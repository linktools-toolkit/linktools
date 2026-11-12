#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test selection and allowlisted synthetic storage cost reporting."""

import hashlib
import json
import math
import os
import re
import sqlite3
import sys
import typing
from fnmatch import fnmatchcase
from pathlib import Path

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addini(
        "asyncio_default_fixture_loop_scope", "Default loop scope for async fixtures",
        type="string", default="function",
    )
    parser.addoption(
        "--test-tier", choices=("daily", "merge", "all"), default="merge",
        help="daily: representative regressions; merge: automatic acceptance; all: include local manual probes",
    )
    parser.addoption(
        "--test-group", default="all",
        help="Select a package test group declared in linktools.yml via manage.py check",
    )
    parser.addoption(
        "--test-shard", default=None,
        help="Select a 1-based file shard INDEX/TOTAL declared for the package test group",
    )


def pytest_configure(config: pytest.Config) -> None:
    config.pluginmanager.register(_ManifestGroups(config), "linktools-manifest-groups")
    if not hasattr(config, "workerinput"):
        config.pluginmanager.register(_StorageCostReports(), "linktools-storage-cost-reports")


class _StorageCostReports:
    """Only publish allowlisted numeric schemas from synthetic native storage fixtures."""

    _PROPERTY = "native_event_lifecycle_cost"
    _TEST = (
        "tests/ai/test_step_event_run_references.py::"
        "test_matched_event_lifecycle_dml_bound_returned_bytes_and_cpu"
    )
    _MODES = {"text_inline", "json_inline", "referenced"}
    _COUNTERS = {
        "bound_bytes", "returned_bytes", "commits", "sql_text_bytes", "returned_rows",
        "application_metadata_encodes", "application_metadata_encoded_bytes",
        "application_metadata_decodes", "json_bind_encodes", "json_bind_encoded_bytes",
        "json_result_decodes", "stored_field_bytes",
    }
    _STATEMENTS = {"BEGIN", "SELECT", "INSERT", "UPDATE", "DELETE", "COMMIT", "ROLLBACK", "SAVEPOINT", "RELEASE"}
    _REPLAY_TEST = (
        "tests/ai/test_native_run_producer.py::"
        "test_cold_registration_replay_confirms_original_without_another_select"
    )
    _REPLAY_COUNTERS = {
        "replay_selects", "replay_inserts", "replay_parameter_bytes", "replay_returned_source_bytes",
        "former_registration_only_payload_bytes", "replay_additional_source_payload_bytes",
        "replay_cpu_ns", "replay_peak_traced_bytes",
    }
    _START_LIFECYCLE_TEST = (
        "tests/ai/test_native_start_projection.py::test_start_sharing_matches_inline_workload_sql_budget"
    )
    _START_CASES = {
        "[0-success]": (0, "success"), "[0-shortcircuit]": (0, "shortcircuit"),
        "[65537-success]": (65537, "success"), "[65537-shortcircuit]": (65537, "shortcircuit"),
    }
    _START_LIFECYCLE_FIELDS = {
        "%s_%s" % (backend, field) for backend in ("owned", "subclass") for field in (
            "select", "insert", "update", "delete", "sql_text_bytes", "bound_text_binary_bytes",
            "scenario_wall_ns", "scenario_cpu_ns", "tracemalloc_session_peak_bytes", "tracemalloc_inherited_session",
        )
    } | {
        "shared_start_sidecar_logical_json_bytes", "shared_start_added_proof_ascii_bytes",
        "shared_start_former_inline_logical_json_bytes", "shared_start_signed_logical_byte_delta",
        "shared_total_dml", "inline_total_dml", "shared_native_event_insert_statements", "inline_native_event_insert_statements",
    }
    _START_READ_TEST = (
        "tests/ai/test_native_start_projection.py::test_large_metadata_is_not_returned_by_cold_trace_select"
    )
    _START_READ_FIELDS = {
        "metadata_%d_%s" % (size, field) for size in (0, 65537) for field in (
            "select", "insert", "update", "delete", "sql_text_bytes", "bound_text_binary_bytes",
            "returned_sql_text_binary_bytes",
        )
    }
    _START_PROOF_TEST = (
        "tests/ai/test_native_started_source.py::test_managed_start_write_records_proof_encoding_cost"
    )
    _START_PROOF_FIELDS = {
        "managed_start_%s_%s" % (label, field) for label in ("body", "fields")
        for field in ("calls", "encoded_bytes", "cpu_ns")
    } | {"managed_start_write_cpu_ns", "managed_start_metadata_utf8_bytes", "managed_start_selects_sqlite"}
    _START_FLAGS = {"owned_tracemalloc_inherited_session", "subclass_tracemalloc_inherited_session"}
    _TERMINAL_CASES = {
        **{
            "test_terminal_outcome_and_full_source_survive_cold_replay[%s]" % outcome:
            ("cold_terminal_replay", {"case": "cold_replay", "outcome": outcome}, {"source_row_upper_bound"})
            for outcome in ("succeeded", "interrupted", "failed", "cancelled")
        },
        **{
            "test_owned_runtime_terminal_is_native_with_only_trace_position[%s]" % outcome:
            (outcome, {"case": "runtime", "outcome": outcome, "fixture_metadata_bytes": 65537}, {
                "terminal_sidecar_logical_json_bytes", "terminal_former_inline_logical_json_bytes",
                "terminal_native_proof_ascii_bytes",
            }) for outcome in ("success", "failure", "cancel", "shortcircuit")
        },
        **{
            "test_terminal_seal_unknown_readback_checks_source_in_existing_run_read[%s]" % damage:
            ("terminal_seal_unknown", {"case": "seal_unknown", "damage": damage}, set())
            for damage in ("none", "missing", "outcome", "metadata", "hash")
        },
        "test_native_terminal_unknown_rejects_changed_registration_in_one_select":
        ("native_terminal_registration_unknown", {"case": "registration_unknown"}, set()),
        **{
            "test_terminal_publishers_reuse_one_source_without_second_write[%s-%s]" % (native_first, failed_echo):
            ("terminal_trace_publication", {"case": "publication", "native_first": native_first,
                                            "failed_echo": failed_echo}, set())
            for native_first, failed_echo in ((False, False), (True, False), (False, True))
        },
    }

    _MESSAGE_SOURCE = "tests/ai/test_native_message_ownership_cost.py::"
    _MESSAGE_CASES = {
        "%s[%d]" % (name, size): (case, size)
        for name, case, sizes in (
            ("test_shared_owner_writes_and_cold_reads_have_linear_message_cost", "shared_batch", (10, 100, 1000)),
            ("test_incremental_owner_chain_reuses_confirmed_prefixes_each_turn", "incremental_turns", (3,)),
        ) for size in sizes
    }
    _MESSAGE_PHASES = {
        "shared_batch": {"shared_write", "cold_read", "checkpoint_publication", "checkpoint_recovery"},
        "incremental_turns": {"incremental_lifecycle"},
    }
    _MESSAGE_COUNTERS = {
        "materialized_rows", "returned_body_bytes", "encoded_messages", "prepared_payloads",
        "encoded_bytes", "snapshot_digest_input_bytes", "context_history_digest_input_bytes",
        "native_bound_text_and_body_bytes", "process_cpu_ns",
    }
    _MESSAGE_EXTRAS = {
        "shared_batch": {"stored_message_bytes"},
        "incremental_turns": {
            "incremental_stored_message_bytes", "unconfirmed_context_encoded_messages",
            "incremental_turns_completed", "incremental_max_cold_tail_rows", "incremental_sample_turn",
        },
    }
    _MESSAGE_ENCODING_FIELDS = {
        "delta_freeze", "mutable_snapshot_freeze", "frozen_projection", "confirmed_context",
    }

    def __init__(self) -> None:
        self.reports = {}  # type: typing.Dict[bool, typing.Dict[str, typing.Any]]
        self.replay = None  # type: typing.Optional[typing.Dict[str, typing.Any]]
        self.start_reports = {}  # type: typing.Dict[typing.Tuple[str, str], typing.Dict[str, typing.Any]]
        self.terminal_reports = {}  # type: typing.Dict[str, typing.Dict[str, typing.Any]]
        self.message_reports = {}  # type: typing.Dict[str, typing.Dict[str, typing.Any]]

    @classmethod
    def _count(cls, value: object) -> bool:
        return type(value) is int and 0 <= value <= 2 ** 63 - 1

    @classmethod
    def _start_value(cls, name: str, value: object) -> bool:
        if name in cls._START_FLAGS:
            return type(value) is bool
        if name in {"owned_tracemalloc_session_peak_bytes", "subclass_tracemalloc_session_peak_bytes"}:
            return value is None or cls._count(value)
        if name == "shared_start_signed_logical_byte_delta":
            return type(value) is int and -(2 ** 63) <= value <= 2 ** 63 - 1
        return cls._count(value)

    @classmethod
    def _parse(cls, value: object, matches: bool) -> "typing.Optional[typing.Dict[str, typing.Any]]":
        if not isinstance(value, str) or len(value) > 16384:
            return None
        try:
            data = json.loads(value)
        except (ValueError, RecursionError):
            return None
        if not isinstance(data, dict) or set(data) != {"metadata_matches_native_run", "samples"}:
            return None
        if data["metadata_matches_native_run"] is not matches:
            return None
        samples = data["samples"]
        if not isinstance(samples, dict) or set(samples) != cls._MODES:
            return None
        for metrics in samples.values():
            if not isinstance(metrics, dict) or set(metrics) != cls._COUNTERS | {"statements", "cpu_seconds"}:
                return None
            statements = metrics["statements"]
            if not isinstance(statements, dict) or not statements or not set(statements) <= cls._STATEMENTS:
                return None
            counts = list(statements.values()) + [metrics[name] for name in cls._COUNTERS]
            if not all(cls._count(count) for count in counts):
                return None
            cpu = metrics["cpu_seconds"]
            if type(cpu) not in (int, float) or not 0 <= cpu <= 2 ** 63 - 1 or not math.isfinite(cpu):
                return None
        return data

    def _record_start(self, report: pytest.TestReport, outcome: str) -> bool:
        identity = {}
        if report.nodeid.startswith(self._START_LIFECYCLE_TEST):
            case = self._START_CASES.get(report.nodeid[len(self._START_LIFECYCLE_TEST):])
            if case is None:
                return False
            identity = {"metadata_bytes": case[0], "scenario_outcome": case[1]}
            label, fields = "native_start_lifecycle_cost", self._START_LIFECYCLE_FIELDS
        elif report.nodeid == self._START_READ_TEST:
            label, fields = "native_start_read_cost", self._START_READ_FIELDS
        elif report.nodeid == self._START_PROOF_TEST:
            label, fields = "native_start_proof_cost", self._START_PROOF_FIELDS
        else:
            return False
        values = report.user_properties
        names = [name for name, _ in values]
        if (any(type(name) is not str or name not in fields for name in names)
                or len(set(names)) != len(names)):
            result = {"status": "invalid"}
        elif not all(self._start_value(name, value) for name, value in values):
            result = {"status": "invalid"}
        elif set(names) != fields:
            result = {"status": "unavailable"}
        else:
            result = dict(values)
        result.update(identity)
        result["test_outcome"] = outcome
        self.start_reports[label, report.nodeid] = result
        return True

    @classmethod
    def _message_json(cls, value: object) -> "typing.Optional[typing.Dict[str, typing.Any]]":
        if type(value) is not str or len(value) > 4096:
            return None
        try:
            def unique_pairs(pairs):
                result = {}
                for key, item in pairs:
                    if key in result:
                        raise ValueError("duplicate property")
                    result[key] = item
                return result
            result = json.loads(value, object_pairs_hook=unique_pairs)
        except (ValueError, RecursionError):
            return None
        return result if type(result) is dict else None

    def _record_message(self, report: pytest.TestReport, outcome: str) -> bool:
        if not report.nodeid.startswith(self._MESSAGE_SOURCE):
            return False
        case = self._MESSAGE_CASES.get(report.nodeid[len(self._MESSAGE_SOURCE):])
        if case is None:
            return False
        label, size = case
        phases, extras = self._MESSAGE_PHASES[label], self._MESSAGE_EXTRAS[label]
        names = [name for name, _ in report.user_properties]
        expected = phases | extras
        if label == "incremental_turns":
            expected = expected | {"incremental_encoding_phases"}
        result = {}
        invalid = (any(type(name) is not str or name not in expected for name in names)
                   or len(set(names)) != len(names))
        if not invalid:
            for name, value in report.user_properties:
                if name in extras:
                    if not self._count(value):
                        invalid = True
                        break
                    if name in {"incremental_turns_completed", "incremental_sample_turn"} and value != size:
                        invalid = True
                        break
                    result[name] = value
                    continue
                parsed = self._message_json(value)
                if parsed is None:
                    invalid = True
                    break
                if name == "incremental_encoding_phases":
                    valid = (set(parsed) == self._MESSAGE_ENCODING_FIELDS
                             and all(self._count(item) for item in parsed.values()))
                else:
                    statements = parsed.get("sql")
                    peak = "sampled_peak_bytes" if label == "incremental_turns" else "traced_peak_bytes"
                    counters = self._MESSAGE_COUNTERS | {peak}
                    valid = (set(parsed) == counters | {"sql"}
                             and type(statements) is dict and set(statements) <= self._STATEMENTS
                             and all(self._count(item) for item in statements.values())
                             and all(self._count(parsed[field]) for field in counters))
                if not valid:
                    invalid = True
                    break
                result[name] = parsed
        if invalid:
            result = {"status": "invalid"}
        elif set(names) != expected:
            result = {"status": "unavailable"}
        result.update({"case": label, "size": size, "test_outcome": outcome})
        self.message_reports[report.nodeid] = result
        return True

    def pytest_runtest_logreport(self, report: pytest.TestReport) -> None:
        # xdist forwards user_properties on call reports to the controller, too.
        if report.when != "call":
            return
        outcome = report.outcome if report.outcome in {"passed", "failed", "skipped"} else "unknown"
        if self._record_message(report, outcome):
            return
        if self._record_start(report, outcome):
            return
        if self._record_terminal(report, outcome):
            return
        if report.nodeid == self._REPLAY_TEST:
            values = [(name, value) for name, value in report.user_properties
                      if isinstance(name, str) and name in self._REPLAY_COUNTERS]
            self.replay = {"status": "invalid" if values else "unavailable"}
            if (len(values) == len(self._REPLAY_COUNTERS)
                    and {name for name, _ in values} == self._REPLAY_COUNTERS
                    and all(self._count(value) for _, value in values)):
                self.replay = dict(values)
            self.replay["test_outcome"] = outcome
            return
        if report.nodeid not in (self._TEST + "[True]", self._TEST + "[False]"):
            return
        matches = report.nodeid.endswith("[True]")
        result = {"metadata_matches_native_run": matches, "status": "unavailable"}
        values = [value for name, value in report.user_properties if name == self._PROPERTY]
        if values:
            parsed = self._parse(values[0], matches) if len(values) == 1 else None
            result = parsed if parsed is not None else {"metadata_matches_native_run": matches, "status": "invalid"}
        result["test_outcome"] = outcome
        self.reports[matches] = result

    def _record_terminal(self, report: pytest.TestReport, outcome: str) -> bool:
        source = "tests/ai/test_native_terminal_projection.py::"
        if not report.nodeid.startswith(source):
            return False
        case = self._TERMINAL_CASES.get(report.nodeid[len(source):])
        if case is None:
            return False
        prefix, identity, extra = case
        fields = {prefix + "_" + field for field in (
            "select", "insert", "update", "delete", "sql_text_bytes", "bound_text_binary_bytes",
        )} | extra
        values = report.user_properties
        names = [name for name, _ in values]
        if (any(type(name) is not str or name not in fields for name in names)
                or len(set(names)) != len(names) or not all(self._count(value) for _, value in values)):
            result = {"status": "invalid"}
        elif set(names) != fields:
            result = {"status": "unavailable"}
        else:
            result = dict(values)
        result.update(identity)
        result["test_outcome"] = outcome
        self.terminal_reports[report.nodeid] = result
        return True

    def pytest_terminal_summary(self, terminalreporter: object) -> None:
        if (not self.reports and self.replay is None and not self.start_reports
                and not self.terminal_reports and not self.message_reports):
            return
        terminalreporter.section("Synthetic native storage costs")
        sha = os.environ.get("GITHUB_SHA", "")
        terminalreporter.write_line("storage_cost_context " + json.dumps({
            "backend": "sqlite+aiosqlite",
            "sqlite_version": sqlite3.sqlite_version,
            "python": "%d.%d.%d" % sys.version_info[:3],
            "checkout_sha": sha if re.fullmatch("[0-9a-fA-F]{40}", sha) else "unknown",
        }, sort_keys=True))
        terminalreporter.write_line(
            "CPU: event process_time seconds; replay/START process_time_ns. "
            "Replay enables tracemalloc; START samples only an externally enabled session (null means not measured). "
            "Not strictly comparable across machines. "
            "Bytes: counted fields/parameters/results, not database-file or network-wire size."
        )
        if self.reports:
            terminalreporter.write_line(
                "Event baselines: json_inline=current JSON inline; text_inline=former TEXT event codec only. "
                "All modes use the current Run lifecycle, not a whole-release comparison."
            )
        for matches in sorted(self.reports):
            terminalreporter.write_line(self._PROPERTY + " " + json.dumps(self.reports[matches], sort_keys=True))
        if self.replay is not None:
            terminalreporter.write_line(
                "Replay baseline: former_registration_only_payload_bytes counts only the original registration ID; "
                "not a measured old-release SQL/CPU run. replay_peak_traced_bytes is traced Python allocation."
            )
            terminalreporter.write_line("native_registration_replay_cost " + json.dumps(self.replay, sort_keys=True))
        if self.start_reports:
            terminalreporter.write_line(
                "Owned/subclass lifecycle baseline: current algorithm across the whole lifecycle, including terminal "
                "sharing, not a historical-version benchmark. START's 128-byte proof may offset savings for a short start."
            )
            terminalreporter.write_line(
                "Lifecycle timing: whole-scenario wall/CPU nanoseconds; SQL counters cover the runtime listener window. "
                "Tracemalloc peak is session-wide when inherited_session=true, not an isolated allocation increment."
            )
            for key in sorted(self.start_reports):
                terminalreporter.write_line(key[0] + " " + json.dumps(self.start_reports[key], sort_keys=True))
        if self.terminal_reports:
            terminalreporter.write_line(
                "Terminal observations: fixed SQLite fixtures only. Runtime fixture metadata is 65537 input bytes; "
                "logical sidecar/inline/proof sizes are not a historical-version benchmark. "
                "Terminal cases do not record CPU, peak memory or measured response-byte costs."
            )
            terminalreporter.write_line(
                "Runtime counters cover the full runtime listener window; publication counters cover only Trace publication. "
                "source_row_upper_bound and terminal_native_proof_ascii_bytes are contract-derived constants, not measurements."
            )
            for key in sorted(self.terminal_reports):
                terminalreporter.write_line("native_terminal_cost " + json.dumps(self.terminal_reports[key], sort_keys=True))
        if self.message_reports:
            terminalreporter.write_line(
                "Message costs: shared_batch is one N-message publication; incremental_turns is N growing commits. "
                "CPU is process_time_ns; traced_peak_bytes is the observation session peak; "
                "sampled_peak_bytes covers only incremental_sample_turn. "
                "Native bound bytes count UTF-8 text and binary parameters for message/sequence DML, "
                "excluding numeric binds and protocol framing. Returned/stored message bytes count binary bodies, "
                "not attachment objects or database-file size. "
                "Mutable input freezing and full-history digest bytes are reported separately from reuse; "
                "no whole-lifecycle linear CPU claim or historical-version baseline."
            )
            for key in sorted(self.message_reports):
                terminalreporter.write_line("native_message_cost " + json.dumps(self.message_reports[key], sort_keys=True))


class _ManifestGroups:

    def __init__(self, config: pytest.Config) -> None:
        self.group = config.getoption("--test-group")
        try:
            check = json.loads(os.environ.get("LINKTOOLS_PYTEST", "{}"))
        except ValueError as error:
            raise pytest.UsageError("Invalid package pytest configuration: %s" % error)
        self.groups = check.get("groups", {})
        if self.group != "all" and self.group not in self.groups:
            raise pytest.UsageError(
                "Unknown test group %r; use manage.py check with one of: %s"
                % (self.group, ", ".join(("all",) + tuple(self.groups)))
            )
        self.roots = tuple(Path(path) for path in check.get("paths", ()))
        self.fallback = next((name for name, patterns in self.groups.items() if not patterns), None)
        self.shard = None
        shard = config.getoption("--test-shard")
        if shard is not None:
            if not re.fullmatch(r"[1-9][0-9]*/[1-9][0-9]*", shard):
                raise pytest.UsageError("Invalid test shard %r; expected INDEX/TOTAL with 1 <= INDEX <= TOTAL" % shard)
            index, total = (int(part) for part in shard.split("/"))
            if index > total:
                raise pytest.UsageError("Invalid test shard %r; INDEX exceeds TOTAL" % shard)
            expected = check.get("shards", {}).get(self.group, 1)
            if not self.roots or total != expected:
                raise pytest.UsageError(
                    "Test shard %r does not match manifest shard count %s for group %r; use manage.py check"
                    % (shard, expected, self.group)
                )
            self.shard = index - 1, total

    def _excludes(self, path: Path, validate: bool = True) -> bool:
        if not any(root == path or root in path.parents for root in self.roots):
            return False
        matches = [
            name for name, patterns in self.groups.items()
            if any(fnmatchcase(path.name, pattern) for pattern in patterns)
        ]
        if len(matches) > 1:
            if validate:
                raise pytest.UsageError("Ambiguous test groups for %s: %s" % (path, ", ".join(matches)))
            # Collect conflicts so item validation still fails, without rejecting support files.
            return False
        family = matches[0] if matches else self.fallback
        if self.group != "all" and family != self.group:
            return True
        if self.shard is not None:
            index, total = self.shard
            # Keep file ownership independent of collection order, machine paths and Python hash seeds.
            relative = path.relative_to(Path(__file__).resolve().parent).as_posix()
            owner = int.from_bytes(hashlib.sha256(relative.encode("utf-8")).digest(), "big") % total
            return owner != index
        return False

    def _ignore_collect(self, path: Path) -> "typing.Optional[bool]":
        # Filename groups cannot exclude directories, which may contain other families.
        # Older pytest also needs __init__.py to establish package collectors and setup.
        if (
            (self.group != "all" or self.shard is not None) and path.name != "__init__.py"
            and path.is_file() and self._excludes(path, validate=False)
        ):
            return True
        return None

    # pytest 7 added pathlib hook arguments; pytest 9 removed the legacy arguments.
    if int(pytest.__version__.split(".")[0]) >= 7:
        def pytest_ignore_collect(self, collection_path: Path) -> "typing.Optional[bool]":
            return self._ignore_collect(collection_path)
    else:
        def pytest_ignore_collect(self, path: object) -> "typing.Optional[bool]":
            return self._ignore_collect(Path(str(path)))

    def pytest_collection_modifyitems(self, config: pytest.Config, items: "typing.List[pytest.Item]") -> None:
        tier = config.getoption("--test-tier")
        selected = []
        deselected = []
        for item in items:
            path = getattr(item, "path", None)
            if path is None:
                path = Path(str(item.fspath))
            excluded = self._excludes(path) or (
                tier != "all" and item.get_closest_marker("manual") is not None
            ) or (
                tier == "daily" and item.get_closest_marker("merge") is not None
            )
            (deselected if excluded else selected).append(item)
        items[:] = selected
        if deselected:
            config.hook.pytest_deselected(items=deselected)
