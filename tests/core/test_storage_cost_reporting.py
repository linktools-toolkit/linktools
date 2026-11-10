#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Synthetic storage costs survive quiet and distributed reporting without payloads."""

import importlib.util
import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest_plugins = ("pytester",)

_EVENT_NODE = (
    "tests/ai/test_step_event_run_references.py::"
    "test_matched_event_lifecycle_dml_bound_returned_bytes_and_cpu"
)
_REPLAY_NODE = (
    "tests/ai/test_native_run_producer.py::"
    "test_cold_registration_replay_confirms_original_without_another_select"
)
_REPLAY = {
    "replay_selects": 1, "replay_inserts": 1, "replay_parameter_bytes": 128,
    "replay_returned_source_bytes": 96, "former_registration_only_payload_bytes": 16,
    "replay_additional_source_payload_bytes": 80, "replay_cpu_ns": 123456,
    "replay_peak_traced_bytes": 4096,
}
_START_LIFECYCLE_NODE = "tests/ai/test_native_start_projection.py::test_start_sharing_matches_inline_workload_sql_budget"
_START_READ_NODE = "tests/ai/test_native_start_projection.py::test_large_metadata_is_not_returned_by_cold_trace_select"
_START_PROOF_NODE = "tests/ai/test_native_started_source.py::test_managed_start_write_records_proof_encoding_cost"
_START_LIFECYCLE = {
    "%s_%s" % (backend, field): 1 for backend in ("owned", "subclass") for field in (
        "select", "insert", "update", "delete", "sql_text_bytes", "bound_text_binary_bytes",
        "scenario_wall_ns", "scenario_cpu_ns", "tracemalloc_session_peak_bytes",
    )
}
_START_LIFECYCLE.update({
    "owned_tracemalloc_inherited_session": True, "subclass_tracemalloc_inherited_session": False,
    "shared_start_sidecar_logical_json_bytes": 64, "shared_start_added_proof_ascii_bytes": 128,
    "shared_start_former_inline_logical_json_bytes": 256, "shared_start_signed_logical_byte_delta": -64,
    "shared_total_dml": 3, "inline_total_dml": 3,
    "shared_native_event_insert_statements": 1, "inline_native_event_insert_statements": 1,
})
_START_READ = {
    "metadata_%d_%s" % (size, field): 1 for size in (0, 65537) for field in (
        "select", "insert", "update", "delete", "sql_text_bytes", "bound_text_binary_bytes",
        "returned_sql_text_binary_bytes",
    )
}
_START_PROOF = {
    "managed_start_%s_%s" % (label, field): 1 for label in ("body", "fields")
    for field in ("calls", "encoded_bytes", "cpu_ns")
}
_START_PROOF.update({"managed_start_write_cpu_ns": 1, "managed_start_metadata_utf8_bytes": 393216,
                     "managed_start_selects_sqlite": 0})
_MESSAGE_NODE = ("tests/ai/test_native_message_ownership_cost.py::"
                 "test_shared_owner_writes_and_cold_reads_have_linear_message_cost")


def _message_counts(sampled: bool = False) -> dict:
    counts = {field: index for index, field in enumerate((
        "materialized_rows", "returned_body_bytes", "encoded_messages", "prepared_payloads",
        "encoded_bytes", "snapshot_digest_input_bytes", "context_history_digest_input_bytes",
        "native_bound_text_and_body_bytes", "process_cpu_ns",
        "sampled_peak_bytes" if sampled else "traced_peak_bytes",
    ), 1)}
    counts["sql"] = {"SELECT": 3, "INSERT": 10}
    return counts


def _shared_message(size: int) -> dict:
    return dict({phase: _message_counts() for phase in (
        "shared_write", "cold_read", "checkpoint_publication", "checkpoint_recovery",
    )}, stored_message_bytes=512 * size)


def _event(matches: bool = True) -> dict:
    return {
        "metadata_matches_native_run": matches,
        "samples": {mode: {
            "statements": {"BEGIN": 33, "INSERT": 35, "SELECT": 3},
            "bound_bytes": 128, "returned_bytes": 96, "commits": 33,
            "sql_text_bytes": 256, "returned_rows": 34,
            "application_metadata_encodes": 35, "application_metadata_encoded_bytes": 64,
            "application_metadata_decodes": 33, "json_bind_encodes": 1,
            "json_bind_encoded_bytes": 16, "json_result_decodes": 0,
            "stored_field_bytes": 48, "cpu_seconds": 0.125,
        } for mode in ("text_inline", "json_inline", "referenced")},
    }


@pytest.fixture
def reporter() -> object:
    source = Path(__file__).resolve().parents[2] / "conftest.py"
    spec = importlib.util.spec_from_file_location("storage_cost_conftest", str(source))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._StorageCostReports()


def _report(node: str, properties: list, when: str = "call") -> SimpleNamespace:
    return SimpleNamespace(nodeid=node, when=when, outcome="passed", user_properties=properties)


def _output(reporter: object) -> str:
    lines = []
    reporter.pytest_terminal_summary(SimpleNamespace(section=lines.append, write_line=lines.append))
    return "\n".join(lines)


@pytest.mark.parametrize("workers", (None, "2"))
def test_quiet_cost_summary_keeps_exact_values_and_failure_outcomes(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch, workers: "str | None",
) -> None:
    root = Path(__file__).resolve().parents[2]
    pytester.makeconftest((root / "conftest.py").read_text(encoding="utf-8"))
    pytester.makeini((root / "pytest.ini").read_text(encoding="utf-8"))
    monkeypatch.delenv("PYTEST_ADDOPTS", raising=False)
    monkeypatch.delenv("LINKTOOLS_PYTEST", raising=False)
    monkeypatch.setenv("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
    monkeypatch.setenv("GITHUB_SHA", "a" * 40)
    directory = pytester.path / "tests" / "ai"
    directory.mkdir(parents=True)
    (directory / "test_step_event_run_references.py").write_text(
        "import json, pytest\n"
        "@pytest.mark.parametrize('metadata_matches_run', [True, False])\n"
        "def test_matched_event_lifecycle_dml_bound_returned_bytes_and_cpu(record_property, metadata_matches_run):\n"
        "    data = %r\n"
        "    data['metadata_matches_native_run'] = metadata_matches_run\n"
        "    record_property('native_event_lifecycle_cost', json.dumps(data))\n"
        "    record_property('sql_traffic', 'private-payload-sentinel')\n"
        "    assert metadata_matches_run\n" % _event(), encoding="utf-8",
    )
    (directory / "test_native_run_producer.py").write_text(
        "def test_cold_registration_replay_confirms_original_without_another_select(record_property):\n"
        "    for name, value in %r.items():\n"
        "        record_property(name, value)\n"
        "    record_property('source', 'private-payload-sentinel')\n" % _REPLAY, encoding="utf-8",
    )
    (directory / "test_native_start_projection.py").write_text(
        "import pytest\n"
        "@pytest.mark.parametrize('outcome', ['success', 'shortcircuit'])\n"
        "@pytest.mark.parametrize('metadata_bytes', [0, 65537])\n"
        "def test_start_sharing_matches_inline_workload_sql_budget(record_property, outcome, metadata_bytes):\n"
        "    for name, value in %r.items():\n"
        "        record_property(name, value)\n"
        "def test_large_metadata_is_not_returned_by_cold_trace_select(record_property):\n"
        "    for name, value in %r.items():\n"
        "        record_property(name, value)\n" % (_START_LIFECYCLE, _START_READ), encoding="utf-8",
    )
    (directory / "test_native_started_source.py").write_text(
        "def test_managed_start_write_records_proof_encoding_cost(record_property):\n"
        "    for name, value in %r.items():\n"
        "        record_property(name, value)\n" % _START_PROOF, encoding="utf-8",
    )
    (directory / "test_native_message_ownership_cost.py").write_text(
        "import json, pytest\n"
        "@pytest.mark.parametrize('size', [10, 100, 1000])\n"
        "def test_shared_owner_writes_and_cold_reads_have_linear_message_cost(record_property, size):\n"
        "    data = %r\n"
        "    for name, value in data[size].items():\n"
        "        record_property(name, json.dumps(value) if isinstance(value, dict) else value)\n"
        % {size: _shared_message(size) for size in (10, 100, 1000)}, encoding="utf-8",
    )
    pytester.makepyfile(test_unrelated=(
        "def test_unrelated(record_property):\n"
        "    record_property('native_event_lifecycle_cost', 'private-payload-sentinel')\n"
        "    record_property('replay_selects', 'private-payload-sentinel')\n"
    ))
    args = ["-q", "--tb=short", "-p", "no:cacheprovider"]
    if workers is not None:
        args.extend(("-p", "xdist.plugin", "-n", workers, "--dist=loadfile"))
    result = pytester.runpytest_subprocess(*args)
    result.assert_outcomes(passed=12, failed=1)
    summary = result.stdout.str().split("Synthetic native storage costs", 1)[1]
    events = [json.loads(line.split(" ", 1)[1]) for line in summary.splitlines()
              if line.startswith("native_event_lifecycle_cost ")]
    assert events == [dict(_event(False), test_outcome="failed"), dict(_event(), test_outcome="passed")]
    replay = [json.loads(line.split(" ", 1)[1]) for line in summary.splitlines()
              if line.startswith("native_registration_replay_cost ")]
    assert replay == [dict(_REPLAY, test_outcome="passed")]
    starts = [json.loads(line.split(" ", 1)[1]) for line in summary.splitlines()
              if line.startswith("native_start_lifecycle_cost ")]
    assert len(starts) == 4
    assert {(item["metadata_bytes"], item["scenario_outcome"]) for item in starts} == {
        (size, outcome) for size in (0, 65537) for outcome in ("success", "shortcircuit")
    }
    for item in starts:
        assert item == dict(_START_LIFECYCLE, metadata_bytes=item["metadata_bytes"],
                            scenario_outcome=item["scenario_outcome"], test_outcome="passed")
    for label, expected in (("native_start_read_cost", _START_READ), ("native_start_proof_cost", _START_PROOF)):
        records = [json.loads(line.split(" ", 1)[1]) for line in summary.splitlines() if line.startswith(label + " ")]
        assert records == [dict(expected, test_outcome="passed")]
    messages = [json.loads(line.split(" ", 1)[1]) for line in summary.splitlines()
                if line.startswith("native_message_cost ")]
    assert sorted(messages, key=lambda item: item["size"]) == [
        dict(_shared_message(size), case="shared_batch", size=size, test_outcome="passed")
        for size in (10, 100, 1000)
    ]
    context = next(json.loads(line.split(" ", 1)[1]) for line in summary.splitlines()
                   if line.startswith("storage_cost_context "))
    assert context["backend"] == "sqlite+aiosqlite" and context["checkout_sha"] == "a" * 40
    assert context["sqlite_version"] == sqlite3.sqlite_version
    assert len(context["python"].split(".")) == 3
    assert "not a whole-release comparison" in summary
    assert "Not strictly comparable across machines" in summary
    assert "not a measured old-release SQL/CPU run" in summary
    assert "not a historical-version benchmark" in summary
    assert "128-byte proof may offset savings" in summary
    assert "Native bound bytes count UTF-8 text and binary parameters" in summary
    assert "excluding numeric binds and protocol framing" in summary
    assert "Returned/stored message bytes count binary bodies" in summary
    assert "sampled_peak_bytes covers only incremental_sample_turn" in summary
    assert "Native bind JSON bytes" not in summary
    assert "returned body UTF-8 bytes" not in summary
    assert "private-payload-sentinel" not in summary


@pytest.mark.parametrize("field,value", (
    ("payload", "private-payload-sentinel"),
    ("statements", {"SELECT private-payload-sentinel": 1}),
    ("bound_bytes", "private-payload-sentinel"),
    ("bound_bytes", {"private-payload-sentinel": 1}),
    ("bound_bytes", True), ("bound_bytes", -1), ("bound_bytes", 1.0), ("bound_bytes", 2 ** 63),
    ("cpu_seconds", True), ("cpu_seconds", -1.0),
    ("cpu_seconds", float("nan")), ("cpu_seconds", float("inf")),
))
def test_event_cost_summary_rejects_unknown_fields_and_nonnumeric_counts(
    reporter: object, field: str, value: object,
) -> None:
    data = _event()
    data["samples"]["referenced"][field] = value
    reporter.pytest_runtest_logreport(_report(_EVENT_NODE + "[True]", [
        ("native_event_lifecycle_cost", json.dumps(data)),
    ]))
    output = _output(reporter)
    assert '"status": "invalid"' in output
    assert '"samples"' not in output
    assert "private-payload-sentinel" not in output


@pytest.mark.parametrize("payload", ("not-json", "[[]]", "x" * 16385, None))
def test_event_cost_summary_rejects_malformed_or_oversized_properties(reporter: object, payload: object) -> None:
    reporter.pytest_runtest_logreport(_report(_EVENT_NODE + "[True]", [("native_event_lifecycle_cost", payload)]))
    assert '"status": "invalid"' in _output(reporter)


@pytest.mark.parametrize("change", ("mode", "scenario", "missing", "duplicate"))
def test_event_cost_summary_requires_the_complete_unambiguous_schema(reporter: object, change: str) -> None:
    data = _event()
    if change == "mode":
        data["samples"]["private-payload-sentinel"] = data["samples"].pop("text_inline")
    elif change == "scenario":
        data["metadata_matches_native_run"] = False
    elif change == "missing":
        data["samples"]["referenced"].pop("returned_bytes")
    properties = [("native_event_lifecycle_cost", json.dumps(data))] * (2 if change == "duplicate" else 1)
    reporter.pytest_runtest_logreport(_report(_EVENT_NODE + "[True]", properties))
    output = _output(reporter)
    assert '"status": "invalid"' in output
    assert "private-payload-sentinel" not in output


@pytest.mark.parametrize("change", ("missing", "duplicate", "text", "boolean", "negative", "float"))
def test_replay_cost_summary_requires_each_allowlisted_integer_once(reporter: object, change: str) -> None:
    properties = list(_REPLAY.items())
    if change == "missing":
        properties.pop()
    elif change == "duplicate":
        properties.append(properties[0])
    else:
        properties[0] = (properties[0][0], {
            "text": "private-payload-sentinel", "boolean": True, "negative": -1, "float": 1.0,
        }[change])
    reporter.pytest_runtest_logreport(_report(_REPLAY_NODE, properties))
    output = _output(reporter)
    assert '"status": "invalid"' in output
    assert '"replay_selects"' not in output
    assert "private-payload-sentinel" not in output


def test_cost_summary_ignores_other_tests_and_report_phases(reporter: object) -> None:
    properties = [("native_event_lifecycle_cost", json.dumps(_event()))] + list(_REPLAY.items())
    for node, when in (("test_other.py::test_other", "call"), (_EVENT_NODE + "[True]", "setup"),
                       (_REPLAY_NODE, "teardown")):
        reporter.pytest_runtest_logreport(_report(node, properties, when))
    assert _output(reporter) == ""


def test_cost_summary_marks_missing_observations_and_does_not_echo_ci_context(
    reporter: object, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GITHUB_SHA", "private-payload-sentinel\n::warning::payload")
    reporter.pytest_runtest_logreport(_report(_EVENT_NODE + "[True]", []))
    reporter.pytest_runtest_logreport(_report(_REPLAY_NODE, []))
    output = _output(reporter)
    assert output.count('"status": "unavailable"') == 2
    assert '"checkout_sha": "unknown"' in output
    assert "private-payload-sentinel" not in output and "::warning::payload" not in output


@pytest.mark.parametrize("node,data", (
    (_START_LIFECYCLE_NODE + "[0-success]", _START_LIFECYCLE),
    (_START_READ_NODE, _START_READ), (_START_PROOF_NODE, _START_PROOF),
))
@pytest.mark.parametrize("change,status", (
    ("extra", "invalid"), ("duplicate", "invalid"), ("type", "invalid"), ("missing", "unavailable"),
))
def test_start_cost_schemas_reject_extra_duplicate_and_bad_values_without_payloads(
    reporter: object, node: str, data: dict, change: str, status: str,
) -> None:
    properties = list(data.items())
    if change == "extra":
        properties.append(("private-payload-sentinel", "private-payload-sentinel"))
    elif change == "duplicate":
        properties.append(properties[0])
    elif change == "type":
        properties[0] = (properties[0][0], "private-payload-sentinel")
    else:
        properties.pop()
    reporter.pytest_runtest_logreport(_report(node, properties))
    output = _output(reporter)
    assert '"status": "%s"' % status in output
    assert '"%s"' % properties[0][0] not in output
    assert "private-payload-sentinel" not in output


@pytest.mark.parametrize("name,value", (
    ("owned_select", True), ("owned_select", -1), ("owned_select", 1.0), ("owned_select", 2 ** 63),
    ("owned_tracemalloc_inherited_session", 1), ("subclass_tracemalloc_inherited_session", "false"),
    ("shared_start_signed_logical_byte_delta", True), ("shared_start_signed_logical_byte_delta", -1.0),
    ("shared_start_signed_logical_byte_delta", -(2 ** 63) - 1),
))
def test_start_cost_schema_distinguishes_counts_flags_and_signed_delta(
    reporter: object, name: str, value: object,
) -> None:
    data = dict(_START_LIFECYCLE)
    data[name] = value
    reporter.pytest_runtest_logreport(_report(_START_LIFECYCLE_NODE + "[65537-shortcircuit]", list(data.items())))
    assert '"status": "invalid"' in _output(reporter)


@pytest.mark.parametrize("node", (
    _START_LIFECYCLE_NODE + "[0-failure]", _START_LIFECYCLE_NODE + "[1-success]",
    _START_LIFECYCLE_NODE + "[private-payload-sentinel]", _START_READ_NODE + "[0]",
    _START_PROOF_NODE + "_other",
))
def test_start_cost_reporting_does_not_expand_to_unapproved_test_cases(reporter: object, node: str) -> None:
    reporter.pytest_runtest_logreport(_report(node, list(_START_LIFECYCLE.items())))
    assert _output(reporter) == ""


@pytest.mark.parametrize("change", (None, "whole_lifecycle_peak", "wrong_sample_turn"))
def test_incremental_memory_report_identifies_final_turn_sample(reporter: object, change: "str | None") -> None:
    node = ("tests/ai/test_native_message_ownership_cost.py::"
            "test_incremental_owner_chain_reuses_confirmed_prefixes_each_turn[1000]")
    counts = _message_counts(sampled=True)
    counts["sql"] = {"SELECT": 3000}
    if change == "whole_lifecycle_peak":
        counts["traced_peak_bytes"] = counts.pop("sampled_peak_bytes")
    properties = [
        ("incremental_lifecycle", json.dumps(counts)),
        ("incremental_encoding_phases", json.dumps({field: 1 for field in (
            "delta_freeze", "mutable_snapshot_freeze", "frozen_projection", "confirmed_context",
        )})),
        ("incremental_stored_message_bytes", 1000),
        ("unconfirmed_context_encoded_messages", 0),
        ("incremental_turns_completed", 1000),
        ("incremental_max_cold_tail_rows", 10),
        ("incremental_sample_turn", 100 if change == "wrong_sample_turn" else 1000),
    ]
    reporter.pytest_runtest_logreport(_report(node, properties))
    output = _output(reporter)
    if change is None:
        assert '"sampled_peak_bytes": 10' in output
        assert '"incremental_sample_turn": 1000' in output
        assert '"status"' not in output
    else:
        assert '"status": "invalid"' in output
        assert '"sampled_peak_bytes"' not in output
    assert '"traced_peak_bytes"' not in output


@pytest.mark.parametrize("field", (
    "returned_body_bytes", "native_bound_text_and_body_bytes", "stored_message_bytes",
))
@pytest.mark.parametrize("value", (True, -1, 1.0, 2 ** 63, "private-payload-sentinel", {"private-payload-sentinel": 1}))
def test_message_cost_summary_requires_integer_byte_counts(reporter: object, field: str, value: object) -> None:
    data = _shared_message(10)
    if field == "stored_message_bytes":
        data[field] = value
    else:
        data["shared_write"][field] = value
    properties = [(name, json.dumps(item) if isinstance(item, dict) else item) for name, item in data.items()]
    reporter.pytest_runtest_logreport(_report(_MESSAGE_NODE + "[10]", properties))
    output = _output(reporter)
    assert '"status": "invalid"' in output
    assert '"shared_write"' not in output
    assert "private-payload-sentinel" not in output


@pytest.mark.parametrize("change,status", (
    ("legacy_returned", "invalid"), ("legacy_bound", "invalid"), ("legacy_stored", "invalid"),
    ("unknown_property", "invalid"), ("unknown_counter", "invalid"), ("unknown_sql", "invalid"),
    ("duplicate_property", "invalid"), ("duplicate_json_key", "invalid"),
    ("missing_counter", "invalid"), ("missing_phase", "unavailable"),
))
def test_message_cost_summary_keeps_exact_schema_without_payloads(
    reporter: object, change: str, status: str,
) -> None:
    data = _shared_message(10)
    counts = data["shared_write"]
    if change == "legacy_returned":
        counts["returned_body_utf8_bytes"] = counts.pop("returned_body_bytes")
    elif change == "legacy_bound":
        counts["native_bind_json_bytes"] = counts.pop("native_bound_text_and_body_bytes")
    elif change == "legacy_stored":
        data["stored_message_utf8_bytes"] = data.pop("stored_message_bytes")
    elif change == "unknown_property":
        data["private-payload-sentinel"] = "private-payload-sentinel"
    elif change == "unknown_counter":
        counts["payload"] = "private-payload-sentinel"
    elif change == "unknown_sql":
        counts["sql"] = {"SELECT private-payload-sentinel": 1}
    elif change == "missing_counter":
        counts.pop("returned_body_bytes")
    elif change == "missing_phase":
        data.pop("shared_write")
    properties = [(name, json.dumps(value) if isinstance(value, dict) else value) for name, value in data.items()]
    if change == "duplicate_property":
        properties.append(properties[0])
    elif change == "duplicate_json_key":
        properties[0] = ("shared_write", properties[0][1][:-1] + ', "returned_body_bytes": 2}')
    reporter.pytest_runtest_logreport(_report(_MESSAGE_NODE + "[10]", properties))
    output = _output(reporter)
    assert '"status": "%s"' % status in output
    assert '"shared_write"' not in output
    assert "private-payload-sentinel" not in output
