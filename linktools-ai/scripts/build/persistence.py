#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build-time validation for the Runtime persistence compatibility contract."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import cast

from linktools.ai.agent import AgentBindingSnapshot
from linktools.ai.core import (
    IdempotencyStatus,
    JsonValue,
    OperationStatus,
    canonical_json_bytes,
)
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime._message import (
    decode_model_messages,
    encode_model_messages,
)
from linktools.ai.runtime.state import _codec as runtime_codec
from linktools.ai.runtime.state._contracts import (
    IdempotencyTerminalUpdate,
    OperationTerminalUpdate,
)
from linktools.ai.spec import AgentSpec
from linktools.ai.task import TaskNode
from pydantic_ai.messages import ModelRequest, UserPromptPart

_MODEL_MESSAGE_FIXTURE_V1 = "runtime_model_messages_v1.json"
_CUSTOM_WIRE_FIXTURE_V1 = "runtime_custom_wire_v1.json"


def load_manifest(path: str | Path) -> dict[str, JsonValue]:
    value = _load_json(Path(path))
    if not isinstance(value, dict):
        raise ValueError("runtime persistence manifest must be an object")
    return cast("dict[str, JsonValue]", value)


def validate_runtime_persistence_manifest(
    manifest: Mapping[str, JsonValue],
) -> tuple[str, ...]:
    expected = runtime_codec._runtime_persistence_manifest()
    return () if dict(manifest) == expected else (
        "runtime persistence manifest does not match the current codec contract",
    )


def _positive_version(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        return None
    return value


def _validate_external_append_only(
    name: str,
    baseline: object,
    candidate: object,
) -> tuple[str, ...]:
    if name != "agent_binding_snapshot":
        return () if candidate == baseline else (
            f"historical external contract changed: {name}",
        )
    if not isinstance(baseline, Mapping) or not isinstance(candidate, Mapping):
        return ("agent_binding_snapshot external contract is invalid",)
    if baseline.get("owner") != candidate.get("owner"):
        return ("agent_binding_snapshot owner changed",)
    baseline_version = _positive_version(baseline.get("version"))
    candidate_version = _positive_version(candidate.get("version"))
    if baseline_version is None or candidate_version is None:
        return ("agent_binding_snapshot version is invalid",)
    if candidate_version < baseline_version:
        return ("agent_binding_snapshot version regressed",)
    return ()


def validate_append_only(
    baseline: Mapping[str, JsonValue],
    candidate: Mapping[str, JsonValue],
) -> tuple[str, ...]:
    errors: list[str] = []
    if baseline.get("wire_version") != candidate.get("wire_version"):
        errors.append("wire_version changed")

    baseline_dataclasses = baseline.get("dataclasses")
    candidate_dataclasses = candidate.get("dataclasses")
    if not isinstance(baseline_dataclasses, Mapping) or not isinstance(
        candidate_dataclasses, Mapping
    ):
        return tuple((*errors, "dataclasses manifest is invalid"))
    for wire_id, raw_baseline in baseline_dataclasses.items():
        raw_candidate = candidate_dataclasses.get(wire_id)
        if not isinstance(raw_baseline, Mapping) or not isinstance(
            raw_candidate, Mapping
        ):
            errors.append(f"historical dataclass removed: {wire_id}")
            continue
        if raw_baseline.get("legacy_revision") != raw_candidate.get(
            "legacy_revision"
        ):
            errors.append(f"legacy revision changed: {wire_id}")
        baseline_revisions = raw_baseline.get("revisions")
        candidate_revisions = raw_candidate.get("revisions")
        if not isinstance(baseline_revisions, Mapping) or not isinstance(
            candidate_revisions, Mapping
        ):
            errors.append(f"revision manifest is invalid: {wire_id}")
            continue
        for revision, fingerprint in baseline_revisions.items():
            if candidate_revisions.get(revision) != fingerprint:
                errors.append(
                    f"historical revision changed: {wire_id}@{revision}"
                )
    for wire_id, raw_candidate in candidate_dataclasses.items():
        if wire_id in baseline_dataclasses:
            continue
        if not isinstance(raw_candidate, Mapping):
            errors.append(f"new dataclass manifest is invalid: {wire_id}")
            continue
        if raw_candidate.get("legacy_revision") is not None:
            errors.append(
                f"new dataclass cannot claim unversioned legacy: {wire_id}"
            )

    baseline_enums = baseline.get("enums")
    candidate_enums = candidate.get("enums")
    if not isinstance(baseline_enums, Mapping) or not isinstance(
        candidate_enums, Mapping
    ):
        errors.append("enum manifest is invalid")
    else:
        for wire_id, raw_values in baseline_enums.items():
            candidate_values = candidate_enums.get(wire_id)
            if not isinstance(raw_values, list) or not isinstance(
                candidate_values, list
            ):
                errors.append(f"historical enum removed: {wire_id}")
                continue
            if not set(raw_values).issubset(candidate_values):
                errors.append(f"historical enum values removed: {wire_id}")

    baseline_external = baseline.get("external")
    candidate_external = candidate.get("external")
    if not isinstance(baseline_external, Mapping) or not isinstance(
        candidate_external, Mapping
    ):
        errors.append("external manifest is invalid")
    else:
        for name, value in baseline_external.items():
            errors.extend(
                _validate_external_append_only(
                    str(name),
                    value,
                    candidate_external.get(name),
                )
            )
    return tuple(errors)


def validate_fixture_append_only(
    baseline: Mapping[str, JsonValue],
    candidate: Mapping[str, JsonValue],
    *,
    label: str = "upgrade fixture",
) -> tuple[str, ...]:
    return tuple(
        f"historical {label} changed: {key}"
        for key, value in baseline.items()
        if candidate.get(key) != value
    )


def _binding_fixture_name(version: int) -> str:
    return f"runtime_agent_binding_snapshot_v{version}.json"


def _agent_binding_snapshot_fixture_value(version: int) -> AgentBindingSnapshot:
    if version != 1:
        raise ValueError(
            f"add a canonical AgentBindingSnapshot fixture value for version {version}"
        )
    return AgentBindingSnapshot(
        version=1,
        agent_spec=AgentSpec("runtime-persistence-v1"),
        agent_digest="a" * 64,
        output_type_module="linktools.ai.runtime.fixture",
        output_type_qualname="FixtureOutput",
        output_schema_id="runtime.persistence.fixture",
        output_schema_revision=1,
        output_schema_fingerprint="b" * 64,
        local_runtime_capability_descriptors=(),
        binding_digest="c" * 64,
    )


def _load_json(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid persistence fixture: {path}: {error}") from error


def _model_message_fixture_values() -> tuple[ModelRequest, ...]:
    fixed = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return (
        ModelRequest(
            parts=[
                UserPromptPart(
                    content="runtime-persistence-v1",
                    timestamp=fixed,
                )
            ]
        ),
    )


def _validate_binding_fixtures(
    binding_contract: Mapping[str, object],
    root: Path,
) -> tuple[str, ...]:
    if binding_contract.get("owner") != "agent":
        return ("agent_binding_snapshot owner changed",)
    current_version = _positive_version(binding_contract.get("version"))
    if current_version is None:
        return ("agent_binding_snapshot version is invalid",)
    errors: list[str] = []
    for version in range(1, current_version + 1):
        path = root / _binding_fixture_name(version)
        if not path.is_file():
            errors.append(f"missing external persistence fixture: {path}")
            continue
        try:
            value = _load_json(path)
            expected = _agent_binding_snapshot_fixture_value(version)
        except ValueError as error:
            errors.append(str(error))
            continue
        if value != expected.to_payload():
            errors.append(
                f"agent_binding_snapshot V{version} writer drifted from its fixture"
            )
            continue
        try:
            decoded = AgentBindingSnapshot.from_payload(value)
        except Exception as error:
            errors.append(
                f"agent_binding_snapshot V{version} fixture is no longer readable: "
                f"{error}"
            )
            continue
        if decoded != expected or decoded.to_payload() != value:
            errors.append(
                f"agent_binding_snapshot V{version} fixture semantics changed"
            )
    return tuple(errors)


def validate_external_fixtures(
    manifest: Mapping[str, JsonValue],
    matrix_dir: str | Path,
) -> tuple[str, ...]:
    external = manifest.get("external")
    if not isinstance(external, Mapping):
        return ("external manifest is invalid",)
    root = Path(matrix_dir)

    binding_contract = external.get("agent_binding_snapshot")
    if not isinstance(binding_contract, Mapping):
        return ("agent_binding_snapshot external contract is invalid",)
    binding_errors = _validate_binding_fixtures(binding_contract, root)
    if binding_errors:
        return binding_errors

    model_contract = external.get("model_message")
    if not isinstance(model_contract, Mapping):
        return ("model_message external contract is invalid",)
    fixture_name = model_contract.get("fixture")
    if fixture_name != _MODEL_MESSAGE_FIXTURE_V1:
        return ("model_message V1 fixture declaration changed",)
    model_path = root / _MODEL_MESSAGE_FIXTURE_V1
    if not model_path.is_file():
        return (f"missing external persistence fixture: {model_path}",)
    try:
        model_value = _load_json(model_path)
    except ValueError as error:
        return (str(error),)
    expected_model = json.loads(
        encode_model_messages(_model_message_fixture_values()).decode("utf-8")
    )
    if model_value != expected_model:
        return ("model_message writer drifted from the frozen V1 fixture",)
    try:
        decoded_model = decode_model_messages(canonical_json_bytes(model_value))
    except Exception as error:
        return (f"model_message V1 fixture is no longer readable: {error}",)
    if decoded_model != _model_message_fixture_values():
        return ("model_message V1 fixture decodes to different semantics",)
    return ()


def _current_contract(wire_id: str):
    contracts = runtime_codec._DATACLASS_PERSISTENCE_BY_VERSION.get(
        runtime_codec._CURRENT_CODEC.version
    )
    if contracts is None or wire_id not in contracts:
        raise RuntimeError(f"missing persistence contract: {wire_id}")
    return contracts[wire_id]


def _custom_wire_values() -> dict[str, JsonValue]:
    task_node = TaskNode(
        "node",
        ("dependency",),
        input={"key": "value"},
        budget_cost=2,
    )
    idempotency = IdempotencyTerminalUpdate(
        scope="scope",
        idempotency_key_digest="a" * 64,
        expected_status=IdempotencyStatus.STARTED,
        next_status=IdempotencyStatus.COMPLETED,
        request_digest="b" * 64,
        result_digest="c" * 64,
        error_code="terminal-error",
    )
    operation = OperationTerminalUpdate(
        operation_id="operation",
        expected_status=OperationStatus.RUNNING,
        next_status=OperationStatus.SUCCEEDED,
        result_ref="result",
        result_digest="d" * 64,
        error_code="terminal-error",
    )
    task_revision = _current_contract("task_node").current_revision
    terminal_revision = _current_contract(
        "execution_terminal_commit"
    ).current_revision
    return {
        f"task_node@{task_revision}": runtime_codec._encode_persisted_domain(
            task_node
        ),
        (
            f"execution_terminal_commit@{terminal_revision}:"
            "idempotency_terminal_update"
        ): runtime_codec._encode_external(
            idempotency,
            runtime_codec._CURRENT_CODEC,
        ),
        (
            f"execution_terminal_commit@{terminal_revision}:"
            "operation_terminal_update"
        ): runtime_codec._encode_external(
            operation,
            runtime_codec._CURRENT_CODEC,
        ),
    }


def _custom_wire_keys() -> set[str]:
    task_contract = _current_contract("task_node")
    terminal_contract = _current_contract("execution_terminal_commit")
    values = {f"task_node@{revision}" for revision in task_contract.fingerprints}
    for revision in terminal_contract.fingerprints:
        values.add(
            f"execution_terminal_commit@{revision}:idempotency_terminal_update"
        )
        values.add(
            f"execution_terminal_commit@{revision}:operation_terminal_update"
        )
    return values


def validate_custom_wire_fixture(matrix_dir: str | Path) -> tuple[str, ...]:
    path = Path(matrix_dir) / _CUSTOM_WIRE_FIXTURE_V1
    if not path.is_file():
        return (f"missing custom persistence fixture: {path}",)
    try:
        value = _load_json(path)
    except ValueError as error:
        return (str(error),)
    if not isinstance(value, Mapping):
        return ("Runtime custom V1 wire fixture must be an object",)
    expected_keys = _custom_wire_keys()
    actual_keys = set(value)
    errors = [
        *(
            f"missing custom wire fixture: {key}"
            for key in sorted(expected_keys - actual_keys)
        ),
        *(
            f"unknown custom wire fixture: {key}"
            for key in sorted(actual_keys - expected_keys)
        ),
    ]
    for key, expected in _custom_wire_values().items():
        if value.get(key) != expected:
            errors.append(f"current custom wire drifted: {key}")
    return tuple(errors)


def _historical_revision_keys(
    manifest: Mapping[str, JsonValue],
) -> set[str]:
    dataclasses = manifest.get("dataclasses")
    if not isinstance(dataclasses, Mapping):
        return set()
    expected: set[str] = set()
    for wire_id, raw_contract in dataclasses.items():
        if not isinstance(wire_id, str) or not isinstance(raw_contract, Mapping):
            continue
        revisions = raw_contract.get("revisions")
        if not isinstance(revisions, Mapping):
            continue
        numbers: list[int] = []
        for raw_revision in revisions:
            if not isinstance(raw_revision, str):
                continue
            try:
                numbers.append(int(raw_revision))
            except ValueError:
                continue
        if numbers:
            current = max(numbers)
            expected.update(
                f"{wire_id}@{revision}"
                for revision in numbers
                if revision < current
            )
    return expected


def _decode_historical_fields(
    wire_id: str,
    revision: int,
    raw_fields: Mapping[str, object],
    target: type[object],
) -> object:
    return runtime_codec._decode_dataclass(
        {
            "$dataclass": wire_id,
            "schema": revision,
            "fields": dict(raw_fields),
        },
        target,
        runtime_codec._CURRENT_CODEC,
        persisted=True,
    )


def _validate_historical_strictness(
    wire_id: str,
    revision: int,
    raw_fields: Mapping[str, object],
    target: type[object],
) -> tuple[str, ...]:
    malformed: list[tuple[str, dict[str, object]]] = []
    with_extra = dict(raw_fields)
    with_extra["__compat_unknown_field__"] = None
    malformed.append(("extra", with_extra))
    keys = tuple(sorted(str(key) for key in raw_fields))
    if keys:
        missing = dict(raw_fields)
        missing.pop(keys[0], None)
        malformed.append(("missing", missing))

    errors: list[str] = []
    for label, fields_value in malformed:
        try:
            _decode_historical_fields(
                wire_id,
                revision,
                fields_value,
                target,
            )
        except AIError as error:
            if error.code is ErrorCode.STORAGE_INTEGRITY_ERROR:
                continue
            errors.append(
                f"historical schema corruption has wrong error: "
                f"{wire_id}@{revision}:{label}:{error.code.value}"
            )
        except Exception as error:
            errors.append(
                f"historical schema corruption escaped codec contract: "
                f"{wire_id}@{revision}:{label}:{type(error).__name__}"
            )
        else:
            errors.append(
                f"historical schema corruption was accepted: "
                f"{wire_id}@{revision}:{label}"
            )
    return tuple(errors)


def validate_upgrade_fixtures(
    manifest: Mapping[str, JsonValue],
    fixtures: Mapping[str, JsonValue],
) -> tuple[str, ...]:
    errors: list[str] = []
    expected = _historical_revision_keys(manifest)
    if set(fixtures) != expected:
        errors.extend(
            f"missing upgrade fixture: {key}"
            for key in sorted(expected - set(fixtures))
        )
        errors.extend(
            f"unknown upgrade fixture: {key}"
            for key in sorted(set(fixtures) - expected)
        )
        return tuple(errors)

    contracts = runtime_codec._DATACLASS_PERSISTENCE_BY_VERSION.get(
        runtime_codec._CURRENT_CODEC.version
    )
    if contracts is None:
        return ("current dataclass persistence contracts are unavailable",)
    for key in sorted(expected):
        wire_id, raw_revision = key.rsplit("@", 1)
        revision = int(raw_revision)
        fixture = fixtures[key]
        if not isinstance(fixture, Mapping) or set(fixture) != {
            "fields",
            "current_fields",
        }:
            errors.append(f"invalid upgrade fixture: {key}")
            continue
        raw_fields = fixture.get("fields")
        expected_fields = fixture.get("current_fields")
        if not isinstance(raw_fields, Mapping) or not isinstance(
            expected_fields, Mapping
        ):
            errors.append(f"invalid upgrade fixture fields: {key}")
            continue
        contract = contracts.get(wire_id)
        target = runtime_codec._CURRENT_CODEC.domain_types.get(wire_id)
        if contract is None or target is None:
            errors.append(f"upgrade fixture target is unavailable: {key}")
            continue
        try:
            upgraded = runtime_codec._apply_persisted_upgrades(
                raw_fields,
                revision,
                contract,
                runtime_codec._CURRENT_CODEC,
            )
            if dict(upgraded) != dict(expected_fields):
                errors.append(f"upgrade fixture semantics changed: {key}")
                continue
            decoded = _decode_historical_fields(
                wire_id,
                revision,
                raw_fields,
                target,
            )
            rewritten = runtime_codec._encode_persisted_domain(decoded)
        except Exception as error:
            errors.append(
                f"upgrade fixture is no longer readable: {key}: {error}"
            )
            continue
        if not isinstance(rewritten, Mapping):
            errors.append(f"upgrade fixture rewrite is invalid: {key}")
            continue
        rewritten_fields = rewritten.get("fields")
        if not isinstance(rewritten_fields, Mapping) or dict(
            rewritten_fields
        ) != dict(expected_fields):
            errors.append(f"upgrade fixture rewrite changed: {key}")
            continue
        errors.extend(
            _validate_historical_strictness(
                wire_id,
                revision,
                raw_fields,
                target,
            )
        )
    return tuple(errors)


def _git(repository: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ("git", "-C", str(repository), *arguments),
        check=False,
        capture_output=True,
        text=True,
    )


def _repository_path(path_value: str | Path) -> tuple[Path, str] | None:
    path = Path(path_value).resolve()
    repository = Path(__file__).resolve().parents[3]
    if not (repository / ".git").exists():
        return None
    try:
        relative = path.relative_to(repository.resolve()).as_posix()
    except ValueError:
        return None
    return repository, relative


def _baseline_commit(
    repository: Path,
    *,
    base_ref: str | None,
) -> str:
    ref = base_ref or os.environ.get(
        "LINKTOOLS_PERSISTENCE_BASE_REF", "origin/master"
    )
    verified = _git(repository, "rev-parse", "--verify", f"{ref}^{{commit}}")
    if verified.returncode != 0 or not verified.stdout.strip():
        raise ValueError(f"persistence baseline ref is unavailable: {ref}")
    baseline = verified.stdout.strip()

    head = _git(repository, "rev-parse", "HEAD")
    if head.returncode != 0 or not head.stdout.strip():
        raise ValueError("current Git HEAD is unavailable")
    if head.stdout.strip() != baseline:
        return baseline

    parent = _git(repository, "rev-parse", "HEAD^")
    if parent.returncode != 0 or not parent.stdout.strip():
        raise ValueError("persistence baseline parent is unavailable")
    return parent.stdout.strip()


def load_git_json_baseline(
    path_value: str | Path,
    *,
    base_ref: str | None = None,
) -> object | None:
    """Load one JSON contract from the target branch's immutable history."""
    resolved = _repository_path(path_value)
    if resolved is None:
        return None
    repository, relative = resolved
    baseline = _baseline_commit(repository, base_ref=base_ref)
    historical = _git(repository, "show", f"{baseline}:{relative}")
    if historical.returncode != 0:
        # A missing path is legitimate only when this contract is first introduced.
        return None
    try:
        return json.loads(historical.stdout)
    except json.JSONDecodeError as error:
        raise ValueError(
            f"historical JSON contract is invalid at {baseline}: {relative}"
        ) from error


def load_git_baseline(
    path_value: str | Path,
    *,
    base_ref: str | None = None,
) -> dict[str, JsonValue] | None:
    value = load_git_json_baseline(path_value, base_ref=base_ref)
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("historical JSON contract is not an object")
    return cast("dict[str, JsonValue]", value)


def _matrix(name: str) -> Path:
    return Path(__file__).with_name("matrix") / name


def _load_optional_baseline(
    path: Path,
    explicit: Path | None,
    base_ref: str | None,
    errors: list[str],
) -> dict[str, JsonValue] | None:
    if explicit is not None:
        return load_manifest(explicit)
    try:
        return load_git_baseline(path, base_ref=base_ref)
    except ValueError as error:
        errors.append(str(error))
        return None


def _freeze_json_fixture(
    path: Path,
    *,
    base_ref: str | None,
    label: str,
    errors: list[str],
) -> None:
    try:
        baseline = load_git_json_baseline(path, base_ref=base_ref)
    except ValueError as error:
        errors.append(str(error))
        return
    if baseline is None:
        return
    try:
        candidate = _load_json(path)
    except ValueError as error:
        errors.append(str(error))
        return
    if candidate != baseline:
        errors.append(f"historical {label} changed")


def _binding_versions(manifest: Mapping[str, JsonValue]) -> range:
    external = manifest.get("external")
    if not isinstance(external, Mapping):
        return range(0)
    binding = external.get("agent_binding_snapshot")
    if not isinstance(binding, Mapping):
        return range(0)
    version = _positive_version(binding.get("version"))
    return range(1, 0 if version is None else version + 1)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=_matrix("runtime-persistence-v1.json"),
    )
    parser.add_argument(
        "--upgrade-fixtures",
        type=Path,
        default=_matrix("runtime-persistence-upgrades-v1.json"),
    )
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--upgrade-baseline", type=Path)
    parser.add_argument("--baseline-ref")
    args = parser.parse_args()

    candidate = load_manifest(args.manifest)
    upgrade_fixtures = load_manifest(args.upgrade_fixtures)
    errors = list(validate_runtime_persistence_manifest(candidate))
    errors.extend(validate_external_fixtures(candidate, args.manifest.parent))
    errors.extend(validate_custom_wire_fixture(args.manifest.parent))
    errors.extend(validate_upgrade_fixtures(candidate, upgrade_fixtures))

    baseline = _load_optional_baseline(
        args.manifest,
        args.baseline,
        args.baseline_ref,
        errors,
    )
    if baseline is not None:
        errors.extend(validate_append_only(baseline, candidate))

    upgrade_baseline = _load_optional_baseline(
        args.upgrade_fixtures,
        args.upgrade_baseline,
        args.baseline_ref,
        errors,
    )
    if upgrade_baseline is not None:
        errors.extend(
            validate_fixture_append_only(
                upgrade_baseline,
                upgrade_fixtures,
                label="upgrade fixture",
            )
        )

    custom_path = args.manifest.parent / _CUSTOM_WIRE_FIXTURE_V1
    custom_baseline = _load_optional_baseline(
        custom_path,
        None,
        args.baseline_ref,
        errors,
    )
    if custom_baseline is not None:
        custom_fixture = load_manifest(custom_path)
        errors.extend(
            validate_fixture_append_only(
                custom_baseline,
                custom_fixture,
                label="custom wire fixture",
            )
        )

    for version in _binding_versions(candidate):
        _freeze_json_fixture(
            args.manifest.parent / _binding_fixture_name(version),
            base_ref=args.baseline_ref,
            label=f"binding snapshot V{version} fixture",
            errors=errors,
        )

    _freeze_json_fixture(
        args.manifest.parent / _MODEL_MESSAGE_FIXTURE_V1,
        base_ref=args.baseline_ref,
        label="model message V1 fixture",
        errors=errors,
    )

    for error in errors:
        print(error)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "load_git_baseline",
    "load_git_json_baseline",
    "load_manifest",
    "validate_append_only",
    "validate_custom_wire_fixture",
    "validate_external_fixtures",
    "validate_fixture_append_only",
    "validate_runtime_persistence_manifest",
    "validate_upgrade_fixtures",
]
