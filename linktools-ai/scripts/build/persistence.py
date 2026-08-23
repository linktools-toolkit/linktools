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

_BINDING_FIXTURE_V1 = "runtime_agent_binding_snapshot_v1.json"
_MODEL_MESSAGE_FIXTURE_V1 = "runtime_model_messages_v1.json"
_CUSTOM_WIRE_FIXTURE_V1 = "runtime_custom_wire_v1.json"


def load_manifest(path: str | Path) -> dict[str, JsonValue]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
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
            if candidate_external.get(name) != value:
                errors.append(f"historical external contract changed: {name}")
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


def _agent_binding_snapshot_fixture_value() -> AgentBindingSnapshot:
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
    if binding_contract.get("owner") != "agent" or binding_contract.get("version") != 1:
        return ("agent_binding_snapshot V1 external contract changed",)
    binding_path = root / _BINDING_FIXTURE_V1
    if not binding_path.is_file():
        return (f"missing external persistence fixture: {binding_path}",)
    try:
        binding_value = _load_json(binding_path)
    except ValueError as error:
        return (str(error),)
    expected_binding = _agent_binding_snapshot_fixture_value()
    if binding_value != expected_binding.to_payload():
        return ("agent_binding_snapshot writer drifted from the frozen V1 fixture",)
    try:
        decoded_binding = AgentBindingSnapshot.from_payload(binding_value)
    except Exception as error:
        return (f"agent_binding_snapshot V1 fixture is no longer readable: {error}",)
    if decoded_binding != expected_binding:
        return ("agent_binding_snapshot V1 fixture decodes to different semantics",)

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
        result_digest=None,
        error_code=None,
    )
    operation = OperationTerminalUpdate(
        operation_id="operation",
        expected_status=OperationStatus.RUNNING,
        next_status=OperationStatus.SUCCEEDED,
        result_ref=None,
        result_digest=None,
        error_code=None,
    )
    return {
        "task_node": runtime_codec._encode_persisted_domain(task_node),
        "idempotency_terminal_update": runtime_codec._encode_external(
            idempotency,
            runtime_codec._CURRENT_CODEC,
        ),
        "operation_terminal_update": runtime_codec._encode_external(
            operation,
            runtime_codec._CURRENT_CODEC,
        ),
    }


def validate_custom_wire_fixture(matrix_dir: str | Path) -> tuple[str, ...]:
    path = Path(matrix_dir) / _CUSTOM_WIRE_FIXTURE_V1
    if not path.is_file():
        return (f"missing custom persistence fixture: {path}",)
    try:
        value = _load_json(path)
    except ValueError as error:
        return (str(error),)
    expected = _custom_wire_values()
    return () if value == expected else (
        "Runtime custom V1 wire drifted from its frozen fixture",
    )


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
            decoded = runtime_codec._decode_dataclass(
                {
                    "$dataclass": wire_id,
                    "schema": revision,
                    "fields": dict(raw_fields),
                },
                target,
                runtime_codec._CURRENT_CODEC,
                persisted=True,
            )
            rewritten = runtime_codec._encode_persisted_domain(decoded)
        except Exception as error:
            errors.append(f"upgrade fixture is no longer readable: {key}: {error}")
            continue
        if not isinstance(rewritten, Mapping):
            errors.append(f"upgrade fixture rewrite is invalid: {key}")
            continue
        rewritten_fields = rewritten.get("fields")
        if not isinstance(rewritten_fields, Mapping) or dict(rewritten_fields) != dict(
            expected_fields
        ):
            errors.append(f"upgrade fixture rewrite changed: {key}")
    return tuple(errors)


def _git(repository: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ("git", "-C", str(repository), *arguments),
        check=False,
        capture_output=True,
        text=True,
    )


def load_git_baseline(
    path_value: str | Path,
    *,
    base_ref: str | None = None,
) -> dict[str, JsonValue] | None:
    """Load one JSON contract at the immutable branch/release comparison point."""
    path = Path(path_value).resolve()
    repository = Path(__file__).resolve().parents[3]
    if not (repository / ".git").exists():
        return None
    try:
        relative = path.relative_to(repository.resolve()).as_posix()
    except ValueError:
        return None

    ref = base_ref or os.environ.get(
        "LINKTOOLS_PERSISTENCE_BASE_REF", "origin/master"
    )
    verified = _git(repository, "rev-parse", "--verify", f"{ref}^{{commit}}")
    if verified.returncode != 0:
        return None
    head = _git(repository, "rev-parse", "HEAD")
    base = _git(repository, "merge-base", "HEAD", ref)
    if head.returncode != 0 or base.returncode != 0:
        return None
    head_sha = head.stdout.strip()
    base_sha = base.stdout.strip()
    if not head_sha or not base_sha:
        return None
    if head_sha == base_sha:
        parent = _git(repository, "rev-parse", "HEAD^")
        if parent.returncode != 0 or not parent.stdout.strip():
            return None
        base_sha = parent.stdout.strip()

    historical = _git(repository, "show", f"{base_sha}:{relative}")
    if historical.returncode != 0:
        return None
    try:
        value = json.loads(historical.stdout)
    except json.JSONDecodeError as error:
        raise ValueError(f"historical JSON contract is invalid at {base_sha}") from error
    if not isinstance(value, dict):
        raise ValueError(f"historical JSON contract is not an object at {base_sha}")
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

    binding_path = args.manifest.parent / _BINDING_FIXTURE_V1
    binding_baseline = _load_optional_baseline(
        binding_path,
        None,
        args.baseline_ref,
        errors,
    )
    if binding_baseline is not None:
        binding_fixture = load_manifest(binding_path)
        errors.extend(
            validate_fixture_append_only(
                binding_baseline,
                binding_fixture,
                label="binding snapshot fixture",
            )
        )

    for error in errors:
        print(error)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "load_git_baseline",
    "load_manifest",
    "validate_append_only",
    "validate_custom_wire_fixture",
    "validate_external_fixtures",
    "validate_fixture_append_only",
    "validate_runtime_persistence_manifest",
    "validate_upgrade_fixtures",
]
