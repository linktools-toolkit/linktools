#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build-time validation for exact Runtime persistence fixtures."""

import argparse
import json
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
    ContextProjection,
    ContextProjectionItem,
    IdempotencyTerminalUpdate,
    OperationTerminalUpdate,
)
from linktools.ai.spec import AgentSpec
from linktools.ai.task import TaskNode
from pydantic_ai.messages import ModelRequest, UserPromptPart

_BINDING_FIXTURE = "runtime_agent_binding_snapshot_v1.json"
_CUSTOM_WIRE_FIXTURE = "runtime_custom_wire_v1.json"
_MODEL_MESSAGE_FIXTURE = "runtime_model_messages_v1.json"
_MATRIX_DIR = Path(__file__).with_name("matrix")


def _load_json(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid persistence fixture: {path}: {error}") from error


def _compare_fixture(
    path: Path,
    expected: object,
    label: str,
) -> tuple[str, ...]:
    if not path.is_file():
        return (f"missing exact persistence fixture: {path}",)
    try:
        value = _load_json(path)
    except ValueError as error:
        return (str(error),)
    if value != expected:
        return (f"{label} writer drifted from its fixture",)
    return ()


def _binding_fixture_value() -> AgentBindingSnapshot:
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


def validate_agent_binding_fixture(matrix_dir: str | Path) -> tuple[str, ...]:
    expected = _binding_fixture_value()
    path = Path(matrix_dir) / _BINDING_FIXTURE
    errors = list(
        _compare_fixture(path, expected.to_payload(), "AgentBindingSnapshot")
    )
    if errors:
        return tuple(errors)
    try:
        value = _load_json(path)
        decoded = AgentBindingSnapshot.from_payload(value)
    except (AIError, KeyError, TypeError, ValueError) as error:
        return (f"AgentBindingSnapshot fixture is not readable: {error}",)
    if decoded != expected or decoded.to_payload() != value:
        return ("AgentBindingSnapshot fixture semantics changed",)
    return ()


def _model_message_values() -> tuple[ModelRequest, ...]:
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


def validate_model_message_fixture(matrix_dir: str | Path) -> tuple[str, ...]:
    path = Path(matrix_dir) / _MODEL_MESSAGE_FIXTURE
    expected = json.loads(
        encode_model_messages(_model_message_values()).decode("utf-8")
    )
    errors = list(_compare_fixture(path, expected, "Model message"))
    if errors:
        return tuple(errors)
    try:
        decoded = decode_model_messages(
            canonical_json_bytes(cast(JsonValue, _load_json(path)))
        )
    except (AIError, KeyError, TypeError, ValueError) as error:
        return (f"Model message fixture is not readable: {error}",)
    if decoded != _model_message_values():
        return ("Model message fixture semantics changed",)
    return ()


def _custom_wire_values() -> dict[str, JsonValue]:
    task_node = TaskNode(
        "node",
        ("dependency",),
        input={"key": "value"},
        budget_cost=2,
    )
    task_wire = cast(dict[str, JsonValue], runtime_codec.encode_domain(task_node))
    task_wire = dict(task_wire)
    task_wire["schema"] = runtime_codec.CURRENT_DATA_VERSION
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
    version = runtime_codec.CURRENT_DATA_VERSION
    return {
        f"task_node@{version}": task_wire,
        (
            f"execution_terminal_commit@{version}:"
            "idempotency_terminal_update"
        ): runtime_codec.encode_domain(idempotency),
        (
            f"execution_terminal_commit@{version}:"
            "operation_terminal_update"
        ): runtime_codec.encode_domain(operation),
    }


def _decode_custom_wire_values(
    value: Mapping[str, object],
) -> tuple[TaskNode, IdempotencyTerminalUpdate, OperationTerminalUpdate]:
    version = runtime_codec.CURRENT_DATA_VERSION
    task_wire = value[f"task_node@{version}"]
    if not isinstance(task_wire, Mapping) or task_wire.get("schema") != version:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    task_payload = dict(task_wire)
    task_payload.pop("schema")
    task = runtime_codec.decode_domain(
        cast(JsonValue, task_payload),
        TaskNode,
    )
    idempotency = runtime_codec.decode_domain(
        cast(
            JsonValue,
            value[
                f"execution_terminal_commit@{version}:"
                "idempotency_terminal_update"
            ],
        ),
        IdempotencyTerminalUpdate,
    )
    operation = runtime_codec.decode_domain(
        cast(
            JsonValue,
            value[
                f"execution_terminal_commit@{version}:"
                "operation_terminal_update"
            ],
        ),
        OperationTerminalUpdate,
    )
    return task, idempotency, operation


def validate_custom_wire_fixture(matrix_dir: str | Path) -> tuple[str, ...]:
    path = Path(matrix_dir) / _CUSTOM_WIRE_FIXTURE
    expected = _custom_wire_values()
    errors = list(_compare_fixture(path, expected, "Custom wire"))
    if errors:
        return tuple(errors)
    try:
        value = _load_json(path)
        if not isinstance(value, Mapping):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        task, idempotency, operation = _decode_custom_wire_values(value)
    except (AIError, KeyError, TypeError, ValueError) as error:
        return (f"Custom wire fixture is not readable: {error}",)
    expected_task, expected_idempotency, expected_operation = (
        _decode_custom_wire_values(expected)
    )
    if (
        task != expected_task
        or idempotency != expected_idempotency
        or operation != expected_operation
    ):
        return ("Custom wire fixture semantics changed",)
    return ()


def _validate_generic_writer_closure() -> tuple[str, ...]:
    valid = ContextProjection((), "d" * 64)
    try:
        payload = runtime_codec._encode_persisted_domain(valid)
        decoded = runtime_codec._decode_enveloped_domain(
            runtime_codec.encode_envelope(
                {
                    "type": runtime_codec.wire_type_id(valid),
                    "payload": payload,
                }
            ),
            ContextProjection,
        )
    except (AIError, KeyError, TypeError, ValueError):
        return ("generic persistence writer failed valid round-trip",)
    if decoded != valid:
        return ("generic persistence writer failed valid round-trip",)

    invalid = ContextProjection(
        cast("tuple[ContextProjectionItem, ...]", ("invalid",)),
        "d" * 64,
    )
    try:
        runtime_codec._encode_persisted_domain(invalid)
    except TypeError:
        return ()
    except Exception:
        return ("generic persistence writer returned the wrong failure type",)
    return ("generic persistence writer accepted unreadable runtime value",)


def validate_exact_fixtures(matrix_dir: str | Path) -> tuple[str, ...]:
    root = Path(matrix_dir)
    return (
        *validate_agent_binding_fixture(root),
        *validate_custom_wire_fixture(root),
        *validate_model_message_fixture(root),
        *_validate_generic_writer_closure(),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix-dir", type=Path, default=_MATRIX_DIR)
    args = parser.parse_args()
    errors = validate_exact_fixtures(args.matrix_dir)
    for error in errors:
        print(error)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "main",
    "validate_agent_binding_fixture",
    "validate_custom_wire_fixture",
    "validate_exact_fixtures",
    "validate_model_message_fixture",
]
