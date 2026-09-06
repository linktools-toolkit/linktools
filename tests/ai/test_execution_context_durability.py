#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Durable Execution correlation and recovery regressions."""

from dataclasses import replace
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from linktools.ai.agent import AgentBindingSnapshot
from linktools.ai.core import (
    ExecutionLineageKind,
    ExecutionStatus,
    Principal,
)
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime._local import LocalExecutionBackend
from linktools.ai.runtime.service_api import ExecutionRequest
from linktools.ai.runtime.state._codec import (
    _decode_enveloped_domain,
    _encode_persisted_domain,
    encode_envelope,
)
from linktools.ai.runtime.state._contracts import (
    ExecutionRecord,
    RecoveryExecutionInput,
    RecoveryIdempotencyInput,
)
from linktools.ai.spec import AgentSpec


def _binding() -> AgentBindingSnapshot:
    return AgentBindingSnapshot(
        version=1,
        agent_spec=AgentSpec("agent", model="default"),
        model={"provider": "test", "model": "fixture"},
        selected=(),
        subagents=(),
        output_mode="text",
        output_schema={"type": "string"},
        binding_digest="a" * 64,
    )


def _execution(*, correlation: dict[str, str | int]) -> ExecutionRecord:
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    binding = _binding()
    return ExecutionRecord(
        execution_id="execution",
        tenant_id="tenant",
        session_id=None,
        binding_digest=binding.binding_digest,
        parent_execution_id=None,
        root_execution_id="execution",
        source_execution_id=None,
        base_execution_id=None,
        lineage_kind=ExecutionLineageKind.RUN,
        status=ExecutionStatus.PENDING_START,
        revision=0,
        event_sequence=0,
        agent_run_sequence=0,
        error_code=None,
        safe_error_details={},
        created_at=now,
        updated_at=now,
        mode="run",
        planning=False,
        thinking=False,
        binding=binding,
        correlation=correlation,
    )


def _recovery_input(*, correlation: dict[str, str | int]) -> RecoveryExecutionInput:
    binding = _binding()
    return RecoveryExecutionInput(
        user_prompt="hello",
        user_prompt_codec="text",
        principal_id="user",
        principal_kind="user",
        session_id=None,
        memory_scope=None,
        binding_digest=binding.binding_digest,
        lineage_kind=ExecutionLineageKind.RUN.value,
        parent_execution_id=None,
        root_execution_id="execution",
        source_execution_id=None,
        base_execution_id=None,
        conversation_step_run_id=None,
        idempotency=RecoveryIdempotencyInput(
            scope="execution.run",
            idempotency_key_digest="b" * 64,
            request_digest="c" * 64,
        ),
        mode="run",
        planning=False,
        thinking=False,
        binding=binding,
        correlation=correlation,
    )


def _backend(execution: ExecutionRecord) -> LocalExecutionBackend:
    backend = object.__new__(LocalExecutionBackend)
    backend._accepting = True
    backend._tenant_id = execution.tenant_id
    backend._catalog = SimpleNamespace(
        binding=lambda digest: SimpleNamespace(
            snapshot=execution.binding,
            digest=digest,
        )
    )
    return backend


@pytest.mark.asyncio
async def test_local_start_rejects_correlation_drift_from_durable_execution() -> None:
    execution = _execution(correlation={"trace_id": "durable", "attempt": 1})
    backend = _backend(execution)
    request = ExecutionRequest(
        user_prompt="hello",
        user_prompt_codec="text",
        principal=Principal("user", "tenant"),
        idempotency_key="execution-correlation-start-0001",
        memory_scope=None,
        mode="run",
        planning=False,
        thinking=False,
        correlation={"trace_id": "request", "attempt": 1},
    )

    with pytest.raises(AIError) as raised:
        await backend._validate_start(request, execution)

    assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


def test_recovery_identity_rejects_correlation_drift() -> None:
    execution = _execution(correlation={"trace_id": "durable"})
    backend = _backend(execution)
    recovery = _recovery_input(correlation={"trace_id": "other"})

    with pytest.raises(AIError) as raised:
        backend._validate_recovery_identity(execution, recovery)

    assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


def test_recovery_execution_input_v1_correlation_wire_round_trips() -> None:
    empty = _recovery_input(correlation={})
    payload = _encode_persisted_domain(empty)
    assert isinstance(payload, dict)
    assert payload["$dataclass"] == "recovery_execution_input"
    fields = payload["fields"]
    assert isinstance(fields, dict)
    assert "correlation" not in fields

    decoded = _decode_enveloped_domain(
        encode_envelope({"type": "recovery_execution_input", "payload": payload}),
        RecoveryExecutionInput,
    )
    assert decoded == empty
    assert dict(decoded.correlation) == {}

    populated = _recovery_input(correlation={"trace_id": "trace", "attempt": 2})
    populated_payload = _encode_persisted_domain(populated)
    assert isinstance(populated_payload, dict)
    populated_fields = populated_payload["fields"]
    assert isinstance(populated_fields, dict)
    assert "correlation" in populated_fields
    assert (
        _decode_enveloped_domain(
            encode_envelope(
                {"type": "recovery_execution_input", "payload": populated_payload}
            ),
            RecoveryExecutionInput,
        )
        == populated
    )


def test_execution_correlation_is_normalized_and_immutable() -> None:
    execution = _execution(correlation={"trace_id": "trace", "attempt": 3})
    assert dict(execution.correlation) == {"attempt": 3, "trace_id": "trace"}

    changed = replace(execution, correlation={"trace_id": "other"})
    assert dict(execution.correlation) == {"attempt": 3, "trace_id": "trace"}
    assert dict(changed.correlation) == {"trace_id": "other"}
