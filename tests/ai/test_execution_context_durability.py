#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Durable Execution correlation and recovery-owner regressions."""

from dataclasses import replace
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from linktools.ai.agent import AgentBindingSnapshot
from linktools.ai.core import ExecutionLineageKind, ExecutionStatus, Principal
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime._local import LocalExecutionBackend
from linktools.ai.runtime.service_api import ExecutionRequest
from linktools.ai.runtime.state._codec import decode_domain, encode_domain
from linktools.ai.runtime.state._contracts import ExecutionRecord, StoredUserInput
from linktools.ai.spec import AgentSpec
from linktools.ai.storage import StoredPayload


def _binding() -> AgentBindingSnapshot:
    return AgentBindingSnapshot(
        version=1,
        agent_spec=AgentSpec("agent", model="default"),
        base_model={"provider": "test", "model": "fixture"},
        selected=(),
        subagents=(),
        output_mode="text",
        output_schema={"type": "string"},
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
        principal_id="user",
        principal_kind="user",
        stored_user_input=StoredUserInput(
            1,
            "text",
            StoredPayload.inline_text("hello"),
        ),
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
async def test_local_start_accepts_matching_durable_correlation() -> None:
    execution = _execution(correlation={"trace_id": "durable", "attempt": 1})
    backend = _backend(execution)
    request = ExecutionRequest(
        user_prompt="hello",
        principal=Principal("user", "tenant"),
        idempotency_key="execution-correlation-start-0001",
        memory_scope=None,
        mode="run",
        planning=False,
        thinking=False,
        correlation={"trace_id": "durable", "attempt": 1},
    )

    await backend._validate_start(request, execution)


@pytest.mark.asyncio
async def test_local_start_rejects_correlation_drift_from_durable_execution() -> None:
    execution = _execution(correlation={"trace_id": "durable", "attempt": 1})
    backend = _backend(execution)
    request = ExecutionRequest(
        user_prompt="hello",
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

    assert raised.value.code is ErrorCode.IDEMPOTENCY_CONFLICT
    assert dict(execution.correlation) == {"attempt": 1, "trace_id": "durable"}


def test_execution_record_correlation_wire_round_trips_current_shape() -> None:
    empty = _execution(correlation={})
    payload = encode_domain(empty)
    assert isinstance(payload, dict)
    assert payload["$dataclass"] == "execution_record"
    fields = payload["fields"]
    assert isinstance(fields, dict)
    assert "correlation" in fields

    decoded = decode_domain(payload, ExecutionRecord)
    assert decoded == empty
    assert dict(decoded.correlation) == {}

    populated = _execution(correlation={"trace_id": "trace", "attempt": 2})
    populated_payload = encode_domain(populated)
    assert isinstance(populated_payload, dict)
    populated_fields = populated_payload["fields"]
    assert isinstance(populated_fields, dict)
    assert "correlation" in populated_fields
    assert decode_domain(populated_payload, ExecutionRecord) == populated


def test_execution_correlation_is_normalized_and_immutable() -> None:
    execution = _execution(correlation={"trace_id": "trace", "attempt": 3})
    assert dict(execution.correlation) == {"attempt": 3, "trace_id": "trace"}

    changed = replace(execution, correlation={"trace_id": "other"})
    assert dict(execution.correlation) == {"attempt": 3, "trace_id": "trace"}
    assert dict(changed.correlation) == {"trace_id": "other"}
