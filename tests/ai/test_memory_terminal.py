#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""In-memory terminal execution persistence checks."""

from dataclasses import replace
from datetime import datetime, timezone

import pytest
from linktools.ai.agent import AgentBindingSnapshot
from linktools.ai.agent._output import bind_output
from linktools.ai.core import (
    ExecutionEventType,
    ExecutionLineageKind,
    ExecutionStatus,
    IdempotencyStatus,
    ResourceKind,
    StopReason,
    UsageMetrics,
)
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime import RuntimeState
from linktools.ai.runtime.state._contracts import (
    ExecutionRecord,
    ExecutionTerminalCommit,
    IdempotencyRecord,
    IdempotencyTerminalUpdate,
    ResultRecord,
)
from linktools.ai.spec import AgentSpec
from linktools.ai.storage import ObjectRef, StoredPayload
from ._runtime_test_helpers import execution_owner_fields


def _binding_snapshot() -> AgentBindingSnapshot:
    output = bind_output()
    return AgentBindingSnapshot(
        agent_spec=AgentSpec("default"),
        base_model={"route_id": "default", "model_identity": "test:model"},
        selected=(),
        subagents=(),
        output_mode=output.mode,
        output_schema=output.schema_definition,
    )


@pytest.mark.asyncio
async def test_execution_idempotency_repository_owns_resource_kind() -> None:
    state = RuntimeState.in_memory()
    await state.initialize(namespace="idempotency-owner", tenant_id="tenant")
    try:
        now = datetime.now(timezone.utc)
        record = IdempotencyRecord(
            tenant_id="tenant",
            scope="execution.run",
            idempotency_key_digest="a" * 64,
            request_digest="b" * 64,
            resource_kind=ResourceKind.EVALUATION,
            resource_id="evaluation",
            status=IdempotencyStatus.RESERVED,
            result_digest=None,
            error_code=None,
            created_at=now,
            updated_at=now,
        )
        with pytest.raises(AIError) as error:
            await state.execution.idempotency.reserve(record)
        assert error.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
    finally:
        await state.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("payload_kind", ["inline", "object"])
async def test_in_memory_terminal_commit_validates_success_result(
    payload_kind: str,
) -> None:
    state = RuntimeState.in_memory()
    await state.initialize(namespace="memory-terminal", tenant_id="tenant")
    try:
        now = datetime.now(timezone.utc)
        execution = ExecutionRecord(
            execution_id="execution",
            tenant_id="tenant",
            session_id=None,
            parent_execution_id=None,
            root_execution_id="execution",
            source_execution_id=None,
            base_execution_id=None,
            lineage_kind=ExecutionLineageKind.RUN,
            status=ExecutionStatus.STARTED,
            revision=1,
            event_sequence=1,
            agent_run_sequence=1,
            error_code=None,
            safe_error_details={},
            created_at=now,
            updated_at=now,
            mode="run",
            planning=False,
            thinking=False,
            binding=_binding_snapshot(),
            **execution_owner_fields(),
        )
        identity = IdempotencyRecord(
            tenant_id="tenant",
            scope="execution.run",
            idempotency_key_digest="a" * 64,
            request_digest="b" * 64,
            resource_kind=ResourceKind.EXECUTION,
            resource_id="execution",
            status=IdempotencyStatus.STARTED,
            result_digest=None,
            error_code=None,
            created_at=now,
            updated_at=now,
        )
        await state.execution.executions.create(execution)
        await state.execution.idempotency.reserve(identity)

        result_ref = ObjectRef("memory", "result", "c" * 64, 0)
        result_payload = (
            StoredPayload.inline_json({"text": "result"})
            if payload_kind == "inline"
            else StoredPayload.object(result_ref)
        )
        terminal = replace(
            execution,
            status=ExecutionStatus.SUCCEEDED,
            revision=2,
            event_sequence=2,
            updated_at=now,
        )
        result = ResultRecord(
            "execution",
            "tenant",
            result_payload,
            StopReason.END_TURN,
            UsageMetrics(),
            now,
        )
        committed = await state.execution.executions.commit_terminal(
            ExecutionTerminalCommit(
                1,
                1,
                terminal,
                result,
                ExecutionEventType.EXECUTION_SUCCEEDED,
                {},
                IdempotencyTerminalUpdate(
                    identity.scope,
                    identity.idempotency_key_digest,
                    identity.status,
                    IdempotencyStatus.COMPLETED,
                    identity.request_digest,
                    result_payload.digest,
                    None,
                ),
            )
        )

        assert committed.execution.status is ExecutionStatus.SUCCEEDED
        assert committed.result == result
        persisted_execution = await state.execution.executions.get(
            "execution",
            tenant_id="tenant",
        )
        persisted_result = await state.execution.executions.get_result(
            "execution",
            tenant_id="tenant",
        )
        assert persisted_execution is not None
        assert persisted_execution.status is ExecutionStatus.SUCCEEDED
        assert persisted_execution.result == result
        assert persisted_result == result
    finally:
        await state.close()
