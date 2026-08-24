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
from linktools.ai.runtime import RuntimeDomain, RuntimeState
from linktools.ai.runtime.state._contracts import (
    ExecutionRecord,
    ExecutionTerminalCommit,
    IdempotencyRecord,
    IdempotencyTerminalUpdate,
    ResultRecord,
)
from linktools.ai.spec import AgentSpec
from linktools.ai.storage import ObjectRef, StoredPayload


def _binding_snapshot() -> AgentBindingSnapshot:
    output = bind_output()
    return AgentBindingSnapshot(
        version=1,
        agent_spec=AgentSpec("default", 1, "default"),
        agent_digest="b" * 64,
        output_type_module=output.value_type.__module__,
        output_type_qualname=output.value_type.__qualname__,
        output_schema_id=output.schema_id,
        output_schema_revision=output.schema_revision,
        output_schema_fingerprint=output.schema_fingerprint,
        local_runtime_capability_descriptors=(),
        binding_digest="a" * 64,
    )


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
            "execution",
            "tenant",
            None,
            "a" * 64,
            None,
            "execution",
            None,
            None,
            ExecutionLineageKind.RUN,
            ExecutionStatus.STARTED,
            1,
            1,
            1,
            None,
            {},
            now,
            now,
            False,
            False,
            _binding_snapshot(),
        )
        identity = IdempotencyRecord(
            "tenant",
            RuntimeDomain.EXECUTION,
            "execution.run",
            "a" * 64,
            "b" * 64,
            ResourceKind.EXECUTION,
            "execution",
            IdempotencyStatus.STARTED,
            None,
            None,
            now,
            now,
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
            "schema",
            1,
            "fingerprint",
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
