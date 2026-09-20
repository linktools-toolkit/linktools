#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime usage aggregation over durable model-request facts."""

from types import SimpleNamespace

import pytest

from linktools.ai.core import (
    ExecutionStatus,
    HmacCursorSigner,
    Principal,
    TaskStatus,
    TenantAuthorizationPolicy,
    UsageMetrics,
    step_conversation_id,
    step_run_id,
)
from linktools.ai.runtime import RuntimeHistory, UsageReadCutoff, UsageSummary
from linktools.ai.runtime._history import StepExecutionHistoryReader
from linktools.ai.runtime.service_api import ExecutionHistoryService
from linktools.ai.runtime.state import RuntimeDomain
from linktools.ai.runtime.state._contracts import (
    ContextProjection,
    ModelInteractionRecord,
    RuntimePayloadRef,
)
from linktools.ai.runtime.state._step_contracts import RunRecord
from linktools.ai.storage import StoredPayload


def _interaction(
    run_id: str,
    sequence: int,
    *,
    status: str,
    usage: UsageMetrics | None,
    duration_ns: int,
    output_retry_index: int | None = None,
) -> ModelInteractionRecord:
    return ModelInteractionRecord(
        run_id,
        0,
        sequence,
        "agent",
        output_retry_index,
        {"provider": "test", "model": "usage"},
        ContextProjection(()),
        RuntimePayloadRef(StoredPayload.inline_bytes(b"{}"), None),
        ContextProjection(()) if status == "SUCCEEDED" else None,
        status,
        "MODEL_FAILED" if status == "FAILED" else None,
        duration_ns,
        usage,
    )


class _Executions:
    def __init__(self, record: object) -> None:
        self.record = record

    async def get(self, execution_id: str, *, tenant_id: str):
        assert tenant_id == "tenant"
        if execution_id != self.record.execution_id:
            return None
        return self.record


class _UsageStore:
    def __init__(
        self,
        run: RunRecord,
        interactions: tuple[ModelInteractionRecord, ...],
        *,
        high_water: int,
    ) -> None:
        self.run = run
        self.interactions = interactions
        self.high_water = high_water

    async def get_run(self, *, run_id: str):
        return self.run if run_id == self.run.run_id else None

    async def model_interaction_count(self, *, run_id: str) -> int:
        assert run_id == self.run.run_id
        return self.high_water

    async def list_model_interactions(
        self,
        *,
        run_id: str,
        after_request_sequence: int | None = None,
        limit: int | None = None,
    ):
        assert run_id == self.run.run_id
        after = 0 if after_request_sequence is None else after_request_sequence
        selected = tuple(
            value
            for value in self.interactions
            if value.request_sequence > after
        )
        return list(selected if limit is None else selected[:limit])


@pytest.mark.asyncio
async def test_usage_reads_only_captured_model_interaction_prefix() -> None:
    namespace = "runtime-usage"
    tenant_id = "tenant"
    execution_id = "execution"
    run_id = step_run_id(
        namespace=namespace,
        tenant_id=tenant_id,
        execution_id=execution_id,
        segment_sequence=1,
    )
    conversation_id = step_conversation_id(
        namespace=namespace,
        tenant_id=tenant_id,
        execution_id=execution_id,
    )
    run = RunRecord(
        run_id,
        conversation_id,
        agent_name="agent",
        metadata={"segment_sequence": "1", "agent_name": "agent"},
    )
    interactions = (
        _interaction(
            run_id,
            1,
            status="SUCCEEDED",
            usage=UsageMetrics(
                input_tokens=10,
                output_tokens=20,
                cache_read_tokens=3,
                cache_write_tokens=4,
            ),
            duration_ns=100,
        ),
        _interaction(
            run_id,
            2,
            status="FAILED",
            usage=None,
            duration_ns=200,
            output_retry_index=1,
        ),
        _interaction(
            run_id,
            3,
            status="CANCELLED",
            usage=UsageMetrics(
                input_tokens=1,
                output_tokens=2,
                cache_read_tokens=5,
                cache_write_tokens=6,
            ),
            duration_ns=300,
        ),
        _interaction(
            run_id,
            4,
            status="SUCCEEDED",
            usage=UsageMetrics(input_tokens=999, output_tokens=999),
            duration_ns=999,
        ),
    )
    record = SimpleNamespace(
        execution_id=execution_id,
        binding_kind="agent",
        status=ExecutionStatus.STARTED,
        agent_run_sequence=1,
    )
    reader = StepExecutionHistoryReader(
        namespace=namespace,
        executions=_Executions(record),  # type: ignore[arg-type]
        store=_UsageStore(run, interactions, high_water=3),  # type: ignore[arg-type]
        cursor_signer=HmacCursorSigner("usage", b"usage-key"),
    )

    usage = await reader.usage(execution_id, tenant_id=tenant_id)

    assert usage == UsageSummary(
        logical_requests=3,
        succeeded_requests=1,
        failed_requests=1,
        cancelled_requests=1,
        output_correction_retries=1,
        input_tokens=11,
        output_tokens=22,
        cache_read_tokens=8,
        cache_write_tokens=10,
        model_duration_ns=600,
        unknown_usage_requests=1,
        transport_retries=None,
        cutoffs=(UsageReadCutoff(execution_id, 1, 3),),
    )


class _UsageService:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def usage(
        self,
        execution_id: str,
        *,
        principal: Principal,
    ) -> UsageSummary:
        del principal
        self.calls.append(execution_id)
        if execution_id == "root":
            return UsageSummary(
                logical_requests=1,
                succeeded_requests=1,
                input_tokens=5,
                output_tokens=7,
                cutoffs=(UsageReadCutoff("root", 1, 1),),
            )
        if execution_id == "child":
            return UsageSummary(
                logical_requests=1,
                failed_requests=1,
                unknown_usage_requests=1,
                cutoffs=(UsageReadCutoff("child", 1, 1),),
            )
        raise AssertionError(execution_id)


class _GraphExecutions:
    async def get(self, execution_id: str, *, tenant_id: str):
        assert tenant_id == "tenant"
        if execution_id == "root":
            return SimpleNamespace(
                execution_id="root",
                tenant_id=tenant_id,
                parent_execution_id=None,
            )
        if execution_id == "child":
            return SimpleNamespace(
                execution_id="child",
                tenant_id=tenant_id,
                parent_execution_id="root",
            )
        return None

    async def list_children(self, execution_id: str, *, tenant_id: str):
        assert tenant_id == "tenant"
        if execution_id == "root":
            return (
                SimpleNamespace(
                    execution_id="child",
                    tenant_id=tenant_id,
                    parent_execution_id="root",
                ),
            )
        return ()


class _GraphUsageHistory(RuntimeHistory):
    async def task_graph(self, graph_id: str, *, principal: Principal):
        assert graph_id == "graph"
        assert principal.tenant_id == "tenant"
        return SimpleNamespace(
            node_states=(
                SimpleNamespace(
                    execution_id="root",
                    status=TaskStatus.SUCCEEDED,
                ),
                SimpleNamespace(
                    execution_id=None,
                    status=TaskStatus.SUCCEEDED,
                ),
            )
        )


@pytest.mark.asyncio
async def test_graph_usage_counts_only_linked_execution_tree() -> None:
    service = _UsageService()
    history = _GraphUsageHistory(
        service,  # type: ignore[arg-type]
        tenant_id="tenant",
        executions=_GraphExecutions(),  # type: ignore[arg-type]
        authorization=TenantAuthorizationPolicy("tenant"),
    )
    principal = Principal("caller", "tenant", "service")

    usage = await history.graph_usage("graph", principal=principal)

    assert service.calls == ["root", "child"]
    assert usage.logical_requests == 2
    assert usage.succeeded_requests == 1
    assert usage.failed_requests == 1
    assert usage.input_tokens == 5
    assert usage.output_tokens == 7
    assert usage.unknown_usage_requests == 1
    assert usage.unrecorded_executions == 1
    assert usage.transport_retries is None
    assert usage.cutoffs == (
        UsageReadCutoff("child", 1, 1),
        UsageReadCutoff("root", 1, 1),
    )
