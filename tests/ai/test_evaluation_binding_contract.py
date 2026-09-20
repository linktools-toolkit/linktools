#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluation binding ownership and replay regression coverage."""

from dataclasses import replace
from datetime import datetime, timezone

import pytest

from linktools.ai.agent import AgentBindingSnapshot
from linktools.ai.core import (
    EvaluationStatus,
    ExecutionLineageKind,
    ExecutionStatus,
    Principal,
)
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime._evaluation import DefaultEvaluationService
from linktools.ai.runtime.service_api import (
    ExecutionHandle,
    ExecutionRequest,
    ReplayEvaluationRequest,
    StartEvaluationRequest,
)
from linktools.ai.runtime.state import RuntimeState
from linktools.ai.runtime.state._contracts import (
    EvaluationRecord,
    ExecutionRecord,
    StoredUserInput,
)
from linktools.ai.spec import AgentSpec
from linktools.ai.storage import StoredPayload


def _binding(agent_id: str = "agent") -> AgentBindingSnapshot:
    return AgentBindingSnapshot(
        agent_spec=AgentSpec(agent_id),
        base_model={"model_identity": "test:model"},
        selected=(),
        subagents=(),
        output_mode="text",
        output_schema={
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
            "additionalProperties": False,
        },
    )


def _execution(
    binding: AgentBindingSnapshot,
    *,
    execution_id: str,
) -> ExecutionRecord:
    now = datetime.now(timezone.utc)
    return ExecutionRecord(
        execution_id=execution_id,
        session_id=None,
        parent_execution_id=None,
        root_execution_id=execution_id,
        source_execution_id=None,
        base_execution_id=None,
        lineage_kind=ExecutionLineageKind.RUN,
        status=ExecutionStatus.SUCCEEDED,
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
        binding=binding,
        principal_id="principal",
        principal_kind="service",
        stored_user_input=StoredUserInput(
            "text",
            StoredPayload.inline_text("evaluation"),
        ),
    )


class _RecordingExecution:
    def __init__(self) -> None:
        self.binding_digest: str | None = None
        self.binding_snapshot: AgentBindingSnapshot | None = None
        self.request: ExecutionRequest | None = None

    async def start(
        self,
        binding_digest: str,
        request: ExecutionRequest,
        *,
        dependency_hold_id: str | None = None,
        binding_snapshot: AgentBindingSnapshot | None = None,
    ) -> ExecutionHandle:
        del dependency_hold_id
        self.binding_digest = binding_digest
        self.binding_snapshot = binding_snapshot
        self.request = request
        return ExecutionHandle("execution")


class _Allow:
    async def authorize(self, *args: object) -> None:
        del args


@pytest.mark.asyncio
async def test_evaluation_start_persists_source_execution_identity() -> None:
    state = RuntimeState.in_memory()
    await state.initialize(namespace="evaluation", tenant_id="tenant")
    binding = _binding()
    execution = _RecordingExecution()
    service = DefaultEvaluationService(
        state.evaluation,
        state.execution.executions,
        _Allow(),  # type: ignore[arg-type]
        execution,  # type: ignore[arg-type]
    )
    try:
        handle = await service.start(
            binding.binding_digest,
            StartEvaluationRequest(
                Principal("principal", "tenant"),
                "dataset",
                "evaluation-memory",
                "evaluation-start",
            ),
            binding_snapshot=binding,
        )

        record = await state.evaluation.records.get(
            handle.evaluation_id,
            tenant_id="tenant",
        )
        assert record is not None
        assert record.execution_id == "execution"
        assert execution.binding_snapshot == binding
    finally:
        await state.close()


@pytest.mark.asyncio
async def test_evaluation_replay_uses_historical_execution_binding() -> None:
    state = RuntimeState.in_memory()
    await state.initialize(namespace="evaluation", tenant_id="tenant")
    historical = _binding("agent")
    source = _execution(historical, execution_id="source-execution")
    now = datetime.now(timezone.utc)
    evaluation = EvaluationRecord(
        evaluation_id="evaluation",
        execution_id=source.execution_id,
        dataset_digest="dataset",
        status=EvaluationStatus.SUCCEEDED,
        revision=1,
        created_at=now,
        updated_at=now,
    )
    execution = _RecordingExecution()
    try:
        await state.execution.executions.create(source)
        await state.evaluation.records.create(evaluation)
        service = DefaultEvaluationService(
            state.evaluation,
            state.execution.executions,
            _Allow(),  # type: ignore[arg-type]
            execution,  # type: ignore[arg-type]
        )

        replayed = await service.replay(
            "agent",
            evaluation.evaluation_id,
            ReplayEvaluationRequest(
                Principal("principal", "tenant"),
                "evaluation-memory",
                "evaluation-replay",
            ),
        )

        assert replayed.execution_id == "execution"
        assert execution.binding_digest == source.binding_digest
        assert execution.binding_snapshot == source.binding
        assert execution.request is not None
        assert execution.request.user_prompt == "evaluation:dataset"
    finally:
        await state.close()


@pytest.mark.asyncio
async def test_evaluation_status_cannot_lead_source_execution() -> None:
    state = RuntimeState.in_memory()
    await state.initialize(namespace="evaluation", tenant_id="tenant")
    source = replace(
        _execution(_binding("agent"), execution_id="source-execution"),
        status=ExecutionStatus.PENDING_START,
        revision=0,
        event_sequence=0,
        agent_run_sequence=0,
    )
    now = datetime.now(timezone.utc)
    await state.execution.executions.create(source)
    await state.evaluation.records.create(
        EvaluationRecord(
            evaluation_id="state-ahead",
            execution_id=source.execution_id,
            dataset_digest="dataset",
            status=EvaluationStatus.RUNNING,
            revision=1,
            created_at=now,
            updated_at=now,
        )
    )
    service = DefaultEvaluationService(
        state.evaluation,
        state.execution.executions,
        _Allow(),  # type: ignore[arg-type]
        _RecordingExecution(),  # type: ignore[arg-type]
    )
    try:
        with pytest.raises(AIError) as raised:
            await service.inspect(
                "state-ahead",
                principal=Principal("principal", "tenant"),
            )
        assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
    finally:
        await state.close()


@pytest.mark.parametrize(
    "status",
    (EvaluationStatus.PENDING, EvaluationStatus.SUCCEEDED),
)
@pytest.mark.asyncio
async def test_evaluation_missing_source_execution_fails_closed(
    status: EvaluationStatus,
) -> None:
    state = RuntimeState.in_memory()
    await state.initialize(namespace="evaluation", tenant_id="tenant")
    now = datetime.now(timezone.utc)
    await state.evaluation.records.create(
        EvaluationRecord(
            evaluation_id="missing-source",
            execution_id="missing-execution",
            dataset_digest="dataset",
            status=status,
            revision=0,
            created_at=now,
            updated_at=now,
        )
    )
    service = DefaultEvaluationService(
        state.evaluation,
        state.execution.executions,
        _Allow(),  # type: ignore[arg-type]
        _RecordingExecution(),  # type: ignore[arg-type]
    )
    try:
        with pytest.raises(AIError) as raised:
            await service.inspect(
                "missing-source",
                principal=Principal("principal", "tenant"),
            )
        assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
    finally:
        await state.close()
