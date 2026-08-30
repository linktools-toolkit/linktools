#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regression coverage for Runtime execution error-contract closure."""

import asyncio
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
from linktools.ai.agent import AgentBindingSnapshot
from linktools.ai.capability import SkillSourceRegistry
from linktools.ai.core import ExecutionLineageKind, ExecutionStatus, OperationStatus
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime._agent_executor import AgentExecutor
from linktools.ai.runtime._capabilities import compose_platform_capabilities
from linktools.ai.runtime._execution import CancelEffectOutcome, DefaultExecutionService
from linktools.ai.runtime.service_api import CancelExecutionRequest
from linktools.ai.runtime.state import ExecutionRecord
from linktools.ai.observe import MiddlewarePipeline
from linktools.ai.spec import AgentSpec
from linktools.ai.workspace import RepositoryInstructions, trusted_workspace_principal
from pydantic_ai_harness.compaction import DeduplicateFileReads
from pydantic_ai_harness.step_persistence import InMemoryStepStore


class _EmptyRepositoryInstructionResolver:
    async def resolve(
        self,
        path: str | Path = ".",
        *,
        exclude_sources: frozenset[str] = frozenset(),
    ) -> RepositoryInstructions:
        del path, exclude_sources
        return RepositoryInstructions(())


def _binding_snapshot() -> AgentBindingSnapshot:
    return AgentBindingSnapshot(
        version=1,
        agent_spec=AgentSpec("agent", model="default"),
        model={"version": 1, "id": "default"},
        selected=(),
        subagents=(),
        output_mode="text",
        output_schema={"type": "object", "properties": {"text": {"type": "string"}}},
        binding_digest="a" * 64,
    )


@pytest.mark.asyncio
async def test_default_platform_composition_keeps_file_read_deduplication() -> None:
    capabilities = await compose_platform_capabilities(
        agent_name="agent",
        conversation_id=None,
        step_run_id="run",
        segment_sequence=1,
        history_id=None,
        memory_scope=None,
        step_store=InMemoryStepStore(),
        memory_store=None,
        runtime_tool_names=(),
        plan_mode=False,
        trusted_tool_classes=(),
        trusted_mcp_selectors=(),
        context_target_tokens=None,
        parent_step_run_id=None,
        tool_operations=None,
        background_tasks=set(),
        plan_store_resolver=None,
    )
    assert any(isinstance(capability, DeduplicateFileReads) for capability in capabilities)


@pytest.mark.asyncio
async def test_agent_executor_cancellation_is_not_replaced_by_usage_sink_failure() -> None:
    executor = AgentExecutor(
        SkillSourceRegistry(),
        instruction_resolver=_EmptyRepositoryInstructionResolver(),
        middleware=MiddlewarePipeline(()),
    )

    async def cancelled(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise asyncio.CancelledError

    async def usage_sink(_usage: object) -> None:
        raise RuntimeError("usage sink failed")

    executor._execute = cancelled  # type: ignore[method-assign]
    scope = SimpleNamespace(
        binding=SimpleNamespace(
            definition=SimpleNamespace(spec=SimpleNamespace(usage_limits=None))
        ),
        usage_sink=usage_sink,
        step_run_id="run",
        background_tasks=set(),
    )

    with pytest.raises(asyncio.CancelledError):
        await executor.execute(scope)  # type: ignore[arg-type]

    pending = executor.pending_background_tasks
    assert len(pending) == 1
    await asyncio.gather(*pending, return_exceptions=True)


@pytest.mark.asyncio
async def test_confirmed_cancel_persists_canonical_terminal_error() -> None:
    now = datetime.now(timezone.utc)
    principal = trusted_workspace_principal("tenant")
    execution = ExecutionRecord(
        execution_id="execution",
        tenant_id="tenant",
        session_id=None,
        binding_digest="a" * 64,
        parent_execution_id=None,
        root_execution_id="execution",
        source_execution_id=None,
        base_execution_id=None,
        lineage_kind=ExecutionLineageKind.RUN,
        status=ExecutionStatus.STARTED,
        revision=1,
        event_sequence=1,
        agent_run_sequence=0,
        error_code=None,
        safe_error_details={},
        created_at=now,
        updated_at=now,
        mode="run",
        planning=False,
        thinking=False,
        binding=_binding_snapshot(),
    )
    cancelling = replace(
        execution,
        status=ExecutionStatus.CANCELLING,
        revision=2,
        event_sequence=2,
    )
    operation = SimpleNamespace(operation_id="operation", status=OperationStatus.PENDING)

    class Operations:
        def __init__(self) -> None:
            self.created = False

        async def get(self, operation_id: str, *, tenant_id: str) -> object | None:
            del operation_id, tenant_id
            return operation if self.created else None

        async def append(self, _value: object) -> object:
            self.created = True
            return operation

    class Executions:
        async def get(self, execution_id: str, *, tenant_id: str) -> ExecutionRecord:
            del execution_id, tenant_id
            return cancelling if operations.created else execution

    class Idempotency:
        async def list_by_resource(self, *args: object, **kwargs: object) -> tuple[object, ...]:
            del args, kwargs
            return ()

    class Backend:
        async def commit_cancel_checkpoint(self, *args: object, **kwargs: object) -> ExecutionRecord:
            del args, kwargs
            return cancelling

        async def cancel(self, _execution: ExecutionRecord) -> CancelEffectOutcome:
            return CancelEffectOutcome.CONFIRMED

        async def abort_start(self, _execution: ExecutionRecord) -> None:
            raise AssertionError("started execution must not use pending-start cleanup")

    class Committer:
        def __init__(self) -> None:
            self.commit = None

        async def commit_terminal_checkpoint(self, commit: object, *, session_id: str | None) -> object:
            del session_id
            self.commit = commit
            return SimpleNamespace()

    operations = Operations()
    committer = Committer()
    service = object.__new__(DefaultExecutionService)
    service._state = SimpleNamespace(
        operations=operations,
        executions=Executions(),
        idempotency=Idempotency(),
    )
    service._backend = Backend()
    service._subagent_cancellation = None
    service._terminal_committer = committer

    async def load_authorized(*args: object, **kwargs: object) -> ExecutionRecord:
        del args, kwargs
        return execution

    async def resolve_cancel_race(*args: object, **kwargs: object) -> None:
        del args, kwargs

    async def verify_terminal(*args: object, **kwargs: object) -> None:
        del args, kwargs

    service._load_authorized = load_authorized
    service._resolve_cancel_race = resolve_cancel_race
    service._terminal_verifier = verify_terminal

    result = await service._cancel(
        "execution",
        CancelExecutionRequest(
            principal=principal,
            idempotency_key="cancel-review-regression",
        ),
    )

    assert result.cancelled is True
    assert committer.commit is not None
    assert committer.commit.execution.status is ExecutionStatus.CANCELLED
    assert committer.commit.execution.error_code == ErrorCode.EXECUTION_CANCELLED.value
    assert committer.commit.execution.safe_error_details == {}
    assert committer.commit.terminal_event_payload == {
        "error_code": ErrorCode.EXECUTION_CANCELLED.value,
        "safe_error_details": {},
    }
