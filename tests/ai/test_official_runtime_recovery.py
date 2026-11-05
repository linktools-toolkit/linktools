#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Cold Runtime continuations use distinct native attempts without losing their source."""

import asyncio
from contextlib import AsyncExitStack, asynccontextmanager
from pathlib import Path

import pytest
from pydantic_ai.exceptions import ApprovalRequired
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart, ToolReturnPart, UserPromptPart
from pydantic_ai_harness.step_persistence import RunRecord, StepPersistence
from sqlalchemy.ext.asyncio import create_async_engine

from linktools.ai.capability import AgentContext, ExecutionCapability
from linktools.ai.core import ApprovalDecision, ExecutionStatus, Principal
from linktools.ai.execution import ApprovalDecisionRequest
from linktools.ai.migrate import provision_database
from linktools.ai.model.messages import decode_model_messages
from linktools.ai.persistence import Persistence, PersistenceStorage, SqlStepStore
from linktools.ai.runtime import Runtime, RuntimeFeatures

from ._runtime_test_helpers import _Models, _UsageFunctionModel, _wait_for_committed, model_features


@asynccontextmanager
async def _runtime(database: Path, model, tool):
    engine = create_async_engine(f"sqlite+aiosqlite:///{database}")
    await provision_database(engine)
    steps = SqlStepStore(engine, namespace="native-recovery", tenant_id="default")
    await steps.initialize()
    features = RuntimeFeatures("native-recovery")
    features.capability(ExecutionCapability(retain_history=True), scope="runtime")
    features.capability(StepPersistence(store=steps, capture_frontier=True), scope="runtime")
    features.tool(tool, effect_policy="replay_safe")
    features.agent("default", model="default", allow_tools=(tool.__name__,))
    try:
        async with Persistence.open("native-recovery", storage=PersistenceStorage.sql(engine)) as persistence:
            async with Runtime.attach(persistence=persistence,
                features=model_features(_Models(_UsageFunctionModel(model)), features), auto_recover=False) as runtime:
                yield runtime, persistence, steps
    finally:
        await engine.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize("conversation", (False, True))
@pytest.mark.parametrize(("unstarted_attempts", "registered_attempt"), ((0, False), (2, False), (0, True)))
async def test_cold_runtime_recovers_settled_tool_in_new_attempt(
    tmp_path: Path, unstarted_attempts: int, registered_attempt: bool,
    conversation: bool,
) -> None:
    effects = []
    blocked = asyncio.Event()

    async def work(_context: AgentContext[None]) -> str:
        effects.append("performed")
        return "saved result"

    async def interrupted(messages, _info):
        if any(isinstance(part, ToolReturnPart) for message in messages for part in message.parts):
            blocked.set()
            await asyncio.Event().wait()
        return ModelResponse(parts=[ToolCallPart("work", {}, tool_call_id="work-1")])

    database = tmp_path / "runtime.db"
    async with _runtime(database, interrupted, work) as (runtime, persistence, steps):
        agent = runtime.agents.get()
        runner = await agent.create_session("session") if conversation else agent
        execution = await runner.start("question", idempotency_key="start")
        await asyncio.wait_for(blocked.wait(), 5)
        await asyncio.wait_for(runtime.handoff(), 5)
        source = await persistence.execution.executions.get_recovery(execution.execution_id, tenant_id="default")
        assert source is not None and source.recovery is not None
        source_run_id = source.recovery.agent_run_id
        assert len(await steps.list_runs()) == 1
    calls = []

    async def recovered(messages, _info):
        calls.append(messages)
        assert [part.content for message in messages for part in message.parts if isinstance(part, ToolReturnPart)] == ["saved result"]
        return ModelResponse(parts=[TextPart("recovered")])

    # Each claim is a real producer admission whose process stops before Agent.run.
    for _ in range(unstarted_attempts):
        async with _runtime(database, recovered, work) as (runtime, persistence, steps):
            backend = runtime._execution_service.runtime_backend()
            current = await persistence.execution.executions.get(execution.execution_id, tenant_id="default")
            checkpoint = await backend._commit_recovery_resume(current)
            assert checkpoint.recovery is not None
            assert checkpoint.recovery.agent_run_id != source_run_id
            assert checkpoint.recovery.source_agent_run_id == source_run_id
            assert await steps.get_run(run_id=checkpoint.recovery.agent_run_id) is None
    if registered_attempt:
        async with _runtime(database, recovered, work) as (runtime, persistence, steps):
            backend = runtime._execution_service.runtime_backend()
            current = await persistence.execution.executions.get(execution.execution_id, tenant_id="default")
            checkpoint = await backend._commit_recovery_resume(current)
            previous = await steps.get_run(run_id=source_run_id)
            assert previous is not None and checkpoint.recovery is not None
            await steps.register_run(RunRecord(
                run_id=checkpoint.recovery.agent_run_id,
                conversation_id=previous.conversation_id,
                parent_run_id=previous.parent_run_id,
                agent_name=previous.agent_name,
                metadata=dict(previous.metadata),
            ))
            assert checkpoint.recovery is not None
            assert checkpoint.recovery.source_agent_run_id == source_run_id
            assert await steps.get_run(run_id=checkpoint.recovery.agent_run_id) is not None
            assert await steps.latest_snapshot(run_id=checkpoint.recovery.agent_run_id, include_interrupted=True) is None
    async with _runtime(database, recovered, work) as (runtime, persistence, steps):
        await runtime.executions.recover(execution.execution_id, principal=runtime.default_principal, idempotency_key="recover")
        outcome = await runtime.executions.wait(execution.execution_id, principal=runtime.default_principal, timeout_seconds=5)
        assert outcome.result.status is ExecutionStatus.SUCCEEDED, outcome.result
        assert outcome.result.output == {"text": "recovered"}
        current = await persistence.execution.executions.get(execution.execution_id, tenant_id="default")
        assert current.agent_run_seq == unstarted_attempts + 2 + registered_attempt
        runs = await steps.list_runs()
        assert len(runs) == 2 + registered_attempt
        assert len({run.run_id for run in runs}) == len(runs)
        assert source_run_id in {run.run_id for run in runs}
        assert effects == ["performed"] and len(calls) == 1
        if conversation:
            page = await runtime.history.messages("session", principal=runtime.default_principal, limit=100)
            parts = [part for message in page.items for part in message.parts]
            assert [part.content for part in parts if isinstance(part, UserPromptPart)] == ["question"]
            assert [part.content for part in parts if isinstance(part, ToolReturnPart)] == ["saved result"]
            assert [part.content for part in parts if isinstance(part, TextPart)] == ["recovered"]


@pytest.mark.asyncio
@pytest.mark.parametrize("interruption", (None, "claim", "effect"))
async def test_cold_runtime_approval_uses_source_frontier_and_fresh_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, interruption: str | None,
) -> None:
    entries = []
    effects = []
    blocked = asyncio.Event()
    finish = interruption != "effect"

    async def checked(_context: AgentContext[None]) -> str:
        entries.append("called")
        if len(entries) == 1:
            raise ApprovalRequired()
        effects.append("approved")
        return "approved result"

    async def model(messages, _info):
        if any(isinstance(part, ToolReturnPart) for message in messages for part in message.parts):
            if not finish:
                blocked.set()
                await asyncio.Event().wait()
            return ModelResponse(parts=[TextPart("approved completion")])
        return ModelResponse(parts=[ToolCallPart("checked", {}, tool_call_id="approval-1")])

    database = tmp_path / "approval.db"
    async with _runtime(database, model, checked) as (runtime, persistence, steps):
        execution = await runtime.agents.get().start("question", idempotency_key="start")
        await _wait_for_committed(
            lambda: persistence.execution.executions.get(execution.execution_id, tenant_id="default"),
            lambda value: value is not None and value.status is ExecutionStatus.WAITING_DEFERRED,
        )
        await asyncio.wait_for(runtime.handoff(), 5)
        assert effects == []
    async with AsyncExitStack() as stack:
        runtime, persistence, steps = await stack.enter_async_context(_runtime(database, model, checked))
        principal = Principal("approver", "default", "service")
        approvals = await runtime.approvals.list(execution.execution_id, principal=principal)
        assert len(approvals) == 1
        async def stopped_before_run(*args, **kwargs):
            pass

        with monkeypatch.context() as fault:
            if interruption == "claim":
                fault.setattr(runtime._execution_service.runtime_backend(), "launch", stopped_before_run)
            await runtime.approvals.decide(execution.execution_id, ApprovalDecisionRequest(
                principal, approvals[0].approval_id, "approve", ApprovalDecision.APPROVE,
            ))
        if interruption is not None:
            if interruption == "effect":
                await asyncio.wait_for(blocked.wait(), 5)
                await asyncio.wait_for(runtime.handoff(), 5)
            checkpoint = await persistence.execution.executions.get_recovery(execution.execution_id, tenant_id="default")
            assert checkpoint.recovery.pending_tools is not None
            assert checkpoint.recovery.source_agent_run_id == checkpoint.recovery.pending_tools.source_agent_run_id
            await stack.aclose()
            finish = True
            runtime, persistence, steps = await stack.enter_async_context(_runtime(database, model, checked))
            await runtime.executions.recover(
                execution.execution_id, principal=runtime.default_principal, idempotency_key="resume-approved",
            )
        outcome = await runtime.executions.wait(execution.execution_id, principal=runtime.default_principal, timeout_seconds=5)
        assert outcome.result.status is ExecutionStatus.SUCCEEDED, outcome.result
        assert outcome.result.output == {"text": "approved completion"}
        current = await persistence.execution.executions.get(execution.execution_id, tenant_id="default")
        assert current.agent_run_seq == (2 if interruption is None else 3)
        assert len(await steps.list_runs()) == (3 if interruption == "effect" else 2)
        assert effects == ["approved"]


@pytest.mark.asyncio
@pytest.mark.parametrize("force", (False, True))
async def test_native_cold_cancel_keeps_late_tool_result_after_owner_drains(
    tmp_path: Path, force: bool,
) -> None:
    from linktools.ai.core import ToolOperationStatus
    from linktools.ai.execution import CancelExecutionRequest

    entered = asyncio.Event()
    release = asyncio.Event()
    effects = []
    requests = []

    async def work(_context: AgentContext[None]) -> str:
        entered.set()
        await release.wait()
        effects.append("performed")
        return "known result after remote cancellation"

    async def model(messages, _info):
        requests.append(messages)
        if len(requests) == 1:
            return ModelResponse(parts=[ToolCallPart("work", {}, tool_call_id="work-1")])
        return ModelResponse(parts=[TextPart("must not dispatch after cancellation")])

    database = tmp_path / "cancel.db"
    async with AsyncExitStack() as stack:
        owner, persistence, _ = await stack.enter_async_context(_runtime(database, model, work))
        remote, _, _ = await stack.enter_async_context(_runtime(database, model, work))
        execution = await owner.agents.get().start("question", idempotency_key="start")
        try:
            await asyncio.wait_for(entered.wait(), 5)
            active = await persistence.execution.executions.get_recovery(execution.execution_id, tenant_id="default")
            assert active is not None and active.recovery is not None
            run_id = active.recovery.agent_run_id
            cancellation = await remote.executions.cancel(
                execution.execution_id, CancelExecutionRequest(owner.default_principal, "stop", force=force),
            )
            assert not cancellation.cancelled
            current = await persistence.execution.executions.get(execution.execution_id, tenant_id="default")
            assert current.status is ExecutionStatus.CANCELLING
            uncertain = await persistence.checkpoint.tools.get_by_call(run_id, "work-1", tenant_id="default")
            assert uncertain.status is ToolOperationStatus.CLAIMED
            release.set()
            operation = await _wait_for_committed(
                lambda: persistence.checkpoint.tools.get_by_call(run_id, "work-1", tenant_id="default"),
                lambda value: value is not None and value.status is ToolOperationStatus.COMPLETED,
            )
            assert operation.result_payload is not None
            saved_payload = operation.result_payload
            saved_messages = decode_model_messages(saved_payload.decode())
            assert [part.content for message in saved_messages for part in message.parts
                    if isinstance(part, ToolReturnPart)] == ["known result after remote cancellation"]
            assert len(requests) == 1 and effects == ["performed"]
            outcome = await owner.executions.wait(
                execution.execution_id, principal=owner.default_principal, timeout_seconds=10,
            )
            assert outcome.result.status is ExecutionStatus.CANCELLED
            current = await persistence.execution.executions.get(execution.execution_id, tenant_id="default")
            await asyncio.wait_for(owner.close(), 5)
            assert await persistence.execution.executions.get(execution.execution_id, tenant_id="default") == current
        finally:
            release.set()
    async with _runtime(database, model, work) as (runtime, persistence, _):
        operation = await persistence.checkpoint.tools.get_by_call(run_id, "work-1", tenant_id="default")
        assert operation is not None and operation.status is ToolOperationStatus.COMPLETED
        assert operation.result_payload == saved_payload
        outcome = await runtime.executions.wait(
            execution.execution_id, principal=runtime.default_principal, timeout_seconds=10,
        )
        assert outcome.result.status is ExecutionStatus.CANCELLED
        operation = await persistence.checkpoint.tools.get_by_call(run_id, "work-1", tenant_id="default")
        assert operation is not None and operation.status is ToolOperationStatus.COMPLETED
        assert operation.result_payload == saved_payload
        assert len(requests) == 1 and effects == ["performed"]
