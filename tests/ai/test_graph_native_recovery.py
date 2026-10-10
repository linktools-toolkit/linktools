#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Native graph recovery preserves captured input and the admitted principal."""

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models.function import AgentInfo, FunctionModel

from linktools.ai.capability import CapabilityGroup
from linktools.ai.core import (
    AuthorizationAction,
    ExecutionStatus,
    OperationKind,
    OperationStatus,
    Principal,
    ResourceKind,
    ResourceRef,
    TaskStatus,
    TenantAuthorizationPolicy,
)
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime import (
    AgentTaskInput,
    CaptureInputRequest,
    Runtime,
    RuntimeStorage,
)
from linktools.ai.task import TaskGraph, TaskNode
from linktools.ai.runtime.state._contracts import ExecutionRecord

from ._runtime_test_helpers import _wait_for_committed
from .test_live_history_readback_integration import _Models


@pytest.mark.asyncio
async def test_run_only_actor_recovers_native_captured_agent_graph(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    origin = Principal("executor", "default")
    actor = Principal("operator", "default")
    entered = asyncio.Event()
    stop_owner = asyncio.Event()
    model_inputs: list[str] = []

    async def model(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        del info
        model_inputs.append(str(messages))
        if len(model_inputs) == 2:
            entered.set()
            await stop_owner.wait()
            raise RuntimeError("injected stopped execution owner")
        yield "recovered answer" if len(model_inputs) == 3 else "source answer"

    class Authorization:
        def __init__(self) -> None:
            self.allow_actor_read = True
            self.calls: list[tuple[Principal, AuthorizationAction]] = []
            self.default = TenantAuthorizationPolicy("default")

        async def authorize(
            self, principal: Principal, action: AuthorizationAction, resource: ResourceRef,
        ) -> None:
            self.calls.append((principal, action))
            if action is AuthorizationAction.EXECUTION_RECOVER:
                raise AIError(ErrorCode.AUTHORIZATION_DENIED)
            if principal == actor:
                if resource.id == "native-recovery" and (
                    action is AuthorizationAction.TASK_RUN
                    or self.allow_actor_read and action is AuthorizationAction.TASK_READ
                ):
                    return
                raise AIError(ErrorCode.AUTHORIZATION_DENIED)
            await self.default.authorize(principal, action, resource)

    authorization = Authorization()
    group = CapabilityGroup("native-graph-recovery")
    group.agent("default", model="default", allow_tools=(), allow_skills=(), allow_subagents=())
    storage = RuntimeStorage.filesystem(tmp_path / "state")
    try:
        async with Runtime.open(
            "native-graph-recovery", models=_Models(FunctionModel(stream_function=model)),
            storage=storage, capabilities=(group,), authorization=authorization,
        ) as runtime:
            task = runtime.tasks.from_agent("test.native-captured-agent", runtime.agents.get("default"))
            engine = runtime.tasks.bind(task)
            source = await engine.start(
                TaskGraph("capture-source", (
                    TaskNode("agent", task=task, input=AgentTaskInput("immutable recovery prompt")),
                )),
                principal=origin, idempotency_key="source-graph",
            )
            assert (await source.wait(timeout_seconds=10)).result.status is TaskStatus.SUCCEEDED
            source_execution = await source.execution("agent")
            capture = await runtime.executions.capture_input(
                source_execution.execution_id, CaptureInputRequest(origin, "source-input"),
            )
            task_capture = await runtime._input_captures.task_input(capture, principal=origin)
            run = await engine.start(
                TaskGraph("native-recovery", (
                    TaskNode("agent", task=task, input_capture=task_capture),
                )),
                principal=origin, idempotency_key="captured-graph",
            )
            await asyncio.wait_for(entered.wait(), 10)
            bound = await _wait_for_committed(
                lambda: storage.task.tasks.graph_state(run.graph_id, tenant_id=origin.tenant_id),
                lambda state: state is not None and state.node_states[0].execution_id is not None,
            )
            execution_id = bound.node_states[0].execution_id
            assert execution_id is not None
            backend = runtime._execution_service.runtime_backend()
            worker = backend._tasks[execution_id]
            before = await storage.execution.executions.get(execution_id, tenant_id=origin.tenant_id)
            head = await storage.execution.executions.get_history_head(execution_id, tenant_id=origin.tenant_id)
            assert before is not None and head is not None
            assert before.status is ExecutionStatus.STARTED
            assert bound.nodes[0].input_capture == task_capture

            commit_failure = backend._commit_failure

            async def commit_stopped_owner(
                execution: ExecutionRecord, error: Exception, *,
                agent_run_id: str | None = None, producer_generation: int | None = None,
            ) -> ExecutionRecord:
                if execution.execution_id == execution_id:
                    return await backend._commit_recovery_required(
                        execution, AIError(ErrorCode.TOOL_EFFECT_UNKNOWN), (),
                        producer_generation=producer_generation,
                    )
                return await commit_failure(
                    execution, error, agent_run_id=agent_run_id,
                    producer_generation=producer_generation,
                )

            with monkeypatch.context() as fault:
                fault.setattr(backend, "_commit_failure", commit_stopped_owner)
                stop_owner.set()
                await asyncio.wait_for(asyncio.shield(worker), 10)
            interrupted = (await run.wait(timeout_seconds=10)).result
            assert interrupted.status is TaskStatus.RECOVERY_REQUIRED
            assert not backend.worker_installed(execution_id)

        storage = RuntimeStorage.filesystem(tmp_path / "state")
        async with Runtime.open(
            "native-graph-recovery", models=_Models(FunctionModel(stream_function=model)),
            storage=storage, capabilities=(group,), authorization=authorization,
        ) as runtime:
            task = runtime.tasks.from_agent("test.native-captured-agent", runtime.agents.get("default"))
            engine = runtime.tasks.bind(task)
            run = await engine.get("native-recovery", principal=origin)
            recovery_handle = await engine.get(run.graph_id, principal=actor)
            authorization.allow_actor_read = False
            authorization.calls.clear()
            backend = runtime._execution_service.runtime_backend()
            assert not backend.worker_installed(execution_id)

            await recovery_handle.recover(idempotency_key="recover-captured-graph")
            completed = (await run.wait(timeout_seconds=10)).result
            assert completed.status is TaskStatus.SUCCEEDED
            result = await runtime.executions.result(execution_id, principal=origin)
            assert result.output == {"text": "recovered answer"}
            current = await storage.execution.executions.get(execution_id, tenant_id=origin.tenant_id)
            recovered_head = await storage.execution.executions.get_history_head(
                execution_id, tenant_id=origin.tenant_id,
            )
            assert current is not None and recovered_head is not None
            assert current.principal_id == origin.principal_id
            assert current.principal_kind == origin.kind
            assert current.stored_user_input == before.stored_user_input
            assert recovered_head.producer_generation > head.producer_generation
            actor_actions = [action for principal, action in authorization.calls if principal == actor]
            assert actor_actions and set(actor_actions) == {AuthorizationAction.TASK_RUN}
            assert not any(action is AuthorizationAction.EXECUTION_RECOVER
                           for _principal, action in authorization.calls)
            final_graph = await run.state(include_content=True)
            assert final_graph.nodes[0].input_capture == task_capture
            assert final_graph.node_states[0].execution_id == execution_id
            receipts = await storage.execution.operations.list_pending(
                ResourceKind.EXECUTION, execution_id, tenant_id=origin.tenant_id,
                limit=10, states=frozenset({OperationStatus.SUCCEEDED}),
            )
            assert len(receipts) == 1
            assert receipts[0].operation_kind is OperationKind.EXECUTION_RECOVER
            assert receipts[0].result_ref == recovered_head.producer_claim_id
            await recovery_handle.recover(idempotency_key="recover-captured-graph")
            assert len(model_inputs) == 3
            assert all("immutable recovery prompt" in messages for messages in model_inputs)
    finally:
        stop_owner.set()


@pytest.mark.asyncio
async def test_graph_takeover_rejects_active_local_worker_without_child_claim(tmp_path: Path) -> None:
    entered = asyncio.Event()
    release = asyncio.Event()
    model_calls = 0

    async def model(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        nonlocal model_calls
        del messages, info
        model_calls += 1
        entered.set()
        await release.wait()
        yield "original worker answer"

    group = CapabilityGroup("local-graph-recovery")
    group.agent("default", model="default", allow_tools=(), allow_skills=(), allow_subagents=())
    storage = RuntimeStorage.filesystem(tmp_path / "state")
    async with Runtime.open(
        "local-graph-recovery", models=_Models(FunctionModel(stream_function=model)),
        storage=storage, capabilities=(group,),
    ) as runtime:
        try:
            task = runtime.tasks.from_agent("test.local-agent", runtime.agents.get("default"))
            run = await runtime.tasks.bind(task).start(
                TaskGraph("local-active-graph", (
                    TaskNode("agent", task=task, input=AgentTaskInput("keep original worker")),
                )),
                idempotency_key="local-active-start",
            )
            await asyncio.wait_for(entered.wait(), 10)
            bound = await _wait_for_committed(
                lambda: storage.task.tasks.graph_state(run.graph_id, tenant_id=runtime.tenant_id),
                lambda state: state is not None and state.node_states[0].execution_id is not None,
            )
            execution_id = bound.node_states[0].execution_id
            assert execution_id is not None
            head = await storage.execution.executions.get_history_head(
                execution_id, tenant_id=runtime.tenant_id,
            )
            assert head is not None
            receipts = await storage.execution.operations.list_pending(
                ResourceKind.EXECUTION, execution_id, tenant_id=runtime.tenant_id,
                limit=10, states=frozenset(OperationStatus),
            )
            assert receipts == ()
            with pytest.raises(AIError) as raised:
                await run.recover(idempotency_key="reject-local-takeover")
            assert raised.value.code is ErrorCode.STORAGE_CONFLICT
            unchanged_head = await storage.execution.executions.get_history_head(
                execution_id, tenant_id=runtime.tenant_id,
            )
            assert unchanged_head.producer_generation == head.producer_generation
            assert unchanged_head.producer_claim_id == head.producer_claim_id
            assert await storage.execution.operations.list_pending(
                ResourceKind.EXECUTION, execution_id, tenant_id=runtime.tenant_id,
                limit=10, states=frozenset(OperationStatus),
            ) == ()
            assert model_calls == 1
            release.set()
            assert (await run.wait(timeout_seconds=10)).result.status is TaskStatus.SUCCEEDED
        finally:
            release.set()
