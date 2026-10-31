#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from linktools.ai.persistence import Persistence
"""Focused regression coverage for reliable mixed TaskGraph nodes."""
from ._runtime_test_helpers import model_features
from linktools.ai.model import create_model

from linktools.ai.persistence import Persistence
from ._runtime_test_helpers import workspace_features

import asyncio
import faulthandler
import hashlib
import sys
from collections.abc import Mapping
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from time import monotonic
from types import CoroutineType, GeneratorType

import pytest
from ._task_test_helpers import (
    RuntimeFeatures,
    TaskFunction,
    admit_graph,
    agent_task_definition,
    agent_task_node,
    clear_task_test_state,
    run_task_graph,
    start_task_graph,
    start_task_request,
    task_engine,
    task_engine_scope,
    task_graph_resume,
    task_graph_state,
    task_graph_wait,
    task_result,
)
from linktools.ai.agent import restore_output
from linktools.ai.capability import ExecutionCapability, MemoryCapability
from linktools.ai.core import (
    JsonValue,
    OperationKind,
    Principal,
    PrincipalKind,
    TaskStatus,
    WorkspaceFileInput,
    canonical_sha256,
    idempotency_key_digest,
    principal_identity_payload,
)
from linktools.ai.errors import AIError, ErrorCode, ObservationError

from linktools.ai.inputs import AgentTaskInput, AgentTaskInputContext
from linktools.ai.execution import ExecutionInputContext, ExecutionService
from linktools.ai.runtime import Runtime
from linktools.ai.persistence import PersistenceStorage
from linktools.ai.task.engine import TaskEngine, TaskGraphRun
from linktools.ai.task.engine._agent import AgentTaskRunner, _agent_task_input_identity, _agent_task_request_identity
from linktools.ai.persistence import PersistenceDomain
from linktools.ai.persistence.snapshots import SnapshotLimits
from linktools.ai.persistence.codec import (
    _encode_persisted_domain,
    decode_domain,
    iter_runtime_object_refs,
)
from linktools.ai.persistence.contracts import StoredUserInput
from linktools.ai.persistence._task_state import (
    _effective_graph_status,
    _isolated_graph_status,
)
from linktools.ai.storage import InMemoryObjectStore, StoredPayload, read_object
from linktools.ai.task import (
    LocalTaskGraphLauncher,
    TaskBindingContract,
    TaskDependencyState,
    TaskGraph,
    TaskGraphAdmission,
    TaskGraphLaunch,
    TaskGraphLimits,
    TaskGraphInfo,
    TaskGraphRequest,
    TaskGraphState,
    TaskGraphView,
    TaskNode,
    TaskNodeContext,
    TaskNodeInvocation,
    TaskNodeView,
    Task,
    TaskRef,
    TaskEffectResolution,
    TaskInputSupplyRequest,
    TaskExpansionContext,
    TaskExpander,
    TaskExpanderRef,
    TaskNodeRunControl,
    TaskNodeRunError,
    TaskNodeRunResult,
)
from linktools.ai.task import _local as task_local
from linktools.ai.workspace import Workspace
from pydantic import BaseModel
from pydantic_ai.messages import BinaryContent
from pydantic_ai.models.test import TestModel


@pytest.fixture(autouse=True)
def _reset_task_registry() -> None:
    clear_task_test_state()


async def _report_graph_wait_timeout(run: TaskGraphRun[object]) -> None:
    started = monotonic()
    print(f"Graph wait timed out: {run.graph_id}", file=sys.stderr)
    for task in sorted(asyncio.all_tasks(), key=lambda value: value.get_name()):
        print(f"Async task: {task.get_name()}", file=sys.stderr)
        pending = task.get_coro()
        while isinstance(pending, (CoroutineType, GeneratorType)):
            frame = pending.cr_frame if isinstance(pending, CoroutineType) else pending.gi_frame
            if frame is not None:
                print(f"  {frame.f_code.co_filename}:{frame.f_lineno} in {frame.f_code.co_name}", file=sys.stderr)
            pending = pending.cr_await if isinstance(pending, CoroutineType) else pending.gi_yieldfrom
    faulthandler.dump_traceback(file=sys.stderr, all_threads=True)

    def consume_result(task: asyncio.Task[TaskGraphInfo]) -> None:
        if not task.cancelled():
            task.exception()

    probe = asyncio.create_task(run.state(), name="timeout-graph-state")
    try:
        done, _ = await asyncio.wait({probe}, timeout=0.5)
        if not done:
            print("Post-timeout graph state unavailable: read exceeded 0.5s", file=sys.stderr)
            return
        if probe.cancelled():
            print("Post-timeout graph state unavailable: read cancelled", file=sys.stderr)
            return
        snapshot = probe.result()
        print(
            f"Post-timeout graph state (+{monotonic() - started:.3f}s): "
            f"id={snapshot.graph_id} status={snapshot.status.value} event_seq={snapshot.event_seq}",
            file=sys.stderr,
        )
        for node in snapshot.node_states:
            print(
                f"Node: id={node.node_id} dependencies={node.dependencies} status={node.status.value} "
                f"execution_id={node.execution_id} fence={node.fence} "
                f"lease_expires_at={node.lease_expires_at} error_code={node.error_code}",
                file=sys.stderr,
            )
    finally:
        if not probe.done():
            probe.cancel()
            await asyncio.wait({probe}, timeout=0.1)
        if probe.done():
            consume_result(probe)
        else:
            print("Post-timeout state read cancellation pending", file=sys.stderr)
            probe.add_done_callback(consume_result)


@pytest.mark.asyncio
@pytest.mark.parametrize("workspace_input", (False, True))
@pytest.mark.parametrize("dynamic", (False, True))
async def test_graph_freezes_attachments_before_dependencies_finish(
    tmp_path: Path,
    workspace_input: bool,
    dynamic: bool,
) -> None:
    gate = asyncio.Event()
    started = asyncio.Event()

    async def hold(context: TaskNodeContext[None]) -> JsonValue:
        if context.node_id == "plan":
            return "ready"
        started.set()
        await gate.wait()
        return "ready"

    source = tmp_path / "input.txt"
    source.write_text("accepted content", encoding="utf-8")
    workspace = Workspace.load(tmp_path)
    application = workspace_features("application", workspace=workspace, features_type=RuntimeFeatures)
    handler = TaskFunction[None]("example.hold", 1, hold)
    application.task(handler, effect_policy="none")
    attachment = (
        WorkspaceFileInput("input.txt", identifier="source")
        if workspace_input
        else BinaryContent(
            data=b"accepted content",
            media_type="text/plain",
            identifier="source",
        )
    )

    class Expand:
        id = "example.attachments"
        revision = 1

        def __init__(self, consumer: TaskNode) -> None:
            self.consumer = consumer

        def expand(self, context: TaskExpansionContext) -> tuple[TaskNode, ...]:
            del context
            return (
                handler.node("hold"),
                self.consumer,
            )

    application.agent(
        "default",
        model="default",
        allow_tools=(),
        allow_skills=(),
        allow_subagents=(),
    )
    storage_root = tmp_path / "state"
    state = PersistenceStorage.filesystem(storage_root)
    async with Persistence.open('attachment-test', storage=state) as persistence_220, Runtime.attach(persistence=persistence_220, features=model_features(_TaskTestModels(), application)) as runtime, task_engine_scope(runtime, persistence_220) as runtime_engine:
        node = agent_task_node(runtime,
            "consumer",
            ("inspect", attachment),
            dependencies=("hold",),
        )
        if dynamic:
            application.task_expander(Expand(node))
        nodes = (
            (handler.node("plan", expander=TaskExpanderRef("example.attachments", 1)),)
            if dynamic
            else (handler.node("hold"), node)
        )
        graph = TaskGraph("attachment-graph", nodes)
        run = await start_task_graph(runtime, graph, idempotency_key="attachment-1")
        await asyncio.wait_for(started.wait(), 10)
        source.unlink()
        repeated = await start_task_graph(runtime, graph, idempotency_key="attachment-1")
        assert repeated.graph_id == run.graph_id
        snapshot = await runtime._state.task.tasks.scheduler_state(
            run.graph_id,
            tenant_id=runtime.tenant_id,
        )
        frozen_node = next(n for n in snapshot.nodes if n.node_id == "consumer")
        stored_prompt = frozen_node.input.get("prompt")
        assert isinstance(stored_prompt, Mapping)
        assert stored_prompt.get("kind") == "stored-user-content-v1"
        stored = decode_domain(stored_prompt.get("value"), StoredUserInput)
        assert stored.payload.ref is not None
        frozen_bytes = await read_object(
            runtime._state.object_store(PersistenceDomain.TASK),
            stored.payload.ref.key,
            expected_digest=stored.payload.ref.digest,
            expected_size=stored.payload.ref.size,
        )
        refs = tuple(
            iter_runtime_object_refs(
                _encode_persisted_domain(frozen_node),
                default_domain=PersistenceDomain.TASK,
            )
        )
        assert (PersistenceDomain.TASK, stored.payload.ref) in refs
        with pytest.raises(AIError) as rejected:
            await start_task_graph(runtime,
                TaskGraph("forged-input", (handler.node("hold"), frozen_node)),
                idempotency_key="forged-input-1",
            )
        assert rejected.value.code is ErrorCode.REQUEST_FIELD_INVALID
        gate.set()
        try:
            completed = (await run.wait(timeout_seconds=10)).result
        except AIError as error:
            if error.code is ErrorCode.WAIT_TIMEOUT:
                try:
                    await _report_graph_wait_timeout(run)
                except Exception as diagnostic_error:
                    print(f"Graph timeout diagnostic unavailable: {type(diagnostic_error).__name__}", file=sys.stderr)
            raise
        assert completed.wait_status is TaskStatus.SUCCEEDED
        consumer = next(n for n in completed.node_states if n.node_id == "consumer")
        assert consumer.status is TaskStatus.SUCCEEDED
        record = await runtime._state.execution.executions.get(
            consumer.execution_id, tenant_id=runtime.tenant_id,
        )
        assert record.stored_user_input.view["attachments"] == stored.view["attachments"]
        assert record.stored_user_input.view["files"] == stored.view["files"]

    archive = InMemoryObjectStore("archive")
    read_state = PersistenceStorage.filesystem(storage_root)
    read_state = Persistence(read_state)
    await read_state.initialize(
        namespace="attachment-test", tenant_id="default", read_only=True
    )
    limits = SnapshotLimits(max_entries=4096, max_bytes=16 * 1024 * 1024)
    try:
        reference = await read_state.export_snapshot(
            object_store=archive, limits=limits
        )
    finally:
        await read_state.close()
    restored_root = tmp_path / "restored"
    await Persistence.restore_snapshot(
        reference, object_store=archive, root=restored_root, limits=limits
    )
    restored = PersistenceStorage.from_root(restored_root)
    restored = Persistence(restored)
    await restored.initialize(
        namespace="attachment-test", tenant_id="default", read_only=True
    )
    try:
        assert frozen_bytes == await read_object(
            restored.object_store(PersistenceDomain.TASK),
            stored.payload.ref.key,
            expected_digest=stored.payload.ref.digest,
            expected_size=stored.payload.ref.size,
        )
    finally:
        await restored.close()


class _TaskTestModelBinding:
    _input_binding = create_model(api='openai',
        model="gpt-task-test", input_types=("mime:text/plain",),
    )
    route_id = "default"
    provider = _input_binding.provider
    model_identity = _input_binding.model_identity
    model_digest = "a" * 64
    contract: dict[str, JsonValue] = dict(_input_binding.contract)

    def materialize(self) -> TestModel:
        return TestModel()


class _TaskTestModels(_TaskTestModelBinding):
    pass


class _ThinkingTaskTestModelBinding(_TaskTestModelBinding):
    def materialize(self) -> TestModel:
        return TestModel(
            profile={
                "supports_thinking": True,
                "thinking_always_enabled": False,
            }
        )


class _ThinkingTaskTestModels(_ThinkingTaskTestModelBinding):
    pass


class _StructuredTaskTestModelBinding(_TaskTestModelBinding):
    def materialize(self) -> TestModel:
        return TestModel(
            custom_output_args={"value": "accepted"},
            profile={"supports_json_schema_output": False},
        )


class _StructuredTaskTestModels(_StructuredTaskTestModelBinding):
    pass


class _ContractTaskRunner:
    def __init__(self) -> None:
        self.calls = 0

    async def run(
        self,
        invocation: TaskNodeInvocation,
        *,
        control: TaskNodeRunControl,
    ) -> TaskNodeRunResult:
        del invocation, control
        self.calls += 1
        return TaskNodeRunResult(
            canonical_sha256({"value": "accepted"}),
            "contract-runner-execution",
        )

    async def wait_bound(
        self,
        invocation: TaskNodeInvocation,
        execution_id: str,
    ) -> TaskNodeRunResult:
        del invocation
        return TaskNodeRunResult(
            canonical_sha256({"value": "accepted"}),
            execution_id,
        )

    async def supply_input(
        self,
        invocation: TaskNodeInvocation,
        execution_id: str,
        value: JsonValue,
    ) -> TaskNodeRunResult:
        del invocation, execution_id, value
        raise AIError(ErrorCode.TASK_NOT_READY)

    async def resolve_effect(
        self,
        invocation: TaskNodeInvocation,
        execution_id: str,
        resolution: TaskEffectResolution,
    ) -> TaskNodeRunResult | None:
        del invocation, execution_id, resolution
        raise AIError(ErrorCode.TASK_NOT_READY)

    async def cancel(self, invocation: TaskNodeInvocation) -> None:
        del invocation


class _ExecutionBackedContractRunner:
    def __init__(self, execution: ExecutionService) -> None:
        self._execution = execution
        self.calls = 0

    async def run(
        self,
        invocation: TaskNodeInvocation,
        *,
        control: TaskNodeRunControl,
    ) -> TaskNodeRunResult:
        self.calls += 1
        binding = TaskBindingContract(
            id="example.contract-runner.execution",
            revision=1,
            effect_policy="none",
            output_contract={"kind": "json"},
            timeout_seconds=None,
            max_attempts=1,
            retry_delay_seconds=0,
        )
        handle = await self._execution.start_task(
            binding,
            principal=invocation.principal,
            input=invocation.node.input,
            idempotency_key=(
                f"runner:{invocation.graph_id}:{invocation.node.node_id}"
            ),
            correlation=invocation.correlation,
        )
        await control.bind_execution(handle.execution_id)
        result = await self._execution.complete_task(
            handle.execution_id,
            principal=invocation.principal,
            output={"value": "accepted"},
        )
        return TaskNodeRunResult(
            canonical_sha256(result.output),
            handle.execution_id,
        )

    async def wait_bound(
        self,
        invocation: TaskNodeInvocation,
        execution_id: str,
    ) -> TaskNodeRunResult:
        result = await self._execution.result(
            execution_id,
            principal=invocation.principal,
        )
        return TaskNodeRunResult(
            canonical_sha256(result.output),
            execution_id,
        )

    async def supply_input(
        self,
        invocation: TaskNodeInvocation,
        execution_id: str,
        value: JsonValue,
    ) -> TaskNodeRunResult:
        del invocation, execution_id, value
        raise AIError(ErrorCode.TASK_NOT_READY)

    async def resolve_effect(
        self,
        invocation: TaskNodeInvocation,
        execution_id: str,
        resolution: TaskEffectResolution,
    ) -> TaskNodeRunResult | None:
        del invocation, execution_id, resolution
        raise AIError(ErrorCode.TASK_NOT_READY)

    async def cancel(self, invocation: TaskNodeInvocation) -> None:
        if invocation.execution_id is None:
            return
        await self._execution.cancel_task(
            invocation.execution_id,
            principal=invocation.principal,
        )


def test_task_definitions_keep_explicit_identity_and_contract() -> None:
    definition = Task("example.direct", _echo_task, effect_policy="none")

    assert (definition.ref.id, definition.ref.revision) == ("example.direct", 1)
    assert definition.contract["type"] == "function"
    assert definition.contract["effect_policy"] == "none"
    assert definition.contract["cancel"] is False

    async def cancel(_context: TaskNodeContext[None]) -> None:
        return None

    cancellable = Task(
        "example.cancellable",
        _echo_task,
        effect_policy="none",
        cancel=cancel,
    )
    assert cancellable.contract["cancel"] is True
    assert TaskNode("node", task=definition).task == definition.ref


@pytest.mark.asyncio
async def test_execution_cancel_winning_during_output_store_projects_cancelled_node(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = PersistenceStorage.in_memory()
    store_entered = asyncio.Event()
    release_store = asyncio.Event()
    async with Persistence.open('execution-task-output-race', storage=state) as persistence_528, Runtime.attach(persistence=persistence_528, features=model_features(_TaskTestModels())) as runtime, task_engine_scope(runtime, persistence_528) as runtime_engine:
        class _OutputRaceRunner:
            execution_id: str | None = None

            async def run(
                self,
                invocation: TaskNodeInvocation,
                *,
                control: TaskNodeRunControl,
            ) -> TaskNodeRunResult:
                binding = TaskBindingContract(
                    id="example.execution-output-race",
                    revision=1,
                    effect_policy="none",
                    output_contract={"kind": "json"},
                    timeout_seconds=None,
                    max_attempts=1,
                    retry_delay_seconds=0,
                )
                handle = await runtime_engine._execution_service.start_task(
                    binding,
                    principal=invocation.principal,
                    input=invocation.node.input,
                    idempotency_key="execution-output-race-start-0001",
                    correlation=invocation.correlation,
                )
                self.execution_id = handle.execution_id
                await control.bind_execution(handle.execution_id)
                await control.handoff_execution(handle.execution_id)
                result = await runtime_engine._execution_service.complete_task(
                    handle.execution_id,
                    principal=invocation.principal,
                    output={"value": "old-output"},
                )
                return TaskNodeRunResult(
                    canonical_sha256(result.output),
                    handle.execution_id,
                )

            async def wait_bound(
                self,
                invocation: TaskNodeInvocation,
                execution_id: str,
            ) -> TaskNodeRunResult:
                result = await runtime_engine._execution_service.result(
                    execution_id,
                    principal=invocation.principal,
                )
                if result.status.value != "SUCCEEDED":
                    raise TaskNodeRunError(
                        ErrorCode.EXECUTION_CANCELLED,
                        execution_id,
                    )
                return TaskNodeRunResult(
                    canonical_sha256(result.output),
                    execution_id,
                )

            async def supply_input(
                self,
                invocation: TaskNodeInvocation,
                execution_id: str,
                value: JsonValue,
            ) -> TaskNodeRunResult:
                del invocation, execution_id, value
                raise AIError(ErrorCode.TASK_NOT_READY)

            async def resolve_effect(
                self,
                invocation: TaskNodeInvocation,
                execution_id: str,
                resolution: TaskEffectResolution,
            ) -> TaskNodeRunResult | None:
                del invocation, execution_id, resolution
                raise AIError(ErrorCode.TASK_NOT_READY)

            async def cancel(self, invocation: TaskNodeInvocation) -> None:
                del invocation

        runner = _OutputRaceRunner()
        original_store = runtime_engine._execution_service._store_task_output

        async def delayed_store(
            output: JsonValue,
            *,
            tenant_id: str,
        ) -> StoredPayload:
            store_entered.set()
            await release_store.wait()
            return await original_store(output, tenant_id=tenant_id)

        monkeypatch.setattr(
            runtime_engine._execution_service,
            "_store_task_output",
            delayed_store,
        )
        task = Task.from_runner(
            "example.execution-output-race",
            runner,
            contract={
                "version": 1,
                "type": "example.execution-output-race",
                "effect_policy": "none",
                "output_contract": {"kind": "json"},
                "reconcile": False,
            },
        )
        graph = TaskGraph(
            "execution-output-race",
            (TaskNode("node", task=task),),
        )
        graph_run = await runtime_engine.with_definitions(task).start(
            graph,
            idempotency_key="execution-output-race-graph-0001",
         principal=runtime.default_principal)
        try:
            await asyncio.wait_for(store_entered.wait(), 3)
            assert runner.execution_id is not None
            cancelled = await runtime_engine._execution_service.cancel_task(
                runner.execution_id,
                principal=runtime.default_principal,
            )
            assert cancelled.cancelled
        finally:
            release_store.set()

        result = (await graph_run.wait(timeout_seconds=5)).result
        execution = await runtime_engine._execution_service.inspect(
            runner.execution_id,
            principal=runtime.default_principal,
        )
        graph_state = await runtime._state.task.tasks.graph_state(
            graph.graph_id,
            tenant_id=runtime.tenant_id,
        )

        assert result.wait_status is TaskStatus.CANCELLED
        assert execution.status.value == "CANCELLED"
        assert graph_state is not None
        assert graph_state.node_states[0].status is TaskStatus.CANCELLED
        assert graph_state.node_states[0].result_digest is None


@pytest.mark.asyncio
async def test_function_task_cancel_winning_during_output_store_projects_cancelled_node(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = PersistenceStorage.in_memory()
    store_entered = asyncio.Event()
    release_store = asyncio.Event()

    async def run_task(context: TaskNodeContext[None]) -> JsonValue:
        del context
        return {"value": "old-output"}

    task = Task("example.function-output-race", run_task, effect_policy="none")
    graph = TaskGraph(
        "function-output-race",
        (TaskNode("node", task=task),),
    )
    async with Persistence.open('function-task-output-race', storage=state) as persistence_692, Runtime.attach(persistence=persistence_692, features=model_features(_TaskTestModels())) as runtime, task_engine_scope(runtime, persistence_692) as runtime_engine:
        original_store = runtime_engine._execution_service._store_task_output

        async def delayed_store(
            output: JsonValue,
            *,
            tenant_id: str,
        ) -> StoredPayload:
            store_entered.set()
            await release_store.wait()
            return await original_store(output, tenant_id=tenant_id)

        monkeypatch.setattr(
            runtime_engine._execution_service,
            "_store_task_output",
            delayed_store,
        )
        graph_run = await runtime_engine.with_definitions(task).start(
            graph,
            idempotency_key="function-output-race-graph-0001",
         principal=runtime.default_principal)
        try:
            await asyncio.wait_for(store_entered.wait(), 3)
            graph_state = await runtime._state.task.tasks.graph_state(
                graph.graph_id,
                tenant_id=runtime.tenant_id,
            )
            assert graph_state is not None
            execution_id = graph_state.node_states[0].execution_id
            assert execution_id is not None
            cancelled = await runtime_engine._execution_service.cancel_task(
                execution_id,
                principal=runtime.default_principal,
            )
            assert cancelled.cancelled
        finally:
            release_store.set()

        result = (await graph_run.wait(timeout_seconds=5)).result
        execution = await runtime_engine._execution_service.inspect(
            execution_id,
            principal=runtime.default_principal,
        )
        graph_state = await runtime._state.task.tasks.graph_state(
            graph.graph_id,
            tenant_id=runtime.tenant_id,
        )

        assert result.wait_status is TaskStatus.CANCELLED
        assert execution.status.value == "CANCELLED"
        assert graph_state is not None
        assert graph_state.node_states[0].status is TaskStatus.CANCELLED
        assert graph_state.node_states[0].result_digest is None


@pytest.mark.asyncio
async def test_pending_node_cancel_recovery_keeps_running_sibling_alive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage = PersistenceStorage.in_memory()
    target_started = asyncio.Event()
    sibling_started = asyncio.Event()
    release_sibling = asyncio.Event()
    sibling_calls = 0

    async def run_target(context: TaskNodeContext[None]) -> JsonValue:
        del context
        target_started.set()
        await asyncio.Event().wait()
        return {"target": "unreachable"}

    async def run_sibling(context: TaskNodeContext[None]) -> JsonValue:
        nonlocal sibling_calls
        del context
        sibling_calls += 1
        sibling_started.set()
        await release_sibling.wait()
        return {"sibling": "completed"}

    target = Task("example.node-cancel-target", run_target, effect_policy="none")
    sibling = Task("example.node-cancel-sibling", run_sibling, effect_policy="none")
    graph = TaskGraph(
        "node-cancel-recovery",
        (
            TaskNode("target", task=target),
            TaskNode("sibling", task=sibling),
        ),
    )
    async with Persistence.open('node-cancel-recovery-runtime', storage=storage) as persistence_784, Runtime.attach(persistence=persistence_784, features=model_features(_TaskTestModels())) as runtime, task_engine_scope(runtime, persistence_784) as runtime_engine:
        graph_run = await runtime_engine.with_definitions(target, sibling).start(
            graph,
            idempotency_key="node-cancel-recovery-graph-0001",
         principal=runtime.default_principal)
        await asyncio.wait_for(target_started.wait(), 3)
        await asyncio.wait_for(sibling_started.wait(), 3)

        original_record_success = runtime_engine._graph_service._record_success
        injected_ack_failure = False

        async def fail_node_cancel_ack(operation, tenant_id, view, **kwargs):
            nonlocal injected_ack_failure
            if (
                getattr(operation, "operation_kind", None) is OperationKind.TASK_CANCEL
                and getattr(operation, "execution_id", None) is not None
                and not injected_ack_failure
            ):
                injected_ack_failure = True
                raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)
            return await original_record_success(
                operation,
                tenant_id,
                view,
                **kwargs,
            )

        monkeypatch.setattr(
            runtime_engine._graph_service,
            "_record_success",
            fail_node_cancel_ack,
        )
        graph_state = await graph_run.state(include_content=True)
        target_state = next(
            item for item in graph_state.node_states if item.node_id == "target"
        )
        assert target_state.execution_id is not None
        execution = await graph_run.execution("target")
        with pytest.raises(AIError) as cancel_error:
            await execution.cancel(idempotency_key="node-cancel-recovery-cancel-0001")
        assert cancel_error.value.code is ErrorCode.STORAGE_RECOVERY_REQUIRED
        assert injected_ack_failure

        target_execution = await runtime.executions.inspect(
            target_state.execution_id,
            principal=runtime.default_principal,
        )
        cancelled_graph_state = await graph_run.state(include_content=True)
        cancelled_target = next(
            item
            for item in cancelled_graph_state.node_states
            if item.node_id == "target"
        )
        still_running_sibling = next(
            item
            for item in cancelled_graph_state.node_states
            if item.node_id == "sibling"
        )
        assert target_execution.status.value == "CANCELLED"
        assert cancelled_target.status is TaskStatus.CANCELLED
        assert still_running_sibling.status is TaskStatus.RUNNING
        pending_cancel = await runtime._state.task.operations.get(
            idempotency_key_digest("node-cancel-recovery-cancel-0001"),
            tenant_id=runtime.tenant_id,
        )
        assert pending_cancel is not None
        assert pending_cancel.status.value in {"RUNNING", "EFFECT_UNKNOWN"}
        assert pending_cancel.execution_id == target_state.execution_id

        recovered = await graph_run.recover(
            idempotency_key="node-cancel-recovery-recover-0001",
        )
        assert recovered.status is TaskStatus.RUNNING
        after_recovery = await graph_run.state(include_content=True)
        after_recovery_sibling = next(
            item for item in after_recovery.node_states if item.node_id == "sibling"
        )
        assert after_recovery_sibling.status is TaskStatus.RUNNING
        assert sibling_calls == 1
        settled_cancel = await runtime._state.task.operations.get(
            idempotency_key_digest("node-cancel-recovery-cancel-0001"),
            tenant_id=runtime.tenant_id,
        )
        assert settled_cancel is not None
        assert settled_cancel.status.value == "SUCCEEDED"

        release_sibling.set()
        completed = (await graph_run.wait(timeout_seconds=5)).result
        final_state = await graph_run.state(include_content=True)
        completed_sibling = next(
            item for item in final_state.node_states if item.node_id == "sibling"
        )
        assert completed.wait_status is TaskStatus.CANCELLED
        assert completed_sibling.status is TaskStatus.SUCCEEDED
        assert sibling_calls == 1


@pytest.mark.asyncio
async def test_terminal_node_cancel_recovery_settles_and_replays_same_receipt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage = PersistenceStorage.in_memory()
    task_started = asyncio.Event()

    async def hold_task(context: TaskNodeContext[None]) -> JsonValue:
        del context
        task_started.set()
        await asyncio.Event().wait()
        return {"unreachable": True}

    task = Task("example.terminal-node-cancel-recovery", hold_task, effect_policy="none")
    graph = TaskGraph(
        "terminal-node-cancel-recovery",
        (TaskNode("target", task=task),),
    )
    async with Persistence.open('terminal-node-cancel-recovery-runtime', storage=storage) as persistence_903, Runtime.attach(persistence=persistence_903, features=model_features(_TaskTestModels())) as runtime, task_engine_scope(runtime, persistence_903) as runtime_engine:
        graph_run = await runtime_engine.with_definitions(task).start(
            graph,
            idempotency_key="terminal-node-cancel-recovery-graph-0001",
         principal=runtime.default_principal)
        await asyncio.wait_for(task_started.wait(), 3)
        initial_state = await graph_run.state(include_content=True)
        target = initial_state.node_states[0]
        assert target.execution_id is not None

        original_cancel_node = runtime._state.task.tasks.cancel_node
        interrupted_projection = False

        async def fail_confirmed_projection(
            graph_id: str,
            node_id: str,
            *,
            tenant_id: str,
            execution_id: str,
            cancel_confirmed: bool = False,
            expected_fence: int | None = None,
        ) -> TaskGraphView:
            nonlocal interrupted_projection
            if cancel_confirmed and not interrupted_projection:
                interrupted_projection = True
                raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)
            return await original_cancel_node(
                graph_id,
                node_id,
                tenant_id=tenant_id,
                execution_id=execution_id,
                cancel_confirmed=cancel_confirmed,
                expected_fence=expected_fence,
            )

        monkeypatch.setattr(
            runtime._state.task.tasks,
            "cancel_node",
            fail_confirmed_projection,
        )
        original_execution_cancel = runtime_engine._execution_service.cancel_task
        execution_cancel_calls = 0

        async def count_execution_cancel(
            execution_id: str,
            *,
            principal: Principal,
        ):
            nonlocal execution_cancel_calls
            execution_cancel_calls += 1
            return await original_execution_cancel(
                execution_id,
                principal=principal,
            )

        monkeypatch.setattr(
            runtime_engine._execution_service,
            "cancel_task",
            count_execution_cancel,
        )
        execution = await graph_run.execution("target")
        with pytest.raises(AIError) as cancel_error:
            await execution.cancel(
                idempotency_key="terminal-node-cancel-recovery-cancel-0001"
            )
        assert cancel_error.value.code is ErrorCode.STORAGE_RECOVERY_REQUIRED
        assert interrupted_projection
        assert execution_cancel_calls == 1

        durable_execution = await runtime.executions.inspect(
            target.execution_id,
            principal=runtime.default_principal,
        )
        interrupted_state = await graph_run.state(include_content=True)
        assert durable_execution.status.value == "CANCELLED"
        assert interrupted_state.status is TaskStatus.RECOVERY_REQUIRED
        assert interrupted_state.node_states[0].status is TaskStatus.RECOVERY_REQUIRED

        recovery_key = "terminal-node-cancel-recovery-recover-0001"
        recovered = await graph_run.recover(idempotency_key=recovery_key)
        cancel_receipt = await runtime._state.task.operations.get(
            idempotency_key_digest("terminal-node-cancel-recovery-cancel-0001"),
            tenant_id=runtime.tenant_id,
        )
        recovery_receipt = await runtime._state.task.operations.get(
            idempotency_key_digest(recovery_key),
            tenant_id=runtime.tenant_id,
        )
        assert recovered.status is TaskStatus.CANCELLED
        assert cancel_receipt is not None
        assert cancel_receipt.status.value == "SUCCEEDED"
        assert recovery_receipt is not None
        assert recovery_receipt.status.value == "SUCCEEDED"
        assert execution_cancel_calls == 1

        replayed = await graph_run.recover(idempotency_key=recovery_key)
        assert replayed.graph_id == recovered.graph_id
        assert replayed.status is TaskStatus.CANCELLED
        assert execution_cancel_calls == 1


@pytest.mark.asyncio
async def test_task_observer_exception_is_visible_without_changing_graph_state() -> None:
    entered = asyncio.Event()
    release = asyncio.Event()

    async def hold_task(context: TaskNodeContext[None]) -> JsonValue:
        del context
        entered.set()
        await release.wait()
        return {"value": "finished"}

    task = Task("example.observer-failure", hold_task, effect_policy="none")
    graph = TaskGraph(
        "observer-failure",
        (TaskNode("node", task=task),),
    )
    async with Persistence.open('task-observer-failure', storage=PersistenceStorage.in_memory()) as persistence_1024, Runtime.attach(persistence=persistence_1024, features=model_features(_TaskTestModels())) as runtime, task_engine_scope(runtime, persistence_1024) as runtime_engine:
        graph_run = await runtime_engine.with_definitions(task).start(
            graph,
            idempotency_key="observer-failure-graph-0001",
         principal=runtime.default_principal)
        await asyncio.wait_for(entered.wait(), 3)
        observed = 0

        async def observer(event: object) -> None:
            nonlocal observed
            del event
            observed += 1
            raise RuntimeError("observer callback failed")

        with pytest.raises(ObservationError) as observer_error:
            await graph_run.wait(on_event=observer, timeout_seconds=5)
        current = await graph_run.state(include_content=True)
        assert observer_error.value.code is ErrorCode.OBSERVER_FAILED
        assert observer_error.value.origin == "callback"
        assert observed == 1
        assert current.status is TaskStatus.RUNNING

        release.set()
        completed = (await graph_run.wait(timeout_seconds=5)).result
        assert completed.wait_status is TaskStatus.SUCCEEDED


@pytest.mark.asyncio
async def test_task_execution_cancel_invokes_cancel_callback_once() -> None:
    started = asyncio.Event()
    cancel_calls = 0

    async def run_task(context: TaskNodeContext[None]) -> JsonValue:
        del context
        started.set()
        await asyncio.Event().wait()
        return {"unreachable": True}

    async def cancel_task(context: TaskNodeContext[None]) -> None:
        nonlocal cancel_calls
        del context
        cancel_calls += 1

    task = Task(
        "example.cancel-callback-once",
        run_task,
        effect_policy="none",
        cancel=cancel_task,
    )
    graph = TaskGraph(
        "cancel-callback-once",
        (TaskNode("node", task=task),),
    )
    async with Persistence.open('cancel-callback-once-runtime', storage=PersistenceStorage.in_memory()) as persistence_1081, Runtime.attach(persistence=persistence_1081, features=model_features(_TaskTestModels())) as runtime, task_engine_scope(runtime, persistence_1081) as runtime_engine:
        graph_run = await runtime_engine.with_definitions(task).start(
            graph,
            idempotency_key="cancel-callback-once-graph-0001",
         principal=runtime.default_principal)
        await asyncio.wait_for(started.wait(), 3)
        execution = await graph_run.execution("node")
        first = await execution.cancel(
            idempotency_key="cancel-callback-once-operation-0001"
        )
        replay = await execution.cancel(
            idempotency_key="cancel-callback-once-operation-0001"
        )
        result = (await graph_run.wait(timeout_seconds=5)).result

        assert first.cancelled
        assert replay.cancelled
        assert result.wait_status is TaskStatus.CANCELLED
        assert cancel_calls == 1


@pytest.mark.asyncio
async def test_cancel_recovery_required_execution_preserves_recovery_boundary() -> None:
    started = asyncio.Event()

    async def uncertain_task(context: TaskNodeContext[None]) -> JsonValue:
        del context
        started.set()
        raise RuntimeError("effect outcome unknown")

    task = Task(
        "example.cancel-recovery-required",
        uncertain_task,
        effect_policy="non_replay_safe",
    )
    graph = TaskGraph(
        "cancel-recovery-required",
        (TaskNode("node", task=task),),
    )
    async with Persistence.open('cancel-recovery-required-runtime', storage=PersistenceStorage.in_memory()) as persistence_1124, Runtime.attach(persistence=persistence_1124, features=model_features(_TaskTestModels())) as runtime, task_engine_scope(runtime, persistence_1124) as runtime_engine:
        graph_run = await runtime_engine.with_definitions(task).start(
            graph,
            idempotency_key="cancel-recovery-required-graph-0001",
         principal=runtime.default_principal)
        await asyncio.wait_for(started.wait(), 3)
        initial = (await graph_run.wait(timeout_seconds=5)).result
        assert initial.wait_status is TaskStatus.RECOVERY_REQUIRED

        execution = await graph_run.execution("node")
        with pytest.raises(AIError) as raised:
            await execution.cancel(
                idempotency_key="cancel-recovery-required-operation-0001"
            )
        assert raised.value.code is ErrorCode.STORAGE_RECOVERY_REQUIRED

        current = await graph_run.state(include_content=True)
        assert current.status is TaskStatus.RECOVERY_REQUIRED
        assert current.node_states[0].status is TaskStatus.RECOVERY_REQUIRED


@pytest.mark.asyncio
async def test_task_execution_cancel_persists_node_intent_before_execution_io(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = PersistenceStorage.in_memory()
    async with Persistence.open('task-execution-cancel-order', storage=state) as persistence_1154, Runtime.attach(persistence=persistence_1154, features=model_features(_TaskTestModels())) as runtime, task_engine_scope(runtime, persistence_1154) as runtime_engine:
        class _WaitingExecutionRunner:
            def __init__(self) -> None:
                self.execution_id: str | None = None
                self.entered = asyncio.Event()

            async def run(
                self,
                invocation: TaskNodeInvocation,
                *,
                control: TaskNodeRunControl,
            ) -> TaskNodeRunResult:
                binding = TaskBindingContract(
                    id="example.execution-cancel-order",
                    revision=1,
                    effect_policy="none",
                    output_contract={"kind": "json"},
                    timeout_seconds=None,
                    max_attempts=1,
                    retry_delay_seconds=0,
                )
                handle = await runtime_engine._execution_service.start_task(
                    binding,
                    principal=invocation.principal,
                    input=invocation.node.input,
                    idempotency_key="execution-cancel-order-start-0001",
                    correlation=invocation.correlation,
                )
                self.execution_id = handle.execution_id
                await control.bind_execution(handle.execution_id)
                await control.handoff_execution(handle.execution_id)
                self.entered.set()
                await asyncio.Event().wait()
                raise AssertionError("cancelled runner unexpectedly resumed")

            async def wait_bound(
                self,
                invocation: TaskNodeInvocation,
                execution_id: str,
            ) -> TaskNodeRunResult:
                del invocation
                raise TaskNodeRunError(ErrorCode.EXECUTION_CANCELLED, execution_id)

            async def supply_input(
                self,
                invocation: TaskNodeInvocation,
                execution_id: str,
                value: JsonValue,
            ) -> TaskNodeRunResult:
                del invocation, execution_id, value
                raise AIError(ErrorCode.TASK_NOT_READY)

            async def resolve_effect(
                self,
                invocation: TaskNodeInvocation,
                execution_id: str,
                resolution: TaskEffectResolution,
            ) -> TaskNodeRunResult | None:
                del invocation, execution_id, resolution
                raise AIError(ErrorCode.TASK_NOT_READY)

            async def cancel(self, invocation: TaskNodeInvocation) -> None:
                del invocation

        runner = _WaitingExecutionRunner()
        original_cancel = runtime_engine._execution_service.cancel_task
        observed_intent = asyncio.Event()

        async def verify_intent_before_cancel(
            execution_id: str,
            *,
            principal: Principal,
        ):
            graph_state = await runtime._state.task.tasks.graph_state(
                "task-execution-cancel-order",
                tenant_id=principal.tenant_id,
            )
            assert graph_state is not None
            node_state = graph_state.node_states[0]
            assert node_state.execution_id == execution_id
            assert node_state.status is TaskStatus.RECOVERY_REQUIRED
            observed_intent.set()
            return await original_cancel(execution_id, principal=principal)

        monkeypatch.setattr(
            runtime_engine._execution_service,
            "cancel_task",
            verify_intent_before_cancel,
        )
        task = Task.from_runner(
            "example.execution-cancel-order",
            runner,
            contract={
                "version": 1,
                "type": "example.execution-cancel-order",
                "effect_policy": "none",
                "output_contract": {"kind": "json"},
                "reconcile": False,
            },
        )
        graph = TaskGraph(
            "task-execution-cancel-order",
            (TaskNode("node", task=task),),
        )
        graph_run = await runtime_engine.with_definitions(task).start(
            graph,
            idempotency_key="task-execution-cancel-order-graph-0001",
         principal=runtime.default_principal)
        await asyncio.wait_for(runner.entered.wait(), 3)
        assert runner.execution_id is not None
        execution = await graph_run.execution("node")
        cancelled = await execution.cancel(
            idempotency_key="task-execution-cancel-order-cancel-0001"
        )

        result = (await graph_run.wait(timeout_seconds=5)).result
        assert cancelled.cancelled
        assert observed_intent.is_set()
        assert result.wait_status is TaskStatus.CANCELLED


@pytest.mark.asyncio
async def test_cancel_and_reopen_preserve_accepted_execution_before_task_binding(
    tmp_path: Path,
) -> None:
    storage_root = tmp_path / "accepted-before-bind"

    class _AcceptedBeforeBindRunner:
        def __init__(self) -> None:
            self.engine: TaskEngine[object] | None = None
            self.execution_id: str | None = None
            self.run_calls = 0
            self.accepted = asyncio.Event()
            self.runner_cancelled = asyncio.Event()
            self.cancel_received = asyncio.Event()

        async def run(
            self,
            invocation: TaskNodeInvocation,
            *,
            control: TaskNodeRunControl,
        ) -> TaskNodeRunResult:
            del control
            self.run_calls += 1
            assert self.engine is not None
            binding = TaskBindingContract(
                id="example.accepted-before-bind",
                revision=1,
                effect_policy="none",
                output_contract={"kind": "json"},
                timeout_seconds=None,
                max_attempts=1,
                retry_delay_seconds=0,
            )
            handle = await self.engine._execution_service.start_task(
                binding,
                principal=invocation.principal,
                input=invocation.node.input,
                idempotency_key="accepted-before-bind-execution-0001",
                correlation=invocation.correlation,
            )
            self.execution_id = handle.execution_id
            self.accepted.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                self.runner_cancelled.set()
                raise
            raise AssertionError("unbound runner unexpectedly continued")

        async def wait_bound(
            self,
            invocation: TaskNodeInvocation,
            execution_id: str,
        ) -> TaskNodeRunResult:
            del invocation, execution_id
            raise AssertionError("unbound accepted execution must stay unresolved")

        async def inspect_bound(
            self,
            invocation: TaskNodeInvocation,
            execution_id: str,
        ) -> TaskNodeRunResult | None:
            del invocation, execution_id
            return None

        async def cancel(self, invocation: TaskNodeInvocation) -> None:
            assert invocation.execution_id is None
            self.cancel_received.set()

    runner = _AcceptedBeforeBindRunner()
    task = Task.from_runner(
        "example.accepted-before-bind",
        runner,
        contract={
            "version": 1,
            "type": "example.accepted-before-bind",
            "effect_policy": "none",
            "output_contract": {"kind": "json"},
            "reconcile": False,
        },
    )
    graph = TaskGraph("accepted-before-bind", (TaskNode("node", task=task),))
    principal: Principal

    async with Persistence.open('default', storage=PersistenceStorage.filesystem(storage_root)) as persistence_1363, Runtime.attach(persistence=persistence_1363, features=model_features(_TaskTestModels())) as runtime, task_engine_scope(runtime, persistence_1363) as runtime_engine:
        runner.engine = runtime_engine
        principal = runtime.default_principal
        graph_run = await runtime_engine.with_definitions(task).start(
            graph,
            idempotency_key="accepted-before-bind-graph-0001",
         principal=runtime.default_principal)
        await asyncio.wait_for(runner.accepted.wait(), 3)
        assert runner.execution_id is not None

        before_cancel = await task_graph_state(
            runtime,
            graph.graph_id,
            principal=principal,
        )
        assert before_cancel.node_states[0].status is TaskStatus.RUNNING
        assert before_cancel.node_states[0].execution_id is None

        cancelled = await graph_run.cancel(
            idempotency_key="accepted-before-bind-cancel-0001",
        )
        execution = await runtime.executions.inspect(
            runner.execution_id,
            principal=principal,
        )
        after_cancel = await task_graph_state(
            runtime,
            graph.graph_id,
            principal=principal,
        )

        assert cancelled.status is TaskStatus.RECOVERY_REQUIRED
        assert runner.runner_cancelled.is_set()
        assert not runner.cancel_received.is_set()
        assert after_cancel.node_states[0].status is TaskStatus.RECOVERY_REQUIRED
        assert after_cancel.node_states[0].execution_id is None
        assert after_cancel.node_states[0].error_code is not None
        assert execution.status.value == "STARTED"
        assert execution.task_attempt == 0

    async with Persistence.open('default', storage=PersistenceStorage.filesystem(storage_root)) as persistence_1407, Runtime.attach(persistence=persistence_1407, features=model_features(_TaskTestModels())) as reopened_runtime, task_engine_scope(reopened_runtime, persistence_1407) as reopened_runtime_engine:
        runner.runtime = reopened_runtime
        reopened_run = await reopened_runtime_engine.with_definitions(task).get(
            graph.graph_id,
            principal=reopened_runtime.default_principal,
        )
        recovered = await reopened_run.recover(
            idempotency_key="accepted-before-bind-recover-0001",
        )
        state_after_reopen = await task_graph_state(
            reopened_runtime,
            graph.graph_id,
            principal=reopened_runtime.default_principal,
        )
        execution = await reopened_runtime.executions.inspect(
            runner.execution_id,
            principal=reopened_runtime.default_principal,
        )

        assert recovered.status is TaskStatus.RECOVERY_REQUIRED
        assert state_after_reopen.node_states[0].status is TaskStatus.RECOVERY_REQUIRED
        assert state_after_reopen.node_states[0].execution_id is None
        assert execution.status.value == "STARTED"
        assert execution.task_attempt == 0
        assert runner.run_calls == 1


@pytest.mark.asyncio
async def test_reconciled_attempt_rejects_late_callback_from_same_execution() -> None:
    state = PersistenceStorage.in_memory()
    async with Persistence.open('task-attempt-fence-race', storage=state) as persistence_1441, Runtime.attach(persistence=persistence_1441, features=model_features(_TaskTestModels())) as runtime, task_engine_scope(runtime, persistence_1441) as runtime_engine:
        principal = runtime.default_principal
        binding = TaskBindingContract(
            id="example.task-attempt-fence",
            revision=1,
            effect_policy="non_replay_safe",
            output_contract={"kind": "json"},
            timeout_seconds=None,
            max_attempts=3,
            retry_delay_seconds=0,
        )
        execution = await runtime_engine._execution_service.start_task(
            binding,
            principal=principal,
            input={},
            idempotency_key="task-attempt-fence-start-0001",
            correlation={},
        )
        first = await runtime_engine._execution_service.claim_task_attempt(
            execution.execution_id,
            principal=principal,
        )
        assert first.task_attempt == 1
        old_callback_entered = asyncio.Event()
        release_old_callback = asyncio.Event()

        async def old_callback() -> object:
            old_callback_entered.set()
            await release_old_callback.wait()
            return await runtime_engine._execution_service.complete_task(
                execution.execution_id,
                principal=principal,
                output={"value": "late-old-attempt"},
                attempt=first,
            )

        late_callback = asyncio.create_task(old_callback())

        async def assert_stale(action: object) -> None:
            with pytest.raises(AIError) as rejected:
                await action()  # type: ignore[operator]
            assert rejected.value.code is ErrorCode.TASK_FENCE_STALE

        try:
            await asyncio.wait_for(old_callback_entered.wait(), 1)
            await runtime_engine._execution_service.require_task_recovery(
                execution.execution_id,
                principal=principal,
                error_code=ErrorCode.TASK_EFFECT_UNKNOWN.value,
                attempt=first,
            )
            resolved = await runtime_engine._execution_service.resolve_task_effect(
                execution.execution_id,
                principal=principal,
                resolution=TaskEffectResolution("not_applied"),
            )
            assert resolved.status.value == "WAITING_RETRY"
            second = await runtime_engine._execution_service.claim_task_attempt(
                execution.execution_id,
                principal=principal,
            )
            assert second.task_attempt == 2

            release_old_callback.set()
            await assert_stale(lambda: late_callback)
        finally:
            release_old_callback.set()
            await asyncio.gather(late_callback, return_exceptions=True)

        await assert_stale(
            lambda: runtime_engine._execution_service.fail_task(
                execution.execution_id,
                principal=principal,
                error=AIError(ErrorCode.TASK_NODE_FAILED),
                attempt=first,
            )
        )
        await assert_stale(
            lambda: runtime_engine._execution_service.schedule_task_retry(
                execution.execution_id,
                principal=principal,
                error_code=ErrorCode.TASK_NODE_FAILED.value,
                attempt=first,
            )
        )
        await assert_stale(
            lambda: runtime_engine._execution_service.require_task_recovery(
                execution.execution_id,
                principal=principal,
                error_code=ErrorCode.TASK_EFFECT_UNKNOWN.value,
                attempt=first,
            )
        )

        current = await runtime_engine._execution_service.inspect(
            execution.execution_id,
            principal=principal,
        )
        assert current.status.value == "STARTED"
        assert current.task_attempt == 2


def test_task_node_separates_authoring_from_resolved_contract() -> None:
    with pytest.raises(TypeError):
        TaskNode("node", effect_policy="replay_safe")  # type: ignore[call-arg]

    resolved = TaskNode.from_resolved(
        "node",
        task=TaskRef("example.direct", 1),
        effect_policy="replay_safe",
        output_contract={"mode": "structured", "schema": {"type": "object"}},
        reconcile=True,
    )

    assert resolved.effect_policy == "replay_safe"
    assert resolved.reconcile is True
    assert resolved.output_contract == {
        "mode": "structured",
        "schema": {"type": "object"},
    }


def test_runner_task_contract_matches_durable_capture_shape() -> None:
    runner = _ContractTaskRunner()
    contract: dict[str, JsonValue] = {
        "version": 1,
        "type": "example.runner",
        "effect_policy": "none",
        "output_contract": {"kind": "json"},
        "reconcile": False,
    }

    with pytest.raises(ValueError):
        Task.from_runner(
            "example.runner.extra",
            runner,
            contract={**contract, "unexpected": True},
        )

    with pytest.raises(ValueError):
        Task.from_runner(
            "example.runner.schema",
            runner,
            contract={
                **contract,
                "output_contract": {
                    "kind": "schema",
                    "schema": {"type": "not-a-json-schema-type"},
                },
            },
        )

    configured = Task.from_runner(
        "example.runner.configured",
        runner,
        contract={**contract, "config": {"owner": {"mode": "custom"}}},
    )
    assert configured.contract["config"] == {"owner": {"mode": "custom"}}


@pytest.mark.asyncio
async def test_runtime_runner_cannot_commit_unreadable_success() -> None:
    runner = _ContractTaskRunner()
    task = Task.from_runner(
        "example.unreadable-runner",
        runner,
        contract={
            "version": 1,
            "type": "example.runner",
            "effect_policy": "none",
            "output_contract": {"kind": "json"},
            "reconcile": False,
        },
    )
    state = PersistenceStorage.in_memory()
    async with Persistence.open('unreadable-runner', storage=state) as persistence_1620, Runtime.attach(persistence=persistence_1620, features=model_features(_TaskTestModels())) as runtime, task_engine_scope(runtime, persistence_1620) as runtime_engine:
        run = await runtime_engine.with_definitions(task).start(
            TaskGraph(
                "unreadable-runner-graph",
                (TaskNode("runner", task=task),),
            ),
            idempotency_key="unreadable-runner-run-0001",
         principal=runtime.default_principal)
        completed = (await run.wait(timeout_seconds=10)).result

        assert completed.wait_status is TaskStatus.FAILED
        assert completed.node_states[0].error_code == (
            ErrorCode.STORAGE_INTEGRITY_ERROR.value
        )
        with pytest.raises(AIError) as result_error:
            await run.result("runner")
        assert result_error.value.code is ErrorCode.TASK_NODE_FAILED

    assert runner.calls == 1


@pytest.mark.parametrize("authoring", (False, True))
def test_agent_task_input_keeps_but_excludes_unknown_additive_fields(authoring: bool) -> None:
    value = dict(
        AgentTaskInput(
            "prompt",
            parameters={"task_id": "business"},
            planning=False,
            thinking=False,
        )
    )
    value["metadata"] = {"source": "host"}

    value["capture_context"] = ExecutionInputContext.from_messages(()).to_payload()
    value["capture_files"] = [{"path": "input.txt"}]
    restored = (
        AgentTaskInput.from_authoring(value) if authoring
        else AgentTaskInput.from_mapping(value)
    )

    assert restored.prompt == "prompt"
    assert restored.parameters == {"task_id": "business"}
    assert restored["metadata"] == {"source": "host"}
    assert "metadata" not in restored.execution_payload()
    assert dict(restored.execution_payload()) == {
        key: item for key, item in value.items() if key != "metadata"
    }


def test_agent_task_input_authoring_defaults_and_durable_required_fields() -> None:
    authored = AgentTaskInput.from_authoring({"prompt": "prompt"})
    assert set(authored) == {
        "kind",
        "version",
        "prompt",
        "parameters",
        "files",
        "session_id",
        "planning",
        "thinking",
    }
    assert authored.planning is None
    assert authored.thinking is None

    durable = dict(
        AgentTaskInput("prompt", planning=False, thinking=False)
    )
    assert AgentTaskInput.from_mapping(durable).thinking is False
    assert AgentTaskInput.from_mapping(durable).planning is False
    for field in durable:
        missing = dict(durable)
        del missing[field]
        with pytest.raises(AIError) as error:
            AgentTaskInput.from_mapping(missing)
        assert error.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("kind", 1),
        ("version", "1"),
        ("prompt", []),
        ("parameters", []),
        ("files", "file.txt"),
        ("session_id", 1),
        ("planning", None),
        ("thinking", 1),
    ),
)
def test_agent_task_input_durable_known_fields_fail_closed(
    field: str,
    value: JsonValue,
) -> None:
    durable = dict(AgentTaskInput("prompt", planning=False, thinking=False))
    durable[field] = value

    with pytest.raises(AIError) as error:
        AgentTaskInput.from_mapping(durable)

    assert error.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


@pytest.mark.parametrize("name", ("metadata", "memory_scope"))
def test_agent_task_input_extensions_stay_out_of_execution_payload(name: str) -> None:
    authored = AgentTaskInput.from_authoring({
        "prompt": "prompt", "planning": False, "thinking": False,
        name: {"application": "value"},
    })
    restored = AgentTaskInput.from_mapping(dict(authored))
    assert dict(restored) == dict(authored)
    assert restored[name] == {"application": "value"}
    assert name not in restored.execution_payload()


@pytest.mark.parametrize("thinking", (None, False, "max"))
def test_agent_task_input_durable_thinking_preserves_inheritance_and_explicit_values(thinking) -> None:
    durable = dict(AgentTaskInput("prompt", planning=False, thinking=thinking))
    restored = AgentTaskInput.from_mapping(durable)
    assert restored.thinking == thinking
    assert restored["thinking"] == thinking


def test_agent_task_input_unknown_durable_version_is_unsupported() -> None:
    durable = dict(AgentTaskInput("prompt", planning=False, thinking=False))
    durable["version"] = 2

    with pytest.raises(AIError) as error:
        AgentTaskInput.from_mapping(durable)

    assert error.value.code is ErrorCode.STORAGE_VERSION_UNSUPPORTED


@pytest.mark.asyncio
async def test_recovery_preflight_preserves_unknown_agent_input_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entered = asyncio.Event()
    release = asyncio.Event()

    async def hold(context: TaskNodeContext[None]) -> JsonValue:
        del context
        entered.set()
        await release.wait()
        return {"ready": True}

    application = RuntimeFeatures[None]("application")
    gate = TaskFunction[None]("test.recovery-version-gate", 1, hold)
    application.task(gate, effect_policy="none")
    application.agent(
        "default",
        model="default",
        allow_tools=(),
        allow_skills=(),
        allow_subagents=(),
    )

    async with Persistence.open('recovery-version', storage=PersistenceStorage.in_memory()) as persistence_1771, Runtime.attach(persistence=persistence_1771, features=model_features(_TaskTestModels(), application)) as runtime, task_engine_scope(runtime, persistence_1771) as runtime_engine:
        agent_task = Task.from_agent(
            "test.recovery-version-agent",
            runtime.agents.get("default"),
        )
        agent_node = TaskNode(
            "agent",
            ("gate",),
            task=agent_task,
            input=AgentTaskInput("prompt"),
        )
        engine = runtime_engine.with_definitions(gate, agent_task)
        graph = TaskGraph(
            "recovery-version",
            (gate.node("gate"), agent_node),
        )
        run = await engine.start(
            graph,
            idempotency_key="recovery-version-run-0001",
        )
        await asyncio.wait_for(entered.wait(), 10)

        preflight = runtime_engine._graph_service._preflight
        original_validate_recovery = preflight.validate_recovery

        def corrupting_validate_recovery(state: TaskGraphState) -> None:
            nodes = []
            for node in state.nodes:
                if node.node_id == "agent":
                    input_value = dict(node.input)
                    input_value["version"] = 2
                    node = TaskNode.from_resolved(
                        node.node_id,
                        node.dependencies,
                        task=node.task,
                        input=input_value,
                        budget_cost=node.budget_cost,
                        expander=node.expander,
                        input_refs=node.input_refs,
                        timeout_seconds=node.timeout_seconds,
                        max_attempts=node.max_attempts,
                        retry_delay_seconds=node.retry_delay_seconds,
                        output_contract=node.output_contract,
                        effect_policy=node.effect_policy,
                        reconcile=node.reconcile,
                        dependency_policy=node.dependency_policy,
                        failure_policy=node.failure_policy,
                    )
                nodes.append(node)
            original_validate_recovery(replace(state, nodes=tuple(nodes)))

        monkeypatch.setattr(
            preflight,
            "validate_recovery",
            corrupting_validate_recovery,
        )
        with pytest.raises(AIError) as unsupported:
            await runtime_engine.with_definitions(gate, agent_task).recover_pending( principal=runtime.default_principal)
        assert unsupported.value.code is ErrorCode.STORAGE_VERSION_UNSUPPORTED

        release.set()
        result = (await run.wait(timeout_seconds=10)).result
        assert result.wait_status is TaskStatus.SUCCEEDED


def test_agent_task_input_uses_current_request_digest_projection() -> None:
    original = dict(AgentTaskInput("prompt", planning=False, thinking=False))
    restored = AgentTaskInput.from_mapping(original)
    principal = Principal("principal", "tenant")
    invocation = TaskNodeInvocation(
        TaskNode("node", task=TaskRef("example.agent", 1)), "graph", principal, {}, {},
    )
    input_identity = canonical_sha256({
        "version": 1, "graph_id": "graph", "node_id": "node", "task_id": "example.agent",
        "task_revision": 1, "binding_digest": "a" * 64,
        "principal": principal_identity_payload(principal), "input": original, "output_contract": None,
    })
    assert _agent_task_input_identity(invocation, restored, task_id="example.agent", task_revision=1,
                                     binding_digest="a" * 64) == input_identity
    expected = canonical_sha256({
        "version": 1, "graph_id": "graph", "node_id": "node", "task_id": "example.agent",
        "task_revision": 1, "agent_binding_digest": "a" * 64,
        "principal": principal_identity_payload(principal), "input_identity": input_identity,
        "prompt": original["prompt"], "files": [], "final_input_digest": "b" * 64,
        "session_id": None, "planning": False,
        "thinking": False, "output_contract": None,
    })
    assert _agent_task_request_identity(invocation, task_input=restored, prompt="prompt",
                                       final_input_digest="b" * 64, input_identity=input_identity,
                                       task_id="example.agent", task_revision=1,
                                       binding_digest="a" * 64) == expected


@pytest.mark.parametrize("name", ("metadata", "memory_scope"))
def test_agent_task_input_identity_excludes_additive_fields(name: str) -> None:
    authored = dict(
        AgentTaskInput("prompt", planning=False, thinking=False)
    )
    plain = AgentTaskInput.from_mapping(authored)
    authored[name] = {"source": "host"}
    extended = AgentTaskInput.from_mapping(authored)
    invocation = TaskNodeInvocation(
        TaskNode("node", task=TaskRef("example.agent", 1)),
        "graph",
        Principal("principal", "tenant"),
        {},
        {},
    )

    assert _agent_task_input_identity(
        invocation,
        plain,
        task_id="example.agent",
        task_revision=1,
        binding_digest="a" * 64,
    ) == _agent_task_input_identity(
        invocation,
        extended,
        task_id="example.agent",
        task_revision=1,
        binding_digest="a" * 64,
    )


@pytest.mark.asyncio
async def test_runtime_agent_task_binding_is_owned_by_its_runtime() -> None:
    first_capabilities = RuntimeFeatures[None]("runtime-owner-first")
    first_capabilities.agent(
        "default",
        model="default",
        allow_tools=(),
        allow_skills=(),
        allow_subagents=(),
    )
    second_capabilities = RuntimeFeatures[None]("runtime-owner-second")
    second_capabilities.agent(
        "default",
        model="default",
        allow_tools=(),
        allow_skills=(),
        allow_subagents=(),
    )
    async with Persistence.open('runtime-owner-first', storage=PersistenceStorage.in_memory()) as persistence_1888, Runtime.attach(persistence=persistence_1888, features=model_features(_TaskTestModels(), first_capabilities)) as first_runtime, task_engine_scope(first_runtime, persistence_1888) as first_runtime_engine:
        async with Persistence.open('runtime-owner-second', storage=PersistenceStorage.in_memory()) as persistence_1893, Runtime.attach(persistence=persistence_1893, features=model_features(_TaskTestModels(), second_capabilities)) as second_runtime, task_engine_scope(second_runtime, persistence_1893) as second_runtime_engine:
            agent = first_runtime.agents.get("default")
            first_task = Task.from_agent("example.agent.first", agent)
            second_task = Task.from_agent("example.agent.second", agent)
            first_runtime_engine.with_definitions(first_task, second_task)

            with pytest.raises(AIError) as foreign_runtime:
                second_runtime_engine.with_definitions(first_task)
            assert foreign_runtime.value.code is ErrorCode.RUNTIME_SERVICE_MISMATCH

            changed_contract = dict(first_task.contract)
            changed_config = dict(changed_contract["config"])
            changed_config["agent_id"] = "edited"
            changed_contract["config"] = changed_config
            edited_task = Task.from_agent(first_task.id, agent)
            object.__setattr__(edited_task, "_contract", changed_contract)
            with pytest.raises(AIError) as edited_declaration:
                first_runtime_engine.with_definitions(edited_task)
            assert edited_declaration.value.code is ErrorCode.REQUEST_FIELD_INVALID

            class FakeAgentRunner:
                pass

            fake_agent_task = Task.from_runner(
                "example.agent.fake",
                FakeAgentRunner(),  # type: ignore[arg-type]
                contract={
                    "version": 1,
                    "type": "agent",
                    "effect_policy": "none",
                    "output_contract": {"kind": "json"},
                    "reconcile": False,
                    "config": {"agent_id": "default", "agent_revision": 1},
                },
            )
            with pytest.raises(AIError) as fake_runner:
                first_runtime_engine.with_definitions(fake_agent_task)
            assert fake_runner.value.code is ErrorCode.REQUEST_FIELD_INVALID

            incomplete_input_graph = TaskGraph(
                "incomplete-agent-task-input",
                (
                    TaskNode(
                        "node",
                        task=first_task,
                        input={"kind": "agent-task-input", "version": 1},
                    ),
                ),
            )
            with pytest.raises(AIError) as invalid_input:
                await first_runtime_engine.with_definitions(first_task).start(
                    incomplete_input_graph,
                    idempotency_key="incomplete-agent-input-run-0001",
                 principal=first_runtime.default_principal)
            assert invalid_input.value.code is ErrorCode.REQUEST_FIELD_INVALID
            assert await first_runtime_engine.persistence.task.admissions.get(
                incomplete_input_graph.graph_id,
                tenant_id=first_runtime.tenant_id,
            ) is None


@pytest.mark.asyncio
async def test_task_results_page_reads_only_page_states_and_preserves_null(
    monkeypatch,
) -> None:
    async def return_null(_context: TaskNodeContext[None]) -> JsonValue:
        return None

    tasks = tuple(
        Task(f"example.result-{name}", return_null, effect_policy="none")
        for name in ("a", "b", "c")
    )
    graph = TaskGraph(
        "bounded-task-results",
        tuple(
            TaskNode(f"node-{name}", task=task)
            for name, task in zip(("c", "a", "b"), tasks)
        ),
    )
    state = PersistenceStorage.in_memory()
    async with Persistence.open('bounded-task-results', storage=state) as persistence_1980, Runtime.attach(persistence=persistence_1980, features=model_features(_TaskTestModels())) as runtime, task_engine_scope(runtime, persistence_1980) as runtime_engine:
        run = await runtime_engine.with_definitions(*tasks).start(
            graph,
            idempotency_key="bounded-task-results-run-0001",
         principal=runtime.default_principal)
        completed = (await run.wait(timeout_seconds=10)).result
        assert completed.wait_status is TaskStatus.SUCCEEDED

        repository = runtime._state.task.tasks
        original_get_node_states = repository.get_node_states
        page_reads: list[tuple[str, ...]] = []

        async def get_node_states(
            graph_id: str,
            node_ids: tuple[str, ...],
            *,
            tenant_id: str,
        ):
            page_reads.append(node_ids)
            return await original_get_node_states(
                graph_id,
                node_ids,
                tenant_id=tenant_id,
            )

        async def reject_full_state(*args, **kwargs):
            del args, kwargs
            raise AssertionError("results() must not load full graph state")

        monkeypatch.setattr(repository, "get_node_states", get_node_states)
        monkeypatch.setattr(repository, "graph_state", reject_full_state)

        first = await run.results(limit=2)
        assert first.next_cursor is not None
        assert [item.node_id for item in first.items] == ["node-a", "node-b"]
        assert all(item.output is None and not item.content_included for item in first.items)

        second = await run.results(cursor=first.next_cursor, limit=2)
        assert second.next_cursor is None
        assert [item.node_id for item in second.items] == ["node-c"]
        assert page_reads == [("node-a", "node-b"), ("node-c",)]

        content_page = await run.results(limit=1, include_content=True)
        assert content_page.items[0].output is None
        assert content_page.items[0].content_included is True
        assert page_reads[-1] == ("node-a",)


@pytest.mark.asyncio
@pytest.mark.parametrize("agent_thinking", (None, "high"))
async def test_agent_task_thinking_false_and_none_resolve_before_model_calls(agent_thinking) -> None:
    application = RuntimeFeatures[None]("thinking-agent")
    application.agent(
        "default",
        model="default",
        allow_tools=(),
        allow_skills=(),
        allow_subagents=(),
        thinking=agent_thinking,
    )
    state = PersistenceStorage.in_memory()
    async with Persistence.open('agent-task-thinking', storage=state) as persistence_2045, Runtime.attach(persistence=persistence_2045, features=model_features(_ThinkingTaskTestModels(), application)) as runtime, task_engine_scope(runtime, persistence_2045) as runtime_engine:
        task = Task.from_agent("example.thinking-worker", runtime.agents.get())
        graph = TaskGraph(
            "agent-task-thinking-graph",
            (
                TaskNode(
                    "explicit-false",
                    task=task,
                    input=AgentTaskInput("false request", thinking=False),
                ),
                TaskNode(
                    "default-thinking",
                    task=task,
                    input=AgentTaskInput("default request"),
                ),
            ),
        )
        run = await runtime_engine.with_definitions(task).start(
            graph,
            idempotency_key="agent-task-thinking-run-0001",
         principal=runtime.default_principal)
        result = (await run.wait(timeout_seconds=10)).result

        assert result.wait_status is TaskStatus.SUCCEEDED, result.node_states
        state_view = await run.state(include_content=True)
        inputs = {node.node_id: node.input for node in state_view.nodes}
        assert inputs["explicit-false"]["thinking"] is False
        assert inputs["default-thinking"]["thinking"] == agent_thinking

        for node_id, expected in (("explicit-false", False), ("default-thinking", agent_thinking)):
            node_result = next(item for item in result.node_states if item.node_id == node_id)
            assert node_result.execution_id is not None
            execution = await runtime._state.execution.executions.get(
                node_result.execution_id,
                tenant_id=runtime.tenant_id,
            )
            assert execution.thinking == expected
            interactions = await runtime.history.model_interactions(
                node_result.execution_id,
                principal=runtime.default_principal,
                include_content=True,
                limit=10,
            )
            assert len(interactions.items) == 1
            assert interactions.items[0].status == "SUCCEEDED"


@pytest.mark.asyncio
async def test_task_view_rejects_duplicate_exact_definitions() -> None:
    definition = Task("example.echo", _echo_task)
    async with Persistence.open('duplicate-task-view', storage=PersistenceStorage.in_memory()) as persistence_2099, Runtime.attach(persistence=persistence_2099, features=model_features(_TaskTestModels())) as runtime, task_engine_scope(runtime, persistence_2099) as runtime_engine:
        with pytest.raises(AIError) as duplicate:
            runtime_engine.with_definitions(definition, definition)
    assert duplicate.value.code is ErrorCode.BINDING_CONFLICT


async def _echo_task(context: TaskNodeContext[None]) -> JsonValue:
    if not context.dependencies:
        return {"value": context.input.get("value")}
    name = next(iter(context.dependencies))
    dependency = context.dependencies[name]
    return {
        "upstream": await context.read_dependency(name),
        "execution_id": dependency.execution_id,
    }


def test_task_dependency_state_exposes_only_terminal_semantics() -> None:
    failed = TaskDependencyState(
        TaskStatus.FAILED,
        error_code=ErrorCode.REQUEST_FIELD_INVALID.value,
        error_digest="a" * 64,
    )
    assert failed.to_payload() == {
        "status": TaskStatus.FAILED.value,
        "error_code": ErrorCode.REQUEST_FIELD_INVALID.value,
        "error_digest": "a" * 64,
    }
    assert "execution_id" not in failed.to_payload()


@pytest.mark.parametrize(
    "dependency_policy",
    ("all_succeeded", "all_terminal", "any_succeeded"),
)
@pytest.mark.parametrize("failure_policy", ("propagate", "isolate"))
def test_task_node_policies_round_trip_on_the_explicit_single_wire(
    dependency_policy: str,
    failure_policy: str,
) -> None:
    node = TaskNode(
        "node",
        dependency_policy=dependency_policy,
        failure_policy=failure_policy,
    )

    wire = _encode_persisted_domain(node)

    assert wire["$dataclass"] == "task_node"
    assert wire["fields"]["dependency_policy"] == dependency_policy
    assert wire["fields"]["failure_policy"] == failure_policy
    assert decode_domain(wire, TaskNode) == node


@pytest.mark.parametrize(
    ("dependency_policy", "dependency_states", "expected"),
    (
        ("all_succeeded", {}, TaskStatus.READY),
        ("all_terminal", {}, TaskStatus.READY),
        ("any_succeeded", {}, TaskStatus.BLOCKED),
        (
            "all_succeeded",
            {"ok": TaskStatus.SUCCEEDED, "ignored": TaskStatus.FAILED},
            TaskStatus.READY,
        ),
        (
            "all_succeeded",
            {"bad": TaskStatus.FAILED, "waiting": TaskStatus.PENDING},
            TaskStatus.BLOCKED,
        ),
        (
            "all_succeeded",
            {"recovering": TaskStatus.RECOVERY_REQUIRED},
            TaskStatus.PENDING,
        ),
        (
            "all_terminal",
            {"ok": TaskStatus.SUCCEEDED, "bad": TaskStatus.CANCELLED},
            TaskStatus.READY,
        ),
        (
            "all_terminal",
            {"ok": TaskStatus.SUCCEEDED, "waiting": TaskStatus.WAITING},
            TaskStatus.PENDING,
        ),
        (
            "any_succeeded",
            {"ok": TaskStatus.SUCCEEDED, "waiting": TaskStatus.RUNNING},
            TaskStatus.PENDING,
        ),
        (
            "any_succeeded",
            {"ok": TaskStatus.SUCCEEDED, "bad": TaskStatus.BLOCKED},
            TaskStatus.READY,
        ),
        (
            "any_succeeded",
            {"bad": TaskStatus.FAILED, "cancelled": TaskStatus.CANCELLED},
            TaskStatus.BLOCKED,
        ),
        (
            "any_succeeded",
            {"bad": TaskStatus.FAILED, "recovering": TaskStatus.RECOVERY_REQUIRED},
            TaskStatus.PENDING,
        ),
    ),
)
def test_task_dependency_status_policy_truth_table(
    dependency_policy: str,
    dependency_states: Mapping[str, TaskStatus],
    expected: TaskStatus,
) -> None:
    node = TaskNode(
        "consumer",
        tuple(key for key in dependency_states if key != "ignored"),
        dependency_policy=dependency_policy,
    )

    assert node.dependency_status(dependency_states) is expected


def test_task_dependency_status_requires_declared_dependency_state() -> None:
    node = TaskNode("consumer", ("missing",), dependency_policy="all_terminal")

    with pytest.raises(KeyError):
        node.dependency_status({})


def test_task_graph_request_identity_includes_both_node_policies() -> None:
    principal = Principal("owner", "tenant")
    original = TaskGraph(
        "policy-identity",
        (
            TaskNode(
                "node",
                dependency_policy="all_terminal",
                failure_policy="propagate",
            ),
        ),
    )
    admission = TaskGraphAdmission.from_request(
        TaskGraphRequest(original, principal, "policy-identity-request-0001")
    )

    for changed in (
        TaskNode("node", dependency_policy="any_succeeded"),
        TaskNode("node", dependency_policy="all_terminal", failure_policy="isolate"),
    ):
        with pytest.raises(AIError) as raised:
            admission.validate_graph(TaskGraph("policy-identity", (changed,)))
        assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


@pytest.mark.parametrize(
    ("node_statuses", "expected"),
    (
        (
            ((TaskStatus.FAILED, "isolate"), (TaskStatus.BLOCKED, "isolate")),
            TaskStatus.SUCCEEDED,
        ),
        (
            ((TaskStatus.FAILED, "propagate"), (TaskStatus.BLOCKED, "isolate")),
            TaskStatus.FAILED,
        ),
        (
            ((TaskStatus.FAILED, "isolate"), (TaskStatus.BLOCKED, "propagate")),
            TaskStatus.BLOCKED,
        ),
        (
            ((TaskStatus.FAILED, "propagate"), (TaskStatus.CANCELLED, "isolate")),
            TaskStatus.FAILED,
        ),
        (
            ((TaskStatus.FAILED, "isolate"), (TaskStatus.CANCELLED, "isolate")),
            TaskStatus.CANCELLED,
        ),
    ),
)
def test_task_graph_aggregate_obeys_failure_and_cancel_priority(
    node_statuses: tuple[tuple[TaskStatus, str], ...],
    expected: TaskStatus,
) -> None:
    nodes = tuple(
        TaskNode(f"node-{index}", failure_policy=failure_policy)
        for index, (_status, failure_policy) in enumerate(node_statuses)
    )
    states = tuple(
        TaskNodeView(
            "aggregate-policy",
            f"node-{index}",
            (),
            status,
            None,
            0,
            None,
            None,
            ErrorCode.TASK_DEPENDENCY_FAILED.value
            if status is TaskStatus.BLOCKED
            else None,
            None,
        )
        for index, (status, _failure_policy) in enumerate(node_statuses)
    )

    assert _isolated_graph_status(states, nodes) is expected


def test_cancelled_graph_projection_rejects_nonterminal_nodes() -> None:
    nodes = (TaskNode("node"),)
    states = (
        TaskNodeView(
            "invalid-cancelled-graph",
            "node",
            (),
            TaskStatus.RUNNING,
            "worker",
            1,
            datetime.now(timezone.utc) + timedelta(seconds=30),
            None,
            None,
            None,
        ),
    )
    graph = TaskGraphView(
        "invalid-cancelled-graph",
        TaskStatus.CANCELLED,
        nodes,
    )

    with pytest.raises(AIError) as raised:
        _effective_graph_status(graph, states)
    assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


def test_cancelled_graph_projection_requires_cancelled_node() -> None:
    nodes = (
        TaskNode("failed"),
        TaskNode("blocked"),
    )
    states = (
        TaskNodeView(
            "invalid-cancelled-graph",
            "failed",
            (),
            TaskStatus.FAILED,
            None,
            0,
            None,
            None,
            ErrorCode.TASK_NODE_FAILED.value,
            "a" * 64,
        ),
        TaskNodeView(
            "invalid-cancelled-graph",
            "blocked",
            (),
            TaskStatus.BLOCKED,
            None,
            0,
            None,
            None,
            ErrorCode.TASK_DEPENDENCY_FAILED.value,
            None,
        ),
    )
    graph = TaskGraphView(
        "invalid-cancelled-graph",
        TaskStatus.CANCELLED,
        nodes,
    )

    with pytest.raises(AIError) as raised:
        _effective_graph_status(graph, states)
    assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


def test_whole_graph_cancel_overrides_terminal_node_aggregate() -> None:
    nodes = (
        TaskNode("failed", failure_policy="propagate"),
        TaskNode("cancelled"),
    )
    states = (
        TaskNodeView(
            "aggregate-cancel",
            "failed",
            (),
            TaskStatus.FAILED,
            None,
            0,
            None,
            None,
            ErrorCode.TASK_NODE_FAILED.value,
            "a" * 64,
        ),
        TaskNodeView(
            "aggregate-cancel",
            "cancelled",
            (),
            TaskStatus.CANCELLED,
            None,
            0,
            None,
            None,
            None,
            None,
        ),
    )
    graph = TaskGraphView("aggregate-cancel", TaskStatus.CANCELLED, nodes)

    assert _isolated_graph_status(states, nodes) is TaskStatus.FAILED
    assert _effective_graph_status(graph, states) is TaskStatus.CANCELLED
    recovering = (
        states[0],
        replace(
            states[1],
            status=TaskStatus.RECOVERY_REQUIRED,
            fence=1,
            error_code=ErrorCode.TASK_EFFECT_UNKNOWN.value,
            error_digest="b" * 64,
        ),
    )
    assert _effective_graph_status(graph, recovering) is TaskStatus.RECOVERY_REQUIRED


@pytest.mark.asyncio
async def test_all_terminal_tasks_run_after_failed_and_blocked_dependencies() -> None:
    observed: dict[str, TaskStatus] = {}

    async def fail(context: TaskNodeContext[None]) -> JsonValue:
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID)

    async def collect(context: TaskNodeContext[None]) -> JsonValue:
        observed.update(
            {key: value.status for key, value in context.dependency_states.items()}
        )
        for name in context.dependency_states:
            with pytest.raises(AIError) as raised:
                await context.read_dependency(name)
            assert raised.value.code is ErrorCode.TASK_DEPENDENCY_FAILED
        return "collected"

    group = RuntimeFeatures[None]("application")
    failure = group.task(TaskFunction("example.fail", 1, fail), effect_policy="none")
    collector = group.task(TaskFunction("example.collect", 1, collect), effect_policy="none")
    group.agent(
        "default", model="default", allow_tools=(), allow_skills=(), allow_subagents=()
    )
    async with Persistence.open('terminal-dependencies', storage=PersistenceStorage.in_memory()) as persistence_2450, Runtime.attach(persistence=persistence_2450, features=model_features(_TaskTestModels(), group)) as runtime, task_engine_scope(runtime, persistence_2450) as runtime_engine:
        run = await start_task_graph(runtime,
            TaskGraph(
                "graph",
                (
                    failure.node("failed"),
                    failure.node("blocked", dependencies=("failed",)),
                    collector.node(
                        "collect",
                        dependencies=("failed", "blocked"),
                        dependency_policy="all_terminal",
                    ),
                    agent_task_node(runtime,
                        "summary",
                        "Summarize upstream states",
                        dependencies=("failed", "blocked", "collect"),
                        dependency_policy="all_terminal",
                    ),
                ),
            ),
            idempotency_key="terminal-dependencies",
        )
        result = (await run.wait(timeout_seconds=10)).result
    assert observed == {"failed": TaskStatus.FAILED, "blocked": TaskStatus.BLOCKED}
    assert {node.node_id: node.status for node in result.node_states} == {
        "failed": TaskStatus.FAILED,
        "blocked": TaskStatus.BLOCKED,
        "collect": TaskStatus.SUCCEEDED,
        "summary": TaskStatus.SUCCEEDED,
    }


@pytest.mark.asyncio
async def test_any_succeeded_waits_for_all_terminal_dependencies_and_filters_results() -> None:
    held_started = asyncio.Event()
    release_held = asyncio.Event()
    success_returned = asyncio.Event()
    observed: dict[str, object] = {}
    blocked_calls = 0

    async def fail(context: TaskNodeContext[None]) -> JsonValue:
        if context.node_id == "held-failure":
            held_started.set()
            await release_held.wait()
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID)

    async def succeed(_context: TaskNodeContext[None]) -> JsonValue:
        success_returned.set()
        return "successful dependency"

    async def collect(context: TaskNodeContext[None]) -> JsonValue:
        observed["states"] = {
            node_id: state.status
            for node_id, state in context.dependency_states.items()
        }
        observed["results"] = set(context.dependencies)
        return "collected"

    async def forbidden(_context: TaskNodeContext[None]) -> JsonValue:
        nonlocal blocked_calls
        blocked_calls += 1
        return "must stay blocked"

    application = RuntimeFeatures[None]("application")
    failure = application.task(
        TaskFunction[None]("example.policy-failure", 1, fail),
        effect_policy="none",
    )
    success = application.task(
        TaskFunction[None]("example.policy-success", 1, succeed),
        effect_policy="none",
    )
    mixed = application.task(
        TaskFunction[None]("example.policy-mixed", 1, collect),
        effect_policy="none",
    )
    blocked = application.task(
        TaskFunction[None]("example.policy-blocked", 1, forbidden),
        effect_policy="none",
    )
    application.agent(
        "default", model="default", allow_tools=(), allow_skills=(), allow_subagents=()
    )
    graph = TaskGraph(
        "any-succeeded-runtime",
        (
            failure.node("held-failure", failure_policy="isolate"),
            failure.node("failed-fast", failure_policy="isolate"),
            success.node("success"),
            mixed.node(
                "mixed",
                dependencies=("held-failure", "success"),
                dependency_policy="any_succeeded",
            ),
            blocked.node(
                "blocked-any",
                dependencies=("held-failure", "failed-fast"),
                dependency_policy="any_succeeded",
                failure_policy="isolate",
            ),
        ),
    )

    async with Persistence.open('any-succeeded-runtime', storage=PersistenceStorage.in_memory()) as persistence_2557, Runtime.attach(persistence=persistence_2557, features=model_features(_TaskTestModels(), application)) as runtime, task_engine_scope(runtime, persistence_2557) as runtime_engine:
        run = await runtime_engine.with_definitions(failure, success, mixed, blocked).start(
            graph,
            idempotency_key="any-succeeded-runtime-0001",
         principal=runtime.default_principal)
        await asyncio.wait_for(held_started.wait(), 10)
        await asyncio.wait_for(success_returned.wait(), 10)
        deadline = asyncio.get_running_loop().time() + 10
        while True:
            state = await run.state(include_content=True)
            state_by_id = {node.node_id: node for node in state.node_states}
            if state_by_id["success"].status is TaskStatus.SUCCEEDED:
                break
            if asyncio.get_running_loop().time() >= deadline:
                raise AssertionError("success dependency did not become terminal")
            await asyncio.sleep(0.001)
        assert "mixed" not in observed
        assert blocked_calls == 0

        release_held.set()
        result = (await run.wait(timeout_seconds=10)).result
        final_state = await run.state(include_content=True)

    assert result.wait_status is TaskStatus.SUCCEEDED
    assert observed == {
        "states": {
            "held-failure": TaskStatus.FAILED,
            "success": TaskStatus.SUCCEEDED,
        },
        "results": {"success"},
    }
    assert blocked_calls == 0
    assert {
        node.node_id: node.status for node in result.node_states
    } == {
        "held-failure": TaskStatus.FAILED,
        "failed-fast": TaskStatus.FAILED,
        "success": TaskStatus.SUCCEEDED,
        "mixed": TaskStatus.SUCCEEDED,
        "blocked-any": TaskStatus.BLOCKED,
    }
    blocked_state = next(
        node for node in final_state.node_states if node.node_id == "blocked-any"
    )
    assert blocked_state.error_code == ErrorCode.TASK_DEPENDENCY_FAILED.value


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "filesystem", "sqlite"))
async def test_task_policy_matrix_persists_in_each_runtime_backend(
    backend: str,
    tmp_path: Path,
) -> None:
    called: list[str] = []

    async def succeed(context: TaskNodeContext[None]) -> JsonValue:
        called.append(context.node_id)
        return context.node_id

    application = RuntimeFeatures[None]("application")
    task = application.task(
        TaskFunction[None](f"example.persist-policy-{backend}", 1, succeed),
        effect_policy="none",
    )
    application.agent(
        "default", model="default", allow_tools=(), allow_skills=(), allow_subagents=()
    )
    if backend == "memory":
        storage = PersistenceStorage.in_memory()
        reopen_storage = None
    elif backend == "filesystem":
        storage_path = tmp_path / "runtime"
        storage = PersistenceStorage.filesystem(storage_path)
        reopen_storage = PersistenceStorage.filesystem(storage_path)
    else:
        storage_path = tmp_path / "runtime.sqlite"
        storage = PersistenceStorage.sqlite(storage_path)
        reopen_storage = PersistenceStorage.sqlite(storage_path)

    policies = tuple(
        (dependency_policy, failure_policy)
        for dependency_policy in ("all_succeeded", "all_terminal", "any_succeeded")
        for failure_policy in ("propagate", "isolate")
    )
    graph = TaskGraph(
        f"persist-policy-{backend}",
        tuple(
            task.node(
                f"{dependency_policy}-{failure_policy}",
                dependency_policy=dependency_policy,
                failure_policy=failure_policy,
            )
            for dependency_policy, failure_policy in policies
        ),
    )

    async with Persistence.open(f'persist-policy-{backend}', storage=storage) as persistence_2657, Runtime.attach(persistence=persistence_2657, features=model_features(_TaskTestModels(), application)) as runtime, task_engine_scope(runtime, persistence_2657) as runtime_engine:
        run = await runtime_engine.with_definitions(task).start(
            graph,
            idempotency_key=f"persist-policy-{backend}-0001",
         principal=runtime.default_principal)
        result = (await run.wait(timeout_seconds=10)).result
        state = await run.state()

    restored_state = None
    if reopen_storage is not None:
        async with Persistence.open(f'persist-policy-{backend}', storage=reopen_storage) as persistence_2671, Runtime.attach(persistence=persistence_2671, features=model_features(_TaskTestModels(), application)) as runtime, task_engine_scope(runtime, persistence_2671) as runtime_engine:
            restored_run = await runtime_engine.with_definitions(task).get(graph.graph_id, principal=runtime.default_principal)
            restored_result = (await restored_run.wait(timeout_seconds=10)).result
            restored_state = await restored_run.state()
        assert restored_result.wait_status is result.wait_status

    expected_policies = {
        f"{dependency_policy}-{failure_policy}": (
            dependency_policy,
            failure_policy,
        )
        for dependency_policy, failure_policy in policies
    }
    assert {
        node.node_id: (node.dependency_policy, node.failure_policy)
        for node in state.nodes
    } == expected_policies
    if restored_state is not None:
        assert {
            node.node_id: (node.dependency_policy, node.failure_policy)
            for node in restored_state.nodes
        } == expected_policies
        assert {node.node_id: node.status for node in restored_state.node_states} == {
            node.node_id: node.status for node in state.node_states
        }
    assert set(called) == {
        "all_succeeded-propagate",
        "all_succeeded-isolate",
        "all_terminal-propagate",
        "all_terminal-isolate",
    }
    statuses = {node.node_id: node.status for node in result.node_states}
    assert statuses["any_succeeded-propagate"] is TaskStatus.BLOCKED
    assert statuses["any_succeeded-isolate"] is TaskStatus.BLOCKED
    assert result.wait_status is TaskStatus.BLOCKED


class _TestTaskExpander:
    def __init__(self, expander_id: str, revision: int = 1) -> None:
        self.id = expander_id
        self.revision = revision

    def expand(self, context: object) -> tuple[TaskNode, ...]:
        del context
        return ()


@pytest.mark.asyncio
async def test_expansion_applies_dependency_policies_to_each_new_node() -> None:
    ran: list[str] = []
    observed_dependencies: dict[str, tuple[dict[str, TaskStatus], set[str]]] = {}

    async def succeed(context: TaskNodeContext[None]) -> JsonValue:
        ran.append(context.node_id)
        observed_dependencies[context.node_id] = (
            {
                node_id: state.status
                for node_id, state in context.dependency_states.items()
            },
            set(context.dependencies),
        )
        return context.node_id

    async def fail(_context: TaskNodeContext[None]) -> JsonValue:
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID)

    handler = TaskFunction[None](
        "example.expansion-policy-handler",
        1,
        succeed,
        effect_policy="none",
    )
    failure = TaskFunction[None](
        "example.expansion-policy-failure",
        1,
        fail,
        effect_policy="none",
    )
    expanded_names = (
        "empty-any",
        "empty-terminal",
        "on-source",
        "on-mixed",
        "all-succeeded-success",
        "all-succeeded-mixed",
        "terminal-after-failures",
        "all-failed",
        "source-excluded",
        "nested-any",
    )

    def expand(_context: TaskExpansionContext) -> tuple[TaskNode, ...]:
        return (
            handler.node(
                "empty-any",
                dependency_policy="any_succeeded",
                failure_policy="isolate",
            ),
            handler.node(
                "empty-terminal",
                dependency_policy="all_terminal",
                failure_policy="isolate",
            ),
            handler.node(
                "on-source",
                dependencies=("expansion-source",),
                dependency_policy="any_succeeded",
                failure_policy="isolate",
            ),
            handler.node(
                "on-mixed",
                dependencies=("expansion-source", "failed-root"),
                dependency_policy="any_succeeded",
                failure_policy="isolate",
            ),
            handler.node(
                "all-succeeded-success",
                dependencies=("expansion-source", "success-root"),
                dependency_policy="all_succeeded",
                failure_policy="isolate",
            ),
            handler.node(
                "all-succeeded-mixed",
                dependencies=("success-root", "failed-root"),
                dependency_policy="all_succeeded",
                failure_policy="isolate",
            ),
            handler.node(
                "terminal-after-failures",
                dependencies=("failed-root", "dependency-blocked"),
                dependency_policy="all_terminal",
                failure_policy="isolate",
            ),
            handler.node(
                "all-failed",
                dependencies=("failed-root", "dependency-blocked"),
                dependency_policy="any_succeeded",
                failure_policy="isolate",
            ),
            handler.node(
                "source-excluded",
                dependencies=("failed-root",),
                dependency_policy="any_succeeded",
                failure_policy="isolate",
            ),
            handler.node(
                "nested-any",
                dependencies=("on-mixed", "all-failed"),
                dependency_policy="any_succeeded",
                failure_policy="isolate",
            ),
        )

    expander = TaskExpander("example.expansion-policy", expand)
    application = RuntimeFeatures[None]("application")
    application.task(handler)
    application.task(failure)
    application.task_expander(expander)
    application.agent(
        "default", model="default", allow_tools=(), allow_skills=(), allow_subagents=()
    )
    graph = TaskGraph(
        "expansion-dependency-policies",
        (
            handler.node("expansion-source", expander=expander.ref),
            handler.node("success-root"),
            failure.node("failed-root", failure_policy="isolate"),
            handler.node(
                "dependency-blocked",
                dependencies=("failed-root",),
                failure_policy="isolate",
            ),
        ),
    )

    async with Persistence.open('expansion-dependency-policies', storage=PersistenceStorage.in_memory()) as persistence_2850, Runtime.attach(persistence=persistence_2850, features=model_features(_TaskTestModels(), application)) as runtime, task_engine_scope(runtime, persistence_2850) as runtime_engine:
        run = await runtime_engine.with_definitions(handler, failure, expander).start(
            graph,
            idempotency_key="expansion-dependency-policies-0001",
         principal=runtime.default_principal)
        result = (await run.wait(timeout_seconds=10)).result
        state = await run.state()

    assert result.wait_status is TaskStatus.SUCCEEDED
    assert set(ran) == {
        "expansion-source",
        "success-root",
        "empty-terminal",
        "on-source",
        "on-mixed",
        "all-succeeded-success",
        "terminal-after-failures",
        "nested-any",
    }
    result_statuses = {node.node_id: node.status for node in result.node_states}
    assert result_statuses["on-source"] is TaskStatus.SUCCEEDED
    assert result_statuses["on-mixed"] is TaskStatus.SUCCEEDED
    assert result_statuses["empty-terminal"] is TaskStatus.SUCCEEDED
    assert result_statuses["all-succeeded-success"] is TaskStatus.SUCCEEDED
    assert result_statuses["all-succeeded-mixed"] is TaskStatus.BLOCKED
    assert result_statuses["terminal-after-failures"] is TaskStatus.SUCCEEDED
    assert result_statuses["all-failed"] is TaskStatus.BLOCKED
    assert result_statuses["source-excluded"] is TaskStatus.BLOCKED
    assert result_statuses["empty-any"] is TaskStatus.BLOCKED
    assert result_statuses["nested-any"] is TaskStatus.SUCCEEDED
    assert observed_dependencies["all-succeeded-success"] == (
        {
            "expansion-source": TaskStatus.SUCCEEDED,
            "success-root": TaskStatus.SUCCEEDED,
        },
        {"expansion-source", "success-root"},
    )
    assert observed_dependencies["terminal-after-failures"] == (
        {
            "dependency-blocked": TaskStatus.BLOCKED,
            "failed-root": TaskStatus.FAILED,
        },
        set(),
    )
    node_infos = {node.node_id: node for node in state.nodes}
    assert set(expanded_names) <= set(node_infos)
    assert {
        node_id: (node_infos[node_id].dependency_policy, node_infos[node_id].failure_policy)
        for node_id in expanded_names
    } == {
        "empty-any": ("any_succeeded", "isolate"),
        "on-source": ("any_succeeded", "isolate"),
        "on-mixed": ("any_succeeded", "isolate"),
        "all-succeeded-success": ("all_succeeded", "isolate"),
        "all-succeeded-mixed": ("all_succeeded", "isolate"),
        "terminal-after-failures": ("all_terminal", "isolate"),
        "all-failed": ("any_succeeded", "isolate"),
        "source-excluded": ("any_succeeded", "isolate"),
        "nested-any": ("any_succeeded", "isolate"),
        "empty-terminal": ("all_terminal", "isolate"),
    }


@pytest.mark.asyncio
async def test_expanded_join_failure_propagates_after_isolated_candidate_failure() -> None:
    ran: list[str] = []
    join_dependencies: dict[str, TaskStatus] = {}

    async def succeed(_context: TaskNodeContext[None]) -> JsonValue:
        return "source"

    async def fail(context: TaskNodeContext[None]) -> JsonValue:
        ran.append(context.node_id)
        if context.node_id == "join":
            join_dependencies.update(
                {
                    node_id: state.status
                    for node_id, state in context.dependency_states.items()
                }
            )
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID)

    application = RuntimeFeatures[None]("application")
    source_task = TaskFunction[None](
        "example.expanded-join-source", 1, succeed, effect_policy="none"
    )
    candidate_task = TaskFunction[None](
        "example.expanded-join-candidate", 1, fail, effect_policy="none"
    )
    application.task(source_task)
    application.task(candidate_task)

    def expand(_context: TaskExpansionContext) -> tuple[TaskNode, ...]:
        return (
            candidate_task.node(
                "join",
                dependencies=("candidate",),
                dependency_policy="all_terminal",
            ),
        )

    expander = TaskExpander("example.expanded-join-failure", expand)
    application.task_expander(expander)
    graph = TaskGraph(
        "expanded-join-failure",
        (
            source_task.node("source", expander=expander.ref),
            candidate_task.node("candidate", failure_policy="isolate"),
        ),
    )

    async with Persistence.open('expanded-join-failure', storage=PersistenceStorage.in_memory()) as persistence_2965, Runtime.attach(persistence=persistence_2965, features=model_features(_TaskTestModels(), application)) as runtime, task_engine_scope(runtime, persistence_2965) as runtime_engine:
        run = await runtime_engine.with_definitions(source_task, candidate_task, expander).start(
            graph,
            idempotency_key="expanded-join-failure-0001",
         principal=runtime.default_principal)
        result = (await run.wait(timeout_seconds=10)).result

    assert result.wait_status is TaskStatus.FAILED
    assert set(ran) == {"candidate", "join"}
    assert join_dependencies == {"candidate": TaskStatus.FAILED}
    assert {
        node.node_id: node.status for node in result.node_states
    } == {
        "source": TaskStatus.SUCCEEDED,
        "candidate": TaskStatus.FAILED,
        "join": TaskStatus.FAILED,
    }


@pytest.mark.asyncio
async def test_nested_expander_source_does_not_implicitly_wait_for_its_child() -> None:
    grandchild_started = asyncio.Event()
    release_grandchild = asyncio.Event()

    async def run(context: TaskNodeContext[None]) -> JsonValue:
        if context.node_id == "grandchild":
            grandchild_started.set()
            await release_grandchild.wait()
        return context.node_id

    application = RuntimeFeatures[None]("application")
    task = TaskFunction[None](
        "example.nested-expansion-boundary",
        1,
        run,
        effect_policy="none",
    )
    application.task(task)
    reference = TaskExpanderRef("example.nested-expansion-boundary", 1)

    def expand(context: TaskExpansionContext) -> tuple[TaskNode, ...]:
        if context.source_node.node_id == "root":
            return (
                task.node(
                    "branch",
                    dependencies=("root",),
                    expander=reference,
                ),
                task.node(
                    "join",
                    dependencies=("branch",),
                    dependency_policy="all_succeeded",
                ),
            )
        if context.source_node.node_id == "branch":
            return (task.node("grandchild", dependencies=("branch",)),)
        return ()

    expander = TaskExpander("example.nested-expansion-boundary", expand)
    application.task_expander(expander)
    graph = TaskGraph(
        "nested-expansion-boundary",
        (task.node("root", expander=expander.ref),),
    )

    async with Persistence.open('nested-expansion-boundary', storage=PersistenceStorage.in_memory()) as persistence_3034, Runtime.attach(persistence=persistence_3034, features=model_features(_TaskTestModels(), application)) as runtime, task_engine_scope(runtime, persistence_3034) as runtime_engine:
        run_handle = await runtime_engine.with_definitions(task, expander).start(
            graph,
            idempotency_key="nested-expansion-boundary-0001",
         principal=runtime.default_principal)
        try:
            await asyncio.wait_for(grandchild_started.wait(), 10)
            deadline = asyncio.get_running_loop().time() + 10
            while True:
                state = await run_handle.state()
                states = {node.node_id: node.status for node in state.node_states}
                if states.get("join") is TaskStatus.SUCCEEDED:
                    break
                if asyncio.get_running_loop().time() >= deadline:
                    raise AssertionError("join did not follow its declared source")
                await asyncio.sleep(0.001)
            assert states["grandchild"] is TaskStatus.RUNNING
            assert state.status is TaskStatus.RUNNING
        finally:
            release_grandchild.set()

        result = (await run_handle.wait(timeout_seconds=10)).result

    assert result.wait_status is TaskStatus.SUCCEEDED
    assert {
        node.node_id: node.status for node in result.node_states
    } == {
        "root": TaskStatus.SUCCEEDED,
        "branch": TaskStatus.SUCCEEDED,
        "join": TaskStatus.SUCCEEDED,
        "grandchild": TaskStatus.SUCCEEDED,
    }


class _ApplicationGraphExpander:
    id = "application.graph"
    revision = 1

    def __init__(self, handler: TaskFunction[None]) -> None:
        self._handler = handler

    def expand(self, context: TaskExpansionContext) -> tuple[TaskNode, ...]:
        reference = TaskExpanderRef(self.id, self.revision)
        source_id = context.source_node.node_id
        if source_id == "application-root":
            return (
                self._handler.node(
                    "child-b",
                    input={"value": "b"},
                    dependencies=("child-a",),
                ),
                self._handler.node(
                    "disconnected",
                    input={"value": "disconnected"},
                ),
                self._handler.node(
                    "child-a",
                    input={"value": "a"},
                    expander=reference,
                ),
            )
        if source_id == "child-a":
            return (
                self._handler.node(
                    "grandchild",
                    input={"value": "grandchild"},
                    dependencies=("child-a",),
                ),
            )
        return ()


class _AgentGraphExpander:
    id = "application.agent-graph"
    revision = 1

    def __init__(self, task: Task[None]) -> None:
        self._task = task

    def expand(self, context: TaskExpansionContext) -> tuple[TaskNode, ...]:
        del context
        return (
            TaskNode(
                "agent-child",
                task=self._task,
                input=AgentTaskInput(
                    "return a child result",
                    files=("context.txt",),
                    session_id="expander-session",
                ),
                timeout_seconds=12,
                max_attempts=2,
                retry_delay_seconds=0.25,
            ),
        )


@pytest.mark.asyncio
async def test_task_handler_revisions_are_exact_and_reserved_namespace_is_closed() -> (
    None
):
    v1 = TaskFunction[None]("example.echo", 1, _echo_task)
    v2 = TaskFunction[None]("example.echo", 2, _echo_task)
    async with Persistence.open('task-revision-view', storage=PersistenceStorage.in_memory()) as persistence_3142, Runtime.attach(persistence=persistence_3142, features=model_features(_TaskTestModels())) as runtime, task_engine_scope(runtime, persistence_3142) as runtime_engine:
        view = runtime_engine.with_definitions(v1, v2)
        assert view is not None
        with pytest.raises(AIError) as duplicate:
            runtime_engine.with_definitions(v1, TaskFunction[None]("example.echo", 1, _echo_task))
        assert duplicate.value.code is ErrorCode.BINDING_CONFLICT
    with pytest.raises(ValueError):
        Task("linktools.ai.custom", _echo_task)

    assert TaskExpander("application.expand", _TestTaskExpander("application.expand").expand)
    with pytest.raises(ValueError):
        TaskExpander("linktools.ai.expand", _TestTaskExpander("linktools.ai.expand").expand)


@pytest.mark.asyncio
async def test_graph_nodes_store_only_the_exact_task_reference() -> None:
    handler = TaskFunction[None]("example.mutable-handler", 1, _echo_task)
    node = handler.node("node")
    assert node.task == handler.ref
    async with Persistence.open('frozen-task-identity', storage=PersistenceStorage.in_memory()) as persistence_3165, Runtime.attach(persistence=persistence_3165, features=model_features(_TaskTestModels())) as runtime, task_engine_scope(runtime, persistence_3165) as runtime_engine:
        run = await start_task_graph(
            runtime,
            TaskGraph("frozen-task-identity", (node,)),
            idempotency_key="frozen-task-identity-0001",
        )
        result = (await run.wait(timeout_seconds=10)).result
        assert await run.result("node") == {"value": None}

    assert result.wait_status is TaskStatus.SUCCEEDED


@pytest.mark.asyncio
async def test_runtime_accepts_multiple_task_handler_revisions() -> None:
    group = RuntimeFeatures[None]("application")
    v1 = TaskFunction[None]("example.runtime-revision", 1, _echo_task)
    v2 = TaskFunction[None]("example.runtime-revision", 2, _echo_task)
    group.task(v1, effect_policy="none")
    group.task(v2, effect_policy="none")
    group.task_expander(_TestTaskExpander("example.runtime-expand", 1))
    group.task_expander(_TestTaskExpander("example.runtime-expand", 2))
    state = PersistenceStorage.in_memory()

    async with Persistence.open('task-revisions', storage=state) as persistence_3192, Runtime.attach(persistence=persistence_3192, features=model_features(_TaskTestModels(), group)) as runtime, task_engine_scope(runtime, persistence_3192) as runtime_engine:
        run = await start_task_graph(runtime,
            TaskGraph(
                "task-revisions",
                (v1.node("v1"), v2.node("v2")),
            ),
            idempotency_key="task-revisions-run-0001",
        )
        result = (await run.wait(timeout_seconds=10)).result

    assert result.wait_status is TaskStatus.SUCCEEDED
    assert {item.node_id for item in result.node_states} == {"v1", "v2"}


@pytest.mark.asyncio
async def test_task_result_commit_preserves_early_execution_binding() -> None:
    state = PersistenceStorage.in_memory()
    state = Persistence(state)
    await state.initialize(namespace="task-result-regression", tenant_id="tenant")
    try:
        repository = state.task.tasks
        graph = TaskGraph("result-graph", (TaskNode("node"),))
        await admit_graph(state, graph)
        lease = await repository.claim(
            graph.graph_id,
            "node",
            tenant_id="tenant",
            owner="worker",
            lease_seconds=30,
        )
        await repository.handoff_execution(
            lease,
            tenant_id="tenant",
            execution_id="execution",
        )
        payload = StoredPayload.inline_json({"ok": True})

        terminal = await repository.complete(
            None,
            tenant_id="tenant",
            execution_id="execution",
            result_digest=payload.digest,
            graph_id=graph.graph_id,
            node_id="node",
        )

        assert terminal.execution_id == "execution"
        snapshot = await repository.graph_state(graph.graph_id, tenant_id="tenant")
        assert snapshot is not None
        assert snapshot.status is TaskStatus.SUCCEEDED
        assert snapshot.node_states[0].execution_id == "execution"
        assert snapshot.node_states[0].result_digest == payload.digest
        results = await repository.get_results(
            graph.graph_id,
            ("node",),
            tenant_id="tenant",
        )
        assert results["node"].execution_id == "execution"
    finally:
        await state.close()


@pytest.mark.asyncio
async def test_runtime_executes_custom_agent_custom_graph_and_persists_each_result(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    workspace = Workspace.load(workspace_root)
    application = workspace_features("application", workspace=workspace, features_type=RuntimeFeatures)
    handler = TaskFunction[None]("example.echo", 1, _echo_task)
    application.task(handler, effect_policy="none")
    application.agent(
        "default",
        model="default",
        allow_tools=(),
        allow_skills=(),
        allow_subagents=(),
    )
    state = PersistenceStorage.in_memory()

    async with Persistence.open('default', storage=state) as persistence_3277, Runtime.attach(persistence=persistence_3277, features=model_features(_TaskTestModels(), application)) as runtime, task_engine_scope(runtime, persistence_3277) as runtime_engine:
        first = handler.node("custom-first", input={"value": "seed"})
        agent = agent_task_node(runtime,
            "agent",
            "Return a short test response.",
            dependencies=("custom-first",),
        )
        last = handler.node("custom-last", dependencies=("agent",))
        graph = TaskGraph("mixed-graph", (first, agent, last))

        result = await run_task_graph(runtime,
            graph,
            idempotency_key="mixed-graph-run-0001",
            timeout_seconds=10,
        )

        assert result.wait_status is TaskStatus.SUCCEEDED
        assert all(node.status is TaskStatus.SUCCEEDED for node in result.node_states)
        first_output = await task_result(runtime, graph.graph_id, "custom-first")
        agent_output = await task_result(runtime, graph.graph_id, "agent")
        last_output = await task_result(runtime, graph.graph_id, "custom-last")
        assert first_output == {"value": "seed"}
        assert isinstance(last_output, dict)
        assert last_output["upstream"] == agent_output
        assert isinstance(last_output["execution_id"], str)
        persisted = await runtime._state.task.tasks.get_results(
            graph.graph_id,
            ("custom-first", "agent", "custom-last"),
            tenant_id="default",
        )
        assert set(persisted) == {"custom-first", "agent", "custom-last"}


@pytest.mark.asyncio
async def test_projected_agent_input_persists_only_declared_source_and_final_input() -> None:
    application = RuntimeFeatures[None]("application")
    source = TaskFunction[None]("example.projected-source", 1, _echo_task)
    application.task(source, effect_policy="none")
    application.agent(
        "default",
        model="default",
        allow_tools=(),
        allow_skills=(),
        allow_subagents=(),
    )
    calls: list[str] = []

    async def build_input(context: AgentTaskInputContext) -> str:
        assert context.input == {"request": "review"}
        calls.append(context.node_id)
        value = await context.result("source")
        assert await context.result("source") == value
        return f"{context.prompt}: {value['value']}"

    state = PersistenceStorage.in_memory()
    async with Persistence.open('projected-input', storage=state) as persistence_3336, Runtime.attach(persistence=persistence_3336, features=model_features(_TaskTestModels(), application)) as runtime, task_engine_scope(runtime, persistence_3336) as runtime_engine:
        agent_task = Task.from_agent(
            "example.projected-agent",
            runtime.agents.get("default"),
            build_input=build_input,
        )
        graph = TaskGraph(
            "projected-input-graph",
            (
                source.node("source", input={"value": "accepted"}),
                TaskNode(
                    "consumer",
                    ("source",),
                    task=agent_task,
                    input=AgentTaskInput(
                        "base prompt",
                        parameters={"request": "review"},
                        thinking=False,
                    ),
                ),
            ),
        )
        run = await runtime_engine.with_definitions(source, agent_task).start(
            graph,
            idempotency_key="projected-input-run-0001",
         principal=runtime.default_principal)
        result = (await run.wait(timeout_seconds=10)).result

        assert result.wait_status is TaskStatus.SUCCEEDED, result.node_states
        assert calls == ["consumer"]
        prepared = await runtime._state.task.tasks.get_prepared_input(
            graph.graph_id,
            "consumer",
            tenant_id=runtime.tenant_id,
        )
        assert prepared is not None
        assert len(prepared.source_refs) == 1
        source_name, source_ref = prepared.source_refs[0]
        assert source_name == "source"
        source_record = await runtime._state.task.tasks.get_results(
            graph.graph_id,
            ("source",),
            tenant_id=runtime.tenant_id,
        )
        assert source_ref.result_digest == source_record["source"].result_digest
        consumer_result = next(
            node for node in result.node_states if node.node_id == "consumer"
        )
        execution = await runtime._state.execution.executions.get(
            consumer_result.execution_id,
            tenant_id=runtime.tenant_id,
        )
        persisted_graph = await runtime._state.task.tasks.graph_state(
            graph.graph_id,
            tenant_id=runtime.tenant_id,
        )
        assert persisted_graph is not None
        persisted_consumer = next(
            node for node in persisted_graph.nodes if node.node_id == "consumer"
        )
        assert persisted_consumer.input["thinking"] is False
        assert execution.thinking is False
        assert execution.stored_user_input is not None
        assert execution.stored_user_input.codec == "text"
        assert execution.stored_user_input.payload.decode() == "base prompt: accepted"


@pytest.mark.asyncio
async def test_invalid_graph_request_does_not_reserve_task_definitions() -> None:
    async def run_first(context: TaskNodeContext[None]) -> JsonValue:
        del context
        return {"owner": "first"}

    async def run_second(context: TaskNodeContext[None]) -> JsonValue:
        del context
        return {"owner": "second"}

    first = Task("example.admission-owner", run_first, effect_policy="none")
    second = Task(
        "example.admission-owner",
        run_second,
        effect_policy="replay_safe",
    )
    async with Persistence.open('task-admission-owner', storage=PersistenceStorage.in_memory()) as persistence_3423, Runtime.attach(persistence=persistence_3423, features=model_features(_TaskTestModels())) as runtime, task_engine_scope(runtime, persistence_3423) as runtime_engine:
        invalid_graph = TaskGraph(
            "task-admission-owner-graph",
            (TaskNode("node", task=first),),
        )
        with pytest.raises(AIError) as invalid_request:
            await runtime_engine.with_definitions(first).start(
                invalid_graph,
                idempotency_key="",
             principal=runtime.default_principal)
        assert invalid_request.value.code is ErrorCode.IDEMPOTENCY_KEY_INVALID

        valid_graph = TaskGraph(
            invalid_graph.graph_id,
            (TaskNode("node", task=second),),
        )
        result = await runtime_engine.with_definitions(second).start(
            valid_graph,
            idempotency_key="task-admission-owner-valid-0001",
         principal=runtime.default_principal)
        completed = (await result.wait(timeout_seconds=10)).result

    assert completed.wait_status is TaskStatus.SUCCEEDED
    assert completed.node_states[0].status is TaskStatus.SUCCEEDED


@pytest.mark.asyncio
async def test_duplicate_graph_start_keeps_the_original_task_definition_owner() -> None:
    entered = asyncio.Event()
    release = asyncio.Event()
    calls: list[str] = []

    async def run_first(context: TaskNodeContext[None]) -> JsonValue:
        del context
        calls.append("first")
        entered.set()
        await release.wait()
        return {"owner": "first"}

    async def run_second(context: TaskNodeContext[None]) -> JsonValue:
        del context
        calls.append("second")
        return {"owner": "second"}

    first = Task("example.concurrent-owner", run_first, effect_policy="none")
    second = Task("example.concurrent-owner", run_second, effect_policy="none")
    graph = TaskGraph(
        "concurrent-owner-graph",
        (TaskNode("node", task=first),),
    )
    async with Persistence.open('concurrent-task-owner', storage=PersistenceStorage.in_memory()) as persistence_3477, Runtime.attach(persistence=persistence_3477, features=model_features(_TaskTestModels())) as runtime, task_engine_scope(runtime, persistence_3477) as runtime_engine:
        first_run = await runtime_engine.with_definitions(first).start(
            graph,
            idempotency_key="concurrent-owner-graph-0001",
         principal=runtime.default_principal)
        await asyncio.wait_for(entered.wait(), 10)
        repeated = await runtime_engine.with_definitions(second).start(
            graph,
            idempotency_key="concurrent-owner-graph-0001",
         principal=runtime.default_principal)
        release.set()
        first_result = (await first_run.wait(timeout_seconds=10)).result
        repeated_result = (await repeated.wait(timeout_seconds=10)).result

    assert first_result.wait_status is TaskStatus.SUCCEEDED
    assert repeated_result.wait_status is TaskStatus.SUCCEEDED
    assert calls == ["first"]


@pytest.mark.asyncio
async def test_projected_agent_input_reads_json_null_dependency() -> None:
    async def return_null(_context: TaskNodeContext[None]) -> JsonValue:
        return None

    source = Task("example.projected-null-source", return_null, effect_policy="none")
    application = RuntimeFeatures[None]("projected-null-agent")
    application.agent(
        "default",
        model="default",
        allow_tools=(),
        allow_skills=(),
        allow_subagents=(),
    )
    projected_values: list[JsonValue] = []

    async def build_input(context: AgentTaskInputContext) -> str:
        value = await context.result("source")
        projected_values.append(value)
        return "source is null"

    state = PersistenceStorage.in_memory()
    async with Persistence.open('projected-null-input', storage=state) as persistence_3522, Runtime.attach(persistence=persistence_3522, features=model_features(_TaskTestModels(), application)) as runtime, task_engine_scope(runtime, persistence_3522) as runtime_engine:
        consumer = Task.from_agent(
            "example.projected-null-consumer",
            runtime.agents.get("default"),
            build_input=build_input,
        )
        graph = TaskGraph(
            "projected-null-input-graph",
            (
                TaskNode("source", task=source),
                TaskNode(
                    "consumer",
                    ("source",),
                    task=consumer,
                    input=AgentTaskInput("base", parameters={"mode": "null"}),
                ),
            ),
        )
        run = await runtime_engine.with_definitions(source, consumer).start(
            graph,
            idempotency_key="projected-null-input-run-0001",
         principal=runtime.default_principal)
        completed = (await run.wait(timeout_seconds=10)).result

        assert completed.wait_status is TaskStatus.SUCCEEDED
        assert await run.result("source") is None

    assert projected_values == [None]


@pytest.mark.asyncio
async def test_runner_task_contract_is_persisted_and_validated_after_reopen(
    tmp_path: Path,
) -> None:
    storage_root = tmp_path / "runner-contract-state"
    schema = _EffectOutput.model_json_schema()
    contract: dict[str, JsonValue] = {
        "version": 1,
        "type": "example.runner",
        "effect_policy": "none",
        "output_contract": {"kind": "schema", "schema": schema},
        "reconcile": False,
    }
    expanded_outputs: list[JsonValue] = []

    def expand(context: TaskExpansionContext) -> tuple[TaskNode, ...]:
        expanded_outputs.append(context.output)
        return ()

    expander = TaskExpander("example.contract-runner.expander", expand)
    state = PersistenceStorage.filesystem(storage_root)
    async with Persistence.open('runner-contract', storage=state) as persistence_3577, Runtime.attach(persistence=persistence_3577, features=model_features(_TaskTestModels())) as runtime, task_engine_scope(runtime, persistence_3577) as runtime_engine:
        runner = _ExecutionBackedContractRunner(runtime_engine._execution_service)
        task = Task.from_runner(
            "example.contract-runner",
            runner,
            contract=contract,
        )
        graph = TaskGraph(
            "runner-contract-graph",
            (
                TaskNode.wait("input"),
                TaskNode(
                    "runner",
                    ("input",),
                    task=task,
                    expander=expander,
                ),
            ),
        )
        await runtime_engine.with_definitions(task, expander).start(
            graph,
            idempotency_key="runner-contract-graph-0001",
         principal=runtime.default_principal)
        wait_id = await _wait_for_input_execution(runtime, graph.graph_id)
        deferred_execution = await runtime_engine._execution_service.inspect(
            wait_id,
            principal=runtime.default_principal,
        )
        initial = await runtime._state.task.tasks.graph_state(
            graph.graph_id,
            tenant_id=runtime.tenant_id,
        )
        assert initial is not None
        input_state = next(
            item for item in initial.node_states if item.node_id == "input"
        )
        assert deferred_execution.status.value == "WAITING_DEFERRED"
        assert input_state.status is TaskStatus.WAITING
        assert input_state.execution_id == wait_id
        runner_node = next(node for node in initial.nodes if node.node_id == "runner")
        assert runner_node.output_contract == {
            "mode": "structured",
            "schema": schema,
        }
        assert runner_node.reconcile is False
        assert runner.calls == 0

    recovered_storage = PersistenceStorage.filesystem(storage_root)
    async with Persistence.open('runner-contract', storage=recovered_storage) as persistence_3629, Runtime.attach(persistence=persistence_3629, features=model_features(_TaskTestModels())) as runtime, task_engine_scope(runtime, persistence_3629) as runtime_engine:
        runner = _ExecutionBackedContractRunner(runtime_engine._execution_service)
        task = Task.from_runner(
            "example.contract-runner",
            runner,
            contract=contract,
        )
        engine = runtime_engine.with_definitions(task, expander)
        await engine.recover_pending(principal=runtime.default_principal)
        resumed = await engine.get(graph.graph_id, principal=runtime.default_principal)
        reopened_input_state = next(
            item
            for item in (await resumed.state(include_content=True)).node_states
            if item.node_id == "input"
        )
        reopened_execution = await runtime_engine._execution_service.inspect(
            wait_id,
            principal=runtime.default_principal,
        )
        assert reopened_execution.status.value == "WAITING_DEFERRED"
        assert reopened_input_state.status is TaskStatus.WAITING
        assert reopened_input_state.execution_id == wait_id
        await resumed.resume(
            "input",
            TaskInputSupplyRequest(
                runtime.default_principal,
                wait_id,
                {"value": "input"},
                "runner-contract-input-0001",
            ),
        )
        completed = (await resumed.wait(timeout_seconds=10)).result
        assert await resumed.result("runner") == {"value": "accepted"}
        result_ref = await resumed.result_ref("runner")
        assert result_ref.result_digest == canonical_sha256({"value": "accepted"})

        recovered_state = await runtime._state.task.tasks.graph_state(
            graph.graph_id,
            tenant_id=runtime.tenant_id,
        )
        assert recovered_state is not None
        runner_node = next(
            node for node in recovered_state.nodes if node.node_id == "runner"
        )
        assert runner_node.output_contract == {
            "mode": "structured",
            "schema": schema,
        }
        assert runner_node.reconcile is False
        assert runner.calls == 1

    assert completed.wait_status is TaskStatus.SUCCEEDED
    assert expanded_outputs == [{"value": "accepted"}]


@pytest.mark.asyncio
async def test_projected_workers_issue_distinct_requests_and_persist_history(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    attachment = b"PROJECTED_ATTACHMENT_PAYLOAD"
    (workspace_root / "brief.txt").write_bytes(attachment)
    workspace = Workspace.load(workspace_root)
    application = workspace_features("application", workspace=workspace, features_type=RuntimeFeatures)
    application.agent(
        "default",
        model="default",
        allow_tools=(),
        allow_skills=(),
        allow_subagents=(),
    )
    models = _TaskTestModels()
    state = PersistenceStorage.in_memory()

    async def read_source(context: TaskNodeContext[None]) -> JsonValue:
        del context
        return {
            "customer": "Harbor-ALPHA",
            "queue": "Queue-BETA",
            "unprojected": "SECRET-UNPROJECTED-SOURCE-FIELD",
        }

    source = Task("example.dual-worker-source", read_source, effect_policy="none")

    async def build_input(context: AgentTaskInputContext) -> str:
        value = await context.result("source")
        assert isinstance(value, dict)
        if context.input["worker"] == "left":
            return f"left projection customer={value['customer']}"
        return f"right projection queue={value['queue']}"

    async with Persistence.open('dual-projected-workers', storage=state) as persistence_3725, Runtime.attach(persistence=persistence_3725, features=model_features(models, application)) as runtime, task_engine_scope(runtime, persistence_3725) as runtime_engine:
        left = Task.from_agent(
            "example.dual-worker-left",
            runtime.agents.get("default"),
            build_input=build_input,
        )
        right = Task.from_agent(
            "example.dual-worker-right",
            runtime.agents.get("default"),
            build_input=build_input,
        )
        graph = TaskGraph(
            "dual-projected-workers-graph",
            (
                TaskNode("source", task=source),
                TaskNode(
                    "left",
                    ("source",),
                    task=left,
                    input=AgentTaskInput(
                        "left request",
                        parameters={"worker": "left"},
                        files=("brief.txt",),
                    ),
                ),
                TaskNode(
                    "right",
                    ("source",),
                    task=right,
                    input=AgentTaskInput(
                        "right request",
                        parameters={"worker": "right"},
                    ),
                ),
            ),
        )
        run = await runtime_engine.with_definitions(source, left, right).start(
            graph,
            idempotency_key="dual-projected-workers-run-0001",
         principal=runtime.default_principal)
        result = (await run.wait(timeout_seconds=10)).result

        assert result.wait_status is TaskStatus.SUCCEEDED, result.node_states

        source_record = (
            await runtime._state.task.tasks.get_results(
                graph.graph_id,
                ("source",),
                tenant_id=runtime.tenant_id,
            )
        )["source"]
        node_results = {node.node_id: node for node in result.node_states}
        for node_id, marker in (
            ("left", "Harbor-ALPHA"),
            ("right", "Queue-BETA"),
        ):
            prepared = await runtime._state.task.tasks.get_prepared_input(
                graph.graph_id,
                node_id,
                tenant_id=runtime.tenant_id,
            )
            assert prepared is not None
            assert len(prepared.source_refs) == 1
            source_name, source_ref = prepared.source_refs[0]
            assert source_name == "source"
            assert source_ref.result_digest == source_record.result_digest

            execution_id = node_results[node_id].execution_id
            assert execution_id is not None
            interactions = await runtime.history.model_interactions(
                execution_id,
                principal=runtime.default_principal,
                include_content=True,
                limit=100,
            )
            assert interactions.next_cursor is None
            assert len(interactions.items) == 1
            interaction = interactions.items[0]
            assert interaction.status == "SUCCEEDED"
            assert interaction.content_included is True
            assert marker in str(interaction.request)
            other_marker = "Queue-BETA" if node_id == "left" else "Harbor-ALPHA"
            assert other_marker not in str(interaction.request)
            assert "SECRET-UNPROJECTED-SOURCE-FIELD" not in str(interaction.request)

            history = await runtime.history.history(
                execution_id,
                principal=runtime.default_principal,
                include_content=True,
                limit=100,
            )
            assert history.next_cursor is None
            assert any(
                item.content_included and marker in str(item.content)
                for item in history.items
            )


@pytest.mark.asyncio
async def test_projected_agent_file_is_reused_from_prepared_input_on_recovery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    attachment = b"RECOVERED_PROJECTED_ATTACHMENT"
    attachment_path = workspace_root / "brief.txt"
    attachment_path.write_bytes(attachment)
    workspace = Workspace.load(workspace_root)
    storage_root = tmp_path / "projected-file-state"
    monkeypatch.setattr(task_local, "_LEASE_SECONDS", 1)
    monkeypatch.setattr(task_local, "_HEARTBEAT_SECONDS", 0.1)
    application = workspace_features("application", workspace=workspace, features_type=RuntimeFeatures)
    application.agent(
        "default",
        model="default",
        allow_tools=(),
        allow_skills=(),
        allow_subagents=(),
    )
    models = _TaskTestModels()
    builds: list[str] = []
    entered_start = asyncio.Event()

    async def source_task(context: TaskNodeContext[None]) -> JsonValue:
        del context
        return {"ticket": "RECOVERY-317"}

    async def build_input(context: AgentTaskInputContext) -> str:
        builds.append(context.node_id)
        source = await context.result("source")
        assert isinstance(source, dict)
        return f"inspect projected ticket {source['ticket']}"

    source = Task("example.projected-file-source", source_task, effect_policy="none")
    state = PersistenceStorage.filesystem(storage_root)
    graph_id = "projected-file-recovery-graph"

    async with Persistence.open('projected-file-recovery', storage=state) as persistence_3867, Runtime.attach(persistence=persistence_3867, features=model_features(models, application)) as runtime, task_engine_scope(runtime, persistence_3867) as runtime_engine:
        worker = Task.from_agent(
            "example.projected-file-worker",
            runtime.agents.get("default"),
            build_input=build_input,
        )
        graph = TaskGraph(
            graph_id,
            (
                TaskNode("source", task=source),
                TaskNode(
                    "consumer",
                    ("source",),
                    task=worker,
                    input=AgentTaskInput(
                        "base request",
                        files=("brief.txt",),
                    ),
                ),
            ),
        )
        original_start = type(worker.agent_execution).start

        async def pause_before_execution_start(
            *args: object,
            **kwargs: object,
        ) -> object:
            entered_start.set()
            await asyncio.Event().wait()
            return await original_start(*args, **kwargs)  # type: ignore[arg-type]

        with monkeypatch.context() as patch:
            patch.setattr(type(worker.agent_execution), "start", pause_before_execution_start)
            await runtime_engine.with_definitions(source, worker).start(
                graph,
                idempotency_key="projected-file-recovery-run-0001",
             principal=runtime.default_principal)
            await asyncio.wait_for(entered_start.wait(), 10)
            prepared = await runtime._state.task.tasks.get_prepared_input(
                graph.graph_id,
                "consumer",
                tenant_id=runtime.tenant_id,
            )
            assert prepared is not None
            file_views = prepared.stored_user_input.view["files"]
            assert isinstance(file_views, list) and len(file_views) == 1
            assert file_views[0]["digest"] == hashlib.sha256(attachment).hexdigest()
            assert builds == ["consumer"]
            await runtime.close()

    attachment_path.unlink()
    await asyncio.sleep(1.1)
    recovered_storage = PersistenceStorage.filesystem(storage_root)
    async with Persistence.open('projected-file-recovery', storage=recovered_storage) as persistence_3924, Runtime.attach(persistence=persistence_3924, features=model_features(models, application)) as runtime, task_engine_scope(runtime, persistence_3924) as runtime_engine:
        worker = Task.from_agent(
            "example.projected-file-worker",
            runtime.agents.get("default"),
            build_input=build_input,
        )
        engine = runtime_engine.with_definitions(source, worker)
        await engine.recover_pending()
        recovered_run = await engine.get(graph.graph_id)
        completed = (await recovered_run.wait(timeout_seconds=10)).result

        assert completed.wait_status is TaskStatus.SUCCEEDED, completed.node_states
        assert builds == ["consumer"]
        consumer_result = next(
            node for node in completed.node_states if node.node_id == "consumer"
        )
        assert consumer_result.execution_id is not None
        interactions = await runtime.history.model_interactions(
            consumer_result.execution_id,
            principal=runtime.default_principal,
            include_content=True,
            limit=100,
        )
        assert interactions.next_cursor is None
        assert len(interactions.items) == 1
        assert interactions.items[0].status == "SUCCEEDED"
        assert "RECOVERY-317" in str(interactions.items[0].request)
        assert "brief.txt" in str(interactions.items[0].request)
        history = await runtime.history.history(
            consumer_result.execution_id,
            principal=runtime.default_principal,
            include_content=True,
            limit=100,
        )
        assert any(
            item.content_included and "RECOVERY-317" in str(item.content)
            for item in history.items
        )


@pytest.mark.asyncio
async def test_runtime_expands_application_and_agent_tasks_across_batches(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    (workspace_root / "context.txt").write_text("context", encoding="utf-8")
    workspace = Workspace.load(workspace_root)
    application = workspace_features("application", workspace=workspace, features_type=RuntimeFeatures)
    application.capability(ExecutionCapability(retain_history=True), scope="runtime")
    handler = TaskFunction[None]("example.echo", 1, _echo_task)
    application.task(handler, effect_policy="none")
    application.task_expander(_ApplicationGraphExpander(handler))
    application.agent(
        "default",
        model="default",
        allow_tools=(),
        allow_skills=(),
        allow_subagents=(),
    )
    application.agent(
        "worker",
        model="default",
        allow_tools=(),
        allow_skills=(),
        allow_subagents=(),
    )
    app_reference = TaskExpanderRef("application.graph", 1)
    agent_reference = TaskExpanderRef("application.agent-graph", 1)
    state = PersistenceStorage.in_memory()

    application.capability(MemoryCapability(scope="expander-memory"))

    async with Persistence.open('default', storage=state) as persistence_3999, Runtime.attach(persistence=persistence_3999, features=model_features(_TaskTestModels(), application)) as runtime, task_engine_scope(runtime, persistence_3999) as runtime_engine:
        worker_task = agent_task_definition(runtime, "worker", "test.agent.worker")
        application.task_expander(_AgentGraphExpander(worker_task))
        await runtime.agents.get("worker").create_session(
            "expander-session",
            idempotency_key="expander-session-create-0001",
        )
        graph = TaskGraph(
            "dynamic-expansion",
            (
                handler.node(
                    "application-root",
                    input={"value": "root"},
                    expander=app_reference,
                ),
                handler.node("empty-root", expander=app_reference),
                agent_task_node(runtime,
                    "agent-root",
                    "return a root result",
                    expander=agent_reference,
                ),
            ),
        )

        result = await run_task_graph(runtime,
            graph,
            idempotency_key="dynamic-expansion-run-0001",
            timeout_seconds=10,
        )

        assert result.wait_status is TaskStatus.SUCCEEDED, result.node_states
        assert await task_result(runtime,
            graph.graph_id,
            "agent-child",
        ) is not None
        assert {node.node_id for node in result.node_states} == {
            "agent-child",
            "agent-root",
            "application-root",
            "child-a",
            "child-b",
            "disconnected",
            "empty-root",
            "grandchild",
        }
        snapshot = await task_graph_state(runtime,
            graph.graph_id,
            principal=runtime.default_principal,
        )
        assert [node.node_id for node in snapshot.nodes] == sorted(
            node.node_id for node in snapshot.nodes
        )
        agent_child = next(
            node for node in snapshot.nodes if node.node_id == "agent-child"
        )
        assert agent_child.input_refs == {}
        assert agent_child.timeout_seconds == 12
        assert agent_child.max_attempts == 2
        assert agent_child.retry_delay_seconds == 0.25
        assert agent_child.task == worker_task.ref
        assert "binding_contract" not in agent_child.input
        child_state = next(
            state for state in snapshot.node_states if state.node_id == "agent-child"
        )
        assert child_state.execution_id is not None
        child_execution = await runtime.executions.inspect(
            child_state.execution_id,
            principal=runtime.default_principal,
        )
        assert child_execution.session_id == "expander-session"
        persisted = await runtime._state.task.tasks.graph_state(
            graph.graph_id,
            tenant_id=runtime.default_principal.tenant_id,
        )
        assert persisted is not None
        persisted_agent_child = next(
            node for node in persisted.nodes if node.node_id == "agent-child"
        )
        persisted_input = AgentTaskInput.from_mapping(persisted_agent_child.input)
        assert persisted_input.files == ("context.txt",)
        assert persisted_input.session_id == "expander-session"
        assert "memory_scope" not in persisted_input
        child_record = await runtime._state.execution.executions.get(
            child_state.execution_id, tenant_id=runtime.default_principal.tenant_id,
        )
        assert child_record is not None
        assert child_record.memory_scope == "expander-memory"
        assert snapshot.node_states[-1].status is TaskStatus.SUCCEEDED
        child_a_state = next(
            state for state in snapshot.node_states if state.node_id == "child-a"
        )
        assert child_a_state.execution_id is not None
        assert await task_result(runtime,
            graph.graph_id,
            "grandchild",
        ) == {
            "upstream": await task_result(runtime, graph.graph_id, "child-a"),
            "execution_id": child_a_state.execution_id,
        }
        events = await runtime._state.task.tasks.list_events(
            graph.graph_id,
            tenant_id="default",
            after_event_seq=0,
            limit=100,
        )
        expanded = {
            event.source_node_id: event.added_node_ids
            for event in events.items
            if event.event_type.value == "GRAPH_EXPANDED"
        }
        assert expanded == {
            "agent-root": ("agent-child",),
            "application-root": ("child-a", "child-b", "disconnected"),
            "child-a": ("grandchild",),
        }


@pytest.mark.asyncio
async def test_task_input_extensions_do_not_override_runtime_memory() -> None:
    features = model_features(_TaskTestModels())
    features.capability(MemoryCapability(scope="host-memory"))
    features.capability(ExecutionCapability(retain_history=True))
    features.agent("default", model="default", allow_tools=(), allow_skills=(), allow_subagents=())
    async with Persistence.open("task-memory-owner", storage=PersistenceStorage.in_memory()) as persistence:
        async with Runtime.attach(persistence=persistence, features=features) as runtime, task_engine_scope(runtime, persistence) as runtime_engine:
            worker = Task.from_agent("memory.worker", runtime.agents.get())
            run = await runtime_engine.with_definitions(worker).start(
                TaskGraph("memory-owner", (TaskNode("agent", task=worker, input={
                    "prompt": "return a result", "memory_scope": {"application": "value"},
                }),)),
                principal=runtime.default_principal, idempotency_key="memory-owner",
            )
            completed = (await run.wait(timeout_seconds=10)).result
            assert completed.wait_status is TaskStatus.SUCCEEDED
            execution = await run.execution("agent")
            record = await persistence.execution.executions.get(execution.execution_id, tenant_id=runtime.tenant_id)
            assert record is not None and record.memory_scope == "host-memory"


@pytest.mark.asyncio
async def test_task_execution_port_replay_rejects_changed_runtime_memory_scope() -> None:
    features = model_features(_TaskTestModels())
    features.capability(MemoryCapability(scope="task-memory"))
    features.capability(ExecutionCapability(retain_history=True))
    features.agent("default", model="default", allow_tools=(), allow_skills=(), allow_subagents=())
    async with Persistence.open("task-memory-replay", storage=PersistenceStorage.in_memory()) as persistence:
        async with Runtime.attach(persistence=persistence, features=features) as runtime:
            task = Task.from_agent("example.memory-worker", runtime.agents.get("default"))
            port = task.agent_execution
            assert port is not None
            assert port.memory_scope == "task-memory"
            start = dict(
                files=(), output=None, principal=runtime.default_principal,
                session_id=None, idempotency_key="task-memory-replay-0001",
                mode="run", planning=False, thinking=False,
            )
            # A lost admission response retries the same request, not a new memory scope.
            admitted = await port.start("return a result", **start)
            replayed = await port.start("return a result", **start)
            assert replayed.execution_id == admitted.execution_id
            changed = replace(port, memory_scope="other-memory")
            with pytest.raises(AIError) as conflict:
                await changed.start("return a result", **start)
            assert conflict.value.code is ErrorCode.IDEMPOTENCY_CONFLICT
            record = await persistence.execution.executions.get(
                admitted.execution_id, tenant_id=runtime.tenant_id,
            )
            assert record is not None
            assert record.memory_scope == "task-memory"
            await admitted.wait()


@pytest.mark.asyncio
@pytest.mark.parametrize("admitted_scope", (None, "task-memory"))
@pytest.mark.parametrize("phase", ("unknown-admission", "bound"))
async def test_task_execution_scope_survives_unknown_and_bound_recovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, admitted_scope: str | None, phase: str,
) -> None:
    from linktools.ai.persistence import _filesystem

    entered = asyncio.Event()
    execution_ids: list[str] = []
    request_keys: list[str] = []
    storage_root = tmp_path / "task-recovery"
    model_calls: list[None] = []
    from contextlib import asynccontextmanager

    original_request_stream = TestModel.request_stream

    @asynccontextmanager
    async def count_request_stream(self, *args, **kwargs):
        model_calls.append(None)
        async with original_request_stream(self, *args, **kwargs) as stream:
            yield stream

    monkeypatch.setattr(TestModel, "request_stream", count_request_stream)

    def features(scope: str | None) -> RuntimeFeatures:
        result = model_features(_TaskTestModels())
        result.agent("default", model="default", allow_tools=(), allow_skills=(), allow_subagents=())
        result.capability(ExecutionCapability(retain_history=True))
        if scope is not None:
            result.capability(MemoryCapability(scope=scope))
        return result

    original_wait = AgentTaskRunner._wait_execution

    async with Persistence.open("task-recovery", storage=PersistenceStorage.filesystem(storage_root)) as persistence:
        async with Runtime.attach(persistence=persistence, features=features(admitted_scope)) as runtime, task_engine_scope(runtime, persistence) as runtime_engine:
            worker = Task.from_agent("recovery.worker", runtime.agents.get())
            port_type = type(worker.agent_execution)
            original_start = port_type.start

            async def paused_start(self, *args, **kwargs):
                execution = await original_start(self, *args, **kwargs)
                execution_ids.append(execution.execution_id)
                request_keys.append(kwargs["idempotency_key"])
                if phase == "unknown-admission":
                    await execution.wait(timeout_seconds=10)
                    entered.set()
                    await asyncio.Event().wait()
                return execution

            async def paused_wait(self, invocation, execution, execution_id, budget_scope):
                await execution.wait(timeout_seconds=10)
                entered.set()
                await asyncio.Event().wait()
                return await original_wait(self, invocation, execution, execution_id, budget_scope)

            with monkeypatch.context() as patch:
                patch.setattr(port_type, "start", paused_start)
                if phase.startswith("bound"):
                    patch.setattr(AgentTaskRunner, "_wait_execution", paused_wait)
                await runtime_engine.with_definitions(worker).start(
                    TaskGraph("recovery-graph", (TaskNode("agent", task=worker, input=AgentTaskInput("recovery prompt")),)),
                    principal=runtime.default_principal, idempotency_key="recovery-graph-start",
                )
                await asyncio.wait_for(entered.wait(), 10)
                source = await persistence.task.tasks.graph_state("recovery-graph", tenant_id=runtime.tenant_id)
                assert source is not None
                source_input = dict(source.nodes[0].input)
                assert "memory_scope" not in source_input
                state = source.node_states[0]
                assert (state.execution_id is None) == (phase == "unknown-admission")
                admitted_calls = len(model_calls)
                assert admitted_calls > 0
                await runtime.close()
                persisted = await persistence.execution.executions.get(
                    execution_ids[0], tenant_id=runtime.tenant_id,
                )
                assert persisted is not None
                if phase == "bound":
                    assert "task:recovery-graph:agent" in persisted.dependency_hold_ids
                    assert persisted.retention_closed is False

    expired_at = datetime.now(timezone.utc) + timedelta(seconds=task_local._LEASE_SECONDS + 1)

    class AfterLeaseExpiry:
        @classmethod
        def now(cls, tz: timezone | None = None) -> datetime:
            return expired_at.astimezone(tz)

    # Keep the live admission window intact, then expire only the abandoned lease.
    monkeypatch.setattr(task_local, "datetime", AfterLeaseExpiry)
    monkeypatch.setattr(_filesystem, "datetime", AfterLeaseExpiry)
    async with Persistence.open("task-recovery", storage=PersistenceStorage.filesystem(storage_root)) as persistence:
        async with Runtime.attach(persistence=persistence, features=features(admitted_scope if phase == "unknown-admission" else "changed-host")) as runtime, task_engine_scope(runtime, persistence) as runtime_engine:
            worker = Task.from_agent("recovery.worker", runtime.agents.get())

            async def replay_start(self, *args, **kwargs):
                execution = await original_start(self, *args, **kwargs)
                execution_ids.append(execution.execution_id)
                request_keys.append(kwargs["idempotency_key"])
                return execution

            with monkeypatch.context() as patch:
                patch.setattr(port_type, "start", replay_start)
                engine = runtime_engine.with_definitions(worker)
                await engine.recover_pending()
                run = await engine.get("recovery-graph")
                completed = (await run.wait(timeout_seconds=10)).result
            assert completed.wait_status is TaskStatus.SUCCEEDED, completed.node_states
            assert len(execution_ids) == (2 if phase == "unknown-admission" else 1)
            assert len(set(execution_ids)) == 1
            assert len(set(request_keys)) == 1
            record = await persistence.execution.executions.get(execution_ids[0], tenant_id=runtime.tenant_id)
            assert record is not None
            assert record.memory_scope == admitted_scope
            assert len(model_calls) == admitted_calls
            assert record.dependency_hold_ids == ()
            restored = await persistence.task.tasks.graph_state("recovery-graph", tenant_id=runtime.tenant_id)
            assert restored is not None
            assert dict(restored.nodes[0].input) == source_input
            # A concurrent launcher finalizer can own the handoff cleanup. Check
            # its durable closing result after all owned work has quiesced.
            await runtime.close()
            repository = persistence.execution.executions
            closed = await repository.state_store.read(lambda transaction: repository.get_in_transaction(
                transaction, execution_ids[0], tenant_id=runtime.tenant_id,
            ))
            assert closed is not None and closed.retention_closed is True


async def _wait_for_input_execution(runtime: Runtime, graph_id: str) -> str:
    async def wait() -> str:
        while True:
            snapshot = await task_graph_state(runtime,
                graph_id,
                principal=runtime.default_principal,
            )
            input_state = next(
                state for state in snapshot.node_states if state.node_id == "input"
            )
            if input_state.status is TaskStatus.WAITING:
                assert input_state.execution_id is not None
                return input_state.execution_id
            await asyncio.sleep(0)

    return await asyncio.wait_for(wait(), timeout=10)


@pytest.mark.asyncio
async def test_reopened_expansion_uses_captured_candidates_and_supports_nesting(
    tmp_path: Path,
) -> None:
    async def run_task(context: TaskNodeContext[None]) -> JsonValue:
        return {"node": context.node_id}

    handler = TaskFunction[None]("example.reopened-expansion", 1, run_task)
    unused = TaskFunction[None]("example.unused-expansion", 1, run_task)
    expander_reference = TaskExpanderRef("example.reopened-expander", 1)

    class Expander:
        id = expander_reference.id
        revision = expander_reference.revision

        def expand(self, context: TaskExpansionContext) -> tuple[TaskNode, ...]:
            if context.source_node.node_id == "root":
                return (
                    handler.node(
                        "child",
                        dependencies=("root",),
                        expander=expander_reference,
                    ),
                )
            if context.source_node.node_id == "child":
                return (handler.node("grandchild", dependencies=("child",)),)
            return ()

    expander = Expander()
    graph = TaskGraph(
        "reopened-expansion",
        (
            TaskNode.wait("input"),
            handler.node(
            "root",
            dependencies=("input",),
            expander=expander_reference,
        ),
        ),
    )
    storage_root = tmp_path / "state"
    initial_application = RuntimeFeatures[None]("application")
    initial_application.task(handler, effect_policy="none")
    initial_application.task(unused, effect_policy="none")
    initial_application.task_expander(expander)
    async with Persistence.open('reopened-expansion', storage=PersistenceStorage.filesystem(storage_root)) as persistence_4178, Runtime.attach(persistence=persistence_4178, features=model_features(_TaskTestModels(), initial_application)) as runtime, task_engine_scope(runtime, persistence_4178) as runtime_engine:
        await start_task_graph(runtime,
            graph,
            idempotency_key="reopened-expansion-run-0001",
        )
        wait_id = await _wait_for_input_execution(runtime, graph.graph_id)

    application = RuntimeFeatures[None]("application")
    application.task(handler, effect_policy="none")
    application.task(unused, effect_policy="replay_safe")
    application.task_expander(expander)
    async with Persistence.open('reopened-expansion', storage=PersistenceStorage.filesystem(storage_root)) as persistence_4193, Runtime.attach(persistence=persistence_4193, features=model_features(_TaskTestModels(), application)) as runtime, task_engine_scope(runtime, persistence_4193) as runtime_engine:
        principal = runtime.default_principal
        await task_graph_resume(runtime,
            graph.graph_id,
            "input",
            TaskInputSupplyRequest(
                principal,
                wait_id,
                {"value": "seed"},
                "reopened-expansion-input-0001",
            ),
        )
        result = await task_graph_wait(runtime,
            graph.graph_id,
            principal=principal,
            timeout_seconds=10,
        )

    assert result.wait_status is TaskStatus.SUCCEEDED
    assert {node.node_id for node in result.node_states} == {
        "input",
        "root",
        "child",
        "grandchild",
    }


@pytest.mark.asyncio
async def test_reopened_expansion_rejects_a_new_runtime_task_candidate(
    tmp_path: Path,
) -> None:
    async def run_task(context: TaskNodeContext[None]) -> JsonValue:
        return {"node": context.node_id}

    required = TaskFunction[None]("example.reopened-required", 1, run_task)
    added = TaskFunction[None]("example.added-after-capture", 1, run_task)
    expander = TaskExpander(
        "example.added-candidate-expander",
        lambda _context: (added.node("added"),),
    )
    graph = TaskGraph(
        "reopened-added-candidate",
        (
            TaskNode.wait("input"),
            required.node(
                "root",
                dependencies=("input",),
                expander=expander.ref,
            ),
        ),
    )
    storage_root = tmp_path / "state"
    async with Persistence.open('reopened-added-candidate', storage=PersistenceStorage.filesystem(storage_root)) as persistence_4249, Runtime.attach(persistence=persistence_4249, features=model_features(_TaskTestModels())) as runtime, task_engine_scope(runtime, persistence_4249) as runtime_engine:
        await runtime_engine.with_definitions(required, expander).start(
            graph,
            idempotency_key="reopened-added-candidate-run-0001",
         principal=runtime.default_principal)
        wait_id = await _wait_for_input_execution(runtime, graph.graph_id)

    async with Persistence.open('reopened-added-candidate', storage=PersistenceStorage.filesystem(storage_root)) as persistence_4260, Runtime.attach(persistence=persistence_4260, features=model_features(_TaskTestModels())) as runtime, task_engine_scope(runtime, persistence_4260) as runtime_engine:
        principal = runtime.default_principal
        run = await runtime_engine.with_definitions(required, added, expander).get(
            graph.graph_id,
            principal=principal,
        )
        await run.resume(
            "input",
            TaskInputSupplyRequest(
                principal,
                wait_id,
                {"value": "seed"},
                "reopened-added-candidate-input-0001",
            ),
        )
        result = (await run.wait(timeout_seconds=10)).result

    root = next(node for node in result.node_states if node.node_id == "root")
    assert result.wait_status is TaskStatus.FAILED
    assert root.error_code == ErrorCode.BINDING_NOT_REGISTERED.value
    assert all(node.node_id != "added" for node in result.node_states)


class _EffectOutput(BaseModel):
    value: str


class _ChangedEffectOutput(BaseModel):
    value: int


@pytest.mark.asyncio
async def test_public_graph_start_canonicalizes_registered_task_semantics() -> None:
    async def valid_output(context: TaskNodeContext[None]) -> JsonValue:
        del context
        return {"value": "ok"}

    async def reconcile(
        context: TaskNodeContext[None],
    ) -> TaskEffectResolution:
        del context
        return TaskEffectResolution("unknown")

    application = RuntimeFeatures[None]("application")
    handler = TaskFunction[None]("example.public-start", 1, valid_output)
    application.task(
        handler,
        effect_policy="non_replay_safe",
        output_type=_EffectOutput,
        reconcile=reconcile,
    )
    state = PersistenceStorage.in_memory()

    async with Persistence.open('task-public-start', storage=state) as persistence_4317, Runtime.attach(persistence=persistence_4317, features=model_features(_TaskTestModels(), application)) as runtime, task_engine_scope(runtime, persistence_4317) as runtime_engine:
        request = TaskGraphRequest(
            TaskGraph("task-public-start", (handler.node("node"),)),
            runtime.default_principal,
            "task-public-start-0001",
        )
        await start_task_request(runtime, request)
        graph_state = await runtime._state.task.tasks.graph_state(
            "task-public-start",
            tenant_id=runtime.default_principal.tenant_id,
        )

    assert graph_state is not None
    node = graph_state.nodes[0]
    assert node.effect_policy == "non_replay_safe"
    assert node.output_contract is not None
    assert node.output_contract["mode"] == "structured"
    assert node.reconcile is True


@pytest.mark.asyncio
async def test_persisted_node_output_contracts_survive_runtime_reopen(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    workspace = Workspace.load(workspace_root)
    storage_root = tmp_path / "state"

    async def copy_input(context: TaskNodeContext[None]) -> JsonValue:
        return await context.read_dependency("input")

    application = workspace_features("application", workspace=workspace, features_type=RuntimeFeatures)
    handler = TaskFunction[None]("example.typed-node", 1, copy_input)
    application.task(handler, effect_policy="none")
    application.agent(
        "default",
        model="default",
        allow_tools=(),
        allow_skills=(),
        allow_subagents=(),
    )

    state = PersistenceStorage.filesystem(storage_root)
    async with Persistence.open('typed-contract-reopen', storage=state) as persistence_4365, Runtime.attach(persistence=persistence_4365, features=model_features(_StructuredTaskTestModels(), application)) as runtime, task_engine_scope(runtime, persistence_4365) as runtime_engine:
        graph = TaskGraph(
            "typed-contract-reopen",
            (
                TaskNode.wait("input", output_type=_EffectOutput),
                TaskNode(
                    "typed-custom",
                    dependencies=("input",),
                    task=handler,
                    output_type=_EffectOutput,
                ),
                agent_task_node(runtime,
                    "typed-agent",
                    "return a structured result",
                    dependencies=("typed-custom",),
                    output_type=_EffectOutput,
                ),
            ),
        )
        await start_task_graph(runtime,
            graph,
            idempotency_key="typed-contract-reopen-0001",
        )
        wait_id = await _wait_for_input_execution(runtime, graph.graph_id)
        invalid_custom_graph = TaskGraph(
            "typed-custom-invalid-reopen",
            (
                TaskNode.wait("input"),
                TaskNode(
                    "typed-custom-invalid",
                    dependencies=("input",),
                    task=handler,
                    output_type=_ChangedEffectOutput,
                ),
            ),
        )
        await start_task_graph(runtime,
            invalid_custom_graph,
            idempotency_key="typed-custom-invalid-reopen-0001",
        )
        invalid_custom_wait_id = await _wait_for_input_execution(
            runtime,
            invalid_custom_graph.graph_id,
        )

    state = PersistenceStorage.filesystem(storage_root)
    async with Persistence.open('typed-contract-reopen', storage=state) as persistence_4415, Runtime.attach(persistence=persistence_4415, features=model_features(_StructuredTaskTestModels(), application)) as runtime, task_engine_scope(runtime, persistence_4415) as runtime_engine:
        recovered = await task_graph_state(runtime,
            "typed-contract-reopen",
            principal=runtime.default_principal,
        )
        contracts = {node.node_id: node.output_contract for node in recovered.nodes}
        assert all(
            contracts[node_id] is not None
            and contracts[node_id]["mode"] == "structured"
            for node_id in ("input", "typed-custom", "typed-agent")
        )
        agent_contract = contracts["typed-agent"]
        assert agent_contract is not None
        restored_agent_output = restore_output(
            agent_contract["mode"],
            agent_contract["schema"],
        )
        with pytest.raises(AIError) as invalid_agent_output:
            restored_agent_output.validate_payload({"wrong": True})
        assert invalid_agent_output.value.code is ErrorCode.OUTPUT_VALIDATION_FAILED

        with pytest.raises(AIError) as invalid_input:
            await task_graph_resume(runtime,
                "typed-contract-reopen",
                "input",
                TaskInputSupplyRequest(
                    runtime.default_principal,
                    wait_id,
                    {"wrong": True},
                    "typed-contract-invalid-input-0001",
                ),
            )
        assert invalid_input.value.code is ErrorCode.OUTPUT_CONTRACT_INVALID

        await task_graph_resume(runtime,
            "typed-contract-reopen",
            "input",
            TaskInputSupplyRequest(
                runtime.default_principal,
                wait_id,
                {"value": "accepted"},
                "typed-contract-valid-input-0001",
            ),
        )
        result = await task_graph_wait(runtime,
            "typed-contract-reopen",
            principal=runtime.default_principal,
            timeout_seconds=10,
        )
        assert result.wait_status is TaskStatus.SUCCEEDED
        output = await task_result(runtime,
            "typed-contract-reopen",
            "typed-agent",
        )
        _EffectOutput.model_validate(output)
        await task_graph_resume(runtime,
            "typed-custom-invalid-reopen",
            "input",
            TaskInputSupplyRequest(
                runtime.default_principal,
                invalid_custom_wait_id,
                {"value": "not-an-integer"},
                "typed-custom-invalid-input-0001",
            ),
        )
        invalid_custom_result = await task_graph_wait(runtime,
            "typed-custom-invalid-reopen",
            principal=runtime.default_principal,
            timeout_seconds=10,
        )

    invalid_custom = next(
        node
        for node in invalid_custom_result.node_states
        if node.node_id == "typed-custom-invalid"
    )
    assert invalid_custom_result.wait_status is TaskStatus.FAILED
    assert invalid_custom.error_code == ErrorCode.OUTPUT_VALIDATION_FAILED.value


async def _invalid_effect_output(context: TaskNodeContext[None]) -> JsonValue:
    del context
    return {"wrong": True}


@pytest.mark.asyncio
async def test_non_replay_safe_applied_resolution_is_owned_by_execution(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    workspace = Workspace.load(workspace_root)
    application = workspace_features("application", workspace=workspace, features_type=RuntimeFeatures)
    handler = TaskFunction[None]("example.effect-applied", 1, _invalid_effect_output)
    application.task(
        handler,
        effect_policy="non_replay_safe",
        output_type=_EffectOutput,
    )
    application.agent(
        "default",
        model="default",
        allow_tools=(),
        allow_skills=(),
        allow_subagents=(),
    )
    state = PersistenceStorage.in_memory()

    async with Persistence.open('default', storage=state) as persistence_4527, Runtime.attach(persistence=persistence_4527, features=model_features(_TaskTestModels(), application)) as runtime, task_engine_scope(runtime, persistence_4527) as runtime_engine:
        run = await start_task_graph(runtime,
            TaskGraph("effect-applied", (handler.node("node"),)),
            idempotency_key="effect-applied-run-0001",
        )
        initial = (await run.wait(timeout_seconds=10)).result
        assert initial.wait_status is TaskStatus.RECOVERY_REQUIRED

        snapshot = await task_graph_state(runtime,
            run.graph_id,
            principal=runtime.default_principal,
        )
        node_state = snapshot.node_states[0]
        assert node_state.status is TaskStatus.RECOVERY_REQUIRED
        assert node_state.execution_id is not None

        resolved = await run.resolve_effect(
            "node",
            node_state.fence,
            TaskEffectResolution("applied", {"value": "recovered"}),
            idempotency_key="effect-applied-resolution-0001",
        )

        assert resolved.status is TaskStatus.SUCCEEDED
        assert await run.result("node") == {"value": "recovered"}
        execution = await runtime.executions.result(
            node_state.execution_id,
            principal=runtime.default_principal,
        )
        assert execution.status.value == "SUCCEEDED"
        events = await runtime._state.execution.events.list(
            node_state.execution_id,
            tenant_id="default",
            after_event_seq=0,
            limit=100,
        )
        assert events.items[-1].payload["task_effect"] == "applied"
        assert events.items[-1].payload["task_effect_value_digest"] == canonical_sha256(
            {"value": "recovered"}
        )


@pytest.mark.asyncio
async def test_non_replay_safe_invalid_applied_value_preserves_effect_fact(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    workspace = Workspace.load(workspace_root)
    application = workspace_features("application", workspace=workspace, features_type=RuntimeFeatures)
    handler = TaskFunction[None]("example.effect-invalid", 1, _invalid_effect_output)
    application.task(
        handler,
        effect_policy="non_replay_safe",
        output_type=_EffectOutput,
    )
    application.agent(
        "default",
        model="default",
        allow_tools=(),
        allow_skills=(),
        allow_subagents=(),
    )

    async with Persistence.open('default', storage=PersistenceStorage.in_memory()) as persistence_4595, Runtime.attach(persistence=persistence_4595, features=model_features(_TaskTestModels(), application)) as runtime, task_engine_scope(runtime, persistence_4595) as runtime_engine:
        run = await start_task_graph(runtime,
            TaskGraph("effect-invalid", (handler.node("node"),)),
            idempotency_key="effect-invalid-run-0001",
        )
        initial = (await run.wait(timeout_seconds=10)).result
        assert initial.wait_status is TaskStatus.RECOVERY_REQUIRED
        snapshot = await task_graph_state(runtime,
            run.graph_id,
            principal=runtime.default_principal,
        )
        node_state = snapshot.node_states[0]
        assert node_state.execution_id is not None

        resolved = await run.resolve_effect(
            "node",
            node_state.fence,
            TaskEffectResolution("applied", {"wrong": True}),
            idempotency_key="effect-invalid-resolution-0001",
        )

        assert resolved.status is TaskStatus.FAILED
        execution = await runtime.executions.result(
            node_state.execution_id,
            principal=runtime.default_principal,
        )
        assert execution.error_code == ErrorCode.OUTPUT_CONTRACT_INVALID.value
        assert execution.safe_error_details["task_effect"] == "applied"
        assert execution.safe_error_details["task_effect_value_digest"] == canonical_sha256(
            {"wrong": True}
        )


@pytest.mark.asyncio
async def test_not_applied_retries_same_execution_once(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    workspace = Workspace.load(workspace_root)
    calls = 0

    async def flaky_effect(context: TaskNodeContext[None]) -> JsonValue:
        nonlocal calls
        del context
        calls += 1
        if calls == 1:
            raise RuntimeError("effect outcome is unknown")
        return {"ok": True}

    application = workspace_features("application", workspace=workspace, features_type=RuntimeFeatures)
    handler = TaskFunction[None]("example.effect-retry", 1, flaky_effect)
    application.task(handler, effect_policy="non_replay_safe")
    application.agent(
        "default",
        model="default",
        allow_tools=(),
        allow_skills=(),
        allow_subagents=(),
    )

    async with Persistence.open('default', storage=PersistenceStorage.in_memory()) as persistence_4660, Runtime.attach(persistence=persistence_4660, features=model_features(_TaskTestModels(), application)) as runtime, task_engine_scope(runtime, persistence_4660) as runtime_engine:
        run = await start_task_graph(runtime,
            TaskGraph(
                "effect-retry",
                (handler.node("node", max_attempts=2, retry_delay_seconds=0),),
            ),
            idempotency_key="effect-retry-run-0001",
        )
        initial = (await run.wait(timeout_seconds=10)).result
        assert initial.wait_status is TaskStatus.RECOVERY_REQUIRED
        before = await task_graph_state(runtime,
            run.graph_id,
            principal=runtime.default_principal,
        )
        state_before = before.node_states[0]
        assert state_before.execution_id is not None

        resumed = await run.resolve_effect(
            "node",
            state_before.fence,
            TaskEffectResolution("not_applied"),
            idempotency_key="effect-retry-resolution-0001",
        )
        assert resumed.status in {TaskStatus.PENDING, TaskStatus.RUNNING}
        final = (await run.wait(timeout_seconds=10)).result

        assert final.wait_status is TaskStatus.SUCCEEDED
        assert calls == 2
        after = await task_graph_state(runtime,
            run.graph_id,
            principal=runtime.default_principal,
        )
        assert after.node_states[0].execution_id == state_before.execution_id
        execution = await runtime.executions.inspect(
            state_before.execution_id,
            principal=runtime.default_principal,
        )
        assert execution.task_attempt == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("reconcile_case", ("unknown", "exception", "invalid"))
async def test_runtime_reconcile_unknown_exception_and_invalid_stay_recoverable(
    reconcile_case: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    run_calls = 0
    reconcile_calls = 0

    async def run_task(context: TaskNodeContext[None]) -> JsonValue:
        nonlocal run_calls
        del context
        run_calls += 1
        raise RuntimeError("effect outcome is unknown")

    async def reconcile(
        context: TaskNodeContext[None],
    ) -> TaskEffectResolution:
        nonlocal reconcile_calls
        del context
        reconcile_calls += 1
        if reconcile_case == "exception":
            raise ValueError("reconciliation unavailable")
        if reconcile_case == "invalid":
            return {"kind": "unknown"}  # type: ignore[return-value]
        return TaskEffectResolution("unknown")

    task = Task(
        f"example.reconcile-{reconcile_case}",
        run_task,
        effect_policy="non_replay_safe",
        reconcile=reconcile,
    )
    graph = TaskGraph(
        f"reconcile-{reconcile_case}",
        (TaskNode("node", task=task),),
    )
    async with Persistence.open(f'runtime-reconcile-{reconcile_case}', storage=PersistenceStorage.in_memory()) as persistence_4741, Runtime.attach(persistence=persistence_4741, features=model_features(_TaskTestModels())) as runtime, task_engine_scope(runtime, persistence_4741) as runtime_engine:
        graph_run = await runtime_engine.with_definitions(task).start(
            graph,
            idempotency_key=f"reconcile-{reconcile_case}-graph-0001",
         principal=runtime.default_principal)
        initial = (await graph_run.wait(timeout_seconds=10)).result
        assert initial.wait_status is TaskStatus.RECOVERY_REQUIRED
        initial_state = await graph_run.state(include_content=True)
        initial_node = initial_state.node_states[0]
        assert initial_node.execution_id is not None
        assert reconcile_calls == 0

        await graph_run.recover(
            idempotency_key=f"reconcile-{reconcile_case}-recover-0001",
        )
        recovered = (await graph_run.wait(timeout_seconds=10)).result
        recovered_state = await graph_run.state(include_content=True)
        recovered_node = recovered_state.node_states[0]
        execution = await runtime.executions.inspect(
            initial_node.execution_id,
            principal=runtime.default_principal,
        )

        assert recovered.wait_status is TaskStatus.RECOVERY_REQUIRED
        assert recovered_node.status is TaskStatus.RECOVERY_REQUIRED
        assert recovered_node.execution_id == initial_node.execution_id
        assert recovered_node.error_code == ErrorCode.TASK_EFFECT_UNKNOWN.value
        assert recovered_node.error_origin == "execution"
        assert run_calls == 1
        assert reconcile_calls == 1
        assert execution.status.value == "RECOVERY_REQUIRED"
        messages = [record.getMessage() for record in caplog.records]
        if reconcile_case == "exception":
            assert recovered_node.safe_error_details == {
                "reconcile_exception_type": "ValueError"
            }
            assert any(
                "task effect reconciliation failed" in message
                and "type=ValueError" in message
                for message in messages
            )
        elif reconcile_case == "invalid":
            assert recovered_node.safe_error_details == {
                "reconcile_result_type": "dict"
            }
            assert any(
                "task effect reconciliation returned an invalid value" in message
                and "type=dict" in message
                for message in messages
            )
        else:
            assert recovered_node.safe_error_details == {
                "reconcile_outcome": "unknown"
            }


@pytest.mark.asyncio
async def test_not_applied_after_attempt_budget_exhaustion_fails_execution_and_node(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    workspace = Workspace.load(workspace_root)
    calls = 0

    async def uncertain_effect(context: TaskNodeContext[None]) -> JsonValue:
        nonlocal calls
        del context
        calls += 1
        raise RuntimeError("effect outcome is unknown")

    application = workspace_features("application", workspace=workspace, features_type=RuntimeFeatures)
    handler = TaskFunction[None](
        "example.effect-attempt-budget",
        1,
        uncertain_effect,
    )
    application.task(handler, effect_policy="non_replay_safe")
    application.agent(
        "default",
        model="default",
        allow_tools=(),
        allow_skills=(),
        allow_subagents=(),
    )

    async with Persistence.open('effect-attempt-budget', storage=PersistenceStorage.in_memory()) as persistence_4831, Runtime.attach(persistence=persistence_4831, features=model_features(_TaskTestModels(), application)) as runtime, task_engine_scope(runtime, persistence_4831) as runtime_engine:
        run = await start_task_graph(
            runtime,
            TaskGraph(
                "effect-attempt-budget",
                (handler.node("node", max_attempts=1),),
            ),
            idempotency_key="effect-attempt-budget-run-0001",
        )
        initial = (await run.wait(timeout_seconds=10)).result
        assert initial.wait_status is TaskStatus.RECOVERY_REQUIRED
        before = await task_graph_state(
            runtime,
            run.graph_id,
            principal=runtime.default_principal,
        )
        node_before = before.node_states[0]
        assert node_before.execution_id is not None

        resolved = await run.resolve_effect(
            "node",
            node_before.fence,
            TaskEffectResolution("not_applied"),
            idempotency_key="effect-attempt-budget-resolution-0001",
        )
        execution = await runtime_engine._execution_service.result(
            node_before.execution_id,
            principal=runtime.default_principal,
        )

        assert resolved.status is TaskStatus.FAILED
        assert resolved.node_results[0].status is TaskStatus.FAILED
        assert resolved.node_results[0].error_code == ErrorCode.TASK_NODE_FAILED.value
        assert execution.status.value == "FAILED"
        assert execution.safe_error_details == {
            "task_effect": "not_applied",
            "reason": "attempts_exhausted",
        }
        assert calls == 1


@pytest.mark.asyncio
async def test_not_applied_after_deadline_expires_fails_execution_and_node(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    workspace = Workspace.load(workspace_root)
    entered = asyncio.Event()

    async def uncertain_effect(context: TaskNodeContext[None]) -> JsonValue:
        del context
        entered.set()
        await asyncio.Event().wait()
        raise AssertionError("timed out effect unexpectedly completed")

    application = workspace_features("application", workspace=workspace, features_type=RuntimeFeatures)
    handler = TaskFunction[None](
        "example.effect-deadline-budget",
        1,
        uncertain_effect,
    )
    application.task(handler, effect_policy="non_replay_safe")
    application.agent(
        "default",
        model="default",
        allow_tools=(),
        allow_skills=(),
        allow_subagents=(),
    )

    async with Persistence.open('effect-deadline-budget', storage=PersistenceStorage.in_memory()) as persistence_4906, Runtime.attach(persistence=persistence_4906, features=model_features(_TaskTestModels(), application)) as runtime, task_engine_scope(runtime, persistence_4906) as runtime_engine:
        run = await start_task_graph(
            runtime,
            TaskGraph(
                "effect-deadline-budget",
                (handler.node("node", timeout_seconds=0.05, max_attempts=2),),
            ),
            idempotency_key="effect-deadline-budget-run-0001",
        )
        await asyncio.wait_for(entered.wait(), 3)
        initial = (await run.wait(timeout_seconds=10)).result
        assert initial.wait_status is TaskStatus.RECOVERY_REQUIRED
        before = await task_graph_state(
            runtime,
            run.graph_id,
            principal=runtime.default_principal,
        )
        node_before = before.node_states[0]
        assert node_before.execution_id is not None

        resolved = await run.resolve_effect(
            "node",
            node_before.fence,
            TaskEffectResolution("not_applied"),
            idempotency_key="effect-deadline-budget-resolution-0001",
        )
        execution = await runtime_engine._execution_service.result(
            node_before.execution_id,
            principal=runtime.default_principal,
        )

        assert resolved.status is TaskStatus.FAILED
        assert resolved.node_results[0].status is TaskStatus.FAILED
        assert resolved.node_results[0].error_code == ErrorCode.EXECUTION_WAIT_TIMEOUT.value
        assert execution.status.value == "FAILED"
        assert execution.error_code == ErrorCode.EXECUTION_WAIT_TIMEOUT.value
        assert execution.safe_error_details == {
            "task_effect": "not_applied",
            "reason": "deadline_exceeded",
        }


@pytest.mark.asyncio
async def test_retry_beyond_task_deadline_fails_execution_instead_of_leaving_started(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    workspace = Workspace.load(workspace_root)
    calls = 0
    retry_requested = asyncio.Event()

    async def retryable_failure(context: TaskNodeContext[None]) -> JsonValue:
        nonlocal calls
        del context
        calls += 1
        raise AIError(ErrorCode.MODEL_TIMEOUT)

    application = workspace_features("application", workspace=workspace, features_type=RuntimeFeatures)
    handler = TaskFunction[None](
        "example.retry-after-deadline",
        1,
        retryable_failure,
        effect_policy="none",
    )
    application.task(handler, effect_policy="none")
    application.agent(
        "default",
        model="default",
        allow_tools=(),
        allow_skills=(),
        allow_subagents=(),
    )

    async with Persistence.open('retry-after-deadline', storage=PersistenceStorage.in_memory()) as persistence_4985, Runtime.attach(persistence=persistence_4985, features=model_features(_TaskTestModels(), application)) as runtime, task_engine_scope(runtime, persistence_4985) as runtime_engine:
        original_schedule = runtime_engine._execution_service.schedule_task_retry

        async def observe_retry_request(*args: object, **kwargs: object):
            retry_requested.set()
            return await original_schedule(*args, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(
            runtime_engine._execution_service,
            "schedule_task_retry",
            observe_retry_request,
        )
        run = await start_task_graph(
            runtime,
            TaskGraph(
                "retry-after-deadline",
                (
                    handler.node(
                        "node",
                        timeout_seconds=3,
                        max_attempts=3,
                        retry_delay_seconds=30,
                    ),
                ),
            ),
            idempotency_key="retry-after-deadline-run-0001",
        )
        result = (await run.wait(timeout_seconds=10)).result
        node = result.node_states[0]
        assert node.execution_id is not None
        execution = await runtime_engine._execution_service.result(
            node.execution_id,
            principal=runtime.default_principal,
        )
        execution_view = await runtime_engine._execution_service.inspect(
            node.execution_id,
            principal=runtime.default_principal,
        )

        assert retry_requested.is_set()
        assert calls == 1
        assert result.wait_status is TaskStatus.FAILED
        assert node.status is TaskStatus.FAILED
        assert node.error_code == ErrorCode.EXECUTION_WAIT_TIMEOUT.value
        assert execution.status.value == "FAILED"
        assert execution.error_code == ErrorCode.EXECUTION_WAIT_TIMEOUT.value
        assert execution_view.task_attempt == 1


@pytest.mark.asyncio
async def test_deferred_input_is_committed_by_execution_and_allows_json_null(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    workspace = Workspace.load(workspace_root)
    application = workspace_features("application", workspace=workspace, features_type=RuntimeFeatures)
    application.agent(
        "default",
        model="default",
        allow_tools=(),
        allow_skills=(),
        allow_subagents=(),
    )

    async with Persistence.open('default', storage=PersistenceStorage.in_memory()) as persistence_5054, Runtime.attach(persistence=persistence_5054, features=model_features(_TaskTestModels(), application)) as runtime, task_engine_scope(runtime, persistence_5054) as runtime_engine:
        run = await start_task_graph(runtime,
            TaskGraph(
                "deferred-input",
                (
                    TaskNode.wait("input"),
                ),
            ),
            idempotency_key="deferred-input-run-0001",
        )
        waiting = (await run.wait(timeout_seconds=10)).result
        node_result = waiting.node_states[0]
        assert node_result.status is TaskStatus.WAITING
        assert node_result.execution_id is not None

        request = TaskInputSupplyRequest(
            runtime.default_principal,
            node_result.execution_id,
            None,
            "deferred-input-value-0001",
        )
        resolved = await run.resume("input", request)

        assert resolved.status is TaskStatus.SUCCEEDED
        assert await run.result("input") is None

        replay = await run.resume("input", request)
        assert replay.status is TaskStatus.SUCCEEDED


class _BindingRunner:
    def __init__(self) -> None:
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.control: TaskNodeRunControl | None = None

    async def run(
        self,
        invocation: TaskNodeInvocation,
        *,
        control: TaskNodeRunControl,
    ) -> TaskNodeRunResult:
        del invocation
        self.control = control
        self.entered.set()
        await self.release.wait()
        payload = StoredPayload.inline_json({"done": True})
        return TaskNodeRunResult(payload.digest)

    async def cancel(self, invocation: TaskNodeInvocation) -> None:
        del invocation

    async def inspect_bound(
        self,
        invocation: TaskNodeInvocation,
        execution_id: str,
    ) -> TaskNodeRunResult | None:
        del invocation, execution_id
        return None


@pytest.mark.asyncio
async def test_local_activity_generation_does_not_lose_pre_wait_handoff_signal() -> None:
    state = PersistenceStorage.in_memory()
    state = Persistence(state)
    await state.initialize(namespace="task-observation-regression", tenant_id="tenant")
    launcher: LocalTaskGraphLauncher | None = None
    try:
        graph = TaskGraph("observation-graph", (TaskNode("node"),))
        repository = state.task.tasks
        principal = Principal("workspace", "tenant", PrincipalKind.LOCAL_TRUSTED.value)
        request = TaskGraphRequest(
            graph,
            principal,
            idempotency_key="observation-graph-run-0001",
        )
        admission = TaskGraphAdmission.from_request(request)
        await state.task.admissions.admit(admission, graph)
        runner = _BindingRunner()
        launcher = LocalTaskGraphLauncher(repository, runner, owner="worker")
        await launcher.start(
            TaskGraphLaunch(graph.graph_id, principal, TaskGraphLimits())
        )
        await asyncio.wait_for(runner.entered.wait(), 1)

        generation = launcher.graph_activity_generation(
            graph.graph_id,
            tenant_id="tenant",
        )
        assert generation is not None
        assert runner.control is not None
        await runner.control.handoff_execution("execution")
        await asyncio.wait_for(
            launcher.wait_graph_activity(
                graph.graph_id,
                tenant_id="tenant",
                after_generation=generation,
            ),
            0.2,
        )
        snapshot = await repository.graph_state(graph.graph_id, tenant_id="tenant")
        assert snapshot is not None
        assert snapshot.node_states[0].status is TaskStatus.WAITING
        assert snapshot.node_states[0].execution_id == "execution"
        runner.release.set()
    finally:
        if launcher is not None:
            await launcher.shutdown()
        await state.close()


@pytest.mark.asyncio
async def test_waiting_recovery_reestablishes_hold_until_task_commit() -> None:
    state = PersistenceStorage.in_memory()
    state = Persistence(state)
    await state.initialize(namespace="task-waiting-hold", tenant_id="tenant")
    launcher: LocalTaskGraphLauncher | None = None
    calls: list[str] = []
    try:
        graph = TaskGraph("waiting-hold", (TaskNode("node"),))
        principal = Principal("workspace", "tenant", PrincipalKind.LOCAL_TRUSTED.value)
        request = TaskGraphRequest(
            graph,
            principal,
            idempotency_key="waiting-hold-submit-0001",
        )
        await state.task.admissions.admit(
            TaskGraphAdmission.from_request(request),
            graph,
        )
        lease = await state.task.tasks.claim(
            graph.graph_id,
            "node",
            tenant_id="tenant",
            owner="worker",
            lease_seconds=30,
        )
        await state.task.tasks.handoff_execution(
            lease,
            tenant_id="tenant",
            execution_id="execution",
        )

        payload = StoredPayload.inline_json({"done": True})

        class WaitingRunner:
            async def run(
                self,
                invocation: TaskNodeInvocation,
                *,
                control: TaskNodeRunControl,
            ) -> TaskNodeRunResult:
                del invocation, control
                raise AssertionError("WAITING recovery must not start a task")

            async def wait_bound(
                self,
                invocation: TaskNodeInvocation,
                execution_id: str,
            ) -> TaskNodeRunResult:
                del invocation
                calls.append(f"wait:{execution_id}")
                return TaskNodeRunResult(
                    payload.digest,
                    execution_id=execution_id,
                )

            async def inspect_bound(
                self,
                invocation: TaskNodeInvocation,
                execution_id: str,
            ) -> TaskNodeRunResult | None:
                del invocation, execution_id
                return None

            async def cancel(self, invocation: TaskNodeInvocation) -> None:
                del invocation

        async def acquire(
            execution_id: str,
            *,
            tenant_id: str,
            hold_id: str,
        ) -> None:
            del tenant_id
            calls.append(f"acquire:{execution_id}:{hold_id}")

        async def release(
            execution_id: str,
            *,
            tenant_id: str,
            hold_id: str,
        ) -> None:
            del tenant_id
            snapshot = await state.task.tasks.graph_state(
                graph.graph_id,
                tenant_id="tenant",
            )
            assert snapshot is not None
            assert snapshot.node_states[0].status is TaskStatus.SUCCEEDED
            calls.append(f"release:{execution_id}:{hold_id}")

        launcher = LocalTaskGraphLauncher(
            state.task.tasks,
            WaitingRunner(),
            owner="worker",
            acquire_execution_hold=acquire,
            release_execution_hold=release,
        )
        await launcher.start(
            TaskGraphLaunch(graph.graph_id, principal, TaskGraphLimits())
        )

        async def wait_terminal() -> None:
            while True:
                snapshot = await state.task.tasks.graph_state(
                    graph.graph_id,
                    tenant_id="tenant",
                )
                assert snapshot is not None
                if snapshot.status is TaskStatus.SUCCEEDED:
                    return
                await asyncio.sleep(0)

        await asyncio.wait_for(wait_terminal(), 1)
        # Durable task completion precedes the owned dependency-hold cleanup.
        await launcher.shutdown()
        assert calls == [
            "acquire:execution:task:waiting-hold:node",
            "wait:execution",
            "release:execution:task:waiting-hold:node",
        ]
    finally:
        if launcher is not None:
            await launcher.shutdown()
        await state.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ("unknown", "deferred", "retry"))
async def test_bound_wait_outcome_preserves_its_waiting_classification(
    outcome: str,
) -> None:
    state = PersistenceStorage.in_memory()
    state = Persistence(state)
    await state.initialize(namespace=f"bound-wait-{outcome}", tenant_id="tenant")
    principal = Principal("workspace", "tenant", PrincipalKind.LOCAL_TRUSTED.value)
    graph = TaskGraph(f"bound-wait-{outcome}", (TaskNode("node"),))
    request = TaskGraphRequest(
        graph,
        principal,
        f"bound-wait-{outcome}-submit-0001",
    )
    await state.task.admissions.admit(
        TaskGraphAdmission.from_request(request),
        graph,
    )
    lease = await state.task.tasks.claim(
        graph.graph_id,
        "node",
        tenant_id="tenant",
        owner="worker",
        lease_seconds=30,
    )
    await state.task.tasks.handoff_execution(
        lease,
        tenant_id="tenant",
        execution_id=f"execution-{outcome}",
        occupies_concurrency=False,
    )
    retry_at = datetime.now(timezone.utc) + timedelta(hours=1)

    class BoundWaitRunner:
        def __init__(self) -> None:
            self.calls = 0
            self.reentered = asyncio.Event()

        async def run(
            self,
            invocation: TaskNodeInvocation,
            *,
            control: TaskNodeRunControl,
        ) -> TaskNodeRunResult:
            del invocation, control
            raise AssertionError("bound recovery must not start another execution")

        async def wait_bound(
            self,
            invocation: TaskNodeInvocation,
            execution_id: str,
        ) -> TaskNodeRunResult:
            del invocation
            self.calls += 1
            if outcome == "unknown":
                raise TaskNodeRunError(
                    ErrorCode.TOOL_EFFECT_UNKNOWN,
                    execution_id,
                    safe_details={"bound_wait": "unknown"},
                )
            if outcome == "retry":
                return TaskNodeRunResult(
                    canonical_sha256({"retry_at": retry_at.isoformat()}),
                    execution_id,
                    retry_at=retry_at,
                )
            if self.calls > 1:
                self.reentered.set()
                await asyncio.Event().wait()
            return TaskNodeRunResult(
                canonical_sha256({"deferred": True}),
                execution_id,
                deferred=True,
            )

        async def supply_input(
            self,
            invocation: TaskNodeInvocation,
            execution_id: str,
            value: JsonValue,
        ) -> TaskNodeRunResult:
            del invocation, execution_id, value
            raise AIError(ErrorCode.TASK_NOT_READY)

        async def resolve_effect(
            self,
            invocation: TaskNodeInvocation,
            execution_id: str,
            resolution: TaskEffectResolution,
        ) -> TaskNodeRunResult | None:
            del invocation, execution_id, resolution
            raise AIError(ErrorCode.TASK_NOT_READY)

        async def cancel(self, invocation: TaskNodeInvocation) -> None:
            del invocation

    runner = BoundWaitRunner()
    launcher = LocalTaskGraphLauncher(
        state.task.tasks,
        runner,
        owner="worker",
    )
    try:
        await launcher.start(
            TaskGraphLaunch(graph.graph_id, principal, TaskGraphLimits())
        )

        async def projected_outcome():
            while True:
                snapshot = await state.task.tasks.graph_state(
                    graph.graph_id,
                    tenant_id="tenant",
                )
                assert snapshot is not None
                node_state = snapshot.node_states[0]
                if outcome == "unknown" and node_state.status is TaskStatus.RECOVERY_REQUIRED:
                    return node_state
                if outcome == "retry" and (
                    node_state.status is TaskStatus.READY
                    and node_state.next_attempt_at == retry_at
                ):
                    return node_state
                if outcome == "deferred" and runner.calls >= 2:
                    return node_state
                await asyncio.sleep(0)

        node_state = await asyncio.wait_for(projected_outcome(), timeout=2)
        assert node_state.execution_id == f"execution-{outcome}"
        if outcome == "unknown":
            assert node_state.error_code == ErrorCode.TOOL_EFFECT_UNKNOWN.value
            assert node_state.safe_error_details["bound_wait"] == "unknown"
            assert runner.calls == 1
        elif outcome == "retry":
            assert node_state.status is TaskStatus.READY
            assert node_state.next_attempt_at == retry_at
            assert runner.calls == 1
        else:
            assert node_state.status is TaskStatus.WAITING
            assert runner.reentered.is_set()
    finally:
        await launcher.shutdown()
        await state.close()


@pytest.mark.asyncio
async def test_runtime_recovery_rejects_same_revision_task_semantic_drift(
    tmp_path: Path,
) -> None:
    storage_root = tmp_path / "semantic-drift-state"
    entered = asyncio.Event()

    async def blocking_task(context: TaskNodeContext[None]) -> JsonValue:
        del context
        entered.set()
        await asyncio.Event().wait()
        return {"value": "unreachable"}

    async def reconcile(
        context: TaskNodeContext[None],
    ) -> TaskEffectResolution:
        del context
        return TaskEffectResolution("unknown")

    first = RuntimeFeatures[None]("application")
    handler = TaskFunction[None]("example.semantic-drift", 1, blocking_task)
    first.task(
        handler,
        effect_policy="none",
        output_type=_EffectOutput,
        reconcile=reconcile,
    )
    state = PersistenceStorage.filesystem(storage_root)
    graph = TaskGraph("semantic-drift", (handler.node("node"),))

    async with Persistence.open('semantic-drift', storage=state) as persistence_5469, Runtime.attach(persistence=persistence_5469, features=model_features(_TaskTestModels(), first)) as runtime, task_engine_scope(runtime, persistence_5469) as runtime_engine:
        await start_task_graph(runtime,
            graph,
            idempotency_key="semantic-drift-run-0001",
        )
        await asyncio.wait_for(entered.wait(), 10)

    reconcile_removed = RuntimeFeatures[None]("application")
    reconcile_removed.task(
        TaskFunction[None]("example.semantic-drift", 1, blocking_task),
        effect_policy="none",
        output_type=_EffectOutput,
    )
    with pytest.raises(AIError) as reconcile_error:
        async with Persistence.open('semantic-drift', storage=PersistenceStorage.filesystem(storage_root)) as persistence_5487, Runtime.attach(persistence=persistence_5487, features=model_features(_TaskTestModels(), reconcile_removed)) as runtime, task_engine_scope(runtime, persistence_5487) as runtime_engine:
            await task_engine(runtime).recover_pending()
    assert reconcile_error.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
    assert reconcile_error.value.safe_details["reason"] == "task_reconcile_changed"

    async def cancel_task(context: TaskNodeContext[None]) -> None:
        del context

    cancel_added = RuntimeFeatures[None]("application")
    cancel_added.task(
        TaskFunction[None]("example.semantic-drift", 1, blocking_task),
        effect_policy="none",
        output_type=_EffectOutput,
        cancel=cancel_task,
        reconcile=reconcile,
    )
    with pytest.raises(AIError) as cancel_error:
        async with Persistence.open('semantic-drift', storage=PersistenceStorage.filesystem(storage_root)) as persistence_5508, Runtime.attach(persistence=persistence_5508, features=model_features(_TaskTestModels(), cancel_added)) as runtime, task_engine_scope(runtime, persistence_5508) as runtime_engine:
            await task_engine(runtime).recover_pending()
    assert cancel_error.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
    assert cancel_error.value.safe_details["reason"] == "task_cancel_changed"

    output_changed = RuntimeFeatures[None]("application")
    output_changed.task(
        TaskFunction[None]("example.semantic-drift", 1, blocking_task),
        effect_policy="none",
        output_type=_ChangedEffectOutput,
        reconcile=reconcile,
    )
    with pytest.raises(AIError) as output_error:
        async with Persistence.open('semantic-drift', storage=PersistenceStorage.filesystem(storage_root)) as persistence_5525, Runtime.attach(persistence=persistence_5525, features=model_features(_TaskTestModels(), output_changed)) as runtime, task_engine_scope(runtime, persistence_5525) as runtime_engine:
            await task_engine(runtime).recover_pending()
    assert output_error.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
    assert output_error.value.safe_details["reason"] == "task_output_contract_changed"

    output_removed = RuntimeFeatures[None]("application")
    output_removed.task(
        TaskFunction[None]("example.semantic-drift", 1, blocking_task),
        effect_policy="none",
        reconcile=reconcile,
    )
    with pytest.raises(AIError) as removed_output_error:
        async with Persistence.open('semantic-drift', storage=PersistenceStorage.filesystem(storage_root)) as persistence_5541, Runtime.attach(persistence=persistence_5541, features=model_features(_TaskTestModels(), output_removed)) as runtime, task_engine_scope(runtime, persistence_5541) as runtime_engine:
            await task_engine(runtime).recover_pending()
    assert removed_output_error.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
    assert (
        removed_output_error.value.safe_details["reason"]
        == "task_output_contract_changed"
    )

    effect_changed = RuntimeFeatures[None]("application")
    effect_changed.task(
        TaskFunction[None]("example.semantic-drift", 1, blocking_task),
        effect_policy="replay_safe",
        output_type=_EffectOutput,
        reconcile=reconcile,
    )
    with pytest.raises(AIError) as effect_error:
        async with Persistence.open('semantic-drift', storage=PersistenceStorage.filesystem(storage_root)) as persistence_5561, Runtime.attach(persistence=persistence_5561, features=model_features(_TaskTestModels(), effect_changed)) as runtime, task_engine_scope(runtime, persistence_5561) as runtime_engine:
            await task_engine(runtime).recover_pending()
    assert effect_error.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
    assert effect_error.value.safe_details["reason"] == "task_effect_changed"


@pytest.mark.asyncio
async def test_runtime_shutdown_leaves_running_custom_task_recoverable(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    workspace = Workspace.load(workspace_root)
    storage_root = tmp_path / "state"
    entered = asyncio.Event()
    cancelled = asyncio.Event()

    async def blocking_task(context: TaskNodeContext[None]) -> JsonValue:
        del context
        entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise

    application = workspace_features("application", workspace=workspace, features_type=RuntimeFeatures)
    handler = TaskFunction[None]("example.block", 1, blocking_task)
    application.task(handler, effect_policy="none")
    application.agent(
        "default",
        model="default",
        allow_tools=(),
        allow_skills=(),
        allow_subagents=(),
    )
    state = PersistenceStorage.filesystem(storage_root)
    graph = TaskGraph("shutdown-graph", (handler.node("node"),))

    async with Persistence.open('default', storage=state) as persistence_5604, Runtime.attach(persistence=persistence_5604, features=model_features(_TaskTestModels(), application)) as runtime, task_engine_scope(runtime, persistence_5604) as runtime_engine:
        await start_task_graph(runtime,
            graph,
            idempotency_key="shutdown-graph-run-0001",
        )
        await asyncio.wait_for(entered.wait(), 10)
        snapshot = await task_graph_state(runtime,
            graph.graph_id,
            principal=runtime.default_principal,
        )
        assert snapshot.node_states[0].status is TaskStatus.RUNNING

    assert cancelled.is_set()
    probe = PersistenceStorage.filesystem(storage_root)
    probe = Persistence(probe)
    await probe.initialize(namespace="default", tenant_id="default")
    try:
        snapshot = await probe.task.tasks.graph_state(
            graph.graph_id,
            tenant_id="default",
        )
        assert snapshot is not None
        assert snapshot.node_states[0].status is TaskStatus.RUNNING
        assert snapshot.node_states[0].owner is not None
        assert snapshot.node_states[0].lease_expires_at is not None
    finally:
        await probe.close()


@pytest.mark.asyncio
async def test_any_succeeded_barrier_survives_runtime_restart(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from linktools.ai.persistence import _filesystem
    from linktools.ai.task import _local

    storage_root = tmp_path / "state"
    held_started = asyncio.Event()
    held_cancelled = asyncio.Event()
    success_calls = 0
    held_calls = 0
    join_calls = 0
    joined: dict[str, object] = {}

    async def succeed(_context: TaskNodeContext[None]) -> JsonValue:
        nonlocal success_calls
        success_calls += 1
        return {"value": "committed"}

    async def hold_then_fail(_context: TaskNodeContext[None]) -> JsonValue:
        nonlocal held_calls
        held_calls += 1
        if held_calls == 1:
            held_started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                held_cancelled.set()
                raise
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID)

    async def join(context: TaskNodeContext[None]) -> JsonValue:
        nonlocal join_calls
        join_calls += 1
        joined["states"] = {
            node_id: state.status
            for node_id, state in context.dependency_states.items()
        }
        joined["results"] = set(context.dependencies)
        return "joined"

    application = RuntimeFeatures[None]("application")
    success = TaskFunction[None]("example.restart-success", 1, succeed)
    held = TaskFunction[None]("example.restart-held-failure", 1, hold_then_fail)
    joiner = TaskFunction[None]("example.restart-join", 1, join)
    application.task(success, effect_policy="none")
    application.task(held, effect_policy="none")
    application.task(joiner, effect_policy="none")
    graph = TaskGraph(
        "any-succeeded-restart",
        (
            success.node("success"),
            held.node("held", failure_policy="isolate"),
            joiner.node(
                "join",
                dependencies=("success", "held"),
                dependency_policy="any_succeeded",
            ),
        ),
    )

    async with Persistence.open('any-succeeded-restart', storage=PersistenceStorage.filesystem(storage_root)) as persistence_5700, Runtime.attach(persistence=persistence_5700, features=model_features(_TaskTestModels(), application)) as runtime, task_engine_scope(runtime, persistence_5700) as runtime_engine:
        first_run = await runtime_engine.with_definitions(success, held, joiner).start(
            graph,
            idempotency_key="any-succeeded-restart-0001",
         principal=runtime.default_principal)
        await asyncio.wait_for(held_started.wait(), 10)
        deadline = asyncio.get_running_loop().time() + 10
        while True:
            first_state = await first_run.state()
            state_by_id = {node.node_id: node for node in first_state.node_states}
            if (
                state_by_id["success"].status is TaskStatus.SUCCEEDED
                and state_by_id["held"].status is TaskStatus.RUNNING
            ):
                break
            if asyncio.get_running_loop().time() >= deadline:
                raise AssertionError("success branch did not commit before shutdown")
            await asyncio.sleep(0.001)
        committed_result = state_by_id["success"].result_digest
        committed_execution = state_by_id["success"].execution_id
        assert committed_result is not None
        assert committed_execution is not None
        assert state_by_id["join"].status is TaskStatus.PENDING
        assert join_calls == 0

    assert held_cancelled.is_set()
    expired_at = datetime.now(timezone.utc) + timedelta(seconds=_local._LEASE_SECONDS + 1)

    class AfterLeaseExpiry:
        @classmethod
        def now(cls, tz: timezone | None = None) -> datetime:
            return expired_at.astimezone(tz)

    # Expire the abandoned lease without shortening the live commit window.
    monkeypatch.setattr(_local, "datetime", AfterLeaseExpiry)
    monkeypatch.setattr(_filesystem, "datetime", AfterLeaseExpiry)

    async with Persistence.open('any-succeeded-restart', storage=PersistenceStorage.filesystem(storage_root)) as persistence_5741, Runtime.attach(persistence=persistence_5741, features=model_features(_TaskTestModels(), application)) as runtime, task_engine_scope(runtime, persistence_5741) as runtime_engine:
        bound_tasks = runtime_engine.with_definitions(success, held, joiner)
        await bound_tasks.recover_pending()
        recovered_run = await bound_tasks.get(graph.graph_id)
        result = (await recovered_run.wait(timeout_seconds=10)).result
        final_state = await recovered_run.state()

    assert result.wait_status is TaskStatus.RECOVERY_REQUIRED
    assert success_calls == 1
    assert held_calls == 1
    assert join_calls == 0
    assert joined == {}
    final_by_id = {node.node_id: node for node in final_state.node_states}
    assert final_by_id["success"].status is TaskStatus.SUCCEEDED
    assert final_by_id["success"].result_digest == committed_result
    assert final_by_id["success"].execution_id == committed_execution
    assert final_by_id["held"].status is TaskStatus.RECOVERY_REQUIRED
    assert final_by_id["join"].status is TaskStatus.PENDING
