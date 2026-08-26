#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Focused regressions for durable composition and close ownership."""

import asyncio
from dataclasses import replace
from types import SimpleNamespace

import pytest
from linktools.ai.agent import AgentCompiler, bind_output
from linktools.ai.capability import RuntimeCapability
from linktools.ai.core import ExecutionStatus, Principal, ResourceKind, ResourceRef, TaskStatus
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.model import ModelRegistry
from linktools.ai.runtime._factory import _RuntimeCloseCoordinator
from linktools.ai.runtime._execution import DefaultExecutionService
from linktools.ai.runtime._local import LocalExecutionBackend
from linktools.ai.runtime.state._durability import (
    CommitObservation,
    DurableCommitState,
    run_durable_commit,
)
from linktools.ai.runtime.state._materializer import _RuntimeObjectRouter
from linktools.ai.runtime.state._plan import RuntimeDomain
from linktools.ai.runtime.state._retention import RuntimeRetentionController
from linktools.ai.runtime.state._steps import RuntimeStepStore
from linktools.ai.storage import FilesystemObjectStore, SqlObjectStore
from linktools.ai.task._api import open_local_task_api
from linktools.ai.task._service_impl import DefaultTaskService
from linktools.ai.spec import AgentSpec
from pydantic import BaseModel
from pydantic_ai.capabilities import AbstractCapability
from sqlalchemy.ext.asyncio import create_async_engine


class _RegisteredOutput(BaseModel):
    value: str


class _RecursiveOutput(BaseModel):
    child: "_RecursiveOutput | None" = None


_RecursiveOutput.model_rebuild()


class _RegisteredCapability(AbstractCapability[None]):
    @classmethod
    def get_serialization_name(cls) -> "str | None":
        return "durable-contract-capability"

    @classmethod
    def from_spec(cls, **kwargs: object) -> "_RegisteredCapability":
        del kwargs
        return cls()


def _compiler(
    *,
    capabilities: tuple[object, ...] = (),
) -> AgentCompiler:
    return AgentCompiler(
        model_resolver=ModelRegistry.openai(model="gpt-test").snapshot(),
        capabilities=capabilities,
        runtime_fingerprint="a" * 64,
    )


def _spec() -> AgentSpec:
    return AgentSpec("durable-contract", model="default")


def test_custom_output_restore_uses_persisted_schema() -> None:
    compiler = _compiler()
    definition = compiler.compile(_spec())
    binding = compiler.bind(definition, output=_RegisteredOutput)
    snapshot = binding.snapshot
    assert snapshot.output_schema_definition == binding.output_binding.schema_definition
    assert not hasattr(compiler, "_outputs_by_type")

    fresh = _compiler()
    restored = fresh.restore(snapshot)
    assert restored.snapshot == snapshot
    assert restored.output_type is not _RegisteredOutput
    assert not hasattr(fresh, "_outputs_by_type")


def test_custom_output_restore_rejects_missing_or_tampered_schema() -> None:
    compiler = _compiler()
    binding = compiler.bind(compiler.compile(_spec()), output=_RegisteredOutput)
    snapshot = binding.snapshot
    fresh = _compiler()

    historical = replace(snapshot, output_schema_definition=None)
    with pytest.raises(AIError) as missing_error:
        fresh.restore(historical)
    assert missing_error.value.code is ErrorCode.AGENT_DEFINITION_UNAVAILABLE

    tampered_schema = dict(snapshot.output_schema_definition or {})
    tampered_schema["title"] = "TamperedOutput"
    tampered = replace(snapshot, output_schema_definition=tampered_schema)
    with pytest.raises(AIError) as tampered_error:
        fresh.restore(tampered)
    assert tampered_error.value.code is ErrorCode.AGENT_DEFINITION_UNAVAILABLE


def test_custom_output_rejects_non_durable_schema_at_bind_time() -> None:
    with pytest.raises(AIError) as error:
        bind_output(_RecursiveOutput)
    assert error.value.code is ErrorCode.OUTPUT_CONTRACT_INVALID
    assert error.value.safe_details == {"reason": "output_schema_not_durable"}


def test_local_runtime_capability_restore_uses_persisted_locator() -> None:
    capability = RuntimeCapability.from_spec(
        "local",
        _RegisteredCapability,
        config={"mode": "strict"},
    )
    compiler = _compiler()
    definition = compiler.compile(_spec(), capabilities=(capability,))
    binding = compiler.bind(definition)
    assert not hasattr(compiler, "_runtime_capability_types")

    fresh = _compiler()
    assert fresh.restore(binding.snapshot).snapshot == binding.snapshot
    assert not hasattr(fresh, "_runtime_capability_types")

    descriptor = dict(binding.snapshot.local_runtime_capability_descriptors[0])
    descriptor["restore_locator"] = {
        "module": "missing.historical.module",
        "qualname": "MovedCapability",
    }
    historical = replace(
        binding.snapshot,
        local_runtime_capability_descriptors=(descriptor,),
    )
    with pytest.raises(AIError) as missing_error:
        fresh.restore(historical)
    assert missing_error.value.code is ErrorCode.AGENT_DEFINITION_UNAVAILABLE


def test_global_runtime_capability_is_restored_from_composition() -> None:
    capability = RuntimeCapability.from_spec(
        "global",
        _RegisteredCapability,
        config={"mode": "global"},
    )
    compiler = _compiler(capabilities=(capability,))
    binding = compiler.bind(compiler.compile(_spec()))
    fresh = _compiler(capabilities=(capability,))
    assert fresh.restore(binding.snapshot).snapshot == binding.snapshot

    descriptor = dict(binding.snapshot.global_runtime_capability_descriptors[0])
    descriptor["restore_locator"] = {
        "module": "missing.historical.module",
        "qualname": "MovedCapability",
    }
    historical = replace(
        binding.snapshot,
        global_runtime_capability_descriptors=(descriptor,),
    )
    assert fresh.restore(historical).snapshot == historical


@pytest.mark.asyncio
async def test_durable_commit_cancellation_settles_operation_before_returning() -> None:
    started = asyncio.Event()
    release = asyncio.Event()
    owner: set[asyncio.Task[object]] = set()

    async def operation() -> str:
        started.set()
        await release.wait()
        return "committed"

    async def readback() -> object:
        raise AssertionError("readback is not used after a committed operation")

    caller = asyncio.create_task(
        run_durable_commit(operation, readback, background_tasks=owner)
    )
    await started.wait()
    caller.cancel()
    await asyncio.sleep(0)
    assert not caller.done()
    assert len(owner) == 1

    release.set()
    result = await caller
    assert result.state is DurableCommitState.COMMITTED
    assert result.value == "committed"
    assert result.cancelled
    assert owner == set()


@pytest.mark.asyncio
async def test_durable_commit_cancellation_settles_readback_before_unknown() -> None:
    started = asyncio.Event()
    release = asyncio.Event()
    owner: set[asyncio.Task[object]] = set()

    async def operation() -> None:
        raise RuntimeError("commit failed")

    async def readback() -> object:
        started.set()
        await release.wait()
        raise RuntimeError("readback failed")

    caller = asyncio.create_task(
        run_durable_commit(operation, readback, background_tasks=owner)
    )
    await started.wait()
    caller.cancel()
    await asyncio.sleep(0)
    assert not caller.done()
    assert len(owner) == 1

    release.set()
    result = await caller
    assert result.state is DurableCommitState.UNRESOLVED
    assert isinstance(result.error, RuntimeError)
    assert result.cancelled
    assert owner == set()


@pytest.mark.asyncio
async def test_durable_commit_cancellation_accepts_committed_readback() -> None:
    started = asyncio.Event()
    release = asyncio.Event()
    owner: set[asyncio.Task[object]] = set()

    async def operation() -> None:
        raise RuntimeError("commit response lost")

    async def readback() -> CommitObservation[str]:
        started.set()
        await release.wait()
        return CommitObservation(DurableCommitState.COMMITTED, value="reconciled")

    caller = asyncio.create_task(
        run_durable_commit(operation, readback, background_tasks=owner)
    )
    await started.wait()
    caller.cancel()
    await asyncio.sleep(0)
    assert not caller.done()

    release.set()
    result = await caller
    assert result.state is DurableCommitState.COMMITTED
    assert result.value == "reconciled"
    assert result.cancelled
    assert owner == set()


def _step_store_for_preflight() -> RuntimeStepStore:
    store = object.__new__(RuntimeStepStore)
    store._background_tasks = set()
    store._durability_flights = {}
    store._terminal_seals = {}
    store._preflight = False
    return store


@pytest.mark.asyncio
async def test_step_preflight_rejects_flights_tasks_and_terminal_seals() -> None:
    store = _step_store_for_preflight()
    store._durability_flights["run"] = object()
    with pytest.raises(AIError) as flight_error:
        await store.preflight_close()
    assert flight_error.value.code is ErrorCode.STORAGE_RECOVERY_REQUIRED
    assert not store._preflight

    store._durability_flights.clear()
    store._terminal_seals["run"] = object()
    with pytest.raises(AIError) as seal_error:
        await store.preflight_close()
    assert seal_error.value.code is ErrorCode.STORAGE_RECOVERY_REQUIRED
    store._terminal_seals.clear()

    release = asyncio.Event()
    task = asyncio.create_task(release.wait())
    store._background_tasks.add(task)
    with pytest.raises(AIError) as task_error:
        await store.preflight_close()
    assert task_error.value.code is ErrorCode.STORAGE_RECOVERY_REQUIRED
    assert not store._preflight
    assert task_error.value.safe_details["pending_tasks"] == 1
    release.set()
    await task
    await store.preflight_close()
    assert store._preflight


@pytest.mark.asyncio
async def test_runtime_object_preflight_rejects_pending_filesystem_work(
    tmp_path,
) -> None:
    store = FilesystemObjectStore(tmp_path)
    router = _RuntimeObjectRouter(
        {RuntimeDomain.EXECUTION: store},
        close_guard_stores=(store,),
    )
    release = asyncio.Event()
    task = asyncio.create_task(release.wait())
    store._background_tasks.add(task)
    with pytest.raises(AIError) as error:
        await router.preflight_close()
    assert error.value.code is ErrorCode.STORAGE_RECOVERY_REQUIRED
    release.set()
    await task
    await router.preflight_close()


@pytest.mark.asyncio
async def test_runtime_object_preflight_ignores_external_filesystem_work(
    tmp_path,
) -> None:
    store = FilesystemObjectStore(tmp_path)
    router = _RuntimeObjectRouter(
        {RuntimeDomain.EXECUTION: store},
        close_guard_stores=(),
    )
    release = asyncio.Event()
    task = asyncio.create_task(release.wait())
    store._background_tasks.add(task)
    try:
        await router.preflight_close()
    finally:
        release.set()
        await task


@pytest.mark.asyncio
async def test_execution_retention_requires_runtime_release_before_cleanup() -> None:
    calls: list[str] = []

    async def runtime_release(execution_id: str, *, tenant_id: str) -> None:
        calls.append(f"runtime:{execution_id}:{tenant_id}")
        raise AIError(ErrorCode.STORAGE_CONFLICT)

    async def execution_get(execution_id: str, *, tenant_id: str) -> object:
        calls.append(f"execution:{execution_id}:{tenant_id}")
        return object()

    controller = object.__new__(RuntimeRetentionController)
    controller._execution_runtime_release = runtime_release
    controller._execution = SimpleNamespace(
        executions=SimpleNamespace(get=execution_get),
    )

    with pytest.raises(AIError) as error:
        await controller.release_execution_handoff("execution", tenant_id="tenant")

    assert error.value.code is ErrorCode.STORAGE_CONFLICT
    assert calls == ["runtime:execution:tenant"]


@pytest.mark.asyncio
async def test_runtime_object_preflight_rejects_pending_sql_work(tmp_path) -> None:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'objects.db'}")
    store = SqlObjectStore(engine)
    router = _RuntimeObjectRouter(
        {RuntimeDomain.EXECUTION: store},
        close_guard_stores=(store,),
    )
    release = asyncio.Event()
    task = asyncio.create_task(release.wait())
    store._background_tasks.add(task)
    try:
        with pytest.raises(AIError) as error:
            await router.preflight_close()
        assert error.value.code is ErrorCode.STORAGE_RECOVERY_REQUIRED
        release.set()
        await task
        await router.preflight_close()
    finally:
        await store._context.close()
        await engine.dispose()


@pytest.mark.asyncio
async def test_runtime_object_preflight_ignores_external_sql_work(tmp_path) -> None:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'objects.db'}")
    store = SqlObjectStore(engine)
    router = _RuntimeObjectRouter(
        {RuntimeDomain.EXECUTION: store},
        close_guard_stores=(),
    )
    release = asyncio.Event()
    task = asyncio.create_task(release.wait())
    store._background_tasks.add(task)
    try:
        await router.preflight_close()
    finally:
        release.set()
        await task
        await store._context.close()
        await engine.dispose()


@pytest.mark.asyncio
async def test_runtime_close_coordinator_does_not_close_lower_resources_after_preflight(
) -> None:
    calls: list[str] = []
    blocked = True

    async def preflight() -> None:
        calls.append("preflight")
        if blocked:
            raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)

    async def lower() -> None:
        calls.append("lower")

    coordinator = _RuntimeCloseCoordinator((preflight, lower))
    with pytest.raises(AIError) as error:
        await coordinator.close()
    assert error.value.code is ErrorCode.STORAGE_RECOVERY_REQUIRED
    assert calls == ["preflight"]

    blocked = False
    await coordinator.close()
    assert calls == ["preflight", "preflight", "lower"]


@pytest.mark.asyncio
async def test_local_execution_close_rejects_pending_command_owned_work() -> None:
    backend = object.__new__(LocalExecutionBackend)
    backend._accepting = True
    backend._tasks = {}
    backend._executor = SimpleNamespace(pending_background_tasks=())
    backend._subagent_dispatcher = None
    backend._captured_usage = {}
    backend._terminal_events = {}
    backend._worker_failures = {}
    backend._pending_audit_events = {}
    backend._pending_audit_locks = {}
    release = asyncio.Event()
    task = asyncio.create_task(release.wait())
    backend._checkpoint_tasks = {task}
    backend._execution_durable_tasks = {"execution": {task}}

    with pytest.raises(AIError) as error:
        await backend.close()
    assert error.value.code is ErrorCode.STORAGE_RECOVERY_REQUIRED
    assert error.value.safe_details["pending_checkpoint_tasks"] == 1
    assert error.value.safe_details["pending_execution_tasks"] == 1
    assert error.value.safe_details["pending_background_tasks"] == 1

    release.set()
    await task
    await asyncio.sleep(0)
    await backend.close()
    assert backend._execution_durable_tasks == {}


@pytest.mark.asyncio
async def test_runtime_release_waits_for_execution_scoped_durable_task() -> None:
    backend = object.__new__(LocalExecutionBackend)
    backend._tenant_id = "tenant"
    backend._tasks = {}
    backend._terminal_events = {}
    backend._worker_failures = {}
    backend._captured_usage = {}
    backend._pending_audit_events = {}
    backend._pending_audit_locks = {}
    release = asyncio.Event()
    task = asyncio.create_task(release.wait())
    backend._execution_durable_tasks = {"execution": {task}}

    with pytest.raises(AIError) as error:
        await backend.release_runtime_execution("execution", tenant_id="tenant")
    assert error.value.code is ErrorCode.STORAGE_CONFLICT

    release.set()
    await task
    await backend.release_runtime_execution("execution", tenant_id="tenant")
    assert backend._execution_durable_tasks == {}


@pytest.mark.asyncio
async def test_execution_wait_rechecks_after_local_worker_quiescence() -> None:
    started = asyncio.Event()
    release = asyncio.Event()

    class Waiter:
        owned = True

        def owns_execution(self, execution_id: str, *, tenant_id: str) -> bool:
            del execution_id, tenant_id
            return self.owned

        async def wait_terminal(self, execution_id: str, *, tenant_id: str) -> None:
            del execution_id, tenant_id
            started.set()
            await release.wait()
            self.owned = False

    service = object.__new__(DefaultExecutionService)
    service._local_waiter = Waiter()
    service._backend = None

    async def inspect(execution_id: str, *, principal: Principal) -> object:
        del execution_id, principal
        return SimpleNamespace(status=ExecutionStatus.SUCCEEDED)

    async def result(execution_id: str, *, principal: Principal) -> str:
        del execution_id, principal
        return "terminal"

    service.inspect = inspect
    service.result = result
    wait = DefaultExecutionService.wait.__wrapped__
    task = asyncio.create_task(
        wait(
            service,
            "execution",
            principal=Principal("principal", "tenant", "service"),
            timeout_seconds=1,
        )
    )
    await started.wait()
    assert not task.done()
    release.set()
    assert await task == "terminal"


class _RunningTaskRepository:
    async def get_header(
        self,
        graph_id: str,
        *,
        tenant_id: str,
    ) -> ResourceRef:
        return ResourceRef(ResourceKind.TASK_GRAPH, graph_id, tenant_id)

    async def reconcile_graph(
        self,
        graph_id: str,
        *,
        tenant_id: str,
    ) -> object:
        del graph_id, tenant_id
        return SimpleNamespace(status=TaskStatus.RUNNING)


class _AllowAuthorization:
    async def authorize(self, *args: object, **kwargs: object) -> None:
        del args, kwargs


class _LostTaskWaiter:
    def owns_graph(self, graph_id: str, *, tenant_id: str) -> bool:
        del graph_id, tenant_id
        return False

    async def wait_graph_activity(self, graph_id: str, *, tenant_id: str) -> None:
        del graph_id, tenant_id
        raise AssertionError("lost owner must not wait for local activity")


class _TerminalTaskRepository:
    async def get_header(
        self,
        graph_id: str,
        *,
        tenant_id: str,
    ) -> ResourceRef:
        return ResourceRef(ResourceKind.TASK_GRAPH, graph_id, tenant_id)

    async def reconcile_graph(
        self,
        graph_id: str,
        *,
        tenant_id: str,
    ) -> object:
        return SimpleNamespace(
            graph_id=graph_id,
            tenant_id=tenant_id,
            status=TaskStatus.SUCCEEDED,
        )


class _TerminalTaskWaiter:
    def __init__(self, error: AIError | None = None) -> None:
        self.error = error
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.owned = True

    def owns_graph(self, graph_id: str, *, tenant_id: str) -> bool:
        del graph_id, tenant_id
        return self.owned

    async def wait_graph_activity(self, graph_id: str, *, tenant_id: str) -> None:
        del graph_id, tenant_id
        self.started.set()
        if self.error is not None:
            raise self.error
        await self.release.wait()
        self.owned = False


def _terminal_task_service(waiter: _TerminalTaskWaiter) -> DefaultTaskService:
    service = DefaultTaskService(
        SimpleNamespace(tasks=_TerminalTaskRepository()),
        _AllowAuthorization(),
        local_waiter=waiter,
    )

    async def result(view: object, tenant_id: str) -> object:
        del tenant_id
        return view

    async def request_release(graph_id: str, tenant_id: str) -> None:
        del graph_id, tenant_id

    service._result = result
    service._request_graph_release = request_release
    return service


@pytest.mark.asyncio
async def test_task_wait_rechecks_terminal_after_local_scheduler_quiescence() -> None:
    waiter = _TerminalTaskWaiter()
    service = _terminal_task_service(waiter)
    task = asyncio.create_task(
        service.wait_graph(
            "graph",
            principal=Principal("principal", "tenant", "service"),
            timeout_seconds=1,
        )
    )
    await waiter.started.wait()
    assert not task.done()
    waiter.release.set()
    result = await task
    assert result.status is TaskStatus.SUCCEEDED


@pytest.mark.asyncio
async def test_task_wait_prefers_terminal_truth_after_scheduler_failure() -> None:
    waiter = _TerminalTaskWaiter(AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED))
    service = _terminal_task_service(waiter)
    result = await service.wait_graph(
        "graph",
        principal=Principal("principal", "tenant", "service"),
        timeout_seconds=1,
    )
    assert result.status is TaskStatus.SUCCEEDED


@pytest.mark.asyncio
async def test_task_wait_fails_immediately_after_local_owner_is_lost() -> None:
    persistence = SimpleNamespace(tasks=_RunningTaskRepository())
    service = DefaultTaskService(
        persistence,
        _AllowAuthorization(),
        local_waiter=_LostTaskWaiter(),
    )
    with pytest.raises(AIError) as error:
        await service.wait_graph(
            "graph",
            principal=Principal("principal", "tenant", "service"),
            timeout_seconds=1,
        )
    assert error.value.code is ErrorCode.STORAGE_RECOVERY_REQUIRED


@pytest.mark.asyncio
async def test_task_service_preflight_blocks_pending_detached_finalizer() -> None:
    service = DefaultTaskService(SimpleNamespace(), _AllowAuthorization())
    release = asyncio.Event()
    task = asyncio.create_task(release.wait())
    service._detached_finalizers.add(task)
    with pytest.raises(AIError) as error:
        await service.preflight_close()
    assert error.value.code is ErrorCode.STORAGE_RECOVERY_REQUIRED
    release.set()
    await task
    await asyncio.sleep(0)
    await service.preflight_close()


@pytest.mark.asyncio
async def test_standalone_task_api_preflights_before_launcher_shutdown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Launcher:
        shutdown_called = False

        def __init__(self, tasks: object, runner: object, *, owner: str) -> None:
            del tasks, runner, owner

        async def shutdown(self) -> None:
            self.shutdown_called = True

    class Service:
        def __init__(self, persistence: object, authorization: object, launcher: object) -> None:
            del persistence, authorization
            self.launcher = launcher

        async def preflight_close(self) -> None:
            raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)

    import linktools.ai.task._api as api_module

    monkeypatch.setattr(api_module, "LocalTaskGraphLauncher", Launcher)
    monkeypatch.setattr(api_module, "DefaultTaskService", Service)
    with pytest.raises(AIError) as error:
        async with open_local_task_api(
            SimpleNamespace(tasks=object()),
            _AllowAuthorization(),
            runner=object(),
            owner="owner",
        ):
            raise AssertionError("context must fail during preflight")
    assert error.value.code is ErrorCode.STORAGE_RECOVERY_REQUIRED
