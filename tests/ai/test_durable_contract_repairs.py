#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Focused regressions for durable composition and close ownership."""

import asyncio
from types import SimpleNamespace

import pytest
from linktools.ai.agent import AgentCompiler, bind_output
from linktools.ai.capability import RuntimeCapability
from linktools.ai.core import Principal, ResourceKind, ResourceRef, TaskStatus
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.model import ModelRegistry
from linktools.ai.runtime._factory import _RuntimeCloseCoordinator
from linktools.ai.runtime._local import LocalExecutionBackend
from linktools.ai.runtime.state._durability import (
    DurableCommitState,
    run_durable_commit,
)
from linktools.ai.runtime.state._materializer import _RuntimeObjectRouter
from linktools.ai.runtime.state._plan import RuntimeDomain
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


class _RegisteredCapability(AbstractCapability[None]):
    @classmethod
    def get_serialization_name(cls) -> "str | None":
        return "durable-contract-capability"

    @classmethod
    def from_spec(cls, **kwargs: object) -> "_RegisteredCapability":
        del kwargs
        return cls()


class _ConflictingCapability(AbstractCapability[None]):
    @classmethod
    def get_serialization_name(cls) -> "str | None":
        return "durable-contract-capability"

    @classmethod
    def from_spec(cls, **kwargs: object) -> "_ConflictingCapability":
        del kwargs
        return cls()


def _compiler(
    *,
    outputs: tuple[object, ...] = (),
    capabilities: tuple[object, ...] = (),
    runtime_capability_types: tuple[type[AbstractCapability[None]], ...] = (),
) -> AgentCompiler:
    return AgentCompiler(
        model_resolver=ModelRegistry.openai(model="gpt-test").snapshot(),
        capabilities=capabilities,
        runtime_fingerprint="a" * 64,
        outputs=outputs,
        runtime_capability_types=runtime_capability_types,
    )


def _spec() -> AgentSpec:
    return AgentSpec("durable-contract", model="default")


def test_custom_output_restore_uses_frozen_composition_registry() -> None:
    compiler = _compiler(
        outputs=(bind_output(_RegisteredOutput, schema_id="durable.output"),),
    )
    definition = compiler.compile(_spec())
    binding = compiler.bind(definition, output=_RegisteredOutput)
    snapshot = binding.snapshot

    fresh = _compiler(
        outputs=(bind_output(_RegisteredOutput, schema_id="durable.output"),),
    )
    restored = fresh.restore(snapshot)
    assert restored.snapshot == snapshot

    unregistered = _compiler()
    before = dict(unregistered._outputs_by_type)
    with pytest.raises(AIError) as bind_error:
        unregistered.bind(unregistered.compile(_spec()), output=_RegisteredOutput)
    assert bind_error.value.code is ErrorCode.OUTPUT_CONTRACT_INVALID
    assert dict(unregistered._outputs_by_type) == before

    with pytest.raises(AIError) as restore_error:
        unregistered.restore(snapshot)
    assert restore_error.value.code is ErrorCode.AGENT_DEFINITION_UNAVAILABLE


def test_local_runtime_capability_restore_requires_frozen_type_registration() -> None:
    capability = RuntimeCapability.from_spec(
        "local",
        _RegisteredCapability,
        config={"mode": "strict"},
    )
    compiler = _compiler(runtime_capability_types=(_RegisteredCapability,))
    before = dict(compiler._runtime_capability_types)
    definition = compiler.compile(_spec(), capabilities=(capability,))
    binding = compiler.bind(definition)
    assert dict(compiler._runtime_capability_types) == before

    fresh = _compiler(runtime_capability_types=(_RegisteredCapability,))
    assert fresh.restore(binding.snapshot).snapshot == binding.snapshot

    unregistered = _compiler()
    with pytest.raises(AIError) as missing_error:
        unregistered.compile(_spec(), capabilities=(capability,))
    assert missing_error.value.code is ErrorCode.CAPABILITY_RESOLUTION_INVALID

    conflicting = _compiler(runtime_capability_types=(_ConflictingCapability,))
    with pytest.raises(AIError) as conflicting_error:
        conflicting.compile(_spec(), capabilities=(capability,))
    assert conflicting_error.value.code is ErrorCode.CAPABILITY_RESOLUTION_INVALID


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


@pytest.mark.asyncio
async def test_durable_commit_cancellation_keeps_operation_owned_until_done() -> None:
    started = asyncio.Event()
    release = asyncio.Event()
    owner: set[asyncio.Task[object]] = set()

    async def operation() -> str:
        started.set()
        await release.wait()
        return "committed"

    async def readback() -> object:
        raise AssertionError("readback is not used for a pending operation")

    caller = asyncio.create_task(
        run_durable_commit(operation, readback, background_tasks=owner)
    )
    await started.wait()
    caller.cancel()
    result = await caller
    assert result.state is DurableCommitState.UNRESOLVED
    assert result.cancelled
    assert len(owner) == 1
    owned = next(iter(owner))
    assert not owned.done()

    release.set()
    await owned
    await asyncio.sleep(0)
    assert owner == set()


@pytest.mark.asyncio
async def test_durable_commit_cancellation_owns_failed_readback_until_done(
    caplog: pytest.LogCaptureFixture,
) -> None:
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
    result = await caller
    assert result.state is DurableCommitState.UNRESOLVED
    assert len(owner) == 1
    readback_task = next(iter(owner))

    release.set()
    await asyncio.gather(readback_task, return_exceptions=True)
    await asyncio.sleep(0)
    assert owner == set()
    assert any(
        "detached durable commit readback failed" in record.message
        for record in caplog.records
    )


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
    router = _RuntimeObjectRouter({RuntimeDomain.EXECUTION: store})
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
async def test_runtime_object_preflight_rejects_pending_sql_work(tmp_path) -> None:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'objects.db'}")
    store = SqlObjectStore(engine)
    router = _RuntimeObjectRouter({RuntimeDomain.EXECUTION: store})
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

    with pytest.raises(AIError) as error:
        await backend.close()
    assert error.value.code is ErrorCode.STORAGE_RECOVERY_REQUIRED

    release.set()
    await task
    await asyncio.sleep(0)
    await backend.close()


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
