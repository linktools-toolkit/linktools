#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from linktools.ai.persistence import Persistence
"""Graceful shutdown preserves checkpointed conversation execution identity."""
from ._runtime_test_helpers import model_features

import asyncio
import getpass
from collections.abc import AsyncIterator, Callable
from pathlib import Path

import pytest
import pytest_asyncio
from pydantic_ai.models.function import FunctionModel

from linktools.ai.core import ExecutionStatus
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime import Runtime
from linktools.ai.persistence import PersistenceStorage

from ._runtime_test_helpers import _Models
from .test_terminal_commit_shared_runtime import _group


@pytest_asyncio.fixture(params=("sqlite", "filesystem", "postgresql", "mysql"))
async def handoff_storage(request: pytest.FixtureRequest, tmp_path: Path) -> AsyncIterator[Callable[[], PersistenceStorage]]:
    backend = request.param
    if backend in {"sqlite", "filesystem"}:
        yield (lambda: PersistenceStorage.sqlite(tmp_path / "runtime.db")) if backend == "sqlite" else (
            lambda: PersistenceStorage.filesystem(tmp_path / "runtime"))
        return
    from sqlalchemy.engine import URL
    from sqlalchemy.ext.asyncio import create_async_engine
    from linktools.ai.migrate import provision_database
    from ._metrics_server_test_helpers import _server_database

    server = request.getfixturevalue(f"_{backend}_server")
    with _server_database(server) as (_, command):
        if backend == "postgresql":
            url = URL.create("postgresql+asyncpg", username=getpass.getuser(), database=command[-1],
                query={"host": command[command.index("-h") + 1]})
        else:
            socket = next(value.split("=", 1)[1] for value in command if value.startswith("--socket="))
            url = URL.create("mysql+asyncmy", username="root", database=command[-1],
                query={"unix_socket": socket, "charset": "utf8mb4"})
        engines = []

        def storage() -> PersistenceStorage:
            engine = create_async_engine(url, isolation_level="READ COMMITTED")
            engines.append(engine)
            return PersistenceStorage.sql(engine)

        provisioner = create_async_engine(url)
        try:
            await provision_database(provisioner)
            yield storage
        finally:
            for engine in engines:
                await engine.dispose()
            await provisioner.dispose()


@pytest.mark.asyncio
async def test_handoff_recovers_same_execution_and_next_session_turn(handoff_storage) -> None:
    storage = handoff_storage
    entered = asyncio.Event()

    async def blocked(messages, info):
        entered.set()
        await asyncio.Event().wait()
        yield "unreachable"

    async with Persistence.open('shutdown-handoff', storage=storage()) as persistence_71, Runtime.attach(persistence=persistence_71, features=model_features(_Models(FunctionModel(stream_function=blocked)), _group(False))) as owner:
        session = await owner.agents.get().create_session("session")
        execution = await session.start("Remember blue", idempotency_key="first")
        await asyncio.wait_for(entered.wait(), 10)
        await asyncio.wait_for(owner.handoff(), 10)
        await owner.handoff()
        await owner.close()

    observed = []

    async def answer(messages, info):
        observed.append(messages)
        yield "blue"

    async with Persistence.open('shutdown-handoff', storage=storage()) as persistence_86, Runtime.attach(persistence=persistence_86, features=model_features(_Models(FunctionModel(stream_function=answer)), _group(False)), auto_recover=False) as successor:
        principal = successor.default_principal
        before = await successor.executions.inspect(execution.execution_id, principal=principal)
        assert before.status is ExecutionStatus.STARTED
        recovered = await successor.executions.recover(execution.execution_id,
            principal=principal, idempotency_key="handoff-owner")
        assert recovered.execution_id == execution.execution_id
        await successor.executions.wait(recovered.execution_id, principal=principal, timeout_seconds=10)
        assert (await successor.executions.inspect(execution.execution_id,
            principal=principal)).status is ExecutionStatus.SUCCEEDED
        session = successor.agents.get().session("session")
        replay = await session.start("Remember blue", idempotency_key="first")
        assert replay.execution_id == execution.execution_id
        following = await session.start("What colour?", idempotency_key="next")
        await asyncio.wait_for(following.wait(), 10)
        assert following.execution_id != execution.execution_id
        from pydantic_ai.messages import TextPart, UserPromptPart
        first_parts = [part for message in observed[0] for part in message.parts]
        assert any(isinstance(part, UserPromptPart) and "Remember blue" in str(part.content)
                   for part in first_parts)
        next_parts = [part for message in observed[-1] for part in message.parts]
        assert any(isinstance(part, TextPart) and part.content == "blue" for part in next_parts)
        assert any(isinstance(part, UserPromptPart) and "What colour?" in str(part.content)
                   for part in next_parts)


@pytest.mark.asyncio
async def test_handoff_requires_checkpoint_capability_without_closing_runtime(tmp_path: Path) -> None:
    async def answer(messages, info):
        yield "ready"

    async with Persistence.open('handoff-required', storage=PersistenceStorage.sqlite(tmp_path / 'state.db')) as persistence_118, Runtime.attach(persistence=persistence_118, features=model_features(_Models(FunctionModel(stream_function=answer)), _group(True))) as runtime:
        with pytest.raises(AIError) as error:
            await runtime.handoff()
        assert error.value.code is ErrorCode.CAPABILITY_REQUIRED_MISSING
        execution = await runtime.agents.get().start("Still usable", idempotency_key="request")
        await asyncio.wait_for(execution.wait(), 10)


@pytest.mark.asyncio
@pytest.mark.parametrize("unknown", [False, True])
async def test_handoff_settles_tool_effect_once_before_recovery(handoff_storage, unknown: bool) -> None:
    from linktools.ai.capability import AgentContext, CheckpointCapability, ExecutionCapability
    from linktools.ai.core import CommitPolicy
    from linktools.ai.runtime import RuntimeFeatures
    from .test_task_mixed_node_reliability import _TaskTestModels

    entered, release = asyncio.Event(), asyncio.Event()
    effects = []

    async def write(context: AgentContext[None]) -> str:
        entered.set()
        await release.wait()
        effects.append("written")
        if unknown:
            raise RuntimeError("provider outcome is unknown")
        return "stored once"

    def features():
        group = RuntimeFeatures("handoff-tool")
        group.capability(ExecutionCapability(retain_history=True, commit=CommitPolicy.sync()))
        group.capability(CheckpointCapability())
        group.tool(write, effect_policy="non_replay_safe")
        group.agent("default", allow_tools=("write",))
        return group

    async with Persistence.open('handoff-tool', storage=handoff_storage()) as persistence_154, Runtime.attach(persistence=persistence_154, features=model_features(_TaskTestModels(), features())) as owner:
        execution = await owner.agents.get().start("Write once", idempotency_key="write")
        await asyncio.wait_for(entered.wait(), 10)
        handoff = asyncio.create_task(owner.handoff())
        await asyncio.sleep(.05)
        assert not handoff.done()
        assert effects == []
        release.set()
        await asyncio.wait_for(handoff, 10)
        assert effects == ["written"]
    async with Persistence.open('handoff-tool', storage=handoff_storage()) as persistence_165, Runtime.attach(persistence=persistence_165, features=model_features(_TaskTestModels(), features()), auto_recover=False) as successor:
        if unknown:
            view = await successor.executions.inspect(execution.execution_id,
                principal=successor.default_principal)
            assert view.status is ExecutionStatus.RECOVERY_REQUIRED
            with pytest.raises(AIError) as error:
                await successor.executions.recover(execution.execution_id,
                    principal=successor.default_principal, idempotency_key="recover-write")
            assert error.value.code is ErrorCode.TOOL_EFFECT_UNKNOWN
            assert effects == ["written"]
            return
        await successor.executions.recover(execution.execution_id,
            principal=successor.default_principal, idempotency_key="recover-write")
        await successor.executions.wait(execution.execution_id,
            principal=successor.default_principal, timeout_seconds=10)
        assert effects == ["written"]
        assert (await successor.executions.inspect(execution.execution_id,
            principal=successor.default_principal)).status is ExecutionStatus.SUCCEEDED


@pytest.mark.parametrize("signal_name", ["SIGTERM", "SIGKILL"])
def test_process_shutdown_recovery_preserves_execution_id(tmp_path: Path, signal_name: str) -> None:
    import os
    import signal
    from ._runtime_process_test_helpers import _RuntimeProcess

    owner = _RuntimeProcess(tmp_path / "process.db", phase="graceful")
    successor = None
    try:
        execution_id = owner.initial["execution_id"]
        os.kill(owner.process.pid, getattr(signal, signal_name))
        if signal_name == "SIGTERM":
            assert owner.receive() == {"closed": True}
        owner.process.join(15)
        assert not owner.process.is_alive()
        assert owner.process.exitcode == (0 if signal_name == "SIGTERM" else -signal.SIGKILL)
        successor = _RuntimeProcess(tmp_path / "process.db", execution_id=execution_id)
        assert successor.initial["status"] == "STARTED"
        successor.request("recover")
        started = successor.request("await_model")
        assert started["execution_id"] == execution_id
        assert started["model_calls"] == 1
        assert successor.request("finish")["status"] == "SUCCEEDED"
        successor.close(check=True)
    finally:
        owner.close()
        if successor is not None:
            successor.close()


@pytest.mark.asyncio
async def test_handoff_waiter_cancellation_keeps_owned_cleanup(tmp_path: Path, monkeypatch) -> None:
    from linktools.ai.runtime._local import LocalExecutionBackend

    entered, release = asyncio.Event(), asyncio.Event()
    original = LocalExecutionBackend.handoff

    async def pause_handoff(backend):
        entered.set()
        await release.wait()
        await original(backend)

    monkeypatch.setattr(LocalExecutionBackend, "handoff", pause_handoff)
    async def answer(messages, info):
        yield "answer"

    async with Persistence.open('handoff-waiter', storage=PersistenceStorage.sqlite(tmp_path / 'waiter.db')) as persistence_232, Runtime.attach(persistence=persistence_232, features=model_features(_Models(FunctionModel(stream_function=answer)), _group(False))) as runtime:
        waiter = asyncio.create_task(runtime.handoff())
        await asyncio.wait_for(entered.wait(), 5)
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        close = asyncio.create_task(runtime.close())
        await asyncio.sleep(0)
        assert not close.done()
        release.set()
        await asyncio.wait_for(close, 5)
        await runtime.handoff()


@pytest.mark.asyncio
async def test_handoff_does_not_claim_success_after_default_close(handoff_storage) -> None:
    entered = asyncio.Event()

    async def blocked(messages, info):
        entered.set()
        await asyncio.Event().wait()
        yield "unreachable"

    models = _Models(FunctionModel(stream_function=blocked))
    async with Persistence.open('handoff-close', storage=handoff_storage()) as persistence_257, Runtime.attach(persistence=persistence_257, features=model_features(models, _group(False))) as runtime:
        execution = await runtime.agents.get().start("Cancel normally", idempotency_key="close")
        await asyncio.wait_for(entered.wait(), 10)
        await runtime.close()
        with pytest.raises(AIError) as error:
            await runtime.handoff()
        assert error.value.code is ErrorCode.RUNTIME_DEPENDENCY_NOT_READY
    async with Persistence.open('handoff-close', storage=handoff_storage()) as persistence_265, Runtime.attach(persistence=persistence_265, features=model_features(models, _group(False)), auto_recover=False) as reader:
        view = await reader.executions.inspect(execution.execution_id, principal=reader.default_principal)
        assert view.status is ExecutionStatus.CANCELLED


@pytest.mark.asyncio
async def test_handoff_before_tool_leaf_defers_without_unknown_effect() -> None:
    from linktools.ai.runtime._handoff import ExecutionHandoffScope, execution_handoff_scope
    from linktools.ai.runtime._tool_boundary import ManagedToolDescriptor
    from .test_tool_effect_semantics import _Bridge, _call

    scope = ExecutionHandoffScope()
    class StopAfterAdmission(_Bridge):
        async def begin(self, *args, **kwargs):
            decision = await super().begin(*args, **kwargs)
            scope.stopping = True
            return decision

    async def write() -> str:
        raise AssertionError("leaf cannot enter after handoff")

    bridge = StopAfterAdmission(False)
    token = execution_handoff_scope.set(scope)
    try:
        with pytest.raises(asyncio.CancelledError):
            await _call(write, ManagedToolDescriptor("tool_operation", "non_replay_safe", "business"),
                        bridge=bridge)
    finally:
        execution_handoff_scope.reset(token)
    assert "defer" in bridge.calls
    assert "unknown" not in bridge.calls


@pytest.mark.asyncio
async def test_pending_user_cancel_interrupts_tool_during_handoff(tmp_path: Path, monkeypatch) -> None:
    from linktools.ai.capability import AgentContext, CheckpointCapability, ExecutionCapability
    from linktools.ai.core import CommitPolicy
    from linktools.ai.runtime import RuntimeFeatures
    from linktools.ai.runtime._execution import DefaultExecutionService
    from .test_task_mixed_node_reliability import _TaskTestModels

    tool_entered, tool_cancelled = asyncio.Event(), asyncio.Event()
    cancel_entered, release_cancel = asyncio.Event(), asyncio.Event()
    original = DefaultExecutionService.cancel_authorized

    async def pause_cancel(service, *args, **kwargs):
        cancel_entered.set()
        await release_cancel.wait()
        return await original(service, *args, **kwargs)

    monkeypatch.setattr(DefaultExecutionService, "cancel_authorized", pause_cancel)
    async def write(context: AgentContext[None]) -> str:
        tool_entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            tool_cancelled.set()
            raise
        return "unreachable"

    group = RuntimeFeatures("cancel-handoff")
    group.capability(ExecutionCapability(retain_history=True, commit=CommitPolicy.sync()))
    group.capability(CheckpointCapability())
    group.tool(write, effect_policy="non_replay_safe")
    group.agent("default", allow_tools=("write",))
    async with Persistence.open('cancel-handoff', storage=PersistenceStorage.sqlite(tmp_path / 'cancel.db')) as persistence_331, Runtime.attach(persistence=persistence_331, features=model_features(_TaskTestModels(), group)) as runtime:
        execution = await runtime.agents.get().start("Write", idempotency_key="write")
        await asyncio.wait_for(tool_entered.wait(), 10)
        cancellation = asyncio.create_task(execution.cancel(idempotency_key="cancel"))
        await asyncio.wait_for(cancel_entered.wait(), 5)
        handoff = asyncio.create_task(runtime.handoff())
        await asyncio.sleep(.05)
        release_cancel.set()
        await asyncio.wait_for(tool_cancelled.wait(), 5)
        await asyncio.wait_for(cancellation, 10)
        await asyncio.wait_for(handoff, 10)
    async with Persistence.open('cancel-handoff', storage=PersistenceStorage.sqlite(tmp_path / 'cancel.db')) as persistence_343, Runtime.attach(persistence=persistence_343, features=model_features(_TaskTestModels(), group), auto_recover=False) as successor:
        view = await successor.executions.inspect(execution.execution_id,
            principal=successor.default_principal)
        assert view.status in {ExecutionStatus.CANCELLED, ExecutionStatus.RECOVERY_REQUIRED}
        with pytest.raises(AIError):
            await successor.executions.recover(execution.execution_id,
                principal=successor.default_principal, idempotency_key="must-not-replay")


@pytest.mark.asyncio
async def test_handoff_joins_admission_before_relinquishing_storage(tmp_path: Path, monkeypatch) -> None:
    from linktools.ai.runtime._local import LocalExecutionBackend

    admitted, release = asyncio.Event(), asyncio.Event()
    original = LocalExecutionBackend.launch

    async def pause_launch(backend, *args, **kwargs):
        admitted.set()
        await release.wait()
        return await original(backend, *args, **kwargs)

    monkeypatch.setattr(LocalExecutionBackend, "launch", pause_launch)
    async def blocked(messages, info):
        await asyncio.Event().wait()
        yield "unreachable"

    database = tmp_path / "admission.db"
    async with Persistence.open('handoff-admission', storage=PersistenceStorage.sqlite(database)) as persistence_371, Runtime.attach(persistence=persistence_371, features=model_features(_Models(FunctionModel(stream_function=blocked)), _group(False))) as owner:
        start = asyncio.create_task(owner.agents.get().start("admitted", idempotency_key="start"))
        await asyncio.wait_for(admitted.wait(), 5)
        shutdown = asyncio.create_task(owner.handoff())
        await asyncio.sleep(.05)
        assert not shutdown.done()
        release.set()
        execution = await asyncio.wait_for(start, 10)
        await asyncio.wait_for(shutdown, 10)
    async def answer(messages, info):
        yield "recovered"
    async with Persistence.open('handoff-admission', storage=PersistenceStorage.sqlite(database)) as persistence_383, Runtime.attach(persistence=persistence_383, features=model_features(_Models(FunctionModel(stream_function=answer)), _group(False)), auto_recover=False) as successor:
        await successor.executions.recover(execution.execution_id,
            principal=successor.default_principal, idempotency_key="recover-admitted")
        await successor.executions.wait(execution.execution_id,
            principal=successor.default_principal, timeout_seconds=10)
        assert (await successor.executions.inspect(execution.execution_id,
            principal=successor.default_principal)).status is ExecutionStatus.SUCCEEDED


@pytest.mark.parametrize("keyed", [False, True])
def test_resident_runtime_finishes_prepared_result_without_model_replay(tmp_path: Path, keyed: bool) -> None:
    from ._runtime_process_test_helpers import _RuntimeProcess

    owner = _RuntimeProcess(tmp_path / "finalizing.db", phase="finalizing")
    successor = None
    try:
        assert owner.initial["status"] == "FINALIZING"
        successor = _RuntimeProcess(tmp_path / "finalizing.db", phase="finalizing",
            execution_id=owner.initial["execution_id"], auto_recover=False)
        assert successor.initial["status"] == "FINALIZING"
        owner.process.kill()
        owner.process.join(10)
        command = "recover_keyed" if keyed else "recover"
        result = successor.request(command)
        assert result["status"] == "SUCCEEDED"
        assert result["model_calls"] == 0
        assert result["generation"] == owner.initial["generation"]
        assert successor.request("result")["output"] == {"text": "owner answer"}
        if keyed:
            assert successor.request(command)["status"] == "SUCCEEDED"
        successor.close(check=True)
    finally:
        owner.close()
        if successor is not None:
            successor.close()


@pytest.mark.parametrize("ordered_publication", (False, True))
@pytest.mark.parametrize("synchronized", (False, True))
def test_concurrent_finalizing_takeover_keeps_one_result_and_replay_receipt(
    tmp_path: Path, synchronized: bool, ordered_publication: bool,
) -> None:
    from concurrent.futures import ThreadPoolExecutor
    import multiprocessing
    from ._runtime_process_test_helpers import _RuntimeProcess

    owner = _RuntimeProcess(tmp_path / "race.db", phase="finalizing")
    successors = []
    context = multiprocessing.get_context("spawn")
    barrier = context.Barrier(2) if synchronized else None
    publication_events = tuple(context.Event() for _ in range(3)) if ordered_publication else ()
    try:
        for role in ("delayed", "first"):
            successors.append(_RuntimeProcess(tmp_path / "race.db", phase="finalizing",
                execution_id=owner.initial["execution_id"], auto_recover=False, recovery_barrier=barrier,
                publication_order=(role, *publication_events) if ordered_publication else None))
        owner.process.kill()
        owner.process.join(10)
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(lambda runtime: runtime.request("try_recover_keyed"), successors))
        assert any(result.get("status") == "SUCCEEDED" for result in results), results
        for result in results:
            assert result.get("error_code") in {None, "STORAGE_CONFLICT", "STORAGE_RECOVERY_REQUIRED"}
        for successor in successors:
            result = successor.request("recover_keyed")
            assert result["status"] == "SUCCEEDED" and result["model_calls"] == 0
            assert result["handoff_phase"] == "completed"
            assert result["active_execution_id"] is None
            assert successor.request("history")["answers"] == ["owner answer"]
            successor.close(check=True)
    finally:
        owner.close()
        for successor in successors:
            successor.close()


def test_recovery_receipt_waits_for_committed_result_publication(tmp_path: Path) -> None:
    from ._runtime_process_test_helpers import _RuntimeProcess

    owner = _RuntimeProcess(tmp_path / "publication.db", phase="finalizing")
    successor = _RuntimeProcess(tmp_path / "publication.db", phase="finalizing",
        execution_id=owner.initial["execution_id"], auto_recover=False)
    try:
        owner.process.kill()
        owner.process.join(10)
        interrupted = successor.request("recover_before_publication")
        assert interrupted["status"] == "SUCCEEDED"
        assert interrupted["handoff_phase"] == "prepared"
        result = successor.request("recover_keyed")
        assert result["status"] == "SUCCEEDED"
        assert result["handoff_phase"] == "completed"
        assert result["active_execution_id"] is None
        assert result["model_calls"] == 0
        assert successor.request("history")["answers"] == ["owner answer"]
        successor.close(check=True)
    finally:
        owner.close()
        successor.close()
