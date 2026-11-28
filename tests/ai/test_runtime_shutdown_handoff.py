#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from linktools.ai.persistence import Persistence
"""Graceful shutdown preserves checkpointed conversation execution identity."""
from ._runtime_test_helpers import configure_execution
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
from .test_terminal_commit_shared_runtime import _HostAuthorization, _group


def _checkpointed_execution_group():
    from linktools.ai.capability import StepPersistenceCapability
    from linktools.ai.core import CommitPolicy

    features = _group(True)
    configure_execution(features, retain_history=True)
    features.capability(StepPersistenceCapability())
    return features


@pytest_asyncio.fixture(params=("sqlite", "local_sqlite", "postgresql", "mysql"))
async def handoff_storage(request: pytest.FixtureRequest, tmp_path: Path) -> AsyncIterator[Callable[[], PersistenceStorage]]:
    backend = request.param
    if backend in {"sqlite", "local_sqlite"}:
        yield (lambda: PersistenceStorage.sqlite(tmp_path / "runtime.db")) if backend == "sqlite" else (
            lambda: PersistenceStorage.from_root(tmp_path / "runtime", exclusive_writer=True))
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
    from pydantic_ai.messages import TextPart, UserPromptPart

    storage = handoff_storage
    entered = asyncio.Event()

    async def blocked(messages, info):
        entered.set()
        await asyncio.Event().wait()
        yield "unreachable"

    async with (
        Persistence.open("shutdown-handoff", storage=storage()) as state,
        Runtime.open(
            persistence=state, features=model_features(_Models(FunctionModel(stream_function=blocked)), _group(False)),
            authorization=_HostAuthorization(),
        ) as owner,
    ):
        principal = owner.default_principal
        await owner.sessions.create_turn_session("session", principal=principal)
        turn = await owner.sessions.start(
            "session", "Remember blue", agent="default", principal=principal, idempotency_key="first",
        )
        await asyncio.wait_for(entered.wait(), 10)
        await asyncio.wait_for(owner.handoff(), 10)
        await owner.handoff()
        await owner.close()

    observed = []

    async def answer(messages, info):
        observed.append(messages)
        yield "blue"

    async with (
        Persistence.open("shutdown-handoff", storage=storage()) as state,
        Runtime.open(
            persistence=state, features=model_features(_Models(FunctionModel(stream_function=answer)), _group(False)),
            authorization=_HostAuthorization(),
        ) as successor,
    ):
        retained = await state.session_turns.get_turn(turn.key)
        assert retained["state"] == "accepted" and retained["generation"] == 1
        assert (await successor.history.turn_session_state("session", principal=principal))["active_turn_id"] == turn.key.turn_id
        # Expire the retained lease without sleeping through its production duration.
        turns = state.session_turns
        async with turns._context.engine.begin() as connection:
            changed = await connection.execute(
                turns._turns.update().filter_by(**turns._turn_values(turn.key), generation=1).values(lease_until=0),
            )
            assert changed.rowcount == 1
        assert await successor.sessions.recover_turn("session", turn.key.turn_id, mode="resume", principal=principal)
        after = await successor.history.turn_result("session", turn.key.turn_id, principal=principal)
        assert after["status"] == "SUCCEEDED" and after["output"] == {"text": "blue"}
        replay = await successor.sessions.start(
            "session", "Remember blue", agent="default", principal=principal, idempotency_key="first",
        )
        assert replay.key == turn.key and len(observed) == 1
        following = await successor.sessions.start(
            "session", "What colour?", agent="default", principal=principal, idempotency_key="next",
        )
        assert (await following.wait_final(timeout_seconds=10)).status is ExecutionStatus.SUCCEEDED
        assert following.key.turn_id != turn.key.turn_id and len(observed) == 2
        first_parts = [part for message in observed[0] for part in message.parts]
        assert any(isinstance(part, UserPromptPart) and "Remember blue" in str(part.content)
                   for part in first_parts)
        next_parts = [part for message in observed[-1] for part in message.parts]
        assert any(isinstance(part, TextPart) and part.content == "blue" for part in next_parts)
        assert any(isinstance(part, UserPromptPart) and "What colour?" in str(part.content)
                   for part in next_parts)


@pytest.mark.asyncio
@pytest.mark.parametrize("pending", [False, True])
async def test_handoff_rejects_nonrecoverable_execution_without_closing_runtime(tmp_path: Path, pending: bool) -> None:
    from pydantic_ai.capabilities import CombinedCapability
    from linktools.ai.capability import StepPersistenceCapability
    from linktools.ai.core import CommitPolicy
    from linktools.ai.storage import InMemoryCacheBackend

    entered, release = asyncio.Event(), asyncio.Event()

    async def answer(messages, info):
        entered.set()
        await release.wait()
        yield "ready"

    features = _group(True)
    features.capability(StepPersistenceCapability(), id="runtime.steps")
    configure_execution(
        features, retain_history=True, commit=CommitPolicy.terminal() if pending else CommitPolicy.sync(),
    )
    model_features(_Models(FunctionModel(stream_function=answer)), features)
    cache = InMemoryCacheBackend(max_bytes=2**20) if pending else None
    async with Runtime.open('handoff-required', storage=PersistenceStorage.sqlite(tmp_path / 'state.db',
                            cache=cache, coordination=cache), features=features) as runtime:
        features.capability(CombinedCapability([]), id="runtime.steps")
        active = await runtime.agents.get().start("Nonrecoverable", idempotency_key="first")
        await asyncio.wait_for(entered.wait(), 5)
        features.capability(StepPersistenceCapability(), id="runtime.steps")
        with pytest.raises(AIError) as error:
            await runtime.handoff()
        assert error.value.code is ErrorCode.CAPABILITY_REQUIRED_MISSING
        assert error.value.safe_details["reason"] == "nonrecoverable_active_execution"
        release.set()
        assert (await active.wait(timeout_seconds=5)).result.status is ExecutionStatus.SUCCEEDED
        execution = await runtime.agents.get().start("Still usable", idempotency_key="request")
        await asyncio.wait_for(execution.wait(), 10)
        features.capability(CombinedCapability([]), id="runtime.steps")
        await runtime.handoff()


@pytest.mark.asyncio
@pytest.mark.parametrize("late_checkpoint", [False, True])
async def test_handoff_keeps_admitted_checkpoint_guarantee_across_policy_updates(
    tmp_path: Path, late_checkpoint: bool,
) -> None:
    from pydantic_ai.capabilities import CombinedCapability
    from linktools.ai.capability import StepPersistenceCapability
    from linktools.ai.core import CommitPolicy

    entered = asyncio.Event()

    async def blocked(messages, info):
        entered.set()
        await asyncio.Event().wait()
        yield "unreachable"

    features = _group(True)
    configure_execution(features, retain_history=True)
    if not late_checkpoint:
        features.capability(StepPersistenceCapability(), id="runtime.steps")
    model_features(_Models(FunctionModel(stream_function=blocked)), features)
    storage = PersistenceStorage.from_root(tmp_path)
    async with Runtime.open("live-handoff", storage=storage, features=features) as runtime:
        if late_checkpoint:
            features.capability(StepPersistenceCapability(), id="runtime.steps")
        execution = await runtime.agents.get().start("Resume me", idempotency_key="first")
        await asyncio.wait_for(entered.wait(), 5)
        features.capability(CombinedCapability([]), id="runtime.steps")
        await asyncio.wait_for(runtime.handoff(), 5)

    async def answer(messages, info):
        yield "recovered"

    features.capability(StepPersistenceCapability(), id="runtime.steps")
    model_features(_Models(FunctionModel(stream_function=answer)), features)
    async with Runtime.open("live-handoff", storage=storage, features=features) as successor:
        recovered = await successor.executions.recover(execution.execution_id,
            principal=successor.default_principal, idempotency_key="recover")
        assert recovered.execution_id == execution.execution_id
        result = await successor.executions.wait(recovered.execution_id,
            principal=successor.default_principal, timeout_seconds=5)
        assert result.result.status is ExecutionStatus.SUCCEEDED


@pytest.mark.asyncio
async def test_handoff_rejects_recovery_admission_without_stopping_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from linktools.ai.runtime._local import LocalExecutionBackend

    entered, release = asyncio.Event(), asyncio.Event()

    async def blocked_model(messages, info):
        entered.set()
        await asyncio.Event().wait()
        yield "unreachable"

    features = model_features(_Models(FunctionModel(stream_function=blocked_model)), _checkpointed_execution_group())
    storage = PersistenceStorage.sqlite(tmp_path / "state.db")
    async with Runtime.open("handoff-recovery", storage=storage, features=features) as owner:
        execution = await owner.agents.get().start("recover", idempotency_key="first")
        await asyncio.wait_for(entered.wait(), 5)
        await owner.handoff()

    entered.clear()
    original = LocalExecutionBackend.launch

    async def blocked_launch(backend, *args, **kwargs):
        entered.set()
        await release.wait()
        return await original(backend, *args, **kwargs)

    async def answer(messages, info):
        yield "recovered"

    monkeypatch.setattr(LocalExecutionBackend, "launch", blocked_launch)
    model_features(_Models(FunctionModel(stream_function=answer)), features)
    async with Runtime.open("handoff-recovery", storage=storage, features=features) as runtime:
        recovery = asyncio.create_task(runtime.executions.recover(execution.execution_id,
            principal=runtime.default_principal, idempotency_key="recover"))
        await asyncio.wait_for(entered.wait(), 5)
        try:
            with pytest.raises(AIError) as error:
                await asyncio.wait_for(runtime.handoff(), 5)
            assert error.value.safe_details["reason"] == "admission_in_progress"
            assert error.value.retryable
        finally:
            release.set()
        recovered = await asyncio.wait_for(recovery, 5)
        result = await runtime.executions.wait(recovered.execution_id,
            principal=runtime.default_principal, timeout_seconds=5)
        assert result.result.status is ExecutionStatus.SUCCEEDED
        await runtime.handoff()


@pytest.mark.asyncio
async def test_handoff_rejects_worker_without_admission_identity(tmp_path: Path) -> None:
    entered = asyncio.Event()

    async def blocked(messages, info):
        entered.set()
        await asyncio.Event().wait()
        yield "unreachable"

    features = model_features(_Models(FunctionModel(stream_function=blocked)), _checkpointed_execution_group())
    async with Runtime.open("handoff-unknown", storage=PersistenceStorage.sqlite(tmp_path / "state.db"),
                            features=features) as runtime:
        execution = await runtime.agents.get().start("first", idempotency_key="first")
        await asyncio.wait_for(entered.wait(), 5)
        backend = runtime._execution_service.runtime_backend()
        owner = backend._turn_owners.pop(execution.execution_id)
        try:
            with pytest.raises(AIError) as error:
                await runtime.handoff()
            assert error.value.safe_details["reason"] == "unknown_active_execution"
        finally:
            backend._turn_owners[execution.execution_id] = owner
        await asyncio.wait_for(runtime.handoff(), 5)


@pytest.mark.asyncio
@pytest.mark.parametrize("unknown", [False, True])
async def test_handoff_settles_tool_effect_once_before_recovery(handoff_storage, unknown: bool) -> None:
    from linktools.ai.capability import AgentContext
    from pydantic_ai_harness.step_persistence import StepPersistence
    from tests.ai._runtime_test_helpers import native_checkpoint
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
        group = RuntimeFeatures("handoff-tool", defaults={"execution": {"retain_history": True}})
        group.capability(native_checkpoint())
        group.tool(write, effect_policy="non_replay_safe")
        group.agent("default", allow_tools=("write",))
        return group

    async with Persistence.open('handoff-tool', storage=handoff_storage()) as persistence_154, Runtime.open(persistence=persistence_154, features=model_features(_TaskTestModels(), features())) as owner:
        execution = await owner.agents.get().start("Write once", idempotency_key="write")
        await asyncio.wait_for(entered.wait(), 10)
        handoff = asyncio.create_task(owner.handoff())
        await asyncio.sleep(.05)
        assert not handoff.done()
        assert effects == []
        release.set()
        await asyncio.wait_for(handoff, 10)
        assert effects == ["written"]
    async with Persistence.open('handoff-tool', storage=handoff_storage()) as persistence_165, Runtime.open(persistence=persistence_165, features=model_features(_TaskTestModels(), features())) as successor:
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

    owner = _RuntimeProcess(tmp_path / "process.db", mode="standalone", phase="graceful")
    successor = None
    try:
        execution_id = owner.initial["execution_id"]
        os.kill(owner.process.pid, getattr(signal, signal_name))
        if signal_name == "SIGTERM":
            assert owner.receive() == {"closed": True}
        owner.process.join(15)
        assert not owner.process.is_alive()
        assert owner.process.exitcode == (0 if signal_name == "SIGTERM" else -signal.SIGKILL)
        successor = _RuntimeProcess(tmp_path / "process.db", mode="standalone", execution_id=execution_id)
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

    async with Persistence.open('handoff-waiter', storage=PersistenceStorage.sqlite(tmp_path / 'waiter.db')) as persistence_232, Runtime.open(persistence=persistence_232, features=model_features(_Models(FunctionModel(stream_function=answer)), _group(False))) as runtime:
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
    async with Persistence.open('handoff-close', storage=handoff_storage()) as persistence_257, Runtime.open(persistence=persistence_257, features=model_features(models, _group(False))) as runtime:
        execution = await runtime.agents.get().start("Cancel normally", idempotency_key="close")
        await asyncio.wait_for(entered.wait(), 10)
        await runtime.close()
        with pytest.raises(AIError) as error:
            await runtime.handoff()
        assert error.value.code is ErrorCode.RUNTIME_DEPENDENCY_NOT_READY
    async with Persistence.open('handoff-close', storage=handoff_storage()) as persistence_265, Runtime.open(persistence=persistence_265, features=model_features(models, _group(False))) as reader:
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
    from linktools.ai.capability import AgentContext
    from pydantic_ai_harness.step_persistence import StepPersistence
    from tests.ai._runtime_test_helpers import native_checkpoint
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

    group = RuntimeFeatures("cancel-handoff", defaults={"execution": {"retain_history": True}})
    group.capability(native_checkpoint())
    group.tool(write, effect_policy="non_replay_safe")
    group.agent("default", allow_tools=("write",))
    async with Persistence.open('cancel-handoff', storage=PersistenceStorage.sqlite(tmp_path / 'cancel.db')) as persistence_331, Runtime.open(persistence=persistence_331, features=model_features(_TaskTestModels(), group)) as runtime:
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
    async with Persistence.open('cancel-handoff', storage=PersistenceStorage.sqlite(tmp_path / 'cancel.db')) as persistence_343, Runtime.open(persistence=persistence_343, features=model_features(_TaskTestModels(), group)) as successor:
        view = await successor.executions.inspect(execution.execution_id,
            principal=successor.default_principal)
        assert view.status in {ExecutionStatus.CANCELLED, ExecutionStatus.RECOVERY_REQUIRED}
        with pytest.raises(AIError):
            await successor.executions.recover(execution.execution_id,
                principal=successor.default_principal, idempotency_key="must-not-replay")


@pytest.mark.asyncio
async def test_handoff_retry_retains_admitted_execution_identity(tmp_path: Path, monkeypatch) -> None:
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
    async with Persistence.open('handoff-admission', storage=PersistenceStorage.sqlite(database)) as persistence_371, Runtime.open(persistence=persistence_371, features=model_features(_Models(FunctionModel(stream_function=blocked)), _checkpointed_execution_group())) as owner:
        start = asyncio.create_task(owner.agents.get().start("admitted", idempotency_key="start"))
        await asyncio.wait_for(admitted.wait(), 5)
        try:
            with pytest.raises(AIError) as error:
                await owner.handoff()
            assert error.value.safe_details["reason"] == "admission_in_progress"
            assert error.value.retryable
            assert persistence_371.ready
        finally:
            release.set()
        execution = await asyncio.wait_for(start, 10)
        await asyncio.wait_for(owner.handoff(), 10)
    async def answer(messages, info):
        yield "recovered"
    async with Persistence.open('handoff-admission', storage=PersistenceStorage.sqlite(database)) as persistence_383, Runtime.open(persistence=persistence_383, features=model_features(_Models(FunctionModel(stream_function=answer)), _checkpointed_execution_group())) as successor:
        await successor.executions.recover(execution.execution_id,
            principal=successor.default_principal, idempotency_key="recover-admitted")
        await successor.executions.wait(execution.execution_id,
            principal=successor.default_principal, timeout_seconds=10)
        assert (await successor.executions.inspect(execution.execution_id,
            principal=successor.default_principal)).status is ExecutionStatus.SUCCEEDED


@pytest.mark.parametrize("keyed", [False, True])
def test_resident_runtime_finishes_prepared_result_without_model_replay(tmp_path: Path, keyed: bool) -> None:
    from ._runtime_process_test_helpers import _RuntimeProcess

    owner = _RuntimeProcess(tmp_path / "finalizing.db", mode="standalone", phase="finalizing")
    successor = None
    try:
        assert owner.initial["status"] == "STARTED" and owner.initial["handoff_phase"] == "prepared"
        successor = _RuntimeProcess(tmp_path / "finalizing.db", mode="standalone", phase="finalizing",
            execution_id=owner.initial["execution_id"], auto_recover=False)
        assert successor.initial["status"] == "STARTED" and successor.initial["handoff_phase"] == "prepared"
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


@pytest.mark.parametrize("synchronized,ordered_publication", (
    pytest.param(False, False, id="natural_concurrent_takeover"),
    pytest.param(True, False, id="prepared_contenders_rendezvous"),
    pytest.param(False, True, id="ordered_terminal_publication"),
))
def test_concurrent_finalizing_takeover_keeps_one_result_and_replay_receipt(
    tmp_path: Path, synchronized: bool, ordered_publication: bool,
) -> None:
    from concurrent.futures import ThreadPoolExecutor
    import multiprocessing
    from ._runtime_process_test_helpers import _RuntimeProcess

    owner = _RuntimeProcess(tmp_path / "race.db", mode="standalone", phase="finalizing")
    successors = []
    context = multiprocessing.get_context("spawn")
    barrier = context.Barrier(2) if synchronized else None
    publication_events = tuple(context.Event() for _ in range(3)) if ordered_publication else ()
    try:
        for role in ("delayed", "first"):
            successors.append(_RuntimeProcess(tmp_path / "race.db", mode="standalone", phase="finalizing",
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


@pytest.mark.parametrize("boundary", ("before_terminal", "before_publication"))
def test_recovery_receipt_waits_for_committed_result_publication(tmp_path: Path, boundary: str) -> None:
    from ._runtime_process_test_helpers import _RuntimeProcess

    owner = _RuntimeProcess(tmp_path / "publication.db", mode="standalone", phase="finalizing")
    successor = _RuntimeProcess(tmp_path / "publication.db", mode="standalone", phase="finalizing",
        execution_id=owner.initial["execution_id"], auto_recover=False)
    try:
        owner.process.kill()
        owner.process.join(10)
        interrupted = successor.request("recover_" + boundary)
        assert interrupted["error_code"] == "STORAGE_RECOVERY_REQUIRED"
        assert interrupted["status"] == ("STARTED" if boundary == "before_terminal" else "SUCCEEDED")
        assert interrupted["handoff_phase"] == "prepared"
        successor.close(check=True)
        successor = _RuntimeProcess(tmp_path / "publication.db", mode="standalone", phase="finalizing",
            execution_id=owner.initial["execution_id"], auto_recover=False)
        assert successor.initial["status"] == interrupted["status"]
        assert successor.initial["handoff_phase"] == "prepared"
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
