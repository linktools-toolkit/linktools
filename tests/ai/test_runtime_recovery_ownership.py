#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared SQL startup observes live producers; explicit recovery owns takeover."""

import asyncio
from contextlib import AsyncExitStack
from datetime import datetime, timezone
import multiprocessing
from multiprocessing.connection import Connection
from pathlib import Path
import traceback

import pytest
from pydantic_ai.models.function import FunctionModel

from linktools.ai.capability import CapabilityGroup
from linktools.ai.core import ApprovalDecision, ApprovalStatus, ExecutionEventType, ExecutionStatus
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime import Runtime, RuntimeStorage, RuntimeStoragePlan, RuntimeStorageRoute
from linktools.ai.runtime.service_api import CancelExecutionRequest

from ._session_tool_test_helpers import _split_sqlite_storage
from ._runtime_test_helpers import _wait_for_committed
from .test_execution_recovery_commands import _commands as _recovery_commands, _execution
from .test_live_history_readback_integration import _Models


_NAMESPACE = "recovery-ownership"


def _capabilities() -> CapabilityGroup:
    group = CapabilityGroup(_NAMESPACE)
    group.agent("default", model="default", allow_tools=())
    return group


def _storage(database: Path, phase: str) -> RuntimeStorage:
    if phase in {"pending", "handoff"}:
        return _split_sqlite_storage(database)
    if phase == "filesystem":
        return RuntimeStorage(RuntimeStoragePlan(
            conversation=RuntimeStorageRoute.sqlite(database.with_suffix(".conversation.db")),
            execution=RuntimeStorageRoute.filesystem(database.with_suffix(".execution")),
        ))
    return RuntimeStorage.sqlite(database)


def _runtime_process(
    connection: Connection, database: str, mode: str, phase: str,
    execution_id: str | None,
) -> None:
    async def run() -> None:
        nonlocal execution_id
        entered = asyncio.Event()
        release_model = asyncio.Event()
        admitted = asyncio.Event()
        release_admission = asyncio.Event()
        calls = 0
        start_task = None
        if phase == "handoff":
            release_model.set()

        async def model(messages, info):
            nonlocal calls
            del messages, info
            calls += 1
            entered.set()
            await release_model.wait()
            yield "owner answer"

        state = _storage(Path(database), phase)
        async with Runtime.open(
            _NAMESPACE, models=_Models(FunctionModel(stream_function=model)),
            storage=state, capabilities=(_capabilities(),),
        ) as runtime:
            async def snapshot() -> dict[str, object]:
                assert execution_id is not None
                tenant = runtime.default_principal.tenant_id
                execution = await state.execution.executions.get(execution_id, tenant_id=tenant)
                head = await state.execution.executions.get_history_head(execution_id, tenant_id=tenant)
                assert execution is not None and head is not None
                return {
                    "execution_id": execution_id, "status": execution.status.value,
                    "revision": execution.revision, "generation": head.producer_generation,
                    "claim_id": head.producer_claim_id, "model_calls": calls,
                }

            if execution_id is None:
                backend = runtime._execution_service.runtime_backend()
                if phase == "admitted":
                    original_launch = backend.launch

                    async def pause_launch(request, execution, **kwargs):
                        nonlocal execution_id
                        execution_id = execution.execution_id
                        admitted.set()
                        await release_admission.wait()
                        return await original_launch(request, execution, **kwargs)

                    backend.launch = pause_launch
                elif phase == "pending":
                    original_claim = state.execution.executions.claim_start

                    async def pause_claim(claim):
                        nonlocal execution_id
                        execution_id = claim.execution_id
                        admitted.set()
                        await release_admission.wait()
                        return await original_claim(claim)

                    state.execution.executions.claim_start = pause_claim
                elif phase == "handoff":
                    original_handoff = backend._reconcile_handoff

                    async def pause_handoff(checkpoint):
                        nonlocal execution_id
                        execution_id = checkpoint.execution_id
                        admitted.set()
                        await release_admission.wait()
                        return await original_handoff(checkpoint)

                    backend._reconcile_handoff = pause_handoff

                async def start():
                    agent = runtime.agents.get("default")
                    if mode == "session":
                        await agent.create_session("session")
                        return await agent.session("session").start("Wait", idempotency_key="start")
                    return await agent.start("Wait", idempotency_key="start")

                start_task = asyncio.create_task(start())
                if phase in {"active", "filesystem"}:
                    execution = await start_task
                    execution_id = execution.execution_id
                    await asyncio.wait_for(entered.wait(), 10)
                else:
                    await asyncio.wait_for(admitted.wait(), 10)
            connection.send(await snapshot())
            while True:
                command = await asyncio.to_thread(connection.recv)
                if command == "snapshot":
                    connection.send(await snapshot())
                elif command == "recover":
                    await runtime.executions.recover(
                        execution_id, principal=runtime.default_principal,
                    )
                    connection.send(await snapshot())
                elif command == "await_model":
                    await asyncio.wait_for(entered.wait(), 10)
                    connection.send(await snapshot())
                elif command == "cancel":
                    result = await runtime.executions.cancel(
                        execution_id,
                        CancelExecutionRequest(runtime.default_principal, "cancel-owner"),
                    )
                    connection.send({**await snapshot(), "cancelled": result.cancelled})
                elif command == "release_admission":
                    release_admission.set()
                    assert start_task is not None
                    await start_task
                    backend = runtime._execution_service.runtime_backend()
                    worker = backend._tasks.get(execution_id)
                    if worker is not None:
                        await asyncio.wait_for(asyncio.shield(worker), 10)
                    failure = backend.worker_failure(
                        execution_id, tenant_id=runtime.default_principal.tenant_id,
                    )
                    assert failure is None, failure
                    connection.send(await snapshot())
                elif command in {"finish", "result"}:
                    if command == "finish":
                        release_admission.set()
                        release_model.set()
                        if start_task is not None:
                            await start_task
                        waited = await runtime.executions.wait(
                            execution_id, principal=runtime.default_principal, timeout_seconds=10,
                        )
                        result = waited.result
                    else:
                        result = await runtime.executions.result(
                            execution_id, principal=runtime.default_principal,
                        )
                    connection.send({
                        "status": result.status.value, "output": result.output,
                        "model_calls": calls,
                    })
                elif command == "close":
                    release_admission.set()
                    release_model.set()
                    if start_task is not None:
                        await start_task
                    return
                else:
                    raise AssertionError(command)

    try:
        asyncio.run(run())
        connection.send({"closed": True})
    except BaseException as error:
        connection.send({"error": type(error).__name__, "traceback": traceback.format_exc()})
    finally:
        connection.close()


class _RuntimeProcess:
    def __init__(
        self, database: Path, *, mode: str = "session", phase: str = "active",
        execution_id: str | None = None,
    ) -> None:
        context = multiprocessing.get_context("spawn")
        self.connection, child = context.Pipe()
        self.process = context.Process(
            target=_runtime_process,
            args=(child, str(database), mode, phase, execution_id),
        )
        self.process.start()
        child.close()
        self._closed = False
        try:
            self.initial = self.receive(timeout=45)
        except BaseException:
            self.close()
            raise

    def receive(self, *, timeout: float = 20) -> dict[str, object]:
        assert self.connection.poll(timeout), "Runtime process did not answer"
        result = self.connection.recv()
        assert "error" not in result, result
        return result

    def request(self, command: str) -> dict[str, object]:
        self.connection.send(command)
        return self.receive()

    def crash(self) -> None:
        self.process.terminate()
        self.process.join(10)
        assert not self.process.is_alive()

    def close(self, *, check: bool = False) -> None:
        if self._closed:
            return
        outcome = None
        if self.process.is_alive():
            try:
                self.connection.send("close")
                if self.connection.poll(15):
                    outcome = self.connection.recv()
            except (BrokenPipeError, EOFError, OSError):
                pass
            self.process.join(10)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(10)
        self.connection.close()
        self._closed = True
        if check:
            assert outcome == {"closed": True}, outcome
            assert self.process.exitcode == 0


@pytest.mark.parametrize(
    ("mode", "phase"),
    (("session", "active"), ("standalone", "active"),
     ("session", "admitted"), ("session", "pending")),
)
def test_sql_startup_does_not_take_over_live_process(
    tmp_path: Path, mode: str, phase: str,
) -> None:
    owner = _RuntimeProcess(tmp_path / "runtime.db", mode=mode, phase=phase)
    remote = None
    try:
        remote = _RuntimeProcess(
            tmp_path / "runtime.db", mode=mode, phase=phase,
            execution_id=owner.initial["execution_id"],
        )
        assert owner.process.is_alive()
        for field in ("status", "revision", "generation", "claim_id"):
            assert remote.initial[field] == owner.initial[field]
        assert remote.initial["model_calls"] == 0
        expected = ExecutionStatus.PENDING_START if phase == "pending" else ExecutionStatus.STARTED
        assert remote.initial["status"] == expected.value
        assert owner.request("finish") == {
            "status": "SUCCEEDED", "output": {"text": "owner answer"}, "model_calls": 1,
        }
        assert remote.request("result") == {
            "status": "SUCCEEDED", "output": {"text": "owner answer"}, "model_calls": 0,
        }
        owner.close(check=True)
        remote.close(check=True)
    finally:
        owner.close()
        if remote is not None:
            remote.close()


@pytest.mark.parametrize("phase", ("active", "admitted", "pending"))
def test_sql_orphan_waits_for_explicit_recovery(tmp_path: Path, phase: str) -> None:
    owner = _RuntimeProcess(tmp_path / "runtime.db", phase=phase)
    remote = None
    try:
        owner.crash()
        remote = _RuntimeProcess(
            tmp_path / "runtime.db", phase=phase, execution_id=owner.initial["execution_id"],
        )
        assert remote.initial["generation"] == owner.initial["generation"]
        assert remote.initial["model_calls"] == 0
        remote.request("recover")
        resumed = remote.request("await_model")
        assert resumed["generation"] > owner.initial["generation"]
        assert resumed["model_calls"] == 1
        assert remote.request("finish")["status"] == "SUCCEEDED"
        remote.close(check=True)
    finally:
        owner.close()
        if remote is not None:
            remote.close()


def test_explicit_recovery_finishes_orphaned_cancel_without_restarting_model(tmp_path: Path) -> None:
    owner = _RuntimeProcess(tmp_path / "runtime.db")
    remote = None
    try:
        remote = _RuntimeProcess(
            tmp_path / "runtime.db", execution_id=owner.initial["execution_id"],
        )
        cancelled = remote.request("cancel")
        assert cancelled["status"] == "CANCELLING"
        assert cancelled["cancelled"] is False
        owner.crash()
        recovered = remote.request("recover")
        assert recovered["status"] == "CANCELLED"
        assert remote.request("result") == {
            "status": "CANCELLED", "output": None, "model_calls": 0,
        }
        remote.close(check=True)
    finally:
        owner.close()
        if remote is not None:
            remote.close()


def test_sql_startup_completes_a_prepared_terminal_handoff(tmp_path: Path) -> None:
    owner = _RuntimeProcess(tmp_path / "runtime.db", phase="handoff")
    remote = None
    try:
        remote = _RuntimeProcess(
            tmp_path / "runtime.db", phase="handoff", execution_id=owner.initial["execution_id"],
        )
        assert owner.process.is_alive()
        assert remote.initial["status"] == "SUCCEEDED"
        assert remote.initial["generation"] == owner.initial["generation"]
        assert remote.request("result") == {
            "status": "SUCCEEDED", "output": {"text": "owner answer"}, "model_calls": 0,
        }
        assert owner.request("finish")["status"] == "SUCCEEDED"
        owner.close(check=True)
        remote.close(check=True)
    finally:
        owner.close()
        if remote is not None:
            remote.close()


def test_filesystem_execution_route_preserves_automatic_crash_recovery(tmp_path: Path) -> None:
    owner = _RuntimeProcess(tmp_path / "runtime.db", phase="filesystem")
    remote = None
    try:
        owner.crash()
        remote = _RuntimeProcess(
            tmp_path / "runtime.db", phase="filesystem", execution_id=owner.initial["execution_id"],
        )
        assert remote.initial["generation"] > owner.initial["generation"]
        assert remote.request("await_model")["model_calls"] == 1
        assert remote.request("finish")["status"] == "SUCCEEDED"
        remote.close(check=True)
    finally:
        owner.close()
        if remote is not None:
            remote.close()


def test_explicit_recovery_wins_before_original_worker_installation(tmp_path: Path) -> None:
    owner = _RuntimeProcess(tmp_path / "runtime.db", phase="admitted")
    remote = None
    try:
        remote = _RuntimeProcess(
            tmp_path / "runtime.db", phase="admitted",
            execution_id=owner.initial["execution_id"],
        )
        remote.request("recover")
        winner = remote.request("await_model")
        assert winner["model_calls"] == 1
        late_owner = owner.request("release_admission")
        assert late_owner["status"] == "STARTED"
        assert late_owner["model_calls"] == 0
        assert late_owner["generation"] == winner["generation"]
        assert late_owner["claim_id"] == winner["claim_id"]
        assert remote.request("finish")["status"] == "SUCCEEDED"
        assert owner.request("result") == {
            "status": "SUCCEEDED", "output": {"text": "owner answer"}, "model_calls": 0,
        }
        owner.close(check=True)
        remote.close(check=True)
    finally:
        owner.close()
        if remote is not None:
            remote.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("same_session", (False, True))
async def test_sql_runtime_session_admission_remains_independent(
    tmp_path: Path, same_session: bool,
) -> None:
    entered = (asyncio.Event(), asyncio.Event())
    release = asyncio.Event()
    calls = []

    async def model(messages, info):
        del messages, info
        index = len(calls)
        calls.append(index)
        entered[index].set()
        await release.wait()
        yield f"answer-{index}"

    first_state = RuntimeStorage.sqlite(tmp_path / "runtime.db")
    second_state = RuntimeStorage.sqlite(tmp_path / "runtime.db")
    try:
        async with Runtime.open(
            _NAMESPACE, models=_Models(FunctionModel(stream_function=model)),
            storage=first_state, capabilities=(_capabilities(),),
        ) as first_runtime, Runtime.open(
            _NAMESPACE, models=_Models(FunctionModel(stream_function=model)),
            storage=second_state, capabilities=(_capabilities(),),
        ) as second_runtime:
            await first_runtime.agents.get("default").create_session("first")
            if not same_session:
                await second_runtime.agents.get("default").create_session("second")
            first = await first_runtime.agents.get("default").session("first").start(
                "First", idempotency_key="first",
            )
            await asyncio.wait_for(entered[0].wait(), 10)
            if same_session:
                with pytest.raises(AIError) as caught:
                    await second_runtime.agents.get("default").session("first").start(
                        "Second", idempotency_key="second",
                    )
                assert caught.value.code is ErrorCode.SESSION_BUSY
                second = None
                assert len(calls) == 1
            else:
                second = await second_runtime.agents.get("default").session("second").start(
                    "Second", idempotency_key="second",
                )
                await asyncio.wait_for(entered[1].wait(), 10)
                assert first.execution_id != second.execution_id
                assert len(calls) == 2
            release.set()
            assert (await first.wait(timeout_seconds=10)).result.status is ExecutionStatus.SUCCEEDED
            if second is not None:
                assert (await second.wait(timeout_seconds=10)).result.status is ExecutionStatus.SUCCEEDED
    finally:
        release.set()


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ("active", "admitted"))
async def test_explicit_recovery_does_not_adopt_a_newer_contender_revision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, phase: str,
) -> None:
    owner = _RuntimeProcess(tmp_path / "runtime.db", phase=phase)
    original = owner.initial
    owner.crash()
    owner.close()
    entered = asyncio.Event()
    release_model = asyncio.Event()
    stale_ready = asyncio.Event()
    release_stale = asyncio.Event()
    calls = []
    stale_task = None

    async def model(messages, info):
        del messages, info
        calls.append(1)
        entered.set()
        await release_model.wait()
        yield "recovered"

    first_state = RuntimeStorage.sqlite(tmp_path / "runtime.db")
    winner_state = RuntimeStorage.sqlite(tmp_path / "runtime.db")
    try:
        async with Runtime.open(
            _NAMESPACE, models=_Models(FunctionModel(stream_function=model)),
            storage=first_state, capabilities=(_capabilities(),),
        ) as first_runtime, Runtime.open(
            _NAMESPACE, models=_Models(FunctionModel(stream_function=model)),
            storage=winner_state, capabilities=(_capabilities(),),
        ) as winner_runtime:
            backend = first_runtime._execution_service.runtime_backend()
            recover = backend.recover_execution

            async def paused_recover(*args, **kwargs):
                stale_ready.set()
                await release_stale.wait()
                return await recover(*args, **kwargs)

            monkeypatch.setattr(backend, "recover_execution", paused_recover)
            execution_id = original["execution_id"]
            stale_task = asyncio.create_task(first_runtime.executions.recover(
                execution_id, principal=first_runtime.default_principal,
            ))
            await asyncio.wait_for(stale_ready.wait(), 10)
            await winner_runtime.executions.recover(
                execution_id, principal=winner_runtime.default_principal,
            )
            await asyncio.wait_for(entered.wait(), 10)
            tenant = winner_runtime.default_principal.tenant_id
            winner_head = await winner_state.execution.executions.get_history_head(
                execution_id, tenant_id=tenant,
            )
            release_stale.set()
            with pytest.raises(AIError) as stale:
                await stale_task
            assert stale.value.code is ErrorCode.STORAGE_CONFLICT
            with pytest.raises(AIError) as active:
                await winner_runtime.executions.recover(
                    execution_id, principal=winner_runtime.default_principal,
                )
            assert active.value.code is ErrorCode.STORAGE_CONFLICT
            current_head = await winner_state.execution.executions.get_history_head(
                execution_id, tenant_id=tenant,
            )
            assert current_head.producer_generation == winner_head.producer_generation
            assert current_head.producer_claim_id == winner_head.producer_claim_id
            assert len(calls) == 1
            release_model.set()
            waited = await winner_runtime.executions.wait(
                execution_id, principal=winner_runtime.default_principal, timeout_seconds=10,
            )
            result = waited.result
            assert result.status is ExecutionStatus.SUCCEEDED
            assert result.output == {"text": "recovered"}
    finally:
        release_model.set()
        release_stale.set()
        if stale_task is not None:
            await asyncio.gather(stale_task, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_through_original", (False, True))
async def test_superseded_live_producer_hands_wait_and_watch_to_winner(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, cancel_through_original: bool,
) -> None:
    owner_entered = asyncio.Event()
    emit_prefix = asyncio.Event()
    prefix_observed = asyncio.Event()
    release_owner = asyncio.Event()
    winner_entered = asyncio.Event()
    release_winner = asyncio.Event()
    calls = []
    observed = []
    consumers = []

    async def original_model(messages, info):
        del messages, info
        calls.append("owner")
        owner_entered.set()
        await emit_prefix.wait()
        yield "superseded prefix"
        await release_owner.wait()
        yield "discarded continuation"

    async def recovered_model(messages, info):
        del messages, info
        calls.append("winner")
        winner_entered.set()
        await release_winner.wait()
        yield "winner answer"

    owner_state = RuntimeStorage.sqlite(tmp_path / "runtime.db")
    winner_state = RuntimeStorage.sqlite(tmp_path / "runtime.db")
    try:
        async with Runtime.open(
            _NAMESPACE, models=_Models(FunctionModel(stream_function=original_model)),
            storage=owner_state, capabilities=(_capabilities(),),
        ) as owner, Runtime.open(
            _NAMESPACE, models=_Models(FunctionModel(stream_function=recovered_model)),
            storage=winner_state, capabilities=(_capabilities(),),
        ) as winner:
            execution = await owner.agents.get("default").start("Wait", idempotency_key="start")
            await asyncio.wait_for(owner_entered.wait(), 10)

            async def watch() -> None:
                async for item in execution.watch(include_content=True):
                    observed.append(item.event)
                    if item.event.payload.get("text") == "superseded prefix":
                        prefix_observed.set()

            watched = asyncio.create_task(watch())
            waited = asyncio.create_task(execution.wait(timeout_seconds=30))
            consumers.extend((watched, waited))
            emit_prefix.set()
            await asyncio.wait_for(prefix_observed.wait(), 10)
            backend = owner._execution_service.runtime_backend()
            original_worker = backend._tasks[execution.execution_id]
            original_pending = tuple(backend._pending_audit_events.get(execution.execution_id, ()))
            if cancel_through_original:
                assert original_pending, "The original producer must have an uncommitted audit prefix"
            await winner.executions.recover(
                execution.execution_id, principal=winner.default_principal,
            )
            await asyncio.wait_for(winner_entered.wait(), 10)
            tenant = owner.default_principal.tenant_id
            claimed_head = await winner_state.execution.executions.get_history_head(
                execution.execution_id, tenant_id=tenant,
            )
            if cancel_through_original:
                before_cancel = await winner_state.execution.executions.get(
                    execution.execution_id, tenant_id=tenant,
                )
                cancelled = await owner.executions.cancel(
                    execution.execution_id,
                    CancelExecutionRequest(owner.default_principal, "cancel-through-superseded-runtime"),
                )
                assert cancelled.cancelled is False
                cancel_tail = await winner_state.execution.events.list(
                    execution.execution_id, tenant_id=tenant,
                    after_event_seq=before_cancel.event_seq, limit=100,
                )
                assert [event.event_type for event in cancel_tail.items] == [ExecutionEventType.CANCEL_REQUESTED]
                assert not any(
                    event.event_type == pending.event_type and event.payload == pending.payload
                    for event in cancel_tail.items for pending in original_pending
                )
            release_owner.set()
            await asyncio.wait_for(asyncio.shield(original_worker), 10)
            current = await winner.executions.inspect(
                execution.execution_id, principal=winner.default_principal,
            )
            expected_status = ExecutionStatus.CANCELLING if cancel_through_original else ExecutionStatus.STARTED
            assert current.status is expected_status
            current_head = await winner_state.execution.executions.get_history_head(
                execution.execution_id, tenant_id=tenant,
            )
            assert current_head.producer_generation == claimed_head.producer_generation
            assert current_head.producer_claim_id == claimed_head.producer_claim_id
            assert current_head.state.value == "open"
            if watched.done():
                await watched
                pytest.fail("Watch ended before the winning producer reached its boundary")
            assert not waited.done()
            assert owner_state.run_store._projection_dirty == set()
            assert owner_state.run_store._durability_flights == {}
            assert owner_state.run_store._execution_producers == {}
            assert owner_state.run_store._staging._runs == {}
            release_winner.set()
            result = (await waited).result
            await asyncio.wait_for(watched, 10)
            assert result.status is (
                ExecutionStatus.CANCELLED if cancel_through_original else ExecutionStatus.SUCCEEDED
            )
            assert result.output == (None if cancel_through_original else {"text": "winner answer"})
            assert calls == ["owner", "winner"]
            assert any(event.event_type == ExecutionEventType.EXECUTION_RESUMED for event in observed)
            assert observed[-1].event_type == (
                ExecutionEventType.EXECUTION_CANCELLED if cancel_through_original else ExecutionEventType.EXECUTION_SUCCEEDED
            )
            sequences = [event.durable_seq for event in observed if event.durable_seq is not None]
            assert sequences == sorted(set(sequences))
        assert not any(record.levelname == "ERROR" for record in caplog.records)
    finally:
        emit_prefix.set()
        release_owner.set()
        release_winner.set()
        for consumer in consumers:
            if not consumer.done():
                consumer.cancel()
        await asyncio.gather(*consumers, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "sqlite"))
async def test_recovery_required_transition_keeps_the_original_producer_fence(
    tmp_path: Path, backend: str,
) -> None:
    state = RuntimeStorage.in_memory() if backend == "memory" else RuntimeStorage.sqlite(tmp_path / "runtime.db")
    await state.initialize(namespace=_NAMESPACE, tenant_id="tenant")
    try:
        execution = _execution(datetime.now(timezone.utc))
        await state.execution.executions.create_with_history_head(execution)
        original_head = await state.execution.executions.get_history_head(
            execution.execution_id, tenant_id="tenant",
        )
        commands = _recovery_commands(state)
        resumed = await commands.commit_resumed(execution)
        with pytest.raises(AIError) as stale:
            await commands.commit_recovery_required(
                resumed, error_code=ErrorCode.TOOL_EFFECT_UNKNOWN.value,
                safe_error_details={}, producer_generation=original_head.producer_generation,
            )
        assert stale.value.code is ErrorCode.STORAGE_CONFLICT
        assert await state.execution.executions.get(execution.execution_id, tenant_id="tenant") == resumed
        winner_head = await state.execution.executions.get_history_head(
            execution.execution_id, tenant_id="tenant",
        )
        recovered = await commands.commit_recovery_required(
            resumed, error_code=ErrorCode.TOOL_EFFECT_UNKNOWN.value,
            safe_error_details={}, producer_generation=winner_head.producer_generation,
        )
        assert recovered.status is ExecutionStatus.RECOVERY_REQUIRED
    finally:
        await state.close()


@pytest.mark.asyncio
async def test_concurrent_sql_startup_converges_on_one_deferred_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from linktools.ai.runtime._local import LocalExecutionBackend
    from linktools.ai.workspace import Workspace, WorkspacePolicy, ToolPermissionPolicy

    from .test_runtime_approval_composition import _ToolModels

    workspace = Workspace.load(
        tmp_path, policy=WorkspacePolicy(tool_permissions=ToolPermissionPolicy(default="ask")),
    )
    application = CapabilityGroup("application")
    application.agent("default", model="default", allow_tools=("read_file",))
    capabilities = (CapabilityGroup("workspace", workspace=workspace), application)
    original_state = RuntimeStorage.sqlite(tmp_path / "runtime.db")
    async with Runtime.open(
        _NAMESPACE, models=_ToolModels(), storage=original_state, capabilities=capabilities,
    ) as original:
        execution = await original.agents.get("default").start("read a file")
        tenant = original.default_principal.tenant_id
        waiting = await _wait_for_committed(
            lambda: original_state.execution.executions.get(execution.execution_id, tenant_id=tenant),
            lambda record: record is not None and record.status is ExecutionStatus.WAITING_DEFERRED,
        )
        worker = original._execution_service.runtime_backend()._tasks.get(execution.execution_id)
        if worker is not None:
            await asyncio.wait_for(asyncio.shield(worker), 10)
        approvals = await original_state.recovery.approvals.list_pending(execution.execution_id, tenant_id=tenant)
        assert len(approvals) == 1
        await original_state.recovery.approvals.decide(
            approvals[0].approval_id, tenant_id=tenant, expected_status=ApprovalStatus.PENDING,
            idempotency_key_digest="a" * 64, decision=ApprovalDecision.DENY,
            principal_id=original.default_principal.principal_id, decision_digest="b" * 64,
            decided_at=datetime.now(timezone.utc),
        )

    both_claiming = asyncio.Event()
    claims = 0
    original_claim = LocalExecutionBackend.claim_deferred_resume

    async def synchronized_claim(backend, checkpoint, current):
        nonlocal claims
        claims += 1
        if claims == 2:
            both_claiming.set()
        await asyncio.wait_for(both_claiming.wait(), 10)
        return await original_claim(backend, checkpoint, current)

    monkeypatch.setattr(LocalExecutionBackend, "claim_deferred_resume", synchronized_claim)
    states = [RuntimeStorage.sqlite(tmp_path / "runtime.db") for _ in range(2)]
    async with AsyncExitStack() as stack:
        opened = await asyncio.gather(*(stack.enter_async_context(Runtime.open(
            _NAMESPACE, models=_ToolModels(), storage=state, capabilities=capabilities,
        )) for state in states), return_exceptions=True)
        assert not any(isinstance(value, BaseException) for value in opened), opened
        assert claims == 2
        runtime = opened[0]
        finished = await runtime.executions.wait(
            execution.execution_id, principal=runtime.default_principal, timeout_seconds=10,
        )
        assert finished.result.status is ExecutionStatus.SUCCEEDED
        current = await states[0].execution.executions.get(execution.execution_id, tenant_id=tenant)
        assert current.agent_run_seq == waiting.agent_run_seq + 1
