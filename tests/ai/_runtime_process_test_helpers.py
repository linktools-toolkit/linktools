#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from linktools.ai.persistence import Persistence
"""Independent Runtime processes for live ownership and crash recovery tests."""
from ._runtime_test_helpers import model_features

import asyncio
import multiprocessing
import signal
from dataclasses import replace
from multiprocessing.connection import Connection
from pathlib import Path
import traceback

from pydantic_ai.models.function import FunctionModel

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime import RuntimeFeatures
from linktools.ai.capability import ExecutionCapability, TraceCapability
from pydantic_ai_harness.step_persistence import StepPersistence
from tests.ai._runtime_test_helpers import native_checkpoint
from linktools.ai.runtime import Runtime
from linktools.ai.persistence import PersistenceStorage, PersistencePlan
from linktools.ai.storage import StorageRoute
from linktools.ai.execution import CancelExecutionRequest

from ._session_tool_test_helpers import _split_sqlite_storage
from ._runtime_test_helpers import _Models


_NAMESPACE = "recovery-ownership"


def _capabilities(
    *, conversation: bool = False, native_database: Path | None = None,
) -> RuntimeFeatures:
    group = RuntimeFeatures(_NAMESPACE)
    if conversation:
        group.capability(ExecutionCapability(retain_history=True), scope="runtime")
    group.capability(native_checkpoint(database=native_database), scope="runtime")
    group.capability(TraceCapability(), scope="runtime")
    group.agent("default", model="default", allow_tools=())
    return group


def _storage(database: Path, phase: str) -> PersistenceStorage:
    if phase in {"pending", "handoff", "finalizing"}:
        return _split_sqlite_storage(database)
    if phase == "local_sqlite":
        execution = StorageRoute.sqlite(database.with_suffix(".execution.db"), exclusive_writer=True)
        plan = PersistencePlan(
            conversation=StorageRoute.sqlite(database.with_suffix(".conversation.db")),
            execution=execution,
        )
        return PersistenceStorage.from_plan(replace(plan, execution=execution.with_cache(None)))
    return PersistenceStorage.sqlite(database)


def _runtime_process(
    connection: Connection, database: str, mode: str, phase: str,
    execution_id: str | None, auto_recover: bool, recovery_barrier, publication_order=None,
    native_database: Path | None = None,
) -> None:
    async def run() -> None:
        nonlocal execution_id
        stopping = asyncio.Event()
        if phase == "graceful":
            asyncio.get_running_loop().add_signal_handler(signal.SIGTERM, stopping.set)
        entered = asyncio.Event()
        release_model = asyncio.Event()
        admitted = asyncio.Event()
        release_admission = asyncio.Event()
        calls = 0
        start_task = None
        if phase in {"handoff", "finalizing"}:
            release_model.set()

        async def model(messages, info):
            nonlocal calls
            del messages, info
            calls += 1
            entered.set()
            await release_model.wait()
            yield "owner answer"

        state = _storage(Path(database), phase)
        async with Persistence.open(_NAMESPACE, storage=state) as persistence_77, Runtime.attach(persistence=persistence_77, features=model_features(_Models(FunctionModel(stream_function=model)), _capabilities(conversation=mode == 'session', native_database=native_database)), auto_recover=auto_recover) as runtime:
            async def snapshot() -> dict[str, object]:
                assert execution_id is not None
                tenant = runtime.default_principal.tenant_id
                execution = await runtime._state.execution.executions.get(execution_id, tenant_id=tenant)
                assert execution is not None
                checkpoint = await runtime._state.execution.executions.get(execution_id, tenant_id=tenant)
                session = await runtime._state.conversation.sessions.get("session", tenant_id=tenant)
                return {
                    "handoff_phase": None if checkpoint is None or checkpoint.recovery is None else checkpoint.recovery.handoff_phase.value,
                    "active_execution_id": None if session is None else session.active_execution_id,
                    "execution_id": execution_id, "status": execution.status.value,
                    "revision": execution.revision, "generation": execution.producer_generation,
                    "claim_id": execution.producer_claim_id, "model_calls": calls,
                }

            if recovery_barrier is not None:
                backend = runtime._execution_service.runtime_backend()
                original_reconcile = backend._reconcile_handoff
                rendezvous = True

                async def synchronized_reconcile(checkpoint):
                    nonlocal rendezvous
                    if rendezvous:
                        rendezvous = False
                        await asyncio.to_thread(recovery_barrier.wait, 15)
                    return await original_reconcile(checkpoint)

                backend._reconcile_handoff = synchronized_reconcile
            if publication_order is not None:
                role, prepared, cursor_committed, publication_finished = publication_order
                backend = runtime._execution_service.runtime_backend()
                original_terminal = backend._commit_reconciled_terminal
                original_conversation = backend._resolve_handoff_conversation
                waiting = True

                async def ordered_conversation(checkpoint, handoff):
                    nonlocal waiting
                    if not waiting:
                        return await original_conversation(checkpoint, handoff)
                    waiting = False
                    if role == "delayed":
                        prepared.set()
                        assert await asyncio.to_thread(cursor_committed.wait, 15)
                    else:
                        assert await asyncio.to_thread(prepared.wait, 15)
                    try:
                        return await original_conversation(checkpoint, handoff)
                    finally:
                        if role == "delayed":
                            publication_finished.set()

                async def ordered_terminal(*args, **kwargs):
                    if role == "first":
                        cursor_committed.set()
                        assert await asyncio.to_thread(publication_finished.wait, 15)
                    return await original_terminal(*args, **kwargs)

                # Conversation content is durable before PREPARED. Race the real
                # cursor reconciliation boundary, before terminal/result commit.
                backend._resolve_handoff_conversation = ordered_conversation
                backend._commit_reconciled_terminal = ordered_terminal
            if execution_id is None:
                backend = runtime._execution_service.runtime_backend()
                if phase == "reserved":
                    # This crash boundary belongs to independently reserved starts.
                    runtime._execution_service._runtime_bridge.prepare_reserved_start = None
                    original_prepare = backend.prepare_start

                    async def pause_prepare(request, execution, identity):
                        nonlocal execution_id
                        execution_id = execution.execution_id
                        admitted.set()
                        await release_admission.wait()
                        return await original_prepare(request, execution, identity)

                    backend.prepare_start = pause_prepare
                elif phase == "admitted":
                    original_launch = backend.launch

                    async def pause_launch(request, execution, **kwargs):
                        nonlocal execution_id
                        execution_id = execution.execution_id
                        admitted.set()
                        await release_admission.wait()
                        return await original_launch(request, execution, **kwargs)

                    backend.launch = pause_launch
                elif phase == "pending":
                    original_claim = runtime._state.execution.executions.claim_start

                    async def pause_claim(claim):
                        nonlocal execution_id
                        execution_id = claim.execution_id
                        admitted.set()
                        await release_admission.wait()
                        return await original_claim(claim)

                    runtime._state.execution.executions.claim_start = pause_claim
                elif phase == "finalizing":
                    original_finalizing = backend._claim_session_or_recovery_finalizing

                    async def pause_finalizing(execution):
                        nonlocal execution_id
                        current = await original_finalizing(execution)
                        execution_id = current.execution_id
                        admitted.set()
                        await release_admission.wait()
                        return current

                    backend._claim_session_or_recovery_finalizing = pause_finalizing
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
                if phase in {"active", "local_sqlite", "graceful"}:
                    execution = await start_task
                    execution_id = execution.execution_id
                    await asyncio.wait_for(entered.wait(), 10)
                else:
                    await asyncio.wait_for(admitted.wait(), 10)
            connection.send(await snapshot())
            if phase == "graceful":
                await stopping.wait()
                await runtime.handoff()
                return
            while True:
                command = await asyncio.to_thread(connection.recv)
                if command == "snapshot":
                    connection.send(await snapshot())
                elif command == "replay_start":
                    agent = runtime.agents.get("default")
                    handle = (
                        await agent.session("session").start("Wait", idempotency_key="start")
                        if mode == "session"
                        else await agent.start("Wait", idempotency_key="start")
                    )
                    assert handle.execution_id == execution_id
                    backend = runtime._execution_service.runtime_backend()
                    failure = backend.worker_failure(
                        execution_id, tenant_id=runtime.default_principal.tenant_id,
                    )
                    connection.send({
                        **await snapshot(),
                        "worker_installed": backend.worker_installed(execution_id),
                        "local_producer": backend._live_broker.is_local_producer(execution_id),
                        "worker_failure": None if failure is None else failure.code.value,
                    })
                elif command == "recover_before_publication":
                    backend = runtime._execution_service.runtime_backend()
                    original_validate = backend._validate_committed_handoff

                    async def unavailable(checkpoint):
                        raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)

                    backend._validate_committed_handoff = unavailable
                    try:
                        await runtime.executions.recover(execution_id,
                            principal=runtime.default_principal, idempotency_key="owner-handoff")
                    except AIError as error:
                        connection.send({**await snapshot(), "error_code": error.code.value})
                    else:
                        raise AssertionError("unfinished publication must not be acknowledged")
                    finally:
                        backend._validate_committed_handoff = original_validate
                elif command == "try_recover_keyed":
                    try:
                        await runtime.executions.recover(execution_id,
                            principal=runtime.default_principal, idempotency_key="owner-handoff")
                    except AIError as error:
                        connection.send({"error_code": error.code.value, "details": dict(error.safe_details), "traceback": traceback.format_exc()})
                    else:
                        connection.send(await snapshot())
                elif command == "history":
                    from pydantic_ai.messages import TextPart
                    page = await runtime.history.messages("session", principal=runtime.default_principal)
                    connection.send({"answers": [part.content for message in page.items
                        for part in message.parts if isinstance(part, TextPart)]})
                elif command in {"recover", "recover_keyed"}:
                    await runtime.executions.recover(
                        execution_id, principal=runtime.default_principal,
                        idempotency_key="owner-handoff" if command == "recover_keyed" else None,
                    )
                    connection.send(await snapshot())
                elif command == "await_model":
                    await asyncio.wait_for(entered.wait(), 10)
                    connection.send(await snapshot())
                elif command == "cancel_events":
                    events = await runtime._state.execution.events.list(
                        execution_id, tenant_id=runtime.default_principal.tenant_id,
                        after_event_seq=0, limit=100,
                    )
                    assert events.next_cursor is None
                    connection.send({
                        "requested": sum(
                            event.event_type == "CANCEL_REQUESTED" for event in events.items
                        ),
                    })
                elif command == "flush_trace":
                    await runtime._run_store.flush_dirty_execution_projections(
                        execution_id=execution_id,
                    )
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
        connection.send({"error": type(error).__name__, "code": error.code.value if isinstance(error, AIError) else None,
                         "traceback": traceback.format_exc()})
    finally:
        connection.close()


class _RuntimeProcess:
    def __init__(
        self, database: Path, *, mode: str = "session", phase: str = "active",
        execution_id: str | None = None, auto_recover: bool = True, recovery_barrier=None,
        publication_order=None, native_database: Path | None = None,
    ) -> None:
        context = multiprocessing.get_context("spawn")
        self.connection, child = context.Pipe()
        self.process = context.Process(
            target=_runtime_process,
            args=(child, str(database), mode, phase, execution_id, auto_recover, recovery_barrier, publication_order, native_database),
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
