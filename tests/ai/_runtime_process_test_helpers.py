#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Independent Runtime processes for live ownership and crash recovery tests."""

import asyncio
import multiprocessing
import signal
from multiprocessing.connection import Connection
from pathlib import Path
import traceback

from pydantic_ai.models.function import FunctionModel

from linktools.ai.errors import AIError
from linktools.ai.runtime import RuntimeFeatures
from linktools.ai.capability import CheckpointCapability, ExecutionCapability, TraceCapability
from linktools.ai.runtime import Runtime, RuntimeStorage, RuntimeStoragePlan
from linktools.ai.storage import StorageRoute
from linktools.ai.runtime.service_api import CancelExecutionRequest

from ._session_tool_test_helpers import _split_sqlite_storage
from ._runtime_test_helpers import _Models


_NAMESPACE = "recovery-ownership"


def _capabilities(*, conversation: bool = False) -> RuntimeFeatures:
    group = RuntimeFeatures(_NAMESPACE)
    if conversation:
        group.capability(ExecutionCapability(retain_history=True), scope="runtime")
    group.capability(CheckpointCapability(), scope="runtime")
    group.capability(TraceCapability(), scope="runtime")
    group.agent("default", model="default", allow_tools=())
    return group


def _storage(database: Path, phase: str) -> RuntimeStorage:
    if phase in {"pending", "handoff", "finalizing"}:
        return _split_sqlite_storage(database)
    if phase == "filesystem":
        return RuntimeStorage(RuntimeStoragePlan(
            conversation=StorageRoute.sqlite(database.with_suffix(".conversation.db")),
            execution=StorageRoute.filesystem(database.with_suffix(".execution")),
        ))
    return RuntimeStorage.sqlite(database)


def _runtime_process(
    connection: Connection, database: str, mode: str, phase: str,
    execution_id: str | None, auto_recover: bool,
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
        async with Runtime.open(
            _NAMESPACE, models=_Models(FunctionModel(stream_function=model)),
            storage=state, features=_capabilities(conversation=mode == "session"),
            auto_recover=auto_recover,
        ) as runtime:
            async def snapshot() -> dict[str, object]:
                assert execution_id is not None
                tenant = runtime.default_principal.tenant_id
                execution = await runtime._state.execution.executions.get(execution_id, tenant_id=tenant)
                head = await runtime._state.execution.executions.get_history_head(execution_id, tenant_id=tenant)
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
                if phase in {"active", "filesystem", "graceful"}:
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
                elif command == "try_recover_keyed":
                    try:
                        await runtime.executions.recover(execution_id,
                            principal=runtime.default_principal, idempotency_key="owner-handoff")
                    except AIError as error:
                        connection.send({"error_code": error.code.value})
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
                    await runtime._state.run_store.flush_dirty_execution_projections(
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
        connection.send({"error": type(error).__name__, "traceback": traceback.format_exc()})
    finally:
        connection.close()


class _RuntimeProcess:
    def __init__(
        self, database: Path, *, mode: str = "session", phase: str = "active",
        execution_id: str | None = None, auto_recover: bool = True,
    ) -> None:
        context = multiprocessing.get_context("spawn")
        self.connection, child = context.Pipe()
        self.process = context.Process(
            target=_runtime_process,
            args=(child, str(database), mode, phase, execution_id, auto_recover),
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
