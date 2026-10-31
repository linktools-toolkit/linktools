#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from linktools.ai.persistence import Persistence
"""Remote terminal ownership releases the previous producer's local tail."""
from ._runtime_test_helpers import model_features

import asyncio
from pathlib import Path

import pytest
from pydantic_ai.models.function import FunctionModel

from linktools.ai.runtime import RuntimeFeatures
from linktools.ai.core import ExecutionStatus
from linktools.ai.capability import ExecutionCapability, TraceCapability
from linktools.ai.runtime import Runtime
from linktools.ai.persistence import PersistenceStorage
from linktools.ai.storage import InMemoryCacheBackend
from linktools.ai.persistence import PersistenceDomain
from linktools.ai.core import CommitPolicy
from linktools.ai.runtime._local import LocalExecutionBackend
from linktools.ai.execution import CancelExecutionRequest

from .test_live_history_readback_integration import _Models


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_at", ("provider", "success"))
@pytest.mark.parametrize("watch_owner", (False, True))
async def test_remote_cancel_releases_revoked_producer_staging(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch,
    cancel_at: str, watch_owner: bool,
) -> None:
    entered = asyncio.Event()
    release = asyncio.Event()
    watchers: list[asyncio.Task[None]] = []
    model_exited = asyncio.Event()
    success_exited = asyncio.Event()

    async def model(messages, info):
        del messages, info
        try:
            if cancel_at == "provider":
                yield "received "
                entered.set()
                await release.wait()
            yield "late answer"
        finally:
            model_exited.set()

    if cancel_at == "success":
        original_success = LocalExecutionBackend._commit_success

        async def pause_success(backend, *args, **kwargs):
            try:
                entered.set()
                await release.wait()
                return await original_success(backend, *args, **kwargs)
            finally:
                success_exited.set()

        monkeypatch.setattr(LocalExecutionBackend, "_commit_success", pause_success)

    owner_storage = PersistenceStorage.sqlite(tmp_path / "runtime.db")
    group = RuntimeFeatures("remote-cancel-release")
    group.capability(ExecutionCapability(retain_history=True), scope="runtime")
    group.capability(TraceCapability(commit=CommitPolicy.batch(buffer=InMemoryCacheBackend(max_bytes=65536), batch_size=128, flush_interval=3.0)), scope="runtime")
    group.agent("default", model="default", allow_tools=())
    remote_storage = PersistenceStorage.sqlite(tmp_path / "runtime.db")
    try:
        async with Persistence.open('remote-cancel-release', storage=owner_storage) as persistence_69, Runtime.attach(persistence=persistence_69, features=model_features(_Models(FunctionModel(stream_function=model)), group)) as owner, Persistence.open('remote-cancel-release', storage=remote_storage) as persistence_72, Runtime.attach(persistence=persistence_72, features=model_features(_Models(FunctionModel(stream_function=model)), group)) as remote:
            await owner.agents.get("default").create_session("session")
            session = owner.agents.get("default").session("session")
            for turn in range(3):
                entered.clear()
                release.clear()
                model_exited.clear()
                success_exited.clear()
                execution = await session.start("Wait", idempotency_key=f"turn-{turn}")
                await asyncio.wait_for(entered.wait(), 10)
                observed = []
                prefix_observed = asyncio.Event()
                watched = None
                if watch_owner:
                    async def watch() -> None:
                        async for item in execution.watch(include_content=True):
                            observed.append(item.event)
                            if item.event.payload.get("text") == (
                                "received " if cancel_at == "provider" else "late answer"
                            ):
                                prefix_observed.set()

                    watched = asyncio.create_task(watch())
                    watchers.append(watched)
                    await asyncio.wait_for(prefix_observed.wait(), 10)
                outcome = await remote.executions.cancel(
                    execution.execution_id,
                    CancelExecutionRequest(owner.default_principal, f"remote-stop-{turn}"),
                )
                assert outcome.cancelled is False
                tenant = owner.default_principal.tenant_id
                pending_head = await remote._state.execution.executions.get_history_head(
                    execution.execution_id, tenant_id=tenant,
                )
                await execution.history(include_content=True)
                while_active = await owner.executions.inspect(execution.execution_id, principal=owner.default_principal)
                assert while_active.status is ExecutionStatus.CANCELLING
                assert pending_head.state.value == "open"
                assert owner._run_store._staging._runs
                backend = owner._execution_service.runtime_backend()
                worker = backend._tasks[execution.execution_id]
                run_stores = (owner._run_store, remote._run_store)
                observation_tasks = {
                    buffer.scope._writer
                    for store in run_stores for buffer in store.trace_persistence._buffers.values()
                    if buffer.scope is not None and buffer.scope._writer is not None
                }
                release.set()
                worker_outcomes = await asyncio.wait_for(
                    asyncio.gather(asyncio.shield(worker), return_exceptions=True), 10,
                )
                assert worker_outcomes[0] is None or isinstance(worker_outcomes[0], asyncio.CancelledError)
                await asyncio.wait_for(model_exited.wait(), 10)
                if cancel_at == "success":
                    await asyncio.wait_for(success_exited.wait(), 10)
                if watched is not None:
                    from linktools.ai.core import ExecutionEventType

                    await asyncio.wait_for(watched, 10)
                    assert observed[-1].event_type == ExecutionEventType.EXECUTION_CANCELLED
                    sequences = [item.durable_seq for item in observed if item.durable_seq is not None]
                    assert sequences == sorted(set(sequences))
                async def released() -> None:
                    while owner._execution_service._handoff._states:
                        await asyncio.sleep(0.01)
                await asyncio.wait_for(released(), 10)
                result = await remote.executions.result(execution.execution_id, principal=owner.default_principal)
                sealed = await remote._state.execution.executions.get_history_head(
                    execution.execution_id, tenant_id=tenant,
                )
                history = await execution.history(include_content=True)
                assert [item.content for item in history.items if item.item_kind == "assistant"] == [
                    "received late answer" if cancel_at == "provider" else "late answer"
                ]
                assert result.status is ExecutionStatus.CANCELLED
                assert result.output is None
                async def settled() -> None:
                    joined = set()
                    while True:
                        tasks = observation_tasks | backend._checkpoint_tasks
                        for store in run_stores:
                            tasks.update(store._background_tasks)
                            tasks.update(
                                buffer.scope._writer for buffer in store.trace_persistence._buffers.values()
                                if buffer.scope is not None and buffer.scope._writer is not None
                            )
                        tasks.difference_update(joined)
                        if not tasks:
                            return
                        await asyncio.gather(*(asyncio.shield(task) for task in tasks))
                        joined.update(tasks)

                await asyncio.wait_for(settled(), 10)
                assert await owner._state.execution.executions.get_history_head(
                    execution.execution_id, tenant_id=tenant,
                ) == sealed
                assert (await execution.history(include_content=True)).items == history.items
                run_store = owner._run_store
                assert run_store.trace_persistence.dirty == set()
                assert run_store.trace_persistence.offsets == {}
                assert run_store._durability_flights == {}
                assert run_store._execution_producers == {}
                assert run_store._staging._runs == {}
                assert run_store._staging._events == {}
                assert run_store._staging._interactions == {}
                assert run_store._staging._payloads == {}
                assert owner._execution_service._handoff._states == {}
                assert backend._terminal_events == {}
                assert backend._pending_audit_locks == {}
                assert backend._execution_durable_tasks == {}
                assert backend._worker_failures == {}
            assert not any(record.levelname == "ERROR" for record in caplog.records)
    finally:
        release.set()
        for watcher in watchers:
            if not watcher.done():
                watcher.cancel()
        await asyncio.gather(*watchers, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("sibling_settles", (True, False))
async def test_remote_cancel_settles_individual_tool_before_stopping_sibling(
    tmp_path: Path, sibling_settles: bool,
) -> None:
    from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart

    from linktools.ai.capability import AgentContext
    from linktools.ai.core import ToolOperationStatus
    from linktools.ai.errors import AIError, ErrorCode

    from ._runtime_test_helpers import _UsageFunctionModel

    fast_entered = asyncio.Event()
    slow_entered = asyncio.Event()
    finish_fast = asyncio.Event()
    slow_stopped = asyncio.Event()
    requests = 0

    async def fast(_ctx: AgentContext[None]) -> str:
        fast_entered.set()
        await finish_fast.wait()
        return "completed effect"

    async def slow(_ctx: AgentContext[None]) -> str:
        slow_entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            if sibling_settles:
                return "settled sibling"
            raise
        finally:
            slow_stopped.set()
        return "unreachable"

    async def model(messages, info):
        nonlocal requests
        del messages, info
        requests += 1
        if requests > 1:
            return ModelResponse(parts=[TextPart("unexpected continuation")])
        return ModelResponse(parts=[
            ToolCallPart("fast", {}, tool_call_id="fast-call"),
            ToolCallPart("slow", {}, tool_call_id="slow-call"),
        ])

    group = RuntimeFeatures("individual-tool-cancel")
    group.capability(TraceCapability(), scope="runtime")
    group.tool(fast, effect_policy="replay_safe")
    group.tool(slow, effect_policy="replay_safe")
    group.agent("default", model="default", allow_tools=("fast", "slow"))
    owner_storage = PersistenceStorage.sqlite(tmp_path / "runtime.db")
    async with Persistence.open('individual-tool-cancel', storage=owner_storage) as persistence_248, Runtime.attach(persistence=persistence_248, features=model_features(_Models(_UsageFunctionModel(model)), group)) as owner, Persistence.open('individual-tool-cancel', storage=PersistenceStorage.sqlite(tmp_path / 'runtime.db')) as persistence_251, Runtime.attach(persistence=persistence_251, features=model_features(_Models(_UsageFunctionModel(model)), group)) as remote:
        execution = await owner.agents.get("default").start("run both")
        try:
            await asyncio.wait_for(asyncio.gather(fast_entered.wait(), slow_entered.wait()), 10)
            await owner._run_store.flush_dirty_execution_projections(
                execution_id=execution.execution_id,
            )
            outcome = await remote.executions.cancel(
                execution.execution_id,
                CancelExecutionRequest(owner.default_principal, "stop-at-completion"),
            )
            assert not outcome.cancelled
            assert not slow_stopped.is_set()
            finish_fast.set()
            if sibling_settles:
                result = (await execution.wait(timeout_seconds=15)).result
                assert result.status is ExecutionStatus.CANCELLED
                assert result.output is None
            else:
                with pytest.raises(AIError) as unknown:
                    await execution.wait(timeout_seconds=15)
                assert unknown.value.code is ErrorCode.TOOL_EFFECT_UNKNOWN
                view = await owner.executions.inspect(execution.execution_id, principal=owner.default_principal)
                assert view.status is ExecutionStatus.RECOVERY_REQUIRED
            assert slow_stopped.is_set()
            assert requests == 1
            operations = await owner._state.recovery.tools.list_by_execution(
                execution.execution_id, tenant_id=owner.default_principal.tenant_id,
            )
            assert next(item for item in operations if item.tool_call_id == "fast-call").status is ToolOperationStatus.COMPLETED
            if not sibling_settles:
                assert next(item for item in operations if item.tool_call_id == "slow-call").status is ToolOperationStatus.EFFECT_UNKNOWN
            history = (await execution.history(tool_call_id="fast-call", include_content=True)).items
            assert [item.item_kind for item in history] == ["tool_call", "tool_result"]
            assert history[-1].content == "completed effect"
        finally:
            finish_fast.set()


@pytest.mark.asyncio
@pytest.mark.parametrize("cas_conflict", (True, False))
async def test_history_projection_cancel_conflict_preserves_explicit_retry_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cas_conflict: bool,
) -> None:
    import sqlite3
    from contextlib import closing

    from pydantic_ai.messages import ModelResponse, ToolCallPart

    from linktools.ai.capability import AgentContext
    from linktools.ai.core import OperationStatus, idempotency_key_digest
    from linktools.ai.errors import AIError, ErrorCode
    from linktools.ai.persistence._sql import _SqlTransaction

    from ._runtime_test_helpers import _UsageFunctionModel

    entered = asyncio.Event()
    release = asyncio.Event()
    cancel_read = asyncio.Event()
    projection: asyncio.Task[None] | None = None

    async def tool(_ctx: AgentContext[None]) -> str:
        entered.set()
        await release.wait()
        return "completed effect"

    async def model(messages, info):
        del messages, info
        return ModelResponse(parts=[ToolCallPart("tool", {}, tool_call_id="call")])

    owner_storage = PersistenceStorage.sqlite(tmp_path / "runtime.db")
    group = RuntimeFeatures("projection-cancel-conflict")
    group.capability(TraceCapability(commit=CommitPolicy.batch(buffer=InMemoryCacheBackend(max_bytes=65536), batch_size=128, flush_interval=60.0)), scope="runtime")
    group.tool(tool, effect_policy="replay_safe")
    group.agent("default", model="default", allow_tools=("tool",))
    original_replace = _SqlTransaction.replace_record
    cancel_updates = 0

    async def project_before_cancel_replace(transaction, record, *, expected_storage_revision):
        nonlocal cancel_updates
        if record.kind == "execution" and record.state == "CANCELLING":
            cancel_updates += 1
        if record.kind == "execution" and record.state == "CANCELLING" and not cancel_read.is_set():
            cancel_read.set()
            assert projection is not None
            await asyncio.wait_for(asyncio.shield(projection), 10)
            if cas_conflict:
                # A nullable CAS conflict is distinct from SQLite's retryable
                # snapshot-upgrade failure on the native UPDATE path.
                return None
        return await original_replace(
            transaction, record, expected_storage_revision=expected_storage_revision,
        )

    monkeypatch.setattr(_SqlTransaction, "replace_record", project_before_cancel_replace)
    async with Persistence.open('projection-cancel-conflict', storage=owner_storage) as persistence_336, Runtime.attach(persistence=persistence_336, features=model_features(_Models(_UsageFunctionModel(model)), group)) as owner, Persistence.open('projection-cancel-conflict', storage=PersistenceStorage.sqlite(tmp_path / 'runtime.db')) as persistence_339, Runtime.attach(persistence=persistence_339, features=model_features(_Models(_UsageFunctionModel(model)), group)) as remote:
        # The paused cancellation snapshot must coexist with the projection commit.
        with closing(sqlite3.connect(tmp_path / "runtime.db")) as connection:
            assert connection.execute("PRAGMA journal_mode=WAL").fetchone() == ("wal",)

        execution = await owner.agents.get("default").start("run tool")
        try:
            await asyncio.wait_for(entered.wait(), 10)
            tenant = owner.default_principal.tenant_id
            repository = owner._state.execution.executions
            before = await repository.get(execution.execution_id, tenant_id=tenant)
            key = repository._key("execution", execution.execution_id)
            before_stored = await repository.state_store.read(lambda tx: tx.get_record(key))
            request = CancelExecutionRequest(owner.default_principal, "stop-projection-race")

            async def project_after_cancel_read() -> None:
                await cancel_read.wait()
                await owner._run_store.flush_dirty_execution_projections(
                    execution_id=execution.execution_id,
                )

            projection = asyncio.create_task(project_after_cancel_read())
            if cas_conflict:
                with pytest.raises(AIError) as conflict:
                    await remote.executions.cancel(execution.execution_id, request)
                assert conflict.value.code is ErrorCode.STORAGE_CONFLICT
                assert cancel_read.is_set()
                after = await repository.get(execution.execution_id, tenant_id=tenant)
                after_stored = await repository.state_store.read(lambda tx: tx.get_record(key))
                assert before_stored is not None and after_stored is not None
                assert after_stored.storage_revision > before_stored.storage_revision
                assert after_stored.data == before_stored.data
                assert before is not None and after is not None
                assert after.status is before.status is ExecutionStatus.STARTED
                assert (after.revision, after.event_seq) == (before.revision, before.event_seq)
                events = await owner._state.execution.events.list(
                    execution.execution_id, tenant_id=tenant, after_event_seq=0, limit=100,
                )
                assert not any(item.event_type == "CANCEL_REQUESTED" for item in events.items)

                outcome = await remote.executions.cancel(execution.execution_id, request)
            else:
                outcome = await remote.executions.cancel(execution.execution_id, request)
                assert cancel_read.is_set()
            assert not outcome.cancelled and cancel_updates == 2
            release.set()
            result = (await execution.wait(timeout_seconds=10)).result
            assert result.status is ExecutionStatus.CANCELLED
            events = await owner._state.execution.events.list(
                execution.execution_id, tenant_id=tenant, after_event_seq=0, limit=100,
            )
            assert sum(item.event_type == "CANCEL_REQUESTED" for item in events.items) == 1
            operation = await owner._state.execution.operations.get(
                idempotency_key_digest(request.idempotency_key), tenant_id=tenant,
            )
            assert operation is not None and operation.status is OperationStatus.EFFECT_UNKNOWN
            assert operation.resource_id == execution.execution_id
            repeated = await remote.executions.cancel(execution.execution_id, request)
            assert repeated.cancelled
            reconciled = await owner._state.execution.operations.get(operation.operation_id, tenant_id=tenant)
            assert reconciled is not None and reconciled.status is OperationStatus.SUCCEEDED
            repeated_events = await owner._state.execution.events.list(
                execution.execution_id, tenant_id=tenant, after_event_seq=0, limit=100,
            )
            assert repeated_events == events
        finally:
            release.set()
            if projection is not None:
                projection.cancel()
                await asyncio.gather(projection, return_exceptions=True)
