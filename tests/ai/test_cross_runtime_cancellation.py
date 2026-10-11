#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Remote terminal ownership releases the previous producer's local tail."""

import asyncio
from pathlib import Path

import pytest
from pydantic_ai.models.function import FunctionModel

from linktools.ai.capability import CapabilityGroup
from linktools.ai.core import ExecutionStatus
from linktools.ai.runtime import Runtime, RuntimeStorage
from linktools.ai.runtime._local import LocalExecutionBackend
from linktools.ai.runtime.service_api import CancelExecutionRequest

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

    async def model(messages, info):
        del messages, info
        if cancel_at == "provider":
            yield "received "
            entered.set()
            await release.wait()
        yield "late answer"

    if cancel_at == "success":
        original_success = LocalExecutionBackend._commit_success

        async def pause_success(backend, *args, **kwargs):
            entered.set()
            await release.wait()
            return await original_success(backend, *args, **kwargs)

        monkeypatch.setattr(LocalExecutionBackend, "_commit_success", pause_success)

    group = CapabilityGroup("remote-cancel-release")
    group.agent("default", model="default", allow_tools=())
    owner_storage = RuntimeStorage.sqlite(tmp_path / "runtime.db")
    remote_storage = RuntimeStorage.sqlite(tmp_path / "runtime.db")
    try:
        async with Runtime.open(
            "remote-cancel-release", models=_Models(FunctionModel(stream_function=model)),
            storage=owner_storage, capabilities=(group,),
        ) as owner, Runtime.open(
            "remote-cancel-release", models=_Models(FunctionModel(stream_function=model)),
            storage=remote_storage, capabilities=(group,),
        ) as remote:
            await owner.agents.get("default").create_session("session")
            session = owner.agents.get("default").session("session")
            for turn in range(3):
                entered.clear()
                release.clear()
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
                pending_head = await remote_storage.execution.executions.get_history_head(
                    execution.execution_id, tenant_id=tenant,
                )
                await execution.history(include_content=True)
                while_active = await owner.executions.inspect(execution.execution_id, principal=owner.default_principal)
                assert while_active.status is ExecutionStatus.CANCELLING
                assert pending_head.state.value == "open"
                assert owner_storage.run_store._staging._runs
                backend = owner._execution_service.runtime_backend()
                worker = backend._tasks[execution.execution_id]
                release.set()
                worker_outcomes = await asyncio.wait_for(
                    asyncio.gather(asyncio.shield(worker), return_exceptions=True), 10,
                )
                assert worker_outcomes[0] is None or isinstance(worker_outcomes[0], asyncio.CancelledError)
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
                sealed = await remote_storage.execution.executions.get_history_head(
                    execution.execution_id, tenant_id=tenant,
                )
                history = await execution.history(include_content=True)
                assert [item.content for item in history.items if item.item_kind == "assistant"] == [
                    "received late answer" if cancel_at == "provider" else "late answer"
                ]
                assert result.status is ExecutionStatus.CANCELLED
                assert result.output is None
                await asyncio.sleep(1.1)
                assert await owner_storage.execution.executions.get_history_head(
                    execution.execution_id, tenant_id=tenant,
                ) == sealed
                assert (await execution.history(include_content=True)).items == history.items
                run_store = owner_storage.run_store
                assert run_store._projection_dirty == set()
                assert run_store._projection_offsets == {}
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

    group = CapabilityGroup("individual-tool-cancel")
    group.tool(fast, effect_policy="replay_safe")
    group.tool(slow, effect_policy="replay_safe")
    group.agent("default", model="default", allow_tools=("fast", "slow"))
    owner_storage = RuntimeStorage.sqlite(tmp_path / "runtime.db")
    async with Runtime.open(
        "individual-tool-cancel", models=_Models(_UsageFunctionModel(model)),
        storage=owner_storage, capabilities=(group,),
    ) as owner, Runtime.open(
        "individual-tool-cancel", models=_Models(_UsageFunctionModel(model)),
        storage=RuntimeStorage.sqlite(tmp_path / "runtime.db"), capabilities=(group,),
    ) as remote:
        execution = await owner.agents.get("default").start("run both")
        try:
            await asyncio.wait_for(asyncio.gather(fast_entered.wait(), slow_entered.wait()), 10)
            await owner_storage.run_store.flush_dirty_execution_projections(
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
            operations = await owner_storage.recovery.tools.list_by_execution(
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
async def test_history_projection_cancel_conflict_preserves_explicit_retry_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pydantic_ai.messages import ModelResponse, ToolCallPart

    from linktools.ai.capability import AgentContext
    from linktools.ai.errors import AIError, ErrorCode
    from linktools.ai.runtime.state._sql import _SqlTransaction
    from linktools.ai.runtime.state._steps import RuntimeAgentRunStore

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

    group = CapabilityGroup("projection-cancel-conflict")
    group.tool(tool, effect_policy="replay_safe")
    group.agent("default", model="default", allow_tools=("tool",))
    owner_storage = RuntimeStorage.sqlite(tmp_path / "runtime.db")
    original_replace = _SqlTransaction.replace_record

    async def project_before_cancel_replace(transaction, record, *, expected_storage_version):
        if record.kind == "execution" and record.state == "CANCELLING" and not cancel_read.is_set():
            cancel_read.set()
            assert projection is not None
            await asyncio.wait_for(asyncio.shield(projection), 10)
        return await original_replace(
            transaction, record, expected_storage_version=expected_storage_version,
        )

    # Keep the batch pending until the cancel CAS, rather than race a timer.
    monkeypatch.setattr(RuntimeAgentRunStore, "_schedule_observation_flush", lambda self: None)
    monkeypatch.setattr(_SqlTransaction, "replace_record", project_before_cancel_replace)
    async with Runtime.open(
        "projection-cancel-conflict", models=_Models(_UsageFunctionModel(model)),
        storage=owner_storage, capabilities=(group,),
    ) as owner, Runtime.open(
        "projection-cancel-conflict", models=_Models(_UsageFunctionModel(model)),
        storage=RuntimeStorage.sqlite(tmp_path / "runtime.db"), capabilities=(group,),
    ) as remote:
        execution = await owner.agents.get("default").start("run tool")
        try:
            await asyncio.wait_for(entered.wait(), 10)
            tenant = owner.default_principal.tenant_id
            repository = owner_storage.execution.executions
            before = await repository.get(execution.execution_id, tenant_id=tenant)
            key = repository._key("execution", execution.execution_id)
            before_stored = await repository.state_store.read(lambda tx: tx.get_record(key))
            request = CancelExecutionRequest(owner.default_principal, "stop-projection-race")

            async def project_after_cancel_read() -> None:
                await cancel_read.wait()
                await owner_storage.run_store.flush_dirty_execution_projections(
                    execution_id=execution.execution_id,
                )

            projection = asyncio.create_task(project_after_cancel_read())
            with pytest.raises(AIError) as conflict:
                await remote.executions.cancel(execution.execution_id, request)
            assert conflict.value.code is ErrorCode.STORAGE_CONFLICT
            assert cancel_read.is_set()
            after = await repository.get(execution.execution_id, tenant_id=tenant)
            after_stored = await repository.state_store.read(lambda tx: tx.get_record(key))
            assert before_stored is not None and after_stored is not None
            assert after_stored.storage_version > before_stored.storage_version
            assert after_stored.data == before_stored.data
            assert before is not None and after is not None
            assert after.status is before.status is ExecutionStatus.STARTED
            assert (after.revision, after.event_seq) == (before.revision, before.event_seq)
            events = await owner_storage.execution.events.list(
                execution.execution_id, tenant_id=tenant, after_event_seq=0, limit=100,
            )
            assert not any(item.event_type == "CANCEL_REQUESTED" for item in events.items)

            outcome = await remote.executions.cancel(execution.execution_id, request)
            assert not outcome.cancelled
            release.set()
            result = (await execution.wait(timeout_seconds=10)).result
            assert result.status is ExecutionStatus.CANCELLED
            events = await owner_storage.execution.events.list(
                execution.execution_id, tenant_id=tenant, after_event_seq=0, limit=100,
            )
            assert sum(item.event_type == "CANCEL_REQUESTED" for item in events.items) == 1
        finally:
            release.set()
            if projection is not None:
                projection.cancel()
                await asyncio.gather(projection, return_exceptions=True)
