#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from linktools.ai.persistence import Persistence
"""Session terminal handoff across tool-using turns."""
from ._runtime_test_helpers import model_features

from linktools.ai.persistence import Persistence
import asyncio
import shutil
import sqlite3
from collections.abc import Awaitable, Callable, Mapping
from contextlib import closing
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pytest
from linktools.ai.core import (
    ExecutionEventType,
    ExecutionStatus,
    JsonValue,
    OperationKind,
    OperationLedgerRecord,
    OperationStatus,
    Principal,
    ResourceKind,
    TenantAuthorizationPolicy,
    ToolOperationStatus,
)
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.migrate import provision_runtime_database
from linktools.ai.capability import SessionCapability, TraceCapability
from pydantic_ai_harness.step_persistence import StepPersistence
from tests.ai._runtime_test_helpers import native_checkpoint, seed_native_snapshot
from linktools.ai.runtime import Runtime, RuntimeFeatures
from linktools.ai.execution import ToolEffectApplied, ToolEffectFailed, ToolEffectNotApplied
from linktools.ai.persistence import PersistenceStorage
from linktools.ai.runtime._local import LocalExecutionBackend
from linktools.ai.runtime._tool import RuntimeToolOperationBridge
from linktools.ai.execution import CancelExecutionRequest
from linktools.ai.persistence import PersistenceDomain
from linktools.ai.persistence import _execution_repositories
from linktools.ai.persistence._runtime_commands import StateCommands
from linktools.ai.persistence._step_archive import StateStepArchive
from linktools.ai.persistence._step_contracts import (
    AgentRunCheckpoint,
    AgentRunRecord,
)
from linktools.ai.persistence.runs import RuntimeAgentRunStore
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import RunContext, ToolDefinition
from pydantic_ai.usage import RunUsage
from sqlalchemy.engine import URL
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from ._runtime_test_helpers import _Models, _UsageFunctionModel, _wait_for_committed
from ._session_tool_test_helpers import (
    _ToolModelBinding,
    _ToolModels,
    _application,
    _exit_at_boundary,
    _relevant_kinds,
    _split_sqlite_storage,
)


def _recovery_tool_model(
    messages: list[ModelMessage],
) -> ModelResponse:
    calls = tuple(
        part
        for message in messages
        if isinstance(message, ModelResponse)
        for part in message.parts
        if isinstance(part, ToolCallPart) and part.tool_name == "lookup"
    )
    returned = {
        part.tool_call_id
        for message in messages
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, ToolReturnPart) and part.outcome != "interrupted"
    }
    pending = next(
        (part for part in reversed(calls) if part.tool_call_id not in returned),
        None,
    )
    if not calls or pending is not None:
        return ModelResponse(
            parts=[
                ToolCallPart(
                    "lookup",
                    {},
                    tool_call_id=(
                        "recovery-lookup" if pending is None else pending.tool_call_id
                    ),
                )
            ]
        )
    return ModelResponse(parts=[TextPart("done")])


class _RecoveryTestModel(TestModel):
    def _request(self, messages: list[ModelMessage], *_args: Any) -> ModelResponse:
        return _recovery_tool_model(messages)


class _RecoveryToolModelBinding(_ToolModelBinding):
    def materialize(self) -> TestModel:
        return _RecoveryTestModel()


class _RecoveryToolModels(_RecoveryToolModelBinding):
    pass


async def _state(
    backend: str,
    tmp_path: Path,
) -> tuple[PersistenceStorage, AsyncEngine | None]:
    if backend == "memory":
        return PersistenceStorage.in_memory(), None
    if backend == "sqlite":
        return PersistenceStorage.sqlite(tmp_path / "runtime-sqlite.db"), None
    if backend == "sql":
        engine = create_async_engine(
            URL.create(
                "sqlite+aiosqlite",
                database=str(tmp_path / "runtime-sql.db"),
            )
        )
        await provision_runtime_database(engine)
        return PersistenceStorage.sql(engine), engine
    raise AssertionError(backend)


class _NativeSessionPolicy(TenantAuthorizationPolicy):
    tool_returns = False

    async def session_tool_return_names(self, principal, session_id, turn) -> frozenset[str]:
        return frozenset({"lookup"}) if self.tool_returns else frozenset()


async def _native_session_tool_turn_commits_terminal_and_history(
    state: PersistenceStorage, calls: list[str],
) -> None:
    features = _application(calls)
    features.capability(SessionCapability(checkpoints=True), scope="runtime")
    features.capability(TraceCapability(), scope="runtime")
    policy = _NativeSessionPolicy()
    async with (
        Persistence.open("session-tool-terminal", storage=state) as persistence,
        Runtime.open(persistence=persistence, features=model_features(_ToolModels(), features),
                     authorization=policy) as runtime,
    ):
        principal = Principal("owner", runtime.tenant_id)
        await runtime.sessions.create_turn_session("session", principal=principal)
        turns = runtime.sessions
        owner = await turns.start("session", "inspect", agent="default", principal=principal, idempotency_key="turn-1")
        result = await owner.wait_final(timeout_seconds=10)
        assert result.status is ExecutionStatus.SUCCEEDED and calls == ["lookup"]
        assert result.view["conversation_committed"]
        call, answer = result.view["items"]
        assert (call["item_kind"], answer["item_kind"]) == ("tool_call", "assistant")
        process, = result.view["process"]
        assert process["key"] == f"tool:{call['tool_call_id']}"
        assert process["status"] == "SUCCEEDED" and process["elapsed_ms"] >= 0

        policy.tool_returns = True
        returned, = await runtime.history.turn_tool_returns(
            "session", owner.key.turn_id, principal=principal, tool_names=frozenset({"lookup"}),
        )
        assert returned == {"tool_name": "lookup", "tool_call_id": call["tool_call_id"], "value": "tool-result"}

        async def trace():
            return await runtime.history.turn_trace("session", owner.key.turn_id, principal=principal)

        observed = await _wait_for_committed(trace, lambda value: bool(value.items), timeout=10)
        projected = [item.payload for item in observed.items
                     if item.payload.get("kind") in {"TOOL_CALL", "TOOL_RESULT", "TOOL_ERROR"}]
        assert [item["kind"] for item in projected] == ["TOOL_CALL", "TOOL_RESULT"]
        assert projected[0]["model_request_seq"] == projected[1]["model_request_seq"] == 1
        assert all(not {"purpose", "content", "args", "arguments", "result", "output", "text", "error_diagnostics"}
                   .intersection(item) for item in projected)

        replay = await turns.start("session", "inspect", agent="default", principal=principal, idempotency_key="turn-1")
        assert replay.key == owner.key and (await replay.wait_final(timeout_seconds=10)).status is ExecutionStatus.SUCCEEDED
        second = await turns.start("session", "continue", agent="default", principal=principal, idempotency_key="turn-2")
        assert (await second.wait_final(timeout_seconds=10)).status is ExecutionStatus.SUCCEEDED
        page = await runtime.history.turn_session("session", principal=principal)
        assert [item["turn_id"] for item in page["timeline"]["items"]] == [owner.key.turn_id, second.key.turn_id]
        assert calls == ["lookup"]


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "sqlite", "sql"))
async def test_session_tool_turn_commits_terminal_and_history(
    tmp_path: Path,
    backend: str,
) -> None:
    state, engine = await _state(backend, tmp_path)
    calls: list[str] = []
    if backend != "memory":
        try:
            await _native_session_tool_turn_commits_terminal_and_history(state, calls)
        finally:
            if engine is not None:
                await engine.dispose()
        return
    application = _application(calls)
    try:
        application.capability(native_checkpoint(), scope="runtime")
        application.capability(TraceCapability(), scope="runtime")
        async with Persistence.open('session-tool-terminal', storage=state) as persistence_151, Runtime.open(persistence=persistence_151, features=model_features(_ToolModels(), application)) as runtime:
            await runtime.agents.get("default").create_session("session")
            session = runtime.agents.get("default").session("session")
            execution = await session.start(
                "inspect",
                idempotency_key="turn-1",
            )

            watched = [
                item
                async for item in execution.watch(include_content=True)
                if item.depth == 0
            ]
            result = (await execution.wait(timeout_seconds=10)).result

            assert result.status is ExecutionStatus.SUCCEEDED
            assert calls == ["lookup"]
            event_types = [item.event.event_type for item in watched]
            assert ExecutionEventType.TOOL_CALL_STARTED in event_types
            assert ExecutionEventType.TOOL_CALL_FINISHED in event_types
            assert ExecutionEventType.EXECUTION_SUCCEEDED in event_types

            session_history = await session.history()
            assert _relevant_kinds(session_history.items) == [
                "user",
                "tool_call",
                "tool_result",
                "assistant",
            ]

            assert runtime.history is not None
            execution_history = await runtime.history.history(
                execution.execution_id,
                principal=runtime.default_principal,
                include_content=True,
            )
            assert _relevant_kinds(execution_history.items) == [
                "user",
                "tool_call",
                "tool_result",
                "assistant",
            ]
            tool_items = tuple(
                item
                for item in execution_history.items
                if item.item_kind in {"tool_call", "tool_result"}
            )
            assert len(tool_items) == 2
            assert tool_items[0].tool_call_id == tool_items[1].tool_call_id
            assert tool_items[0].model_request_seq is not None
            assert tool_items[1].model_request_seq == tool_items[0].model_request_seq
            assert tool_items[0].tool_operation_id is not None
            assert tool_items[1].tool_operation_id == tool_items[0].tool_operation_id
            assert tool_items[0].started_at is not None
            assert tool_items[0].finished_at is not None
            assert tool_items[0].duration_ns is not None
            assert tool_items[0].status == "SUCCEEDED"

            execution_trace = await runtime.history.trace(
                execution.execution_id,
                principal=runtime.default_principal,
            )
            tool_trace = tuple(
                item
                for item in execution_trace.items
                if item.payload.get("kind") in {"TOOL_CALL", "TOOL_RESULT", "TOOL_ERROR"}
            )
            assert [item.payload["kind"] for item in tool_trace] == [
                "TOOL_CALL",
                "TOOL_RESULT",
            ]
            assert (
                tool_trace[0].payload["model_request_seq"]
                == tool_items[0].model_request_seq
            )
            assert (
                tool_trace[1].payload["model_request_seq"]
                == tool_items[0].model_request_seq
            )
            assert all("purpose" not in item.payload for item in tool_trace)
            assert all(
                not {"content", "args", "arguments", "result", "output", "text", "error_diagnostics"}
                .intersection(item.payload)
                for item in execution_trace.items
            )

            retried = await session.start(
                "inspect",
                idempotency_key="turn-1",
            )
            retried_result = (await retried.wait(timeout_seconds=10)).result
            assert retried.execution_id == execution.execution_id
            assert retried_result.status is ExecutionStatus.SUCCEEDED
            assert calls == ["lookup"]

            second = (await session.run(
                "continue",
                idempotency_key="turn-2",
                timeout_seconds=10,
            )).result
            assert second.status is ExecutionStatus.SUCCEEDED
            assert calls == ["lookup"]
            accumulated = await session.history()
            assert _relevant_kinds(accumulated.items) == [
                "user",
                "tool_call",
                "tool_result",
                "assistant",
                "user",
                "assistant",
            ]
    finally:
        await runtime._state.close()
        if engine is not None:
            await engine.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize("storage_mode", ("same_group", "split_conversation"))
@pytest.mark.parametrize(
    ("rejection", "expected_code"),
    (
        ("busy", ErrorCode.SESSION_BUSY),
        ("conflict", ErrorCode.SESSION_CONFLICT),
    ),
)
async def test_rejected_session_start_cleans_only_its_admission(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    storage_mode: str,
    rejection: str,
    expected_code: ErrorCode,
) -> None:
    state = (PersistenceStorage.sqlite(tmp_path / "same-group.db") if storage_mode == "same_group"
             else _split_sqlite_storage(tmp_path / "split.db"))
    entered, release = asyncio.Event(), asyncio.Event()
    calls = []

    async def model(messages, _info):
        calls.append(messages)
        entered.set()
        await release.wait()
        return ModelResponse(parts=[TextPart("answer")])


    features = RuntimeFeatures("session-admission-cleanup")
    features.agent("default", model="default", allow_tools=())
    async with (
        Persistence.open("session-admission-cleanup", storage=state) as persistence,
        Runtime.open(persistence=persistence, features=model_features(_Models(_UsageFunctionModel(model)), features),
                     authorization=_NativeSessionPolicy()) as runtime,
    ):
        principal = runtime.default_principal
        sessions, store = runtime.sessions, persistence.session_turns
        before = await sessions.create_turn_session("session", principal=principal)
        owner = None
        try:
            with monkeypatch.context() as fault:
                if rejection == "busy":
                    owner = await sessions.start_turn("session", "owner", agent="default", principal=principal,
                                                 idempotency_key="turn-owner")
                    await asyncio.wait_for(entered.wait(), timeout=10)
                    before = await runtime.history.turn_session_state("session", principal=principal)
                    assert before["active_turn_id"] == owner.key.turn_id
                else:
                    accept = store.accept

                    async def change_revision(key, **kwargs):
                        assert await sessions.update_turn_session(
                            "session", principal=principal, expected_revision=before["revision"],
                            metadata={"source": "concurrent owner"},
                        )
                        return await accept(key, **kwargs)

                    fault.setattr(store, "accept", change_revision)
                with pytest.raises(AIError) as raised:
                    await sessions.start_turn("session", "rejected", agent="default", principal=principal,
                                         idempotency_key="turn-rejected")
            assert raised.value.code is expected_code
            current = await runtime.history.turn_session_state("session", principal=principal)
            assert current["last_turn_no"] == 0
            assert not await store.find_request("session", "turn-rejected")
            if owner is None:
                assert current["active_turn_id"] is None
                assert current["metadata"] == {"source": "concurrent owner"}
                assert current["revision"] == before["revision"] + 1
                assert not calls
            else:
                assert current == before
                assert len(calls) == 1
        finally:
            release.set()
        if owner is not None:
            assert (await owner.wait_final(timeout_seconds=15)).status is ExecutionStatus.SUCCEEDED
            assert len(calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("committed", (False, True))
async def test_split_admission_unknown_preserves_only_committed_turn_after_restart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, committed: bool,
) -> None:

    database = tmp_path / "start-cleanup-restart.db"
    calls = []
    attempted = []

    async def model(messages, _info):
        calls.append(messages)
        return ModelResponse(parts=[TextPart("answer")])

    features = RuntimeFeatures("start-cleanup-restart")
    features.agent("default", model="default", allow_tools=())
    features = model_features(_Models(_UsageFunctionModel(model)), features)
    async with (
        Persistence.open("start-cleanup-restart", storage=_split_sqlite_storage(database)) as persistence,
        Runtime.open(persistence=persistence, features=features, authorization=_NativeSessionPolicy()) as runtime,
    ):
        principal = runtime.default_principal
        sessions, store = runtime.sessions, persistence.session_turns
        await sessions.create_turn_session("session", principal=principal)
        before = await runtime.history.turn_session_state("session", principal=principal)
        accept = store.accept

        async def lose_ack(key, **kwargs):
            attempted.append(key)
            if committed:
                assert await accept(key, **kwargs)
            raise AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN)

        with monkeypatch.context() as fault:
            fault.setattr(store, "accept", lose_ack)
            if committed:
                replay = await sessions.start_turn("session", "inspect", agent="default", principal=principal,
                                              idempotency_key="turn-cleanup-restart")
                assert replay.key == attempted[0]
            else:
                with pytest.raises(AIError) as rejected:
                    await sessions.start_turn("session", "inspect", agent="default", principal=principal,
                                         idempotency_key="turn-cleanup-restart")
                assert rejected.value.code is ErrorCode.STORAGE_COMMIT_UNKNOWN

        key, = attempted
        assert not calls
        current = await runtime.history.turn_session_state("session", principal=principal)
        if not committed:
            assert current == before
            assert await store.get_turn(key) is None
        else:
            assert current["active_turn_id"] == key.turn_id and current["last_turn_no"] == 0
            row = await store.get_turn(key)
            assert row["state"] == "accepted" and row["generation"] == 1

    async with (
        Persistence.open("start-cleanup-restart", storage=_split_sqlite_storage(database)) as persistence,
        Runtime.open(persistence=persistence, features=features, authorization=_NativeSessionPolicy()) as runtime,
    ):
        store = persistence.session_turns
        current = await runtime.history.turn_session_state("session", principal=principal)
        assert not calls
        if not committed:
            assert current == before and await store.get_turn(key) is None
        else:
            assert current["active_turn_id"] == key.turn_id
            table = store._turns
            async with store._context.engine.begin() as connection:
                expired = await connection.execute(table.update().filter_by(
                    **store._turn_values(key), generation=1,
                ).values(lease_until=0))
                assert expired.rowcount == 1
            assert await runtime.sessions.recover_turn("session", key.turn_id, mode="resume", principal=principal)

            async def released():
                return await runtime.history.turn_result("session", key.turn_id, principal=principal)

            view = await _wait_for_committed(released, lambda item: item["conversation_committed"], timeout=10)
            assert view["status"] == "SUCCEEDED" and len(calls) == 1
            current = await runtime.history.turn_session_state("session", principal=principal)
            assert current["active_turn_id"] is None and current["last_turn_no"] == 1


@pytest.mark.asyncio
async def test_session_start_cancel_before_worker_run_commits_cancelled_terminal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from linktools.ai.runtime._regenerable_turn import _RegenerableTurnRunner

    storage = PersistenceStorage.sqlite(tmp_path / "cancel-start-before-run.db")
    calls: list[str] = []
    worker_entered = asyncio.Event()
    features = _application(calls)
    features.capability(SessionCapability(checkpoints=True), scope="runtime")

    async def pause_worker(*_args, **_kwargs):
        worker_entered.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(_RegenerableTurnRunner, "settle", pause_worker)
    async with (
        Persistence.open("session-start-cancel-before-run", storage=storage) as state,
        Runtime.open(persistence=state, features=model_features(_ToolModels(), features),
                     authorization=_NativeSessionPolicy()) as runtime,
    ):
        principal = runtime.default_principal
        sessions = runtime.sessions
        await sessions.create_turn_session("session", principal=principal)
        before = await runtime.history.turn_session_state("session", principal=principal)
        turn = await sessions.start_turn("session", "inspect", agent="default", principal=principal, idempotency_key="turn-1")
        await asyncio.wait_for(worker_entered.wait(), timeout=10)
        admitted = await runtime.history.turn_session_state("session", principal=principal)
        assert admitted["active_turn_id"] == turn.key.turn_id
        assert await turn.cancel()
        result = await turn.wait_final(timeout_seconds=10)
        assert result.status is ExecutionStatus.CANCELLED
        assert result.view["conversation_committed"]
        assert result.view["error_code"] == ErrorCode.EXECUTION_CANCELLED.value
        assert result.view["output"] is None and result.view["items"] == []
        after = await runtime.history.turn_session_state("session", principal=principal)
        assert after["status"] == before["status"] and after["active_turn_id"] is None
        assert after["last_turn_no"] == before["last_turn_no"] + 1
        assert await turn.cancel()
        replay = await sessions.start_turn("session", "inspect", agent="default", principal=principal, idempotency_key="turn-1")
        assert replay.key == turn.key
        assert (await replay.result()).view == result.view and not calls


@pytest.mark.asyncio
@pytest.mark.parametrize("storage_mode", ("same_group", "split_conversation"))
async def test_native_pending_start_cannot_publish_after_session_closes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    storage_mode: str,
) -> None:
    storage = (PersistenceStorage.sqlite(tmp_path / "pending-start-cancel.db") if storage_mode == "same_group"
               else _split_sqlite_storage(tmp_path / "pending-start-cancel.db"))
    calls = []
    prepared, release = asyncio.Event(), asyncio.Event()
    keys = []

    async def model(messages, _info):
        calls.append(messages)
        return ModelResponse(parts=[TextPart("not expected")])

    features = RuntimeFeatures("pending-start-cancel")
    features.agent("default", model="default", allow_tools=())
    async with (
        Persistence.open("pending-start-cancel", storage=storage) as state,
        Runtime.open(persistence=state, features=model_features(_Models(_UsageFunctionModel(model)), features),
                     authorization=_NativeSessionPolicy()) as runtime,
    ):
        principal = runtime.default_principal
        sessions, store = runtime.sessions, state.session_turns
        await sessions.create_turn_session("session", principal=principal)
        initial = await runtime.history.turn_session_state("session", principal=principal)
        accept = store.accept

        async def pause_accept(key, **kwargs):
            keys.append(key)
            prepared.set()
            await release.wait()
            return await accept(key, **kwargs)

        monkeypatch.setattr(store, "accept", pause_accept)
        request = asyncio.create_task(sessions.start_turn(
            "session", "inspect", agent="default", principal=principal, idempotency_key="turn-pending-start",
        ))
        try:
            await asyncio.wait_for(prepared.wait(), timeout=10)
            key, = keys
            # There is no cancellable turn identifier until native admission commits.
            assert not await sessions.cancel_turn("session", key.turn_id, principal=principal)
            assert await store.get_turn(key) is None
            assert await runtime.history.turn_session_state("session", principal=principal) == initial
            assert await sessions.close_turn_session("session", principal=principal, expected_revision=initial["revision"])
        finally:
            release.set()
            received, = await asyncio.gather(request, return_exceptions=True)
        assert isinstance(received, AIError) and received.code is ErrorCode.SESSION_CONFLICT
        assert await store.get_turn(key) is None
        assert not await store.find_request("session", "turn-pending-start") and not calls
        current = await runtime.history.turn_session_state("session", principal=principal)
        assert current["status"] == "CLOSED" and current["active_turn_id"] is None
        assert current["last_turn_no"] == 0


@pytest.mark.asyncio
async def test_cancel_during_unknown_tool_effect_preserves_recovery_and_cancel_intent(
    tmp_path: Path,
) -> None:
    state = PersistenceStorage.sqlite(tmp_path / "cancel-recovery.db")
    calls: list[str] = []
    tool_started = asyncio.Event()
    release_tool = asyncio.Event()
    application = _application(
        calls,
        effect_policy="non_replay_safe",
        effect_log=tmp_path / "effects.txt",
        started=tool_started,
        release=release_tool,
    )
    try:
        application.capability(native_checkpoint(), scope="runtime")
        application.capability(TraceCapability(), scope="runtime")
        async with Persistence.open('cancel-unknown-tool-effect', storage=state) as persistence_801, Runtime.open(persistence=persistence_801, features=model_features(_ToolModels(), application)) as runtime:
            execution = await runtime.agents.get("default").start(
                "inspect",
                idempotency_key="turn-1",
            )
            await asyncio.wait_for(tool_started.wait(), timeout=10)

            first_cancel = await execution.cancel(idempotency_key="cancel-1")
            assert first_cancel.cancelled is False

            current = await runtime._state.execution.executions.get(
                execution.execution_id,
                tenant_id=runtime.default_principal.tenant_id,
            )
            assert current is not None
            assert current.status is ExecutionStatus.RECOVERY_REQUIRED
            assert current.error_code == ErrorCode.TOOL_EFFECT_UNKNOWN.value
            assert await runtime._state.execution.executions.get_result(
                execution.execution_id,
                tenant_id=runtime.default_principal.tenant_id,
            ) is None

            operations = await runtime._state.checkpoint.tools.list_by_execution(
                execution.execution_id,
                tenant_id=runtime.default_principal.tenant_id,
            )
            assert len(operations) == 1
            assert operations[0].status.value == "EFFECT_UNKNOWN"

            cancel_operations = await runtime._state.execution.operations.list_pending(
                ResourceKind.EXECUTION,
                execution.execution_id,
                tenant_id=runtime.default_principal.tenant_id,
                limit=100,
            )
            assert len(cancel_operations) == 1
            assert cancel_operations[0].operation_kind is OperationKind.EXECUTION_CANCEL
            assert cancel_operations[0].status is OperationStatus.PENDING

            assert current.session_id is None

            replayed_cancel = await execution.cancel(idempotency_key="cancel-1")
            assert replayed_cancel.cancelled is False
            replayed_operations = await runtime._state.execution.operations.list_pending(
                ResourceKind.EXECUTION,
                execution.execution_id,
                tenant_id=runtime.default_principal.tenant_id,
                limit=100,
            )
            assert replayed_operations == cancel_operations

            with pytest.raises(AIError) as raised:
                await execution.wait(timeout_seconds=5)
            assert raised.value.code is ErrorCode.TOOL_EFFECT_UNKNOWN
            with pytest.raises(AIError) as recover_error:
                await execution.recover()
            assert recover_error.value.code is ErrorCode.TOOL_EFFECT_UNKNOWN

            events = await execution.list_events(limit=100)
            event_types = tuple(item.event_type for item in events.items)
            assert ExecutionEventType.EXECUTION_RECOVERY_REQUIRED in event_types
            assert ExecutionEventType.EXECUTION_CANCELLED not in event_types
            assert calls == ["lookup"]
            assert (tmp_path / "effects.txt").read_text().splitlines() == [
                execution.execution_id
            ]
            release_tool.set()
    finally:
        release_tool.set()
        await runtime._state.close()


@pytest.mark.asyncio
async def test_cancel_intent_replay_readback_preserves_sqlite_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = PersistenceStorage.sqlite(tmp_path / "cancel-intent-readback.db")
    calls: list[str] = []
    tool_started = asyncio.Event()
    release_tool = asyncio.Event()
    application = _application(
        calls,
        effect_policy="non_replay_safe",
        started=tool_started,
        release=release_tool,
    )
    cancel_key = "cancel-probe"
    try:
        application.capability(native_checkpoint(), scope="runtime")
        application.capability(TraceCapability(), scope="runtime")
        async with Persistence.open('cancel-intent-readback', storage=state) as persistence_903, Runtime.open(persistence=persistence_903, features=model_features(_ToolModels(), application)) as runtime:
            execution = await runtime.agents.get("default").start(
                "inspect",
                idempotency_key="turn-probe",
            )
            await asyncio.wait_for(tool_started.wait(), timeout=10)

            first = await execution.cancel(idempotency_key=cancel_key)
            assert first.cancelled is False
            principal = runtime.default_principal
            before_conflict = await runtime._state.execution.executions.get(
                execution.execution_id,
                tenant_id=principal.tenant_id,
            )
            assert before_conflict is not None
            assert before_conflict.status is ExecutionStatus.RECOVERY_REQUIRED

            async def read_events() -> tuple[Any, ...]:
                page = await runtime._state.execution.events.list(
                    execution.execution_id,
                    tenant_id=principal.tenant_id,
                    after_event_seq=0,
                    limit=100,
                )
                return page.items

            intent = await runtime._state.execution.operations.get(
                cancel_key,
                tenant_id=principal.tenant_id,
            )
            assert intent is not None
            events_before_conflict = await read_events()

            with pytest.raises(AIError) as conflict:
                await execution.cancel(idempotency_key=cancel_key, force=True)
            assert conflict.value.code is ErrorCode.IDEMPOTENCY_CONFLICT
            assert await runtime._state.execution.operations.get(
                cancel_key,
                tenant_id=principal.tenant_id,
            ) == intent
            assert await runtime._state.execution.executions.get(
                execution.execution_id,
                tenant_id=principal.tenant_id,
            ) == before_conflict
            assert await read_events() == events_before_conflict

            target_group = runtime._state.execution.executions.state_store.storage_group
            group_class = type(target_group)
            original_mutate = group_class.mutate
            original_get = runtime._state.execution.operations.get

            async def cancel_with_unknown_commit(
                key: str,
                *,
                unreadable: bool = False,
                partial_projection: bool = False,
            ) -> tuple[bool | None, ErrorCode | None]:
                injected = False

                async def uncertain_commit(group: Any, stores: Any, callback: Any) -> Any:
                    nonlocal injected
                    value = await original_mutate(group, stores, callback)
                    if (
                        group is target_group
                        and not injected
                        and isinstance(value, OperationLedgerRecord)
                        and value.operation_kind is OperationKind.EXECUTION_CANCEL
                    ):
                        injected = True
                        raise AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN)
                    return value

                async def unavailable_readback(
                    operation_id: str,
                    *,
                    tenant_id: str,
                ) -> Any:
                    if injected:
                        raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)
                    return await original_get(operation_id, tenant_id=tenant_id)

                async def leave_projection_unchanged(
                    transaction: Any,
                    candidate: Any,
                    expected_version: int,
                ) -> None:
                    guarded = await transaction.guard_record(
                        candidate.key,
                        expected_storage_revision=expected_version,
                    )
                    if guarded is None:
                        raise AIError(ErrorCode.STORAGE_CONFLICT)

                error_code = None
                cancelled = None
                with monkeypatch.context() as patcher:
                    patcher.setattr(group_class, "mutate", uncertain_commit)
                    if unreadable:
                        patcher.setattr(
                            runtime._state.execution.operations,
                            "get",
                            unavailable_readback,
                        )
                    if partial_projection:
                        patcher.setattr(
                            _execution_repositories,
                            "_replace_checked",
                            leave_projection_unchanged,
                        )
                    try:
                        result = await execution.cancel(idempotency_key=key)
                        cancelled = result.cancelled
                    except AIError as error:
                        error_code = error.code
                assert injected
                return cancelled, error_code

            before_replay = await runtime._state.execution.executions.get(
                execution.execution_id,
                tenant_id=principal.tenant_id,
            )
            events_before_replay = await read_events()
            replayed, replay_error = await cancel_with_unknown_commit(cancel_key)
            assert replay_error is None
            assert replayed is False
            assert await runtime._state.execution.operations.get(
                cancel_key,
                tenant_id=principal.tenant_id,
            ) == intent
            assert await runtime._state.execution.executions.get(
                execution.execution_id,
                tenant_id=principal.tenant_id,
            ) == before_replay
            assert await read_events() == events_before_replay

            new_key = "cancel-new-probe"
            before_new_intent = await runtime._state.execution.executions.get(
                execution.execution_id,
                tenant_id=principal.tenant_id,
            )
            assert before_new_intent is not None
            events_before_new_intent = await read_events()
            new_cancelled, new_error = await cancel_with_unknown_commit(new_key)
            assert new_error is None
            assert new_cancelled is False
            after_new_intent = await runtime._state.execution.executions.get(
                execution.execution_id,
                tenant_id=principal.tenant_id,
            )
            assert after_new_intent is not None
            assert after_new_intent.status is ExecutionStatus.RECOVERY_REQUIRED
            assert after_new_intent.revision == before_new_intent.revision + 1
            assert after_new_intent.event_seq == before_new_intent.event_seq + 1
            new_intent = await runtime._state.execution.operations.get(
                new_key,
                tenant_id=principal.tenant_id,
            )
            assert new_intent is not None
            assert new_intent.operation_id != intent.operation_id
            pending_intents = await runtime._state.execution.operations.list_pending(
                ResourceKind.EXECUTION,
                execution.execution_id,
                tenant_id=principal.tenant_id,
                limit=100,
            )
            assert {item.operation_id for item in pending_intents} == {
                intent.operation_id,
                new_intent.operation_id,
            }
            events_after_new_intent = await read_events()
            assert events_after_new_intent[:-1] == events_before_new_intent
            assert events_after_new_intent[-1].event_type == ExecutionEventType.CANCEL_REQUESTED
            assert events_after_new_intent[-1].payload == {
                "operation_id": new_intent.operation_id,
            }

            unknown_key = "cancel-unreadable-probe"
            events_before_unreadable = events_after_new_intent
            _, unreadable_error = await cancel_with_unknown_commit(
                unknown_key,
                unreadable=True,
            )
            assert unreadable_error is ErrorCode.STORAGE_COMMIT_UNKNOWN
            after_unreadable = await runtime._state.execution.executions.get(
                execution.execution_id,
                tenant_id=principal.tenant_id,
            )
            assert after_unreadable is not None
            assert after_unreadable.status is ExecutionStatus.RECOVERY_REQUIRED
            assert after_unreadable.revision == after_new_intent.revision + 1
            assert after_unreadable.event_seq == after_new_intent.event_seq + 1
            unreadable_intent = await original_get(
                unknown_key,
                tenant_id=principal.tenant_id,
            )
            assert unreadable_intent is not None
            pending_intents = await runtime._state.execution.operations.list_pending(
                ResourceKind.EXECUTION,
                execution.execution_id,
                tenant_id=principal.tenant_id,
                limit=100,
            )
            assert {item.operation_id for item in pending_intents} == {
                intent.operation_id,
                new_intent.operation_id,
                unreadable_intent.operation_id,
            }
            assert (await read_events())[:-1] == events_before_unreadable
            assert await runtime._state.execution.executions.get_result(
                execution.execution_id,
                tenant_id=principal.tenant_id,
            ) is None
            assert after_unreadable.session_id is None
            assert calls == ["lookup"]

            before_partial = after_unreadable
            events_before_partial = await read_events()
            partial_key = "cancel-partial-probe"
            _, partial_error = await cancel_with_unknown_commit(
                partial_key,
                partial_projection=True,
            )
            assert partial_error is ErrorCode.STORAGE_INTEGRITY_ERROR
            assert await runtime._state.execution.executions.get(
                execution.execution_id,
                tenant_id=principal.tenant_id,
            ) == before_partial
            partial_intent = await original_get(
                partial_key,
                tenant_id=principal.tenant_id,
            )
            assert partial_intent is not None
            partial_events = await read_events()
            assert partial_events[:-1] == events_before_partial
            assert partial_events[-1].event_type == ExecutionEventType.CANCEL_REQUESTED
            assert partial_events[-1].payload == {
                "operation_id": partial_intent.operation_id,
            }
    finally:
        release_tool.set()
        await runtime._state.close()


@dataclass(frozen=True)
class _UnresolvedToolSource:
    directory: Path
    execution_id: str
    agent_run_id: str
    operation_id: str
    fence: int
    producer_generation: int
    calls: tuple[str, ...]
    durable_interaction_count: int | None
    cancel_status: OperationStatus | None


async def _prepare_unresolved_tool_source(
    tmp_path: Path, with_cancel: bool,
) -> _UnresolvedToolSource:
    database = tmp_path / "recovery-resolution.db"
    state = _split_sqlite_storage(database)
    calls: list[str] = []
    tool_started = asyncio.Event()
    release_tool = asyncio.Event()
    effect_log = tmp_path / "recovery-effects.txt"
    application = _application(
        calls,
        effect_policy="non_replay_safe",
        effect_log=effect_log,
        started=tool_started,
        release=release_tool,
    )
    durable_interaction_count = None
    cancel_status = None
    try:
        application.capability(native_checkpoint(database=tmp_path / "native.db"), scope="runtime")
        application.capability(TraceCapability(), scope="runtime")
        async with Persistence.open('session-tool-recovery-resolution', storage=state) as persistence_1204, Runtime.open(persistence=persistence_1204, features=model_features(_RecoveryToolModels(), application)) as runtime:
            execution = await runtime.agents.get("default").start(
                "inspect",
                idempotency_key="turn-resolution",
            )
            await asyncio.wait_for(tool_started.wait(), timeout=10)
            execution_id = execution.execution_id

            if with_cancel:
                cancelled = await execution.cancel(
                    idempotency_key="cancel-after-effect-recovery"
                )
                assert cancelled.cancelled is False
                replayed = await execution.cancel(
                    idempotency_key="cancel-after-effect-recovery"
                )
                assert replayed.cancelled is False
            else:
                backend = runtime._execution_service.runtime_backend()
                worker = backend._tasks[execution_id]
                worker.cancel()
                await asyncio.gather(worker, return_exceptions=True)
            with pytest.raises(AIError) as unknown_result:
                await execution.wait(timeout_seconds=10)
            assert unknown_result.value.code is ErrorCode.TOOL_EFFECT_UNKNOWN

            current = await runtime._state.execution.executions.get(
                execution_id,
                tenant_id=runtime.default_principal.tenant_id,
            )
            assert current is not None
            assert current.status is ExecutionStatus.RECOVERY_REQUIRED
            with pytest.raises(AIError) as unresolved:
                await execution.recover()
            assert unresolved.value.code is ErrorCode.TOOL_EFFECT_UNKNOWN

            effects = await execution.recovery_effects()
            assert len(effects) == 1
            cancel_key = "cancel-after-effect-recovery"
            checkpoint_before = await runtime._state.execution.executions.get(
                execution_id,
                tenant_id=runtime.default_principal.tenant_id,
            )
            assert checkpoint_before is not None
            assert checkpoint_before.recovery is not None and checkpoint_before.recovery.agent_run_id is not None
            agent_run_id = checkpoint_before.recovery.agent_run_id
            if with_cancel:
                trace_archive = runtime._run_store.read_store(PersistenceDomain.TRACE)
                durable_interaction_count = await trace_archive.model_interaction_count(
                    agent_run_id=agent_run_id
                )
            if with_cancel:
                receipt = await runtime._state.execution.operations.get(
                    cancel_key, tenant_id=runtime.default_principal.tenant_id,
                )
                assert receipt is not None
                cancel_status = receipt.status
            operation_id = effects[0].operation_id
            fence = effects[0].fence

    finally:
        release_tool.set()
        await runtime._state.close()
    return _UnresolvedToolSource(
        tmp_path, execution_id, agent_run_id, operation_id, fence,
        checkpoint_before.producer_generation, tuple(calls),
        durable_interaction_count, cancel_status,
    )


@pytest.fixture(scope="module")
def unresolved_tool_source(
    tmp_path_factory: pytest.TempPathFactory,
) -> Callable[[Path, bool], Awaitable[_UnresolvedToolSource]]:
    sources: dict[bool, _UnresolvedToolSource] = {}

    async def prepare(destination: Path, with_cancel: bool) -> _UnresolvedToolSource:
        if with_cancel not in sources:
            root = tmp_path_factory.mktemp("unresolved-tool-cancel" if with_cancel else "unresolved-tool-resume")
            source = await _prepare_unresolved_tool_source(root, with_cancel)
            stopped = root.rename(root.with_name(root.name + "-stopped"))
            sources[with_cancel] = replace(source, directory=stopped)
        source = sources[with_cancel]
        for name in (
            "recovery-resolution-execution.db",
            "recovery-resolution-conversation.db",
            "native.db",
        ):
            with (
                closing(sqlite3.connect(
                    f"{(source.directory / name).as_uri()}?mode=ro", uri=True,
                )) as original,
                closing(sqlite3.connect(destination / name)) as copied,
            ):
                original.backup(copied)
        shutil.copyfile(
            source.directory / "recovery-effects.txt", destination / "recovery-effects.txt",
        )
        return source

    return prepare


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("resolution", "expected_tool_status", "expected_retries"),
    (
        pytest.param(
            ToolEffectApplied({"confirmed": True}),
            ToolOperationStatus.COMPLETED,
            0,
            id="applied",
        ),
        pytest.param(
            ToolEffectNotApplied(),
            ToolOperationStatus.PENDING,
            1,
            id="not-applied",
        ),
        pytest.param(
            ToolEffectFailed(),
            ToolOperationStatus.FAILED,
            0,
            id="failed",
        ),
    ),
)
@pytest.mark.parametrize("with_cancel", (False, True), ids=("resume", "cancel"))
async def test_public_recovery_converges_after_tool_effect_resolution(
    tmp_path: Path,
    resolution: Any,
    expected_tool_status: ToolOperationStatus,
    expected_retries: int,
    with_cancel: bool,
    unresolved_tool_source: Callable[[Path, bool], Awaitable[_UnresolvedToolSource]],
) -> None:
    source = await unresolved_tool_source(tmp_path, with_cancel)
    execution_id, agent_run_id = source.execution_id, source.agent_run_id
    operation_id, fence = source.operation_id, source.fence
    durable_interaction_count = source.durable_interaction_count
    calls = list(source.calls)
    cancel_key = "cancel-after-effect-recovery"
    database = tmp_path / "recovery-resolution.db"
    effect_log = tmp_path / "recovery-effects.txt"
    state = _split_sqlite_storage(database)
    reopened_application = _application(
        calls,
        effect_policy="non_replay_safe",
        effect_log=effect_log,
    )
    reopened_application.capability(native_checkpoint(database=tmp_path / "native.db"), scope="runtime")
    reopened_application.capability(TraceCapability(), scope="runtime")
    async with Persistence.open('session-tool-recovery-resolution', storage=state) as persistence_1273, Runtime.open(persistence=persistence_1273, features=model_features(_RecoveryToolModels(), reopened_application)) as runtime:
        execution = await runtime.executions.get(execution_id)
        current = await runtime._state.execution.executions.get(
            execution_id, tenant_id=runtime.default_principal.tenant_id,
        )
        assert current is not None and current.status is ExecutionStatus.RECOVERY_REQUIRED
        assert current.producer_generation == source.producer_generation
        effects = await execution.recovery_effects()
        assert len(effects) == 1
        assert (effects[0].operation_id, effects[0].fence) == (operation_id, fence)
        operation = await runtime._state.checkpoint.tools.get_operation(
            operation_id, tenant_id=runtime.default_principal.tenant_id,
        )
        assert operation is not None and operation.status is ToolOperationStatus.EFFECT_UNKNOWN
        assert operation.producer_generation == source.producer_generation
        assert current.session_id is None
        with pytest.raises(AIError) as unresolved:
            await execution.recover()
        assert unresolved.value.code is ErrorCode.TOOL_EFFECT_UNKNOWN
        if with_cancel:
            receipt = await runtime._state.execution.operations.get(
                cancel_key, tenant_id=runtime.default_principal.tenant_id,
            )
            assert receipt is not None and receipt.status is source.cancel_status

        resolution_result = await execution.resolve_tool_effect(
            operation_id,
            expected_fence=fence,
            resolution=resolution,
            idempotency_key=f"resolve-{expected_tool_status.value.lower()}",
        )
        assert resolution_result.status is expected_tool_status

        recovered = await execution.recover()
        assert recovered.execution_id == execution_id
        result = (await execution.wait(timeout_seconds=15)).result
        expected_execution_status = (
            ExecutionStatus.CANCELLED
            if with_cancel
            else ExecutionStatus.SUCCEEDED
        )
        assert result.status is expected_execution_status

        tool = await runtime._state.checkpoint.tools.get_operation(
            operation_id,
            tenant_id=runtime.default_principal.tenant_id,
        )
        assert tool is not None
        expected_final_tool_status = (
            ToolOperationStatus.COMPLETED
            if not with_cancel
            and expected_tool_status is ToolOperationStatus.PENDING
            else expected_tool_status
        )
        assert tool.status is expected_final_tool_status
        retry_count = 0 if with_cancel else expected_retries
        assert calls == ["lookup"] * (1 + retry_count)
        assert effect_log.read_text().splitlines() == [execution_id] * (
            1 + retry_count
        )
        checkpoint = await runtime._state.execution.executions.get(
            execution_id,
            tenant_id=runtime.default_principal.tenant_id,
        )
        assert checkpoint is not None and checkpoint.session_id is None
        assert checkpoint.recovery_state.value == "completed"

        if with_cancel:
            interactions_after = await runtime._run_store.read_store(
                PersistenceDomain.TRACE
            ).model_interaction_count(
                agent_run_id=agent_run_id,
            )
            assert interactions_after == durable_interaction_count
            operation = await runtime._state.execution.operations.get(
                cancel_key,
                tenant_id=runtime.default_principal.tenant_id,
            )
            assert operation is not None
            assert operation.status is OperationStatus.SUCCEEDED
        else:
            assert runtime.history is not None
            history = await runtime.history.history(
                execution_id,
                principal=runtime.default_principal,
                include_content=True,
            )
            assert history.items


@pytest.mark.asyncio
async def test_no_worker_cancel_intent_survives_replay_and_recovery(
    tmp_path: Path,
) -> None:
    state = _split_sqlite_storage(tmp_path / "recovery-no-worker.db")
    calls: list[str] = []
    tool_started = asyncio.Event()
    release_tool = asyncio.Event()
    effect_log = tmp_path / "recovery-no-worker-effects.txt"
    application = _application(
        calls,
        effect_policy="non_replay_safe",
        effect_log=effect_log,
        started=tool_started,
        release=release_tool,
    )
    cancel_key = "cancel-while-worker-absent"
    try:
        application.capability(native_checkpoint(), scope="runtime")
        application.capability(TraceCapability(), scope="runtime")
        async with Persistence.open('recovery-no-worker-cancel', storage=state) as persistence_1374, Runtime.open(persistence=persistence_1374, features=model_features(_ToolModels(), application)) as runtime:
            execution = await runtime.agents.get("default").start(
                "inspect",
                idempotency_key="turn-1",
            )
            await asyncio.wait_for(tool_started.wait(), timeout=10)
            backend = runtime._execution_service.runtime_backend()
            worker = backend._tasks.pop(execution.execution_id)
            try:
                first = await execution.cancel(idempotency_key=cancel_key)
            finally:
                backend._tasks[execution.execution_id] = worker
            assert first.cancelled is False

            cancelling = await runtime._state.execution.executions.get(
                execution.execution_id,
                tenant_id=runtime.default_principal.tenant_id,
            )
            assert cancelling is not None
            assert cancelling.status is ExecutionStatus.CANCELLING
            await backend.cancel(cancelling)
            release_tool.set()

            recovery_required = await runtime._state.execution.executions.get(
                execution.execution_id,
                tenant_id=runtime.default_principal.tenant_id,
            )
            assert recovery_required is not None
            assert recovery_required.status is ExecutionStatus.RECOVERY_REQUIRED
            operation = await runtime._state.execution.operations.get(
                cancel_key,
                tenant_id=runtime.default_principal.tenant_id,
            )
            assert operation is not None
            assert operation.status is OperationStatus.EFFECT_UNKNOWN

            replayed = await execution.cancel(idempotency_key=cancel_key)
            assert replayed.cancelled is False
            effects = await execution.recovery_effects()
            assert len(effects) == 1
            interactions_before = await execution.model_interactions(
                include_content=True
            )
            resolution = await execution.resolve_tool_effect(
                effects[0].operation_id,
                expected_fence=effects[0].fence,
                resolution=ToolEffectApplied({"confirmed": True}),
                idempotency_key="resolve-no-worker-effect",
            )
            assert resolution.status is ToolOperationStatus.COMPLETED

            await execution.recover()
            result = (await execution.wait(timeout_seconds=10)).result
            assert result.status is ExecutionStatus.CANCELLED
            assert calls == ["lookup"]
            assert effect_log.read_text().splitlines() == [execution.execution_id]
            settled = await runtime._state.execution.operations.get(
                cancel_key,
                tenant_id=runtime.default_principal.tenant_id,
            )
            assert settled is not None
            assert settled.status is OperationStatus.SUCCEEDED
            interactions_after = await execution.model_interactions(
                include_content=True
            )
            assert interactions_after.items == interactions_before.items
    finally:
        release_tool.set()
        await runtime._state.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", ("unknown_write", "recovery_projection"))
async def test_cancel_at_unknown_effect_boundaries_stays_nonterminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, boundary: str,
) -> None:
    from linktools.ai.runtime._regenerable_turn import _RegenerableTurnRunner

    database = tmp_path / "state.db"
    effect_log = tmp_path / "effects.txt"
    calls = []
    tool_started, release_tool = asyncio.Event(), asyncio.Event()
    boundary_started, release_boundary, stopped = asyncio.Event(), asyncio.Event(), asyncio.Event()
    features = _application(calls, effect_policy="non_replay_safe", effect_log=effect_log,
                            started=tool_started, release=release_tool)
    features.capability(SessionCapability(checkpoints=True), scope="runtime")
    notify_stopped = _RegenerableTurnRunner._producer_stopped

    def producer_stopped(runner, turn):
        notify_stopped(runner, turn)
        stopped.set()

    monkeypatch.setattr(_RegenerableTurnRunner, "_producer_stopped", producer_stopped)
    async with (
        Persistence.open("cancel-unknown-boundary", storage=PersistenceStorage.sqlite(database)) as state,
        Runtime.open(persistence=state, features=model_features(_ToolModels(), features),
                     authorization=_NativeSessionPolicy()) as runtime,
    ):
        principal = runtime.default_principal
        repository = state.checkpoint.tools
        mark_unknown = repository.mark_effect_unknown
        list_operations = repository.list_by_execution
        issued = False

        async def pause_unknown_write(*args, **kwargs):
            nonlocal issued
            if boundary == "unknown_write" and not issued:
                issued = True
                boundary_started.set()
                await release_boundary.wait()
            return await mark_unknown(*args, **kwargs)

        async def pause_recovery_projection(*args, **kwargs):
            nonlocal issued
            operations = await list_operations(*args, **kwargs)
            if boundary == "recovery_projection" and not issued and kwargs.get("_states") and any(
                item.status is ToolOperationStatus.EFFECT_UNKNOWN for item in operations
            ):
                issued = True
                boundary_started.set()
                await release_boundary.wait()
            return operations

        monkeypatch.setattr(repository, "mark_effect_unknown", pause_unknown_write)
        monkeypatch.setattr(repository, "list_by_execution", pause_recovery_projection)
        await runtime.sessions.create_turn_session("session", principal=principal)
        turn = await runtime.sessions.start_turn("session", "inspect", agent="default", principal=principal,
                                            idempotency_key="turn-1")
        try:
            await asyncio.wait_for(tool_started.wait(), timeout=10)
            [claim] = await list_operations(turn.key.turn_id, tenant_id=principal.tenant_id)
            assert claim.status is ToolOperationStatus.CLAIMED
            assert await turn.cancel()
            await asyncio.wait_for(boundary_started.wait(), timeout=10)
            row = await state.session_turns.get_turn(turn.key)
            assert row["state"] == "cancelling" and row["result_json"] is None
            assert row["generation"] == claim.producer_generation
            [effect] = await list_operations(turn.key.turn_id, tenant_id=principal.tenant_id)
            assert effect.status is (ToolOperationStatus.CLAIMED if boundary == "unknown_write"
                                     else ToolOperationStatus.EFFECT_UNKNOWN)
            assert (effect.tool_operation_id, effect.owner, effect.fence) == (
                claim.tool_operation_id, claim.owner, claim.fence)
            assert (await runtime.history.turn_session_state("session", principal=principal))[
                "active_turn_id"] == turn.key.turn_id
        finally:
            release_boundary.set()
            release_tool.set()
        await asyncio.wait_for(stopped.wait(), timeout=10)
        row = await state.session_turns.get_turn(turn.key)
        assert row["state"] == "cancelling" and row["result_json"] is None
        [effect] = await list_operations(turn.key.turn_id, tenant_id=principal.tenant_id)
        assert effect.status is ToolOperationStatus.EFFECT_UNKNOWN
        assert (effect.tool_operation_id, effect.owner, effect.fence) == (
            claim.tool_operation_id, claim.owner, claim.fence)
        page = await runtime.history.turn_session("session", principal=principal)
        assert page["session"]["active_turn_id"] == turn.key.turn_id and page["timeline"]["items"] == []
        assert calls == ["lookup"] and effect_log.read_text().splitlines() == [turn.key.turn_id]


@pytest.mark.asyncio
@pytest.mark.parametrize("readback", ("claimed", "unavailable"))
async def test_unverified_unknown_effect_write_does_not_terminalize_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, readback: str,
) -> None:
    from linktools.ai.runtime._regenerable_turn import _RegenerableTurnRunner

    database = tmp_path / "state.db"
    effect_log = tmp_path / "effects.txt"
    calls = []
    tool_started, release_tool = asyncio.Event(), asyncio.Event()
    attempted, stopped = asyncio.Event(), asyncio.Event()
    proof_reads = []
    features = _application(calls, effect_policy="non_replay_safe", effect_log=effect_log,
                            started=tool_started, release=release_tool)
    features.capability(SessionCapability(checkpoints=True), scope="runtime")
    notify_stopped = _RegenerableTurnRunner._producer_stopped

    def producer_stopped(runner, turn):
        notify_stopped(runner, turn)
        stopped.set()

    monkeypatch.setattr(_RegenerableTurnRunner, "_producer_stopped", producer_stopped)
    async with (
        Persistence.open("unknown-write", storage=PersistenceStorage.sqlite(database)) as state,
        Runtime.open(persistence=state, features=model_features(_ToolModels(), features),
                     authorization=_NativeSessionPolicy()) as runtime,
    ):
        principal = runtime.default_principal
        repository = state.checkpoint.tools
        get_operation = repository.get_operation

        async def fail_unknown_write(*_args, **_kwargs):
            attempted.set()
            raise AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN)

        async def read_operation(operation_id, *, tenant_id):
            if attempted.is_set():
                proof_reads.append(operation_id)
                if readback == "unavailable":
                    raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)
            return await get_operation(operation_id, tenant_id=tenant_id)

        await runtime.sessions.create_turn_session("session", principal=principal)
        turn = await runtime.sessions.start_turn("session", "inspect", agent="default", principal=principal,
                                            idempotency_key="turn-1")
        try:
            await asyncio.wait_for(tool_started.wait(), timeout=10)
            [claim] = await repository.list_by_execution(turn.key.turn_id, tenant_id=principal.tenant_id)
            with monkeypatch.context() as fault:
                fault.setattr(repository, "mark_effect_unknown", fail_unknown_write)
                fault.setattr(repository, "get_operation", read_operation)
                assert await turn.cancel()
                await asyncio.wait_for(stopped.wait(), timeout=10)
                assert attempted.is_set() and set(proof_reads) == {claim.tool_operation_id}
            row = await state.session_turns.get_turn(turn.key)
            assert row["state"] == "cancelling" and row["result_json"] is None
            assert row["generation"] == claim.producer_generation
            assert await repository.get_operation(claim.tool_operation_id, tenant_id=principal.tenant_id) == claim
            assert claim.status is ToolOperationStatus.CLAIMED
            page = await runtime.history.turn_session("session", principal=principal)
            assert page["session"]["active_turn_id"] == turn.key.turn_id and page["timeline"]["items"] == []
            assert calls == ["lookup"] and effect_log.read_text().splitlines() == [turn.key.turn_id]
        finally:
            release_tool.set()

    async with Persistence.open("unknown-write", storage=PersistenceStorage.sqlite(database), read_only=True) as state:
        row = await state.session_turn_reader.get_turn(turn.key)
        assert row["state"] == "cancelling" and row["result_json"] is None
        owner = await state.session_turn_reader.get_session("session")
        assert owner["active_turn_id"] == turn.key.turn_id.encode() and owner["last_turn_no"] == 0
        assert await state.checkpoint.tools.get_operation(claim.tool_operation_id, tenant_id=principal.tenant_id) == claim


@pytest.mark.asyncio
async def test_durable_terminal_survives_local_seal_finalization_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = PersistenceStorage.in_memory()
    calls: list[str] = []
    application = _application(calls)

    async def fail_finalize(
        _store: RuntimeAgentRunStore,
        _plan: object,
    ) -> None:
        raise RuntimeError("injected terminal seal finalization failure")

    monkeypatch.setattr(
        RuntimeAgentRunStore,
        "finalize_execution_terminal_seal",
        fail_finalize,
    )
    try:
        application.capability(native_checkpoint(), scope="runtime")
        application.capability(TraceCapability(), scope="runtime")
        async with Persistence.open('session-tool-terminal-finalize', storage=state) as persistence_1740, Runtime.open(persistence=persistence_1740, features=model_features(_ToolModels(), application)) as runtime:
            await runtime.agents.get("default").create_session("session")
            execution = (
                await runtime.agents.get("default")
                .session("session")
                .start(
                    "inspect",
                    idempotency_key="turn-1",
                )
            )
            watched = [
                item
                async for item in execution.watch(include_content=True)
                if item.depth == 0
            ]
            result = (await execution.wait(timeout_seconds=10)).result

            assert result.status is ExecutionStatus.SUCCEEDED
            assert calls == ["lookup"]
            assert (
                watched[-1].event.event_type == ExecutionEventType.EXECUTION_SUCCEEDED
            )
            stored_execution = await runtime._state.execution.executions.get(
                execution.execution_id,
                tenant_id=runtime.tenant_id,
            )
            stored_session = await runtime._state.conversation.sessions.get(
                "session",
                tenant_id=runtime.tenant_id,
            )
            assert stored_execution is not None and stored_execution.retention_closed
            assert stored_session is not None
            assert stored_session.active_execution_id is None
    finally:
        await runtime._state.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "terminal_code",
    (
        ErrorCode.STORAGE_INTEGRITY_ERROR,
        ErrorCode.STORAGE_COMMIT_UNKNOWN,
    ),
)
async def test_terminal_commit_error_converges_to_failed_terminal(
    monkeypatch: pytest.MonkeyPatch,
    terminal_code: ErrorCode,
) -> None:
    calls: list[str] = []
    application = _application(calls)
    original = StateCommands.commit_terminal_checkpoint
    injected = False

    async def fail_success_once(
        self: StateCommands,
        commit: object,
        **kwargs: object,
    ) -> object:
        nonlocal injected
        execution = getattr(commit, "execution", None)
        if (
            not injected
            and execution is not None
            and execution.status is ExecutionStatus.SUCCEEDED
        ):
            injected = True
            raise AIError(
                terminal_code,
                safe_details={"phase": "terminal_commit"},
            )
        return await original(self, commit, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(
        StateCommands,
        "commit_terminal_checkpoint",
        fail_success_once,
    )
    application.capability(native_checkpoint(), scope="runtime")
    application.capability(TraceCapability(), scope="runtime")
    async with Persistence.open('session-tool-terminal-failure', storage=PersistenceStorage.in_memory()) as persistence_1824, Runtime.open(persistence=persistence_1824, features=model_features(_ToolModels(), application)) as runtime:
        await runtime.agents.get("default").create_session("session")
        session = runtime.agents.get("default").session("session")
        execution = await session.start("inspect", idempotency_key="turn-1")
        watched = [
            item
            async for item in execution.watch(include_content=True)
            if item.depth == 0
        ]
        result = (await execution.wait(timeout_seconds=10)).result

        assert injected
        assert result.status is ExecutionStatus.FAILED
        assert result.error_code == terminal_code.value
        assert calls == ["lookup"]
        assert watched[-1].event.event_type == ExecutionEventType.EXECUTION_FAILED
        assert watched[-1].event.payload["error_code"] == result.error_code

        same = await session.start(
            "inspect",
            idempotency_key="turn-1",
        )
        same_result = (await same.wait(timeout_seconds=10)).result
        assert same.execution_id == execution.execution_id
        assert same_result.status is ExecutionStatus.FAILED
        assert calls == ["lookup"]

        retry = (await session.run(
            "retry",
            idempotency_key="turn-2",
            timeout_seconds=10,
        )).result
        assert retry.status is ExecutionStatus.SUCCEEDED
        assert calls == ["lookup", "lookup"]


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("sqlite", "sql"))
async def test_completed_tool_operation_is_reused_after_reopen(
    tmp_path: Path,
    backend: str,
) -> None:
    database = tmp_path / f"tool-replay-{backend}.db"
    first_engine: AsyncEngine | None = None
    if backend == "sqlite":
        first = PersistenceStorage.sqlite(database)
    else:
        first_engine = create_async_engine(
            URL.create("sqlite+aiosqlite", database=str(database))
        )
        await provision_runtime_database(first_engine)
        first = PersistenceStorage.sql(first_engine)
    first = Persistence(first)
    await first.initialize(namespace="tool-replay", tenant_id="tenant")
    try:
        first_bridge = RuntimeToolOperationBridge(
            first.checkpoint.tools,
            first.object_store(PersistenceDomain.CHECKPOINT),
            namespace="tool-replay",
            tenant_id="tenant",
            execution_id="execution",
            agent_run_id="run-1",
            binding_digest="a" * 64,
            owner="worker-1",
            background_tasks=set(),

        )
        context = RunContext(
            deps=None,
            model=TestModel(),
            usage=RunUsage(),
            run_id="run-1",
        )
        call = ToolCallPart("lookup", {}, tool_call_id="call-1")
        tool = ToolDefinition(name="lookup")
        decision = await first_bridge.begin(context, call, tool, {}, True)
        assert not decision.has_cached_result
        await first_bridge.complete(decision, "tool-result")
    finally:
        await first.close()
        if first_engine is not None:
            await first_engine.dispose()

    second_engine: AsyncEngine | None = None
    if backend == "sqlite":
        second = PersistenceStorage.sqlite(database)
    else:
        second_engine = create_async_engine(
            URL.create("sqlite+aiosqlite", database=str(database))
        )
        second = PersistenceStorage.sql(second_engine)
    second = Persistence(second)
    await second.initialize(namespace="tool-replay", tenant_id="tenant")
    try:
        replay_bridge = RuntimeToolOperationBridge(
            second.checkpoint.tools,
            second.object_store(PersistenceDomain.CHECKPOINT),
            namespace="tool-replay",
            tenant_id="tenant",
            execution_id="execution",
            agent_run_id="run-2",
            recovery_agent_run_id="run-1",
            operation_ids={"call-1": decision.operation_id},
            binding_digest="a" * 64,
            owner="worker-2",
            background_tasks=set(),

        )
        replay_context = RunContext(
            deps=None,
            model=TestModel(),
            usage=RunUsage(),
            run_id="run-2",
        )
        replay = await replay_bridge.begin(
            replay_context,
            call,
            tool,
            {},
            True,
        )
        assert replay.has_cached_result
        assert replay.cached_result == "tool-result"
    finally:
        await second.close()
        if second_engine is not None:
            await second_engine.dispose()


@pytest.fixture(scope="module")
def unconfirmed_tool_source_copy(
    tmp_path_factory: pytest.TempPathFactory,
) -> Callable[[Path], Awaitable[tuple[Path, Path, Path]]]:
    frozen: Path | None = None

    async def copy(destination: Path) -> tuple[Path, Path, Path]:
        nonlocal frozen
        if frozen is None:
            source = tmp_path_factory.mktemp("unconfirmed-tool-process")
            with pytest.MonkeyPatch.context() as patch:
                patch.setenv("LINKTOOLS_TEST_STEP_DATABASE", str(source / "native.db"))
                await _exit_at_boundary(
                    source / "runtime.db", source / "effects.txt", "sqlite", "standalone_effect_unconfirmed",
                )
            frozen = source.rename(source.with_name(source.name + "-stopped"))
        # The process is dead. Copy raw databases and their WAL/SHM sidecars
        # without opening SQLite, so each reader performs its own crash recovery.
        shutil.copytree(frozen, destination, dirs_exist_ok=True)
        return destination / "runtime.db", destination / "effects.txt", destination / "native.db"

    return copy


@pytest.mark.asyncio
async def test_abandoned_tool_claim_waits_for_explicit_recovery_after_restart(
    tmp_path: Path,
    unconfirmed_tool_source_copy: Callable[[Path], Awaitable[tuple[Path, Path, Path]]],
) -> None:
    database, effect_log, native_database = await unconfirmed_tool_source_copy(tmp_path)
    committed_effects = effect_log.read_text().splitlines()
    assert len(committed_effects) == 1
    execution_id = committed_effects[0]
    calls: list[str] = []
    state = PersistenceStorage.sqlite(database)
    application = _application(
        calls,
        effect_policy="non_replay_safe",
        effect_log=effect_log,
    )
    try:
        application.capability(native_checkpoint(database=native_database), scope="runtime")
        application.capability(TraceCapability(), scope="runtime")
        async with Persistence.open('session-tool-crash', storage=state) as persistence_1981, Runtime.open(persistence=persistence_1981, features=model_features(_ToolModels(), application)) as runtime:
            current = await runtime._state.execution.executions.get(
                execution_id,
                tenant_id=runtime.default_principal.tenant_id,
            )
            assert current is not None and current.session_id is None
            assert current.status is ExecutionStatus.STARTED
            tools = await runtime._state.checkpoint.tools.list_by_execution(
                execution_id,
                tenant_id=runtime.default_principal.tenant_id,
            )
            assert len(tools) == 1
            assert tools[0].status is ToolOperationStatus.CLAIMED

            execution = await runtime.executions.get(execution_id)
            assert (await runtime.executions.inspect(
                execution_id, principal=runtime.default_principal,
            )).status is ExecutionStatus.STARTED
            assert not runtime._execution_service.runtime_backend().worker_installed(execution_id)
            with pytest.raises(AIError) as raised:
                await execution.recover()
                await execution.wait(timeout_seconds=10)

            assert raised.value.code is ErrorCode.TOOL_EFFECT_UNKNOWN
            recovered = await runtime._state.execution.executions.get(
                execution_id,
                tenant_id=runtime.default_principal.tenant_id,
            )
            assert recovered is not None
            assert recovered.status is ExecutionStatus.RECOVERY_REQUIRED
            [unresolved] = await runtime._state.checkpoint.tools.list_by_execution(
                execution_id, tenant_id=runtime.default_principal.tenant_id,
            )
            assert unresolved.status is ToolOperationStatus.EFFECT_UNKNOWN
            assert (unresolved.tool_operation_id, unresolved.owner, unresolved.fence, unresolved.producer_generation) == (
                tools[0].tool_operation_id, tools[0].owner, tools[0].fence, tools[0].producer_generation,
            )
            assert calls == []
            assert effect_log.read_text().splitlines() == committed_effects
    finally:
        await runtime._state.close()


@pytest.mark.asyncio
async def test_idempotent_replay_does_not_claim_active_tool_execution(
    tmp_path: Path,
    unconfirmed_tool_source_copy: Callable[[Path], Awaitable[tuple[Path, Path, Path]]],
) -> None:
    database, effect_log, native_database = await unconfirmed_tool_source_copy(tmp_path)
    committed_effects = effect_log.read_text().splitlines()
    assert len(committed_effects) == 1
    execution_id = committed_effects[0]
    calls: list[str] = []
    state = PersistenceStorage.sqlite(database)
    application = _application(
        calls,
        effect_policy="non_replay_safe",
        effect_log=effect_log,
    )
    try:
        application.capability(native_checkpoint(database=native_database), scope="runtime")
        application.capability(TraceCapability(), scope="runtime")
        async with Persistence.open('session-tool-crash', storage=state) as persistence_2047, Runtime.open(persistence=persistence_2047, features=model_features(_ToolModels(), application)) as runtime:
            tools = await runtime._state.checkpoint.tools.list_by_execution(
                execution_id,
                tenant_id=runtime.default_principal.tenant_id,
            )
            assert len(tools) == 1
            assert tools[0].status is ToolOperationStatus.CLAIMED
            before_replay = await runtime._state.execution.executions.get(
                execution_id, tenant_id=runtime.default_principal.tenant_id,
            )
            before_head = await runtime._state.execution.executions.get(
                execution_id, tenant_id=runtime.default_principal.tenant_id,
            )

            same = await runtime.agents.get("default").start("inspect", idempotency_key="turn-1")
            assert same.execution_id == execution_id
            assert await runtime._state.execution.executions.get(
                execution_id, tenant_id=runtime.default_principal.tenant_id,
            ) == before_replay
            assert await runtime._state.execution.executions.get(
                execution_id, tenant_id=runtime.default_principal.tenant_id,
            ) == before_head
            assert not runtime._execution_service.runtime_backend().worker_installed(execution_id)
            assert await runtime._state.checkpoint.tools.list_by_execution(
                execution_id, tenant_id=runtime.default_principal.tenant_id,
            ) == tools
            with pytest.raises(AIError) as raised:
                await same.recover()
                await same.wait(timeout_seconds=10)
            assert raised.value.code is ErrorCode.TOOL_EFFECT_UNKNOWN

            recovered = await runtime._state.execution.executions.get(
                execution_id,
                tenant_id=runtime.default_principal.tenant_id,
            )
            assert recovered is not None
            assert recovered.status is ExecutionStatus.RECOVERY_REQUIRED
            assert await runtime._state.execution.executions.get_result(
                execution_id,
                tenant_id=runtime.default_principal.tenant_id,
            ) is None
            assert calls == []
            assert effect_log.read_text().splitlines() == committed_effects
    finally:
        await runtime._state.close()


def _split_terminal_setup(root):
    from linktools.ai.persistence import PersistencePlan
    from linktools.ai.storage import StorageRoute

    async def model(_messages, _info):
        with (root / "provider.log").open("a") as output:
            output.write("model\n")
        return ModelResponse(parts=[TextPart("durable answer")])

    step = StorageRoute.sqlite(root / "step.db")
    storage = PersistenceStorage(PersistencePlan(
        conversation=StorageRoute.sqlite(root / "conversation.db"), execution=step, checkpoint=step,
    ))
    features = RuntimeFeatures("application")
    features.capability(SessionCapability(checkpoints=True), scope="runtime")
    features.agent("default", model="default", allow_tools=(), allow_subagents=())
    return storage, model_features(_Models(_UsageFunctionModel(model)), features)


async def _send_split_terminal(runtime):
    return await runtime.sessions.start_turn(
        "session", "question", agent="default", principal=runtime.default_principal, idempotency_key="first",
    )


def _crash_split_terminal(root, phase):
    import os

    from linktools.ai.persistence.turns import SqlSessionTurns

    publish = SqlSessionTurns.publish_split_terminal

    async def crashing_publish(store, key, **kwargs):
        assert kwargs["source"].phase == "sealed" and kwargs["state"] == "succeeded"
        (root / "turn-id").write_text(key.turn_id)
        if phase == "target-committed":
            assert await publish(store, key, **kwargs)
        os._exit(91)

    SqlSessionTurns.publish_split_terminal = crashing_publish

    async def main():
        storage, features = _split_terminal_setup(root)
        async with (
            Persistence.open("split-terminal", storage=storage) as state,
            Runtime.open(persistence=state, features=features, authorization=_NativeSessionPolicy()) as runtime,
        ):
            await runtime.sessions.create_turn_session("session", principal=runtime.default_principal)
            turn = await _send_split_terminal(runtime)
            await turn.wait_final(timeout_seconds=10)
        raise AssertionError("expected subprocess exit")

    asyncio.run(main())


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ("source-sealed", "target-committed"))
async def test_split_storage_handoff_reconciles_prepared_checkpoint_after_restart(tmp_path: Path, phase: str) -> None:
    from linktools.ai.persistence.turns import SessionTurnKey

    from .test_session_tool_process_recovery import _expire, _run_until_crash

    await _run_until_crash(_crash_split_terminal, tmp_path, phase)
    key = SessionTurnKey("session", (tmp_path / "turn-id").read_text())
    storage, features = _split_terminal_setup(tmp_path)
    async with (
        Persistence.open("split-terminal", storage=storage) as state,
        Runtime.open(persistence=state, features=features, authorization=_NativeSessionPolicy()) as runtime,
    ):
        principal = runtime.default_principal
        turns = state.session_turns
        source = await runtime.sessions._turn_runner._split_recovery_source()
        sealed = await source._read_pin(source._identity(key, 1))
        assert sealed.phase == "sealed" and sealed.pin.terminal_state == "succeeded"
        row = await turns.get_turn(key)
        if phase == "source-sealed":
            assert row["state"] == "accepted" and row["split_source_receipt"] is None
            assert (await turns.get_session("session"))["active_turn_id"] == key.turn_id.encode()
            await _expire(turns, key, generation=1)
            assert await runtime.sessions.recover_turn("session", key.turn_id, mode="resume", principal=principal)
            published = await source._read_pin(source._identity(key, 1))
            assert published.phase == "published" and published.pin == sealed.pin
        else:
            assert (row["state"], row["split_source_receipt"]) == ("succeeded", sealed.pin.receipt_id.encode())
        assert not await runtime.sessions.recover_turn("session", key.turn_id, mode="resume", principal=principal)
        replay = await _send_split_terminal(runtime)
        assert replay.key == key
        terminal = await replay.wait_final(timeout_seconds=10)
        assert terminal.status is ExecutionStatus.SUCCEEDED
        assert terminal.view["output"] == {"text": "durable answer"}
        row = await turns.get_turn(key)
        assert (row["state"], row["generation"]) == ("succeeded", 1)
        assert row["split_source_receipt"] == sealed.pin.receipt_id.encode()
        assert row["result_json"]["transcript"] == sealed.pin.transcript.to_json()
        page = await runtime.history.turn_session("session", principal=principal)
        assert page["session"]["active_turn_id"] is None and page["timeline"]["items"] == [terminal.view]
    assert (tmp_path / "provider.log").read_text().splitlines() == ["model"]


@pytest.mark.asyncio
async def test_split_storage_success_handoff_wins_cancel_race(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from linktools.ai.persistence._split_step_source import SqlSplitStepSource

    target_committed = asyncio.Event()
    release = asyncio.Event()
    reconcile = SqlSplitStepSource.reconcile

    async def paused(source, connection, **kwargs):
        target_committed.set()
        await release.wait()
        return await reconcile(source, connection, **kwargs)

    monkeypatch.setattr(SqlSplitStepSource, "reconcile", paused)
    storage, features = _split_terminal_setup(tmp_path)
    async with (
        Persistence.open("split-terminal", storage=storage) as state,
        Runtime.open(persistence=state, features=features, authorization=_NativeSessionPolicy()) as runtime,
    ):
        principal = runtime.default_principal
        await runtime.sessions.create_turn_session("session", principal=principal)
        turn = await _send_split_terminal(runtime)
        try:
            await asyncio.wait_for(target_committed.wait(), 10)
            source = await runtime.sessions._turn_runner._split_recovery_source()
            sealed = await source._read_pin(source._identity(turn.key, 1))
            assert sealed.phase == "sealed"
            row = await state.session_turns.get_turn(turn.key)
            assert (row["state"], row["generation"], row["split_source_receipt"]) == (
                "succeeded", 1, sealed.pin.receipt_id.encode(),
            )
            assert not await turn.cancel()
            assert not await turn.cancel()
        finally:
            release.set()
        terminal = await turn.wait_final(timeout_seconds=10)
        published = await source._read_pin(source._identity(turn.key, 1))
        assert published.phase == "published" and published.pin == sealed.pin
        assert terminal.status is ExecutionStatus.SUCCEEDED
        assert terminal.view["output"] == {"text": "durable answer"}
        page = await runtime.history.turn_session("session", principal=principal)
        assert page["session"]["active_turn_id"] is None and page["timeline"]["items"] == [terminal.view]
    assert (tmp_path / "provider.log").read_text().splitlines() == ["model"]


@pytest.mark.asyncio
async def test_trace_checkpoint_readback_ignores_transient_before_coordinate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = PersistenceStorage.in_memory()
    state = Persistence(state)
    await state.initialize(namespace="checkpoint-readback", tenant_id="tenant")
    recovery = state.run_store.read_store(PersistenceDomain.TRACE)
    assert isinstance(recovery, StateStepArchive)
    group = recovery.state_store.storage_group
    original = group.mutate
    injected = False

    depth = 0

    async def commit_then_fail(stores: Any, callback: Any) -> Any:
        nonlocal injected, depth
        outer = depth == 0
        depth += 1
        try:
            result = await original(stores, callback)
            if outer and not injected:
                injected = True
                raise AIError(ErrorCode.STORAGE_UNAVAILABLE)
            return result
        finally:
            depth -= 1

    from .test_step_projection import _execution
    await state.execution.executions.create_execution(_execution())
    monkeypatch.setattr(group, "mutate", commit_then_fail)
    try:
        run = AgentRunRecord("run", agent_conversation_id="conversation", agent_id="default")
        checkpoint = AgentRunCheckpoint(
            agent_run_id="run",
            step_index=1,
            messages=[
                ModelRequest(parts=[UserPromptPart("inspect")]),
                ModelResponse(parts=[TextPart("done")]),
            ],
            state="complete",
            transcript_message_count_before=0,
        )
        await state.run_store.register_agent_run(run)
        await state.run_store.save_checkpoint(checkpoint)
        await state.run_store.flush_execution_projection("run", execution_id="execution")
        assert injected

        recovery = state.run_store.read_store(PersistenceDomain.TRACE)
        assert isinstance(recovery, StateStepArchive)
        assert await recovery.transcript_message_count_for_run(run) == len(
            checkpoint.messages
        )
        stored = await recovery.latest_checkpoint(
            agent_run_id=run.agent_run_id,
            include_interrupted=True,
        )
        assert stored is not None
        assert stored.transcript_message_count_before is None
        assert tuple(stored.messages) == tuple(checkpoint.messages)
    finally:
        await state.close()


@pytest.mark.asyncio
async def test_conversation_checkpoint_relocation_is_idempotent_and_validates_prefix() -> None:
    from dataclasses import replace

    state = PersistenceStorage.in_memory()
    state = Persistence(state)
    await state.initialize(namespace="run-checkpoint-relocation", tenant_id="tenant")
    try:
        recovery = state.run_store.read_store(PersistenceDomain.CONVERSATION)
        assert isinstance(recovery, StateStepArchive)
        run = AgentRunRecord("run", agent_conversation_id="conversation", agent_id="default")
        await recovery.register_agent_run(run)
        first = AgentRunCheckpoint(
            agent_run_id="run", agent_conversation_id="conversation", agent_id="default",
            step_index=1,
            messages=[ModelRequest(parts=[UserPromptPart("inspect")])],
            state="active",
            transcript_message_count_before=0,
        )
        final = AgentRunCheckpoint(
            agent_run_id="run", agent_conversation_id="conversation", agent_id="default",
            step_index=2,
            messages=[
                *first.messages,
                ModelResponse(parts=[TextPart("done")]),
            ],
            state="complete",
            transcript_message_count_before=1,
        )
        await recovery.materialize_checkpoint(run, first)
        await recovery.materialize_checkpoint(run, final)
        before = await recovery.transcript_message_count_for_run(run)

        relocated = await recovery.relocate_conversation_checkpoint(run, final)
        assert relocated.transcript_message_count_before == len(final.messages)
        await recovery.materialize_checkpoint(run, relocated)
        assert await recovery.transcript_message_count_for_run(run) == before

        bad_run = AgentRunRecord(
            "bad-run",
            agent_conversation_id="conversation",
            agent_id="default",
        )
        bad_checkpoint = AgentRunCheckpoint(
            agent_run_id="bad-run", agent_conversation_id="conversation", agent_id="default",
            step_index=2,
            messages=[ModelRequest(parts=[UserPromptPart("stored")]), ModelResponse(parts=[TextPart("done")])],
            state="complete", transcript_message_count_before=0,
        )
        await recovery.materialize_checkpoint(bad_run, bad_checkpoint)
        with pytest.raises(AIError) as raised:
            await recovery.relocate_conversation_checkpoint(
                bad_run, replace(bad_checkpoint, messages=[
                    ModelRequest(parts=[UserPromptPart("different")]), bad_checkpoint.messages[1],
                ]),
            )
        assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
    finally:
        await state.close()


@pytest.mark.asyncio
async def test_conversation_rebases_cumulative_native_tool_checkpoint() -> None:
    state = PersistenceStorage.in_memory()
    state = Persistence(state)
    await state.initialize(namespace="session-tool-recovery", tenant_id="tenant")
    try:
        native = native_checkpoint().store
        state.run_store.bind_step_store(native)
        state.run_store.bind_execution_persistence(
            "run", conversation=True, restart_checkpoints=True, durable_trace=False,
        )
        conversation = state.run_store.read_store(PersistenceDomain.CONVERSATION)
        assert isinstance(conversation, StateStepArchive)

        run = AgentRunRecord(
            "run",
            agent_conversation_id="conversation",
            agent_id="default",
            metadata={"history_id": "history"},
        )
        await state.run_store.register_agent_run(run)
        first_messages = [
            ModelRequest(parts=[UserPromptPart("inspect")]),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        "lookup",
                        {},
                        tool_call_id="call",
                    )
                ]
            ),
        ]
        final_messages = [
            *first_messages,
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        "lookup",
                        "tool-result",
                        tool_call_id="call",
                    )
                ]
            ),
            ModelResponse(parts=[TextPart("done")]),
        ]
        for message in final_messages:
            message.run_id = run.agent_run_id
        for index, messages in enumerate((first_messages, final_messages), 1):
            await seed_native_snapshot(
                native, run_id=run.agent_run_id, messages=messages, step_index=index,
                conversation_id=run.agent_conversation_id, agent_name=run.agent_id,
                metadata=dict(run.metadata),
            )
            await state.run_store.save_checkpoint(AgentRunCheckpoint(
                agent_run_id=run.agent_run_id, step_index=index, messages=messages,
                context_messages=messages, state="complete", transcript_message_count_before=0,
            ))
        for _ in range(2):
            await state.run_store.materialize_conversation(agent_run_id=run.agent_run_id)
            await state.run_store.materialize_from_recovery(
                target=PersistenceDomain.CONVERSATION, agent_run_id=run.agent_run_id,
            )

        stored = await conversation.latest_checkpoint(agent_run_id=run.agent_run_id)
        assert stored is not None
        assert stored.state == "complete"
        assert tuple(stored.messages[-len(final_messages) :]) == tuple(final_messages)
        assert await conversation.transcript_message_count_for_run(run) == len(
            final_messages
        )
    finally:
        await state.close()


def _crash_session_bootstrap(root, phase, turn_id=None):
    import os

    from linktools.ai.persistence.turns import SessionTurnKey, SqlSessionTurns
    from linktools.ai.runtime._regenerable_turn import _RegenerableTurnRunner
    from linktools.ai.runtime._step_store import RuntimeStepStore

    from .test_session_tool_process_recovery import _NAME, _SESSION, _expire, _features, _send

    original_accept = SqlSessionTurns.accept
    original_step = RuntimeStepStore.commit_step

    async def accept(store, key, **kwargs):
        accepted = await original_accept(store, key, **kwargs)
        assert accepted
        (root / "turn-id").write_text(key.turn_id)
        if phase == "activated":
            os._exit(91)
        return accepted

    async def commit_step(store, **kwargs):
        await original_step(store, **kwargs)
        snapshot = kwargs.get("snapshot")
        if (phase == "request_checkpoint" and snapshot is not None and snapshot.messages
                and not any(isinstance(message, ModelResponse) for message in snapshot.messages)):
            os._exit(91)

    async def finish(runner, turn, **kwargs):
        assert phase == "recovered_before_terminal"
        assert kwargs["state"] == "succeeded" and turn.generation == 2
        os._exit(91)

    if phase == "recovered_before_terminal":
        _RegenerableTurnRunner._finish_retained_turn = finish
    else:
        SqlSessionTurns.accept = accept
        RuntimeStepStore.commit_step = commit_step

    async def run():
        async with (
            Persistence.open(_NAME, storage=PersistenceStorage.from_root(root / "runtime")) as state,
            Runtime.open(persistence=state, features=_features(root), authorization=_NativeSessionPolicy()) as runtime,
        ):
            if phase == "recovered_before_terminal":
                await _expire(state.session_turns, SessionTurnKey(_SESSION, turn_id), generation=1)
                assert await runtime.sessions.recover_turn(_SESSION, turn_id, mode="resume", principal=runtime.default_principal)
            else:
                await runtime.agents.get().create_session(_SESSION)
                turn = await _send(runtime)
                await turn.wait_final(timeout_seconds=10)
        raise AssertionError("bootstrap boundary was not reached")

    asyncio.run(run())


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "phase", (
        "activated",
        pytest.param("request_checkpoint", marks=pytest.mark.merge),
        pytest.param("recovered_before_terminal", marks=pytest.mark.merge),
        "effect_unconfirmed",
    )
)
async def test_recovery_preserves_bootstrap_and_effect_confirmation_boundaries(tmp_path: Path, phase: str) -> None:
    from linktools.ai.persistence.turns import SessionTurnKey

    from .test_session_tool_process_recovery import (
        _NAME, _SESSION, _crash_native_process, _expire, _features, _run_until_crash, _send,
    )

    effect_log = tmp_path / "effects.log"
    provider_log = tmp_path / "provider.log"
    interrupted = phase == "effect_unconfirmed"
    if phase in ("activated", "request_checkpoint"):
        await _run_until_crash(_crash_session_bootstrap, tmp_path, phase)
        turn_id = (tmp_path / "turn-id").read_text()
        assert not effect_log.exists() and not provider_log.exists()
    else:
        await _run_until_crash(
            _crash_native_process, str(tmp_path), "tool_interrupted" if interrupted else "tool_completed",
        )
        [turn_id] = effect_log.read_text().splitlines()
        if not interrupted:
            await _run_until_crash(_crash_session_bootstrap, tmp_path, phase, turn_id)

    predecessor = 2 if phase == "recovered_before_terminal" else 1
    key = SessionTurnKey(_SESSION, turn_id)
    async with (
        Persistence.open(_NAME, storage=PersistenceStorage.from_root(tmp_path / "runtime")) as state,
        Runtime.open(persistence=state, features=_features(tmp_path, interrupted=interrupted),
                     authorization=_NativeSessionPolicy()) as runtime,
    ):
        turns = state.session_turns
        principal = runtime.default_principal
        assert (await turns.get_turn(key))["generation"] == predecessor
        before = await state.checkpoint.tools.list_by_execution(turn_id, tenant_id=runtime.tenant_id)
        if phase in ("activated", "request_checkpoint"):
            assert not before
        else:
            [original] = before
            assert original.tool_call_id == "stable-call"
            assert original.status is (ToolOperationStatus.CLAIMED if interrupted else ToolOperationStatus.COMPLETED)
        await _expire(turns, key, generation=predecessor)
        if phase == "activated":
            # An absent checkpoint cannot prove that the previous host never
            # dispatched. Refuse recovery without advancing its expired lease.
            expired = await turns.get_turn(key)
            for _ in range(2):
                with pytest.raises(AIError) as unavailable:
                    await runtime.sessions.recover_turn(_SESSION, turn_id, mode="resume", principal=principal)
                assert unavailable.value.code is ErrorCode.STORAGE_RECOVERY_REQUIRED
                assert await turns.get_turn(key) == expired
            assert (await turns.get_session(_SESSION))["active_turn_id"] == key.turn_id.encode()
            assert not provider_log.exists() and not effect_log.exists()
            return

        assert await runtime.sessions.recover_turn(_SESSION, turn_id, mode="resume", principal=principal)
        recovered = await _send(runtime)
        assert recovered.key == key
        terminal = await recovered.wait_final(timeout_seconds=10)
        assert terminal.status is ExecutionStatus.SUCCEEDED
        assert terminal.view["output"] == {"text": "done"}
        assert terminal.view["conversation_committed"] is True
        page = await runtime.history.turn_session(_SESSION, principal=principal)
        assert page["session"]["active_turn_id"] is None
        assert page["timeline"]["items"] == [terminal.view]
        receipts = await state.checkpoint.tools.list_by_execution(turn_id, tenant_id=runtime.tenant_id)
        if before:
            [original] = before
            [matching] = [item for item in receipts if item.tool_operation_id == original.tool_operation_id]
            assert matching == original
        else:
            [receipt] = receipts
            assert receipt.status is ToolOperationStatus.COMPLETED
    assert effect_log.read_text().splitlines() == [turn_id]


@pytest.mark.asyncio
async def test_tool_effect_waits_for_durable_response_checkpoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from linktools.ai.persistence import SqlStepStore

    from .test_session_tool_process_recovery import _NAME, _SESSION, _features, _send

    original = SqlStepStore.write_prepared_step
    rejected = asyncio.Event()

    async def reject_response(store, transaction, prepared, **kwargs):
        snapshot = prepared.snapshot
        if snapshot is not None and any(
            isinstance(part, ToolCallPart) and part.tool_name == "effect"
            for message in snapshot.messages if isinstance(message, ModelResponse)
            for part in message.parts
        ):
            rejected.set()
            raise AIError(ErrorCode.STORAGE_UNAVAILABLE)
        return await original(store, transaction, prepared, **kwargs)

    monkeypatch.setattr(SqlStepStore, "write_prepared_step", reject_response)
    async with (
        Persistence.open(_NAME, storage=PersistenceStorage.from_root(tmp_path / "runtime")) as state,
        Runtime.open(persistence=state, features=_features(tmp_path), authorization=_NativeSessionPolicy()) as runtime,
    ):
        await runtime.agents.get().create_session(_SESSION)
        turn = await _send(runtime)
        await asyncio.wait_for(rejected.wait(), 10)
        terminal = await turn.wait_final(timeout_seconds=10)
        assert terminal.status is ExecutionStatus.FAILED
        assert terminal.view["error_code"] == ErrorCode.STORAGE_UNAVAILABLE.value
        assert not await state.checkpoint.tools.list_by_execution(turn.key.turn_id, tenant_id=runtime.tenant_id)
    assert (tmp_path / "provider.log").read_text().splitlines() == ["initial"]
    assert not (tmp_path / "effects.log").exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "sqlite"))
async def test_terminal_transaction_rollback_keeps_session_cursor_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend: str
) -> None:
    if backend == "sqlite":
        from linktools.ai.runtime._regenerable_turn import _RegenerableTurnRunner

        storage = PersistenceStorage.sqlite(tmp_path / "session-terminal-rollback.db")
        calls: list[str] = []
        prompts = []

        async def model(messages, _info):
            if any(isinstance(part, ToolReturnPart) for part in messages[-1].parts):
                return ModelResponse(parts=[TextPart("done")])
            assert not any(isinstance(part, ToolReturnPart) for message in messages for part in message.parts)
            prompts.append([part.content for message in messages if isinstance(message, ModelRequest)
                            for part in message.parts if isinstance(part, UserPromptPart)])
            return ModelResponse(parts=[ToolCallPart("lookup", {}, tool_call_id=f"lookup-{len(prompts)}")])

        features = _application(calls)
        features.capability(SessionCapability(checkpoints=True), scope="runtime")
        stopped = asyncio.Event()
        notify_stopped = _RegenerableTurnRunner._producer_stopped

        def producer_stopped(runner, turn):
            notify_stopped(runner, turn)
            stopped.set()

        async with (
            Persistence.open("session-atomic-terminal", storage=storage) as state,
            Runtime.open(persistence=state, features=model_features(_Models(_UsageFunctionModel(model)), features),
                         authorization=_NativeSessionPolicy()) as runtime,
        ):
            principal = runtime.default_principal
            sessions, store = runtime.sessions, state.session_turns
            await sessions.create_turn_session("session", principal=principal)
            before = await runtime.history.turn_session_state("session", principal=principal)
            finish = store._finish_in_group
            injected = []

            async def rollback(transaction, prepared):
                assert await finish(transaction, prepared)
                injected.append(True)
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

            with monkeypatch.context() as fault:
                fault.setattr(store, "_finish_in_group", rollback)
                fault.setattr(_RegenerableTurnRunner, "_producer_stopped", producer_stopped)
                turn = await sessions.start_turn("session", "inspect", agent="default", principal=principal,
                                            idempotency_key="turn-1")
                await asyncio.wait_for(stopped.wait(), timeout=10)

            assert injected == [True] and calls == ["lookup"]
            active = await runtime.history.turn_session("session", principal=principal)
            assert active["session"]["last_turn_no"] == before["last_turn_no"]
            assert active["session"]["active_turn_id"] == turn.key.turn_id
            assert active["timeline"]["items"] == []
            row = await store.get_turn(turn.key)
            assert row["state"] == "accepted" and row["result_json"] is None

            replay = await sessions.start_turn("session", "inspect", agent="default", principal=principal,
                                          idempotency_key="turn-1")
            assert replay.key == turn.key and calls == ["lookup"]
            async with store._context.engine.begin() as connection:
                expired = await connection.execute(store._turns.update().filter_by(
                    **store._turn_values(turn.key), generation=row["generation"],
                ).values(lease_until=0))
                assert expired.rowcount == 1
            assert await sessions.recover_turn("session", turn.key.turn_id,
                                               expected_generation=row["generation"], principal=principal)
            failed = await replay.result()
            assert failed.status is ExecutionStatus.FAILED
            assert failed.view["conversation_committed"] and failed.view["items"] == []
            assert calls == ["lookup"]
            following = await sessions.start_turn("session", "next", agent="default", principal=principal,
                                             idempotency_key="turn-2")
            assert (await following.wait_final(timeout_seconds=10)).status is ExecutionStatus.SUCCEEDED
            assert calls == ["lookup", "lookup"]
            assert prompts == [["inspect"], ["inspect", "next"]]
            return

    from linktools.ai.persistence._conversation_repositories import (
        SessionRepositoryImpl,
    )

    original = SessionRepositoryImpl.commit_timeline_turn_in_transaction
    injected = False

    async def commit_then_fail(
        self: SessionRepositoryImpl, *args: Any, **kwargs: Any
    ) -> Any:
        nonlocal injected
        result = await original(self, *args, **kwargs)
        if not injected:
            injected = True
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return result

    monkeypatch.setattr(
        SessionRepositoryImpl, "commit_timeline_turn_in_transaction", commit_then_fail
    )
    state, engine = await _state(backend, tmp_path)
    calls: list[str] = []
    try:
        application = _application(calls)
        application.capability(native_checkpoint(), scope="runtime")
        application.capability(TraceCapability(), scope="runtime")
        async with Persistence.open('session-atomic-terminal', storage=state) as persistence_2659, Runtime.open(persistence=persistence_2659, features=model_features(_ToolModels(), application)) as runtime:
            await runtime.agents.get("default").create_session("session")
            session = runtime.agents.get("default").session("session")
            execution = await session.start("inspect", idempotency_key="turn-1")
            result = (await execution.wait(timeout_seconds=10)).result
            assert injected
            assert result.status is ExecutionStatus.FAILED
            assert result.error_code == ErrorCode.STORAGE_INTEGRITY_ERROR.value
            assert calls == ["lookup"]
            history = await session.history()
            assert "assistant" not in _relevant_kinds(history.items)
            assert "tool_result" not in _relevant_kinds(history.items)
            replay = await session.start("inspect", idempotency_key="turn-1")
            assert replay.execution_id == execution.execution_id
            assert (
                (await replay.wait(timeout_seconds=10)).result
            ).status is ExecutionStatus.FAILED
            assert calls == ["lookup"]
            following = (await session.run(
                "next", idempotency_key="turn-2", timeout_seconds=10
            )).result
            assert following.status is ExecutionStatus.SUCCEEDED
            assert calls == ["lookup", "lookup"]
    finally:
        await runtime._state.close()
        if engine is not None:
            await engine.dispose()


@pytest.mark.asyncio
async def test_conversation_preparation_failure_keeps_owner_retryable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = Persistence(PersistenceStorage.in_memory())
    await state.initialize(namespace="conversation-preparation", tenant_id="tenant")
    native = native_checkpoint().store
    state.run_store.bind_step_store(native)
    state.run_store.bind_execution_persistence(
        "run", conversation=True, restart_checkpoints=True, durable_trace=False,
    )
    run = AgentRunRecord(
        "run", agent_conversation_id="conversation", agent_id="default",
        metadata={"history_id": "history"},
    )
    messages = [ModelRequest(parts=[UserPromptPart("inspect")], run_id="run")]
    await seed_native_snapshot(
        native, run_id="run", messages=messages, conversation_id="conversation",
        agent_name="default", metadata=dict(run.metadata),
    )
    await state.run_store.register_agent_run(run)
    await state.run_store.save_checkpoint(AgentRunCheckpoint(
        "run", 1, messages, context_messages=messages, transcript_message_count_before=0,
    ))
    conversation = state.run_store.read_store(PersistenceDomain.CONVERSATION)

    original_snapshot = await native.latest_snapshot(run_id="run")

    async def fail_resolution(*args: Any, **kwargs: Any) -> Any:
        raise AIError(ErrorCode.STORAGE_UNAVAILABLE)

    try:
        with monkeypatch.context() as patch:
            patch.setattr(conversation, "relocate_conversation_checkpoint", fail_resolution)
            with pytest.raises(AIError) as raised:
                await state.run_store.materialize_conversation(agent_run_id="run")
            assert raised.value.code is ErrorCode.STORAGE_UNAVAILABLE
        assert await conversation.latest_checkpoint(agent_run_id="run") is None
        assert await native.latest_snapshot(run_id="run") == original_snapshot
        reader = state.run_store.checkpoint_persistence.reader
        assert (await reader.latest_snapshot(run_id="run")).messages == messages
        await state.run_store.materialize_conversation(agent_run_id="run")
        assert (await conversation.latest_checkpoint(agent_run_id="run")).messages == messages
    finally:
        await state.close()


@pytest.mark.asyncio
async def test_cold_cancel_preserves_unconfirmed_tool_effect_after_process_exit(
    tmp_path: Path,
    unconfirmed_tool_source_copy: Callable[[Path], Awaitable[tuple[Path, Path, Path]]],
) -> None:
    database, effect_log, native_database = await unconfirmed_tool_source_copy(tmp_path)
    effects = effect_log.read_text().splitlines()
    execution_id, = effects
    calls: list[str] = []
    application = _application(calls, effect_policy="non_replay_safe", effect_log=effect_log)
    application.capability(native_checkpoint(database=native_database), scope="runtime")
    application.capability(TraceCapability(), scope="runtime")
    async with Persistence.open(
        "session-tool-crash", storage=PersistenceStorage.sqlite(database),
    ) as state, Runtime.open(
        persistence=state, features=model_features(_ToolModels(), application),
    ) as runtime:
        before, = await state.checkpoint.tools.list_by_execution(
            execution_id, tenant_id=runtime.default_principal.tenant_id,
        )
        assert before.status is ToolOperationStatus.CLAIMED
        execution = await runtime.executions.get(execution_id)
        outcome = await execution.cancel(idempotency_key="cold-cancel")
        assert outcome.cancelled is False
        current = await state.execution.executions.get(
            execution_id, tenant_id=runtime.default_principal.tenant_id,
        )
        assert current.status is ExecutionStatus.CANCELLING
        still_claimed, = await state.checkpoint.tools.list_by_execution(
            execution_id, tenant_id=runtime.default_principal.tenant_id,
        )
        assert still_claimed == before
        with pytest.raises(AIError) as recovering:
            await execution.recover()
        assert recovering.value.code is ErrorCode.TOOL_EFFECT_UNKNOWN
        current = await state.execution.executions.get(
            execution_id, tenant_id=runtime.default_principal.tenant_id,
        )
        assert current.status is ExecutionStatus.RECOVERY_REQUIRED
        operation, = await state.checkpoint.tools.list_by_execution(
            execution_id, tenant_id=runtime.default_principal.tenant_id,
        )
        assert operation.status is ToolOperationStatus.EFFECT_UNKNOWN
        assert (operation.owner, operation.fence, operation.producer_generation) == (
            before.owner, before.fence, before.producer_generation,
        )
        assert calls == []
        assert effect_log.read_text().splitlines() == effects
