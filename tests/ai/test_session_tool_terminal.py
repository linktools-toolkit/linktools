#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Session terminal handoff across tool-using turns."""

import asyncio
import multiprocessing
import os
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any, Literal

import pytest
from linktools.ai.capability import AgentContext, CapabilityGroup
from linktools.ai.core import (
    ExecutionEventType,
    ExecutionStatus,
    JsonValue,
    OperationKind,
    OperationStatus,
    ResourceKind,
    SessionStatus,
    ToolOperationStatus,
)
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.migrate import provision_runtime_database
from linktools.ai.runtime import (
    Runtime,
    RuntimeStorage,
    RuntimeStoragePlan,
    RuntimeStorageRoute,
)
from linktools.ai.runtime._local import LocalExecutionBackend
from linktools.ai.runtime._tool import RuntimeToolOperationBridge
from linktools.ai.runtime.service_api import CancelExecutionRequest
from linktools.ai.runtime.state import RuntimeDomain
from linktools.ai.runtime.state._runtime_commands import RuntimeStateCommands
from linktools.ai.runtime.state._step_archive import StateStepArchive
from linktools.ai.runtime.state._step_contracts import (
    AgentRunCheckpoint,
    AgentRunRecord,
)
from linktools.ai.runtime.state._steps import RuntimeAgentRunStore
from linktools.ai.storage import PayloadPolicy
from pydantic_ai.messages import (
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


class _ToolModelBinding:
    route_id = "default"
    provider = "test"
    model_identity = "test:session-tool"
    vision = False
    contract: dict[str, JsonValue] = {
        "provider": "test",
        "model": "session-tool",
    }

    def materialize(self) -> TestModel:
        return TestModel(
            call_tools=["lookup"],
            custom_output_text="done",
        )


class _ToolModels:
    def capture(self) -> "_ToolModels":
        return self

    def resolve(self, route_id: str) -> _ToolModelBinding:
        if route_id != "default":
            raise AssertionError(route_id)
        return _ToolModelBinding()

    def restore(
        self,
        payload: Mapping[str, JsonValue],
        *,
        route_id: str | None = None,
    ) -> _ToolModelBinding:
        if (
            route_id not in {None, "default"}
            or dict(payload) != _ToolModelBinding.contract
        ):
            raise AIError(ErrorCode.MODEL_CONNECTION_NOT_FOUND)
        return _ToolModelBinding()


async def _state(
    backend: str,
    tmp_path: Path,
) -> tuple[RuntimeStorage, AsyncEngine | None]:
    if backend == "memory":
        return RuntimeStorage.in_memory(), None
    if backend == "sqlite":
        return RuntimeStorage.sqlite(tmp_path / "runtime-sqlite.db"), None
    if backend == "sql":
        engine = create_async_engine(
            URL.create(
                "sqlite+aiosqlite",
                database=str(tmp_path / "runtime-sql.db"),
            )
        )
        await provision_runtime_database(engine)
        return RuntimeStorage.sql(engine), engine
    raise AssertionError(backend)


def _application(
    calls: list[str],
    *,
    effect_policy: Literal["replay_safe", "non_replay_safe"] = "replay_safe",
    effect_log: Path | None = None,
    started: asyncio.Event | None = None,
    release: asyncio.Event | None = None,
) -> CapabilityGroup[None]:
    application: CapabilityGroup[None] = CapabilityGroup("application")

    async def lookup(_ctx: AgentContext[None]) -> str:
        calls.append("lookup")
        if effect_log is not None:
            with effect_log.open("a", encoding="utf-8") as handle:
                handle.write(_ctx.execution_id + "\n")
                handle.flush()
                os.fsync(handle.fileno())
        if started is not None:
            started.set()
        if release is not None:
            await release.wait()
        return "tool-result"

    application.tool(lookup, name="lookup", effect_policy=effect_policy)
    application.agent(
        "default",
        model="default",
        allow_tools=("lookup",),
        allow_skills=(),
        allow_subagents=(),
        tool_retries=0,
        output_retries=0,
    )
    return application


def _split_sqlite_storage(database: Path) -> RuntimeStorage:
    return RuntimeStorage(
        RuntimeStoragePlan(
            conversation=RuntimeStorageRoute.sqlite(
                database.with_name(f"{database.stem}-conversation.db")
            ),
            execution=RuntimeStorageRoute.sqlite(
                database.with_name(f"{database.stem}-execution.db")
            ),
        )
    )


def _relevant_kinds(values: object) -> list[str]:
    return [
        item.item_kind
        for item in values  # type: ignore[union-attr]
        if item.item_kind in {"user", "tool_call", "tool_result", "assistant"}
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "sqlite", "sql"))
async def test_session_tool_turn_commits_terminal_and_history(
    tmp_path: Path,
    backend: str,
) -> None:
    state, engine = await _state(backend, tmp_path)
    calls: list[str] = []
    application = _application(calls)
    try:
        async with Runtime.open(
            "session-tool-terminal",
            models=_ToolModels(),  # type: ignore[arg-type]
            storage=state,
            capabilities=(application,),
        ) as runtime:
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
            result = await execution.wait(timeout_seconds=10)

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
            assert tool_items[0].request_sequence is not None
            assert tool_items[1].request_sequence == tool_items[0].request_sequence
            assert tool_items[0].tool_operation_id is not None
            assert tool_items[1].tool_operation_id == tool_items[0].tool_operation_id
            assert tool_items[0].started_at is not None
            assert tool_items[0].finished_at is not None
            assert tool_items[0].duration_ns is not None
            assert tool_items[0].status == "SUCCEEDED"

            execution_trace = await runtime.history.trace(
                execution.execution_id,
                principal=runtime.default_principal,
                include_content=True,
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
                tool_trace[0].payload["request_sequence"]
                == tool_items[0].request_sequence
            )
            assert (
                tool_trace[1].payload["request_sequence"]
                == tool_items[0].request_sequence
            )
            assert all("purpose" not in item.payload for item in tool_trace)

            retried = await session.start(
                "inspect",
                idempotency_key="turn-1",
            )
            retried_result = await retried.wait(timeout_seconds=10)
            assert retried.execution_id == execution.execution_id
            assert retried_result.status is ExecutionStatus.SUCCEEDED
            assert calls == ["lookup"]

            second = await session.run(
                "continue",
                idempotency_key="turn-2",
                timeout_seconds=10,
            )
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
        await state.close()
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
    if storage_mode == "same_group":
        state = RuntimeStorage.sqlite(tmp_path / "same-group.db")
    else:
        state = RuntimeStorage(
            RuntimeStoragePlan(
                conversation=RuntimeStorageRoute.sqlite(
                    tmp_path / "conversation.db"
                ),
                execution=RuntimeStorageRoute.sqlite(tmp_path / "execution.db"),
            )
        )
    calls: list[str] = []
    tool_started = asyncio.Event()
    release_tool = asyncio.Event()
    application = _application(
        calls,
        started=tool_started,
        release=release_tool,
    )
    try:
        async with Runtime.open(
            "session-admission-cleanup",
            models=_ToolModels(),  # type: ignore[arg-type]
            storage=state,
            capabilities=(application,),
        ) as runtime:
            try:
                await runtime.agents.get("default").create_session("session")
                session = runtime.agents.get("default").session("session")
                owner = None
                conflict_snapshot: list[object] = []
                if rejection == "busy":
                    owner = await session.start(
                        "inspect",
                        idempotency_key="turn-owner",
                    )
                    await asyncio.wait_for(tool_started.wait(), timeout=10)
                else:
                    original_prepare_start = LocalExecutionBackend.prepare_start

                    async def conflict_after_reservation(
                        backend: LocalExecutionBackend,
                        request: object,
                        execution_record: object,
                        identity: object,
                    ) -> object:
                        await state.conversation.sessions.transition_status(
                            "session",
                            tenant_id=runtime.default_principal.tenant_id,
                            expected=frozenset({SessionStatus.OPEN}),
                            next_status=SessionStatus.CLOSING,
                        )
                        conflict_snapshot.append(
                            await state.conversation.sessions.get(
                                "session",
                                tenant_id=runtime.default_principal.tenant_id,
                            )
                        )
                        return await original_prepare_start(
                            backend,
                            request,  # type: ignore[arg-type]
                            execution_record,  # type: ignore[arg-type]
                            identity,  # type: ignore[arg-type]
                        )

                    monkeypatch.setattr(
                        LocalExecutionBackend,
                        "prepare_start",
                        conflict_after_reservation,
                    )

                session_before = None
                if owner is not None:
                    session_before = await state.conversation.sessions.get(
                        "session",
                        tenant_id=runtime.default_principal.tenant_id,
                    )
                    assert session_before is not None
                    assert session_before.active_execution_id == owner.execution_id

                with pytest.raises(AIError) as raised:
                    await session.start("rejected", idempotency_key="turn-rejected")
                assert raised.value.code is expected_code
                if rejection == "conflict":
                    assert len(conflict_snapshot) == 1
                    session_before = conflict_snapshot[0]
                assert session_before is not None

                executions = await state.execution.executions.list_by_session(
                    "session",
                    tenant_id=runtime.default_principal.tenant_id,
                )
                rejected = next(
                    item
                    for item in executions
                    if item.error_code == expected_code.value
                )
                assert rejected.status is ExecutionStatus.FAILED
                assert rejected.started_at is None
                assert rejected.error_code == expected_code.value
                assert (
                    await state.execution.executions.get_result(
                        rejected.execution_id,
                        tenant_id=runtime.default_principal.tenant_id,
                    )
                ) is not None
                identities = await state.execution.idempotency.list_by_resource(
                    ResourceKind.EXECUTION,
                    rejected.execution_id,
                    tenant_id=runtime.default_principal.tenant_id,
                )
                assert len(identities) == 1
                assert identities[0].status.value == "FAILED"

                checkpoint = await state.recovery.checkpoints.get(
                    rejected.execution_id,
                    tenant_id=runtime.default_principal.tenant_id,
                )
                if storage_mode == "same_group":
                    assert checkpoint is None
                else:
                    assert checkpoint is not None
                    assert checkpoint.state.value == "completed"
                    await runtime._execution_service.runtime_backend().abort_start(
                        rejected
                    )
                    assert await state.recovery.checkpoints.get(
                        rejected.execution_id,
                        tenant_id=runtime.default_principal.tenant_id,
                    ) == checkpoint

                session_after = await state.conversation.sessions.get(
                    "session",
                    tenant_id=runtime.default_principal.tenant_id,
                )
                assert session_after is not None
                assert session_after.status is session_before.status
                assert session_after.active_execution_id == session_before.active_execution_id
                assert session_after.continuation == session_before.continuation

                release_tool.set()
                if owner is not None:
                    result = await owner.wait(timeout_seconds=15)
                    assert result.status is ExecutionStatus.SUCCEEDED
            finally:
                release_tool.set()
    finally:
        await state.close()


@pytest.mark.asyncio
async def test_session_start_cancel_before_worker_run_commits_cancelled_terminal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = RuntimeStorage.sqlite(tmp_path / "cancel-start-before-run.db")
    calls: list[str] = []
    worker_entered = asyncio.Event()
    release_worker = asyncio.Event()
    application = _application(calls)
    original_run = LocalExecutionBackend._run

    async def pause_worker(
        backend: LocalExecutionBackend,
        request: Any,
        execution_record: Any,
        resume: Any,
    ) -> None:
        worker_entered.set()
        await release_worker.wait()
        await original_run(backend, request, execution_record, resume)

    monkeypatch.setattr(LocalExecutionBackend, "_run", pause_worker)
    try:
        async with Runtime.open(
            "session-start-cancel-before-run",
            models=_ToolModels(),  # type: ignore[arg-type]
            storage=state,
            capabilities=(application,),
        ) as runtime:
            await runtime.agents.get("default").create_session("session")
            session = runtime.agents.get("default").session("session")
            before = await state.conversation.sessions.get(
                "session",
                tenant_id=runtime.default_principal.tenant_id,
            )
            assert before is not None
            execution = await session.start("inspect", idempotency_key="turn-1")
            await asyncio.wait_for(worker_entered.wait(), timeout=10)

            admitted = await state.conversation.sessions.get(
                "session",
                tenant_id=runtime.default_principal.tenant_id,
            )
            assert admitted is not None
            assert admitted.active_execution_id == execution.execution_id
            cancelled = await runtime._execution_service.cancel(
                execution.execution_id,
                CancelExecutionRequest(
                    runtime.default_principal,
                    "cancel-before-worker-run",
                ),
            )
            assert cancelled.cancelled is True
            record = await state.execution.executions.get(
                execution.execution_id,
                tenant_id=runtime.default_principal.tenant_id,
            )
            assert record is not None
            assert record.status is ExecutionStatus.CANCELLED
            assert record.error_code == ErrorCode.EXECUTION_CANCELLED.value
            result = await state.execution.executions.get_result(
                execution.execution_id,
                tenant_id=runtime.default_principal.tenant_id,
            )
            assert result is not None
            identities = await state.execution.idempotency.list_by_resource(
                ResourceKind.EXECUTION,
                execution.execution_id,
                tenant_id=runtime.default_principal.tenant_id,
            )
            assert len(identities) == 1
            assert identities[0].status.value == "CANCELLED"
            assert identities[0].error_code is None
            assert result.output is None
            assert result.stop_reason.value == "CANCELLED"
            events = await execution.list_events(limit=100)
            assert any(
                item.event_type == ExecutionEventType.EXECUTION_CANCELLED
                for item in events.items
            )
            assert not any(
                item.event_type
                in {
                    ExecutionEventType.EXECUTION_SUCCEEDED,
                    ExecutionEventType.EXECUTION_FAILED,
                }
                for item in events.items
            )
            after = await state.conversation.sessions.get(
                "session",
                tenant_id=runtime.default_principal.tenant_id,
            )
            assert after is not None
            assert after.status is before.status
            assert after.active_execution_id is None
            assert after.continuation == before.continuation
            replayed = await runtime._execution_service.cancel(
                execution.execution_id,
                CancelExecutionRequest(
                    runtime.default_principal,
                    "cancel-before-worker-run",
                ),
            )
            assert replayed.cancelled is True
            assert calls == []
    finally:
        release_worker.set()
        await state.close()


@pytest.mark.asyncio
async def test_cancel_during_unknown_tool_effect_preserves_recovery_and_cancel_intent(
    tmp_path: Path,
) -> None:
    state = RuntimeStorage.sqlite(tmp_path / "cancel-recovery.db")
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
        async with Runtime.open(
            "cancel-unknown-tool-effect",
            models=_ToolModels(),  # type: ignore[arg-type]
            storage=state,
            capabilities=(application,),
        ) as runtime:
            await runtime.agents.get("default").create_session("session")
            session = runtime.agents.get("default").session("session")
            execution = await session.start(
                "inspect",
                idempotency_key="turn-1",
            )
            await asyncio.wait_for(tool_started.wait(), timeout=10)

            first_cancel = await execution.cancel(idempotency_key="cancel-1")
            assert first_cancel.cancelled is False

            current = await state.execution.executions.get(
                execution.execution_id,
                tenant_id=runtime.default_principal.tenant_id,
            )
            assert current is not None
            assert current.status is ExecutionStatus.RECOVERY_REQUIRED
            assert current.error_code == ErrorCode.TOOL_EFFECT_UNKNOWN.value
            assert await state.execution.executions.get_result(
                execution.execution_id,
                tenant_id=runtime.default_principal.tenant_id,
            ) is None

            operations = await state.recovery.tools.list_by_execution(
                execution.execution_id,
                tenant_id=runtime.default_principal.tenant_id,
            )
            assert len(operations) == 1
            assert operations[0].status.value == "EFFECT_UNKNOWN"

            cancel_operations = await state.execution.operations.list_pending(
                ResourceKind.EXECUTION,
                execution.execution_id,
                tenant_id=runtime.default_principal.tenant_id,
                limit=100,
            )
            assert len(cancel_operations) == 1
            assert cancel_operations[0].operation_kind is OperationKind.EXECUTION_CANCEL
            assert cancel_operations[0].status is OperationStatus.PENDING

            session_record = await state.conversation.sessions.get(
                "session",
                tenant_id=runtime.default_principal.tenant_id,
            )
            assert session_record is not None
            assert session_record.active_execution_id == execution.execution_id

            replayed_cancel = await execution.cancel(idempotency_key="cancel-1")
            assert replayed_cancel.cancelled is False
            replayed_operations = await state.execution.operations.list_pending(
                ResourceKind.EXECUTION,
                execution.execution_id,
                tenant_id=runtime.default_principal.tenant_id,
                limit=100,
            )
            assert replayed_operations == cancel_operations

            with pytest.raises(AIError) as raised:
                await execution.wait(timeout_seconds=5)
            assert raised.value.code is ErrorCode.TOOL_EFFECT_UNKNOWN

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
        await state.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", ("unknown_write", "recovery_projection"))
async def test_cancel_at_unknown_effect_boundaries_stays_nonterminal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    boundary: str,
) -> None:
    state = RuntimeStorage.sqlite(tmp_path / f"cancel-boundary-{boundary}.db")
    calls: list[str] = []
    tool_started = asyncio.Event()
    release_tool = asyncio.Event()
    boundary_started = asyncio.Event()
    release_boundary = asyncio.Event()
    application = _application(
        calls,
        effect_policy="non_replay_safe",
        effect_log=tmp_path / f"effects-{boundary}.txt",
        started=tool_started,
        release=release_tool,
    )
    try:
        async with Runtime.open(
            f"cancel-unknown-boundary-{boundary}",
            models=_ToolModels(),  # type: ignore[arg-type]
            storage=state,
            capabilities=(application,),
        ) as runtime:
            if boundary == "unknown_write":
                repository = state.recovery.tools
                original_mark = repository.mark_effect_unknown

                async def pause_unknown_write(*args: Any, **kwargs: Any) -> Any:
                    boundary_started.set()
                    await release_boundary.wait()
                    return await original_mark(*args, **kwargs)

                monkeypatch.setattr(
                    repository,
                    "mark_effect_unknown",
                    pause_unknown_write,
                )
            else:
                original_projection = LocalExecutionBackend._commit_recovery_required

                async def pause_recovery_projection(
                    backend: LocalExecutionBackend,
                    execution_record: Any,
                    error: AIError,
                    effects: Any,
                ) -> Any:
                    if error.code is ErrorCode.TOOL_EFFECT_UNKNOWN:
                        boundary_started.set()
                        await release_boundary.wait()
                    return await original_projection(
                        backend,
                        execution_record,
                        error,
                        effects,
                    )

                monkeypatch.setattr(
                    LocalExecutionBackend,
                    "_commit_recovery_required",
                    pause_recovery_projection,
                )

            await runtime.agents.get("default").create_session("session")
            session = runtime.agents.get("default").session("session")
            execution = await session.start("inspect", idempotency_key="turn-1")
            await asyncio.wait_for(tool_started.wait(), timeout=10)
            cancel_task = asyncio.create_task(
                execution.cancel(idempotency_key="cancel-at-boundary")
            )
            await asyncio.wait_for(boundary_started.wait(), timeout=10)

            current = await state.execution.executions.get(
                execution.execution_id,
                tenant_id=runtime.default_principal.tenant_id,
            )
            assert current is not None
            assert current.status is ExecutionStatus.CANCELLING
            assert await state.execution.executions.get_result(
                execution.execution_id,
                tenant_id=runtime.default_principal.tenant_id,
            ) is None
            tools = await state.recovery.tools.list_by_execution(
                execution.execution_id,
                tenant_id=runtime.default_principal.tenant_id,
            )
            assert len(tools) == 1
            assert tools[0].status is (
                ToolOperationStatus.CLAIMED
                if boundary == "unknown_write"
                else ToolOperationStatus.EFFECT_UNKNOWN
            )
            session_record = await state.conversation.sessions.get(
                "session",
                tenant_id=runtime.default_principal.tenant_id,
            )
            assert session_record is not None
            assert session_record.active_execution_id == execution.execution_id
            cancel_intents = await state.execution.operations.list_pending(
                ResourceKind.EXECUTION,
                execution.execution_id,
                tenant_id=runtime.default_principal.tenant_id,
                limit=100,
            )
            assert len(cancel_intents) == 1
            assert cancel_intents[0].operation_kind is OperationKind.EXECUTION_CANCEL
            assert cancel_intents[0].status is OperationStatus.PENDING

            release_boundary.set()
            cancelled = await asyncio.wait_for(cancel_task, timeout=10)
            assert cancelled.cancelled is False
            recovered = await state.execution.executions.get(
                execution.execution_id,
                tenant_id=runtime.default_principal.tenant_id,
            )
            assert recovered is not None
            assert recovered.status is ExecutionStatus.RECOVERY_REQUIRED
            assert await state.execution.executions.get_result(
                execution.execution_id,
                tenant_id=runtime.default_principal.tenant_id,
            ) is None
            final_session = await state.conversation.sessions.get(
                "session",
                tenant_id=runtime.default_principal.tenant_id,
            )
            assert final_session is not None
            assert final_session.active_execution_id == execution.execution_id
            events = await execution.list_events(limit=100)
            terminal_types = {
                ExecutionEventType.EXECUTION_SUCCEEDED,
                ExecutionEventType.EXECUTION_FAILED,
                ExecutionEventType.EXECUTION_CANCELLED,
            }
            assert not any(item.event_type in terminal_types for item in events.items)
            assert calls == ["lookup"]
            assert (tmp_path / f"effects-{boundary}.txt").read_text().splitlines() == [
                execution.execution_id
            ]
    finally:
        release_tool.set()
        release_boundary.set()
        await state.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("readback", "expected_error"),
    (
        ("claimed", ErrorCode.STORAGE_RECOVERY_REQUIRED),
        ("unavailable", ErrorCode.STORAGE_RECOVERY_REQUIRED),
    ),
)
async def test_unverified_unknown_effect_write_does_not_terminalize_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    readback: str,
    expected_error: ErrorCode,
) -> None:
    state = RuntimeStorage.sqlite(tmp_path / f"unknown-write-{readback}.db")
    calls: list[str] = []
    tool_started = asyncio.Event()
    release_tool = asyncio.Event()
    write_attempted = asyncio.Event()
    application = _application(
        calls,
        effect_policy="non_replay_safe",
        effect_log=tmp_path / f"effects-unknown-write-{readback}.txt",
        started=tool_started,
        release=release_tool,
    )
    close_failure_expected = False
    try:
        async with Runtime.open(
            f"unknown-write-{readback}",
            models=_ToolModels(),  # type: ignore[arg-type]
            storage=state,
            capabilities=(application,),
        ) as runtime:
            repository = state.recovery.tools
            original_get = repository.get_operation

            async def fail_unknown_write(*args: Any, **kwargs: Any) -> Any:
                write_attempted.set()
                raise AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN)

            async def read_unknown_write(
                operation_id: str,
                *,
                tenant_id: str,
            ) -> Any:
                if readback == "unavailable" and write_attempted.is_set():
                    raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)
                return await original_get(operation_id, tenant_id=tenant_id)

            monkeypatch.setattr(repository, "mark_effect_unknown", fail_unknown_write)
            monkeypatch.setattr(repository, "get_operation", read_unknown_write)

            await runtime.agents.get("default").create_session("session")
            session = runtime.agents.get("default").session("session")
            execution = await session.start("inspect", idempotency_key="turn-1")
            await asyncio.wait_for(tool_started.wait(), timeout=10)
            cancelled = await asyncio.wait_for(
                execution.cancel(idempotency_key="cancel-unverified-write"),
                timeout=10,
            )
            assert cancelled.cancelled is False
            assert write_attempted.is_set()

            current = await state.execution.executions.get(
                execution.execution_id,
                tenant_id=runtime.default_principal.tenant_id,
            )
            assert current is not None
            assert current.status is ExecutionStatus.CANCELLING
            assert await state.execution.executions.get_result(
                execution.execution_id,
                tenant_id=runtime.default_principal.tenant_id,
            ) is None
            tools = await state.recovery.tools.list_by_execution(
                execution.execution_id,
                tenant_id=runtime.default_principal.tenant_id,
            )
            assert len(tools) == 1
            assert tools[0].status is ToolOperationStatus.CLAIMED
            session_record = await state.conversation.sessions.get(
                "session",
                tenant_id=runtime.default_principal.tenant_id,
            )
            assert session_record is not None
            assert session_record.active_execution_id == execution.execution_id
            backend = runtime._execution_service.runtime_backend()
            failure = backend.worker_failure(
                execution.execution_id,
                tenant_id=runtime.default_principal.tenant_id,
            )
            assert failure is not None
            assert failure.code is expected_error
            events = await execution.list_events(limit=100)
            terminal_types = {
                ExecutionEventType.EXECUTION_SUCCEEDED,
                ExecutionEventType.EXECUTION_FAILED,
                ExecutionEventType.EXECUTION_CANCELLED,
            }
            assert not any(item.event_type in terminal_types for item in events.items)
            assert calls == ["lookup"]
            assert (
                tmp_path / f"effects-unknown-write-{readback}.txt"
            ).read_text().splitlines() == [execution.execution_id]
            close_failure_expected = True
    except AIError as close_error:
        if not close_failure_expected:
            raise
        assert close_error.code is ErrorCode.STORAGE_RECOVERY_REQUIRED
    finally:
        release_tool.set()
        await state.close()


@pytest.mark.asyncio
async def test_durable_terminal_survives_local_seal_finalization_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = RuntimeStorage.in_memory()
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
        async with Runtime.open(
            "session-tool-terminal-finalize",
            models=_ToolModels(),  # type: ignore[arg-type]
            storage=state,
            capabilities=(application,),
        ) as runtime:
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
            result = await execution.wait(timeout_seconds=10)

            assert result.status is ExecutionStatus.SUCCEEDED
            assert calls == ["lookup"]
            assert (
                watched[-1].event.event_type == ExecutionEventType.EXECUTION_SUCCEEDED
            )
            stored_execution = await state.execution.executions.get(
                execution.execution_id,
                tenant_id=runtime.tenant_id,
            )
            stored_session = await state.conversation.sessions.get(
                "session",
                tenant_id=runtime.tenant_id,
            )
            assert stored_execution is not None and stored_execution.retention_closed
            assert stored_session is not None
            assert stored_session.active_execution_id is None
    finally:
        await state.close()


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
    original = RuntimeStateCommands.commit_terminal_checkpoint
    injected = False

    async def fail_success_once(
        self: RuntimeStateCommands,
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
        RuntimeStateCommands,
        "commit_terminal_checkpoint",
        fail_success_once,
    )
    async with Runtime.open(
        "session-tool-terminal-failure",
        models=_ToolModels(),  # type: ignore[arg-type]
        storage=RuntimeStorage.in_memory(),
        capabilities=(application,),
    ) as runtime:
        await runtime.agents.get("default").create_session("session")
        session = runtime.agents.get("default").session("session")
        execution = await session.start("inspect", idempotency_key="turn-1")
        watched = [
            item
            async for item in execution.watch(include_content=True)
            if item.depth == 0
        ]
        result = await execution.wait(timeout_seconds=10)

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
        same_result = await same.wait(timeout_seconds=10)
        assert same.execution_id == execution.execution_id
        assert same_result.status is ExecutionStatus.FAILED
        assert calls == ["lookup"]

        retry = await session.run(
            "retry",
            idempotency_key="turn-2",
            timeout_seconds=10,
        )
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
        first = RuntimeStorage.sqlite(database)
    else:
        first_engine = create_async_engine(
            URL.create("sqlite+aiosqlite", database=str(database))
        )
        await provision_runtime_database(first_engine)
        first = RuntimeStorage.sql(first_engine)
    await first.initialize(namespace="tool-replay", tenant_id="tenant")
    try:
        first_bridge = RuntimeToolOperationBridge(
            first.recovery.tools,
            first.object_store(RuntimeDomain.RECOVERY),
            namespace="tool-replay",
            tenant_id="tenant",
            execution_id="execution",
            agent_run_id="run-1",
            binding_digest="a" * 64,
            owner="worker-1",
            background_tasks=set(),
            payload_policy=PayloadPolicy(),
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
        second = RuntimeStorage.sqlite(database)
    else:
        second_engine = create_async_engine(
            URL.create("sqlite+aiosqlite", database=str(database))
        )
        second = RuntimeStorage.sql(second_engine)
    await second.initialize(namespace="tool-replay", tenant_id="tenant")
    try:
        replay_bridge = RuntimeToolOperationBridge(
            second.recovery.tools,
            second.object_store(RuntimeDomain.RECOVERY),
            namespace="tool-replay",
            tenant_id="tenant",
            execution_id="execution",
            agent_run_id="run-2",
            recovery_agent_run_id="run-1",
            binding_digest="a" * 64,
            owner="worker-2",
            background_tasks=set(),
            payload_policy=PayloadPolicy(),
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


def _crash_session_process(
    database: str, effect_log: str, backend: str, phase: str
) -> None:
    """Exit without cleanup after a selected durable boundary."""
    original_complete = RuntimeToolOperationBridge.complete
    original_checkpoint = RuntimeAgentRunStore.save_checkpoint
    original_success = LocalExecutionBackend._commit_success
    original_reconcile_handoff = LocalExecutionBackend._reconcile_handoff
    original_commit_reconciled = LocalExecutionBackend._commit_reconciled_terminal
    original_activate = RuntimeStateCommands.commit_agent_attempt_checkpoint
    original_admission = RuntimeStateCommands.commit_tool_admission

    async def admit(self: RuntimeStateCommands, request: Any) -> Any:
        if phase == "effect_unconfirmed":
            request = replace(request, lease_seconds=1)
        return await original_admission(self, request)

    async def activate(self: RuntimeStateCommands, *args: Any, **kwargs: Any) -> Any:
        result = await original_activate(self, *args, **kwargs)
        if phase == "activated":
            os._exit(91)
        return result

    async def complete(
        self: RuntimeToolOperationBridge, decision: Any, result: Any
    ) -> bool:
        if phase == "effect_unconfirmed":
            os._exit(91)
        cancelled = await original_complete(self, decision, result)
        if phase == "tool_completed":
            os._exit(91)
        return cancelled

    async def save_checkpoint(
        self: RuntimeAgentRunStore, checkpoint: AgentRunCheckpoint, **kwargs: Any
    ) -> None:
        await original_checkpoint(self, checkpoint, **kwargs)
        if phase == "request_checkpoint" and not any(
            isinstance(message, ModelResponse) for message in checkpoint.messages
        ):
            os._exit(91)
        if (
            phase in {"tool_checkpoint", "projected_tool_checkpoint"}
            and checkpoint.pending_request_index is not None
        ):
            if phase == "projected_tool_checkpoint":
                await self.flush_execution_projection(
                    checkpoint.agent_run_id, execution_id=kwargs["execution_id"]
                )
            os._exit(91)

    async def commit_success(
        self: LocalExecutionBackend, *args: Any, **kwargs: Any
    ) -> Any:
        if phase in {"before_terminal", "recovered_before_terminal"}:
            os._exit(91)
        result = await original_success(self, *args, **kwargs)
        if phase == "after_terminal":
            os._exit(91)
        return result

    async def reconcile_handoff(
        self: LocalExecutionBackend,
        checkpoint: Any,
    ) -> Any:
        if phase == "handoff_prepared":
            os._exit(91)
        return await original_reconcile_handoff(self, checkpoint)

    async def commit_reconciled(
        self: LocalExecutionBackend,
        checkpoint: Any,
    ) -> Any:
        result = await original_commit_reconciled(self, checkpoint)
        if phase == "terminal_before_handoff_complete":
            os._exit(91)
        return result

    RuntimeToolOperationBridge.complete = complete
    RuntimeStateCommands.commit_agent_attempt_checkpoint = activate
    RuntimeStateCommands.commit_tool_admission = admit
    RuntimeAgentRunStore.save_checkpoint = save_checkpoint
    LocalExecutionBackend._commit_success = commit_success
    LocalExecutionBackend._reconcile_handoff = reconcile_handoff
    LocalExecutionBackend._commit_reconciled_terminal = commit_reconciled

    async def run() -> None:
        if backend == "sqlite":
            state = RuntimeStorage.sqlite(database)
        elif backend == "split_sqlite":
            state = _split_sqlite_storage(Path(database))
        else:
            engine = create_async_engine(
                URL.create("sqlite+aiosqlite", database=database)
            )
            if phase != "recovered_before_terminal":
                await provision_runtime_database(engine)
            state = RuntimeStorage.sql(engine)
        application = _application(
            [], effect_policy="non_replay_safe", effect_log=Path(effect_log)
        )
        async with Runtime.open(
            "session-tool-crash",
            models=_ToolModels(),
            storage=state,
            capabilities=(application,),
        ) as runtime:
            if phase != "recovered_before_terminal":
                await runtime.agents.get("default").create_session("session")
            execution = (
                await runtime.agents.get("default")
                .session("session")
                .start("inspect", idempotency_key="turn-1")
            )
            await execution.wait(timeout_seconds=15)
        raise AssertionError("crash boundary was not reached")

    asyncio.run(run())


async def _exit_at_boundary(
    database: Path, effect_log: Path, backend: str, phase: str
) -> None:
    process = multiprocessing.get_context("spawn").Process(
        target=_crash_session_process,
        args=(str(database), str(effect_log), backend, phase),
    )
    process.start()
    try:
        await asyncio.to_thread(process.join, 25)
        assert process.exitcode == 91, (phase, process.exitcode)
    finally:
        if process.is_alive():
            process.kill()
            await asyncio.to_thread(process.join, 5)
        process.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("sqlite", "sql", "split_sqlite"))
@pytest.mark.parametrize(
    "phase",
    (
        "tool_completed",
        "tool_checkpoint",
        "projected_tool_checkpoint",
        "before_terminal",
        "after_terminal",
    ),
)
async def test_session_tool_turn_recovers_after_process_exit_without_replaying_effect(
    tmp_path: Path,
    backend: str,
    phase: str,
) -> None:
    database = tmp_path / "runtime.db"
    effect_log = tmp_path / "effects.txt"
    await _exit_at_boundary(database, effect_log, backend, phase)
    committed_effects = effect_log.read_text().splitlines()
    assert len(committed_effects) == 1
    execution_id = committed_effects[0]
    engine = None
    if backend == "sqlite":
        state = RuntimeStorage.sqlite(database)
    elif backend == "split_sqlite":
        state = _split_sqlite_storage(database)
    else:
        engine = create_async_engine(
            URL.create("sqlite+aiosqlite", database=str(database))
        )
        state = RuntimeStorage.sql(engine)
    calls: list[str] = []
    application = _application(calls, effect_policy="non_replay_safe", effect_log=effect_log)
    try:
        async with Runtime.open(
            "session-tool-crash",
            models=_ToolModels(),
            storage=state,
            capabilities=(application,),
        ) as runtime:
            session = runtime.agents.get("default").session("session")
            same = await session.start("inspect", idempotency_key="turn-1")
            result = await same.wait(timeout_seconds=15)
            assert same.execution_id == execution_id
            assert result.status is ExecutionStatus.SUCCEEDED, result
            assert calls == []
            assert effect_log.read_text().splitlines() == committed_effects
            watched = [
                item
                async for item in same.watch(include_content=True)
                if item.depth == 0
            ]
            assert (
                watched[-1].event.event_type == ExecutionEventType.EXECUTION_SUCCEEDED
            )
            history = await session.history()
            assert _relevant_kinds(history.items) == [
                "user",
                "tool_call",
                "tool_result",
                "assistant",
            ]
            execution_history = await runtime.history.history(
                execution_id, principal=runtime.default_principal, include_content=True
            )
            execution_history_kinds = _relevant_kinds(execution_history.items)
            session_history_kinds = _relevant_kinds(history.items)
            if backend == "split_sqlite":
                assert set(session_history_kinds) <= set(execution_history_kinds)
            else:
                assert execution_history_kinds == session_history_kinds
            interactions = await same.model_interactions(include_content=True)
            assert len(interactions.items) >= 2, interactions.items
            assert interactions.items[-1].response is not None
            assert "done" in str(interactions.items[-1].response)
            repeated = await session.start("inspect", idempotency_key="turn-1")
            assert repeated.execution_id == execution_id
            assert (
                await repeated.wait(timeout_seconds=15)
            ).status is ExecutionStatus.SUCCEEDED
            following = await session.run(
                "continue", idempotency_key="turn-2", timeout_seconds=15
            )
            assert following.status is ExecutionStatus.SUCCEEDED
            assert effect_log.read_text().splitlines() == committed_effects
            if backend == "split_sqlite":
                repeated_history = await session.history()
                assert _relevant_kinds(repeated_history.items) == [
                    "user",
                    "tool_call",
                    "tool_result",
                    "assistant",
                    "user",
                    "assistant",
                ]
    finally:
        await state.close()
        if engine is not None:
            await engine.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("phase", "expected_status"),
    (
        ("handoff_prepared", ExecutionStatus.STARTED),
        ("terminal_before_handoff_complete", ExecutionStatus.SUCCEEDED),
    ),
)
async def test_split_storage_handoff_reconciles_prepared_checkpoint_after_restart(
    tmp_path: Path,
    phase: str,
    expected_status: ExecutionStatus,
) -> None:
    database = tmp_path / "handoff-restart.db"
    effect_log = tmp_path / "handoff-effects.txt"
    await _exit_at_boundary(database, effect_log, "split_sqlite", phase)
    committed_effects = effect_log.read_text().splitlines()
    assert len(committed_effects) == 1
    execution_id = committed_effects[0]

    inspection = _split_sqlite_storage(database)
    await inspection.initialize(
        namespace="session-tool-crash",
        tenant_id="default",
    )
    current = await inspection.execution.executions.get(
        execution_id,
        tenant_id="default",
    )
    checkpoint = await inspection.recovery.checkpoints.get(
        execution_id,
        tenant_id="default",
    )
    assert current is not None
    assert current.status is expected_status
    assert checkpoint is not None
    assert checkpoint.state.value == "handoff"
    assert checkpoint.handoff_phase.value == "prepared"
    await inspection.close()

    state = _split_sqlite_storage(database)
    calls: list[str] = []
    application = _application(
        calls,
        effect_policy="non_replay_safe",
        effect_log=effect_log,
    )
    try:
        async with Runtime.open(
            "session-tool-crash",
            models=_ToolModels(),  # type: ignore[arg-type]
            storage=state,
            capabilities=(application,),
        ) as runtime:
            session = runtime.agents.get("default").session("session")
            execution = await session.start("inspect", idempotency_key="turn-1")
            result = await execution.wait(timeout_seconds=15)
            assert execution.execution_id == execution_id
            assert result.status is ExecutionStatus.SUCCEEDED
            assert calls == []
            assert effect_log.read_text().splitlines() == committed_effects
            completed = await state.recovery.checkpoints.get(
                execution_id,
                tenant_id=runtime.default_principal.tenant_id,
            )
            assert completed is not None
            assert completed.state.value == "completed"
            assert completed.handoff_phase.value == "completed"
            session_record = await state.conversation.sessions.get(
                "session",
                tenant_id=runtime.default_principal.tenant_id,
            )
            assert session_record is not None
            assert session_record.active_execution_id is None
            history = await session.history()
            assert _relevant_kinds(history.items) == [
                "user",
                "tool_call",
                "tool_result",
                "assistant",
            ]
    finally:
        await state.close()


@pytest.mark.asyncio
async def test_split_storage_success_handoff_wins_cancel_race(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _split_sqlite_storage(tmp_path / "handoff-cancel-race.db")
    calls: list[str] = []
    handoff_started = asyncio.Event()
    release_handoff = asyncio.Event()
    application = _application(calls)
    original_resolve = LocalExecutionBackend._resolve_handoff_conversation

    async def pause_handoff(
        backend: LocalExecutionBackend,
        checkpoint: Any,
        handoff: Any,
    ) -> None:
        handoff_started.set()
        await release_handoff.wait()
        await original_resolve(backend, checkpoint, handoff)

    monkeypatch.setattr(
        LocalExecutionBackend,
        "_resolve_handoff_conversation",
        pause_handoff,
    )
    try:
        async with Runtime.open(
            "split-storage-handoff-cancel-race",
            models=_ToolModels(),  # type: ignore[arg-type]
            storage=state,
            capabilities=(application,),
        ) as runtime:
            assert (
                state.execution.executions.state_store.storage_group
                is state.recovery.checkpoints.state_store.storage_group
            )
            assert (
                state.execution.executions.state_store.storage_group
                is not state.conversation.sessions.state_store.storage_group
            )
            await runtime.agents.get("default").create_session("session")
            session = runtime.agents.get("default").session("session")
            execution = await session.start("inspect", idempotency_key="turn-1")
            await asyncio.wait_for(handoff_started.wait(), timeout=10)

            prepared = await state.recovery.checkpoints.get(
                execution.execution_id,
                tenant_id=runtime.default_principal.tenant_id,
            )
            assert prepared is not None
            assert prepared.handoff_phase.value == "prepared"
            current = await state.execution.executions.get(
                execution.execution_id,
                tenant_id=runtime.default_principal.tenant_id,
            )
            assert current is not None
            assert current.status is ExecutionStatus.FINALIZING

            first_cancel = await execution.cancel(
                idempotency_key="cancel-during-success-handoff"
            )
            assert first_cancel.cancelled is False
            replayed_cancel = await execution.cancel(
                idempotency_key="cancel-during-success-handoff"
            )
            assert replayed_cancel.cancelled is False

            release_handoff.set()
            result = await execution.wait(timeout_seconds=15)
            assert result.status is ExecutionStatus.SUCCEEDED
            final_execution = await state.execution.executions.get(
                execution.execution_id,
                tenant_id=runtime.default_principal.tenant_id,
            )
            assert final_execution is not None
            assert final_execution.status is ExecutionStatus.SUCCEEDED
            completed = await state.recovery.checkpoints.get(
                execution.execution_id,
                tenant_id=runtime.default_principal.tenant_id,
            )
            assert completed is not None
            assert completed.state.value == "completed"
            assert completed.handoff_phase.value == "completed"
            events = await execution.list_events(limit=100)
            terminal_types = [
                item.event_type
                for item in events.items
                if item.event_type
                in {
                    ExecutionEventType.EXECUTION_SUCCEEDED,
                    ExecutionEventType.EXECUTION_FAILED,
                    ExecutionEventType.EXECUTION_CANCELLED,
                }
            ]
            assert terminal_types == [ExecutionEventType.EXECUTION_SUCCEEDED]
            session_record = await state.conversation.sessions.get(
                "session",
                tenant_id=runtime.default_principal.tenant_id,
            )
            assert session_record is not None
            assert session_record.active_execution_id is None
            assert session_record.continuation is not None
            history = await session.history()
            assert _relevant_kinds(history.items) == [
                "user",
                "tool_call",
                "tool_result",
                "assistant",
            ]
            assert calls == ["lookup"]
    finally:
        release_handoff.set()
        await state.close()


@pytest.mark.asyncio
async def test_checkpoint_save_readback_ignores_transient_before_coordinate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = RuntimeStorage.in_memory()
    await state.initialize(namespace="checkpoint-readback", tenant_id="tenant")
    original = StateStepArchive.materialize_checkpoint
    injected = False

    async def commit_then_fail(
        self: StateStepArchive,
        run: AgentRunRecord,
        checkpoint: AgentRunCheckpoint,
        *,
        execution_id: str | None = None,
        **kwargs: Any,
    ) -> None:
        nonlocal injected
        await original(
            self,
            run,
            checkpoint,
            execution_id=execution_id,
            **kwargs,
        )
        if self.runtime_domain is RuntimeDomain.RECOVERY and not injected:
            injected = True
            raise AIError(ErrorCode.STORAGE_UNAVAILABLE)

    monkeypatch.setattr(
        StateStepArchive,
        "materialize_checkpoint",
        commit_then_fail,
    )
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
        assert injected

        recovery = state.run_store.read_store(RuntimeDomain.RECOVERY)
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
async def test_run_checkpoint_relocation_is_idempotent_and_validates_prefix() -> None:
    state = RuntimeStorage.in_memory()
    await state.initialize(namespace="run-checkpoint-relocation", tenant_id="tenant")
    try:
        recovery = state.run_store.read_store(RuntimeDomain.RECOVERY)
        assert isinstance(recovery, StateStepArchive)
        run = AgentRunRecord("run", agent_conversation_id="conversation", agent_id="default")
        await recovery.register_agent_run(run)
        first = AgentRunCheckpoint(
            agent_run_id="run",
            step_index=1,
            messages=[ModelRequest(parts=[UserPromptPart("inspect")])],
            state="active",
            transcript_message_count_before=0,
        )
        final = AgentRunCheckpoint(
            agent_run_id="run",
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

        relocated = await recovery.relocate_run_checkpoint(run, final)
        assert relocated.transcript_message_count_before == len(final.messages)
        await recovery.materialize_checkpoint(run, relocated)
        assert await recovery.transcript_message_count_for_run(run) == before

        bad_run = AgentRunRecord(
            "bad-run",
            agent_conversation_id="conversation",
            agent_id="default",
        )
        await recovery.register_agent_run(bad_run)
        await recovery.materialize_checkpoint(
            bad_run,
            AgentRunCheckpoint(
                agent_run_id="bad-run",
                step_index=1,
                messages=[ModelRequest(parts=[UserPromptPart("stored")])],
                state="active",
                transcript_message_count_before=0,
            ),
        )
        with pytest.raises(AIError) as raised:
            await recovery.relocate_run_checkpoint(
                bad_run,
                AgentRunCheckpoint(
                    agent_run_id="bad-run",
                    step_index=2,
                    messages=[
                        ModelRequest(parts=[UserPromptPart("different")]),
                        ModelResponse(parts=[TextPart("done")]),
                    ],
                    state="complete",
                    transcript_message_count_before=1,
                ),
            )
        assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
    finally:
        await state.close()


@pytest.mark.asyncio
async def test_recovery_to_conversation_rebases_cumulative_tool_checkpoint() -> None:
    state = RuntimeStorage.in_memory()
    await state.initialize(namespace="session-tool-recovery", tenant_id="tenant")
    try:
        recovery = state.run_store.read_store(RuntimeDomain.RECOVERY)
        conversation = state.run_store.read_store(RuntimeDomain.CONVERSATION)
        assert isinstance(recovery, StateStepArchive)
        assert isinstance(conversation, StateStepArchive)

        run = AgentRunRecord(
            "run",
            agent_conversation_id="conversation",
            agent_id="default",
            metadata={"history_id": "history"},
        )
        await recovery.register_agent_run(run)
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
        await recovery.materialize_checkpoint(
            run,
            AgentRunCheckpoint(
                agent_run_id=run.agent_run_id,
                step_index=1,
                messages=first_messages,
                state="active",
                transcript_message_count_before=0,
            ),
        )
        await recovery.materialize_checkpoint(
            run,
            AgentRunCheckpoint(
                agent_run_id=run.agent_run_id,
                step_index=2,
                messages=final_messages,
                state="complete",
                transcript_message_count_before=len(first_messages),
            ),
        )

        await state.run_store.materialize_from_recovery(
            target=RuntimeDomain.CONVERSATION,
            agent_run_id=run.agent_run_id,
        )
        await state.run_store.materialize_from_recovery(
            target=RuntimeDomain.CONVERSATION,
            agent_run_id=run.agent_run_id,
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


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "phase", ("activated", "request_checkpoint", "recovered_before_terminal", "effect_unconfirmed")
)
async def test_recovery_preserves_bootstrap_and_effect_confirmation_boundaries(
    tmp_path: Path, phase: str
) -> None:
    database = tmp_path / "runtime.db"
    effect_log = tmp_path / "effects.txt"
    if phase == "recovered_before_terminal":
        await _exit_at_boundary(database, effect_log, "sqlite", "tool_completed")
    await _exit_at_boundary(database, effect_log, "sqlite", phase)
    if phase == "effect_unconfirmed":
        await asyncio.sleep(1.1)
    calls: list[str] = []
    async with Runtime.open(
        "session-tool-crash",
        models=_ToolModels(),
        storage=RuntimeStorage.sqlite(database),
        capabilities=(
            _application(calls, effect_policy="non_replay_safe", effect_log=effect_log),
        ),
    ) as runtime:
        session = runtime.agents.get("default").session("session")
        if phase == "effect_unconfirmed":
            with pytest.raises(AIError) as raised:
                same = await session.start("inspect", idempotency_key="turn-1")
                await same.wait(timeout_seconds=10)
            assert raised.value.code is ErrorCode.TOOL_EFFECT_UNKNOWN
            assert calls == []
        else:
            same = await session.start("inspect", idempotency_key="turn-1")
            result = await same.wait(timeout_seconds=10)
            assert result.status is ExecutionStatus.SUCCEEDED, result
            assert calls == (["lookup"] if phase in {"activated", "request_checkpoint"} else [])
            history = await session.history()
            assert _relevant_kinds(history.items) == [
                "user",
                "tool_call",
                "tool_result",
                "assistant",
            ]
        assert len(effect_log.read_text().splitlines()) == 1


@pytest.mark.asyncio
async def test_tool_effect_waits_for_durable_response_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = StateStepArchive.materialize_checkpoint
    calls: list[str] = []
    rejected = False

    async def reject_response(
        self: StateStepArchive,
        run: AgentRunRecord,
        checkpoint: AgentRunCheckpoint,
        **kwargs: Any,
    ) -> None:
        nonlocal rejected
        if self.runtime_domain is RuntimeDomain.RECOVERY and any(
            isinstance(message, ModelResponse)
            and any(isinstance(part, ToolCallPart) for part in message.parts)
            for message in checkpoint.messages
        ):
            rejected = True
            raise AIError(ErrorCode.STORAGE_UNAVAILABLE)
        return await original(self, run, checkpoint, **kwargs)

    monkeypatch.setattr(StateStepArchive, "materialize_checkpoint", reject_response)
    with pytest.raises(AIError) as raised:
        async with Runtime.open(
            "pre-effect-checkpoint",
            models=_ToolModels(),
            storage=RuntimeStorage.in_memory(),
            capabilities=(_application(calls),),
        ) as runtime:
            await runtime.agents.get("default").create_session("session")
            execution = (
                await runtime.agents.get("default")
                .session("session")
                .start("inspect", idempotency_key="turn-1")
            )
            await execution.wait(timeout_seconds=10)
    assert raised.value.code is ErrorCode.STORAGE_UNAVAILABLE
    assert rejected
    assert calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "sqlite"))
async def test_terminal_transaction_rollback_keeps_session_cursor_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend: str
) -> None:
    from linktools.ai.runtime.state._conversation_repositories import (
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
        async with Runtime.open(
            "session-atomic-terminal",
            models=_ToolModels(),
            storage=state,
            capabilities=(_application(calls),),
        ) as runtime:
            await runtime.agents.get("default").create_session("session")
            session = runtime.agents.get("default").session("session")
            execution = await session.start("inspect", idempotency_key="turn-1")
            result = await execution.wait(timeout_seconds=10)
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
                await replay.wait(timeout_seconds=10)
            ).status is ExecutionStatus.FAILED
            assert calls == ["lookup"]
            following = await session.run(
                "next", idempotency_key="turn-2", timeout_seconds=10
            )
            assert following.status is ExecutionStatus.SUCCEEDED
            assert calls == ["lookup", "lookup"]
    finally:
        await state.close()
        if engine is not None:
            await engine.dispose()


@pytest.mark.asyncio
async def test_recovery_preparation_failure_releases_its_owned_flight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = RuntimeStorage.in_memory()
    await state.initialize(namespace="recovery-preparation", tenant_id="tenant")
    recovery = state.run_store.read_store(RuntimeDomain.RECOVERY)
    run = AgentRunRecord(
        "run",
        agent_conversation_id="conversation",
        agent_id="default",
        metadata={"history_id": "history"},
    )
    checkpoint = AgentRunCheckpoint(
        agent_run_id="run",
        step_index=1,
        agent_conversation_id=run.agent_conversation_id,
        agent_id=run.agent_id,
        messages=[ModelRequest(parts=[UserPromptPart("inspect")])],
        transcript_message_count_before=0,
    )
    await recovery.materialize_checkpoint(run, checkpoint)

    async def fail_resolution(_interactions: Any) -> Any:
        raise AIError(ErrorCode.STORAGE_UNAVAILABLE)

    monkeypatch.setattr(recovery, "resolve_model_interactions", fail_resolution)
    with pytest.raises(AIError) as raised:
        await state.run_store.materialize_from_recovery(
            target=RuntimeDomain.CONVERSATION, agent_run_id="run"
        )
    assert raised.value.code is ErrorCode.STORAGE_UNAVAILABLE
    await state.close()
