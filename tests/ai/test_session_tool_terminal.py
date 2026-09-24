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
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import RunContext, ToolDefinition
from pydantic_ai.usage import RunUsage
from sqlalchemy.engine import URL
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from linktools.ai.capability import AgentContext, CapabilityGroup
from linktools.ai.core import ExecutionEventType, ExecutionStatus, JsonValue
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.migrate import provision_runtime_database
from linktools.ai.runtime import Runtime, RuntimeState
from linktools.ai.runtime._tool import RuntimeToolOperationBridge
from linktools.ai.runtime._local import LocalExecutionBackend
from linktools.ai.runtime.state import RuntimeDomain
from linktools.ai.runtime.state._runtime_commands import RuntimeStateCommands
from linktools.ai.runtime.state._step_archive import StateStepArchive
from linktools.ai.runtime.state._step_contracts import (
    ContinuableSnapshot,
    RunRecord,
)
from linktools.ai.runtime.state._steps import RuntimeStepStore
from linktools.ai.storage import PayloadPolicy

from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)


class _ToolModelBinding:
    route_id = "default"
    provider = "test"
    model_identity = "test:session-tool"
    vision = False
    model_digest = "a" * 64
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
    def snapshot(self) -> "_ToolModels":
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
) -> tuple[RuntimeState, AsyncEngine | None]:
    if backend == "memory":
        return RuntimeState.in_memory(), None
    if backend == "sqlite":
        return RuntimeState.sqlite(tmp_path / "runtime-sqlite.db"), None
    if backend == "sql":
        engine = create_async_engine(
            URL.create(
                "sqlite+aiosqlite",
                database=str(tmp_path / "runtime-sql.db"),
            )
        )
        await provision_runtime_database(engine)
        return RuntimeState.sql(engine), engine
    raise AssertionError(backend)


def _application(
    calls: list[str],
    *,
    effect: Literal["replay_safe", "non_replay_safe"] = "replay_safe",
    effect_log: Path | None = None,
) -> CapabilityGroup[None]:
    application: CapabilityGroup[None] = CapabilityGroup("application")

    async def lookup(_ctx: AgentContext[None]) -> str:
        calls.append("lookup")
        if effect_log is not None:
            with effect_log.open("a", encoding="utf-8") as handle:
                handle.write(_ctx.execution_id + "\n")
                handle.flush()
                os.fsync(handle.fileno())
        return "tool-result"

    application.tool(lookup, name="lookup", effect=effect)
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
            state=state,
            capabilities=(application,),
        ) as runtime:
            await runtime.agent("default").create_session("session")
            session = runtime.agent("default").session("session")
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
async def test_durable_terminal_survives_local_seal_finalization_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = RuntimeState.in_memory()
    calls: list[str] = []
    application = _application(calls)

    async def fail_finalize(
        _store: RuntimeStepStore,
        _plan: object,
    ) -> None:
        raise RuntimeError("injected terminal seal finalization failure")

    monkeypatch.setattr(
        RuntimeStepStore,
        "finalize_execution_terminal_seal",
        fail_finalize,
    )
    try:
        async with Runtime.open(
            "session-tool-terminal-finalize",
            models=_ToolModels(),  # type: ignore[arg-type]
            state=state,
            capabilities=(application,),
        ) as runtime:
            await runtime.agent("default").create_session("session")
            execution = (
                await runtime.agent("default")
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
        state=RuntimeState.in_memory(),
        capabilities=(application,),
    ) as runtime:
        await runtime.agent("default").create_session("session")
        session = runtime.agent("default").session("session")
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
        first = RuntimeState.sqlite(database)
    else:
        first_engine = create_async_engine(
            URL.create("sqlite+aiosqlite", database=str(database))
        )
        await provision_runtime_database(first_engine)
        first = RuntimeState.sql(first_engine)
    await first.initialize(namespace="tool-replay", tenant_id="tenant")
    try:
        first_bridge = RuntimeToolOperationBridge(
            first.recovery.tools,
            first.object_store(RuntimeDomain.RECOVERY),
            namespace="tool-replay",
            tenant_id="tenant",
            execution_id="execution",
            step_run_id="run-1",
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
        second = RuntimeState.sqlite(database)
    else:
        second_engine = create_async_engine(
            URL.create("sqlite+aiosqlite", database=str(database))
        )
        second = RuntimeState.sql(second_engine)
    await second.initialize(namespace="tool-replay", tenant_id="tenant")
    try:
        replay_bridge = RuntimeToolOperationBridge(
            second.recovery.tools,
            second.object_store(RuntimeDomain.RECOVERY),
            namespace="tool-replay",
            tenant_id="tenant",
            execution_id="execution",
            step_run_id="run-2",
            recovery_step_run_id="run-1",
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
    original_snapshot = RuntimeStepStore.save_snapshot
    original_success = LocalExecutionBackend._commit_success
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

    async def save_snapshot(
        self: RuntimeStepStore, snapshot: ContinuableSnapshot, **kwargs: Any
    ) -> None:
        await original_snapshot(self, snapshot, **kwargs)
        if phase == "request_checkpoint" and not any(
            isinstance(message, ModelResponse) for message in snapshot.messages
        ):
            os._exit(91)
        if (
            phase in {"tool_checkpoint", "projected_tool_checkpoint"}
            and snapshot.pending_request_index is not None
        ):
            if phase == "projected_tool_checkpoint":
                await self.flush_execution_projection(
                    snapshot.run_id, execution_id=kwargs["execution_id"]
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

    RuntimeToolOperationBridge.complete = complete
    RuntimeStateCommands.commit_agent_attempt_checkpoint = activate
    RuntimeStateCommands.commit_tool_admission = admit
    RuntimeStepStore.save_snapshot = save_snapshot
    LocalExecutionBackend._commit_success = commit_success

    async def run() -> None:
        if backend == "sqlite":
            state = RuntimeState.sqlite(database)
        else:
            engine = create_async_engine(
                URL.create("sqlite+aiosqlite", database=database)
            )
            if phase != "recovered_before_terminal":
                await provision_runtime_database(engine)
            state = RuntimeState.sql(engine)
        application = _application(
            [], effect="non_replay_safe", effect_log=Path(effect_log)
        )
        async with Runtime.open(
            "session-tool-crash",
            models=_ToolModels(),
            state=state,
            capabilities=(application,),
        ) as runtime:
            if phase != "recovered_before_terminal":
                await runtime.agent("default").create_session("session")
            execution = (
                await runtime.agent("default")
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
@pytest.mark.parametrize("backend", ("sqlite", "sql"))
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
        state = RuntimeState.sqlite(database)
    else:
        engine = create_async_engine(
            URL.create("sqlite+aiosqlite", database=str(database))
        )
        state = RuntimeState.sql(engine)
    calls: list[str] = []
    application = _application(calls, effect="non_replay_safe", effect_log=effect_log)
    try:
        async with Runtime.open(
            "session-tool-crash",
            models=_ToolModels(),
            state=state,
            capabilities=(application,),
        ) as runtime:
            session = runtime.agent("default").session("session")
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
            assert _relevant_kinds(execution_history.items) == _relevant_kinds(
                history.items
            )
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
    finally:
        await state.close()
        if engine is not None:
            await engine.dispose()


@pytest.mark.asyncio
async def test_snapshot_save_readback_ignores_transient_before_coordinate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = RuntimeState.in_memory()
    await state.initialize(namespace="snapshot-readback", tenant_id="tenant")
    original = StateStepArchive.materialize_snapshot
    injected = False

    async def commit_then_fail(
        self: StateStepArchive,
        run: RunRecord,
        snapshot: ContinuableSnapshot,
        *,
        execution_id: str | None = None,
        **kwargs: Any,
    ) -> None:
        nonlocal injected
        await original(
            self,
            run,
            snapshot,
            execution_id=execution_id,
            **kwargs,
        )
        if self.runtime_domain is RuntimeDomain.RECOVERY and not injected:
            injected = True
            raise AIError(ErrorCode.STORAGE_UNAVAILABLE)

    monkeypatch.setattr(
        StateStepArchive,
        "materialize_snapshot",
        commit_then_fail,
    )
    try:
        run = RunRecord("run", conversation_id="conversation", agent_name="default")
        snapshot = ContinuableSnapshot(
            run_id="run",
            step_index=1,
            messages=[
                ModelRequest(parts=[UserPromptPart("inspect")]),
                ModelResponse(parts=[TextPart("done")]),
            ],
            state="complete",
            transcript_message_count_before=0,
        )
        await state.steps.register_run(run)
        await state.steps.save_snapshot(snapshot)
        assert injected

        recovery = state.steps.read_store(RuntimeDomain.RECOVERY)
        assert isinstance(recovery, StateStepArchive)
        assert await recovery.transcript_message_count_for_run(run) == len(
            snapshot.messages
        )
        stored = await recovery.latest_snapshot(
            run_id=run.run_id,
            include_interrupted=True,
        )
        assert stored is not None
        assert stored.transcript_message_count_before is None
        assert tuple(stored.messages) == tuple(snapshot.messages)
    finally:
        await state.close()


@pytest.mark.asyncio
async def test_run_snapshot_relocation_is_idempotent_and_validates_prefix() -> None:
    state = RuntimeState.in_memory()
    await state.initialize(namespace="run-snapshot-relocation", tenant_id="tenant")
    try:
        recovery = state.steps.read_store(RuntimeDomain.RECOVERY)
        assert isinstance(recovery, StateStepArchive)
        run = RunRecord("run", conversation_id="conversation", agent_name="default")
        await recovery.register_run(run)
        first = ContinuableSnapshot(
            run_id="run",
            step_index=1,
            messages=[ModelRequest(parts=[UserPromptPart("inspect")])],
            state="active",
            transcript_message_count_before=0,
        )
        final = ContinuableSnapshot(
            run_id="run",
            step_index=2,
            messages=[
                *first.messages,
                ModelResponse(parts=[TextPart("done")]),
            ],
            state="complete",
            transcript_message_count_before=1,
        )
        await recovery.materialize_snapshot(run, first)
        await recovery.materialize_snapshot(run, final)
        before = await recovery.transcript_message_count_for_run(run)

        relocated = await recovery.relocate_run_snapshot(run, final)
        assert relocated.transcript_message_count_before == len(final.messages)
        await recovery.materialize_snapshot(run, relocated)
        assert await recovery.transcript_message_count_for_run(run) == before

        bad_run = RunRecord(
            "bad-run",
            conversation_id="conversation",
            agent_name="default",
        )
        await recovery.register_run(bad_run)
        await recovery.materialize_snapshot(
            bad_run,
            ContinuableSnapshot(
                run_id="bad-run",
                step_index=1,
                messages=[ModelRequest(parts=[UserPromptPart("stored")])],
                state="active",
                transcript_message_count_before=0,
            ),
        )
        with pytest.raises(AIError) as raised:
            await recovery.relocate_run_snapshot(
                bad_run,
                ContinuableSnapshot(
                    run_id="bad-run",
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
async def test_recovery_to_conversation_rebases_cumulative_tool_snapshot() -> None:
    state = RuntimeState.in_memory()
    await state.initialize(namespace="session-tool-recovery", tenant_id="tenant")
    try:
        recovery = state.steps.read_store(RuntimeDomain.RECOVERY)
        conversation = state.steps.read_store(RuntimeDomain.CONVERSATION)
        assert isinstance(recovery, StateStepArchive)
        assert isinstance(conversation, StateStepArchive)

        run = RunRecord(
            "run",
            conversation_id="conversation",
            agent_name="default",
            metadata={"history_id": "history"},
        )
        await recovery.register_run(run)
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
        await recovery.materialize_snapshot(
            run,
            ContinuableSnapshot(
                run_id=run.run_id,
                step_index=1,
                messages=first_messages,
                state="active",
                transcript_message_count_before=0,
            ),
        )
        await recovery.materialize_snapshot(
            run,
            ContinuableSnapshot(
                run_id=run.run_id,
                step_index=2,
                messages=final_messages,
                state="complete",
                transcript_message_count_before=len(first_messages),
            ),
        )

        await state.steps.materialize_from_recovery(
            target=RuntimeDomain.CONVERSATION,
            step_run_id=run.run_id,
        )
        await state.steps.materialize_from_recovery(
            target=RuntimeDomain.CONVERSATION,
            step_run_id=run.run_id,
        )

        stored = await conversation.latest_snapshot(run_id=run.run_id)
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
        state=RuntimeState.sqlite(database),
        capabilities=(
            _application(calls, effect="non_replay_safe", effect_log=effect_log),
        ),
    ) as runtime:
        session = runtime.agent("default").session("session")
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
    original = StateStepArchive.materialize_snapshot
    calls: list[str] = []
    rejected = False

    async def reject_response(
        self: StateStepArchive,
        run: RunRecord,
        snapshot: ContinuableSnapshot,
        **kwargs: Any,
    ) -> None:
        nonlocal rejected
        if self.runtime_domain is RuntimeDomain.RECOVERY and any(
            isinstance(message, ModelResponse)
            and any(isinstance(part, ToolCallPart) for part in message.parts)
            for message in snapshot.messages
        ):
            rejected = True
            raise AIError(ErrorCode.STORAGE_UNAVAILABLE)
        return await original(self, run, snapshot, **kwargs)

    monkeypatch.setattr(StateStepArchive, "materialize_snapshot", reject_response)
    with pytest.raises(AIError) as raised:
        async with Runtime.open(
            "pre-effect-checkpoint",
            models=_ToolModels(),
            state=RuntimeState.in_memory(),
            capabilities=(_application(calls),),
        ) as runtime:
            await runtime.agent("default").create_session("session")
            execution = (
                await runtime.agent("default")
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
            state=state,
            capabilities=(_application(calls),),
        ) as runtime:
            await runtime.agent("default").create_session("session")
            session = runtime.agent("default").session("session")
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
    state = RuntimeState.in_memory()
    await state.initialize(namespace="recovery-preparation", tenant_id="tenant")
    recovery = state.steps.read_store(RuntimeDomain.RECOVERY)
    run = RunRecord(
        "run",
        conversation_id="conversation",
        agent_name="default",
        metadata={"history_id": "history"},
    )
    snapshot = ContinuableSnapshot(
        run_id="run",
        step_index=1,
        conversation_id=run.conversation_id,
        agent_name=run.agent_name,
        messages=[ModelRequest(parts=[UserPromptPart("inspect")])],
        transcript_message_count_before=0,
    )
    await recovery.materialize_snapshot(run, snapshot)

    async def fail_resolution(_interactions: Any) -> Any:
        raise AIError(ErrorCode.STORAGE_UNAVAILABLE)

    monkeypatch.setattr(recovery, "resolve_model_interactions", fail_resolution)
    with pytest.raises(AIError) as raised:
        await state.steps.materialize_from_recovery(
            target=RuntimeDomain.CONVERSATION, step_run_id="run"
        )
    assert raised.value.code is ErrorCode.STORAGE_UNAVAILABLE
    await state.close()
