#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Session terminal handoff across tool-using turns."""

from collections.abc import Mapping
from pathlib import Path

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
    fingerprint = "a" * 64
    semantic_payload: dict[str, JsonValue] = {
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
            or dict(payload) != _ToolModelBinding.semantic_payload
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


def _application(calls: list[str]) -> CapabilityGroup[None]:
    application: CapabilityGroup[None] = CapabilityGroup("application")

    async def lookup(_ctx: RunContext[AgentContext[None]]) -> str:
        calls.append("lookup")
        return "tool-result"

    application.tool(lookup, name="lookup", effect="replay_safe")
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

    async def fail_finalize(_plan: object) -> None:
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
            execution = await runtime.agent("default").session("session").start(
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
            assert watched[-1].event.event_type is ExecutionEventType.EXECUTION_SUCCEEDED
    finally:
        await state.close()


@pytest.mark.asyncio
async def test_terminal_commit_error_converges_to_failed_terminal(
    monkeypatch: pytest.MonkeyPatch,
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
                ErrorCode.STORAGE_INTEGRITY_ERROR,
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
        assert result.error_code == ErrorCode.STORAGE_INTEGRITY_ERROR.value
        assert calls == ["lookup"]
        assert watched[-1].event.event_type is ExecutionEventType.EXECUTION_FAILED
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

        stored = await conversation.latest_snapshot(run_id=run.run_id)
        assert stored is not None
        assert stored.state == "complete"
        assert tuple(stored.messages[-len(final_messages):]) == tuple(final_messages)
        assert (
            await conversation.transcript_message_count_for_run(run)
            == len(final_messages)
        )
    finally:
        await state.close()
