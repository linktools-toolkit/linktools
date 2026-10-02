#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared models, storage and crash boundaries for session tool contracts."""

import asyncio
import multiprocessing
import os
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any, Literal

from linktools.ai.capability import AgentContext, CapabilityGroup
from linktools.ai.core import ExecutionEventType, ExecutionStatus, JsonValue
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
from linktools.ai.runtime.state._runtime_commands import RuntimeStateCommands
from linktools.ai.runtime.state._step_contracts import AgentRunCheckpoint
from linktools.ai.runtime.state._steps import RuntimeAgentRunStore
from pydantic_ai.messages import ModelResponse
from pydantic_ai.models.test import TestModel
from sqlalchemy.engine import URL
from sqlalchemy.ext.asyncio import create_async_engine


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
        elif phase == "effect_unconfirmed_live_lease":
            request = replace(request, lease_seconds=5)
        return await original_admission(self, request)

    async def activate(self: RuntimeStateCommands, *args: Any, **kwargs: Any) -> Any:
        result = await original_activate(self, *args, **kwargs)
        if phase == "activated":
            os._exit(91)
        return result

    async def complete(
        self: RuntimeToolOperationBridge, decision: Any, result: Any
    ) -> bool:
        if phase in {"effect_unconfirmed", "effect_unconfirmed_live_lease"}:
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


async def _assert_tool_turn_recovers_without_replaying_effect(
    tmp_path: Path, backend: str, phase: str
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
