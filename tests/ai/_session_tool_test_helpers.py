#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from linktools.ai.persistence import Persistence
"""Shared models, storage and crash boundaries for session tool contracts."""
from ._runtime_test_helpers import model_features

import asyncio
import multiprocessing
import os

from pathlib import Path
from typing import Any, Literal

from linktools.ai.capability import AgentContext
from linktools.ai.runtime import RuntimeFeatures
from linktools.ai.core import JsonValue
from linktools.ai.migrate import provision_runtime_database
from linktools.ai.capability import TraceCapability
from pydantic_ai_harness.step_persistence import ContinuableSnapshot, SqliteStepStore
from tests.ai._runtime_test_helpers import native_checkpoint
from linktools.ai.runtime import Runtime
from linktools.ai.persistence import PersistenceStorage, PersistencePlan
from linktools.ai.storage import StorageRoute
from linktools.ai.runtime._local import LocalExecutionBackend
from linktools.ai.runtime._tool import RuntimeToolOperationBridge
from linktools.ai.runtime._step_store import RuntimeStepStore
from linktools.ai.persistence._runtime_commands import StateCommands
from pydantic_ai.messages import ModelResponse, ToolReturnPart
from pydantic_ai.models.test import TestModel
from sqlalchemy.engine import URL
from sqlalchemy.ext.asyncio import create_async_engine


class _ToolModelBinding:
    route_id = "default"
    provider = "test"
    model_identity = "test:session-tool"
    contract: dict[str, JsonValue] = {
        "provider": "test",
        "model": "session-tool",
    }

    def materialize(self) -> TestModel:
        return TestModel(
            call_tools=["lookup"],
            custom_output_text="done",
        )


class _ToolModels(_ToolModelBinding):
    pass


def _application(
    calls: list[str],
    *,
    effect_policy: Literal["replay_safe", "non_replay_safe"] = "replay_safe",
    effect_log: Path | None = None,
    started: asyncio.Event | None = None,
    release: asyncio.Event | None = None,
) -> RuntimeFeatures[None]:
    application: RuntimeFeatures[None] = RuntimeFeatures("application", defaults={"execution": {"retain_history": True}})

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


def _split_sqlite_storage(database: Path) -> PersistenceStorage:
    return PersistenceStorage(
        PersistencePlan(
            conversation=StorageRoute.sqlite(
                database.with_name(f"{database.stem}-conversation.db")
            ),
            execution=StorageRoute.sqlite(
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
    original_step = RuntimeStepStore.commit_step
    original_snapshot = SqliteStepStore.save_snapshot
    original_success = LocalExecutionBackend._commit_success
    original_reconcile_handoff = LocalExecutionBackend._reconcile_handoff
    original_commit_reconciled = LocalExecutionBackend._commit_reconciled_terminal
    original_activate = StateCommands.commit_agent_attempt_checkpoint

    async def activate(self: StateCommands, *args: Any, **kwargs: Any) -> Any:
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

    async def save_snapshot(self: SqliteStepStore, snapshot: ContinuableSnapshot) -> None:
        await original_snapshot(self, snapshot)
        if phase == "request_checkpoint" and snapshot.messages and not any(
            isinstance(message, ModelResponse) for message in snapshot.messages
        ):
            os._exit(91)

    async def commit_step(self: RuntimeStepStore, **kwargs: Any) -> None:
        await original_step(self, **kwargs)
        snapshot = kwargs.get("snapshot")
        if (
            phase in {"tool_checkpoint", "projected_tool_checkpoint"}
            and snapshot is not None
            and snapshot.messages
            and any(isinstance(part, ToolReturnPart) for part in snapshot.messages[-1].parts)
        ):
            if phase == "projected_tool_checkpoint":
                await self._working_set.flush_execution_projection(
                    snapshot.run_id, execution_id=self._tool_operations._execution_id,
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
        *,
        finish: Any,
    ) -> Any:
        async def finish_after_crash(execution: Any) -> Any:
            if phase == "terminal_before_handoff_complete":
                os._exit(91)
            return await finish(execution)

        return await original_commit_reconciled(
            self, checkpoint, finish=finish_after_crash,
        )

    RuntimeToolOperationBridge.complete = complete
    StateCommands.commit_agent_attempt_checkpoint = activate
    RuntimeStepStore.commit_step = commit_step
    SqliteStepStore.save_snapshot = save_snapshot
    LocalExecutionBackend._commit_success = commit_success
    LocalExecutionBackend._reconcile_handoff = reconcile_handoff
    LocalExecutionBackend._commit_reconciled_terminal = commit_reconciled

    async def run() -> None:
        if backend == "sqlite":
            state = PersistenceStorage.sqlite(database)
        elif backend == "split_sqlite":
            state = _split_sqlite_storage(Path(database))
        else:
            engine = create_async_engine(
                URL.create("sqlite+aiosqlite", database=database)
            )
            if phase != "recovered_before_terminal":
                await provision_runtime_database(engine)
            state = PersistenceStorage.sql(engine)
        standalone = phase == "standalone_effect_unconfirmed"
        started = asyncio.Event() if standalone else None
        application = _application(
            [], effect_policy="non_replay_safe", effect_log=Path(effect_log),
            started=started, release=asyncio.Event() if standalone else None,
        )
        application.capability(native_checkpoint(), scope="runtime")
        application.capability(TraceCapability(), scope="runtime")
        async with Persistence.open('session-tool-crash', storage=state) as persistence_222, Runtime.open(persistence=persistence_222, features=model_features(_ToolModels(), application)) as runtime:
            if started is not None:
                await runtime.agents.get("default").start("inspect", idempotency_key="turn-1")
                # The synthetic effect has fsynced, but is held before its
                # result can be committed to the original claim.
                await asyncio.wait_for(started.wait(), 10)
                os._exit(91)
            if phase != "recovered_before_terminal":
                await runtime.agents.get("default").create_session("session")
            execution = (
                await runtime.agents.get("default")
                .session("session")
                .start("inspect", idempotency_key="turn-1")
            )
            if phase == "recovered_before_terminal":
                await execution.recover()
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
