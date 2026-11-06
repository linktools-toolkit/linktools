#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from linktools.ai.persistence import Persistence
"""Shared models, storage and crash boundaries for session tool contracts."""
from ._runtime_test_helpers import model_features

import asyncio
import multiprocessing
import os

import pytest
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any, Literal

from linktools.ai.capability import AgentContext
from linktools.ai.runtime import RuntimeFeatures
from linktools.ai.core import ExecutionEventType, ExecutionStatus, JsonValue, agent_run_id
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.migrate import provision_runtime_database
from linktools.ai.capability import ExecutionCapability, TraceCapability
from pydantic_ai_harness.step_persistence import ContinuableSnapshot, SqliteStepStore
from tests.ai._runtime_test_helpers import native_checkpoint
from linktools.ai.runtime import Runtime
from linktools.ai.persistence import PersistenceStorage, PersistencePlan, PersistenceDomain
from linktools.ai.storage import StorageRoute
from linktools.ai.runtime._local import LocalExecutionBackend
from linktools.ai.runtime._tool import RuntimeToolOperationBridge
from linktools.ai.runtime._step_store import RuntimeStepStore
from linktools.ai.persistence._runtime_commands import StateCommands
from linktools.ai.persistence._step_contracts import AgentRunCheckpoint
from linktools.ai.persistence._step_archive import _decode_step, _encode_step
from linktools.ai.persistence.runs import RuntimeAgentRunStore
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
    application: RuntimeFeatures[None] = RuntimeFeatures("application")
    application.capability(ExecutionCapability(retain_history=True), scope="runtime")

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
        application = _application(
            [], effect_policy="non_replay_safe", effect_log=Path(effect_log)
        )
        application.capability(native_checkpoint(), scope="runtime")
        application.capability(TraceCapability(), scope="runtime")
        async with Persistence.open('session-tool-crash', storage=state) as persistence_222, Runtime.attach(persistence=persistence_222, features=model_features(_ToolModels(), application)) as runtime:
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


async def _assert_tool_turn_recovers_without_replaying_effect(
    tmp_path: Path, backend: str, phase: str, *, trace_damage: str | None = None,
) -> None:
    database = tmp_path / "runtime.db"
    effect_log = tmp_path / "effects.txt"
    await _exit_at_boundary(database, effect_log, backend, phase)
    committed_effects = effect_log.read_text().splitlines()
    assert len(committed_effects) == 1
    execution_id = committed_effects[0]
    engine = None
    if backend == "sqlite":
        state = PersistenceStorage.sqlite(database)
    elif backend == "split_sqlite":
        state = _split_sqlite_storage(database)
    else:
        engine = create_async_engine(
            URL.create("sqlite+aiosqlite", database=str(database))
        )
        state = PersistenceStorage.sql(engine)
    calls: list[str] = []
    application = _application(calls, effect_policy="non_replay_safe", effect_log=effect_log)
    try:
        application.capability(native_checkpoint(), scope="runtime")
        application.capability(TraceCapability(), scope="runtime")
        async with Persistence.open('session-tool-crash', storage=state) as persistence_284, Runtime.attach(persistence=persistence_284, features=model_features(_ToolModels(), application)) as runtime:
            session = runtime.agents.get("default").session("session")
            before_replay = await runtime._state.execution.executions.get(
                execution_id, tenant_id=runtime.default_principal.tenant_id,
            )
            assert before_replay is not None
            same = await session.start("inspect", idempotency_key="turn-1")
            assert same.execution_id == execution_id
            if before_replay.status is ExecutionStatus.STARTED:
                assert await runtime._state.execution.executions.get(
                    execution_id, tenant_id=runtime.default_principal.tenant_id,
                ) == before_replay
                assert not runtime._execution_service.runtime_backend().worker_installed(execution_id)
                await same.recover()
            result = (await same.wait(timeout_seconds=15)).result
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
            if phase == "before_terminal":
                finished = await runtime._state.execution.executions.get(execution_id, tenant_id=runtime.tenant_id)
                assert finished is not None and finished.agent_run_seq == 2
                latest_run_id = agent_run_id(namespace=runtime.namespace, tenant_id=runtime.tenant_id,
                                             execution_id=execution_id, agent_run_seq=2)
                trace = runtime._run_store.read_store(PersistenceDomain.TRACE)
                capture = (await trace.capture_history((latest_run_id,)))[latest_run_id]
                assert capture.run is not None and capture.message_count == capture.model_interaction_count == 0
                assert [event.event_type for event in await trace.list_events(agent_run_id=latest_run_id)] == [
                    "AGENT_RUN_STARTED", "AGENT_RUN_SUCCEEDED",
                ]
                empty_attempt = await runtime.history.history(
                    execution_id, principal=runtime.default_principal,
                    include_content=True, agent_run_seq=2, limit=1,
                )
                assert empty_attempt.items == () and empty_attempt.next_cursor is None
                first_attempt = await runtime.history.history(
                    execution_id, principal=runtime.default_principal,
                    include_content=True, agent_run_seq=1, limit=1,
                )
                assert first_attempt.items and first_attempt.next_cursor is not None
                remaining = await runtime.history.history(
                    execution_id, principal=runtime.default_principal,
                    include_content=True, agent_run_seq=1,
                    cursor=first_attempt.next_cursor, limit=1,
                )
                assert remaining.items
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
                (await repeated.wait(timeout_seconds=15)).result
            ).status is ExecutionStatus.SUCCEEDED
            following = (await session.run(
                "continue", idempotency_key="turn-2", timeout_seconds=15
            )).result
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
            if trace_damage is not None:
                assert phase == "before_terminal"
                trace = runtime._run_store.read_store(PersistenceDomain.TRACE)
                damaged_run_id = agent_run_id(namespace=runtime.namespace, tenant_id=runtime.tenant_id,
                                              execution_id=execution_id,
                                              agent_run_seq=1 if trace_damage == "missing_source" else 2)

                async def damage(transaction):
                    owner_key = trace._agent_run_key(damaged_run_id)
                    if trace_damage in {"missing_source", "missing_latest"}:
                        await transaction.delete_records((owner_key,))
                    else:
                        owner = await transaction.get_record(owner_key)
                        owned_facts = [fact for fact in await transaction.scan_facts()
                                       if fact.owner_key_digest == owner_key]
                        events = sorted((fact for fact in owned_facts
                                         if fact.stream_digest == trace._stream(damaged_run_id, "event")),
                                        key=lambda fact: fact.sequence)
                        assert len(events) == 2
                        terminal = _decode_step(events[-1].data)
                        assert terminal.event_type == "AGENT_RUN_SUCCEEDED"
                        await transaction.guard_record(owner_key, expected_storage_revision=owner.storage_revision)
                        await transaction.delete_fact_streams(owner_key)
                        await transaction.insert_facts(tuple(
                            replace(fact, data=_encode_step(replace(terminal, event_type="AGENT_RUN_INTERRUPTED")))
                            if fact == events[-1] else fact for fact in owned_facts
                        ))

                await trace.state_store.mutate(damage)
                with pytest.raises(AIError) as unavailable:
                    await runtime.history.history(execution_id, principal=runtime.default_principal, include_content=True)
                assert unavailable.value.code in {ErrorCode.EXECUTION_HISTORY_UNAVAILABLE,
                                                   ErrorCode.STORAGE_INTEGRITY_ERROR}
                assert calls == [] and effect_log.read_text().splitlines() == committed_effects
    finally:
        await runtime._state.close()
        if engine is not None:
            await engine.dispose()
