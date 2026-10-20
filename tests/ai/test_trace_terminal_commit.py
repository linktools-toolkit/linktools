#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Terminal trace joins existing authoritative transactions without delaying safety writes."""

import asyncio
import json
from contextvars import ContextVar
from pathlib import Path

import pytest
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from sqlalchemy import event
from sqlalchemy.engine import Engine
from sqlalchemy.ext.asyncio import create_async_engine

from linktools.ai.capability import ExecutionCapability, CheckpointCapability, TraceCapability, RuntimeFeatures, AgentContext
from linktools.ai.core import CommitPolicy, ExecutionStatus
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.migrate import provision_runtime_database
from linktools.ai.runtime import Runtime, RuntimeStorage
from linktools.ai.runtime.state import RuntimeDomain
from linktools.ai.runtime.state._sql import _SqlTransaction
from linktools.ai.storage._database import SqlStorageContext
from linktools.ai.storage import InMemoryCacheBackend, StorageRoute
from .test_explicit_runtime_features import _FeatureModels, _FeatureFunctionModel


async def _run_observed(path: Path, backend: str, policy: CommitPolicy, monkeypatch, *, checkpoint=False, split=False):
    path.mkdir(parents=True, exist_ok=True)
    engine = None
    if backend == "memory":
        storage = RuntimeStorage.in_memory()
    elif backend == "filesystem":
        storage = RuntimeStorage.filesystem(path / "state")
    elif backend == "sqlite":
        storage = RuntimeStorage.sqlite(path / "state.db")
    else:
        engine = create_async_engine(f"sqlite+aiosqlite:///{path / 'state.db'}")
        await provision_runtime_database(engine)
        storage = RuntimeStorage.sql(engine)
    entered, release = asyncio.Event(), asyncio.Event()

    async def lookup(context: AgentContext[None]) -> str:
        return "value"

    async def model(messages, info):
        if not any(isinstance(part, ToolReturnPart) for message in messages for part in message.parts):
            return ModelResponse(parts=[ToolCallPart("lookup", {}, tool_call_id="lookup-call")])
        entered.set()
        await release.wait()
        return ModelResponse(parts=[TextPart("trace tail")])

    features = RuntimeFeatures("trace-terminal")
    features.tool(lookup, effect_policy="none")
    features.agent("default", model="default", allow_tools=("lookup",))
    for capability in (
        ExecutionCapability(retain_history=True, commit=CommitPolicy.terminal(), coordination=InMemoryCacheBackend(max_bytes=65536)),
        TraceCapability(commit=policy, storage=StorageRoute.sqlite(path / "trace.db") if split else None),
        *((CheckpointCapability(),) if checkpoint else ()),
    ):
        features.capability(capability, scope="runtime")
    callback = ContextVar("transaction_callback", default=None)
    trace_frames, terminal_frames, frames = [], [], []
    during = {"trace_sql_transactions": [], "terminal_sql_transactions": []}
    statements = []
    recording = False

    def observe_statement(connection, cursor, statement, parameters, context, executemany):
        if recording:
            statements.append(statement.split(None, 1)[0].upper())

    event.listen(Engine, "before_cursor_execute", observe_statement)
    original_sql_mutation = SqlStorageContext.run_mutation
    try:
        async with Runtime.open(
            "terminal-trace", storage=storage, features=features,
            models=_FeatureModels(_FeatureFunctionModel(model)),
        ) as runtime:
            session = await runtime.agents.get().create_session("session")
            during["sql_mutations"] = []

            async def observe_sql(context, fn, **kwargs):
                async def wrapped(sql_session):
                    if recording:
                        during["sql_mutations"].append(sql_session.sync_session.get_transaction())
                    return await fn(sql_session)
                return await original_sql_mutation(context, wrapped, **kwargs)

            monkeypatch.setattr(SqlStorageContext, "run_mutation", observe_sql)
            archive = runtime._state.run_store.read_store(RuntimeDomain.TRACE)
            repository = runtime._state.execution.executions
            groups = {archive.state_store.storage_group, repository.state_store.storage_group}
            for group in groups:
                mutate = group.mutate

                async def observe_mutation(stores, fn, *, original=mutate):
                    async def wrapped(transaction):
                        if callback.get() is not None:
                            return await fn(transaction)
                        frame = object()
                        token = callback.set(frame)
                        frames.append(frame)
                        try:
                            return await fn(transaction)
                        finally:
                            callback.reset(token)
                    return await original(stores, wrapped)

                monkeypatch.setattr(group, "mutate", observe_mutation)
            sync_projection = archive._sync_projection_in_transaction
            commit_terminal = repository.commit_terminal_in_transaction

            async def observe_projection(transaction, run, **kwargs):
                trace_frames.append(callback.get())
                if isinstance(transaction, _SqlTransaction):
                    during["trace_sql_transactions"].append(transaction._session.sync_session.get_transaction())
                return await sync_projection(transaction, run, **kwargs)

            async def observe_terminal(transaction, commit, **kwargs):
                terminal_frames.append(callback.get())
                if isinstance(transaction, _SqlTransaction):
                    during["terminal_sql_transactions"].append(transaction._session.sync_session.get_transaction())
                return await commit_terminal(transaction, commit, **kwargs)

            monkeypatch.setattr(archive, "_sync_projection_in_transaction", observe_projection)
            monkeypatch.setattr(repository, "commit_terminal_in_transaction", observe_terminal)
            recording = True
            execution = await session.start("question")
            try:
                await asyncio.wait_for(entered.wait(), 5)
                run_id, = runtime._state.run_store.trace_persistence.offsets
                during["projections"] = len(trace_frames)
                during["transactions"] = len(frames)
                during["events"] = await archive.list_events(agent_run_id=run_id)
                during["execution"] = await repository.get(execution.execution_id, tenant_id=runtime.tenant_id)
                if checkpoint:
                    during["checkpoint"] = await runtime._state.run_store.read_store(RuntimeDomain.CHECKPOINT).latest_checkpoint(agent_run_id=run_id)
            finally:
                release.set()
            assert (await execution.wait(timeout_seconds=5)).result.status is ExecutionStatus.SUCCEEDED
            recording = False
            during["sql_statement_count"] = len(statements)
            during["sql_write_statement_count"] = sum(kind in {"INSERT", "UPDATE", "DELETE"} for kind in statements)
            assert (await execution.model_interactions()).items
            assert await archive.list_events(agent_run_id=run_id)
            assert (await repository.get(execution.execution_id, tenant_id=runtime.tenant_id)).status is ExecutionStatus.SUCCEEDED
            return during, trace_frames, terminal_frames, frames
    finally:
        event.remove(Engine, "before_cursor_execute", observe_statement)
        monkeypatch.setattr(SqlStorageContext, "run_mutation", original_sql_mutation)
        release.set()
        if engine is not None:
            await engine.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "filesystem", "sqlite", "sql"))
async def test_terminal_trace_joins_terminal_transaction_and_reduces_intermediate_writes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend: str) -> None:
    deferred = await _run_observed(tmp_path / "terminal", backend, CommitPolicy.terminal(), monkeypatch)
    immediate = await _run_observed(tmp_path / "sync", backend, CommitPolicy.sync(), monkeypatch)
    during, trace_frames, terminal_frames, frames = deferred
    assert during["execution"] is not None  # Admission safety writes remain required with durable trace.
    assert during["transactions"] > 0
    assert during["projections"] == 0
    assert during["events"] == []
    assert len(terminal_frames) == 1
    assert terminal_frames[0] is not None
    assert trace_frames and all(frame is terminal_frames[0] for frame in trace_frames)
    if backend in {"sqlite", "sql"}:
        terminal_sql, = during["terminal_sql_transactions"]
        assert terminal_sql is not None
        assert during["trace_sql_transactions"]
        assert all(transaction is terminal_sql for transaction in during["trace_sql_transactions"])
        assert sum(transaction is terminal_sql for transaction in during["sql_mutations"]) == 1
        assert len(immediate[0]["sql_mutations"]) > len(during["sql_mutations"])
    if backend in {"sqlite", "sql"}:
        report = {}
        for label, observed in (("terminal", deferred), ("sync", immediate)):
            snapshot, projections, terminals, _ = observed
            report[label] = {
                "write_transactions": len(snapshot["sql_mutations"]),
                "intermediate_projection_transactions": len(set(snapshot["trace_sql_transactions"]) - set(snapshot["terminal_sql_transactions"])),
                "terminal_transactions": len(set(snapshot["terminal_sql_transactions"])),
                "sql_statements_including_reads": snapshot["sql_statement_count"],
                "sql_write_statements": snapshot["sql_write_statement_count"],
            }
        (tmp_path / "transaction-counts.json").write_text(json.dumps(report, indent=2))
    assert immediate[0]["projections"] > during["projections"]
    assert len(immediate[1]) > len(trace_frames)
    assert len(immediate[3]) > len(frames)


@pytest.mark.asyncio
async def test_terminal_trace_does_not_delay_checkpoint_safety_boundary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    during, _, terminal_frames, _ = await _run_observed(
        tmp_path, "sqlite", CommitPolicy.terminal(), monkeypatch, checkpoint=True,
    )
    assert during["checkpoint"] is not None
    assert during["execution"].restart_checkpoints
    assert len(terminal_frames) == 1


@pytest.mark.asyncio
async def test_terminal_trace_on_another_database_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(AIError) as rejected:
        await _run_observed(tmp_path, "sqlite", CommitPolicy.terminal(), monkeypatch, split=True)
    assert rejected.value.code is ErrorCode.RUNTIME_DEPENDENCY_NOT_READY
    assert "trace and execution must share a StateStorageGroup" in str(rejected.value)
