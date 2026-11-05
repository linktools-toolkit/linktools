#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Matched two-turn measurement; counts are evidence, not API contracts."""

import asyncio
import json
import os
import traceback
from contextvars import ContextVar
from linktools.ai.storage import SqlStorageContext
from collections import Counter
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart, ToolReturnPart, UserPromptPart
from sqlalchemy import event, text
from sqlalchemy.engine import Engine

from linktools.ai.capability import AgentContext, ExecutionCapability
from pydantic_ai_harness.step_persistence import StepPersistence
from sqlalchemy.ext.asyncio import create_async_engine
from linktools.ai.persistence import SqlStepStore
from linktools.ai.migrate import provision_database
from linktools.ai.core import CommitPolicy, ExecutionStatus
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.persistence import _sql
from linktools.ai.persistence.runs import RuntimeAgentRunStore
from linktools.ai.runtime import Runtime, RuntimeFeatures
from linktools.ai.runtime import _local as local_module
from linktools.ai.storage import InMemoryCacheBackend, SqlObjectStore

from ._runtime_test_helpers import _Models, _UsageFunctionModel, model_features


pytestmark = pytest.mark.skipif(not os.environ.get("TURN_IO_OUTPUT"), reason="Set TURN_IO_OUTPUT to run this measurement probe.")


def _bind_bytes(value: object) -> int:
    if isinstance(value, str):
        return len(value.encode("utf-8"))
    if isinstance(value, (bytes, bytearray, memoryview)):
        return len(value)
    if isinstance(value, dict):
        return sum(_bind_bytes(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return sum(_bind_bytes(item) for item in value)
    return 0


@pytest.mark.asyncio
@pytest.mark.parametrize("checkpointed", (True,))
@pytest.mark.parametrize("coordinated", (False,))
@pytest.mark.parametrize("terminal", (False,))
async def test_two_turn_storage_measurement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, checkpointed: bool,
    coordinated: bool, terminal: bool,
) -> None:
    root = Path(__file__).resolve().parents[2]
    assert Path(local_module.__file__).resolve().is_relative_to(root)
    assert Path(_sql.__file__).resolve().is_relative_to(root)
    payload = "x" * int(os.environ.get("TURN_IO_TOOL_BYTES", "0")) or "done"
    counts: Counter[str] = Counter()
    active = False

    def count(key: str, amount: int = 1) -> None:
        if active:
            counts[key] += amount

    def traced(original, prefix, name):
        async def invoke(*args, **kwargs):
            count(f"{prefix}.{name}")
            result = await original(*args, **kwargs)
            if prefix == "cache":
                if name == "get" and result is not None:
                    count("cache.get_bytes", len(result))
                elif name == "put":
                    count("cache.put_bytes", len(args[2]))
                elif name == "append" and args[4] is not None:
                    count("cache.append_bytes", len(args[4]))
                elif name == "read_batch":
                    count("cache.batch_bytes", sum(len(item[2]) for item in result if item[2] is not None))
            return result
        return invoke

    for cls, prefix, names in (
        (_sql.SqlStateStore, "state", ("read", "mutate", "read_record", "read_records", "refresh_records")),
        (_sql.SqlStateStorageGroup, "group", ("read", "read_many", "get_records", "mutate")),
        (InMemoryCacheBackend, "cache", ("get", "put", "delete", "contains_many", "append", "read_batch", "ack")),
        (InMemoryCacheBackend, "coordination", ("claim_turn", "read_turn", "read_turn_by_request", "read_turn_by_session", "renew_turn", "release_turn", "cancel_turn")),
        (SqlObjectStore, "object", ("stat",)),
    ):
        for name in names:
            monkeypatch.setattr(cls, name, traced(getattr(cls, name), prefix, name))

    original_record = _sql._record_from_row
    def record_from_row(row):
        result = original_record(row)
        count("state.source_records")
        count("state.source_record_json_bytes", len(json.dumps(dict(result.data), ensure_ascii=False, separators=(",", ":")).encode("utf-8")))
        return result
    monkeypatch.setattr(_sql, "_record_from_row", record_from_row)

    original_put, original_open = SqlObjectStore.put, SqlObjectStore.open
    async def put(self, key, chunks, **kwargs):
        count("object.put")
        async def observed() -> AsyncIterator[bytes]:
            async for chunk in chunks:
                count("object.put_bytes", len(chunk))
                yield chunk
        return await original_put(self, key, observed(), **kwargs)
    async def open_object(self, key):
        count("object.open")
        async for chunk in original_open(self, key):
            count("object.open_bytes", len(chunk))
            yield chunk
    monkeypatch.setattr(SqlObjectStore, "put", put)
    monkeypatch.setattr(SqlObjectStore, "open", open_object)

    # The old tree has a three-second owner-observation cache. The fresh-source
    # comparison disables only that memo, preserving its pending-owner guard.
    if os.environ.get("TURN_IO_FRESH_BASELINE") == "1":
        original_require = RuntimeAgentRunStore.require_active_producer
        async def fresh(self, agent_run_id):
            self._producer_observed_until.clear()
            return await original_require(self, agent_run_id)
        monkeypatch.setattr(RuntimeAgentRunStore, "require_active_producer", fresh)

    traces = []
    trace_stack = ContextVar("sql_trace", default=())
    original_statement = SqlStorageContext.run_statement
    async def statement(self, operation):
        token = trace_stack.set(tuple(
            f"{Path(frame.filename).name}:{frame.lineno}:{frame.name}" for frame in traceback.extract_stack()
            if "/linktools-ai/src/" in frame.filename))
        try:
            return await original_statement(self, operation)
        finally:
            trace_stack.reset(token)
    monkeypatch.setattr(SqlStorageContext, "run_statement", statement)
    mutations: dict[int, bool] = {}
    transactions: dict[int, int] = {}
    transaction_seq = 0
    trace_all = os.environ.get("TURN_IO_ALL_SQL") == "1"
    def sql(connection, cursor, statement, parameters, context, many):
        word = statement.lstrip().split()[0].upper()
        if active and (trace_all or word == "SELECT"):
            traces.append({"round": turn, "sql": statement, "stack": trace_stack.get(),
                           "transaction": transactions.get(id(connection)),
                           "bind_text_blob_bytes": _bind_bytes(parameters),
                           "executemany": many})
        count("sql." + word)
        count("sql.bind_text_blob_bytes", _bind_bytes(parameters))
        if word in {"INSERT", "UPDATE", "DELETE"}:
            mutations[id(connection)] = True
    def begin(connection):
        nonlocal transaction_seq
        transaction_seq += 1
        transactions[id(connection)] = transaction_seq
        count("sql.transactions")
        mutations[id(connection)] = False
    def commit(connection):
        count("sql.commits")
        if active and trace_all:
            traces.append({"round": turn, "sql": "COMMIT", "stack": trace_stack.get(),
                           "transaction": transactions.get(id(connection)),
                           "write": bool(mutations.get(id(connection), False))})
        if mutations.pop(id(connection), False):
            count("sql.write_commits")

    calls: Counter[str] = Counter()
    effects: list[str] = []
    model_history: dict[str, list[int]] = {}
    async def work(_context: AgentContext[None]) -> str:
        effects.append("done")
        return payload
    async def model(messages, info):
        question = next(part.content for message in reversed(messages) for part in message.parts
                        if isinstance(part, UserPromptPart) and isinstance(part.content, str) and "question-" in part.content)
        calls[question] += 1
        model_history.setdefault(question, []).append(sum(isinstance(part, ToolReturnPart) for message in messages for part in message.parts))
        if calls[question] <= 3:
            return ModelResponse(parts=[ToolCallPart("work", {}, tool_call_id=f"call-{question}-{calls[question]}")])
        return ModelResponse(parts=[TextPart("answer")])
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'rounds.db'}")
    await provision_database(engine)
    steps = SqlStepStore(engine, namespace="turn-io", tenant_id="default")
    await steps.initialize()
    features = RuntimeFeatures("turn-io")
    features.capability(ExecutionCapability(retain_history=True, commit=CommitPolicy.terminal() if terminal else CommitPolicy.sync()), scope="runtime")
    if checkpointed:
        features.capability(StepPersistence(store=steps, capture_frontier=os.environ.get("TURN_IO_FRONTIER", "1") == "1"), scope="runtime")
    features.tool(work, effect_policy="replay_safe")
    features.agent("default", model="default", allow_tools=("work",))
    cache = InMemoryCacheBackend(max_bytes=16 * 1024 * 1024)
    storage = PersistenceStorage.sql(engine, cache=cache, coordination=cache if coordinated else None)
    rows = []
    async def table_counts():
        async with engine.connect() as connection:
            tables = (await connection.execute(text("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'ai_%'"))).scalars().all()
            return {name: (await connection.execute(text(f'SELECT count(*) FROM "{name}"'))).scalar_one()
                    for name in tables}
    event.listen(Engine, "before_cursor_execute", sql)
    event.listen(Engine, "begin", begin)
    event.listen(Engine, "commit", commit)
    try:
        async with Persistence.open("turn-io", storage=storage) as persistence, Runtime.attach(
            persistence=persistence, features=model_features(_Models(_UsageFunctionModel(model)), features),
        ) as runtime:
            session = await runtime.agents.get().create_session("session")
            inspect_tables = os.environ.get("TURN_IO_TABLE_ROWS") == "1"
            table_rows_before = await table_counts() if inspect_tables else {}
            backend = runtime._execution_service.runtime_backend()
            for turn in range(2):
                counts.clear()
                active = True
                execution = await session.start(f"question-{turn}", idempotency_key=f"round-{turn}")
                result = await execution.wait(timeout_seconds=15)
                assert result.result.status is ExecutionStatus.SUCCEEDED
                assert result.result.output == {"text": "answer"}
                # Include retained lifetime release and already-owned durable work,
                # not only the public result that can precede owner cleanup.
                worker = backend._tasks.get(execution.execution_id)
                if worker is not None:
                    await asyncio.wait_for(asyncio.shield(worker), 10)
                owner = getattr(backend, "_turn_owners", {}).get(execution.execution_id)
                if owner is not None:
                    await asyncio.wait_for(owner.closed.wait(), 10)
                active = False
                rows.append(dict(counts))
            table_rows_after = await table_counts() if inspect_tables else {}
            assert calls == {"question-0": 4, "question-1": 4}
            assert len(effects) == 6
            assert model_history == {"question-0": [0, 1, 2, 3], "question-1": [3, 4, 5, 6]}
            replay = await session.start("question-1", idempotency_key="round-1")
            assert (await replay.wait(timeout_seconds=10)).result.output == {"text": "answer"}
            assert len(effects) == 6 and sum(calls.values()) == 8
            page = await runtime.history.messages("session", principal=runtime.default_principal, limit=100)
            parts = [part for message in page.items for part in message.parts]
            assert [part.content for part in parts if isinstance(part, UserPromptPart)] == ["question-0", "question-1"]
            assert len([part for part in parts if isinstance(part, ToolCallPart)]) == 6
            assert len([part for part in parts if isinstance(part, ToolReturnPart)]) == 6
            assert [part.content for part in parts if isinstance(part, TextPart)] == ["answer", "answer"]
            assert page.next_cursor is None
    finally:
        active = False
        event.remove(Engine, "before_cursor_execute", sql)
        event.remove(Engine, "begin", begin)
        event.remove(Engine, "commit", commit)
        await steps.close()
        await engine.dispose()
    Path(os.environ.get("TURN_IO_TRACE_OUTPUT", ".dev/unified-select-trace.json")).write_text(json.dumps(traces, indent=2))
    output = Path(os.environ.get("TURN_IO_OUTPUT", ".dev/turn-io-measurement.jsonl"))
    with output.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"label": os.environ.get("TURN_IO_LABEL", "current"), "source_root": str(root), "tool_payload_bytes": len(payload), "checkpointed": checkpointed, "capture_frontier": os.environ.get("TURN_IO_FRONTIER", "1") == "1",
                                 "coordinated": coordinated, "terminal": terminal, "rounds": rows,
                                 "table_row_delta": {name: table_rows_after[name] - table_rows_before.get(name, 0) for name in table_rows_after},
                                 "assertions": {"tools": len(effects), "model_calls": sum(calls.values()),
                                                "history_messages": len(page.items), "history_tool_returns": 6,
                                                "outputs": ["answer", "answer"], "replay_no_duplicate_effects": True}}) + "\n")
