#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Exact History reads through the public Runtime boundary."""

import asyncio
import json
import os
from collections import Counter
from contextvars import ContextVar
from dataclasses import replace
from pathlib import Path

import pytest
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from sqlalchemy import event
from sqlalchemy.engine import Engine

from linktools.ai.capability import AgentContext, ExecutionCapability, StepPersistenceCapability, TraceCapability
from linktools.ai.core import CommitPolicy, ExecutionStatus, Principal
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.history import History
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.runtime import Runtime, RuntimeFeatures
from linktools.ai.storage import InMemoryCacheBackend

from ._runtime_test_helpers import _Models, _UsageFunctionModel, model_features
from .test_runtime_storage_io_session import _listen_sql, _record_selects
from .test_transcript_sparse_windows import _history


@pytest.mark.asyncio
@pytest.mark.parametrize("hint", ("present", "missing", "invalid"))
async def test_nonzero_transcript_window_batches_head_and_seek(tmp_path: Path, hint: str) -> None:
    state = Persistence(PersistenceStorage.sqlite(tmp_path / "window.db"))
    try:
        repository = await _history(state, 4)
        if hint != "present":
            async def change_hint(transaction):
                key = repository._seek_key("history", 0)
                current = await transaction.get_record(key)
                assert current is not None
                if hint == "missing":
                    await transaction.delete_record(key)
                else:
                    assert await transaction.replace_record(
                        replace(current, data={}, storage_revision=current.storage_revision + 1),
                        expected_storage_revision=current.storage_revision,
                    )
            await repository._store.mutate(change_hint)
        engine = repository._store.storage_group.context.engine
        statements = []
        listener = _listen_sql(engine, statements)
        try:
            values = [message async for message in repository.iter_message_range("history", start=1, end=3)]
        finally:
            event.remove(engine.sync_engine, "before_cursor_execute", listener)
        assert [message.parts[0].content for message in values] == ["1", "2"]
        metadata = [item for item in _record_selects(statements) if repository._store.store_key in item[1]]
        assert len(metadata) == 1
    finally:
        await state.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("cached", (False, True))
@pytest.mark.parametrize("include_content", (False, True))
async def test_exact_tool_history_keeps_authorized_projection(
    tmp_path: Path, cached: bool, include_content: bool,
) -> None:
    model_calls = 0
    effects = []

    def model(messages, info):
        nonlocal model_calls
        model_calls += 1
        returned = [part for message in messages for part in message.parts if isinstance(part, ToolReturnPart)]
        if not returned:
            return ModelResponse(parts=[ToolCallPart("lookup", {"index": index}, tool_call_id=f"call-{index}")
                                        for index in range(12)])
        assert len(returned) == 12
        return ModelResponse(parts=[TextPart("done")])

    async def lookup(ctx: AgentContext[None], index: int) -> str:
        """Read one deterministic synthetic value."""
        effects.append(index)
        return "item" * 100 + str(index)

    features = RuntimeFeatures("history-read")
    features.agent("default", model="default", allow_tools=("lookup",))
    features.tool(lookup, effect_policy="none")
    features.capability(ExecutionCapability(retain_history=True))
    features.capability(StepPersistenceCapability())
    features.capability(TraceCapability(commit=CommitPolicy.sync()))
    owner: ContextVar[asyncio.Task | None] = ContextVar("history_read_owner", default=None)
    counts = Counter()
    measurements = []
    output = os.environ.get("HISTORY_IO_OUTPUT")

    def active():
        return owner.get() is not None

    def sql(connection, cursor, statement, parameters, context, executemany):
        if active():
            counts[statement.split()[0].upper()] += 1

    def commit(connection):
        if active():
            counts["commits"] += 1

    def rollback(connection):
        if active():
            counts["rollbacks"] += 1

    def storage():
        return PersistenceStorage.from_root(tmp_path / "runtime", cache=(
            InMemoryCacheBackend(max_bytes=16 * 1024 * 1024) if cached else None))

    async def exact(reader, execution_id, principal, phase):
        counts.clear()
        token = owner.set(asyncio.current_task())
        try:
            page = await reader.history(
                execution_id, principal=principal, tool_call_id="call-11",
                agent_run_seq=1, model_request_seq=1, include_content=include_content, limit=2,
            )
        finally:
            owner.reset(token)
        measurements.append({"phase": phase, "counts": dict(counts)})
        return page

    if output:
        event.listen(Engine, "before_cursor_execute", sql)
        event.listen(Engine, "commit", commit)
        event.listen(Engine, "rollback", rollback)
    try:
        async with Runtime.open("history-read", features=model_features(_Models(_UsageFunctionModel(model)), features),
                                storage=storage()) as runtime:
            execution = await runtime.agents.get().start("read synthetic items")
            assert (await execution.wait(timeout_seconds=15)).result.status is ExecutionStatus.SUCCEEDED
            principal = runtime.default_principal
            full = await runtime.executions.history(execution.execution_id, principal=principal,
                                                    include_content=include_content, limit=100)
            expected = tuple(item for item in full.items if item.tool_call_id == "call-11")
            selected = await exact(runtime.executions, execution.execution_id, principal, "live-warm")
            assert selected.items == expected
            assert len(selected.items) == 2 and selected.next_cursor is None
            assert model_calls == 2 and sorted(effects) == list(range(12))
        async with Persistence.open("history-read", storage=storage(), read_only=True) as persistence, History.open(
            persistence=persistence,
        ) as history:
            cold = await exact(history, execution.execution_id, principal, "reopened-cold")
            assert cold.items == expected and cold.next_cursor is None
            warm = await exact(history, execution.execution_id, principal, "reopened-warm")
            assert warm.items == expected and warm.next_cursor is None
            items, cursor = [], None
            while True:
                page = await history.history(execution.execution_id, principal=principal, tool_call_id="call-11",
                    agent_run_seq=1, model_request_seq=1, include_content=include_content, limit=1, cursor=cursor)
                items.extend(page.items)
                cursor = page.next_cursor
                if cursor is None:
                    break
            assert tuple(items) == expected
            assert not (await history.history(execution.execution_id, principal=principal,
                                              tool_call_id="missing", limit=2)).items
            for identity, actor in (("missing-execution", principal),
                                    (execution.execution_id, Principal(principal.principal_id, "other-tenant"))):
                with pytest.raises(AIError) as error:
                    await history.history(identity, principal=actor, tool_call_id="call-11", limit=2)
                assert error.value.code is ErrorCode.AUTHORIZATION_DENIED
    finally:
        if output:
            event.remove(Engine, "before_cursor_execute", sql)
            event.remove(Engine, "commit", commit)
            event.remove(Engine, "rollback", rollback)
    if output:
        with Path(output).open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"label": os.environ.get("HISTORY_IO_LABEL", "current"), "cached": cached,
                                     "include_content": include_content, "measurements": measurements}) + "\n")
