#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from linktools.ai.persistence import Persistence
"""End-to-end Session timeline coverage through the public Runtime API."""
from ._runtime_test_helpers import model_features

from collections.abc import Awaitable, Callable
from functools import partial
from pathlib import Path

import pytest

from linktools.ai.runtime import RuntimeFeatures
from linktools.ai.core import ExecutionStatus
from linktools.ai.capability import ExecutionCapability
from linktools.ai.history import History
from linktools.ai.core import Page
from linktools.ai.execution import SessionTurn
from linktools.ai.runtime import Runtime
from linktools.ai.persistence import PersistenceStorage

from ._runtime_test_helpers import RuntimeUsageModels


@pytest.mark.asyncio
async def test_in_memory_session_run_restores_timeline() -> None:
    application = RuntimeFeatures("application")
    application.capability(ExecutionCapability(retain_history=True), scope="runtime")
    application.agent("default", model="default", allow_tools=())

    async with Persistence.open('default', storage=PersistenceStorage.in_memory()) as persistence_27, Runtime.attach(persistence=persistence_27, features=model_features(RuntimeUsageModels(), application)) as runtime:
        session = await runtime.agents.get("default").create_session("session")
        first = (await session.run("hello", timeout_seconds=10)).result
        second = (await session.run("again", timeout_seconds=10)).result
        assert first.status is ExecutionStatus.SUCCEEDED
        assert second.status is ExecutionStatus.SUCCEEDED

        page = await session.timeline()
        assert [turn.execution_id for turn in page.items] == [
            first.execution_id,
            second.execution_id,
        ]
        assert [turn.status for turn in page.items] == [
            ExecutionStatus.SUCCEEDED,
            ExecutionStatus.SUCCEEDED,
        ]
        assert [turn.user_input["prompt"] for turn in page.items] == [
            {"kind": "text", "text": "hello"},
            {"kind": "text", "text": "again"},
        ]
        assert all(turn.conversation_committed for turn in page.items)
        assert [
            [item.item_kind for item in turn.items] for turn in page.items
        ] == [["assistant"], ["assistant"]]
        assert page.next_cursor is None


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "exclusive_sqlite"))
async def test_session_timeline_fork_reads_match_live_and_reopened_history(
    tmp_path: Path, backend: str,
) -> None:
    application = RuntimeFeatures("application")
    application.capability(ExecutionCapability(retain_history=True), scope="runtime")
    application.agent("default", model="default", allow_tools=())
    storage = (
        PersistenceStorage.in_memory()
        if backend == "memory"
        else PersistenceStorage.from_root(tmp_path, exclusive_writer=True)
    )

    async def check_pages(
        read: Callable[..., Awaitable[Page[SessionTurn]]],
        expected: tuple[SessionTurn, ...],
    ) -> None:
        full = await read()
        assert full.items == expected
        assert full.next_cursor is None
        newest = await read(limit=1)
        assert newest.items == expected[-1:]
        assert newest.next_cursor is not None
        oldest = await read(limit=1, cursor=newest.next_cursor)
        assert oldest.items == expected[:1]
        assert oldest.next_cursor is None

    async with Persistence.open('default', storage=storage) as persistence_86, Runtime.attach(persistence=persistence_86, features=model_features(RuntimeUsageModels(), application)) as runtime:
        principal = runtime.default_principal
        parent = await runtime.agents.get("default").create_session("parent")
        assert (await parent.timeline()).items == ()
        assert (await runtime.history.session_timeline(
            "parent", principal=principal,
        )).items == ()
        parent_turn = (await parent.run("before fork", timeout_seconds=10)).result
        assert parent_turn.status is ExecutionStatus.SUCCEEDED

        child = await parent.fork("child")
        await parent.close()
        child_turn = (await child.run("after fork", timeout_seconds=10)).result
        assert child_turn.status is ExecutionStatus.SUCCEEDED

        page = await child.timeline()
        assert [turn.execution_id for turn in page.items] == [
            parent_turn.execution_id,
            child_turn.execution_id,
        ]
        assert [turn.user_input["prompt"] for turn in page.items] == [
            {"kind": "text", "text": "before fork"},
            {"kind": "text", "text": "after fork"},
        ]
        assert all(turn.conversation_committed for turn in page.items)
        assert all(turn.items for turn in page.items)
        await check_pages(child.timeline, page.items)
        await check_pages(
            partial(runtime.history.session_timeline, "child", principal=principal),
            page.items,
        )

    if backend == "exclusive_sqlite":
        async with Persistence.open('default', storage=PersistenceStorage.from_root(tmp_path, exclusive_writer=True), tenant_id="default", read_only=True) as history_persistence_118, History.open(persistence=history_persistence_118) as history:
            await check_pages(
                partial(history.session_timeline, "child", principal=principal),
                page.items,
            )


@pytest.mark.asyncio
async def test_session_timeline_reads_only_captured_tail_during_append(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from linktools.ai.history import _history as history_module
    from linktools.ai.persistence._sql import _SqlTransaction

    application = RuntimeFeatures("application")
    application.capability(ExecutionCapability(retain_history=True), scope="runtime")
    application.agent("default", model="default", allow_tools=())
    storage = PersistenceStorage.from_root(tmp_path, exclusive_writer=True)

    async with (
        Persistence.open("default", storage=storage) as persistence,
        Runtime.attach(persistence=persistence, features=model_features(RuntimeUsageModels(), application)) as runtime,
    ):
        session = await runtime.agents.get("default").create_session("session")
        principal = runtime.default_principal
        for n in range(6):
            result = (await session.run(f"repeat {n % 2}", timeout_seconds=10)).result
            assert result.status is ExecutionStatus.SUCCEEDED
        full = await runtime.history.session_timeline("session", principal=principal)
        previous_capture = history_module.capture_session_timeline
        get_facts = _SqlTransaction.get_facts
        requested: list[tuple[int, ...]] = []

        async def read_facts(transaction, stream, sequences):
            values = await get_facts(transaction, stream, sequences)
            if values and values[-1].kind == "transcript_chunk":
                requested.append(tuple(sequences))
            return values

        async def capture_then_append(*args, **kwargs):
            captured = await previous_capture(*args, **kwargs)
            assert (await session.run("newest", timeout_seconds=10)).result.status is ExecutionStatus.SUCCEEDED
            return captured

        with monkeypatch.context() as patch:
            patch.setattr(_SqlTransaction, "get_facts", read_facts)
            patch.setattr(history_module, "capture_session_timeline", capture_then_append)
            page = await runtime.history.session_timeline("session", principal=principal, limit=2)
        assert page.items == full.items[-2:]
        assert requested == [(3, 4, 5, 6)]
        older = await runtime.history.session_timeline("session", principal=principal, cursor=page.next_cursor)
        assert older.items == full.items[:-2]


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", ("missing", "malformed", "coordinate_gap", "head"))
async def test_session_timeline_rejects_invalid_captured_tail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str,
) -> None:
    from dataclasses import replace

    from linktools.ai.errors import AIError, ErrorCode
    from linktools.ai.history import _history as history_module
    from linktools.ai.persistence._history import TranscriptRepository
    from linktools.ai.persistence._sql import _SqlTransaction

    application = RuntimeFeatures("application")
    application.capability(ExecutionCapability(retain_history=True), scope="runtime")
    application.agent("default", model="default", allow_tools=())
    storage = PersistenceStorage.from_root(tmp_path, exclusive_writer=True)

    async with (
        Persistence.open("default", storage=storage) as persistence,
        Runtime.attach(persistence=persistence, features=model_features(RuntimeUsageModels(), application)) as runtime,
    ):
        session = await runtime.agents.get("default").create_session("session")
        for _ in range(3):
            assert (await session.run("repeat", timeout_seconds=10)).result.status is ExecutionStatus.SUCCEEDED
        get_facts = _SqlTransaction.get_facts
        decode_chunk = TranscriptRepository.decode_chunk
        capture = history_module.capture_session_timeline

        async def read_facts(transaction, stream, sequences):
            values = await get_facts(transaction, stream, sequences)
            if not values or values[-1].kind != "transcript_chunk":
                return values
            if fault == "missing":
                return values[:-1]
            if fault == "malformed":
                return (*values[:-1], replace(values[-1], data={}))
            return values

        def decode(repository, fact):
            chunk = decode_chunk(repository, fact)
            return replace(chunk, first_message_index=chunk.first_message_index + 1) if (
                fault == "coordinate_gap" and fact.sequence == 3 and fact.kind == "transcript_chunk"
            ) else chunk

        async def capture_bad_head(*args, **kwargs):
            captured = await capture(*args, **kwargs)
            if fault != "head":
                return captured
            return replace(captured, histories={
                identity: (repository, record, replace(head, chunk_count=head.chunk_count + 1))
                for identity, (repository, record, head) in captured.histories.items()
            })

        monkeypatch.setattr(_SqlTransaction, "get_facts", read_facts)
        monkeypatch.setattr(TranscriptRepository, "decode_chunk", decode)
        monkeypatch.setattr(history_module, "capture_session_timeline", capture_bad_head)
        with pytest.raises(AIError) as error:
            await runtime.history.session_timeline("session", principal=runtime.default_principal, limit=1)
        assert error.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
