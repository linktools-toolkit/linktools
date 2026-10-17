#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Bounded transcript reads retain their captured range and integrity checks."""

import asyncio
from contextlib import aclosing
from dataclasses import replace
from pathlib import Path

import pytest
from sqlalchemy import event
from sqlalchemy.ext.asyncio import create_async_engine

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.migrate import provision_database
from linktools.ai.runtime import RuntimeStorage
from linktools.ai.runtime.state import RuntimeDomain
from linktools.ai.runtime.state import _history
from linktools.ai.runtime.state._store import FactQuery, active_state_scope

from .test_incremental_projection_boundaries import _storage
from .test_runtime_storage_io_session import _listen_sql, _record_selects
from .test_session_history_projection import _materialize, _session

pytestmark = pytest.mark.asyncio


@pytest.mark.parametrize("reader", ("owner", "session", "raw_context", "spans"))
async def test_bounded_transcript_read_has_a_metadata_io_budget(
    tmp_path: Path, reader: str,
) -> None:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'transcript.db'}")
    await provision_database(engine)
    state = RuntimeStorage.sql(engine)
    await state.initialize(namespace="transcript-read-io", tenant_id="tenant")
    try:
        session = await state.conversation.sessions.create(_session())
        assert session.history_id is not None
        await _materialize(state, "run", ("one", "two"))
        archive = state.run_store.read_store(RuntimeDomain.CONVERSATION)
        repository = archive.transcript_repository
        statements: list[tuple[str, object]] = []
        listener = _listen_sql(engine, statements)
        begins: list[object] = []
        def capture_begin(connection: object) -> None:
            begins.append(connection)
        event.listen(engine.sync_engine, "begin", capture_begin)
        try:
            if reader == "spans":
                spans = await repository.load_message_spans(
                    session.history_id, ((0, 1), (1, 4)),
                )
                messages = tuple(message for span in spans for message in span)
            elif reader == "raw_context":
                context = await repository.load_session_raw_model_context(
                    session.history_id, tenant_id="tenant", message_count=4,
                )
                messages = context.model_messages()
                assert all(value.source is not None for value in context.messages)
            else:
                stream = (
                    repository.iter_message_range(session.history_id, start=0, end=4)
                    if reader == "owner"
                    else repository.iter_session_message_range(
                        session.history_id, tenant_id="tenant", start=0, end=4,
                    )
                )
                messages = tuple([message async for message in stream])
            assert [message.parts[0].content for message in messages] == [
                "one", "answer:one", "two", "answer:two",
            ]
            # A range from zero needs no seek; session descriptor/head share one batch.
            assert len(_record_selects(statements)) <= 1
            # Bounded metadata, seek, and the first fact page share one checkout/transaction.
            assert len(begins) <= 1
            assert sum("FROM ai_state_facts" in statement for statement, _ in statements) <= 1
        finally:
            event.remove(engine.sync_engine, "before_cursor_execute", listener)
            event.remove(engine.sync_engine, "begin", capture_begin)
    finally:
        await state.close()
        await engine.dispose()


@pytest.mark.parametrize("backend", ("memory", "filesystem", "sqlite"))
@pytest.mark.parametrize("reader", ("owner", "session"))
@pytest.mark.parametrize("page_size", (1, 64))
async def test_transcript_range_excludes_appends_after_reading_starts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    backend: str, reader: str, page_size: int,
) -> None:
    monkeypatch.setattr(_history, "_TRANSCRIPT_PAGE_SIZE", page_size)
    state = _storage(backend, tmp_path)
    await state.initialize(namespace="transcript-read-cut", tenant_id="tenant")
    try:
        session = await state.conversation.sessions.create(_session())
        assert session.history_id is not None
        await _materialize(state, "first", ("one", "two"))
        repository = state.run_store.read_store(RuntimeDomain.CONVERSATION).transcript_repository
        stream = (
            repository.iter_message_range(session.history_id, start=1, end=4)
            if reader == "owner"
            else repository.iter_session_message_range(
                session.history_id, tenant_id="tenant", start=1, end=4,
            )
        )
        first = await anext(stream)
        assert active_state_scope() is None
        await _materialize(state, "later", ("three",))
        messages = [first, *[message async for message in stream]]
        assert [message.parts[0].content for message in messages] == [
            "answer:one", "two", "answer:two",
        ]
        fresh = [message async for message in repository.iter_message_range(
            session.history_id, start=4, end=6,
        )]
        assert [message.parts[0].content for message in fresh] == ["three", "answer:three"]
        with pytest.raises(AIError) as error:
            _ = [message async for message in repository.iter_message_range(
                session.history_id, start=0, end=7,
            )]
        assert error.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
    finally:
        await state.close()


@pytest.mark.parametrize("backend", ("memory", "filesystem", "sqlite"))
@pytest.mark.parametrize("corruption", ("missing_tail", "overlap"))
@pytest.mark.parametrize("page_size", (1, 64))
async def test_transcript_range_checks_all_returned_facts_before_finishing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    backend: str, corruption: str, page_size: int,
) -> None:
    state = _storage(backend, tmp_path)
    await state.initialize(namespace="transcript-range-integrity", tenant_id="tenant")
    try:
        session = await state.conversation.sessions.create(_session())
        await _materialize(state, "first", ("one",))
        await _materialize(state, "second", ("two",))
        repository = state.run_store.read_store(RuntimeDomain.CONVERSATION).transcript_repository
        store = repository._store
        assert session.history_id is not None
        stream = repository.transcript_stream(session.history_id)
        facts = await store.read(lambda tx: tx.list_facts(FactQuery(stream)))
        assert len(facts) == 2

        async def corrupt(transaction):
            owner = facts[0].owner_key_digest
            record = await transaction.get_record(owner)
            assert record is not None
            assert await transaction.guard_record(owner, expected_storage_revision=record.storage_revision) is not None
            if corruption == "missing_tail":
                await transaction.delete_fact_streams(owner)
                await transaction.insert_fact(facts[0])
            else:
                await transaction.insert_fact(replace(facts[-1], sequence=facts[-1].sequence + 1))

        await store.mutate(corrupt)
        monkeypatch.setattr(_history, "_TRANSCRIPT_PAGE_SIZE", page_size)
        with pytest.raises(AIError) as error:
            _ = [message async for message in repository.iter_message_range(
                session.history_id, start=0, end=4,
            )]
        assert error.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
    finally:
        await state.close()


@pytest.mark.parametrize("backend", ("memory", "filesystem", "sqlite"))
@pytest.mark.parametrize("reader", ("owner", "session"))
@pytest.mark.parametrize("finish", ("close", "cancel"))
async def test_transcript_payload_and_yield_release_state_scope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    backend: str, reader: str, finish: str,
) -> None:
    state = _storage(backend, tmp_path)
    await state.initialize(namespace="transcript-payload-scope", tenant_id="tenant")
    task = None
    try:
        session = await state.conversation.sessions.create(_session())
        assert session.history_id is not None
        await _materialize(state, "first", ("x" * 17000,))
        archive = state.run_store.read_store(RuntimeDomain.CONVERSATION)
        repository = archive.transcript_repository
        latest = await repository.latest_chunk(session.history_id)
        assert latest is not None and latest.content.payload.kind == "object"
        objects = repository._object_store
        assert objects is not None
        original_open = objects.open
        reading = asyncio.Event()
        release = asyncio.Event()
        closed = asyncio.Event()
        if finish == "close":
            release.set()

        async def checked_open(key: str):
            assert active_state_scope() is None
            try:
                async with aclosing(original_open(key)) as body:
                    reading.set()
                    await release.wait()
                    async for chunk in body:
                        assert active_state_scope() is None
                        yield chunk
            finally:
                assert active_state_scope() is None
                closed.set()

        monkeypatch.setattr(objects, "open", checked_open)
        stream = (
            repository.iter_message_range(session.history_id, start=0, end=2)
            if reader == "owner"
            else repository.iter_session_message_range(
                session.history_id, tenant_id="tenant", start=0, end=2,
            )
        )
        if finish == "close":
            first = await anext(stream)
            assert first.parts[0].content == "x" * 17000
            assert active_state_scope() is None
            await stream.aclose()
        else:
            task = asyncio.create_task(anext(stream))
            await asyncio.wait_for(reading.wait(), timeout=2)
            # A paused object body must not retain a state lock or transaction.
            await asyncio.wait_for(
                state.conversation.sessions.create(_session("during-payload")), timeout=2,
            )
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            await stream.aclose()
        assert closed.is_set()
        assert active_state_scope() is None
        await state.conversation.sessions.create(_session("after-reader"))
        assert await repository.get_head(session.history_id) is not None
    finally:
        if task is not None and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        await state.close()


@pytest.mark.parametrize("reader", ("owner", "session", "spans"))
async def test_empty_transcript_range_validates_head_without_loading_facts(
    tmp_path: Path, reader: str,
) -> None:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'empty-range.db'}")
    await provision_database(engine)
    state = RuntimeStorage.sql(engine)
    await state.initialize(namespace="transcript-empty-range", tenant_id="tenant")
    try:
        session = await state.conversation.sessions.create(_session())
        assert session.history_id is not None
        await _materialize(state, "first", ("one",))
        repository = state.run_store.read_store(RuntimeDomain.CONVERSATION).transcript_repository

        async def read(start: int, end: int):
            if reader == "spans":
                return await repository.load_message_spans(
                    session.history_id, ((start, start), (end, end)),
                )
            stream = (
                repository.iter_message_range(session.history_id, start=start, end=end)
                if reader == "owner"
                else repository.iter_session_message_range(
                    session.history_id, tenant_id="tenant", start=start, end=end,
                )
            )
            return tuple([message async for message in stream])

        statements: list[tuple[str, object]] = []
        listener = _listen_sql(engine, statements)
        try:
            assert await read(0 if reader == "spans" else 2, 2) == (
                ((), ()) if reader == "spans" else ()
            )
            assert not any("FROM ai_state_facts" in statement for statement, _ in statements)
            with pytest.raises(AIError) as error:
                await read(3, 3)
            assert error.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
        finally:
            event.remove(engine.sync_engine, "before_cursor_execute", listener)
    finally:
        await state.close()
        await engine.dispose()


@pytest.mark.parametrize("reader", ("owner", "session"))
async def test_transcript_first_yield_prefetches_only_one_bounded_fact_page(
    monkeypatch: pytest.MonkeyPatch, reader: str,
) -> None:
    from linktools.ai.runtime.state._memory_transaction import _MemoryTransaction

    state = RuntimeStorage.in_memory()
    await state.initialize(namespace="transcript-lazy-page", tenant_id="tenant")
    try:
        session = await state.conversation.sessions.create(_session())
        assert session.history_id is not None
        monkeypatch.setattr(_history, "_TRANSCRIPT_CHUNK_MAX_MESSAGES", 1)
        await _materialize(state, "first", tuple(f"prompt:{index}" for index in range(35)))
        repository = state.run_store.read_store(RuntimeDomain.CONVERSATION).transcript_repository
        stream_digest = repository.transcript_stream(session.history_id)
        original = _MemoryTransaction.list_facts
        rows = []

        async def capture_facts(transaction, query):
            facts = await original(transaction, query)
            if query.stream_digest == stream_digest:
                assert query.limit is not None and query.limit <= 64
                rows.extend(facts)
            return facts

        monkeypatch.setattr(_MemoryTransaction, "list_facts", capture_facts)
        stream = (
            repository.iter_message_range(session.history_id, start=0, end=70)
            if reader == "owner"
            else repository.iter_session_message_range(
                session.history_id, tenant_id="tenant", start=0, end=70,
            )
        )
        assert (await anext(stream)).parts[0].content == "prompt:0"
        assert active_state_scope() is None
        assert 0 < len(rows) <= 64
        before_close = tuple(rows)
        await stream.aclose()
        assert tuple(rows) == before_close
        assert active_state_scope() is None
    finally:
        await state.close()


async def test_session_transcript_store_fallback_does_not_overlap_history_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from linktools.ai.runtime.state._memory import InMemoryStateStore

    state = RuntimeStorage.in_memory()
    await state.initialize(namespace="transcript-separate-store", tenant_id="tenant")
    other = InMemoryStateStore()
    await other.initialize()
    try:
        session = await state.conversation.sessions.create(_session())
        assert session.history_id is not None
        await _materialize(state, "first", ("one",))
        archive = state.run_store.read_store(RuntimeDomain.CONVERSATION)
        records = await archive.state_store.read(lambda tx: tx.scan_records())
        facts = await archive.state_store.read(lambda tx: tx.scan_facts())

        async def copy(transaction):
            await transaction.insert_records(records)
            await transaction.insert_facts(facts)

        await other.mutate(copy)
        repository = _history.TranscriptRepository(
            other, object_store=None, namespace="transcript-separate-store",
            tenant_id="tenant", runtime_domain=RuntimeDomain.CONVERSATION,
            history_repository=state.conversation.histories,
        )
        original = other.read
        read_separate = False

        async def read_outside_history(callback):
            nonlocal read_separate
            assert active_state_scope() is None
            read_separate = True
            return await original(callback)

        monkeypatch.setattr(other, "read", read_outside_history)
        messages = [message async for message in repository.iter_session_message_range(
            session.history_id, tenant_id="tenant", start=0, end=2,
        )]
        assert [message.parts[0].content for message in messages] == ["one", "answer:one"]
        assert read_separate
        assert active_state_scope() is None
    finally:
        await other.close()
        await state.close()


@pytest.mark.parametrize("adapter", ("independent", "subclass"))
@pytest.mark.parametrize("result", ("valid", "missing", "mismatched_head"))
async def test_session_range_retains_custom_history_scalar_contract(
    adapter: str, result: str,
) -> None:
    from linktools.ai.runtime.state._conversation_repositories import ConversationHistoryRepositoryImpl

    state = RuntimeStorage.in_memory()
    namespace = "transcript-custom-history"
    await state.initialize(namespace=namespace, tenant_id="tenant")
    try:
        session = await state.conversation.sessions.create(_session())
        assert session.history_id is not None
        await _materialize(state, "first", ("one",))
        original = state.conversation.histories
        calls: list[str] = []

        class ScalarReads:
            async def get_in_transaction(self, transaction, history_id, *, tenant_id):
                calls.append("get")
                if result == "missing":
                    return None
                return await original.get_in_transaction(transaction, history_id, tenant_id=tenant_id)

            async def local_head_in_transaction(self, transaction, history_id):
                calls.append("head")
                count = await original.local_head_in_transaction(transaction, history_id)
                return count + (result == "mismatched_head")

            async def get_forest_roots_in_transaction(self, transaction, head_id, *, max_roots):
                calls.append("roots")
                return await original.get_forest_roots_in_transaction(transaction, head_id, max_roots=max_roots)

        class IndependentHistory(ScalarReads):
            state_store = original.state_store

        class CustomizedHistory(ScalarReads, ConversationHistoryRepositoryImpl):
            pass

        custom = (
            IndependentHistory() if adapter == "independent"
            else CustomizedHistory(original.state_store, namespace=namespace, tenant_id="tenant")
        )
        repository = _history.TranscriptRepository(
            original.state_store, object_store=None, namespace=namespace,
            tenant_id="tenant", runtime_domain=RuntimeDomain.CONVERSATION,
            history_repository=custom,
        )
        stream = repository.iter_session_message_range(
            session.history_id, tenant_id="tenant", start=0, end=2,
        )
        if result == "valid":
            messages = [message async for message in stream]
            assert [message.parts[0].content for message in messages] == ["one", "answer:one"]
            assert calls == ["get", "head", "roots"]
        else:
            with pytest.raises(AIError) as error:
                _ = [message async for message in stream]
            assert error.value.code is (
                ErrorCode.SESSION_HISTORY_UNAVAILABLE if result == "missing"
                else ErrorCode.STORAGE_INTEGRITY_ERROR
            )
            assert calls == (["get"] if result == "missing" else ["get", "head"])
    finally:
        await state.close()


@pytest.mark.parametrize("corruption", ("descriptor", "head", "owner", "domain", "tenant"))
@pytest.mark.parametrize("end", (0, 2))
async def test_batched_history_range_validates_descriptor_and_head(
    corruption: str, end: int,
) -> None:
    from linktools.ai.runtime.state._contracts import TranscriptOwnerDomain

    state = RuntimeStorage.in_memory()
    await state.initialize(namespace="transcript-batch-integrity", tenant_id="tenant")
    try:
        session = await state.conversation.sessions.create(_session())
        assert session.history_id is not None
        await _materialize(state, "first", ("one",))
        histories = state.conversation.histories
        repository = state.run_store.read_store(RuntimeDomain.CONVERSATION).transcript_repository
        history_id = session.history_id

        async def corrupt(transaction):
            kind = "conversation_history" if corruption == "descriptor" else "transcript_head"
            key = histories._key(kind, history_id)
            stored = await transaction.get_record(key)
            assert stored is not None
            if corruption in {"descriptor", "head"}:
                assert await transaction.delete_record(key, expected_storage_revision=stored.storage_revision)
            else:
                head = repository.decode_head(stored)
                head = (
                    replace(head, owner_id="another-history") if corruption == "owner"
                    else replace(head, owner_domain=TranscriptOwnerDomain.EXECUTION)
                )
                changed = histories._stored("transcript_head", history_id, head)
                assert await transaction.replace_record(
                    replace(stored, data=changed.data, storage_revision=stored.storage_revision + 1),
                    expected_storage_revision=stored.storage_revision,
                )

        if corruption != "tenant":
            await histories.state_store.mutate(corrupt)
        with pytest.raises(AIError) as error:
            _ = [message async for message in repository.iter_session_message_range(
                history_id, tenant_id="other" if corruption == "tenant" else "tenant", start=0, end=end,
            )]
        assert error.value.code is (
            ErrorCode.SESSION_HISTORY_UNAVAILABLE if corruption == "descriptor"
            else ErrorCode.STORAGE_OWNER_MISMATCH if corruption == "tenant"
            else ErrorCode.STORAGE_INTEGRITY_ERROR
        )
    finally:
        await state.close()


@pytest.mark.parametrize("start", (0, 1))
async def test_invalid_seek_hint_falls_back_to_validated_facts(start: int) -> None:
    state = RuntimeStorage.in_memory()
    await state.initialize(namespace="transcript-seek-hint", tenant_id="tenant")
    try:
        session = await state.conversation.sessions.create(_session())
        assert session.history_id is not None
        await _materialize(state, "first", ("one",))
        repository = state.run_store.read_store(RuntimeDomain.CONVERSATION).transcript_repository

        async def corrupt_hint(transaction):
            key = repository._seek_key(session.history_id, 0)
            stored = await transaction.get_record(key)
            assert stored is not None
            assert await transaction.replace_record(
                replace(stored, data={"invalid_hint": True}, storage_revision=stored.storage_revision + 1),
                expected_storage_revision=stored.storage_revision,
            )

        await repository._store.mutate(corrupt_hint)
        messages = [message async for message in repository.iter_session_message_range(
            session.history_id, tenant_id="tenant", start=start, end=2,
        )]
        assert [message.parts[0].content for message in messages] == ["one", "answer:one"][start:]
    finally:
        await state.close()


@pytest.mark.parametrize("mismatch", ("namespace", "tenant"))
async def test_history_batch_does_not_bypass_transcript_scope(mismatch: str) -> None:
    state = RuntimeStorage.in_memory()
    await state.initialize(namespace="transcript-original", tenant_id="tenant")
    try:
        session = await state.conversation.sessions.create(_session())
        assert session.history_id is not None
        histories = state.conversation.histories
        repository = _history.TranscriptRepository(
            histories.state_store, object_store=None,
            namespace="transcript-other" if mismatch == "namespace" else "transcript-original",
            tenant_id="tenant-other" if mismatch == "tenant" else "tenant",
            runtime_domain=RuntimeDomain.CONVERSATION, history_repository=histories,
        )
        with pytest.raises(AIError) as error:
            _ = [message async for message in repository.iter_session_message_range(
                session.history_id, tenant_id="tenant", start=0, end=0,
            )]
        assert error.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
    finally:
        await state.close()


async def test_history_batch_preserves_head_decode_error_priority() -> None:
    state = RuntimeStorage.in_memory()
    await state.initialize(namespace="transcript-error-priority", tenant_id="tenant")
    try:
        session = await state.conversation.sessions.create(_session())
        assert session.history_id is not None
        histories = state.conversation.histories
        repository = state.run_store.read_store(RuntimeDomain.CONVERSATION).transcript_repository

        async def corrupt(transaction):
            stored = await transaction.get_record(histories._key("transcript_head", session.history_id))
            assert stored is not None
            assert await transaction.delete_record(
                stored.key_digest, expected_storage_revision=stored.storage_revision,
            )
            await transaction.insert_record(replace(
                stored, kind="other-kind", data={**stored.data, "v": 999},
            ))

        await histories.state_store.mutate(corrupt)
        with pytest.raises(AIError) as error:
            _ = [message async for message in repository.iter_session_message_range(
                session.history_id, tenant_id="tenant", start=0, end=0,
            )]
        assert error.value.code is ErrorCode.STORAGE_VERSION_UNSUPPORTED
    finally:
        await state.close()
