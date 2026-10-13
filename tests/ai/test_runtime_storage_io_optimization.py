#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Focused I/O invariants for Runtime storage optimization."""

import hashlib
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pytest
from linktools.ai.core import ApprovalStatus
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.migrate import provision_database
from linktools.ai.runtime._harness import HarnessPlanStoreAdapter
from linktools.ai.runtime._plan import RuntimePlanStore
from linktools.ai.runtime import RuntimeStorage
from linktools.ai.runtime.state import RuntimeDomain
from linktools.ai.runtime.state._codec import (
    _encode_persisted_domain,
    encode_envelope,
)
from linktools.ai.runtime.state._contracts import (
    ApprovalRecord,
    ContextProjection,
    LoadedModelContext,
)
from linktools.ai.runtime.state._filesystem import FilesystemStateStore
from linktools.ai.runtime.state._filesystem_transaction import _FilesystemTransaction
from linktools.ai.runtime.state._history import TranscriptRepository
from linktools.ai.runtime.state._recovery_repositories import (
    RecoveryApprovalRepositoryImpl,
)
from linktools.ai.runtime.state._store import FactQuery, StoredFact, StoredRecord
from linktools.ai.runtime.state._sql import SqlStateStore
from linktools.ai.storage import (
    FilesystemObjectStore,
    SqlObjectStore,
    StoredPayload,
)
from linktools.ai.storage import _object_filesystem as object_module
from pydantic_ai_harness.planning import PlanItem as HarnessPlanItem
from sqlalchemy import event
from sqlalchemy.ext.asyncio import create_async_engine

pytestmark = pytest.mark.asyncio


async def _chunks(value: bytes) -> AsyncIterator[bytes]:
    yield value


def _filesystem_fact_record(owner: bytes) -> StoredRecord:
    return StoredRecord(
        owner,
        None,
        None,
        "owner",
        "owner",
        None,
        0,
        None,
        0,
        None,
        {},
    )


def _filesystem_fact(
    stream: bytes,
    sequence: int,
    owner: bytes,
    subject: bytes | None,
) -> StoredFact:
    return StoredFact(
        stream,
        sequence,
        owner,
        "test_fact",
        subject,
        None,
        {"sequence": sequence},
    )


async def test_filesystem_fact_batch_limits_stream_and_subject_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "state"
    owner = b"o" * 32
    stream = b"s" * 32
    subject_a = b"a" * 32
    subject_b = b"b" * 32
    store = FilesystemStateStore(
        root,
        namespace="fact-batch",
        tenant_id="tenant",
        runtime_domain="execution",
    )
    await store.initialize()
    try:
        async def seed(transaction: _FilesystemTransaction) -> None:
            await transaction.insert_record(_filesystem_fact_record(owner))
            await transaction.insert_facts(
                (
                    _filesystem_fact(stream, 1, owner, subject_a),
                    _filesystem_fact(stream, 2, owner, subject_b),
                )
            )

        await store.mutate(seed)
    finally:
        await store.close()

    reopened = FilesystemStateStore(
        root,
        namespace="fact-batch",
        tenant_id="tenant",
        runtime_domain="execution",
    )
    await reopened.initialize()
    try:
        staged: list[str] = []
        subject_loads = 0
        appended = (
            _filesystem_fact(stream, 3, owner, subject_a),
            _filesystem_fact(stream, 4, owner, subject_a),
            _filesystem_fact(stream, 5, owner, None),
        )

        async def append(
            transaction: _FilesystemTransaction,
        ) -> tuple[StoredFact, ...]:
            nonlocal subject_loads
            original_write = transaction._write
            original_load = transaction._cache.load_fact_subjects

            def write(relative: object, value: object) -> None:
                staged.append(str(relative))
                original_write(relative, value)

            def load_subjects(info: object) -> None:
                nonlocal subject_loads
                subject_loads += 1
                original_load(info)

            assert await transaction.guard_record(
                owner,
                expected_storage_version=0,
            ) is not None
            monkeypatch.setattr(transaction, "_write", write)
            monkeypatch.setattr(transaction._cache, "load_fact_subjects", load_subjects)
            await transaction.insert_facts(appended)
            assert subject_loads == 0
            values = await transaction.list_facts(
                FactQuery(stream, after_sequence=None)
            )
            latest = await transaction.list_facts(
                FactQuery(stream, subject_digest=subject_a, latest=True)
            )
            assert tuple(value.sequence for value in values) == (1, 2, 3, 4, 5)
            assert tuple(value.sequence for value in latest) == (4,)
            return values

        await reopened.mutate(append)
        assert sum(path.endswith("/meta.json") for path in staged) == 1
        assert sum("/subjects/" in path for path in staged) == 1
        assert sum("/items/" in path for path in staged) == 3

        values = await reopened.read(
            lambda transaction: transaction.list_facts(FactQuery(stream))
        )
        assert tuple(value.sequence for value in values) == (1, 2, 3, 4, 5)
        latest = await reopened.read(
            lambda transaction: transaction.list_facts(
                FactQuery(stream, subject_digest=subject_a, latest=True)
            )
        )
        assert tuple(value.sequence for value in latest) == (4,)
    finally:
        await reopened.close()


async def test_filesystem_fact_batch_stages_each_stream_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "state"
    owner = b"o" * 32
    stream_a = b"a" * 32
    stream_b = b"b" * 32
    subject_a = b"c" * 32
    subject_b = b"d" * 32
    store = FilesystemStateStore(
        root,
        namespace="fact-batch-multi",
        tenant_id="tenant",
        runtime_domain="execution",
    )
    await store.initialize()
    try:
        async def seed(transaction: _FilesystemTransaction) -> None:
            await transaction.insert_record(_filesystem_fact_record(owner))
            await transaction.insert_facts(
                (
                    _filesystem_fact(stream_a, 1, owner, subject_a),
                    _filesystem_fact(stream_b, 1, owner, subject_b),
                )
            )

        await store.mutate(seed)
    finally:
        await store.close()

    reopened = FilesystemStateStore(
        root,
        namespace="fact-batch-multi",
        tenant_id="tenant",
        runtime_domain="execution",
    )
    await reopened.initialize()
    try:
        staged: list[str] = []

        async def append(transaction: _FilesystemTransaction) -> None:
            assert await transaction.guard_record(
                owner,
                expected_storage_version=0,
            ) is not None
            original_write = transaction._write

            def write(relative: object, value: object) -> None:
                staged.append(str(relative))
                original_write(relative, value)

            monkeypatch.setattr(transaction, "_write", write)
            await transaction.insert_facts(
                (
                    _filesystem_fact(stream_a, 2, owner, subject_a),
                    _filesystem_fact(stream_b, 2, owner, subject_b),
                    _filesystem_fact(stream_a, 3, owner, subject_a),
                )
            )

        await reopened.mutate(append)
        assert sum("/items/" in path for path in staged) == 3
        assert sum(path.endswith("/meta.json") for path in staged) == 2
        assert sum("/subjects/" in path for path in staged) == 2
        expected_sequences = {stream_a: (1, 2, 3), stream_b: (1, 2)}
        for stream, expected in expected_sequences.items():
            values = await reopened.read(
                lambda transaction, current_stream=stream: transaction.list_facts(
                    FactQuery(current_stream)
                )
            )
            assert tuple(value.sequence for value in values) == expected
    finally:
        await reopened.close()


async def test_filesystem_object_store_syncs_payload_before_publish(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    original_sync = object_module._sync_file
    original_publish = object_module._publish_filesystem_object

    def sync(path: Path) -> None:
        events.append("sync")
        original_sync(path)

    def publish(
        temporary: Path,
        destination: Path,
        key: str,
        size: int,
        digest: str,
    ) -> bool:
        events.append("publish")
        return original_publish(
            temporary,
            destination,
            key,
            size,
            digest,
        )

    monkeypatch.setattr(object_module, "_sync_file", sync)
    monkeypatch.setattr(object_module, "_publish_filesystem_object", publish)
    store = FilesystemObjectStore(tmp_path / "objects")
    payload = b"filesystem-payload"
    digest = hashlib.sha256(payload).hexdigest()

    stat = await store.put(
        "payload",
        _chunks(payload),
        expected_size=len(payload),
        expected_digest=digest,
    )

    assert stat.digest == digest
    assert events[:2] == ["sync", "publish"]


async def test_sql_object_store_does_not_sync_staging_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "objects.db"
    engine = create_async_engine(f"sqlite+aiosqlite:///{path}")
    await provision_database(engine)
    store = SqlObjectStore(engine)
    payload = b"sql-payload"
    digest = hashlib.sha256(payload).hexdigest()

    def fail_sync(_path: Path) -> None:
        raise AssertionError("SQL staging payload must not be fsynced")

    monkeypatch.setattr(object_module, "_sync_file", fail_sync)
    try:
        stat = await store.put(
            "payload",
            _chunks(payload),
            expected_size=len(payload),
            expected_digest=digest,
        )
        assert stat.digest == digest
        assert await store.stat("payload") == stat
    finally:
        await engine.dispose()


async def test_session_model_context_reuses_observed_projection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = TranscriptRepository(
        object(),  # type: ignore[arg-type]
        object_store=None,
        namespace="io-history",
        tenant_id="tenant",
        runtime_domain=RuntimeDomain.CONVERSATION,
    )
    projection = ContextProjection(())
    expected = LoadedModelContext(())
    projection_reads = 0

    async def load_projection(_owner_id: str) -> ContextProjection:
        nonlocal projection_reads
        projection_reads += 1
        return projection

    async def load_projected(
        owner_id: str,
        observed: ContextProjection,
    ) -> LoadedModelContext:
        assert owner_id == "history"
        assert observed is projection
        return expected

    monkeypatch.setattr(repository, "load_projection", load_projection)
    monkeypatch.setattr(
        repository,
        "_load_model_context_from_projection",
        load_projected,
    )

    result = await repository.load_session_model_context(
        "history",
        tenant_id="tenant",
    )

    assert result is expected
    assert projection_reads == 1


async def test_message_spans_keep_the_observed_head_without_reading_a_new_head(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from .test_session_history_projection import _materialize, _session

    state = RuntimeStorage.in_memory()
    await state.initialize(namespace="observed-span-head", tenant_id="tenant")
    try:
        session = await state.conversation.sessions.create(_session())
        assert session.history_id is not None
        await _materialize(state, "first", tuple(f"prompt:{index}" for index in range(80)))
        repository = state.run_store.read_store(RuntimeDomain.CONVERSATION).transcript_repository
        head = await repository.get_head(session.history_id)
        assert head is not None
        await _materialize(state, "later", ("two",))

        async def reject_head_read(*_args, **_kwargs):
            raise AssertionError("a captured head must not be refreshed")

        monkeypatch.setattr(repository, "get_head_in_transaction", reject_head_read)
        spans = await repository.load_message_spans(
            session.history_id, ((128, 130),), observed_head=head,
        )
        assert [message.parts[0].content for message in spans[0]] == ["prompt:64", "answer:prompt:64"]
        with pytest.raises(AIError) as raised:
            await repository.load_message_spans(
                session.history_id, ((160, 162),), observed_head=head,
            )
        assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
        monkeypatch.setattr(repository._store, "read", reject_head_read)
        assert await repository.load_message_spans(
            session.history_id, ((0, 0), (160, 160)), observed_head=head,
        ) == ((), ())
    finally:
        await state.close()


async def test_approval_cancel_batches_known_record_sql(
    tmp_path: Path,
) -> None:
    path = tmp_path / "approvals.db"
    engine = create_async_engine(f"sqlite+aiosqlite:///{path}")
    await provision_database(engine)
    store = SqlStateStore(engine)
    await store.initialize()
    repository = RecoveryApprovalRepositoryImpl(
        store,
        namespace="io-approval",
        tenant_id="tenant",
    )
    now = datetime.now(timezone.utc)
    approval_ids = ("approval-a", "approval-b", "approval-c")
    for approval_id in approval_ids:
        await repository.create(
            ApprovalRecord(
                approval_id=approval_id,
                execution_id="execution",
                status=ApprovalStatus.PENDING,
                idempotency_key_digest=None,
                decision=None,
                decided_by=None,
                decision_digest=None,
                created_at=now,
                decided_at=None,
            )
        )

    statements: list[str] = []

    def capture_sql(
        _connection: object,
        _cursor: object,
        statement: str,
        _parameters: object,
        _context: object,
        _executemany: bool,
    ) -> None:
        statements.append(statement)

    event.listen(engine.sync_engine, "before_cursor_execute", capture_sql)
    try:
        cancelled = await store.mutate(
            lambda transaction: repository.cancel_pending_in_transaction(
                transaction,
                approval_ids,
                execution_id="execution",
                tenant_id="tenant",
                decided_at=now,
            )
        )
        assert tuple(value.approval_id for value in cancelled) == approval_ids
        assert all(value.status is ApprovalStatus.CANCELLED for value in cancelled)
        record_sql = [
            statement.upper()
            for statement in statements
            if "AI_STATE_RECORDS" in statement.upper()
        ]
        assert sum(statement.lstrip().startswith("SELECT") for statement in record_sql) == 1
        assert sum(statement.lstrip().startswith("UPDATE") for statement in record_sql) == 1
    finally:
        event.remove(engine.sync_engine, "before_cursor_execute", capture_sql)
        await store.close()
        await engine.dispose()


async def test_plan_add_uses_one_record_read_and_insert_sql(tmp_path: Path) -> None:
    path = tmp_path / "plan.db"
    engine = create_async_engine(f"sqlite+aiosqlite:///{path}")
    await provision_database(engine)
    store = SqlStateStore(engine)
    await store.initialize()
    adapter = HarnessPlanStoreAdapter(
        RuntimePlanStore(
            store,
            namespace="io-plan",
            tenant_id="tenant",
            owner_kind="execution",
            owner_id="execution",
        )
    )
    statements: list[str] = []

    def capture_sql(
        _connection: object,
        _cursor: object,
        statement: str,
        _parameters: object,
        _context: object,
        _executemany: bool,
    ) -> None:
        statements.append(statement)

    event.listen(engine.sync_engine, "before_cursor_execute", capture_sql)
    try:
        item = await adapter.add_item(HarnessPlanItem(content="first"))
        assert item.content == "first"
        record_sql = [
            statement.upper()
            for statement in statements
            if "AI_STATE_RECORDS" in statement.upper()
        ]
        assert sum(statement.lstrip().startswith("SELECT") for statement in record_sql) == 1
        assert sum(statement.lstrip().startswith("INSERT") for statement in record_sql) == 1
        assert sum(statement.lstrip().startswith("UPDATE") for statement in record_sql) == 0
    finally:
        event.remove(engine.sync_engine, "before_cursor_execute", capture_sql)
        await store.close()
        await engine.dispose()
