#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Focused I/O invariants for Runtime storage optimization."""

import hashlib
from collections.abc import AsyncIterator
from datetime import datetime, timezone
from pathlib import Path

import pytest
from linktools.ai.core import ApprovalStatus
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.migrate import provision_database
from linktools.ai.persistence import Persistence
from linktools.ai.runtime._harness import HarnessPlanStoreAdapter
from linktools.ai.runtime._plan import RuntimePlanStore
from linktools.ai.persistence import PersistenceStorage
from linktools.ai.persistence import PersistenceDomain
from linktools.ai.persistence.contracts import (
    ApprovalRecord,
    ContextProjection,
    LoadedModelContext,
)
from linktools.ai.persistence._history import TranscriptRepository
from linktools.ai.persistence._checkpoint_repositories import (
    RecoveryApprovalRepositoryImpl,
)
from linktools.ai.persistence._sql import SqlStateStore
from linktools.ai.storage import (
    FilesystemObjectStore,
    SqlObjectStore,
)
from linktools.ai.storage import _object_filesystem as object_module
from pydantic_ai_harness.planning import PlanItem as HarnessPlanItem
from sqlalchemy import event
from sqlalchemy.ext.asyncio import create_async_engine

pytestmark = pytest.mark.asyncio


async def _chunks(value: bytes) -> AsyncIterator[bytes]:
    yield value


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
        runtime_domain=PersistenceDomain.CONVERSATION,
    )
    projection = ContextProjection(())
    expected = LoadedModelContext(())
    projection_reads = 0

    async def load_metadata(_owner_id: str) -> tuple[ContextProjection, None]:
        nonlocal projection_reads
        projection_reads += 1
        return projection, None

    async def load_projected(
        owner_id: str,
        observed: ContextProjection,
    ) -> LoadedModelContext:
        assert owner_id == "history"
        assert observed is projection
        return expected

    monkeypatch.setattr(repository, "_load_context_metadata", load_metadata)
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

    state = Persistence(PersistenceStorage.in_memory())
    await state.initialize(namespace="observed-span-head", tenant_id="tenant")
    try:
        session = await state.conversation.sessions.create(_session())
        assert session.history_id is not None
        await _materialize(state, "first", tuple(f"prompt:{index}" for index in range(80)))
        repository = state.run_store.read_store(PersistenceDomain.CONVERSATION).transcript_repository
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
