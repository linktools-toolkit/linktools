#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Current record projections retain complete pagination and snapshot validation."""

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from sqlalchemy.ext.asyncio import create_async_engine

from linktools.ai.core import SessionStatus
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.migrate import provision_database
from linktools.ai.persistence import PersistenceDomain
from linktools.ai.persistence._memory import InMemoryStateStore
from linktools.ai.persistence._repository_common import project_record
from linktools.ai.persistence._snapshot_validation import canonical_snapshot_indexes, validate_snapshot_domain
from linktools.ai.persistence._sql import SqlStateStore
from linktools.ai.persistence.codec import decode_record, encode_record
from linktools.ai.persistence.contracts import SessionRecord
from linktools.ai.persistence.state import RecordQuery, StateStore, StoredRecord, sortable_timestamp


def _record(identity: str) -> StoredRecord:
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    session = SessionRecord(
        session_id=identity, owner_principal_id="principal", status=SessionStatus.OPEN,
        revision=1, cwd=None, metadata={}, created_at=now, updated_at=now,
        closed_at=None, active_execution_id=None, agent_id="agent",
    )
    return project_record(
        namespace="records", tenant_id="tenant", domain=PersistenceDomain.CONVERSATION,
        kind="session", identity=(identity,), value=session, state=session.status.value,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "sqlite"))
async def test_projected_records_page_completely_and_validate_after_roundtrip(
    tmp_path: Path, backend: str,
) -> None:
    engine = None
    store: StateStore
    if backend == "memory":
        store = InMemoryStateStore()
    else:
        engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'records.db'}")
        await provision_database(engine)
        store = SqlStateStore(engine)
    await store.initialize()
    try:
        records = tuple(decode_record(encode_record(_record(identity))) for identity in (
            "session-z", "session-a", "session-中", "session-0", "session-b",
        ))
        await store.mutate(lambda transaction: transaction.insert_records(records))
        assert all(record.sort_key == record.key for record in records)
        expected = tuple(sorted(records, key=lambda value: (value.sort_key, value.key)))
        found: list[StoredRecord] = []
        while True:
            previous = found[-1] if found else None
            page = await store.read(lambda transaction: transaction.list_records(RecordQuery(
                kind="session", limit=2,
                after_sort_key=None if previous is None else previous.sort_key,
                after_key=None if previous is None else previous.key,
            )))
            if not page:
                break
            found.extend(page)
            assert len(found) <= len(records)
        assert tuple(found) == expected
        scope = dict(namespace="records", tenant_id="tenant", domain=PersistenceDomain.CONVERSATION)
        aliases, sequences = canonical_snapshot_indexes(**scope, records=expected, facts=(), operations=())
        validate_snapshot_domain(**scope, records=expected, aliases=aliases, sequences=sequences,
                                 facts=(), operations=())
        with pytest.raises(AIError) as error:
            validate_snapshot_domain(
                **scope, records=(replace(expected[0], sort_key=b"wrong"), *expected[1:]),
                aliases=aliases, sequences=sequences, facts=(), operations=(),
            )
        assert error.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
    finally:
        await store.close()
        if engine is not None:
            await engine.dispose()


def test_timestamp_sorting_preserves_chronology_and_stable_ties() -> None:
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    first, second = _record("first").key, _record("second").key
    same_time = (sortable_timestamp(now, first), sortable_timestamp(now, second))
    assert len(same_time[0]) == 40
    assert same_time[0][8:] == first
    assert same_time[0] != same_time[1]
    assert same_time[0] == sortable_timestamp(now, first)
    assert max(same_time) < sortable_timestamp(now + timedelta(microseconds=1), first)
