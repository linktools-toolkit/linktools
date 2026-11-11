#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Exact globally scoped record addresses across storage and transport boundaries."""

from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pytest
from sqlalchemy.ext.asyncio import create_async_engine

from linktools.ai.migrate import provision_database
from linktools.ai.persistence._memory import InMemoryStateStore
from linktools.ai.persistence._record_cache import StateRecordCache
from linktools.ai.persistence._repository_common import decode_record_cursor, record_cursor
from linktools.ai.persistence._sql import SqlStateStore
from linktools.ai.persistence.codec import decode_alias, decode_fact, decode_record, encode_alias, encode_fact, encode_record
from linktools.ai.persistence.state import (
    RECORD_KEY_MAX_BYTES, RecordQuery, RecordScanCursor, StoredAlias, StoredFact, StoredRecord,
    alias_key, parent_key, record_key, scope_key, sortable_timestamp,
)
from linktools.ai.storage import InMemoryCacheBackend


def _record(key: bytes) -> StoredRecord:
    return StoredRecord(key, None, None, "probe", key, None, 0, None, 0, None, {})


def test_record_addresses_preserve_roles_scopes_and_exact_flat_components() -> None:
    key = record_key("namespace", "tenant", "execution", "probe", ("ab", "c"))
    assert key.startswith(b"\x00\x06record\x00\x09namespace")
    variants = (
        record_key("other", "tenant", "execution", "probe", ("ab", "c")),
        record_key("namespace", "other", "execution", "probe", ("ab", "c")),
        record_key("namespace", "tenant", "trace", "probe", ("ab", "c")),
        record_key("namespace", "tenant", "execution", "other", ("ab", "c")),
        record_key("namespace", "tenant", "execution", "probe", ("a", "bc")),
        scope_key("namespace", "tenant", "execution", "probe", "owner", ("ab", "c")),
        parent_key("namespace", "tenant", "execution", "probe", "owner", ("ab", "c")),
    )
    assert len({key, *variants}) == len(variants) + 1
    assert record_key("namespace", "tenant", "execution", "probe", ("é",)) != record_key(
        "namespace", "tenant", "execution", "probe", ("e\N{COMBINING ACUTE ACCENT}",))
    assert record_key("", "", "archive", "archive_content", ("scope", "content"))
    for invalid in ("scalar", (), (1,), (True,), (("nested",),), ({"key": "value"},)):
        with pytest.raises((TypeError, ValueError)):
            record_key("namespace", "tenant", "execution", "probe", invalid)


def test_raw_record_references_round_trip_at_bounds_without_repr_exposure() -> None:
    key = b"sensitive-address:" + b"x" * (RECORD_KEY_MAX_BYTES - 18)
    assert len(key) == RECORD_KEY_MAX_BYTES
    record = replace(_record(key), parent_key=key)
    alias = StoredAlias(b"lookup", key)
    fact = StoredFact(b"s" * 32, 1, key, "event", None, None, {})
    assert decode_record(encode_record(record)) == record
    assert decode_alias(encode_alias(alias)) == alias
    assert decode_fact(encode_fact(fact)) == fact
    assert decode_record_cursor(record_cursor(record)) == (key, key)
    for value in (record, alias, fact, RecordScanCursor("probe", key),
                  RecordQuery(kind="probe", parent_key=key, after_sort_key=key, after_key=key)):
        assert "sensitive-address" not in repr(value)
    with pytest.raises(ValueError):
        _record(key + b"x")
    with pytest.raises(ValueError):
        sortable_timestamp(datetime(2026, 1, 1, tzinfo=timezone.utc), key)
    assert len(sortable_timestamp(datetime(2026, 1, 1, tzinfo=timezone.utc), key[:-8])) == RECORD_KEY_MAX_BYTES


def test_maximum_generated_turn_owner_fits_global_record_bound() -> None:
    key = record_key("n" * 256, "t" * 128, "execution", "turn_session", ("s" * 256, "r" * 393))
    assert len(key) == 1074


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "sqlite"))
async def test_one_bare_store_keeps_global_record_bindings_and_cleanup_separate(tmp_path: Path, backend: str) -> None:
    engine = None
    if backend == "memory":
        store = InMemoryStateStore()
    else:
        engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'record.db'}")
        await provision_database(engine)
        store = SqlStateStore(engine)
    await store.initialize()
    records = tuple(_record(record_key(ns, tenant, domain, "probe", ("same",)))
                    for ns, tenant, domain in (("one", "tenant", "execution"), ("two", "tenant", "execution"),
                                              ("one", "other", "execution"), ("one", "tenant", "trace")))
    try:
        await store.mutate(lambda tx: tx.insert_records(records))
        assert await store.read_records(tuple(value.key for value in records)) == {value.key: value for value in records}
        await store.mutate(lambda tx: tx.delete_record(records[0].key))
        assert await store.read_records(tuple(value.key for value in records)) == {value.key: value for value in records[1:]}
    finally:
        await store.close()
        if engine is not None:
            await engine.dispose()


@pytest.mark.asyncio
async def test_variable_length_cached_batches_keep_component_boundaries() -> None:
    store = InMemoryStateStore()
    await store.initialize()
    cache = StateRecordCache(store, InMemoryCacheBackend(max_bytes=100_000), scope="same")
    records = tuple(_record(key) for key in (b"a", b"bc", b"ab", b"c"))
    try:
        await store.mutate(lambda tx: tx.insert_records(records))
        first = await cache.read_records((b"a", b"bc"))
        second = await cache.read_records((b"ab", b"c"))
        assert set(first) == {b"a", b"bc"}
        assert set(second) == {b"ab", b"c"}
    finally:
        await cache.close()
        await store.close()
