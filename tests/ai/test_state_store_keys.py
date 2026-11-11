#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Exact State partition families preserve isolation, reopen and current readers."""

import hashlib
from pathlib import Path

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.steps import native_store_key
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.persistence._sql import (
    SqlStateStore, _fact_from_row, _fact_values, _operation_from_row,
    _operation_values, _record_from_row, _record_values,
)
from linktools.ai.persistence.state import (
    STATE_GROUP_ARCHIVE_KEY, STATE_GROUP_METADATA_KEY, STORE_KEY_MAX_BYTES,
    StoredFact, StoredOperation, StoredRecord, capability_store_key,
    standalone_store_key, state_store_key, validate_store_key,
)


def _tuple(*parts: str) -> bytes:
    return b"".join(len(part.encode("utf-8")).to_bytes(2, "big") + part.encode("utf-8") for part in parts)


def _record(value: str) -> StoredRecord:
    return StoredRecord(b"record", None, None, "probe", b"record", None, 0, None, 0, None, {"value": value})


def _keys() -> tuple[bytes, ...]:
    return (
        state_store_key("namespace", "tenant", "execution"),
        state_store_key("other", "tenant", "execution"),
        state_store_key("namespace", "other", "execution"),
        state_store_key("namespace", "tenant", "conversation"),
        state_store_key("namespace", "tenant", "capability:one"),
        state_store_key("namespace", "tenant", "capability:two"),
        capability_store_key("namespace", "tenant", "one"),
        capability_store_key("namespace", "tenant", "two"),
        standalone_store_key("execution"),
        STATE_GROUP_METADATA_KEY,
        STATE_GROUP_ARCHIVE_KEY,
        b"\xff",
        b"\0" * STORE_KEY_MAX_BYTES,
    )


def test_store_families_compact_complete_framed_tuples_once_and_keep_utf8_bounds() -> None:
    assert len(set(_keys())) == len(_keys())
    assert state_store_key("a", "b", "c") == hashlib.sha256(_tuple("state_store", "a", "b", "c")).digest()
    assert capability_store_key("a", "b", "c") == hashlib.sha256(_tuple("capability-store", "a", "b", "c")).digest()
    assert standalone_store_key("c") == hashlib.sha256(_tuple("standalone", "c")).digest()
    assert STATE_GROUP_METADATA_KEY == _tuple("state-group-metadata")
    assert STATE_GROUP_ARCHIVE_KEY == _tuple("state-group-archive")
    assert state_store_key("a", "bc", "d") != state_store_key("ab", "c", "d")
    assert state_store_key("é", "b", "c") != state_store_key("e\u0301", "b", "c")
    namespace, tenant, capability = "界" * 85 + "a", "界" * 42 + "ab", "界" * 85 + "a"
    assert len(state_store_key(namespace, tenant, "capability:" + capability)) == 32
    assert len(capability_store_key(namespace, tenant, capability)) == 32
    for raw in _keys():
        assert validate_store_key(raw) is raw
    assert len(standalone_store_key("x" * 754)) == 32
    with pytest.raises(ValueError):
        standalone_store_key("x" * 755)
    for parts in ((namespace + "x", tenant, "execution"), (namespace, tenant + "x", "execution")):
        with pytest.raises(AIError):
            state_store_key(*parts)
    with pytest.raises(AIError):
        capability_store_key(namespace, tenant, capability + "x")


@pytest.mark.parametrize("backend", ("memory", "sqlite"))
@pytest.mark.asyncio
async def test_complete_partition_families_and_opaque_custom_keys_are_isolated(tmp_path: Path, backend: str) -> None:
    storage = PersistenceStorage.in_memory() if backend == "memory" else PersistenceStorage.sqlite(tmp_path / "state.db")
    async with Persistence.open("scope", storage=storage) as persistence:
        group = persistence.execution.executions.state_store.storage_group
        # Reserve actual group metadata/archive families for their real owning records.
        keys = tuple(key for key in _keys() if key not in (STATE_GROUP_METADATA_KEY, STATE_GROUP_ARCHIVE_KEY))
        stores = [await group.open_partition(key, create=True) for key in keys]
        for index, (key, store) in enumerate(zip(keys, stores)):
            assert store is not None and await group.open_partition(key) is store
            await store.mutate(lambda tx, index=index: tx.insert_record(_record(str(index))))
        for index, store in enumerate(stores):
            assert await store.read_record(b"record") == _record(str(index))
        assert set(stores) <= set(await group.scan_stores())
        dynamic = await persistence.capability_store("one")
        assert dynamic is await group.open_partition(capability_store_key("scope", "default", "one"))
        for invalid in (b"", b"x" * 769, "text", bytearray(b"raw")):
            with pytest.raises(ValueError):
                await group.open_partition(invalid, create=True)


@pytest.mark.asyncio
async def test_raw_sql_store_keys_reopen_and_keep_system_archive_scope(tmp_path: Path) -> None:
    storage = PersistenceStorage.sqlite(tmp_path / "state.db")
    keys = (b"\0" * 32, b"\0" * 768, state_store_key("namespace", "tenant", "execution"))
    scope = None
    for reopened in (False, True):
        async with Persistence.open("scope", storage=storage, read_only=reopened) as persistence:
            group = persistence.execution.executions.state_store.storage_group
            archive = await group.open_archive(create=not reopened)
            assert archive is not None
            if reopened:
                assert archive[0] == scope
            scope = archive[0]
            assert archive[1].store_key == STATE_GROUP_ARCHIVE_KEY
            for index, key in enumerate(keys):
                store = await group.open_partition(key, create=not reopened)
                assert store is not None and store.store_key == key
                alias = SqlStateStore(group.context.engine, metadata=group.metadata, context=group.context,
                                      group=group, store_key=store.store_key)
                await alias.initialize()
                assert alias.store_key == key
                if not reopened:
                    await store.mutate(lambda tx, index=index: tx.insert_record(_record(str(index))))
                assert await store.read_record(b"record") == _record(str(index))
                assert await alias.read_record(b"record") == _record(str(index))
                await alias.close()
            async with group.context.engine.connect() as connection:
                rows = (await connection.execute(text("SELECT DISTINCT store_key, typeof(store_key) FROM ai_state_records"))).all()
                assert all(type(key) is bytes and kind == "blob" for key, kind in rows)
                assert set(keys) | {STATE_GROUP_METADATA_KEY} <= {key for key, _ in rows}


def test_standalone_constructor_uses_compact_partition_without_rehashing_explicit_keys() -> None:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    store = SqlStateStore(engine)
    assert store.store_key == standalone_store_key("conversation")
    assert len(store.store_key) == 32
    alias = SqlStateStore(engine, group=store.storage_group, store_key=store.store_key)
    assert alias.store_key == store.store_key


@pytest.mark.parametrize("invalid", (b"", b"x" * 769, "text", bytearray(b"raw")))
def test_sql_constructor_and_complete_current_rows_reject_invalid_store_keys(invalid: object) -> None:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    with pytest.raises(ValueError):
        SqlStateStore(engine, store_key=invalid)
    record = _record("saved")
    fact = StoredFact(b"stream", 1, record.key, "probe", None, None, {})
    operation = StoredOperation(b"operation", b"stream", 1, "pending", False, {})
    for value, encode, decode in ((record, _record_values, _record_from_row),
                                  (fact, _fact_values, _fact_from_row),
                                  (operation, _operation_values, _operation_from_row)):
        row = {**encode(value), "store_key": b"raw"}
        assert decode(row) == value
        with pytest.raises(AIError) as error:
            decode({**row, "store_key": invalid})
        assert error.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


@pytest.mark.asyncio
@pytest.mark.parametrize("table", ("ai_state_records", "ai_state_aliases", "ai_state_facts", "ai_state_sequences", "ai_state_operations"))
async def test_partition_enumeration_rejects_corrupt_persisted_store_keys(tmp_path: Path, table: str) -> None:
    async with Persistence.open("scope", storage=PersistenceStorage.sqlite(tmp_path / "state.db")) as persistence:
        group = persistence.execution.executions.state_store.storage_group
        store = await group.open_partition(b"custom", create=True)
        assert store is not None

        async def seed(tx):
            from linktools.ai.persistence.state import StoredAlias
            record = _record("saved")
            await tx.insert_record(record)
            await tx.insert_alias(StoredAlias(b"alias", record.key))
            await tx.insert_fact(StoredFact(b"stream", 1, record.key, "probe", None, None, {}))
            await tx.next_sequence(b"stream")
            await tx.insert_operation(StoredOperation(b"operation", b"stream", 1, "pending", False, {}))

        await store.mutate(seed)
        async with group.context.engine.begin() as connection:
            await connection.execute(text(f"UPDATE {table} SET store_key = :invalid WHERE store_key = :key"),
                                     {"invalid": b"x" * 769, "key": b"custom"})
        with pytest.raises(AIError) as error:
            await group.scan_stores()
        assert error.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


def test_raw_store_and_partition_names_are_not_in_object_representations() -> None:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    private = "private-capability-identity"
    store = SqlStateStore(engine, runtime_domain=private, store_key=private.encode())
    assert private not in repr(store)
    assert private not in repr(store.storage_group)
    assert private not in repr(store._reads)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ("record", "fact", "operation"))
@pytest.mark.parametrize("known_group", (False, True))
async def test_native_publication_uses_raw_state_owner_and_native_target_without_extra_lookup(
    monkeypatch: pytest.MonkeyPatch, kind: str, known_group: bool,
) -> None:
    import json
    from types import SimpleNamespace
    from sqlalchemy.dialects import mysql, postgresql, sqlite
    from linktools.ai.persistence import _native_content_index as index
    from linktools.ai.persistence._message_root import NativeMessageRoot, NativeMessageSlice
    from linktools.ai.persistence.contracts import ArchivedContextBlock
    from linktools.ai.persistence.state import record_key

    group_id, native_store, root_id = "a" * 32, native_store_key("native", "tenant"), "c" * 64
    store_key = b"\xff" * STORE_KEY_MAX_BYTES
    root = NativeMessageRoot(group_id, native_store, (NativeMessageSlice(root_id, 1, 0, 1),))
    owner_key = b"owner" if kind != "operation" else b"o" * 1152
    identity = (b"s" * 1280, 1) if kind == "fact" else (owner_key,)
    selects = []
    selected_owners = []
    target_requests = []

    async def scalar(statement):
        selects.append(statement)
        for dialect in (mysql.dialect(), postgresql.dialect(), sqlite.dialect()):
            compiled = statement.compile(dialect=dialect)
            assert STATE_GROUP_METADATA_KEY in compiled.params.values()
            assert record_key("", "", "archive", "archive_scope", ("archive_scope_id",)) in compiled.params.values()
        return json.dumps({"archive_scope_id": group_id})

    async def targets(session, store, roots):
        target_requests.append((store, roots))
        return (17,)

    async def edges(context, session, owner_kind, sources, targets):
        assert owner_kind == kind and targets == [17]
        selected_owners.append(sources)
        for dialect in (mysql.dialect(), postgresql.dialect(), sqlite.dialect()):
            compiled = sources.compile(dialect=dialect)
            assert store_key in compiled.params.values()
            assert store_key.hex() not in compiled.params.values()
            assert len(compiled.params) == (3 if kind == "fact" else 2)
            assert list(sources.selected_columns.keys()) == ["id"]

    monkeypatch.setattr(index, "_owner_targets", targets)
    monkeypatch.setattr(index, "replace_owner_edges", edges)
    session = SimpleNamespace(info={}, scalar=scalar)
    index.stage_owner(session, kind, store_key, identity, ArchivedContextBlock(root, "d" * 64, 2))
    await index.flush_owners(SimpleNamespace(), session, group_id=group_id if known_group else None)
    assert len(selects) == (0 if known_group else 1)
    assert len(selected_owners) == 1
    assert target_requests == [(native_store, (root_id,))]
