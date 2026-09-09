#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Backend-neutral bounded range-index contracts."""

import hashlib
from pathlib import Path

import pytest
from sqlalchemy.dialects import mysql, postgresql, sqlite
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.schema import CreateTable

from linktools.ai.migrate import provision_database
from linktools.ai.runtime.state._filesystem import FilesystemStateStore
from linktools.ai.runtime.state._plan import RuntimeDomain
from linktools.ai.runtime.state._schema import build_runtime_sql_metadata
from linktools.ai.runtime.state._sql import SqlStateStore
from linktools.ai.runtime.state._store import RecordQuery, StoredRecord

pytestmark = pytest.mark.asyncio
_SCOPE = b"s" * 32
_PARTITION = b"p" * 32


def _record(sort_key: str) -> StoredRecord:
    return StoredRecord(
        hashlib.sha256(sort_key.encode("ascii")).digest(),
        _PARTITION,
        _SCOPE,
        None,
        "probe",
        sort_key,
        None,
        0,
        None,
        0,
        None,
        {},
    )


async def _seed(store, values: tuple[StoredRecord, ...]) -> None:
    await store.mutate(lambda transaction: transaction.insert_records(values))


async def _list_prefix(store, prefix: str | None, *, limit: int = 10):
    return await store.read(
        lambda transaction: transaction.list_records(
            RecordQuery(
                scope_digest=_SCOPE,
                kind="probe",
                sort_key_prefix=prefix,
                limit=limit,
            )
        )
    )


async def test_filesystem_range_index_is_generic_and_does_not_enumerate_index(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "generic-range"
    store = FilesystemStateStore(
        root,
        namespace="generic-range",
        tenant_id="tenant",
        runtime_domain=RuntimeDomain.EXECUTION.value,
        _range_index=True,
    )
    await store.initialize()
    await _seed(
        store,
        tuple(_record(value) for value in ("-first", "A-001", "A-002", "a-001")),
    )
    await store.close()

    reopened = FilesystemStateStore(
        root,
        namespace="generic-range",
        tenant_id="tenant",
        runtime_domain=RuntimeDomain.EXECUTION.value,
        _range_index=True,
    )
    await reopened.initialize()
    original_iterdir = Path.iterdir

    def guarded_iterdir(path: Path):
        if "record-index" in path.parts:
            raise AssertionError("bounded range queries must not enumerate the index")
        return original_iterdir(path)

    monkeypatch.setattr(Path, "iterdir", guarded_iterdir)
    values = await _list_prefix(reopened, "A")
    assert [value.sort_key for value in values] == ["A-001", "A-002"]
    first = await _list_prefix(reopened, None, limit=1)
    assert [value.sort_key for value in first] == ["-first"]
    await reopened.close()


async def test_filesystem_range_index_is_opt_in(tmp_path: Path) -> None:
    root = tmp_path / "range-opt-in"
    store = FilesystemStateStore(
        root,
        namespace="range-opt-in",
        tenant_id="tenant",
        runtime_domain=RuntimeDomain.EXECUTION.value,
    )
    await store.initialize()
    await _seed(store, (_record("probe"),))
    assert not (root / "record-index").exists()
    await store.close()


async def test_filesystem_range_index_does_not_write_one_node_per_character(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = FilesystemStateStore(
        tmp_path / "compressed-range",
        namespace="compressed-range",
        tenant_id="tenant",
        runtime_domain=RuntimeDomain.EXECUTION.value,
        _range_index=True,
    )
    await store.initialize()
    commits: list[tuple[tuple[str, ...], tuple[str, ...]]] = []
    original_commit = store._commit_sync

    def capture_commit(transaction, base, target):
        commits.append((tuple(transaction.writes), tuple(transaction.deletes)))
        return original_commit(transaction, base, target)

    monkeypatch.setattr(store, "_commit_sync", capture_commit)
    await _seed(store, (_record("x" * 128),))
    assert len(commits) == 1
    index_writes = [path for path in commits[0][0] if path.startswith("record-index/")]
    assert len(index_writes) == 2
    await store.close()


async def test_sql_prefix_is_case_sensitive_and_matches_filesystem(
    tmp_path: Path,
) -> None:
    path = tmp_path / "range.db"
    engine = create_async_engine(f"sqlite+aiosqlite:///{path}")
    await provision_database(engine)
    store = SqlStateStore(engine)
    await store.initialize()
    try:
        await _seed(
            store,
            tuple(_record(value) for value in ("A-one", "A-two", "a-one")),
        )
        values = await _list_prefix(store, "A")
        assert [value.sort_key for value in values] == ["A-one", "A-two"]
    finally:
        await store.close()
        await engine.dispose()


async def test_sql_sort_key_collation_is_explicit_for_supported_backends() -> None:
    metadata = build_runtime_sql_metadata(frozenset({RuntimeDomain.EXECUTION}))
    column_type = metadata.tables["ai_state_records"].c.sort_key.type
    assert column_type.dialect_impl(postgresql.dialect()).collation == "C"
    assert column_type.dialect_impl(sqlite.dialect()).collation == "BINARY"
    assert column_type.dialect_impl(mysql.dialect()).collation == "utf8mb4_bin"
