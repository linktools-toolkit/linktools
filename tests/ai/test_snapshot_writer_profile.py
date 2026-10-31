#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Restored generations preserve the requested SQLite deployment profile."""

import sqlite3
from pathlib import Path

import pytest

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.snapshot import Snapshot, SnapshotLimits
from linktools.ai.storage import InMemoryObjectStore, ObjectRef
from .test_runtime_snapshot_operations import Offline

LIMITS = SnapshotLimits(max_entries=1000, max_bytes=1_000_000)


async def _snapshot(tmp_path: Path) -> tuple[InMemoryObjectStore, ObjectRef]:
    storage = PersistenceStorage.from_root(tmp_path / "source")
    async with Persistence.open("profile", storage=storage, tenant_id="tenant"):
        pass
    objects = InMemoryObjectStore()
    async with Persistence.open("profile", storage=storage, tenant_id="tenant", read_only=True) as state:
        reference = await Snapshot.create(persistence=state, object_store=objects, exclusive=Offline(), limits=LIMITS)
    return objects, reference


def _profile(root: Path) -> int:
    with sqlite3.connect((root / "runtime.db").as_uri() + "?mode=ro", uri=True) as connection:
        return connection.execute("PRAGMA application_id").fetchone()[0]


@pytest.mark.asyncio
@pytest.mark.parametrize("exclusive", (False, True))
async def test_snapshot_restore_selects_profile_and_reuses_matching_generation(tmp_path: Path, exclusive: bool) -> None:
    objects, reference = await _snapshot(tmp_path)
    options = dict(object_store=objects, target=tmp_path / "target", namespace="profile", tenant_id="tenant", limits=LIMITS)
    restored = await Snapshot.restore(reference, **options, exclusive_writer=exclusive)
    assert bool(_profile(restored.storage_root)) is exclusive
    assert await Snapshot.restore(reference, **options, exclusive_writer=exclusive) == restored
    assert restored.snapshot_digest == reference.digest
    async with restored.open_history():
        async with Persistence.open("profile", storage=PersistenceStorage.from_root(restored.storage_root, exclusive_writer=exclusive), tenant_id="tenant") as writer:
            assert writer.exclusive_execution_writer is exclusive


@pytest.mark.asyncio
@pytest.mark.parametrize("exclusive", (False, True))
async def test_snapshot_same_rejects_opposite_writer_profile(tmp_path: Path, exclusive: bool) -> None:
    objects, reference = await _snapshot(tmp_path)
    options = dict(object_store=objects, target=tmp_path / "target", namespace="profile", tenant_id="tenant", limits=LIMITS)
    restored = await Snapshot.restore(reference, **options, exclusive_writer=exclusive)
    before = (tmp_path / "target" / "current.json").read_bytes()
    database_before = (restored.storage_root / "runtime.db").read_bytes()
    with pytest.raises(AIError) as caught:
        await Snapshot.restore(reference, **options, exclusive_writer=not exclusive)
    assert caught.value.code is ErrorCode.SNAPSHOT_CONFLICT
    assert (tmp_path / "target" / "current.json").read_bytes() == before
    assert bool(_profile(restored.storage_root)) is exclusive
    assert (restored.storage_root / "runtime.db").read_bytes() == database_before


@pytest.mark.asyncio
@pytest.mark.parametrize("winner_profile", (False, True))
async def test_snapshot_same_publish_race_rejects_opposite_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, winner_profile: bool,
) -> None:
    import asyncio

    objects, reference = await _snapshot(tmp_path)
    original = Persistence.restore_snapshot
    ready = {False: asyncio.Event(), True: asyncio.Event()}
    release = {False: asyncio.Event(), True: asyncio.Event()}

    async def restore(cls, ref, *, object_store, root, limits, exclusive_writer=False):
        await original(ref, object_store=object_store, root=root, limits=limits, exclusive_writer=exclusive_writer)
        ready[exclusive_writer].set()
        await release[exclusive_writer].wait()

    monkeypatch.setattr(Persistence, "restore_snapshot", classmethod(restore))
    target = tmp_path / "target"
    options = dict(object_store=objects, target=target, namespace="profile", tenant_id="tenant", limits=LIMITS)
    tasks = {profile: asyncio.create_task(Snapshot.restore(reference, **options, exclusive_writer=profile)) for profile in (False, True)}
    try:
        await asyncio.wait_for(asyncio.gather(*(event.wait() for event in ready.values())), 10)
        assert not (target / "current.json").exists()
        release[winner_profile].set()
        winner = await tasks[winner_profile]
        release[not winner_profile].set()
        with pytest.raises(AIError) as caught:
            await tasks[not winner_profile]
        assert caught.value.code is ErrorCode.SNAPSHOT_CONFLICT
        assert bool(_profile(winner.storage_root)) is winner_profile
        assert len(tuple((target / "generations").iterdir())) == 1
        assert not tuple((target / ".staging").iterdir())
    finally:
        for event in release.values():
            event.set()
        await asyncio.gather(*tasks.values(), return_exceptions=True)


@pytest.mark.asyncio
async def test_snapshot_default_shared_profile_and_uri_path(tmp_path: Path) -> None:
    objects, reference = await _snapshot(tmp_path)
    options = dict(object_store=objects, target=tmp_path / "spaces # question? 百分%", namespace="profile", tenant_id="tenant", limits=LIMITS)
    restored = await Snapshot.restore(reference, **options)
    assert _profile(restored.storage_root) == 0
    assert await Snapshot.restore(reference, **options) == restored


@pytest.mark.asyncio
async def test_snapshot_profile_read_does_not_recreate_missing_database(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from linktools.ai.storage import SqlStorageContext

    objects, reference = await _snapshot(tmp_path)
    options = dict(object_store=objects, target=tmp_path / "target", namespace="profile", tenant_id="tenant", limits=LIMITS)
    restored = await Snapshot.restore(reference, **options)
    original = SqlStorageContext.is_local_sqlite_source

    async def remove_before_open(context: SqlStorageContext, path: Path) -> bool:
        path.unlink()
        return await original(context, path)

    monkeypatch.setattr(SqlStorageContext, "is_local_sqlite_source", remove_before_open)
    with pytest.raises(AIError) as caught:
        await Snapshot.restore(reference, **options)
    assert caught.value.code is ErrorCode.STORAGE_UNAVAILABLE
    assert not (restored.storage_root / "runtime.db").exists()


@pytest.mark.asyncio
async def test_snapshot_profile_read_cancellation_closes_borrowed_observer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import asyncio
    from linktools.ai.storage import SqlStorageContext

    objects, reference = await _snapshot(tmp_path)
    options = dict(object_store=objects, target=tmp_path / "target", namespace="profile", tenant_id="tenant", limits=LIMITS)
    restored = await Snapshot.restore(reference, **options, exclusive_writer=True)
    entered = asyncio.Event()
    contexts = []
    original = SqlStorageContext.is_local_sqlite_source

    async def pause(context: SqlStorageContext, path: Path) -> bool:
        result = await original(context, path)
        contexts.append(context)
        entered.set()
        await asyncio.Event().wait()
        return result

    monkeypatch.setattr(SqlStorageContext, "is_local_sqlite_source", pause)
    task = asyncio.create_task(Snapshot.restore(reference, **options, exclusive_writer=True))
    try:
        await asyncio.wait_for(entered.wait(), 10)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert contexts and all(context.closed and not context.exclusive_writer for context in contexts)
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
    monkeypatch.setattr(SqlStorageContext, "is_local_sqlite_source", original)
    async with Persistence.open("profile", storage=PersistenceStorage.from_root(restored.storage_root, exclusive_writer=True), tenant_id="tenant") as writer:
        assert writer.exclusive_execution_writer


@pytest.mark.asyncio
@pytest.mark.parametrize("initial_profile", (False, True))
async def test_snapshot_profile_replacement_keeps_existing_generation_guards(tmp_path: Path, initial_profile: bool) -> None:
    objects, reference = await _snapshot(tmp_path)
    options = dict(object_store=objects, target=tmp_path / "target", namespace="profile", tenant_id="tenant", limits=LIMITS)
    old = await Snapshot.restore(reference, **options, exclusive_writer=initial_profile)
    before = (old.storage_root / "runtime.db").read_bytes()
    with pytest.raises(AIError) as caught:
        await Snapshot.restore(reference, **options, exclusive_writer=not initial_profile, replace_policy="replace")
    assert caught.value.code is ErrorCode.SNAPSHOT_CONFLICT
    new = await Snapshot.restore(reference, **options, exclusive_writer=not initial_profile,
                                 replace_policy="replace", expected_generation=old.generation, exclusive=Offline())
    assert new.generation != old.generation
    assert new.snapshot_digest == old.snapshot_digest
    assert bool(_profile(new.storage_root)) is not initial_profile
    assert (old.storage_root / "runtime.db").read_bytes() == before


@pytest.mark.asyncio
async def test_snapshot_writer_profile_requires_boolean_before_creating_target(tmp_path: Path) -> None:
    objects, reference = await _snapshot(tmp_path)
    with pytest.raises(TypeError):
        await Snapshot.restore(reference, object_store=objects, target=tmp_path / "target",
                               namespace="profile", tenant_id="tenant", limits=LIMITS, exclusive_writer=1)
    assert not (tmp_path / "target").exists()


@pytest.mark.asyncio
async def test_snapshot_profile_conflict_survives_observer_cleanup_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from linktools.ai.storage import SqlStorageContext

    objects, reference = await _snapshot(tmp_path)
    options = dict(object_store=objects, target=tmp_path / "target", namespace="profile", tenant_id="tenant", limits=LIMITS)
    restored = await Snapshot.restore(reference, **options, exclusive_writer=True)
    before = (restored.storage_root / "runtime.db").read_bytes()
    current = (tmp_path / "target" / "current.json").read_bytes()
    original_read = SqlStorageContext.is_local_sqlite_source
    original_close = SqlStorageContext.close
    contexts = []
    failure = OSError("observer dispose failed")

    async def read(context: SqlStorageContext, path: Path) -> bool:
        contexts.append(context)
        return await original_read(context, path)

    async def close(context: SqlStorageContext) -> None:
        if context in contexts:
            raise failure
        await original_close(context)

    monkeypatch.setattr(SqlStorageContext, "is_local_sqlite_source", read)
    monkeypatch.setattr(SqlStorageContext, "close", close)
    try:
        with pytest.raises(AIError) as caught:
            await Snapshot.restore(reference, **options)
        assert caught.value.code is ErrorCode.SNAPSHOT_CONFLICT
        assert caught.value.__cause__ is failure
        assert contexts and all(not context.exclusive_writer for context in contexts)
        assert (restored.storage_root / "runtime.db").read_bytes() == before
        assert (tmp_path / "target" / "current.json").read_bytes() == current
    finally:
        for context in contexts:
            await original_close(context)
