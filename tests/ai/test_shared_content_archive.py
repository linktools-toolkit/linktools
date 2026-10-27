#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Immutable content ranges retain occurrences without copying prior bodies."""

from pathlib import Path

import pytest

from linktools.ai.errors import AIError
from linktools.ai.persistence import Persistence
from linktools.ai.persistence._archive import ArchiveRoot, ContentArchive, PreparedArchive
from linktools.ai.persistence._plan import PersistenceDomain
from .test_incremental_projection_boundaries import _storage

pytestmark = pytest.mark.asyncio


async def _archive(state: Persistence) -> ContentArchive:
    source = state.run_store.read_store(PersistenceDomain.CHECKPOINT).state_store
    scope, store = await source.storage_group.open_archive(create=True)
    return ContentArchive(store, scope, namespace="archive", tenant_id="tenant", object_store=None)


async def _publish(archive: ContentArchive, prepared: PreparedArchive) -> None:
    await archive.store.mutate(lambda tx: archive.publish(tx, prepared))


@pytest.mark.parametrize("backend", ("memory", "filesystem", "sqlite"))
async def test_shared_content_preserves_occurrences_exact_slices_and_old_roots(tmp_path: Path, backend: str) -> None:
    state = Persistence(_storage(backend, tmp_path))
    await state.initialize(namespace="archive", tenant_id="tenant")
    try:
        archive = await _archive(state)
        root = archive.empty_root()
        roots = []
        for index in range(20):
            prepared = PreparedArchive()
            contents = await archive.prepare_content((("bytes", str(index).encode()),), prepared)
            root = await archive.append(root, contents, prepared)
            await _publish(archive, prepared)
            roots.append(root)
        for index, captured in enumerate(roots):
            identities = await archive.content_ids(captured)
            assert await archive.read_contents(identities, codec="bytes") == tuple(str(i).encode() for i in range(index + 1))
        assert ArchiveRoot.from_json(root.to_json()) == root
        prepared = PreparedArchive()
        repeated = await archive.prepare_content((("bytes", b"new"), ("bytes", b"new")), prepared)
        branch = await archive.append(root.slice(1, 2), repeated, prepared)
        await _publish(archive, prepared)
        assert await archive.read_contents(await archive.content_ids(branch), codec="bytes") == (b"1", b"new", b"new")
        assert len(branch.slices) == 2
        records = await archive.store.read(lambda tx: tx.scan_records())
        assert sum(record.kind == "archive_content" for record in records) == 21
        assert sum(record.kind == "archive_sequence" for record in records) == 22
        assert await archive.read_contents(await archive.content_ids(root, start=8, end=11), codec="bytes") == (b"8", b"9", b"10")
    finally:
        await state.close()


async def test_partial_tail_is_never_extended_into_excluded_history(tmp_path: Path) -> None:
    state = Persistence(_storage("memory", tmp_path))
    await state.initialize(namespace="archive", tenant_id="tenant")
    try:
        archive = await _archive(state)
        prepared = PreparedArchive()
        contents = await archive.prepare_content(tuple(("bytes", value) for value in (b"A", b"B", b"C")), prepared)
        root = await archive.append(archive.empty_root(), contents, prepared)
        await _publish(archive, prepared)
        tail = root.slice(1, 2)
        for index in range(30):
            prepared = PreparedArchive()
            content = await archive.prepare_content((("bytes", str(index).encode()),), prepared)
            tail = await archive.append(tail, content, prepared)
            await _publish(archive, prepared)
            assert len(tail.slices) == 2
        values = await archive.read_contents(await archive.content_ids(tail), codec="bytes")
        assert values == (b"B",) + tuple(str(index).encode() for index in range(30))
        with pytest.raises(AIError):
            await archive.content_ids(ArchiveRoot("other", root.content_scope, root.slices))
    finally:
        await state.close()


async def test_prepared_content_reuses_verified_metadata_with_batch_local_body_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = Persistence(_storage("memory", tmp_path))
    await state.initialize(namespace="archive", tenant_id="tenant")
    try:
        archive = await _archive(state)
        initial = PreparedArchive()
        await archive.prepare_content((("bytes", b"body"),), initial)
        await _publish(archive, initial)
        reads = 0
        body_reads = 0
        read_records = archive.store.read_records
        read_content = archive._read_content

        async def counted_records(keys):
            nonlocal reads
            reads += 1
            return await read_records(keys)

        async def counted_content(content):
            nonlocal body_reads
            body_reads += 1
            return await read_content(content)

        monkeypatch.setattr(archive.store, "read_records", counted_records)
        monkeypatch.setattr(archive, "_read_content", counted_content)
        prepared = PreparedArchive()
        first = await archive.prepare_content((("bytes", b"body"),), prepared)
        second = await archive.prepare_content((("bytes", b"body"),), prepared)
        assert first == second
        assert reads == 1
        assert body_reads == 1
        await _publish(archive, prepared)
        await archive.prepare_content((("bytes", b"body"),), prepared)
        assert body_reads == 2
        fresh = PreparedArchive()
        await archive.prepare_content((("bytes", b"body"),), fresh)
        assert body_reads == 3
        assert await archive.read_contents(first, codec="bytes") == (b"body",)
    finally:
        await state.close()


async def test_combined_preparation_rejects_foreign_scope_and_conflicting_records(tmp_path: Path) -> None:
    from dataclasses import replace

    state = Persistence(_storage("memory", tmp_path))
    await state.initialize(namespace="archive", tenant_id="tenant")
    try:
        archive = await _archive(state)
        left = PreparedArchive()
        contents = await archive.prepare_content((("bytes", b"body"),), left)
        root = await archive.append(archive.empty_root(), contents, left)
        right = PreparedArchive(owner=left.owner, records=dict(left.records), nodes=dict(left.nodes))
        combined = PreparedArchive.combine((left, right))
        assert combined.records == left.records
        assert combined.nodes == left.nodes
        assert combined is not left
        await _publish(archive, combined)
        assert await archive.read_contents(await archive.content_ids(root), codec="bytes") == (b"body",)
        assert left.records == right.records
        with pytest.raises(AIError):
            PreparedArchive.combine((left, PreparedArchive(owner=("foreign", "archive", "tenant"))))
        key, record = next(iter(right.records.items()))
        right.records[key] = replace(record, data={"invalid": True})
        with pytest.raises(AIError):
            PreparedArchive.combine((left, right))
    finally:
        await state.close()


@pytest.mark.parametrize("backend", ("memory", "filesystem", "sqlite"))
async def test_range_metadata_uses_one_snapshot_and_body_reads_leave_it(tmp_path, monkeypatch, backend):
    from linktools.ai.persistence.state import active_state_scope

    state = Persistence(_storage(backend, tmp_path))
    await state.initialize(namespace="archive", tenant_id="tenant")
    try:
        archive = await _archive(state)
        prepared = PreparedArchive()
        contents = await archive.prepare_content(tuple(("bytes", str(i).encode()) for i in range(40)), prepared)
        root = await archive.append(archive.empty_root(), contents, prepared)
        await _publish(archive, prepared)
        group = archive.store.storage_group
        reads = 0
        bodies = 0
        group_read = group.read
        body_read = archive._read_content

        async def count_read(store, callback):
            nonlocal reads
            reads += 1
            return await group_read(store, callback)

        async def outside_scope(content):
            nonlocal bodies
            assert active_state_scope() is None
            bodies += 1
            return await body_read(content)

        monkeypatch.setattr(group, "read", count_read)
        monkeypatch.setattr(archive, "_read_content", outside_scope)
        values = await archive.read_ranges((root.slice(0, 1), root.slice(39, 40), root.slice(39, 40)), codec="bytes")
        assert values == ((b"0",), (b"39",), (b"39",))
        assert reads == 1
        assert bodies == 2
    finally:
        await state.close()
