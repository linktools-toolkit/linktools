#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Prepared-only range resolution keeps one authoritative durable fallback."""

from dataclasses import replace
from pathlib import Path

import pytest

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime.state import RuntimeState
from linktools.ai.runtime.state._archive import ContentArchive, PreparedArchive
from .test_incremental_projection_boundaries import _storage
from .test_shared_content_archive import _archive, _publish

pytestmark = pytest.mark.asyncio


@pytest.mark.parametrize("backend", ("memory", "filesystem", "sqlite"))
async def test_prepared_ranges_resolve_without_a_state_scope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend: str,
) -> None:
    state = RuntimeState(_storage(backend, tmp_path))
    await state.initialize(namespace="prepared-range", tenant_id="tenant")
    try:
        archive = await _archive(state)
        prepared = PreparedArchive()
        values = tuple(("bytes", str(index % 5).encode()) for index in range(32))
        root = await archive.prepare_append(archive.empty_root(), values, prepared)
        expected = tuple(archive.content_record_keys((value,))[0] for value in values)

        async def unexpected_read(callback):
            raise AssertionError("complete prepared ranges must not open a read scope")

        monkeypatch.setattr(archive.store, "read", unexpected_read)
        ranges = ((0, 32), (4, 13), (13, 13), (2, 6), (0, 32))
        actual = await archive.content_ids_many(tuple(root.slice(a, b) for a, b in ranges), prepared=prepared)
        actual_keys = tuple(tuple(archive._key("archive_content", identity) for identity in group) for group in actual)
        assert actual_keys == tuple(expected[a:b] for a, b in ranges)
        assert await archive.content_ids_many((), prepared=prepared) == ()
        assert await archive.content_ids_many(()) == ()
    finally:
        await state.close()


@pytest.mark.parametrize("backend", ("memory", "filesystem", "sqlite"))
async def test_partial_prepared_ranges_fall_back_to_one_scope_and_cold_skips_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend: str,
) -> None:
    state = RuntimeState(_storage(backend, tmp_path))
    await state.initialize(namespace="prepared-range", tenant_id="tenant")
    try:
        archive = await _archive(state)
        initial = PreparedArchive()
        root = await archive.prepare_append(archive.empty_root(), tuple(("bytes", str(i).encode()) for i in range(31)), initial)
        await _publish(archive, initial)
        expected = await archive.content_ids(root)
        partial = PreparedArchive(owner=initial.owner)
        first = root.slices[0].first
        partial.nodes[first] = initial.nodes[first]
        read = archive.store.read
        resolve = ContentArchive._resolve_ranges
        scopes = []
        probes = []

        async def counted_read(callback):
            scopes.append(True)
            return await read(callback)

        async def counted_resolve(self, transaction, roots, **kwargs):
            probes.append(transaction is None)
            return await resolve(self, transaction, roots, **kwargs)

        monkeypatch.setattr(archive.store, "read", counted_read)
        monkeypatch.setattr(ContentArchive, "_resolve_ranges", counted_resolve)
        assert await archive.content_ids(root, prepared=partial) == expected
        assert len(scopes) == 1 and probes == [True, False]
        scopes.clear()
        probes.clear()
        assert await archive.content_ids(root) == expected
        assert len(scopes) == 1 and probes == [False]
    finally:
        await state.close()


@pytest.mark.parametrize("backend", ("memory", "filesystem", "sqlite"))
@pytest.mark.parametrize("fault", ("root_scope", "prepared_owner", "unexpected_next", "missing_node"))
async def test_prepared_range_probe_never_turns_invalid_or_missing_metadata_into_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend: str, fault: str,
) -> None:
    state = RuntimeState(_storage(backend, tmp_path))
    await state.initialize(namespace="prepared-range", tenant_id="tenant")
    try:
        archive = await _archive(state)
        prepared = PreparedArchive()
        root = await archive.prepare_append(archive.empty_root(), (("bytes", b"unpublished"),), prepared)
        first = root.slices[0].first
        if fault == "root_scope":
            root = replace(root, scope_id="another-scope")
        elif fault == "prepared_owner":
            prepared.owner = ("another-store", "another-scope", "another-content-scope")
        elif fault == "unexpected_next":
            prepared.nodes[first] = replace(prepared.nodes[first], next_forest=first)
        else:
            prepared.nodes.clear()
        scopes = []
        read = archive.store.read

        async def counted_read(callback):
            scopes.append(True)
            return await read(callback)

        monkeypatch.setattr(archive.store, "read", counted_read)
        with pytest.raises(AIError) as raised:
            await archive.content_ids(root, prepared=prepared)
        assert raised.value.code == ErrorCode.STORAGE_INTEGRITY_ERROR
        assert len(scopes) == (1 if fault == "missing_node" else 0)
    finally:
        await state.close()
