#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Bounded archive plans preserve exact occurrences on every builtin backend."""

from pathlib import Path

import pytest

from linktools.ai.runtime.state import RuntimeState
from linktools.ai.runtime.state._archive import PreparedArchive
from .test_incremental_projection_boundaries import _storage
from .test_shared_content_archive import _archive, _publish

pytestmark = pytest.mark.asyncio


@pytest.mark.parametrize("backend", ("memory", "filesystem", "sqlite"))
async def test_archive_plans_preserve_all_ranges_and_unpublished_occurrences(tmp_path: Path, backend: str) -> None:
    state = RuntimeState(_storage(backend, tmp_path))
    await state.initialize(namespace="archive", tenant_id="tenant")
    try:
        archive = await _archive(state)
        root = archive.empty_root()
        values = tuple(str(index % 5).encode() for index in range(32))
        for start in range(0, len(values), 2):
            prepared = PreparedArchive()
            root = await archive.prepare_append(root, tuple(("bytes", value) for value in values[start:start + 2]), prepared)
            # The same plan also resolves nodes not yet visible to durable readers.
            assert len(await archive.content_ids(root, prepared=prepared)) == start + 2
            await _publish(archive, prepared)
        ranges = tuple((start, end) for start in range(33) for end in range(start, 33))
        actual = await archive.read_ranges(tuple(root.slice(start, end) for start, end in ranges), codec="bytes")
        assert actual == tuple(values[start:end] for start, end in ranges)
        prepared = PreparedArchive()
        branch = await archive.prepare_append(root.slice(7, 12), (("bytes", b"branch"),), prepared)
        await _publish(archive, prepared)
        assert await archive.read_ranges((branch, root), codec="bytes") == (values[7:12] + (b"branch",), values)
    finally:
        await state.close()


@pytest.mark.parametrize("backend", ("memory", "filesystem", "sqlite"))
async def test_prepared_frontier_reuse_does_not_bypass_publication_integrity(tmp_path: Path, backend: str) -> None:
    from dataclasses import replace

    from linktools.ai.errors import AIError, ErrorCode

    state = RuntimeState(_storage(backend, tmp_path))
    await state.initialize(namespace="archive", tenant_id="tenant")
    try:
        archive = await _archive(state)
        initial = PreparedArchive()
        root = await archive.prepare_append(archive.empty_root(), (("bytes", b"old"),), initial)
        await _publish(archive, initial)
        prepared = PreparedArchive()
        await archive.prepare_append(root, (("bytes", b"new"),), prepared)
        captured = next(record for record in prepared.records.values()
                        if record.kind == "archive_sequence" and record.sort_key == root.slices[0].first)
        corrupt = replace(captured, storage_revision=captured.storage_revision + 1,
                          data={**dict(captured.data), "count": 3})
        assert await archive.store.mutate(lambda transaction: transaction.replace_record(
            corrupt, expected_storage_revision=captured.storage_revision,
        ))
        # Reuse may be local, but the immutable dependency must still match at commit.
        with pytest.raises(AIError) as raised:
            await _publish(archive, prepared)
        assert raised.value.code == ErrorCode.STORAGE_INTEGRITY_ERROR
    finally:
        await state.close()


@pytest.mark.parametrize("backend", ("memory", "filesystem", "sqlite"))
@pytest.mark.parametrize("has_baseline", (False, True))
async def test_empty_model_projection_does_not_open_archive_or_keep_baseline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend: str, has_baseline: bool,
) -> None:
    from pydantic_ai.messages import ModelResponse, TextPart
    from linktools.ai.runtime import RuntimeDomain
    from linktools.ai.runtime.state._contracts import ContextProjection

    state = RuntimeState(_storage(backend, tmp_path))
    await state.initialize(namespace="empty-projection", tenant_id="tenant")
    try:
        repository = state.run_store.read_store(RuntimeDomain.CHECKPOINT).transcript_repository
        baseline = None
        if has_baseline:
            previous = PreparedArchive()
            baseline = await repository.prepare_model_projection(
                (ModelResponse(parts=[TextPart("previous context")]),), previous,
            )
            await repository.publish_archive(previous)

        async def unexpected_archive(*, create=False):
            raise AssertionError("empty context must not open or initialize content storage")

        monkeypatch.setattr(repository, "content_archive", unexpected_archive)
        prepared = PreparedArchive()
        projection = await repository.prepare_model_projection((), prepared, baseline=baseline)
        assert projection == ContextProjection(())
        assert not prepared.records and not prepared.nodes
        assert (await repository.load_projected_context("run", projection)).model_messages() == ()
    finally:
        await state.close()
