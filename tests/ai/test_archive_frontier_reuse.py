#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Loaded append frontiers are bounded, exact, and never replace fresh publication checks."""

from dataclasses import replace
from pathlib import Path

import pytest
from pydantic_ai.messages import ModelRequest, UserPromptPart

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime.state import RuntimeState
from linktools.ai.runtime.state._archive import ContentArchive, PreparedArchive
from linktools.ai.runtime.state._contracts import ContextProjection
from linktools.ai.runtime.state._plan import RuntimeDomain
from linktools.ai.runtime.state._step_contracts import AgentRunCheckpoint, AgentRunRecord
from .test_incremental_projection_boundaries import _storage
from .test_shared_content_archive import _archive, _publish

pytestmark = pytest.mark.asyncio


@pytest.mark.parametrize("backend", ("memory", "filesystem", "sqlite"))
async def test_loaded_frontier_appends_without_metadata_reads_and_preserves_old_roots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend: str,
) -> None:
    state = RuntimeState(_storage(backend, tmp_path))
    await state.initialize(namespace="archive", tenant_id="tenant")
    try:
        archive = await _archive(state)
        initial = PreparedArchive()
        values = tuple(str(index).encode() for index in range(100))
        root = await archive.prepare_append(archive.empty_root(), tuple(("bytes", value) for value in values), initial)
        await _publish(archive, initial)
        reads = []
        read = archive.store.read

        async def counted(callback):
            reads.append(True)
            return await read(callback)

        monkeypatch.setattr(archive.store, "read", counted)
        frontiers = {}
        assert await archive.read_ranges((root,), codec="bytes", append_frontiers=frontiers) == (values,)
        assert len(reads) == 1
        assert len(frontiers[root]) == 2
        assert all(record.kind == "archive_sequence" for record in frontiers[root])
        prepared = PreparedArchive()
        await archive.prepare_content((("bytes", b"new"),), prepared)

        async def forbidden(*args, **kwargs):
            pytest.fail("A validated loaded frontier must not read append metadata again")

        with monkeypatch.context() as patch:
            patch.setattr(archive.store, "read", forbidden)
            patch.setattr(archive.store, "read_records", forbidden)
            archive.reuse_frontier(root, frontiers[root], prepared)
            extended = await archive.prepare_append(root, (("bytes", b"new"),), prepared)
        await _publish(archive, prepared)
        fork_plan = PreparedArchive()
        fork = await archive.prepare_append(root.slice(4, 7), (("bytes", b"fork"),), fork_plan)
        await _publish(archive, fork_plan)
        assert await archive.read_ranges((root, extended, fork), codec="bytes") == (
            values, values + (b"new",), values[4:7] + (b"fork",),
        )
    finally:
        await state.close()


@pytest.mark.parametrize("change", ("root", "scope", "namespace", "count", "first", "next", "missing", "extra"))
async def test_frontier_reuse_rejects_mismatched_roots_and_records(tmp_path: Path, change: str) -> None:
    state = RuntimeState(_storage("memory", tmp_path))
    await state.initialize(namespace="archive", tenant_id="tenant")
    try:
        archive = await _archive(state)
        initial = PreparedArchive()
        root = await archive.prepare_append(archive.empty_root(), (("bytes", b"a"), ("bytes", b"b")), initial)
        other = await archive.prepare_append(archive.empty_root(), (("bytes", b"c"), ("bytes", b"d")), initial)
        await _publish(archive, initial)
        frontiers = {}
        await archive.read_ranges((root,), codec="bytes", append_frontiers=frontiers)
        records = frontiers[root]
        if change == "root":
            root = other
        elif change == "scope":
            root = replace(root, scope_id="foreign")
        elif change == "namespace":
            archive = ContentArchive(archive.store, archive.scope_id, namespace="foreign", tenant_id="tenant", object_store=None)
        elif change == "count":
            root = replace(root, slices=(replace(root.slices[0], count=3, end=3),))
        elif change in {"first", "next"}:
            position = 0 if change == "first" else 1
            bad = replace(records[position], data={**dict(records[position].data), "content": "0" * 64})
            records = tuple(bad if index == position else record for index, record in enumerate(records))
        elif change == "missing":
            records = records[:1]
        elif change == "extra":
            records = (*records, records[0])
        prepared = PreparedArchive()
        with pytest.raises(AIError) as caught:
            archive.reuse_frontier(root, records, prepared)
        assert caught.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
        assert not prepared.records and not prepared.nodes
    finally:
        await state.close()


@pytest.mark.parametrize("position", (0, 1))
async def test_reused_frontier_is_rechecked_against_fresh_publication_records(tmp_path: Path, position: int) -> None:
    state = RuntimeState(_storage("memory", tmp_path))
    await state.initialize(namespace="archive", tenant_id="tenant")
    try:
        archive = await _archive(state)
        initial = PreparedArchive()
        root = await archive.prepare_append(archive.empty_root(), (("bytes", b"a"), ("bytes", b"b")), initial)
        await _publish(archive, initial)
        frontiers = {}
        await archive.read_ranges((root,), codec="bytes", append_frontiers=frontiers)
        prepared = PreparedArchive()
        archive.reuse_frontier(root, frontiers[root], prepared)
        await archive.prepare_append(root, (("bytes", b"new"),), prepared)
        record = frontiers[root][position]
        changed = replace(record, storage_revision=record.storage_revision + 1,
                          data={**dict(record.data), "content": "0" * 64})
        assert await archive.store.mutate(lambda transaction: transaction.replace_record(
            changed, expected_storage_revision=record.storage_revision,
        ))
        with pytest.raises(AIError) as caught:
            await _publish(archive, prepared)
        assert caught.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
    finally:
        await state.close()


async def test_frontier_capture_does_not_load_excluded_bodies_or_escape_failed_reads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = RuntimeState(_storage("memory", tmp_path))
    await state.initialize(namespace="archive", tenant_id="tenant")
    try:
        archive = await _archive(state)
        initial = PreparedArchive()
        root = await archive.prepare_append(archive.empty_root(), tuple(("bytes", str(index).encode()) for index in range(10)), initial)
        await _publish(archive, initial)
        bodies = []
        read = archive._read_content

        async def counted(content):
            bodies.append(content.identity)
            return await read(content)

        monkeypatch.setattr(archive, "_read_content", counted)
        frontiers = {}
        assert await archive.read_ranges((root.slice(2, 3),), codec="bytes", append_frontiers=frontiers) == ((b"2",),)
        assert len(bodies) == 1
        assert frontiers == {}
        missing = next(record for record in initial.records.values() if record.kind == "archive_content")
        await archive.store.mutate(lambda transaction: transaction.delete_records((missing.key_digest,)))
        with pytest.raises(AIError):
            await archive.read_ranges((root,), codec="bytes", append_frontiers=frontiers)
        assert frontiers == {}

        async def corrupt_body(content):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

        await archive.store.mutate(lambda transaction: transaction.insert_records((missing,)))
        monkeypatch.setattr(archive, "_read_content", corrupt_body)
        with pytest.raises(AIError):
            await archive.read_ranges((root,), codec="bytes", append_frontiers=frontiers)
        assert frontiers == {}
    finally:
        await state.close()


@pytest.mark.parametrize("change", ("unchanged", "same_length", "reordered", "summary", "no_receipt", "multiple_blocks", "no_append"))
async def test_checkpoint_frontier_reuse_requires_fresh_exact_loaded_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: str,
) -> None:
    state = RuntimeState(_storage("memory", tmp_path))
    await state.initialize(namespace="archive", tenant_id="tenant")
    try:
        owner = state.run_store.read_store(RuntimeDomain.CHECKPOINT)
        history = owner.transcript_repository
        archive = await history.content_archive(create=True)
        messages = (ModelRequest(parts=[UserPromptPart("aa")]), ModelRequest(parts=[UserPromptPart("bb")]))
        initial = PreparedArchive()
        projection = await history.prepare_model_projection(messages, initial)
        await _publish(archive, initial)
        if change == "multiple_blocks":
            projection = ContextProjection((*projection.items, *projection.items))
        loaded = await history.load_projected_context("previous", projection)
        assert bool(loaded.archive_frontier) == (change != "multiple_blocks")
        if change == "no_receipt":
            loaded = replace(loaded, archive_frontier=())
        if change == "same_length":
            loaded.model_messages()[0].parts[0].content = "cc"
        incoming = ModelRequest(parts=[UserPromptPart("new")])
        incoming_messages = [] if change == "no_append" else [incoming]
        expected = (*loaded.model_messages(), *incoming_messages)
        if change == "reordered":
            expected = (*reversed(loaded.model_messages()), incoming)
        elif change == "summary":
            expected = (ModelRequest(parts=[UserPromptPart("summary")]), incoming)
        seeded = []
        reuse = ContentArchive.reuse_frontier

        def capture(self, root, records, prepared):
            seeded.append(root)
            return reuse(self, root, records, prepared)

        monkeypatch.setattr(ContentArchive, "reuse_frontier", capture)
        state.run_store.register_context_baseline("run", loaded)
        _, batch = await owner.prepare_run_checkpoint(AgentRunRecord("run"), AgentRunCheckpoint(
            "run", 1, incoming_messages, transcript_message_count_before=0,
            context_messages=expected if change in {"reordered", "summary"} else None,
        ))
        assert bool(seeded) == (change == "unchanged")
        await _publish(archive, batch[0].archive)
        observed = await history.load_projected_context("run", batch[0].projection)
        assert observed.model_messages() == expected
    finally:
        await state.close()
