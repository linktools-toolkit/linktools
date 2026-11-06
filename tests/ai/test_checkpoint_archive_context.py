#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Confirmed native views are local to their producing owner and source."""

from dataclasses import replace

import pytest
from pydantic_ai_harness.step_persistence import ContinuableSnapshot, InMemoryStepStore, RunRecord

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence import Persistence
from linktools.ai.persistence._archive import PreparedArchive
from linktools.ai.persistence._official_step_reader import OfficialStepReader

from .test_incremental_projection_boundaries import _storage

pytestmark = pytest.mark.asyncio


async def test_confirmed_checkpoint_reader_cannot_rebind_a_different_source() -> None:
    source = InMemoryStepStore()
    reader = OfficialStepReader(source)
    record = RunRecord(run_id="run", metadata={"linktools.transcript_start": "0"})
    await source.register_run(record)
    reader.confirm_run(record)
    reader.bind_step_store(source)
    with pytest.raises(AIError) as caught:
        reader.bind_step_store(InMemoryStepStore())
    assert caught.value.code is ErrorCode.STORAGE_CONFLICT
    assert reader.step_store is source
    assert await reader.get_run(run_id="run") == record


async def test_releasing_one_confirmed_run_keeps_other_owned_frontiers(monkeypatch: pytest.MonkeyPatch) -> None:
    source = InMemoryStepStore()
    reader = OfficialStepReader(source)
    for run_id in ("first", "second"):
        record = RunRecord(run_id=run_id, metadata={"linktools.transcript_start": "0"})
        snapshot = ContinuableSnapshot(run_id=run_id, step_index=1, messages=[], idempotency_key="first")
        await source.register_run(record)
        await source.save_snapshot(snapshot)
        reader.confirm_run(record)
        reader.confirm_snapshot(snapshot)
    reader.invalidate_run("first")
    advanced = ContinuableSnapshot(run_id="first", step_index=2, messages=[], idempotency_key="next")
    await source.save_snapshot(advanced)
    reads = []
    latest = source.latest_snapshot

    async def tracked(**kwargs):
        reads.append(kwargs["run_id"])
        return await latest(**kwargs)

    monkeypatch.setattr(source, "latest_snapshot", tracked)
    assert await reader.latest_snapshot(run_id="first") == advanced
    assert (await reader.latest_snapshot(run_id="second")).step_index == 1
    assert reads == ["first"]
    reader.clear_confirmed_runs()
    assert (await reader.latest_snapshot(run_id="second")).step_index == 1
    assert reads == ["first", "second"]


async def test_confirmed_run_identity_cannot_change_within_one_owner() -> None:
    reader = OfficialStepReader(InMemoryStepStore())
    original = RunRecord(run_id="run", metadata={"linktools.transcript_start": "0"})
    reader.confirm_run(original)
    with pytest.raises(AIError) as caught:
        reader.confirm_run(replace(original, metadata={"linktools.transcript_start": "1"}))
    assert caught.value.code is ErrorCode.STORAGE_CONFLICT
    assert await reader.get_run(run_id="run") == original


@pytest.mark.parametrize("prefix_count", (1024, 4096))
async def test_archive_publication_dependency_work_tracks_delta_not_confirmed_prefix(
    tmp_path, monkeypatch, prefix_count,
):
    from .test_shared_content_archive import _archive, _publish

    state = Persistence(_storage("memory", tmp_path))
    await state.initialize(namespace="archive", tenant_id="tenant")
    try:
        archive = await _archive(state)
        initial = PreparedArchive()
        root = await archive.prepare_append(
            archive.empty_root(), tuple(("bytes", str(i).encode()) for i in range(prefix_count)), initial,
        )
        await _publish(archive, initial)
        delta = PreparedArchive(owner=initial.owner, nodes=dict(initial.nodes))
        extended = await archive.prepare_append(root, (("bytes", b"next"),), delta)
        sequence_records = [record for record in delta.records.values() if record.kind == "archive_sequence"]
        key = archive._key
        key_work = []

        def count_key(*args):
            key_work.append(args)
            return key(*args)

        with monkeypatch.context() as patch:
            patch.setattr(archive, "_key", count_key)
            publication_keys = archive.publication_record_keys(delta)
        # Each pending node has one content and at most three node dependencies.
        # Growing confirmed history must not add hashing work to this delta.
        assert len(key_work) <= 4 * len(sequence_records)
        assert delta.records.keys() <= set(publication_keys)
        await _publish(archive, delta)
        assert await archive.read_contents(
            await archive.content_ids(extended, start=prefix_count - 1), codec="bytes",
        ) == (str(prefix_count - 1).encode(), b"next")
    finally:
        await state.close()
