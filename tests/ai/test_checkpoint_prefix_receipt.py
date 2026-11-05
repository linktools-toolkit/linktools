#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Native confirmation never hides caller mutation or uncertain durable truth."""

import asyncio
from dataclasses import replace
from pathlib import Path
from typing import Literal

import pytest
from pydantic_ai.messages import ModelRequest, UserPromptPart
from pydantic_ai_harness.step_persistence import ContinuableSnapshot, InMemoryStepStore, RunRecord

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime._step_store import RuntimeStepStore

from ._runtime_test_helpers import native_checkpoint
from .test_checkpoint_metadata_snapshot_io import _checkpoint
from .test_official_checkpoint_owner import _runtime_store


@pytest.mark.asyncio
async def test_confirmed_native_values_detach_caller_and_reader_mutations(tmp_path: Path) -> None:
    native = native_checkpoint(database=tmp_path / "copy.db").store
    store = _runtime_store(native)
    await store.initialize()
    store.bind_execution_producer("run", execution_id="execution", producer_generation=1)
    writer = RuntimeStepStore(native, InMemoryStepStore(), working_set=store)
    run = RunRecord(run_id="run", metadata={"linktools.transcript_start": "0", "label": "original"})
    snapshot = ContinuableSnapshot(run_id="run", step_index=1, idempotency_key="one", messages=[
        ModelRequest(parts=[UserPromptPart("original")], run_id="run"),
    ])
    try:
        await writer.register_run(run)
        await writer.save_snapshot(snapshot)
        run.metadata["label"] = "caller mutation"
        snapshot.messages[0].parts[0].content = "caller mutation"
        reader = store.checkpoint_persistence.reader
        first = await reader.latest_checkpoint(agent_run_id="run")
        assert first.messages[0].parts[0].content == "original"
        first.messages[0].parts[0].content = "reader mutation"
        observed_run = await reader.get_run(run_id="run")
        observed_run.metadata["label"] = "reader mutation"
        assert (await reader.latest_checkpoint(agent_run_id="run")).messages[0].parts[0].content == "original"
        assert (await reader.get_run(run_id="run")).metadata["label"] == "original"
        assert (await native.latest_snapshot(run_id="run")).messages[0].parts[0].content == "original"
    finally:
        await store.preflight_close()
        await store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("state", ("complete", "interrupted"))
async def test_unkeyed_native_snapshot_discards_cached_frontier(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, state: Literal["complete", "interrupted"],
) -> None:
    native = native_checkpoint(database=tmp_path / "unkeyed.db").store
    store = _runtime_store(native)
    await store.initialize()
    store.bind_execution_producer("run", execution_id="execution", producer_generation=1)
    writer = RuntimeStepStore(native, InMemoryStepStore(), working_set=store)
    run = RunRecord(run_id="run", metadata={"linktools.transcript_start": "0"})
    first = ContinuableSnapshot(run_id="run", step_index=1, messages=[], idempotency_key="one")
    try:
        await writer.register_run(run)
        await writer.save_snapshot(first)
        unkeyed = replace(first, step_index=2, state=state, idempotency_key=None)
        await writer.save_snapshot(unkeyed)
        reads = []
        latest = native.latest_snapshot

        async def tracked(**kwargs):
            reads.append(kwargs)
            return await latest(**kwargs)

        monkeypatch.setattr(native, "latest_snapshot", tracked)
        reader = store.checkpoint_persistence.reader
        assert await reader.latest_snapshot(run_id="run", include_interrupted=True) == unkeyed
        assert await reader.latest_snapshot(run_id="run") == (unkeyed if state == "complete" else first)
        assert len(reads) == 2
    finally:
        await store.preflight_close()
        await store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("committed", (False, True))
@pytest.mark.parametrize("cancelled", (False, True))
async def test_unconfirmed_snapshot_cannot_replace_unknown_readback_with_cached_truth(
    monkeypatch: pytest.MonkeyPatch, committed: bool, cancelled: bool,
) -> None:
    native = InMemoryStepStore()
    store = _runtime_store(native)
    await store.initialize()
    store.bind_execution_producer("run", execution_id="execution", producer_generation=1)
    writer = RuntimeStepStore(native, native, working_set=store)
    first = ContinuableSnapshot(run_id="run", step_index=1, messages=[], idempotency_key="one")
    try:
        await writer.register_run(RunRecord(run_id="run", metadata={"linktools.transcript_start": "0"}))
        await writer.save_snapshot(first)
        pending = replace(first, step_index=2, idempotency_key="two")
        save = native.save_snapshot
        latest = native.latest_snapshot
        writes = []

        async def uncertain(snapshot):
            writes.append(snapshot)
            if committed:
                await save(snapshot)
            if cancelled:
                raise asyncio.CancelledError
            raise OSError("source outcome unavailable")

        async def unknown(**kwargs):
            raise AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN)

        with monkeypatch.context() as patch:
            patch.setattr(native, "save_snapshot", uncertain)
            patch.setattr(native, "latest_snapshot", unknown)
            with pytest.raises(asyncio.CancelledError if cancelled else OSError):
                await writer.save_snapshot(pending)
            with pytest.raises(AIError) as caught:
                await store.checkpoint_persistence.materialize_recovery_checkpoint(agent_run_id="run", require_complete=True)
            assert caught.value.code is ErrorCode.STORAGE_COMMIT_UNKNOWN
            assert writes == [pending]
        assert await store.checkpoint_persistence.reader.latest_snapshot(run_id="run") == await latest(run_id="run")
        assert (await latest(run_id="run")).step_index == (2 if committed else 1)
    finally:
        await store.preflight_close()
        await store.close()


def test_shared_checkpoint_prefix_compares_tree_nodes_without_scanning_payloads(monkeypatch: pytest.MonkeyPatch) -> None:
    from linktools.ai.persistence._transcript_staging import MessageArchive, MessageSnapshot

    archive = MessageArchive()
    committed = archive.snapshot(_checkpoint(17).messages)
    extended = archive.snapshot(_checkpoint(23).messages)

    def no_payload_scan(self):
        raise AssertionError("shared immutable prefix must compare existing tree nodes")

    monkeypatch.setattr(MessageSnapshot, "_payloads", no_payload_scan)
    assert extended[:len(committed)] == committed
