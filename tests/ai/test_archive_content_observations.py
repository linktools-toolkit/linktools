#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Metadata observations never stand in for body integrity or publication guards."""
from dataclasses import replace
from pathlib import Path

import pytest

from linktools.ai.errors import AIError
from linktools.ai.persistence import Persistence
from linktools.ai.persistence._archive import ContentArchive, PreparedArchive
from linktools.ai.storage import StoredPayload
from .test_incremental_projection_boundaries import _storage
from .test_shared_content_archive import _archive, _publish

pytestmark = pytest.mark.asyncio


@pytest.mark.parametrize("backend", ("memory", "filesystem", "sqlite"))
async def test_content_observation_consumes_missing_and_preserves_fresh_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend: str,
) -> None:
    state = Persistence(_storage(backend, tmp_path))
    await state.initialize(namespace="archive", tenant_id="tenant")
    try:
        archive = await _archive(state)
        values = (("bytes", b"new"), ("bytes", b"new"))
        keys = archive.content_record_keys(values)
        observed = await archive.store.read_records(keys)
        prepared = PreparedArchive()
        archive.observe_content(prepared, keys, observed)
        assert not prepared.records and not prepared._verified_content
        async def forbidden(*args, **kwargs):
            pytest.fail("The completed observation already covers this metadata read")
        with monkeypatch.context() as patch:
            patch.setattr(archive.store, "read", forbidden)
            patch.setattr(archive.store, "read_records", forbidden)
            root = await archive.prepare_append(archive.empty_root(), values, prepared)
        assert not prepared._observed_content
        rival = PreparedArchive()
        await archive.prepare_content(values[:1], rival)
        record = rival.records[keys[0]]
        rival.records[keys[0]] = replace(record, data={**record.data, "raw_size": 999})
        await archive.store.mutate(lambda tx: tx.insert_records(tuple(rival.records.values())))
        with pytest.raises(AIError):
            await _publish(archive, prepared)
        assert not prepared._observed_content
        assert root.count == 2
    finally:
        await state.close()


async def test_observation_still_checks_corrupt_body_and_does_not_propagate(tmp_path: Path) -> None:
    state = Persistence(_storage("memory", tmp_path))
    await state.initialize(namespace="archive", tenant_id="tenant")
    try:
        archive = await _archive(state)
        values = (("bytes", b"body"),)
        initial = PreparedArchive()
        await archive.prepare_content(values, initial)
        await _publish(archive, initial)
        keys = archive.content_record_keys(values)
        records = await archive.store.read_records(keys)
        original = records[keys[0]]
        corrupt = replace(original, data={**original.data, "payload": StoredPayload.inline_bytes(b"bad").to_json()})
        prepared = PreparedArchive()
        archive.observe_content(prepared, keys, {keys[0]: corrupt})
        with pytest.raises(AIError):
            await archive.prepare_content(values, prepared)
        assert not prepared._observed_content
        archive.observe_content(prepared, keys, records)
        with pytest.raises(AIError):
            archive.observe_content(prepared, keys, {})
        combined = PreparedArchive.combine((prepared,))
        assert not combined._observed_content and not prepared._observed_content
        foreign = ContentArchive(archive.store, archive.scope_id, namespace="other", tenant_id="tenant", object_store=None)
        with pytest.raises(AIError):
            foreign.observe_content(PreparedArchive(), keys, records)
        with pytest.raises(AIError):
            archive.observe_content(PreparedArchive(), (), records)
    finally:
        await state.close()


async def test_observation_unknown_key_failure_does_not_retain_consumed_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = Persistence(_storage("memory", tmp_path))
    await state.initialize(namespace="archive", tenant_id="tenant")
    try:
        archive = await _archive(state)
        values = (("bytes", b"observed"), ("bytes", b"unknown"))
        prepared = PreparedArchive()
        archive.observe_content(prepared, archive.content_record_keys(values[:1]), {})
        async def failure(*args, **kwargs):
            raise RuntimeError("metadata read failed")
        with monkeypatch.context() as patch:
            patch.setattr(archive.store, "read_records", failure)
            with pytest.raises(RuntimeError, match="metadata read failed"):
                await archive.prepare_content(values, prepared)
        assert not prepared._observed_content and not prepared.records
    finally:
        await state.close()


@pytest.mark.parametrize("backend", ("memory", "filesystem"))
async def test_unsupported_group_keeps_preparation_metadata_before_archive_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend: str,
) -> None:
    from linktools.ai.persistence import PersistenceDomain
    from linktools.ai.persistence.state import GroupRecordReader
    from linktools.ai.persistence._step_contracts import AgentRunRecord
    from .test_checkpoint_metadata_snapshot_io import _checkpoint

    state = Persistence(_storage(backend, tmp_path))
    await state.initialize(namespace="archive", tenant_id="tenant")
    try:
        archive = state.run_store.read_store(PersistenceDomain.CONVERSATION)
        assert not isinstance(archive.state_store.storage_group, GroupRecordReader)
        original_read = archive.state_store.read
        original_content = archive.transcript_repository.content_archive
        metadata_read = False
        async def read(callback):
            nonlocal metadata_read
            result = await original_read(callback)
            metadata_read = True
            return result
        async def content(*args, **kwargs):
            assert metadata_read, "Unsupported group must retain the original preparation order"
            return await original_content(*args, **kwargs)
        monkeypatch.setattr(archive.state_store, "read", read)
        monkeypatch.setattr(archive.transcript_repository, "content_archive", content)
        _, prepared = await archive.prepare_conversation_checkpoint(AgentRunRecord("run"), _checkpoint(2))
        assert len(prepared) == 1
    finally:
        await state.close()
