#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Read retired bytes without mutation; restore current Memory snapshots."""

import hashlib
import json
import zipfile
from pathlib import Path, PurePosixPath

import pytest
from pydantic_ai_harness.memory import MemoryOperation

from linktools.ai.persistence import FilesystemSnapshotReader, Persistence, PersistenceDomain, PersistenceStorage
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.memory import PersistentMemoryStore
from linktools.ai.snapshot import SnapshotLimits
from linktools.ai.storage import InMemoryObjectStore, PayloadPolicy


@pytest.mark.asyncio
async def test_legacy_filesystem_fixture_rejects_retired_context_without_source_changes(tmp_path: Path) -> None:
    fixtures = Path(__file__).parent / "fixtures" / "persistence"
    provenance = json.loads((fixtures / "filesystem_core_v1.provenance.json").read_text())
    bundle = fixtures / "filesystem_core_v1.zip"
    assert hashlib.sha256(bundle.read_bytes()).hexdigest() == provenance["zip_sha256"]
    source = tmp_path / "source"
    source.mkdir()
    with zipfile.ZipFile(bundle) as archive:
        assert len(archive.namelist()) == len(set(archive.namelist()))
        for member in archive.infolist():
            relative = PurePosixPath(member.filename)
            assert not relative.is_absolute() and ".." not in relative.parts
            assert "\\" not in member.filename and not member.is_dir()
            assert (member.external_attr >> 16) & 0o170000 == 0o100000
            destination = source.joinpath(*relative.parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(archive.read(member))

    def business_hashes() -> dict[str, str]:
        return {
            path.relative_to(source).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in source.rglob("*")
            if path.is_file() and path.name not in {"state.lock", "publication.lock"}
        }

    before = business_hashes()
    assert before == provenance["business_file_sha256"]
    namespace, tenant = provenance["namespace"], provenance["tenant_id"]
    limits = SnapshotLimits(max_entries=1000, max_bytes=2_000_000)
    objects = InMemoryObjectStore(provenance["snapshot_object_store_id"])
    reader = FilesystemSnapshotReader(source, namespace=namespace, tenant_id=tenant)
    try:
        snapshot = await reader.export_snapshot(object_store=objects, limits=limits)
    finally:
        await reader.close()
    assert snapshot.digest == provenance["snapshot_digest"]
    assert business_hashes() == before

    target = tmp_path / "restored"
    with pytest.raises(AIError) as rejected:
        await Persistence.restore_snapshot(snapshot, object_store=objects, root=target, limits=limits,
                                           exclusive_writer=True)
    assert rejected.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
    assert not (target / "runtime.db").exists()
    assert business_hashes() == before


@pytest.mark.asyncio
async def test_current_memory_snapshot_preserves_content_and_operation_receipt(tmp_path: Path) -> None:
    namespace, tenant = "memory-snapshot", "tenant"
    content = "迁移内存\n" * 5000
    operation = MemoryOperation(id="write-once", fingerprint=hashlib.sha256(content.encode()).hexdigest())
    objects = InMemoryObjectStore("snapshots")
    limits = SnapshotLimits(max_entries=1000, max_bytes=2_000_000)

    def memory(state: Persistence) -> PersistentMemoryStore:
        return PersistentMemoryStore(
            state.memory, object_store=state.object_store(PersistenceDomain.MEMORY), namespace=namespace,
            tenant_id=tenant, execution_id="execution", memory_scope="retained",
            payload_policy=PayloadPolicy(inline_limit_bytes=64),
        )

    async with Persistence.open(namespace, tenant_id=tenant,
                                storage=PersistenceStorage.from_root(tmp_path / "current")) as state:
        store = memory(state)
        first = await store.write("memory/note.md", content, expected_version=None, operation=operation)
        captured = await store.capture()
    async with Persistence.open(namespace, tenant_id=tenant, read_only=True,
                                storage=PersistenceStorage.from_root(tmp_path / "current")) as state:
        snapshot = await state.export_snapshot(object_store=objects, limits=limits)
    target = tmp_path / "restored"
    await Persistence.restore_snapshot(snapshot, object_store=objects, root=target, limits=limits)
    async with Persistence.open(namespace, tenant_id=tenant,
                                storage=PersistenceStorage.from_root(target)) as state:
        store = memory(state)
        assert await store.capture() == captured
        receipt = await store.get_operation(operation)
        assert receipt is not None and receipt.replayed and receipt.version == first.version
        replay = await store.write("memory/note.md", content, expected_version=None, operation=operation)
        assert replay.replayed and replay.version == first.version
        assert await store.capture() == captured
