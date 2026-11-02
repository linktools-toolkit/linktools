#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Preserve migration of real legacy bytes without requiring a legacy writer."""

import hashlib
import json
import zipfile
from pathlib import Path, PurePosixPath

import pytest
from pydantic_ai_harness.memory import MemoryOperation

from linktools.ai.persistence import FilesystemSnapshotReader, Persistence, PersistenceDomain, PersistenceStorage
from linktools.ai.runtime._memory import RuntimeMemoryStore
from linktools.ai.snapshot import SnapshotLimits
from linktools.ai.storage import InMemoryObjectStore, PayloadPolicy


@pytest.mark.asyncio
async def test_legacy_filesystem_fixture_restores_retained_data_without_source_changes(tmp_path: Path) -> None:
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
    await Persistence.restore_snapshot(snapshot, object_store=objects, root=target, limits=limits,
                                       exclusive_writer=True)
    async with Persistence.open(namespace, tenant_id=tenant,
                                storage=PersistenceStorage.from_root(target, exclusive_writer=True)) as state:
        checkpoint = await state.run_store.read_store(PersistenceDomain.CHECKPOINT).latest_checkpoint(agent_run_id="run")
        assert checkpoint is not None
        assert checkpoint.messages[-1].parts[0].content == "archived request body\n" * 5000
        memory = RuntimeMemoryStore(
            state.memory, object_store=state.object_store(PersistenceDomain.MEMORY), namespace=namespace,
            tenant_id=tenant, execution_id="execution", memory_scope="retained",
            payload_policy=PayloadPolicy(inline_limit_bytes=64),
        )
        content = "迁移内存\n" * 5000
        captured = await memory.capture()
        assert captured["memory/note.md"]["content"] == content
        operation = MemoryOperation(id="write-once", fingerprint=hashlib.sha256(content.encode()).hexdigest())
        receipt = await memory.get_operation(operation)
        assert receipt is not None and receipt.replayed
        replay = await memory.write("memory/note.md", content, expected_version=None, operation=operation)
        assert replay.replayed and receipt.version == replay.version == provenance["memory_version"]
        assert await memory.capture() == captured
    assert business_hashes() == before
