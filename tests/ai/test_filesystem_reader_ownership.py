#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A legacy reader owns only readonly parsing and native offline locks."""

import asyncio
import hashlib
import json
import shutil
from pathlib import Path

import pytest

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence import FilesystemSnapshotReader
from linktools.ai.snapshot import SnapshotLimits
from linktools.ai.storage import FilesystemWriterLock, InMemoryObjectStore

from ._filesystem_legacy_fixture_helpers import install_legacy_overlay, load_legacy_source

pytestmark = pytest.mark.asyncio
LIMITS = SnapshotLimits(1000, 2_000_000)


def _source(root: Path) -> Path:
    return load_legacy_source(root)


def _hashes(root: Path) -> dict[str, str]:
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob("*") if p.is_file() and p.name not in {"state.lock", "publication.lock"}}


async def test_offline_reader_discovers_members_only_after_excluding_the_writer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "source"
    _source(root)
    reader = FilesystemSnapshotReader(root, namespace="legacy-reader", tenant_id="tenant")
    entered, release = asyncio.Event(), asyncio.Event()
    acquire = reader._group._group_lock.acquire

    async def gate() -> None:
        entered.set()
        await release.wait()
        await acquire()

    monkeypatch.setattr(reader._group._group_lock, "acquire", gate)
    output = InMemoryObjectStore()
    exporting = asyncio.create_task(reader.export_snapshot(object_store=output, limits=LIMITS))
    try:
        await asyncio.wait_for(entered.wait(), 5)
        writer = FilesystemWriterLock(reader._group._group_lock.path)
        await writer.acquire()
        try:
            install_legacy_overlay(root, "custom")
        finally:
            await writer.release()
        before = _hashes(root)
        release.set()
        with pytest.raises(AIError) as rejected:
            await asyncio.wait_for(exporting, 5)
        assert rejected.value.code is ErrorCode.SNAPSHOT_UNSUPPORTED
        assert [item async for item in output.list_objects()] == []
        assert _hashes(root) == before
    finally:
        release.set()
        await asyncio.gather(exporting, return_exceptions=True)
        await reader.close()


async def test_offline_reader_revalidates_inventory_after_all_member_locks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "source"
    metadata = _source(root)
    manifest = metadata / "manifest.json"
    original = manifest.read_bytes()
    reader = FilesystemSnapshotReader(root, namespace="legacy-reader", tenant_id="tenant")
    validate = reader._group._validate_locked_inventory

    def changed_inventory(before):
        value = json.loads(original)
        value["members"] = [entry for entry in value["members"] if entry["runtime_domain"] != "memory"]
        manifest.write_text(json.dumps(value))
        return validate(before)

    monkeypatch.setattr(reader._group, "_validate_locked_inventory", changed_inventory)
    try:
        with pytest.raises(AIError) as rejected:
            await reader.export_snapshot(object_store=InMemoryObjectStore(), limits=LIMITS)
        assert rejected.value.code is ErrorCode.STORAGE_CONFLICT
        contender = FilesystemWriterLock(metadata / "state.lock")
        await contender.acquire()
        await contender.release()
    finally:
        manifest.write_bytes(original)
        await reader.close()


@pytest.mark.parametrize("cleanup_failure", (False, True))
async def test_offline_reader_cancelled_partial_acquire_settles_before_unlock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cleanup_failure: bool,
) -> None:
    root = tmp_path / "source"
    metadata = _source(root)
    before = _hashes(root)
    reader = FilesystemSnapshotReader(root, namespace="legacy-reader", tenant_id="tenant")
    entered, release = asyncio.Event(), asyncio.Event()
    original_acquire, original_release = FilesystemWriterLock.acquire, FilesystemWriterLock.release
    error = OSError("member unlock unavailable")
    blocked = None

    async def acquire(lock) -> None:
        nonlocal blocked
        await original_acquire(lock)
        if lock.path != metadata / "state.lock" and blocked is None:
            blocked = lock
            entered.set()
            await release.wait()

    async def unlock(lock) -> None:
        if cleanup_failure and lock is blocked:
            raise error
        await original_release(lock)

    monkeypatch.setattr(FilesystemWriterLock, "acquire", acquire)
    monkeypatch.setattr(FilesystemWriterLock, "release", unlock)
    exporting = asyncio.create_task(reader.export_snapshot(object_store=InMemoryObjectStore(), limits=LIMITS))
    try:
        await asyncio.wait_for(entered.wait(), 5)
        exporting.cancel()
        await asyncio.sleep(0)
        assert not exporting.done()
        with pytest.raises(AIError) as second:
            await reader.export_snapshot(object_store=InMemoryObjectStore(), limits=LIMITS)
        assert second.value.code is ErrorCode.RUNTIME_DEPENDENCY_NOT_READY
        with pytest.raises(AIError) as active:
            await reader.close()
        assert active.value.code is ErrorCode.RUNTIME_DEPENDENCY_NOT_READY
        contender = FilesystemWriterLock(metadata / "state.lock")
        with pytest.raises(AIError) as held:
            await contender.acquire()
        assert held.value.code is ErrorCode.STORAGE_CONFLICT
        release.set()
        with pytest.raises(asyncio.CancelledError) as cancelled:
            await asyncio.wait_for(exporting, 5)
        if cleanup_failure:
            assert cancelled.value.__cause__ is error
            with pytest.raises(AIError):
                await contender.acquire()
        cleanup_failure = False
        await reader.close()
        await contender.acquire()
        await contender.release()
    finally:
        release.set()
        cleanup_failure = False
        await asyncio.gather(exporting, return_exceptions=True)
        await reader.close()
    assert _hashes(root) == before


@pytest.mark.parametrize("damage", ("group_version", "member_version", "group_generation", "archive_scope", "archive_directory", "archive_binding", "archive_outside", "member_journal", "foreign_journal"))
async def test_offline_reader_rejects_invalid_metadata_without_repair(tmp_path: Path, damage: str) -> None:
    root = tmp_path / "source"
    metadata = _source(root)
    archive_root = root / ".state-archive" / metadata.name
    member_root = root / json.loads((metadata / "manifest.json").read_text())["members"][0]["relative_path"]
    if damage in {"group_version", "member_version"}:
        target = (metadata if damage == "group_version" else member_root) / "manifest.json"
        value = json.loads(target.read_text())
        value["version" if damage == "group_version" else "layout_version"] = 999
        target.write_text(json.dumps(value))
    elif damage == "group_generation":
        (metadata / "generation").write_text("not a generation")
    elif damage == "archive_scope":
        (metadata / "archive_scope.json").unlink()
    elif damage == "archive_directory":
        shutil.rmtree(archive_root)
    elif damage == "archive_binding":
        (metadata / "archive_scope.json").write_text('{"archive_scope_id": "not a scope"}')
    elif damage == "archive_outside":
        outside = tmp_path / "outside-archive"
        shutil.move(archive_root, outside)
        archive_root.symlink_to(outside, target_is_directory=True)
    elif damage == "member_journal":
        journal = member_root / ".txn"
        journal.mkdir()
        (journal / "commit").write_bytes(b"uncertain member publication")
    else:
        journal = root / ".txn-foreign"
        journal.mkdir()
        (journal / "commit").write_bytes(b"uncertain foreign publication")
        (journal / "plan.json").write_text(json.dumps({"writes": [{"path": member_root.relative_to(root).as_posix() + "/generation"}]}))
    before = _hashes(root)
    reader = FilesystemSnapshotReader(root, namespace="legacy-reader", tenant_id="tenant")
    try:
        with pytest.raises(AIError) as rejected:
            await reader.export_snapshot(object_store=InMemoryObjectStore(), limits=LIMITS)
        assert rejected.value.code is (ErrorCode.STORAGE_RECOVERY_REQUIRED if damage.endswith("journal") else ErrorCode.STORAGE_INTEGRITY_ERROR)
    finally:
        await reader.close()
    assert _hashes(root) == before


async def test_offline_reader_preserves_legal_in_root_archive_alias(tmp_path: Path) -> None:
    root = tmp_path / "source"
    metadata = _source(root)
    archive_root = root / ".state-archive" / metadata.name
    relocated = root / "archive-target"
    shutil.move(archive_root, relocated)
    archive_root.symlink_to(relocated, target_is_directory=True)
    before = _hashes(root)
    provenance = json.loads((Path(__file__).parent / "fixtures/persistence/filesystem_core_v1.provenance.json").read_text())
    objects = InMemoryObjectStore(provenance["snapshot_object_store_id"])
    reader = FilesystemSnapshotReader(root, namespace="legacy-reader", tenant_id="tenant")
    try:
        reference = await reader.export_snapshot(object_store=objects, limits=LIMITS)
        assert reference.digest == provenance["snapshot_digest"]
    finally:
        await reader.close()
    assert _hashes(root) == before
