#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Legacy migration reads clean state without repairing or opening a writer."""

import asyncio
import hashlib
import json
import shutil
import threading
from pathlib import Path

import pytest
from pydantic_ai_harness.memory import MemoryOperation

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence import FilesystemSnapshotReader, Persistence, PersistenceDomain, PersistenceStorage
from linktools.ai.persistence._capability_marker import ensure_capability_partition_marker
from linktools.ai.persistence._filesystem import FilesystemStateStorageGroup
from linktools.ai.persistence._filesystem_layout import _FilesystemCache
from linktools.ai.persistence.runs import AgentRunRecord
from linktools.ai.snapshot import SnapshotLimits
from linktools.ai.storage import InMemoryObjectStore, PayloadPolicy, read_object
from linktools.ai.runtime._memory import RuntimeMemoryStore

from .test_checkpoint_metadata_snapshot_io import _checkpoint

pytestmark = pytest.mark.asyncio
NAMESPACE, TENANT = "legacy-reader", "tenant"
LIMITS = SnapshotLimits(max_entries=1000, max_bytes=2_000_000)


def _contents(root: Path) -> dict[str, bytes]:
    return {path.relative_to(root).as_posix(): path.read_bytes() for path in root.rglob("*")
            if path.is_file() and path.name not in {"state.lock", "publication.lock"}}


def _memory(state: Persistence) -> RuntimeMemoryStore:
    return RuntimeMemoryStore(
        state.memory, object_store=state.object_store(PersistenceDomain.MEMORY), namespace=NAMESPACE,
        tenant_id=TENANT, execution_id="execution", memory_scope="retained",
        payload_policy=PayloadPolicy(inline_limit_bytes=64),
    )


def _retained_checkpoint():
    checkpoint = _checkpoint(3)
    checkpoint.messages[-1].parts[0].content = "archived request body\n" * 5000
    return checkpoint


async def _write(root: Path, *, external=None) -> tuple[str, MemoryOperation]:
    content = "迁移内存\n" * 5000
    operation = MemoryOperation(id="write-once", fingerprint=hashlib.sha256(content.encode()).hexdigest())
    async with Persistence.open(NAMESPACE, storage=PersistenceStorage.filesystem(root, object_store=external),
                                tenant_id=TENANT) as state:
        archive = state.run_store.read_store(PersistenceDomain.CHECKPOINT)
        await archive.register_agent_run(AgentRunRecord("run"))
        await archive.save_checkpoint(_retained_checkpoint())
        await _memory(state).write("memory/note.md", content, expected_version=None, operation=operation)
    return content, operation


@pytest.mark.parametrize("external", (False, True, "false_valued"))
async def test_legacy_reader_preserves_source_and_restores_typed_archive_and_memory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, external: bool | str,
) -> None:
    root, target = tmp_path / "source", tmp_path / "target"
    class FalseValuedStore(InMemoryObjectStore):
        def __bool__(self) -> bool:
            return False

    source_objects = (FalseValuedStore("source") if external == "false_valued" else
                      InMemoryObjectStore("source") if external else None)
    content, operation = await _write(root, external=source_objects)
    before = _contents(root)
    objects = InMemoryObjectStore("snapshot")
    async with Persistence.open(NAMESPACE, storage=PersistenceStorage.filesystem(root, object_store=source_objects),
                                tenant_id=TENANT, read_only=True) as state:
        expected = await state.export_snapshot(object_store=objects, limits=LIMITS)
    assert _contents(root) == before

    async def forbidden(*args, **kwargs):
        pytest.fail("offline reader initialized a StateStore writer")

    reader = FilesystemSnapshotReader(root, namespace=NAMESPACE, tenant_id=TENANT,
                                       source_object_store=source_objects)
    with monkeypatch.context() as patch:
        patch.setattr(FilesystemStateStorageGroup, "initialize", forbidden)
        try:
            reference = await reader.export_snapshot(object_store=objects, limits=LIMITS)
        finally:
            await reader.close()
    assert reference == expected
    assert _contents(root) == before
    manifest = json.loads(await read_object(objects, reference.key, expected_digest=reference.digest,
                                            expected_size=reference.size))
    assert manifest["archives"] and manifest["domains"]["checkpoint"]["facts"]
    assert any(item["domain"] == "memory" for item in manifest["objects"])
    assert any(item["source"]["key"].startswith("v1/archive/") for item in manifest["objects"])
    await Persistence.restore_snapshot(reference, object_store=objects, root=target, limits=LIMITS, exclusive_writer=True)
    async with Persistence.open(NAMESPACE, storage=PersistenceStorage.from_root(target, exclusive_writer=True),
                                tenant_id=TENANT) as state:
        checkpoint = await state.run_store.read_store(PersistenceDomain.CHECKPOINT).latest_checkpoint(agent_run_id="run")
        assert checkpoint is not None and checkpoint.messages == _retained_checkpoint().messages
        captured = await _memory(state).capture()
        receipt = await _memory(state).get_operation(operation)
        replay = await _memory(state).write("memory/note.md", content, expected_version=None, operation=operation)
        assert receipt is not None and receipt.replayed and replay.replayed and receipt.version == replay.version
        assert await _memory(state).capture() == captured
    assert _contents(root) == before


@pytest.mark.parametrize("damage", ("missing_root", "missing_member", "missing_manifest", "partial_members", "dirty_commit", "custom", "marker", "bad_marker"))
async def test_legacy_reader_rejects_incomplete_or_unproven_sources_without_writes(tmp_path: Path, damage: str) -> None:
    root = tmp_path / "source"
    if damage != "missing_root":
        await _write(root)
    if damage in {"custom", "marker", "bad_marker"}:
        async with Persistence.open(NAMESPACE, storage=PersistenceStorage.filesystem(root), tenant_id=TENANT) as state:
            if damage == "custom":
                await state.execution.executions.state_store.storage_group.open_partition(hashlib.sha256(b"opaque").digest(), create=True)
            else:
                await state.execution.executions.state_store.mutate(ensure_capability_partition_marker)
        if damage == "bad_marker":
            marker = next((root / "execution").rglob("operations/by-key/*/*.json"))
            value = json.loads(marker.read_text())
            value["state"] = "invalid"
            marker.write_text(json.dumps(value))
    elif damage in {"missing_member", "missing_manifest", "partial_members"}:
        manifest = next((root / ".state-groups").glob("*/manifest.json"))
        value = json.loads(manifest.read_text())
        member = next(item for item in value["members"] if item["runtime_domain"] == "memory")
        if damage == "partial_members":
            value["members"].remove(member)
            manifest.write_text(json.dumps(value))
        elif damage == "missing_member":
            shutil.rmtree(root / member["relative_path"])
        else:
            (root / member["relative_path"] / "manifest.json").unlink()
    elif damage == "dirty_commit":
        scope = next((root / ".state-groups").iterdir()).name
        journal = root / (".txn-" + scope)
        journal.mkdir()
        (journal / "commit").write_bytes(b"unresolved committed publication")
    before = _contents(root)
    objects = InMemoryObjectStore()
    reader = FilesystemSnapshotReader(root, namespace=NAMESPACE, tenant_id=TENANT)
    try:
        with pytest.raises(AIError) as caught:
            await reader.export_snapshot(object_store=objects, limits=LIMITS)
        expected = (ErrorCode.STORAGE_NOT_FOUND if damage == "missing_root" else
                    ErrorCode.STORAGE_RECOVERY_REQUIRED if damage == "dirty_commit" else
                    ErrorCode.SNAPSHOT_UNSUPPORTED if damage in {"custom", "marker"} else
                    ErrorCode.STORAGE_INTEGRITY_ERROR)
        assert caught.value.code is expected
        assert [item async for item in objects.list_objects()] == []
    finally:
        await reader.close()
    assert _contents(root) == before


@pytest.mark.parametrize("foreign", ("namespace", "tenant"))
async def test_legacy_reader_never_guesses_foreign_identity(tmp_path: Path, foreign: str) -> None:
    await _write(tmp_path)
    before = _contents(tmp_path)
    reader = FilesystemSnapshotReader(tmp_path, namespace="foreign" if foreign == "namespace" else NAMESPACE,
                                       tenant_id="foreign" if foreign == "tenant" else TENANT)
    try:
        with pytest.raises(AIError) as caught:
            await reader.export_snapshot(object_store=InMemoryObjectStore(), limits=LIMITS)
        assert caught.value.code is ErrorCode.STORAGE_NOT_FOUND
    finally:
        await reader.close()
    assert _contents(tmp_path) == before


async def test_legacy_reader_cancellation_drains_file_read_before_releasing_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    await _write(tmp_path)
    before = _contents(tmp_path)
    entered, release = threading.Event(), threading.Event()
    read = _FilesystemCache.scan_records_page

    def blocked(cache, **kwargs):
        entered.set()
        assert release.wait(5)
        return read(cache, **kwargs)

    monkeypatch.setattr(_FilesystemCache, "scan_records_page", blocked)
    reader = FilesystemSnapshotReader(tmp_path, namespace=NAMESPACE, tenant_id=TENANT)
    exporting = asyncio.create_task(reader.export_snapshot(object_store=InMemoryObjectStore(), limits=LIMITS))
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        exporting.cancel()
        await asyncio.sleep(0)
        assert not exporting.done()
        with pytest.raises(AIError) as active:
            await reader.close()
        assert active.value.code is ErrorCode.RUNTIME_DEPENDENCY_NOT_READY
        contender = Persistence(PersistenceStorage.filesystem(tmp_path))
        try:
            with pytest.raises(AIError) as busy:
                await contender.initialize(namespace=NAMESPACE, tenant_id=TENANT)
            assert busy.value.code is ErrorCode.STORAGE_CONFLICT
        finally:
            await contender.close()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(exporting, 5)
        async with Persistence.open(NAMESPACE, storage=PersistenceStorage.filesystem(tmp_path), tenant_id=TENANT):
            pass
    finally:
        release.set()
        await asyncio.gather(exporting, return_exceptions=True)
        await reader.close()
    assert _contents(tmp_path) == before


async def test_legacy_reader_retains_failed_unlock_for_explicit_close_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    await _write(tmp_path)
    before = _contents(tmp_path)
    reader = FilesystemSnapshotReader(tmp_path, namespace=NAMESPACE, tenant_id=TENANT)
    release = reader._group._group_lock.release
    unavailable = OSError("unlock unavailable")
    failed = True

    async def fail_release() -> None:
        if failed:
            raise unavailable
        await release()

    monkeypatch.setattr(reader._group._group_lock, "release", fail_release)
    try:
        with pytest.raises(OSError) as export_error:
            await reader.export_snapshot(object_store=InMemoryObjectStore(), limits=LIMITS)
        assert export_error.value is unavailable
        with pytest.raises(OSError) as close_error:
            await reader.close()
        assert close_error.value is unavailable
        contender = Persistence(PersistenceStorage.filesystem(tmp_path))
        try:
            with pytest.raises(AIError) as busy:
                await contender.initialize(namespace=NAMESPACE, tenant_id=TENANT)
            assert busy.value.code is ErrorCode.STORAGE_CONFLICT
        finally:
            await contender.close()
        failed = False
        await reader.close()
        await reader.close()
        with pytest.raises(AIError) as closed:
            await reader.export_snapshot(object_store=InMemoryObjectStore(), limits=LIMITS)
        assert closed.value.code is ErrorCode.STORAGE_CLOSED
        async with Persistence.open(NAMESPACE, storage=PersistenceStorage.filesystem(tmp_path), tenant_id=TENANT):
            pass
    finally:
        failed = False
        await reader.close()
    assert _contents(tmp_path) == before


async def test_legacy_reader_close_before_export_is_final(tmp_path: Path) -> None:
    reader = FilesystemSnapshotReader(tmp_path, namespace=NAMESPACE)
    await reader.close()
    with pytest.raises(AIError) as closed:
        await reader.export_snapshot(object_store=InMemoryObjectStore(), limits=LIMITS)
    assert closed.value.code is ErrorCode.STORAGE_CLOSED
    assert not tmp_path.exists() or not tuple(tmp_path.iterdir())


@pytest.mark.parametrize("limit", ("entries", "bytes"))
async def test_legacy_reader_preserves_output_admission_limits(tmp_path: Path, limit: str) -> None:
    await _write(tmp_path)
    before = _contents(tmp_path)
    reader = FilesystemSnapshotReader(tmp_path, namespace=NAMESPACE, tenant_id=TENANT)
    limits = SnapshotLimits(max_entries=1 if limit == "entries" else 1000,
                            max_bytes=64 if limit == "bytes" else LIMITS.max_bytes)
    try:
        with pytest.raises(AIError) as rejected:
            await reader.export_snapshot(object_store=InMemoryObjectStore(), limits=limits)
        assert rejected.value.code is ErrorCode.SNAPSHOT_UNSUPPORTED
    finally:
        await reader.close()
    assert _contents(tmp_path) == before


async def test_legacy_reader_ignores_uncommitted_staging_without_deleting_it(tmp_path: Path) -> None:
    await _write(tmp_path)
    reader = FilesystemSnapshotReader(tmp_path, namespace=NAMESPACE, tenant_id=TENANT)
    objects = InMemoryObjectStore()
    try:
        expected = await reader.export_snapshot(object_store=objects, limits=LIMITS)
        scope = next((tmp_path / ".state-groups").iterdir()).name
        stage = tmp_path / (".txn-" + scope) / "stage"
        stage.mkdir(parents=True)
        (stage / "incomplete.json").write_bytes(b"not yet committed")
        before = _contents(tmp_path)
        assert await reader.export_snapshot(object_store=objects, limits=LIMITS) == expected
        assert _contents(tmp_path) == before
    finally:
        await reader.close()


async def test_legacy_reader_keeps_primary_failure_when_unlock_also_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    await _write(tmp_path)
    reader = FilesystemSnapshotReader(tmp_path, namespace=NAMESPACE, tenant_id=TENANT)
    primary, cleanup = ValueError("destination unavailable"), OSError("unlock failed")
    original = reader._group._group_lock.release

    class Destination(InMemoryObjectStore):
        async def put(self, *args, **kwargs):
            raise primary

    async def fail_release() -> None:
        raise cleanup

    monkeypatch.setattr(reader._group._group_lock, "release", fail_release)
    try:
        with pytest.raises(ValueError) as caught:
            await reader.export_snapshot(object_store=Destination(), limits=LIMITS)
        assert caught.value is primary and caught.value.__cause__ is cleanup
    finally:
        monkeypatch.setattr(reader._group._group_lock, "release", original)
        await reader.close()


async def test_legacy_reader_does_not_shield_cancelled_destination_io(tmp_path: Path) -> None:
    await _write(tmp_path)
    reader = FilesystemSnapshotReader(tmp_path, namespace=NAMESPACE, tenant_id=TENANT)
    entered = asyncio.Event()

    class Destination(InMemoryObjectStore):
        async def put(self, *args, **kwargs):
            entered.set()
            await asyncio.Event().wait()

    exporting = asyncio.create_task(reader.export_snapshot(object_store=Destination(), limits=LIMITS))
    try:
        await asyncio.wait_for(entered.wait(), 5)
        exporting.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(exporting, 5)
        async with Persistence.open(NAMESPACE, storage=PersistenceStorage.filesystem(tmp_path), tenant_id=TENANT):
            pass
    finally:
        exporting.cancel()
        await asyncio.gather(exporting, return_exceptions=True)
        await reader.close()


async def test_legacy_reader_keeps_other_tenant_state_and_objects_out_of_snapshot(tmp_path: Path) -> None:
    await _write(tmp_path)
    async with Persistence.open(NAMESPACE, storage=PersistenceStorage.filesystem(tmp_path), tenant_id="other") as state:
        foreign = RuntimeMemoryStore(
            state.memory, object_store=state.object_store(PersistenceDomain.MEMORY), namespace=NAMESPACE,
            tenant_id="other", execution_id="other-execution", memory_scope="retained",
            payload_policy=PayloadPolicy(inline_limit_bytes=1),
        )
        body = "other tenant private body"
        await foreign.write("memory/other.md", body, expected_version=None,
                            operation=MemoryOperation(id="other", fingerprint=hashlib.sha256(body.encode()).hexdigest()))
    before = _contents(tmp_path)
    objects = InMemoryObjectStore()
    reader = FilesystemSnapshotReader(tmp_path, namespace=NAMESPACE, tenant_id=TENANT)
    try:
        reference = await reader.export_snapshot(object_store=objects, limits=LIMITS)
    finally:
        await reader.close()
    exported = []
    async for item in objects.list_objects():
        exported.append(await read_object(objects, item.key, expected_digest=item.digest, expected_size=item.size))
    assert all(b"other tenant private body" not in payload for payload in exported)
    manifest = json.loads(await read_object(objects, reference.key, expected_digest=reference.digest, expected_size=reference.size))
    assert manifest["tenant_id"] == TENANT
    assert len(manifest["domains"]["memory"]["records"]) == 1
    assert _contents(tmp_path) == before
