#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Workspace capture keeps physical object storage out of portable content."""

import hashlib
import json
import os
import stat
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import pytest

from linktools.ai.core import canonical_json_bytes
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence import Persistence, PersistenceDomain, PersistenceStorage
from linktools.ai.persistence.contracts import ArtifactRecord
from linktools.ai.snapshot import Snapshot, SnapshotLimits
from linktools.ai.storage import (
    FilesystemObjectStore, InMemoryObjectStore, ObjectRef, ObjectStat, ObjectStore, read_object,
)
from linktools.ai.workspace import Workspace


class _CustomObjectStore:
    """A provider of object operations without filesystem inspection methods."""

    def __init__(self, store: FilesystemObjectStore | InMemoryObjectStore) -> None:
        self._store = store

    @property
    def store_id(self) -> str:
        return self._store.store_id

    async def put(
        self, key: str, chunks: AsyncIterator[bytes], *,
        expected_size: int, expected_digest: str,
    ) -> ObjectStat:
        return await self._store.put(
            key, chunks, expected_size=expected_size, expected_digest=expected_digest,
        )

    async def stat(self, key: str) -> ObjectStat | None:
        return await self._store.stat(key)

    async def validate_integrity(self) -> None:
        await self._store.validate_integrity()

    def open(self, key: str) -> AsyncIterator[bytes]:
        return self._store.open(key)

    def list_objects(self) -> AsyncIterator[ObjectStat]:
        return self._store.list_objects()


class _OfflineGuard:
    @asynccontextmanager
    async def offline_exclusivity(self) -> AsyncIterator[None]:
        yield


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("builtin", "custom_local", "custom_memory"))
async def test_workspace_snapshot_excludes_object_storage_and_reports_known_paths(
    tmp_path: Path, backend: str,
) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    (root / "README.md").write_text("project content", encoding="utf-8")
    source_root, target_root = root / "source-objects", root / "snapshots"
    if backend == "custom_memory":
        source: ObjectStore = _CustomObjectStore(InMemoryObjectStore("source"))
        target: ObjectStore = _CustomObjectStore(InMemoryObjectStore("snapshot"))
    else:
        source = FilesystemObjectStore(source_root, store_id="source")
        target = FilesystemObjectStore(target_root, store_id="snapshot")
        if backend == "custom_local":
            source, target = _CustomObjectStore(source), _CustomObjectStore(target)

    async def content() -> AsyncIterator[bytes]:
        yield b"unreferenced source object"

    stat = await source.put(
        "unreferenced", content(), expected_size=26,
        expected_digest=hashlib.sha256(b"unreferenced source object").hexdigest(),
    )
    storage = PersistenceStorage.sqlite(root / "runtime.db", object_store=source)
    reference = ObjectRef("runtime", stat.key, stat.digest, stat.size)
    async with Persistence.open("workspace-paths", storage=storage) as persistence:
        await persistence.artifact.records.put_metadata(ArtifactRecord(
            "artifact", "execution", "producer", "text/plain", reference,
            datetime(2026, 1, 1, tzinfo=timezone.utc),
        ))
    limits = SnapshotLimits(1000, 1024 * 1024)
    options = {"workspace_exclude_paths": (str(source_root), target_root)} if backend == "custom_local" else {}
    async with Persistence.open("workspace-paths", storage=storage, read_only=True) as persistence:
        for status in persistence.status():
            assert status["object_store_id"] == "source"
            assert status["object_store_paths"] == ((str(source_root),) if backend == "builtin" else ())
        first = await Snapshot.create(
            persistence=persistence, object_store=target, workspace=Workspace.load(root),
            exclusive=_OfflineGuard(), limits=limits, **options,
        )
        second = await Snapshot.create(
            persistence=persistence, object_store=target, workspace=Workspace.load(root),
            exclusive=_OfflineGuard(), limits=limits, **options,
        )
    assert first == second
    manifest = json.loads(await read_object(
        target, first.key, expected_digest=first.digest, expected_size=first.size,
    ))
    paths = [entry["path"] for entry in manifest["workspace"]["entries"]]
    assert "README.md" in paths
    assert "runtime.db" not in paths
    assert not any(path.split("/")[0] in {"source-objects", "snapshots"} for path in paths)
    restored = await Snapshot.restore(
        first, object_store=target, target=tmp_path / "restored",
        namespace="workspace-paths", tenant_id="default", limits=limits,
    )
    assert restored.workspace_root is not None
    assert (restored.workspace_root / "README.md").read_text(encoding="utf-8") == "project content"
    assert not (restored.workspace_root / "runtime.db").exists()
    assert not (restored.workspace_root / "source-objects").exists()
    assert not (restored.workspace_root / "snapshots").exists()
    async with Persistence.open(
        "workspace-paths", storage=PersistenceStorage.from_root(restored.storage_root), read_only=True,
    ) as persistence:
        artifact = await persistence.artifact.records.get_metadata("artifact", tenant_id="default")
        assert artifact is not None and artifact.object_ref == reference
        assert await persistence.read_object(PersistenceDomain.ARTIFACT, reference) == b"unreferenced source object"


@pytest.mark.asyncio
@pytest.mark.parametrize("custom", (False, True))
@pytest.mark.parametrize("overlap", ("root", "symlink"))
async def test_workspace_snapshot_rejects_excluded_roots_and_links(
    tmp_path: Path, custom: bool, overlap: str,
) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    object_root = root if overlap == "root" else root / "objects"
    object_root.mkdir(exist_ok=True)
    if overlap == "symlink":
        (root / "alias").symlink_to(object_root, target_is_directory=True)
    target: ObjectStore = FilesystemObjectStore(object_root, store_id="snapshot")
    if custom:
        target = _CustomObjectStore(target)
    storage = PersistenceStorage.sqlite(tmp_path / "runtime.db")
    async with Persistence.open("workspace-paths", storage=storage):
        pass
    options = {"workspace_exclude_paths": (object_root,)} if custom else {}
    async with Persistence.open("workspace-paths", storage=storage, read_only=True) as persistence:
        with pytest.raises(AIError) as rejected:
            await Snapshot.create(
                persistence=persistence, object_store=target, workspace=Workspace.load(root),
                exclusive=_OfflineGuard(), limits=SnapshotLimits(1000, 1024 * 1024), **options,
            )
        assert rejected.value.code is ErrorCode.SNAPSHOT_UNSUPPORTED


@pytest.mark.asyncio
async def test_workspace_snapshot_rejects_a_scalar_exclusion_before_publication(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    (root / "README.md").write_text("project content", encoding="utf-8")
    storage = PersistenceStorage.sqlite(tmp_path / "runtime.db")
    target = InMemoryObjectStore("snapshot")
    async with Persistence.open("workspace-paths", storage=storage):
        pass
    async with Persistence.open("workspace-paths", storage=storage, read_only=True) as persistence:
        with pytest.raises(TypeError, match="workspace_exclude_paths must be a sequence of paths"):
            await Snapshot.create(
                persistence=persistence, object_store=target, workspace=Workspace.load(root),
                workspace_exclude_paths=str(tmp_path / "unrelated"),
                exclusive=_OfflineGuard(), limits=SnapshotLimits(1000, 1024 * 1024),
            )
    assert [stat async for stat in target.list_objects()] == []


async def _workspace_mode_snapshot(tmp_path: Path) -> tuple[InMemoryObjectStore, ObjectRef]:
    root = tmp_path / "workspace"
    root.mkdir()
    (root / "content.txt").write_text("content", encoding="utf-8")
    storage = PersistenceStorage.sqlite(tmp_path / "runtime.db")
    archive = InMemoryObjectStore("snapshot")
    async with Persistence.open("workspace-modes", storage=storage):
        pass
    async with Persistence.open("workspace-modes", storage=storage, read_only=True) as persistence:
        ref = await Snapshot.create(
            persistence=persistence, object_store=archive, workspace=Workspace.load(root),
            exclusive=_OfflineGuard(), limits=SnapshotLimits(1000, 1024 * 1024),
        )
    return archive, ref


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", (0o002, -1, 0o200, True, 1.5))
@pytest.mark.parametrize("operation", ("verify", "restore"))
async def test_snapshot_rejects_invalid_execute_bits_before_workspace_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: object, operation: str,
) -> None:
    archive, original = await _workspace_mode_snapshot(tmp_path)
    manifest = json.loads(await read_object(
        archive, original.key, expected_digest=original.digest, expected_size=original.size,
    ))
    entry = next(entry for entry in manifest["workspace"]["entries"] if "content" in entry)
    entry["mode"] = mode
    payload = canonical_json_bytes(manifest)
    digest = hashlib.sha256(payload).hexdigest()

    async def chunks() -> AsyncIterator[bytes]:
        yield payload

    await archive.put("invalid-mode", chunks(), expected_size=len(payload), expected_digest=digest)
    ref = ObjectRef(archive.store_id, "invalid-mode", digest, len(payload))

    def reject_chmod(*args: object, **kwargs: object) -> None:
        pytest.fail("invalid snapshot must be rejected before chmod")

    monkeypatch.setattr(os, "chmod", reject_chmod)
    target = tmp_path / "restored"
    limits = SnapshotLimits(1000, 1024 * 1024)
    with pytest.raises(AIError) as rejected:
        if operation == "verify":
            await Snapshot.verify(ref, object_store=archive, limits=limits)
        else:
            await Snapshot.restore(
                ref, object_store=archive, target=target,
                namespace="workspace-modes", tenant_id="default", limits=limits,
            )
    assert rejected.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
    assert not target.exists()


@pytest.mark.asyncio
@pytest.mark.skipif(os.name != "posix", reason="POSIX execute-bit combinations require POSIX chmod")
async def test_snapshot_round_trips_all_execute_bit_combinations(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    modes = (0, 0o001, 0o010, 0o011, 0o100, 0o101, 0o110, 0o111)
    for mode in modes:
        path = root / f"mode-{mode:o}"
        path.write_text("content", encoding="utf-8")
        path.chmod(0o600 | mode)
    storage = PersistenceStorage.sqlite(tmp_path / "runtime.db")
    archive = InMemoryObjectStore("snapshot")
    async with Persistence.open("workspace-modes", storage=storage):
        pass
    limits = SnapshotLimits(1000, 1024 * 1024)
    async with Persistence.open("workspace-modes", storage=storage, read_only=True) as persistence:
        ref = await Snapshot.create(
            persistence=persistence, object_store=archive, workspace=Workspace.load(root),
            exclusive=_OfflineGuard(), limits=limits,
        )
    await Snapshot.verify(ref, object_store=archive, limits=limits)
    restored = await Snapshot.restore(
        ref, object_store=archive, target=tmp_path / "restored",
        namespace="workspace-modes", tenant_id="default", limits=limits,
    )
    assert restored.workspace_root is not None
    default_file = tmp_path / "default-file"
    default_file.write_text("content", encoding="utf-8")
    non_execute = stat.S_IMODE(default_file.stat().st_mode) & ~0o111
    for mode in modes:
        path = restored.workspace_root / f"mode-{mode:o}"
        assert path.read_text(encoding="utf-8") == "content"
        assert stat.S_IMODE(path.stat().st_mode) == non_execute | mode
