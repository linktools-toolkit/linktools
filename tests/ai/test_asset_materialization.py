#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pinned Asset projections and explicit temporary-directory ownership."""

import asyncio
import os
import threading
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import IO

import pytest
from linktools.ai.asset import (
    AssetKey,
    AssetMaterializer,
    AssetStore,
    AssetVersionRef,
    DirectoryAssetBackend,
    FilesystemAssetBackend,
    InMemoryAssetBackend,
)
from linktools.ai.asset import _materialization
from linktools.ai.capability import CapabilityGroup, SkillResource, validate_resource_path
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.storage import StorageEntryRevision, StorageOverlay


async def _store(root: Path, backend_name: str = "memory") -> AssetStore:
    backend = (
        InMemoryAssetBackend()
        if backend_name == "memory"
        else FilesystemAssetBackend(str(root))
    )
    store = AssetStore(StorageOverlay(backend, writer=backend))
    await store.initialize()
    return store


async def _file(store: AssetStore) -> AssetVersionRef:
    key = AssetKey("custom", "opaque://logical-identity")
    await store.put(key, b"print('pinned')\n")
    return (await store.resolve_versions((key,)))[0]


def _temporary_parent(monkeypatch: pytest.MonkeyPatch, root: Path) -> None:
    root.mkdir()
    create = _materialization.tempfile.mkdtemp

    def create_directory(*, prefix: str) -> str:
        return create(prefix=prefix, dir=root)

    monkeypatch.setattr(_materialization.tempfile, "mkdtemp", create_directory)


@pytest.mark.asyncio
@pytest.mark.parametrize("backend_name", ("memory", "filesystem"))
async def test_materialization_reads_pinned_history_and_owns_files(
    tmp_path: Path, backend_name: str,
) -> None:
    store = await _store(tmp_path / "backend", backend_name)
    try:
        ref = await _file(store)
        await store.put(ref.key, b"changed")
        await store.delete(ref.key)
        async with AssetMaterializer() as materializer:
            result = await materializer.materialize(
                store,
                {"scripts/main.py": ref},
                executable_bits={"scripts/main.py": 0o101},
            )
            assert result.root.is_absolute()
            assert result.files == {"scripts/main.py": result.root / "scripts/main.py"}
            assert result.files["scripts/main.py"].read_bytes() == b"print('pinned')\n"
            if os.name != "nt":
                assert result.root.stat().st_mode & 0o777 == 0o700
                assert result.files["scripts/main.py"].stat().st_mode & 0o7777 == 0o501
            with pytest.raises(TypeError):
                result.files["other"] = result.root / "other"  # type: ignore[index]
        assert not result.root.exists()
        await materializer.close()
        with pytest.raises(AIError) as error:
            await materializer.materialize(store, {})
        assert error.value.code is ErrorCode.STORAGE_CLOSED
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_empty_package_and_independent_owners_have_usable_roots(tmp_path: Path) -> None:
    store = await _store(tmp_path)
    first = AssetMaterializer()
    second = AssetMaterializer()
    try:
        ref = await _file(store)
        empty = await first.materialize(store, {})
        one, two = await asyncio.gather(
            first.materialize(store, {"main.py": ref}),
            second.materialize(store, {"main.py": ref}),
        )
        assert empty.root.is_dir()
        assert dict(empty.files) == {}
        assert len({empty.root, one.root, two.root}) == 3
        assert one.files["main.py"].stat().st_mode & 0o111 == 0
        await first.close()
        assert not empty.root.exists()
        assert not one.root.exists()
        assert two.files["main.py"].read_bytes() == b"print('pinned')\n"
    finally:
        await first.close()
        await second.close()
        await store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("path", (
    "../escape", "/absolute", "a/../escape", "a\\escape", "", ".",
    "a//b", "a/", "./a", "C:/escape", "a:stream", "a\x00b", "\ud800",
    "run:dev.py", "CON", "aux.txt", "script.", "script ", "a?b", "a\x01b",
))
async def test_materialization_rejects_unsafe_paths_before_creating_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, path: str,
) -> None:
    parent = tmp_path / "projections"
    _temporary_parent(monkeypatch, parent)
    store = await _store(tmp_path)
    try:
        ref = await _file(store)
        with pytest.raises(AIError) as declaration_error:
            validate_resource_path(path)
        assert declaration_error.value.code is ErrorCode.CAPABILITY_RESOLUTION_INVALID
        with pytest.raises(AIError) as resource_error:
            SkillResource(path, ref)
        assert resource_error.value.code is ErrorCode.REQUEST_FIELD_INVALID
        async with AssetMaterializer() as materializer:
            with pytest.raises(ValueError):
                await materializer.materialize(store, {path: ref})
        assert tuple(parent.iterdir()) == ()
    finally:
        await store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("modes", ({}, {"main.py": True}, {"main.py": -1},
                                    {"main.py": 0o2}, {"main.py": 0o4111},
                                    {"main.py": 0, "extra": 0}))
async def test_materialization_rejects_modes_outside_the_captured_file_set(
    tmp_path: Path, modes: dict[str, int],
) -> None:
    store = await _store(tmp_path)
    try:
        ref = await _file(store)
        async with AssetMaterializer() as materializer:
            with pytest.raises(ValueError):
                await materializer.materialize(store, {"main.py": ref}, executable_bits=modes)
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_materialization_rejects_overlapping_file_and_directory_paths(tmp_path: Path) -> None:
    store = await _store(tmp_path)
    try:
        ref = await _file(store)
        async with AssetMaterializer() as materializer:
            with pytest.raises(ValueError):
                await materializer.materialize(store, {"a": ref, "a/b": ref})
    finally:
        await store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid", ("integrity", "missing", "layer"))
async def test_invalid_version_references_leave_no_projection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, invalid: str,
) -> None:
    parent = tmp_path / "projections"
    _temporary_parent(monkeypatch, parent)
    store = await _store(tmp_path)
    try:
        ref = await _file(store)
        if invalid == "integrity":
            ref = replace(ref, etag="0" * 64)
            expected = ErrorCode.STORAGE_INTEGRITY_ERROR
        elif invalid == "missing":
            ref = replace(ref, revision=StorageEntryRevision(999))
            expected = ErrorCode.ASSET_VERSION_NOT_FOUND
        else:
            ref = replace(ref, layer_id="absent")
            expected = ErrorCode.ASSET_VERSION_LAYER_UNKNOWN
        async with AssetMaterializer() as materializer:
            with pytest.raises(AIError) as error:
                await materializer.materialize(store, {"main.py": ref})
            assert error.value.code is expected
        assert tuple(parent.iterdir()) == ()
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_partial_write_failure_removes_unpublished_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = tmp_path / "projections"
    _temporary_parent(monkeypatch, parent)
    store = await _store(tmp_path)
    open_file = Path.open

    def failing_open(path: Path, *args: object, **kwargs: object) -> IO[bytes] | IO[str]:
        if path.name == "fail.py":
            raise PermissionError("write failed")
        return open_file(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", failing_open)
    try:
        ref = await _file(store)
        async with AssetMaterializer() as materializer:
            with pytest.raises(AIError) as error:
                await materializer.materialize(store, {"a.py": ref, "fail.py": ref})
            assert error.value.code is ErrorCode.STORAGE_UNAVAILABLE
            assert tuple(parent.iterdir()) == ()
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_cancelled_materialization_waits_for_writer_before_removal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = tmp_path / "projections"
    _temporary_parent(monkeypatch, parent)
    store = await _store(tmp_path)
    started = threading.Event()
    release = threading.Event()
    write_files = _materialization._write_files

    def blocked_write(
        root: Path,
        files: tuple[tuple[str, bytes], ...],
        modes: Mapping[str, int],
    ) -> dict[str, Path]:
        started.set()
        assert release.wait(10)
        return write_files(root, files, modes)

    monkeypatch.setattr(_materialization, "_write_files", blocked_write)
    try:
        ref = await _file(store)
        async with AssetMaterializer() as materializer:
            task = asyncio.create_task(materializer.materialize(store, {"main.py": ref}))
            assert await asyncio.to_thread(started.wait, 5)
            task.cancel()
            await asyncio.sleep(0)
            task.cancel()
            await asyncio.sleep(0)
            assert not task.done()
            assert tuple(parent.iterdir())
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert tuple(parent.iterdir()) == ()
    finally:
        release.set()
        await store.close()


@pytest.mark.asyncio
async def test_close_waits_for_pending_writes_without_publishing_expired_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = tmp_path / "projections"
    _temporary_parent(monkeypatch, parent)
    store = await _store(tmp_path)
    started = threading.Event()
    release = threading.Event()
    write_files = _materialization._write_files

    def blocked_write(
        root: Path,
        files: tuple[tuple[str, bytes], ...],
        modes: Mapping[str, int],
    ) -> dict[str, Path]:
        started.set()
        assert release.wait(10)
        return write_files(root, files, modes)

    monkeypatch.setattr(_materialization, "_write_files", blocked_write)
    materializer = AssetMaterializer()
    try:
        ref = await _file(store)
        write_task = asyncio.create_task(materializer.materialize(store, {"main.py": ref}))
        assert await asyncio.to_thread(started.wait, 5)
        close_task = asyncio.create_task(materializer.close())
        await asyncio.sleep(0)
        assert not close_task.done()
        release.set()
        with pytest.raises(AIError) as error:
            await write_task
        assert error.value.code is ErrorCode.STORAGE_CLOSED
        await close_task
        assert tuple(parent.iterdir()) == ()
    finally:
        release.set()
        await materializer.close()
        await store.close()


@pytest.mark.asyncio
async def test_cancelled_close_finishes_removal_and_is_repeatable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = await _store(tmp_path)
    started = threading.Event()
    release = threading.Event()
    remove_tree = _materialization._remove_tree

    def blocked_remove(root: Path) -> None:
        started.set()
        assert release.wait(10)
        remove_tree(root)

    monkeypatch.setattr(_materialization, "_remove_tree", blocked_remove)
    materializer = AssetMaterializer()
    try:
        result = await materializer.materialize(store, {})
        task = asyncio.create_task(materializer.close())
        assert await asyncio.to_thread(started.wait, 5)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        assert result.root.is_dir()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not result.root.exists()
        await materializer.close()
    finally:
        release.set()
        await materializer.close()
        await store.close()


@pytest.mark.asyncio
async def test_failed_cleanup_preserves_primary_and_can_be_retried(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = await _store(tmp_path)
    materializer = AssetMaterializer()
    remove_tree = _materialization.shutil.rmtree

    def failed_remove(root: Path) -> None:
        raise PermissionError("remove failed")

    primary = RuntimeError("consumer failed")
    try:
        with pytest.raises(RuntimeError) as error:
            async with materializer:
                result = await materializer.materialize(store, {})
                monkeypatch.setattr(_materialization.shutil, "rmtree", failed_remove)
                raise primary
        assert error.value is primary
        assert isinstance(error.value.__cause__, AIError)
        assert error.value.__cause__.code is ErrorCode.STORAGE_UNAVAILABLE
        assert result.root.is_dir()
        monkeypatch.setattr(_materialization.shutil, "rmtree", remove_tree)
        await materializer.close()
        assert not result.root.exists()
    finally:
        monkeypatch.setattr(_materialization.shutil, "rmtree", remove_tree)
        await materializer.close()
        await store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ("skill", "mcp"))
@pytest.mark.parametrize("backend_name", ("memory", "directory"))
async def test_capability_capture_rejects_unmaterializable_resource_paths(
    tmp_path: Path, kind: str, backend_name: str,
) -> None:
    declaration_name, declaration = (
        ("SKILL.md", b"---\nname: review\ndescription: Review files\n---\nReview.")
        if kind == "skill"
        else ("mcp.json", b'{"command":"python"}')
    )
    files = {
        AssetKey(kind, f"review/{declaration_name}"): declaration,
        AssetKey(kind, "review/run:dev.py"): b"print('ready')",
    }
    if backend_name == "directory":
        if os.name == "nt":
            pytest.skip("Windows cannot create a colon-containing fixture filename")
        for key, value in files.items():
            path = tmp_path / key.kind / key.id
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(value)
        backend = DirectoryAssetBackend(str(tmp_path), kinds=(kind,))
    else:
        backend = InMemoryAssetBackend()
        for key, value in files.items():
            await backend.put(key, value)
    store = AssetStore(StorageOverlay(backend))
    await store.initialize()
    try:
        with pytest.raises(AIError) as error:
            await CapabilityGroup("application", assets=store).capture()
        assert error.value.code is ErrorCode.CAPABILITY_RESOLUTION_INVALID
    finally:
        await store.close()
