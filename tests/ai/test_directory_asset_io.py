#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Directory asset async I/O and cancellation boundaries."""

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
import linktools.ai.asset._directory as directory_module
import linktools.ai.capability._skill_source as skill_source_module
from linktools.ai.asset import (
    AssetKey,
    AssetRoot,
    AssetStore,
    AssetVersionRef,
    DirectoryAssetBackend,
    PrefixAssetPathAdapter,
)
from linktools.ai.capability import (
    AssetSkillSource,
    CapabilityGroup,
    SkillDefinition,
)
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.storage import StorageEntryRevision, StorageOverlay


@pytest.mark.asyncio
async def test_directory_asset_stat_hashes_content_off_event_loop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "agents").mkdir()
    (tmp_path / "agents" / "default.json").write_bytes(b"agent")
    backend = DirectoryAssetBackend(
        AssetRoot("file", str(tmp_path)),
        path_adapter=PrefixAssetPathAdapter({"agent": "agents"}),
        kinds=("agent",),
    )
    main_thread = threading.current_thread()
    observed_threads: list[threading.Thread] = []
    original_read = directory_module.read_bytes

    def tracked_read(path: Path) -> bytes:
        observed_threads.append(threading.current_thread())
        return original_read(path)

    monkeypatch.setattr(directory_module, "read_bytes", tracked_read)
    await backend.initialize()
    try:
        info = await backend.stat(AssetKey("agent", "default.json"))
        assert info is not None
        assert observed_threads
        assert all(thread is not main_thread for thread in observed_threads)
    finally:
        await backend.close()


@pytest.mark.asyncio
async def test_asset_skill_package_resolution_runs_off_event_loop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "assets"
    package = root / "skills" / "review"
    package.mkdir(parents=True)
    (package / "SKILL.md").write_text(
        "---\nname: review\ndescription: Review files\n---\n\nReview files.\n",
        encoding="utf-8",
    )
    (package / "run.sh").write_text("#!/bin/sh\n", encoding="utf-8")
    store = AssetStore(
        StorageOverlay(
            DirectoryAssetBackend(
                str(root),
                path_adapter=PrefixAssetPathAdapter({"skill": "skills"}),
                kinds=("skill",),
            )
        )
    )
    main_thread = threading.current_thread()
    observed_threads: list[threading.Thread] = []
    original_resolve = skill_source_module._resolve_local_skill_package

    def tracked_resolve(relatives, paths):
        observed_threads.append(threading.current_thread())
        return original_resolve(relatives, paths)

    monkeypatch.setattr(
        skill_source_module,
        "_resolve_local_skill_package",
        tracked_resolve,
    )
    await store.initialize()
    try:
        capture = await CapabilityGroup("application", assets=store).capture()
        definition = next(
            item.value
            for item in capture.contributions
            if item.kind == "skill"
        )
        assert isinstance(definition, SkillDefinition)
        assert definition.source_ref is not None
        source = AssetSkillSource("application", capture.asset_reader)
        view = await source.inspect(definition.source_ref)

        assert view.location.kind == "local"
        assert observed_threads
        assert all(thread is not main_thread for thread in observed_threads)
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_directory_asset_versions_ignore_revision_but_verify_content(
    tmp_path: Path,
) -> None:
    (tmp_path / "agents").mkdir()
    path = tmp_path / "agents" / "default.json"
    path.write_bytes(b"first")
    store = AssetStore(
        StorageOverlay(
            DirectoryAssetBackend(
                str(tmp_path),
                path_adapter=PrefixAssetPathAdapter({"agent": "agents"}),
                kinds=("agent",),
            )
        )
    )
    await store.initialize()
    try:
        key = AssetKey("agent", "default.json")
        ref = (await store.resolve_versions((key,)))[0]
        other_revision = AssetVersionRef(
            ref.key,
            ref.layer_id,
            StorageEntryRevision(ref.revision.value + 1),
            ref.etag,
            ref.size,
        )
        assert await store.read_versions((other_revision,)) == (b"first",)

        path.write_bytes(b"changed")
        with pytest.raises(AIError) as error:
            await store.read_versions((other_revision,))
        assert error.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_cancelled_stat_cannot_repopulate_a_closed_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "agent").mkdir()
    (tmp_path / "agent" / "default.json").write_bytes(b"agent")
    backend = DirectoryAssetBackend(str(tmp_path), kinds=("agent",))
    loop = asyncio.get_running_loop()
    started = asyncio.Event()
    release = threading.Event()
    original_read = directory_module.read_bytes

    def delayed_read(path: Path) -> bytes:
        loop.call_soon_threadsafe(started.set)
        assert release.wait(5)
        return original_read(path)

    monkeypatch.setattr(directory_module, "read_bytes", delayed_read)
    await backend.initialize()
    with ThreadPoolExecutor(max_workers=1) as executor:
        loop.set_default_executor(executor)
        task = asyncio.create_task(backend.stat(AssetKey("agent", "default.json")))
        try:
            await asyncio.wait_for(started.wait(), 2)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            await backend.close()
            release.set()
            # A single-worker barrier observes the entire abandoned read job.
            await loop.run_in_executor(executor, lambda: None)
            assert backend._entries == {}
        finally:
            release.set()
            await asyncio.gather(task, return_exceptions=True)
            await backend.close()
