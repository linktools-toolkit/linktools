#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Directory asset async I/O and cancellation boundaries."""

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
import linktools.ai.asset._directory as directory_module
from linktools.ai.asset import (
    AssetKey,
    AssetRoot,
    DirectoryAssetBackend,
    PrefixAssetPathAdapter,
)


@pytest.mark.asyncio
async def test_directory_asset_stat_hashes_content_off_event_loop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "agents").mkdir()
    (tmp_path / "agents" / "default.json").write_bytes(b"agent")
    backend = DirectoryAssetBackend(
        AssetRoot("file:assets", "file", str(tmp_path), "assets"),
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
