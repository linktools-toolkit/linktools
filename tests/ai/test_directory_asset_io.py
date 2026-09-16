#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regression coverage for directory asset async I/O boundaries."""

import threading
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
