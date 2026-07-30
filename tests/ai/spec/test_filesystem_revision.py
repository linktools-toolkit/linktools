#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Filesystem-backed SpecLoader content identity and SpecIndex refresh.

The loader exposes ``identity(raw)`` (SHA-256 of content) so a parsed cache can
key by content, not by Python ``hash()`` (randomized per process). These tests
run on real disk (tmp_path)."""

from pathlib import Path

import pytest

from linktools.ai.agent.index import AgentSpecIndex
from linktools.ai.spec.parsing import SpecLoader


def _write(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


@pytest.mark.asyncio
async def test_identity_is_sha256_of_content():
    loader = SpecLoader.from_filesystem(Path("."))
    assert loader.identity("v1") == loader.identity("v1")
    assert loader.identity("v1") != loader.identity("v2")


@pytest.mark.asyncio
async def test_filesystem_list_ids_reflects_current_files(tmp_path):
    root = tmp_path / "agents"
    root.mkdir()
    _write(root / "a.md", "---\nname: a\n---\nbody\n")
    loader = SpecLoader.from_filesystem(root)
    assert await loader.list_ids(".md") == ("a",)

    _write(root / "b.md", "---\nname: b\n---\nbody\n")
    assert await loader.list_ids(".md") == ("a", "b")

    (root / "a.md").unlink()
    assert await loader.list_ids(".md") == ("b",)


@pytest.mark.asyncio
async def test_filesystem_registry_sees_changes_immediately(tmp_path):
    """An AgentSpecIndex over a filesystem root sees new/removed ids and
    changed content immediately -- each get re-reads and re-identifies."""
    root = tmp_path / "agents"
    root.mkdir()
    _write(root / "a.md", "---\nname: a\n---\nbody\n")
    registry = AgentSpecIndex.from_specloader(
        SpecLoader.from_filesystem(root), suffix=".md"
    )

    assert await registry.list_ids() == ("a",)

    _write(root / "b.md", "---\nname: b\n---\nbody\n")
    assert await registry.list_ids() == ("a", "b")

    (root / "a.md").unlink()
    assert await registry.list_ids() == ("b",)


@pytest.mark.asyncio
async def test_filesystem_read_uses_thread_not_loop(tmp_path):
    # Smoke: reading a file works and returns its text.
    root = tmp_path / "agents"
    root.mkdir()
    _write(root / "a.md", "hello")
    loader = SpecLoader.from_filesystem(root)
    assert await loader.read("a.md") == "hello"
