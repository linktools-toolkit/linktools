#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Filesystem-backed SpecLoader revision uses mtime_ns + path + size (not
int(st_mtime) of the max file), so a same-second modify/add/delete is detected
without sleeping. These tests run on real disk (tmp_path)."""

from pathlib import Path

import pytest

from linktools.ai.registry.agent import AgentRegistry
from linktools.ai.registry.parser import SpecLoader


def _write(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


@pytest.mark.asyncio
async def test_filesystem_revision_detects_same_second_modify(tmp_path):
    """Overwriting a file within the same second must change the revision --
    int(st_mtime) would miss this, mtime_ns catches it."""
    root = tmp_path / "agents"
    root.mkdir()
    _write(root / "a.md", "---\nname: a\n---\nv1\n")
    loader = SpecLoader.from_filesystem(root)
    rev1 = await loader.revision()

    _write(root / "a.md", "---\nname: a\n---\nv2\n")
    rev2 = await loader.revision()
    assert rev2 != rev1, "same-second modify must change the revision"


@pytest.mark.asyncio
async def test_filesystem_revision_detects_add_and_delete(tmp_path):
    """Adding then removing a file must change the revision each time."""
    root = tmp_path / "agents"
    root.mkdir()
    _write(root / "a.md", "x")
    loader = SpecLoader.from_filesystem(root)
    rev1 = await loader.revision()

    _write(root / "b.md", "y")
    rev2 = await loader.revision()
    assert rev2 != rev1, "adding a file must change the revision"

    (root / "b.md").unlink()
    rev3 = await loader.revision()
    assert rev3 != rev2, "deleting a file must change the revision"


@pytest.mark.asyncio
async def test_filesystem_registry_list_ids_refreshes_without_sleep(tmp_path):
    """An AgentRegistry over a filesystem root sees new/removed ids immediately
    (same second) -- the cache invalidates because the revision changed."""
    root = tmp_path / "agents"
    root.mkdir()
    _write(root / "a.md", "---\nname: a\n---\nbody\n")
    registry = AgentRegistry(SpecLoader.from_filesystem(root), suffix=".md")

    assert await registry.list_ids() == ("a",)

    _write(root / "b.md", "---\nname: b\n---\nbody\n")
    assert await registry.list_ids() == ("a", "b")

    (root / "a.md").unlink()
    assert await registry.list_ids() == ("b",)


@pytest.mark.asyncio
async def test_filesystem_revision_stable_when_unchanged(tmp_path):
    """An unchanged tree yields the same revision (no spurious invalidation)."""
    root = tmp_path / "agents"
    root.mkdir()
    _write(root / "a.md", "x")
    loader = SpecLoader.from_filesystem(root)
    assert await loader.revision() == await loader.revision()
