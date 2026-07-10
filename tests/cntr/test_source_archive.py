#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SourceContainer safe archive extraction (refactor spec §5.3).

``_handle_source_file`` must route through ``linktools.utils.safe_extract`` so a
malicious source archive cannot escape its destination directory (Zip Slip /
absolute paths). Normal archives must still extract cleanly.
"""
import zipfile

import pytest

from linktools.cntr.container import SourceContainer
from linktools.errors import ToolArchiveError


class _ConcreteSource(SourceContainer):
    """Minimal concrete SourceContainer so _handle_source_file is callable."""
    _source_url = "https://example.test/src.zip"
    _source_path = "."


def _make_source(manager, tmp_path):
    return _ConcreteSource(manager, str(tmp_path), name="tstsrc")


def test_normal_zip_extracts_clean(fresh_manager, tmp_path):
    archive = tmp_path / "ok.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("dir/a.txt", "hello")
        z.writestr("b.txt", "world")
    dest = tmp_path / "out"
    dest.mkdir()

    _make_source(fresh_manager, tmp_path)._handle_source_file(archive, dest)

    assert (dest / "dir" / "a.txt").read_text() == "hello"
    assert (dest / "b.txt").read_text() == "world"


def test_parent_traversal_is_rejected(fresh_manager, tmp_path):
    archive = tmp_path / "evil.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("../escape.txt", "evil")
    dest = tmp_path / "out"
    dest.mkdir()

    with pytest.raises(ToolArchiveError):
        _make_source(fresh_manager, tmp_path)._handle_source_file(archive, dest)

    # Nothing escaped above the destination.
    assert not (tmp_path / "escape.txt").exists()


def test_absolute_path_is_rejected(fresh_manager, tmp_path):
    archive = tmp_path / "abs.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("/etc/evil.txt", "x")
    dest = tmp_path / "out"
    dest.mkdir()

    with pytest.raises(ToolArchiveError):
        _make_source(fresh_manager, tmp_path)._handle_source_file(archive, dest)
