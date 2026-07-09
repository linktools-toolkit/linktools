# -*- coding: utf-8 -*-
"""Tests for the safe archive extractor (spec §10.7 TOL-007)."""
import io
import os
import tarfile
import zipfile

import pytest

from linktools.utils import safe_extract
from linktools.errors import ToolArchiveError


def _make_zip(tmp_path, entries):
    zpath = tmp_path / "a.zip"
    with zipfile.ZipFile(zpath, "w") as z:
        for name, body in entries.items():
            z.writestr(name, body)
    return zpath


def _make_tar(tmp_path, entries, symlinks=None):
    tpath = tmp_path / "a.tar"
    with tarfile.open(tpath, "w") as t:
        for name, body in entries.items():
            info = tarfile.TarInfo(name=name)
            data = body.encode() if isinstance(body, str) else body
            info.size = len(data)
            t.addfile(info, io.BytesIO(data))
        for name, target in (symlinks or {}).items():
            info = tarfile.TarInfo(name=name)
            info.type = tarfile.SYMTYPE
            info.linkname = target
            info.size = 0
            t.addfile(info)
    return tpath


def test_zip_extracts_clean(tmp_path):
    z = _make_zip(tmp_path, {"dir/a.txt": "hello", "b.txt": "world"})
    dest = tmp_path / "out"
    safe_extract(z, dest)
    assert (dest / "dir" / "a.txt").read_text() == "hello"
    assert (dest / "b.txt").read_text() == "world"


def test_tar_extracts_clean(tmp_path):
    t = _make_tar(tmp_path, {"x.txt": "abc"})
    dest = tmp_path / "out"
    safe_extract(t, dest)
    assert (dest / "x.txt").read_text() == "abc"


def test_rejects_path_traversal_zip(tmp_path):
    z = _make_zip(tmp_path, {"../escape.txt": "evil"})
    dest = tmp_path / "out"
    with pytest.raises(ToolArchiveError):
        safe_extract(z, dest)
    assert not (tmp_path / "escape.txt").exists()


def test_rejects_absolute_path_zip(tmp_path):
    # zipfile rejects absolute names on write; craft via writestr with abs path.
    zpath = tmp_path / "abs.zip"
    with zipfile.ZipFile(zpath, "w") as z:
        z.writestr("/etc/evil.txt", "x")
    with pytest.raises(ToolArchiveError):
        safe_extract(zpath, tmp_path / "out")


def test_rejects_symlink_in_tar(tmp_path):
    t = _make_tar(tmp_path, {"real.txt": "ok"}, symlinks={"link": "/etc/passwd"})
    dest = tmp_path / "out"
    with pytest.raises(ToolArchiveError):
        safe_extract(t, dest)
    assert not (dest / "link").exists()


def test_rejects_too_many_files(tmp_path):
    entries = {"f%d.txt" % i: "x" for i in range(20)}
    z = _make_zip(tmp_path, entries)
    with pytest.raises(ToolArchiveError):
        safe_extract(z, tmp_path / "out", max_files=10)


def test_rejects_oversize_single_file(tmp_path):
    z = _make_zip(tmp_path, {"big.txt": "x" * 5000})
    with pytest.raises(ToolArchiveError):
        safe_extract(z, tmp_path / "out", max_file_size=1024)


def test_rejects_zip_bomb_total_size(tmp_path):
    # many small files that exceed total cap
    entries = {"f%d.txt" % i: "x" * 1000 for i in range(20)}
    z = _make_zip(tmp_path, entries)
    with pytest.raises(ToolArchiveError):
        safe_extract(z, tmp_path / "out", max_total_size=5000)


def test_unknown_format_raises(tmp_path):
    p = tmp_path / "x.bin"
    p.write_bytes(b"not an archive")
    with pytest.raises(ToolArchiveError):
        safe_extract(p, tmp_path / "out")
