#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import hashlib
import json
import re

import pytest

from linktools.core import backup_legacy_path


@pytest.mark.parametrize("kind", ["file", "directory", "file_symlink", "directory_symlink"])
def test_backup_preserves_content_and_records_source(tmp_path, kind):
    source = tmp_path / "legacy"
    destination = tmp_path / "config"
    target = tmp_path / "target" if "symlink" in kind else source
    if "directory" in kind:
        target.mkdir()
        (target / "nested").mkdir()
        (target / "nested" / "settings").write_bytes(b"original")
    else:
        target.write_bytes(b"original")
    if "symlink" in kind:
        source.symlink_to(target, target_is_directory="directory" in kind)

    backup_legacy_path(destination, source)

    assert not source.exists() and not source.is_symlink()
    directories = list((destination / "migrations").iterdir())
    assert len(directories) == 1
    backup_dir = directories[0]
    digest = hashlib.sha256(b"original").hexdigest()[:8]
    assert re.fullmatch(r"\d{8}T\d{6}Z-" + digest + r"-[0-9a-f]{8}", backup_dir.name)
    backup = backup_dir / source.name
    payload = backup / "nested" / "settings" if "directory" in kind else backup
    assert payload.read_bytes() == b"original"
    if "symlink" in kind:
        assert backup.is_symlink()
        assert target.exists()
    report = json.loads((backup_dir / "report.json").read_text())
    assert report == {"source": str(source), "backup": str(backup), "migrated_at": backup_dir.name.split("-")[0]}


def test_missing_backup_source_does_not_create_output(tmp_path):
    destination = tmp_path / "config"
    backup_legacy_path(destination, tmp_path / "missing")
    assert not destination.exists()


def test_repeated_backups_do_not_overwrite_previous_content(tmp_path):
    source = tmp_path / "legacy"
    destination = tmp_path / "config"
    for content in (b"first", b"second"):
        source.write_bytes(content)
        backup_legacy_path(destination, source)
    assert sorted(path.read_bytes() for path in destination.glob("migrations/*/legacy")) == [b"first", b"second"]


def test_unreadable_source_is_not_removed(tmp_path):
    source = tmp_path / "legacy"
    source.symlink_to(tmp_path / "missing")
    destination = tmp_path / "config"
    with pytest.raises(FileNotFoundError):
        backup_legacy_path(destination, source)
    assert source.is_symlink()
    assert not destination.exists()


def test_failed_move_preserves_source(tmp_path, monkeypatch):
    import shutil

    source = tmp_path / "legacy"
    source.write_bytes(b"original")
    error = OSError("cannot move")

    def fail_move(*args, **kwargs):
        raise error

    monkeypatch.setattr(shutil, "move", fail_move)
    with pytest.raises(OSError) as caught:
        backup_legacy_path(tmp_path / "config", source)
    assert caught.value is error
    assert source.read_bytes() == b"original"
