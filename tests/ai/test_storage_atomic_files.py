#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Atomic file publication preserves bytes and reports durable uncertainty."""

import errno
from pathlib import Path

import pytest

import linktools.ai.storage._files as files_module
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.storage import read_json, write_bytes_atomic, write_json_atomic


@pytest.mark.parametrize("failure", ("replace", "file_sync", "directory_sync"))
def test_atomic_write_failure_preserves_publication_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    path = tmp_path / "value"
    path.write_bytes(b"previous")
    error = OSError(errno.EIO, "publication failed")

    def fail(*_args: object) -> None:
        raise error

    if failure == "replace":
        monkeypatch.setattr(files_module.os, "replace", fail)
    elif failure == "file_sync":
        monkeypatch.setattr(files_module.os, "fsync", fail)
    else:
        monkeypatch.setattr(files_module, "sync_directory", fail)

    if failure == "directory_sync":
        with pytest.raises(AIError) as raised:
            write_bytes_atomic(path, b"published", fsync=True)
        assert raised.value.code is ErrorCode.STORAGE_RECOVERY_REQUIRED
        assert raised.value.__cause__ is error
        assert path.read_bytes() == b"published"
    else:
        with pytest.raises(OSError) as raised:
            write_bytes_atomic(path, b"published", fsync=True)
        assert raised.value is error
        assert path.read_bytes() == b"previous"
    assert not tuple(tmp_path.glob(".value.*"))


def test_atomic_write_defaults_to_unsynced_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def reject_sync(*_args: object) -> None:
        raise AssertionError("default publication does not request fsync")

    monkeypatch.setattr(files_module.os, "fsync", reject_sync)
    monkeypatch.setattr(files_module, "sync_directory", reject_sync)
    path = tmp_path / "nested" / "value"
    write_bytes_atomic(path, b"\x00\xffvalue")
    assert path.read_bytes() == b"\x00\xffvalue"


@pytest.mark.parametrize("fsync", (False, True))
def test_atomic_json_preserves_sorted_compact_utf8_bytes(tmp_path: Path, fsync: bool) -> None:
    path = tmp_path / "value.json"
    value = {"z": [None, True, 3, 1.25], "a": {"z": "\u4e2d", "a": False}}
    write_json_atomic(path, value, fsync=fsync)
    assert path.read_bytes() == '{"a":{"a":false,"z":"\u4e2d"},"z":[null,true,3,1.25]}'.encode("utf-8")
    assert read_json(path) == value
