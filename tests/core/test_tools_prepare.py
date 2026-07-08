# -*- coding: utf-8 -*-
"""Characterization tests for Tool.prepare (fix-plan §5.4 prerequisite).

These lock the CURRENT main-path install behaviour so the ToolInstaller
takeover can be verified against it. They exercise the real Tool/Tools code
against a fake, tmp-rooted environ and local archives (no network).
"""
import hashlib
import json
import logging
import os
import zipfile
from pathlib import Path

import pytest

from linktools._cache_store import CacheStore
from linktools._download import DownloadManager
from linktools.core._locks import LockManager
from linktools.core._tools import Tools, Tool
from linktools.types import MISSING


class _DummyConfig:
    """Stand-in for environ.wrap_config(): env overrides are never set in tests."""

    def get(self, key, type=None, default=None):
        return default


class FakeEnviron:
    """Minimal environ surface that Tool.prepare / Tool.config touch."""

    def __init__(self, tmp_path):
        self._root = Path(tmp_path)
        self.system = "linux"
        self.machine = "x86_64"
        self.version = "0.0.0"
        self.locks = LockManager(self._root / "locks")
        self._cache = None
        self._downloads = None

    @property
    def cache(self):
        if self._cache is None:
            self._cache = CacheStore(self._root / "cache.db")
        return self._cache

    @property
    def downloads(self):
        if self._downloads is None:
            self._downloads = DownloadManager(self)
        return self._downloads

    def get_data_path(self, *parts, create_parent=False):
        p = self._root / "data"
        if parts:
            p = p.joinpath(*[str(x) for x in parts])
        if create_parent:
            p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def get_temp_path(self, *parts, create_parent=False):
        p = self._root / "temp"
        if parts:
            p = p.joinpath(*[str(x) for x in parts])
        if create_parent:
            p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def get_logger(self, name):
        return logging.getLogger("test." + name)

    def wrap_config(self, namespace=MISSING, env_prefix=MISSING):
        return _DummyConfig()


@pytest.fixture
def env(tmp_path):
    return FakeEnviron(tmp_path)


def _zip(path, entries):
    with zipfile.ZipFile(path, "w") as z:
        for name, body in entries.items():
            z.writestr(name, body)
    return path


# --------------------------------------------------------------------------- #
# install: single-file download (JAR-style) + archive extract
# --------------------------------------------------------------------------- #

def test_prepare_installs_single_file_tool(env, tmp_path):
    jar = tmp_path / "mytool-1.0.jar"
    jar.write_bytes(b"jar-body")
    digest = hashlib.sha256(b"jar-body").hexdigest()
    tools = Tools(env, {"mytool": {
        "version": "1.0", "download_url": str(jar), "sha256": digest}})
    tool = tools["mytool"]
    assert not tool.exists
    tool.prepare()
    assert tool.exists                      # absolute_path now present
    assert Path(tool.absolute_path).read_bytes() == b"jar-body"
    # manifest + active pointer written
    assert os.path.exists(os.path.join(tool.root_path, "manifest.json"))
    active = env.get_data_path("tools", "mytool", "active.json")
    assert json.loads(active.read_text())["version"] == "1.0"


def test_prepare_extracts_archive_tool(env, tmp_path):
    archive = _zip(tmp_path / "pkg.zip", {"bin/run": "#!/bin/sh\necho hi",
                                          "lib/x.txt": "x"})
    tools = Tools(env, {"mytool": {
        "version": "2.0", "download_url": str(archive),
        "unpack_path": ".", "target_path": "bin/run"}})
    tool = tools["mytool"]
    tool.prepare()
    assert tool.exists
    assert (Path(tool.root_path) / "bin" / "run").exists()
    assert (Path(tool.root_path) / "lib" / "x.txt").exists()
    # manifest entrypoint is relative to the install root
    manifest = json.loads((Path(tool.root_path) / "manifest.json").read_text())
    assert manifest["entrypoint"] == "bin/run"


# --------------------------------------------------------------------------- #
# idempotency, dependency order, corrupt-root recovery
# --------------------------------------------------------------------------- #

def test_prepare_is_idempotent(env, tmp_path):
    jar = tmp_path / "t.jar"
    jar.write_bytes(b"body")
    tools = Tools(env, {"t": {"version": "1.0", "download_url": str(jar)}})
    tool = tools["t"]
    tool.prepare()
    root = tool.root_path
    first_mtime = os.path.getmtime(os.path.join(root, "manifest.json"))
    tool.prepare()                          # second prepare must not re-download
    assert os.path.getmtime(os.path.join(root, "manifest.json")) == first_mtime


def test_prepare_installs_dependencies_first(env, tmp_path, monkeypatch):
    order = []

    def fake_download(self, request, *args, **kwargs):
        order.append(request.url)
        Path(request.destination).parent.mkdir(parents=True, exist_ok=True)
        Path(request.destination).write_bytes(b"x")

    monkeypatch.setattr(DownloadManager, "download", fake_download)

    dep_jar = tmp_path / "dep.jar"
    dep_jar.write_bytes(b"dep")
    main_jar = tmp_path / "main.jar"
    main_jar.write_bytes(b"main")
    tools = Tools(env, {
        "dep": {"version": "1.0", "download_url": str(dep_jar)},
        "main": {"version": "1.0", "download_url": str(main_jar),
                 "depends_on": "dep"},
    })
    tools["main"].prepare()
    assert order[0] == str(dep_jar)         # dependency installed first
    assert order[1] == str(main_jar)


def test_prepare_recovers_when_root_exists_but_entry_missing(env, tmp_path):
    # Pre-create an incomplete root (no entry file) -> prepare must replace it
    # with a good install rather than crashing or leaving the bad one.
    jar = tmp_path / "t.jar"
    jar.write_bytes(b"body")
    tools = Tools(env, {"t": {"version": "1.0", "download_url": str(jar)}})
    tool = tools["t"]
    # materialise the root path with junk but no entry, so self.exists is False
    os.makedirs(tool.root_path, exist_ok=True)
    (Path(tool.root_path) / "junk").write_text("partial")
    tool.prepare()
    assert tool.exists                       # good install landed
    assert not (Path(tool.root_path) / "junk").exists()
