# -*- coding: utf-8 -*-
"""Tests for ToolInstaller (spec §10.6/§10.8/§10.9)."""
import hashlib
import json
import zipfile

import pytest

from linktools._cache_store import CacheStore
from linktools._download import DownloadManager
from linktools.core._locks import LockManager
from linktools.core._tools_installer import ToolInstaller, ToolInstallation
from linktools.core._tools_registry import ToolDefinition
from linktools.errors import ToolInstallError


@pytest.fixture
def installer(tmp_path):
    environ = type("E", (), {
        "locks": LockManager(tmp_path / "locks"),
        "cache": CacheStore(tmp_path / "cache.db"),
        "downloads": None,  # set below
    })()
    environ.downloads = DownloadManager(environ)
    return ToolInstaller(environ, tmp_path / "tools")


def _make_archive(tmp_path, entries, name="a.zip"):
    p = tmp_path / name
    with zipfile.ZipFile(p, "w") as z:
        for n, body in entries.items():
            z.writestr(n, body)
    return p


def test_install_extracts_and_writes_manifest(installer, tmp_path):
    archive = _make_archive(tmp_path, {"bin/run": "#!/bin/sh\necho hi", "lib/x.txt": "x"})
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    definition = ToolDefinition(name="mytool", version="1.2.3")
    inst = installer.install(definition, source_url=str(archive), sha256=digest)

    assert isinstance(inst, ToolInstallation)
    assert inst.version == "1.2.3"
    # extracted tree present under <name>/<version>/
    assert (inst.root / "bin" / "run").read_text().startswith("#!/bin/sh")
    # manifest (§10.8)
    manifest = json.loads((inst.root / "manifest.json").read_text())
    assert manifest["name"] == "mytool" and manifest["version"] == "1.2.3"
    assert manifest["sha256"] == digest
    assert "bin/run" in manifest["files"]
    # active pointer (§10.9)
    assert installer.active_version("mytool") == "1.2.3"
    # no staging leftover
    assert not list((tmp_path / "tools" / "mytool").glob("*.staging-*"))
    assert not list((tmp_path / "tools" / "mytool").glob(".staging-*"))


def test_install_reuses_already_installed(installer, tmp_path):
    archive = _make_archive(tmp_path, {"run": "x"})
    definition = ToolDefinition(name="t", version="1.0")
    first = installer.install(definition, source_url=str(archive))
    second = installer.install(definition, source_url=str(archive))
    assert first.root == second.root  # reused, not re-extracted


def test_install_hash_mismatch_raises_and_no_target(installer, tmp_path):
    archive = _make_archive(tmp_path, {"run": "x"})
    definition = ToolDefinition(name="bad", version="1.0")
    with pytest.raises(Exception):
        installer.install(definition, source_url=str(archive),
                          sha256="0" * 64)  # wrong hash
    # no half-installed version dir exposed
    assert not (tmp_path / "tools" / "bad" / "1.0").exists()


def test_remove_refuses_active(installer, tmp_path):
    archive = _make_archive(tmp_path, {"run": "x"})
    definition = ToolDefinition(name="t", version="1.0")
    installer.install(definition, source_url=str(archive))
    with pytest.raises(ToolInstallError):
        installer.remove("t", "1.0")  # active -> refused
    # force removes
    assert installer.remove("t", "1.0", force=True) is True
    assert not installer.is_installed("t", "1.0")


def test_multi_version_layout(installer, tmp_path):
    archive = _make_archive(tmp_path, {"run": "v1"})
    definition = ToolDefinition(name="mv", version="1.0")
    installer.install(definition, source_url=str(archive))
    archive2 = _make_archive(tmp_path, {"run": "v2"}, name="b.zip")
    definition2 = ToolDefinition(name="mv", version="2.0")
    installer.install(definition2, source_url=str(archive2))
    assert installer.is_installed("mv", "1.0") and installer.is_installed("mv", "2.0")
    # active is the last installed
    assert installer.active_version("mv") == "2.0"
    # old version still present (not clobbered)
    assert (tmp_path / "tools" / "mv" / "1.0" / "run").exists()
