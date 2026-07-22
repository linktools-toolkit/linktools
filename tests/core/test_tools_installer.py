# -*- coding: utf-8 -*-
"""Tests for ToolInstaller."""
import hashlib
import json
import zipfile

import pytest

from linktools.core import CacheStore
from linktools.core import DownloadManager
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
    # manifest
    manifest = json.loads((inst.root / "manifest.json").read_text())
    assert manifest["name"] == "mytool" and manifest["version"] == "1.2.3"
    assert manifest["sha256"] == digest
    assert "bin/run" in manifest["files"]
    # active pointer
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


# --------------------------------------------------------------------------- #
# -5: manifest fields + state split
# --------------------------------------------------------------------------- #

def test_manifest_records_platform_arch_size_entrypoint(installer, tmp_path):
    archive = _make_archive(tmp_path, {"bin/run": "#!/bin/sh\necho hi"})
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    definition = ToolDefinition(name="t", version="1.0", entrypoint="bin/run")
    inst = installer.install(definition, source_url=str(archive), sha256=digest)
    m = inst.manifest
    assert m["entrypoint"] == "bin/run"
    assert m["platform"] and m["architecture"]
    assert m["size"] == len("#!/bin/sh\necho hi")


def test_install_missing_entrypoint_raises_no_target(installer, tmp_path):
    archive = _make_archive(tmp_path, {"bin/run": "x"})
    definition = ToolDefinition(name="t", version="1.0", entrypoint="bin/missing")
    with pytest.raises(ToolInstallError):
        installer.install(definition, source_url=str(archive))
    # entrypoint validated in staging -> no half-installed target exposed
    assert not (tmp_path / "tools" / "t" / "1.0").exists()


def test_version_complete_independent_of_active_pointer(installer, tmp_path):
    archive = _make_archive(tmp_path, {"run": "x"})
    definition = ToolDefinition(name="t", version="1.0")
    installer.install(definition, source_url=str(archive))
    # removing active.json must NOT make the version dir incomplete
    (tmp_path / "tools" / "t" / "active.json").unlink()
    assert installer.is_installation_complete("t", "1.0") is True
    assert installer.is_active_valid("t") is False  # no active pointer -> invalid


def test_active_invalid_when_points_at_corrupt_version(installer, tmp_path):
    archive = _make_archive(tmp_path, {"run": "x"})
    definition = ToolDefinition(name="t", version="1.0")
    installer.install(definition, source_url=str(archive))
    # delete a recorded file -> version no longer complete
    (tmp_path / "tools" / "t" / "1.0" / "run").unlink()
    assert installer.is_installation_complete("t", "1.0") is False
    assert installer.is_active_valid("t") is False
    with pytest.raises(ToolInstallError):
        installer.resolve_active("t")


def test_resolve_active_returns_installation(installer, tmp_path):
    archive = _make_archive(tmp_path, {"run": "x"})
    definition = ToolDefinition(name="t", version="1.0")
    installer.install(definition, source_url=str(archive))
    inst = installer.resolve_active("t")
    assert isinstance(inst, ToolInstallation)
    assert inst.name == "t" and inst.version == "1.0"


def test_corrupt_target_quarantined_then_reinstalled(installer, tmp_path):
    # pre-create an incomplete target (no manifest) that would block the move
    bad = tmp_path / "tools" / "t" / "1.0"
    bad.mkdir(parents=True)
    (bad / "junk").write_text("partial")
    archive = _make_archive(tmp_path, {"run": "x"})
    definition = ToolDefinition(name="t", version="1.0")
    inst = installer.install(definition, source_url=str(archive))
    assert (inst.root / "run").exists()           # fresh install succeeded
    assert (tmp_path / "tools" / ".corrupt").exists()  # corrupt moved aside
