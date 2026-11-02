#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Load recorded legacy business bytes without retaining a state writer."""

import hashlib
import json
import zipfile
from pathlib import Path, PurePosixPath


_FIXTURES = Path(__file__).parent / "fixtures" / "persistence"


def legacy_provenance() -> dict:
    return json.loads((_FIXTURES / "filesystem_core_v1.provenance.json").read_text())


def _extract(root: Path, archive: zipfile.ZipFile, name: str, relative: str, digest: str) -> None:
    path = PurePosixPath(relative)
    assert not path.is_absolute() and ".." not in path.parts and "\\" not in relative
    info = archive.getinfo(name)
    assert not info.is_dir() and (info.external_attr >> 16) & 0o170000 == 0o100000
    content = archive.read(info)
    assert hashlib.sha256(content).hexdigest() == digest
    destination = root.joinpath(*path.parts)
    assert root.resolve() in destination.resolve().parents
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(content)


def load_legacy_source(root: Path) -> Path:
    provenance = legacy_provenance()
    bundle = _FIXTURES / "filesystem_core_v1.zip"
    assert hashlib.sha256(bundle.read_bytes()).hexdigest() == provenance["zip_sha256"]
    root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(bundle) as archive:
        expected = provenance["business_file_sha256"]
        assert sorted(archive.namelist()) == sorted(expected)
        for name, digest in expected.items():
            _extract(root, archive, name, name, digest)
    return legacy_group_root(root)


def install_legacy_overlay(root: Path, name: str) -> None:
    provenance = json.loads((_FIXTURES / "filesystem_scope_overlays_v1.provenance.json").read_text())
    assert provenance["base_zip_sha256"] == legacy_provenance()["zip_sha256"]
    bundle = _FIXTURES / "filesystem_scope_overlays_v1.zip"
    assert hashlib.sha256(bundle.read_bytes()).hexdigest() == provenance["zip_sha256"]
    overlay = provenance["overlays"][name]
    assert overlay["deleted_files"] == []
    with zipfile.ZipFile(bundle) as archive:
        names = [value for value in archive.namelist() if value.startswith(name + "/")]
        assert sorted(names) == sorted(name + "/" + value for value in overlay["file_sha256"])
        for path, digest in overlay["file_sha256"].items():
            _extract(root, archive, name + "/" + path, path, digest)


def legacy_group_root(root: Path) -> Path:
    provenance = legacy_provenance()
    matches = [name for name in provenance["business_file_sha256"]
               if name.startswith(".state-groups/") and name.endswith("/manifest.json")]
    assert len(matches) == 1
    return (root / matches[0]).parent


def load_legacy_turn_source(root: Path, *, active: bool) -> dict:
    provenance = json.loads((_FIXTURES / "filesystem_turns_v1.provenance.json").read_text())
    bundle = _FIXTURES / "filesystem_turns_v1.zip"
    assert hashlib.sha256(bundle.read_bytes()).hexdigest() == provenance["zip_sha256"]
    name = "active" if active else "released"
    source = provenance["sources"][name]
    with zipfile.ZipFile(bundle) as archive:
        expected = {variant + "/" + path for variant, value in provenance["sources"].items()
                    for path in value["files"]}
        assert sorted(archive.namelist()) == sorted(expected)
        for path, digest in source["files"].items():
            _extract(root, archive, name + "/" + path, path, digest)
    return {**source, "clock": provenance["clock"]}
