#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SQLAlchemy vendor-neutrality boundary: the core source tree, its declared
dependencies, and the built distribution artifacts must carry zero
vendor-specific DDL kwargs (``mysql_length``, ``mysql_*``, ``postgresql_*``)
and zero environment-specific DB driver dependency (``asyncmy``, ``aiomysql``,
``asyncpg``). A deployment that needs a specific vendor's DDL tuning or driver
brings its own SchemaProvider / extras; the shipped core never hardcodes
either."""

from __future__ import annotations

import re
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
_AI_PKG_ROOT = _REPO_ROOT / "linktools-ai"
_AI_SRC = _AI_PKG_ROOT / "src"
_REQUIREMENTS_YML = _AI_PKG_ROOT / "requirements.yml"
_PYPROJECT_TOML = _AI_PKG_ROOT / "pyproject.toml"

_THIS_FILE = Path(__file__).resolve()

_FORBIDDEN_WORDS = (
    "mysql_length",
    "mysql_",
    "postgresql_",
    "asyncmy",
    "aiomysql",
    "asyncpg",
)

_WORD_PATTERN = re.compile(
    "|".join(re.escape(w) for w in _FORBIDDEN_WORDS)
)


def _scan_text(label: str, text: str, hits: "list[str]") -> None:
    for match in _WORD_PATTERN.finditer(text):
        hits.append(f"{label}: {match.group(0)!r}")


def test_source_tree_has_no_vendor_kwargs_or_drivers():
    """``linktools-ai/src`` carries none of the forbidden markers."""
    hits: "list[str]" = []
    for path in sorted(_AI_SRC.rglob("*.py")):
        if path.resolve() == _THIS_FILE:
            continue
        if "__pycache__" in path.parts:
            continue
        text = path.read_text(encoding="utf-8")
        _scan_text(str(path.relative_to(_REPO_ROOT)), text, hits)
    assert not hits, "vendor-specific markers found in source:\n" + "\n".join(hits)


def test_dependency_manifests_have_no_vendor_drivers():
    """``requirements.yml`` / ``pyproject.toml`` declare no environment-specific
    DB driver and no vendor DDL kwarg reference outside of prose explaining
    the boundary itself (the module docstring in ``requirements.yml`` names
    ``mysql``/``postgres`` only as English words describing what is NOT
    bundled, never as an actual dependency line or kwarg)."""
    hits: "list[str]" = []
    for manifest in (_REQUIREMENTS_YML, _PYPROJECT_TOML):
        text = manifest.read_text(encoding="utf-8")
        for word in ("asyncmy", "aiomysql", "asyncpg", "mysql_length", "mysql_", "postgresql_"):
            if word in text:
                hits.append(f"{manifest.relative_to(_REPO_ROOT)}: {word!r}")
    assert not hits, "vendor driver / DDL kwarg reference found in manifests:\n" + "\n".join(hits)


def _build_wheel(tmp_path: Path) -> Path:
    dist_dir = tmp_path / "dist"
    subprocess.run(
        [sys.executable, "-m", "build", "--wheel", "--outdir", str(dist_dir)],
        cwd=_AI_PKG_ROOT,
        check=True,
        capture_output=True,
    )
    wheels = sorted(dist_dir.glob("*.whl"))
    assert wheels, "wheel build produced no .whl file"
    return wheels[-1]


def test_wheel_filename_metadata_and_source_have_no_vendor_markers(tmp_path):
    """The built wheel's filename, its ``METADATA``, and every ``.py`` file it
    ships carry none of the forbidden vendor markers."""
    wheel_path = _build_wheel(tmp_path)
    hits: "list[str]" = []
    _scan_text("wheel filename", wheel_path.name, hits)
    with zipfile.ZipFile(wheel_path) as zf:
        for name in zf.namelist():
            if name.endswith("METADATA"):
                _scan_text(f"wheel:{name}", zf.read(name).decode("utf-8"), hits)
            elif name.endswith(".py"):
                _scan_text(f"wheel:{name}", zf.read(name).decode("utf-8"), hits)
    assert not hits, "vendor-specific markers found in the built wheel:\n" + "\n".join(hits)
