#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SQLAlchemy driver-dependency boundary: core ships MySQL and PostgreSQL
dialect implementations (``storage/sqlalchemy/dialects/mysql.py`` and
``postgresql.py``, built purely from SQLAlchemy-core's own per-dialect SQL
construction helpers), but it declares zero environment-specific DB driver
dependency (``asyncmy``, ``aiomysql``, ``asyncpg``) in its manifests. A
deployment that wants to actually talk to MySQL or PostgreSQL brings its own
driver via the engine URL; the shipped core never bundles one."""

from __future__ import annotations

from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_AI_PKG_ROOT = _REPO_ROOT / "linktools-ai"
_REQUIREMENTS_YML = _AI_PKG_ROOT / "requirements.yml"
_PYPROJECT_TOML = _AI_PKG_ROOT / "pyproject.toml"

_FORBIDDEN_DRIVERS = ("asyncmy", "aiomysql", "asyncpg")


def test_dependency_manifests_have_no_vendor_drivers():
    """``requirements.yml`` / ``pyproject.toml`` declare no environment-specific
    DB driver dependency line."""
    hits: "list[str]" = []
    for manifest in (_REQUIREMENTS_YML, _PYPROJECT_TOML):
        text = manifest.read_text(encoding="utf-8")
        for word in _FORBIDDEN_DRIVERS:
            if word in text:
                hits.append(f"{manifest.relative_to(_REPO_ROOT)}: {word!r}")
    assert not hits, "vendor driver dependency found in manifests:\n" + "\n".join(hits)
