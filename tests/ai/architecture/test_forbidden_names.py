#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Forbidden runtime names -- the legacy class/module terms the refactor deleted
must stay deleted in both source tokens and file paths, and the legacy
``resource`` term must stay cleared everywhere except migration code (which is
allowed to reference the old on-disk directory and DB table names it renames).

The allowlist excludes: migration packages (``migrations/``), this
``architecture/`` directory (these guard tests legitimately mention every
forbidden term as the literal under test), and the wheel-conformance test
(which asserts old names are ABSENT from the built wheel, so it must list them).
"""

import re
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[3]
_SEARCH_ROOTS = (
    _REPO / "linktools-ai" / "src",
    _REPO / "tests" / "ai",
    _REPO / "examples",
)

FORBIDDEN_RUNTIME_TERMS = (
    "ResourceStore",
    "ResourceBackend",
    "ResourcePath",
    "ResourceInfo",
    "ModelGateway",
    "ModelRouter",
    "CapabilityAssembler",
    "ExecutionBackend",
    "KeywordMemoryIndex",
    "MemoryIndexStatus",
    "MemoryManager",
    "SwarmRunner",
    "SwarmTask",
)


def _is_allowlisted(path: Path) -> bool:
    parts = path.parts
    return (
        "__pycache__" in parts
        or "migrations" in parts
        or "architecture" in parts
        or path.name == "test_wheel_only_conformance.py"
    )


def _py_files():
    for root in _SEARCH_ROOTS:
        if not root.exists():
            continue
        for path in root.rglob("*.py"):
            if _is_allowlisted(path):
                continue
            yield path


@pytest.mark.parametrize("term", FORBIDDEN_RUNTIME_TERMS)
def test_forbidden_term_absent_from_source(term: str) -> None:
    rx = re.compile(rf"\b{re.escape(term)}\b")
    hits = []
    for path in _py_files():
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        if rx.search(text):
            hits.append(str(path.relative_to(_REPO)))
    assert not hits, f"forbidden term {term!r} found in source: {hits}"


@pytest.mark.parametrize("term", FORBIDDEN_RUNTIME_TERMS)
def test_forbidden_term_absent_from_file_paths(term: str) -> None:
    needle = term.lower()
    hits = []
    for path in _py_files():
        if needle in str(path.relative_to(_REPO)).lower():
            hits.append(str(path.relative_to(_REPO)))
    assert not hits, f"forbidden term {term!r} found in file path: {hits}"


def test_capability_resolver_assemble_method_absent() -> None:
    # CapabilityResolver.assemble was renamed to .resolve() -- the old method
    # name must not reappear as a call (".assemble(") anywhere in source.
    rx = re.compile(r"\.assemble\(")
    hits = []
    for path in _py_files():
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        if rx.search(text):
            hits.append(str(path.relative_to(_REPO)))
    assert not hits, f"forbidden '.assemble(' call found: {hits}"


def test_resource_term_absent_outside_migrations() -> None:
    rx = re.compile(r"\bResource[A-Za-z_]*\b|\bresource(s)?\b")
    hits = []
    for path in _py_files():
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        if rx.search(text):
            hits.append(str(path.relative_to(_REPO)))
    assert not hits, f"legacy 'resource' term found outside migrations: {hits}"
