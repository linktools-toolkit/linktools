#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Architecture gate: the storage package's single allowed dependency
direction.

``linktools.ai.storage`` is the lowest layer. It must NOT import any domain
package -- not asset / artifact / run / jobs / runtime / capability. Domains
depend on storage's narrow Protocols; storage never depends back. Today this
is violated broadly (the storage facade + the per-domain persistence files
import the domains they serve -- the root cause the storage-kernel spec
names), so this contract is ``xfail(strict=True)`` until the facade + those
per-domain stores move out of ``storage/`` into ``runtime/persistence/`` and
each domain's own ``persistence/`` package. The moment the last violation is
gone the test XPASSES, strict-xfail turns it red, and the mark must be
removed -- the boundary cannot be silently re-violated."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_AI_SRC = (
    Path(__file__).resolve().parents[4]
    / "linktools-ai"
    / "src"
    / "linktools"
    / "ai"
)
_STORAGE_SRC = _AI_SRC / "storage"

# The domain roots storage must not reach into (asset/artifact/run/jobs/
# runtime/capability). Storage owns only generic object/cache/blob/coordination
# machinery; it may never import a domain that consumes it.
_FORBIDDEN_DOMAINS = frozenset(
    {"asset", "artifact", "run", "jobs", "runtime", "capability"}
)


def _domains_in_absolute(name: str) -> "set[str]":
    """The linktools.ai.<domain> roots targeted by an absolute import name."""
    out: "set[str]" = set()
    if name.startswith("linktools.ai."):
        parts = name.split(".")
        if len(parts) > 2:
            out.add(parts[2])
    return out


def _imported_domain_roots(file_path: Path) -> "set[str]":
    """Every linktools.ai.<domain> root this file imports (absolute or
    relative). Relative imports are resolved against the file's package under
    ``linktools.ai``; the leading segment after that prefix is the domain."""
    try:
        tree = ast.parse(file_path.read_text(encoding="utf-8"))
    except (SyntaxError, OSError, UnicodeDecodeError):
        return set()
    pkg_parts = list(file_path.relative_to(_AI_SRC).parts[:-1])
    domains: "set[str]" = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                domains |= _domains_in_absolute(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0:
                domains |= _domains_in_absolute(node.module or "")
            else:
                base = pkg_parts[: len(pkg_parts) - (node.level - 1)]
                full = ".".join(base + ([node.module] if node.module else []))
                if full:
                    domains.add(full.split(".")[0])
    return domains


def _storage_python_files() -> "list[Path]":
    return [
        p
        for p in sorted(_STORAGE_SRC.rglob("*.py"))
        if "__pycache__" not in p.parts
    ]


def test_storage_imports_no_domain_package() -> None:
    """``linktools.ai.storage`` imports no asset/artifact/run/jobs/runtime/
    capability. Enumerate every violation so a fix is obvious."""
    violations: "dict[str, list[str]]" = {}
    for path in _storage_python_files():
        hits = sorted(_imported_domain_roots(path) & _FORBIDDEN_DOMAINS)
        if hits:
            violations[str(path.relative_to(_AI_SRC))] = hits
    assert not violations, (
        "storage/ imports forbidden domain packages (must depend only on its "
        "own subpackages + storage.object Protocols):\n"
        + "\n".join(f"  {f}: {d}" for f, d in sorted(violations.items()))
    )
