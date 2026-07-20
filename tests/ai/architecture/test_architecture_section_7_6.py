#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Architecture guards for the plan's §7.6 misc invariants that the §3.3
dependency-direction test (test_dependency_rules.py) does not already cover:

- the ``_runtime`` build kernel is imported at RUNTIME only by the composition
  root (``runtime.py``); TYPE_CHECKING imports are typing-only and excluded;
- domain packages never import the concrete storage backends
  (``storage.filesystem`` / ``storage.sqlalchemy``) -- they depend on the
  Storage Protocols, and a backend is injected at the composition root;
- the Protocol module ``storage.protocols`` depends on no concrete backend;
- the core dependency declarations carry no environment-specific storage SDK,
  database driver, or distributed-coordination client (those live in optional
  extras / downstream, never the installable core)."""

from __future__ import annotations

import ast
import tomllib
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[3]
_AI = _REPO / "linktools-ai" / "src" / "linktools" / "ai"
_PYPROJECT = _REPO / "linktools-ai" / "pyproject.toml"


def _is_type_checking_test(test: ast.expr) -> bool:
    return (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING") or (
        isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"
    )


def _resolved_imports(
    file_path: Path, *, exclude_type_checking: bool = False
) -> "set[str]":
    """Full resolved ``linktools.ai.*`` module paths imported by ``file_path``.

    Relative imports are resolved against the file's package. When
    ``exclude_type_checking`` is set, imports nested under an
    ``if TYPE_CHECKING:`` block are dropped (they are typing-only, not runtime
    dependency edges)."""
    try:
        tree = ast.parse(file_path.read_text(encoding="utf-8"))
    except (SyntaxError, OSError, UnicodeDecodeError):
        return set()
    tc_node_ids: "set[int]" = set()
    if exclude_type_checking:
        for node in ast.walk(tree):
            if isinstance(node, ast.If) and _is_type_checking_test(node.test):
                for child in ast.walk(node):
                    tc_node_ids.add(id(child))
    rel = file_path.relative_to(_AI).with_suffix("")
    is_init = rel.name == "__init__"
    if is_init:
        rel = rel.parent
    parts = ("linktools", "ai") + tuple(rel.parts)
    base_pkg = ".".join(parts) if is_init else ".".join(parts[:-1])

    roots: "set[str]" = set()
    for node in ast.walk(tree):
        if exclude_type_checking and id(node) in tc_node_ids:
            continue
        if isinstance(node, ast.Import):
            for alias in node.names:
                roots.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0:
                if node.module:
                    roots.add(node.module)
            else:
                base = base_pkg
                for _ in range(node.level - 1):
                    if "." in base:
                        base = base.rsplit(".", 1)[0]
                    else:
                        base = ""
                        break
                if node.module:
                    roots.add(f"{base}.{node.module}")
                else:
                    for alias in node.names:
                        roots.add(f"{base}.{alias.name}")
    return {r for r in roots if r.startswith("linktools.ai")}


def _src_py_files() -> "list[Path]":
    return [
        p
        for p in _AI.rglob("*.py")
        if "__pycache__" not in p.parts
    ]


def test_runtime_kernel_imported_only_by_composition_root() -> None:
    # The _runtime BUILD kernel (the composition root that assembles Runtime
    # components) is imported at RUNTIME only by runtime.py. Every other src
    # module uses the public Runtime facade. coordinator.py's TYPE_CHECKING
    # import of RuntimeComponents is typing-only and excluded. Other _runtime
    # submodules (lifecycle, inspection, dependencies) are separately shared
    # and not in scope for this §7.6 line -- it names the BUILDER specifically.
    offenders: "list[str]" = []
    for path in _src_py_files():
        rel_parts = path.relative_to(_AI).parts
        if "_runtime" in rel_parts:
            continue  # the kernel itself may import its own submodules
        if rel_parts == ("runtime.py",):
            continue  # the composition root
        for mod in _resolved_imports(path, exclude_type_checking=True):
            if mod == "linktools.ai._runtime.build" or mod.startswith(
                "linktools.ai._runtime.build."
            ):
                offenders.append(f"{path.relative_to(_REPO)}: {mod}")
                break
    assert not offenders, (
        "_runtime.build kernel imported at runtime outside the composition "
        "root (runtime.py):\n  " + "\n  ".join(sorted(offenders))
    )


def test_domains_do_not_import_concrete_storage_backends() -> None:
    # Domain packages depend on Storage Protocols, never on a concrete backend
    # (storage.filesystem / storage.sqlalchemy). A backend is injected at the
    # composition root. TYPE_CHECKING imports count too -- a concrete backend
    # under TYPE_CHECKING is still a hidden dependency.
    concrete = ("linktools.ai.storage.filesystem", "linktools.ai.storage.sqlalchemy")
    offenders: "list[str]" = []
    for path in _src_py_files():
        rel_parts = path.relative_to(_AI).parts
        # Only the storage/ package itself (its facade selects a backend) and
        # _runtime/ (the build kernel wires the commit coordinator) may name a
        # concrete backend. Every other package is a domain and must not.
        if rel_parts[0] in ("storage", "_runtime"):
            continue
        for mod in _resolved_imports(path):
            if mod.startswith(concrete):
                offenders.append(f"{path.relative_to(_REPO)}: {mod}")
                break
    assert not offenders, (
        "domain package imports a concrete storage backend:\n  "
        + "\n  ".join(sorted(offenders))
    )


def test_storage_protocols_does_not_import_concrete_backends() -> None:
    # The Protocol module is the public boundary a downstream implements; it
    # must depend on no concrete backend and no linktools concrete store.
    protocols = _AI / "storage" / "protocols.py"
    mods = _resolved_imports(protocols)
    backend_hits = {
        m
        for m in mods
        if m.startswith(
            ("linktools.ai.storage.filesystem", "linktools.ai.storage.sqlalchemy")
        )
    }
    assert not backend_hits, (
        f"storage.protocols imports concrete backends: {sorted(backend_hits)}"
    )


# Environment-specific dependencies that must NEVER appear in the installable
# core (they live in optional extras or downstream adapters). The plan §6.1/§7.6
# requires the core to stay free of storage SDKs, DB drivers, and distributed-
# coordination clients.
_ENV_SPECIFIC_SDK_PATTERNS = (
    "redis",
    "aioboto3",
    "boto3",
    "azure-storage",
    "azure-identity",
    "google-cloud-storage",
    "asyncpg",
    "aiomysql",
    "asyncmy",
    "psycopg",
    "celery",
    "kombu",
    "dramatiq",
    "huey",
    "rqlite",
    "etcd3",
    "kazoo",
    "pysyncobj",
)


def _core_dependency_names() -> "list[str]":
    data = tomllib.loads(_PYPROJECT.read_text(encoding="utf-8"))
    project = data.get("project", {})
    # The installable CORE deps are `dependencies` (NOT optional-dependencies,
    # which is where SQLAlchemy/aiosqlite legitimately live as extras).
    deps = project.get("dependencies", [])
    return [str(d).strip() for d in deps]


def test_core_dependencies_have_no_env_specific_sdk() -> None:
    # The installable core (`[project] dependencies`) carries no environment-
    # specific storage SDK, DB driver, or distributed-coordination client.
    # Those belong in optional extras (SQLAlchemy/aiosqlite) or downstream
    # adapters, never in `import linktools.ai`'s required footprint.
    deps = _core_dependency_names()
    hits: "list[str]" = []
    for dep in deps:
        # Normalize the PEP 508 spec to its bare distribution name (lower).
        name = dep.split(";")[0].split(">")[0].split("<")[0].split("=")[0]
        name = name.split("[")[0].strip().lower().replace("_", "-")
        for pat in _ENV_SPECIFIC_SDK_PATTERNS:
            if name == pat or name.startswith(pat + "-"):
                hits.append(dep)
                break
    assert not hits, (
        "environment-specific SDK/driver/coordination client in the installable "
        f"core dependencies: {hits}"
    )
