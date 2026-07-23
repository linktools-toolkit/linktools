#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SQLAlchemy adapter boundary guards.

The plan (Storage/Asset/Artifact section, WP2 dialect layer) fixes hard rules
for the in-repo SqlAlchemyStorageAdapter:

1. core receives NO database URL / DSN;
2. core imports NO dialect driver (psycopg / asyncpg / mysql / mssql / ...);
3. dialect-name branching lives ONLY in the isolated ``storage/sqlalchemy/
   dialects/`` strategy package -- the adapter facade and every store stay
   dialect-neutral and delegate conflict classification there;
4. SQLite, MySQL, and PostgreSQL are the supported dialects; SQLite is the only
   one exercised in-repo (MySQL/PostgreSQL run in CI via
   ``LINKTOOLS_AI_TEST_MYSQL_DSN`` / ``LINKTOOLS_AI_TEST_POSTGRESQL_DSN``).

These must hold mechanically, not by goodwill -- a future change that adds
``import asyncpg`` or a ``create_engine(url)`` to the core adapter, or scatters
dialect-name ``if``s outside the strategy package, would silently re-couple
core to a vendor. This module grep-asserts the boundary so the regression
cannot slip in.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[3]
# The boundary rules apply to ALL of core, not just the
# sqlalchemy/ subpackage: a regression that adds `import asyncpg` to
# runtime/builder.py or storage/facade.py would otherwise slip past. The
# constructor-shape test below still targets SqlAlchemyStorage specifically.
_AI = _REPO / "linktools-ai" / "src" / "linktools" / "ai"

# SQLAlchemy's own optional driver packages. ``aiosqlite`` is the ONE allowed
# in-repo integration dialect (SQLite); everything else is a vendor driver that
# must stay out of core.
_FORBIDDEN_DRIVER_PACKAGES = {
    "psycopg2",
    "psycopg",
    "asyncpg",
    "aiomysql",
    "mysql",
    "pymysql",
    "aiopg",
    "cx_Oracle",
    "oracledb",
    "pyodbc",
    "snowflake",
    "bigquery",
}


def _core_py_files() -> "list[Path]":
    return [p for p in _AI.rglob("*.py") if "__pycache__" not in p.parts]


def _imported_top_level_modules(path: Path) -> "set[str]":
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (SyntaxError, OSError, UnicodeDecodeError):
        return set()
    roots: "set[str]" = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                roots.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            # Relative imports (level > 0) are intra-package -- a local module
            # named e.g. ``mysql.py`` under ``dialects/`` is NOT the PyPI mysql
            # driver. Only absolute imports (level == 0) can reach a vendor
            # driver package.
            if node.module and node.level == 0:
                roots.add(node.module.split(".")[0])
    return roots


def test_adapter_imports_no_dialect_driver() -> None:
    """No core source may import a dialect driver package."""
    offenders: "list[str]" = []
    for path in _core_py_files():
        roots = _imported_top_level_modules(path)
        bad = roots & _FORBIDDEN_DRIVER_PACKAGES
        if bad:
            offenders.append(f"{path.relative_to(_REPO)}: imports {sorted(bad)}")
    assert not offenders, "vendor dialect drivers in core adapter:\n  " + "\n  ".join(
        offenders
    )


def test_adapter_parses_no_dsn_or_engine_url() -> None:
    """No core source may parse a database URL/DSN or build an
    engine from one -- the adapter receives a downstream-built session_factory.

    The ONE exemption is the SQLite reference helper
    (``storage/sqlite/facade.py``): explicitly allows the SQLite
    helper to construct an engine ('SQLite helper 可以构造 engine'), and SQLite
    is the only in-repo integration dialect. Every other core module is
    forbidden from calling ``create_engine`` / ``create_async_engine`` /
    ``make_url`` / ``URL.create``."""
    # Calls/attribute references that indicate URL/DSN handling. ``create_engine``
    # (sync) is forbidden outright; ``create_async_engine`` / ``make_url`` /
    # ``engine.url`` indicate the core is parsing a connection string.
    forbidden_calls = {
        "create_engine",
        "create_async_engine",
        "make_url",
        "URL.create",
    }
    offenders: "list[str]" = []
    for path in _core_py_files():
        # The SQLite reference helper is the single core site allowed to
        # construct an engine. Skip it; everywhere else the ban
        # holds.
        if "storage" in path.parts and "sqlite" in path.parts:
            continue
        text = path.read_text(encoding="utf-8")
        for call in forbidden_calls:
            if re.search(rf"\b{re.escape(call)}\s*\(", text):
                offenders.append(f"{path.relative_to(_REPO)}: calls {call}")
    assert not offenders, "DSN/engine construction in core adapter:\n  " + "\n  ".join(
        offenders
    )


def test_adapter_has_no_dialect_name_branching() -> None:
    """No core source OUTSIDE the dialect strategy package may branch on a
    dialect name. Portable SQLAlchemy API (with_for_update(skip_locked=True),
    read-check-mutate upsert) is allowed; ``dialect.name == 'sqlite'`` /
    vendor-conditional SQL is allowed ONLY inside ``storage/sqlalchemy/dialects/``,
    the isolated strategy layer that resolves the dialect once at adapter
    construction."""
    forbidden = [
        r"\.dialect\.name\b",
        r"==\s*['\"]sqlite['\"]",
        r"==\s*['\"]postgresql['\"]",
        r"==\s*['\"]postgres['\"]",
        r"==\s*['\"]mysql['\"]",
        r"dialect\s*==",
    ]
    offenders: "list[str]" = []
    for path in _core_py_files():
        # The dialect strategy package is the one core site that legitimately
        # branches on a dialect name (it maps sqlite/mysql/postgresql to a
        # classifier). Everywhere else the ban still holds.
        if "sqlalchemy" in path.parts and "dialects" in path.parts:
            continue
        text = path.read_text(encoding="utf-8")
        for pat in forbidden:
            m = re.search(pat, text)
            if m:
                offenders.append(f"{path.relative_to(_REPO)}: matches {pat!r}")
    assert not offenders, (
        "dialect-name branching outside the dialects/ strategy package:\n  "
        + "\n  ".join(offenders)
    )


def test_sqlalchemy_storage_constructor_takes_session_factory_not_url() -> None:
    """SqlAlchemyStorage.__init__ must accept a session_factory and must NOT
    accept a url/dsn/engine argument."""
    import inspect

    from linktools.ai.storage.sqlalchemy.facade import SqlAlchemyStorage

    sig = inspect.signature(SqlAlchemyStorage.__init__)
    params = set(sig.parameters) - {"self"}
    assert "session_factory" in params, (
        "SqlAlchemyStorage must take an external session_factory, not build one"
    )
    assert not (params & {"url", "dsn", "engine", "connection_string"}), (
        f"SqlAlchemyStorage must not take a url/dsn/engine; found: {sorted(params & {'url', 'dsn', 'engine', 'connection_string'})}"
    )


def test_sqlalchemy_storage_adapter_has_frozen_constructor() -> None:
    """freezes the generic SqlAlchemyStorageAdapter constructor: it
    takes session_factory + the caller's artifact_blobs + coordination +
    features (and must NOT take a url/dsn/engine). This is the positive contract
    a downstream composes against -- the generic adapter must accept injected
    blob/coordination/features, not construct them internally."""
    import inspect

    from linktools.ai.storage.sqlalchemy.facade import SqlAlchemyStorageAdapter

    sig = inspect.signature(SqlAlchemyStorageAdapter.__init__)
    params = set(sig.parameters) - {"self"}
    required = {"session_factory", "artifact_blobs", "coordination", "features"}
    assert required <= params, (
        f"SqlAlchemyStorageAdapter must take the frozen §4.7 params; missing: "
        f"{sorted(required - params)}"
    )
    assert not (params & {"url", "dsn", "engine", "connection_string"}), (
        f"SqlAlchemyStorageAdapter must not take a url/dsn/engine; found: "
        f"{sorted(params & {'url', 'dsn', 'engine', 'connection_string'})}"
    )
