#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Parametrized object-store factory for the storage.object contract suite.

Every ObjectStore contract runs against each backend the storage kernel ships
(memory / filesystem / sqlite / sqlalchemy) by depending on the
``object_store_factory`` fixture. Backends are imported lazily; a not-yet-
implemented backend SKIPs (rather than erroring) so a contract runs against
every backend that exists today and silently gains coverage as each backend
lands -- no per-backend xfail churn."""

from __future__ import annotations

import pytest


def _make_memory():
    try:
        from linktools.ai.storage.backends.memory.object import MemoryObjectStore
    except ImportError:
        pytest.skip("memory backend not yet implemented")
    return MemoryObjectStore()


def _make_filesystem(tmp_path):
    try:
        from linktools.ai.storage.backends.filesystem.object import (
            FilesystemObjectStore,
        )
    except ImportError:
        pytest.skip("filesystem backend not yet implemented")
    return FilesystemObjectStore(root=tmp_path / "fs-object")


def _make_sqlite(tmp_path):
    try:
        from linktools.ai.storage.backends.sqlite.object import SqliteObjectStore
    except ImportError:
        pytest.skip("sqlite backend not yet implemented")
    return SqliteObjectStore(path=tmp_path / "sqlite-object.db")


def _make_sqlalchemy(tmp_path):
    try:
        from linktools.ai.storage.backends.sqlalchemy.object import (
            SqlAlchemyObjectStore,
        )
    except ImportError:
        pytest.skip("sqlalchemy backend not yet implemented")
    # Test-only engine construction: SqlAlchemyObjectStore itself takes only a
    # session_factory (core parses no DSN / builds no engine), so exercising
    # it against a concrete dialect means the TEST builds the engine here.
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'sa-object.db'}")
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    return SqlAlchemyObjectStore(session_factory=session_factory)


_BACKENDS = {
    "memory": _make_memory,
    "filesystem": _make_filesystem,
    "sqlite": _make_sqlite,
    "sqlalchemy": _make_sqlalchemy,
}


@pytest.fixture(params=sorted(_BACKENDS))
def object_store_factory(request, tmp_path):
    """Return a zero-arg factory that builds a fresh ObjectStore for the
    parametrized backend. Unimplemented backends skip; implemented ones run."""
    name = request.param
    factory = _BACKENDS[name]

    def _build():
        if name == "memory":
            return factory()
        return factory(tmp_path)

    return _build
