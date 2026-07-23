#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared Depth-contract fixtures: one parametrized factory that yields a fresh
AssetWriterBackend for each of Memory, Filesystem, and SqlAlchemy so the same
contract cases run against all three backends."""

import asyncio
import threading

import pytest

from linktools.ai.asset.file import FileAssetBackend
from linktools.ai.asset.memory import MemoryAssetBackend
from linktools.ai.asset.path import AssetPath
from linktools.ai.asset.store import AssetStore


def _seed(store: AssetStore) -> None:
    """Seed the canonical Depth-contract tree.

    Each of these is a stored asset (including ``/a`` and ``/a/dir``, which act
    as both a target and a parent) so ZERO/ONE can return the target itself."""
    coro = _seed_coro(store)
    _run_in_new_loop(coro)


async def _seed_coro(store: AssetStore) -> None:
    for path in ("/a", "/a/file.txt", "/a/dir", "/a/dir/deep.txt", "/b.txt"):
        await store.put(AssetPath(path), b"x")


def _run_in_new_loop(coro):
    # The factory is invoked synchronously from inside a running pytest-asyncio
    # loop, so seeding must complete on a separate thread with its own loop.
    outcome: dict = {}

    def _runner():
        try:
            outcome["value"] = asyncio.run(coro)
        except BaseException as exc:  # noqa: BLE001 - re-raised on caller thread
            outcome["error"] = exc

    thread = threading.Thread(target=_runner)
    thread.start()
    thread.join()
    if "error" in outcome:
        raise outcome["error"]
    return outcome.get("value")


def _memory_backend(tmp_path):
    return MemoryAssetBackend()


def _file_backend(tmp_path, ident):
    return FileAssetBackend(root=tmp_path / f"file-{ident}")


def _sqlalchemy_backend(tmp_path, ident, engines):
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    from linktools.ai.storage.sqlalchemy.asset import SqlAlchemyAssetBackend
    from linktools.ai.storage.sqlalchemy.models import Base

    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path}/sql-{ident}.db")
    engines.append(engine)

    async def _create():
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        await engine.dispose()

    _run_in_new_loop(_create())
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    return SqlAlchemyAssetBackend(session_factory=session_factory)


def _dispose_engines(engines):
    # Dispose every SQL engine created during the test on a fresh loop. The
    # pytest-asyncio test loop has already closed by teardown, so without this
    # aiosqlite's worker thread raises "Event loop is closed" warnings.
    if not engines:
        return

    async def _dispose_all():
        for engine in engines:
            await engine.dispose()

    _run_in_new_loop(_dispose_all())


@pytest.fixture(params=["memory", "filesystem", "sqlalchemy"])
def make_store(request, tmp_path):
    """Return a factory that builds a fresh seeded AssetStore for the current
    backend param. Each call yields an independent store so a case can build
    its own tree (e.g. the unstored-target case)."""
    backend = request.param
    counter = {"n": 0}
    engines: "list" = []

    def _factory():
        counter["n"] += 1
        if backend == "memory":
            primary = _memory_backend(tmp_path)
        elif backend == "filesystem":
            primary = _file_backend(tmp_path, counter["n"])
        elif backend == "sqlalchemy":
            primary = _sqlalchemy_backend(tmp_path, counter["n"], engines)
        else:  # pragma: no cover - param is constrained above
            raise AssertionError(f"unknown backend: {backend}")
        store = AssetStore(primary=primary)
        _seed(store)
        return store

    yield _factory
    _dispose_engines(engines)


@pytest.fixture(params=["memory", "filesystem", "sqlalchemy"])
def make_backend(request, tmp_path):
    """Return a factory that builds a fresh UNSEEDED backend for cases that
    drive raw_list directly (the cursor-pagination contract)."""
    backend = request.param
    counter = {"n": 0}
    engines: "list" = []

    def _factory():
        counter["n"] += 1
        if backend == "memory":
            return _memory_backend(tmp_path)
        if backend == "filesystem":
            return _file_backend(tmp_path, counter["n"])
        if backend == "sqlalchemy":
            return _sqlalchemy_backend(tmp_path, counter["n"], engines)
        raise AssertionError(f"unknown backend: {backend}")  # pragma: no cover

    yield _factory
    _dispose_engines(engines)
