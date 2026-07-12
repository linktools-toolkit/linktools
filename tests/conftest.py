#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Root pytest configuration shared by every test package.

``CacheStore`` (used by nearly every ``Environ``/``ContainerManager``
construction in this suite) opens a real ``sqlite3.Connection`` eagerly in
``__init__``. That connection is not reliably reclaimed by reference
counting OR by a plain ``gc.collect()`` between tests -- confirmed by
direct instrumentation: a fixture returning a bare, unclosed ``CacheStore``
leaks 3 file descriptors (db/wal/shm) per test, monotonically, for the life
of a pytest session, regardless of whether anything still holds a Python
reference to it. A session covering the hundreds of tests in this suite
eventually exceeds ``select()``'s ``FD_SETSIZE`` limit, failing unrelated,
otherwise-correct tests (observed in
``tests/cntr/test_structured_command_runner.py``).

Explicitly calling ``CacheStore.close()`` *does* reliably release the
underlying connection's file descriptors immediately -- it just isn't
guaranteed to be called by every test/fixture that happens to construct
one (directly or via ``Environ``/``ContainerManager``). Rather than
retrofit explicit teardown into every such fixture across the suite, force
every live ``CacheStore`` closed after each test: closing is safe to call
at any time (the next access lazily reopens a fresh connection on that
thread), so this cannot break a still-in-use store, including a
session-scoped one.
"""
import gc

import pytest


@pytest.fixture(autouse=True)
def _close_leaked_cache_stores_after_each_test():
    yield
    from linktools.cache import CacheStore

    for obj in gc.get_objects():
        if isinstance(obj, CacheStore):
            obj.close()
