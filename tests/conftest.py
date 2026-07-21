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
import sys
from pathlib import Path

import pytest

# Make the standalone external_adapter package (AC-15 wheel-only proof, plan
# §4.11) importable repo-wide without an installed wheel: its src/ dir goes on
# sys.path here so test modules under tests/ai/ that import ``external_adapter``
# resolve it. When the package IS installed (isolated-venv wheel proof), this
# insert is a harmless no-op (the installed package is found first).
_EXTERNAL_ADAPTER_SRC = Path(__file__).parent / "external_adapter" / "src"
if str(_EXTERNAL_ADAPTER_SRC) not in sys.path:
    sys.path.insert(0, str(_EXTERNAL_ADAPTER_SRC))

# The storage conformance testkit (``linktools-ai/testing/``) is test-support
# code, not library code -- it is never packaged into the linktools-ai wheel.
# Putting the linktools-ai/ package root on sys.path makes it importable as
# ``testing`` repo-wide without shipping it in the wheel.
_LINKTOOLS_AI_ROOT = Path(__file__).parent.parent / "linktools-ai"
if str(_LINKTOOLS_AI_ROOT) not in sys.path:
    sys.path.insert(0, str(_LINKTOOLS_AI_ROOT))


@pytest.fixture(autouse=True)
def _close_leaked_cache_stores_after_each_test():
    yield
    from linktools.core import CacheStore

    for obj in gc.get_objects():
        if isinstance(obj, CacheStore):
            obj.close()
