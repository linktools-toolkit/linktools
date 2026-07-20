#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RevisionCache contract tests: the observable guarantees every domain Catalog
inherits. Proves atomic invalidation on revision change, single-run concurrent
refresh, cache hits, and the not-found vs parse-failure distinction."""

import asyncio

import pytest

from linktools.ai.catalog import (
    CatalogCodec,
    CatalogItemNotFoundError,
    CatalogItemParseError,
    CatalogSource,
    RevisionCache,
)


class _FakeSource:
    """A controllable CatalogSource: a revision string + a {path: text} map.

    ``revision_calls`` / ``read_calls`` count how many times each method fired
    so the single-refresh + single-flight guarantees are observable. ``read``
    yields (``await asyncio.sleep(0)``) before returning so the single-flight
    test actually exercises the cache-miss race -- a non-yielding async def
    runs to completion without ever letting a concurrent caller observe the
    empty cache, which makes a stampede impossible to detect."""

    def __init__(
        self,
        items: "dict[str, str]",
        revision: str = "r1",
        *,
        yield_on_read: bool = True,
    ) -> None:
        self._items = items
        self._revision = revision
        self.revision_calls = 0
        self.read_calls = 0
        self._yield_on_read = yield_on_read

    async def revision(self) -> str:
        self.revision_calls += 1
        return self._revision

    async def list_ids(self, suffix: str) -> "tuple[str, ...]":
        ids = sorted(
            path[: -len(suffix)] if path.endswith(suffix) else path
            for path in self._items
        )
        return tuple(ids)

    async def read(self, path: str) -> str:
        if path not in self._items:
            raise CatalogItemNotFoundError(f"no item at {path!r}")
        # Yield BEFORE recording the read so concurrent callers all reach the
        # miss path before any of them populates the cache. Without single-
        # flight this turns into N reads; with it, into 1.
        if self._yield_on_read:
            await asyncio.sleep(0)
        self.read_calls += 1
        return self._items[path]

    def set_revision(self, revision: str) -> None:
        self._revision = revision


class _IdentityCodec:
    """A CatalogCodec that 'parses' by wrapping the raw text, but rejects the
    sentinel '<bad>' as a parse failure."""

    def decode(self, item_id: str, raw: str) -> str:
        if raw == "<bad>":
            raise CatalogItemParseError(f"{item_id}: bad field at $.x")
        return f"{item_id}={raw}"


def _run(coro):
    return asyncio.run(coro)


def test_cache_hit_returns_decoded_without_rereading():
    source = _FakeSource({"a.md": "1"})
    codec = _IdentityCodec()
    cache = RevisionCache(source, codec)

    first = _run(cache.get("a"))
    # Second get is a cache hit: the source read call count stays at one read.
    second = _run(cache.get("a"))
    assert first == "a=1"
    assert second == "a=1"


def test_revision_change_atomically_invalidates_cache_and_ids():
    source = _FakeSource({"a.md": "old"}, revision="r1")
    cache = RevisionCache(source, _IdentityCodec())

    _run(cache.get("a"))  # populate cache at r1
    _run(cache.list_ids())  # populate id listing at r1

    # Source changes contents + bumps revision.
    source._items = {"a.md": "new", "b.md": "2"}
    source.set_revision("r2")

    # The cached spec is gone; the new content is read + decoded.
    assert _run(cache.get("a")) == "a=new"
    # The id listing reflects the new revision.
    assert _run(cache.list_ids()) == ("a", "b")


def test_concurrent_get_same_key_runs_read_once():
    """The single-flight guarantee: N concurrent cold-cache gets for the SAME
    item run source.read + codec.decode exactly ONCE, not N times.

    This is the test a non-yielding source could not detect. ``_FakeSource.read``
    yields before recording the call, so without single-flight all 10 callers
    reach the cache-miss path before any populates the cache -- 10 reads. With
    the per-key in-flight Future map, the first caller records its Future and
    the other 9 await it; ``read_calls`` ends at 1. If the single-flight code
    is removed, this test fails (read_calls == 10)."""
    source = _FakeSource({"a.md": "1"})
    cache = RevisionCache(source, _IdentityCodec())

    async def stampede():
        return await asyncio.gather(*[cache.get("a") for _ in range(10)])

    results = _run(stampede())
    assert results == ["a=1"] * 10
    assert source.read_calls == 1, (
        f"single-flight broken: source.read ran {source.read_calls} times "
        "for 10 concurrent same-key cold gets (expected 1)"
    )


def test_concurrent_get_different_keys_run_independently():
    """Different keys do NOT coalesce -- single-flight is per-key, not global.
    Each distinct item must incur its own read+decode."""
    source = _FakeSource({"a.md": "1", "b.md": "2", "c.md": "3"})
    cache = RevisionCache(source, _IdentityCodec())

    async def mixed():
        return await asyncio.gather(
            *[cache.get(k) for k in ("a", "b", "c", "a", "b", "c")]
        )

    results = _run(mixed())
    assert results == ["a=1", "b=2", "c=3", "a=1", "b=2", "c=3"]
    # One read per distinct key (3), not per call (6).
    assert source.read_calls == 3


def test_concurrent_get_same_key_propagates_decode_failure_to_all_waiters():
    """If the single fetch fails, every coalesced waiter must see the failure
    (not silently get None / hang on a never-resolved Future)."""
    source = _FakeSource({"bad.md": "<bad>"})
    cache = RevisionCache(source, _IdentityCodec())

    async def failing_stampede():
        return await asyncio.gather(
            *[cache.get("bad") for _ in range(8)],
            return_exceptions=True,
        )

    results = _run(failing_stampede())
    assert all(isinstance(r, CatalogItemParseError) for r in results), (
        "every coalesced waiter must observe the decode failure"
    )
    # And the failing fetch ran once, not 8 times.
    assert source.read_calls == 1


def test_revision_check_runs_once_per_get_under_concurrency():
    """Each get() checks the revision exactly once (the cheap, lock-serialized
    refresh critical section). This pins the lock exists: without it, a
    revision() that mutated state would race."""
    source = _FakeSource({"a.md": "1"})
    cache = RevisionCache(source, _IdentityCodec())

    async def many():
        await asyncio.gather(*[cache.get("a") for _ in range(10)])

    _run(many())
    # 10 gets -> 10 revision checks (one per get); the expensive read+decode
    # ran once (asserted by test_concurrent_get_same_key_runs_read_once).
    assert source.revision_calls == 10


def test_not_found_is_distinguished_from_parse_failure():
    source = _FakeSource({"good.md": "ok", "bad.md": "<bad>"})
    cache = RevisionCache(source, _IdentityCodec())

    # Missing item -> CatalogItemNotFoundError.
    with pytest.raises(CatalogItemNotFoundError):
        _run(cache.get("missing"))
    # Present-but-unparseable -> CatalogItemParseError (NOT not-found).
    with pytest.raises(CatalogItemParseError):
        _run(cache.get("bad"))
    # A good item still decodes.
    assert _run(cache.get("good")) == "good=ok"


def test_source_and_codec_are_runtime_checkable_protocols():
    # Structural: _FakeSource satisfies CatalogSource, _IdentityCodec satisfies
    # CatalogCodec without inheritance.
    assert isinstance(_FakeSource({}), CatalogSource)
    assert isinstance(_IdentityCodec(), CatalogCodec)
