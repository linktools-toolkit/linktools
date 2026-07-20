#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CatalogSource / CatalogCodec / RevisionCache: the generic, domain-agnostic
contracts a Catalog composes (plan §4.3).

- ``CatalogSource``: the raw byte/text origin. Exposes a revision string (any
  change means the cached specs are stale), an id listing, and a single-item
  read. The revision is a string so a source can encode a content hash, an
  mtime, a directory version, etc.
- ``CatalogCodec[T]``: strict decode of one item's raw text into a typed spec.
  Must reject unknown fields and surface the failing field path in any error.
- ``RevisionCache[T]``: the (item_id, revision)-keyed cache. Holds the cached
  specs + the id listing for the current revision; atomically drops both when
  the source revision changes; runs the revision check exactly once under
  concurrency. ``get`` distinguishes not-found from parse-failure and never
  conflates them.
"""

from __future__ import annotations

import asyncio
from typing import Any, Generic, Protocol, TypeVar, runtime_checkable

T = TypeVar("T")


class CatalogError(Exception):
    """Base for Catalog lookup/parse failures."""


class CatalogItemNotFoundError(CatalogError):
    """The item id is not present in the source. Distinct from a parse failure:
    the source has no bytes for this id at this revision."""


class CatalogItemParseError(CatalogError):
    """The item's raw text was read but the codec rejected it (unknown field,
    bad type, missing required field). Carries the item id, the source path,
    and (when the codec provides it) the offending field path."""


@runtime_checkable
class CatalogSource(Protocol):
    """The raw origin of catalog items."""

    async def revision(self) -> str:
        """A string that changes whenever the source's contents change. Used as
        part of the cache key; a changed value atomically invalidates the
        cache."""
        ...

    async def list_ids(self, suffix: str) -> "tuple[str, ...]":
        """All item ids at the current revision whose names end in ``suffix``."""
        ...

    async def read(self, path: str) -> str:
        """Read one item's raw text. Raise a not-found error when the item does
        not exist at this revision (distinct from a parse error).
        CatalogItemNotFoundError is the recommended type for new sources; a
        domain source may propagate its own not-found type for back-comat."""
        ...


@runtime_checkable
class CatalogCodec(Protocol, Generic[T]):
    """Strict decode of one item's raw text into a typed spec."""

    def decode(self, item_id: str, raw: str) -> T:
        """Parse ``raw`` for ``item_id``. Reject unknown fields; raise a parse
        error carrying the item id + field path on any failure.
        CatalogItemParseError is the recommended type for new codecs; a domain
        codec may raise its own rich parse-error type (the RevisionCache
        propagates whatever the codec raises)."""
        ...


class RevisionCache(Generic[T]):
    """A revision-keyed cache over a CatalogSource + CatalogCodec.

    Holds the decoded specs and the id listing for the CURRENT source revision.
    ``_ensure_fresh`` checks the source revision under a single asyncio lock so
    concurrent callers run the check once and see a consistent cache; a changed
    revision clears both the spec cache and the id listing atomically.
    """

    def __init__(
        self,
        source: CatalogSource,
        codec: "CatalogCodec[T]",
        *,
        suffix: str = ".md",
        source_name: "str | None" = None,
        metrics: "Any | None" = None,
    ) -> None:
        self._source = source
        self._codec = codec
        self._suffix = suffix
        self._source_name = source_name or type(source).__name__
        self._cache: "dict[tuple[str, str], T]" = {}
        self._cached_revision: "str | None" = None
        self._ids: "tuple[str, ...] | None" = None
        self._refresh_lock = asyncio.Lock()
        # Per-(id, revision) in-flight futures: single-flight the cache-miss
        # read+decode so N concurrent cold gets for the SAME item run the source
        # read + codec decode exactly once. Without this, a real (yielding)
        # source stampsedes -- N tasks all see an empty cache before any
        # populates it.
        self._inflight: "dict[tuple[str, str], asyncio.Future[T]]" = {}
        # Optional ObservabilityMetrics sink. When wired, every revision
        # change observed by _ensure_fresh increments
        # ``catalog_revision_refresh_total``. Default None keeps existing
        # callers no-op.
        self._metrics = metrics

    @property
    def source_name(self) -> str:
        return self._source_name

    async def _ensure_fresh(self) -> str:
        """Return the current revision, atomically invalidating the cache if the
        source revision changed. The lock guarantees the revision check + clear
        run as one critical section, so concurrent refresh callers see a single
        revision read and never race a half-cleared cache."""
        async with self._refresh_lock:
            revision = await self._source.revision()
            if revision != self._cached_revision:
                self._cache.clear()
                self._ids = None
                self._cached_revision = revision
                if self._metrics is not None:
                    # A refresh happened: the source revision moved since the
                    # last call. Counted as an observability signal (not a
                    # failure) per the production-hardening metric floor.
                    self._metrics.counter("catalog_revision_refresh_total")
            return revision

    async def list_ids(self) -> "tuple[str, ...]":
        await self._ensure_fresh()
        if self._ids is None:
            self._ids = await self._source.list_ids(self._suffix)
        return self._ids

    async def get(self, item_id: str) -> T:
        revision = await self._ensure_fresh()
        cache_key = (item_id, revision)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached
        # Single-flight the cache-miss fetch: if another task is already reading
        # + decoding this (id, revision), await its in-flight Future instead of
        # re-fetching. Different items still proceed concurrently; only same-key
        # misses coalesce. Proven by test_concurrent_get_same_key_runs_read_once
        # against a YIELDING source.
        existing = self._inflight.get(cache_key)
        if existing is not None:
            return await existing
        future = asyncio.get_running_loop().create_future()
        self._inflight[cache_key] = future
        try:
            path = f"{item_id}{self._suffix}"
            # The source owns the not-found error type (CatalogItemNotFoundError
            # for new sources; a domain may propagate its own not-found type for
            # back-comat). The codec owns the parse-error type. RevisionCache
            # propagates both AS-IS so a domain's rich error hierarchy is
            # preserved -- the caller sees the same errors the inlined registry
            # raised.
            raw = await self._source.read(path)
            spec = self._codec.decode(item_id, raw)
            self._cache[cache_key] = spec
            future.set_result(spec)
        except BaseException as exc:
            # Propagate the failure to any coalesced waiters. The creator does
            # not re-raise directly; it falls through to ``await future`` below,
            # which re-raises -- so the Future's exception is always retrieved
            # by at least one party and never triggers "Future exception was
            # never retrieved".
            if not future.done():
                future.set_exception(exc)
        finally:
            # Drop the in-flight entry before returning so a later retry (or a
            # subsequent revision) can re-attempt. The result/exception is
            # already on the Future for current coalesced waiters.
            self._inflight.pop(cache_key, None)
        # The creator awaits its own Future: retrieves the result for success,
        # re-raises on failure. Identical to what a coalesced waiter does.
        return await future


__all__: "list[str]" = [
    "CatalogCodec",
    "CatalogError",
    "CatalogItemNotFoundError",
    "CatalogItemParseError",
    "CatalogSource",
    "RevisionCache",
]
