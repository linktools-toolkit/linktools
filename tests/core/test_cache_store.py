# -*- coding: utf-8 -*-
"""Tests for the SQLite CacheStore (spec §7.2-7.7, §7.12).

This is the new transactional local-store infrastructure (PR 05); it is not yet
wired into ``environ.cache`` and does not replace ``FileCache`` -- that
migration + deletion is PR 06.
"""
import threading

import pytest

from linktools._cache_store import CacheStore, JsonCodec, BytesCodec
from linktools.errors import (
    CacheValueError, CacheCodecError, CacheBusyError,
)
from linktools.types import MISSING


@pytest.fixture
def store(tmp_path):
    s = CacheStore(tmp_path / "cache.db")
    try:
        yield s
    finally:
        s.close()


@pytest.fixture
def ns(store):
    return store.namespace("test")


# --------------------------------------------------------------------------- #
# §7.5 value semantics -- falsy values round-trip; existence by row presence.
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("value", [None, False, True, 0, 0.0, "", "x", [1], {"a": 1}])
def test_falsy_roundtrip(ns, value):
    ns.set("k", value)
    assert ns.contains("k") is True
    assert ns.get("k") == value


def test_missing_key_returns_default(ns):
    assert ns.get("nope") is None
    assert ns.get("nope", default="sentinel") == "sentinel"
    assert ns.get("nope", default=MISSING) is MISSING
    assert ns.contains("nope") is False


def test_stored_none_is_distinguishable_from_missing(ns):
    ns.set("k", None)
    assert ns.contains("k") is True
    assert ns.get("k", default="sentinel") is None  # stored None, not the default


# --------------------------------------------------------------------------- #
# §7.4 TTL
# --------------------------------------------------------------------------- #

def test_ttl_none_never_expires(ns):
    ns.set("k", "v", ttl=None)
    assert ns.get("k") == "v"


def test_ttl_zero_is_immediately_expired(ns):
    ns.set("k", "v", ttl=0)
    assert ns.contains("k") is False
    assert ns.get("k", default="gone") == "gone"


def test_ttl_negative_is_rejected(ns):
    with pytest.raises(CacheValueError):
        ns.set("k", "v", ttl=-1)


def test_ttl_expires_after_time(ns, monkeypatch):
    # The store uses time.time() (UTC unix) for persistent TTL.
    import linktools._cache_store as mod
    clock = [1000.0]
    monkeypatch.setattr(mod.time, "time", lambda: clock[0])
    ns.set("k", "v", ttl=10)         # expires at 1010
    assert ns.get("k") == "v"
    clock[0] = 1011.0
    assert ns.contains("k") is False  # expired -> lazily removed
    assert ns.get("k", default="gone") == "gone"


# --------------------------------------------------------------------------- #
# §7.7 atomic increment
# --------------------------------------------------------------------------- #

def test_incr_missing_uses_initial_plus_delta(ns):
    assert ns.increment("c", delta=5, initial=0) == 5
    assert ns.get("c") == 5


def test_incr_existing_adds_delta(ns):
    ns.set("c", 7)
    assert ns.increment("c", delta=5) == 12
    assert ns.get("c") == 12


def test_incr_concurrent_same_key_is_lossless(store):
    ns = store.namespace("counters")
    ns.set("c", 0)
    barrier = threading.Event()

    def worker():
        barrier.wait()
        for _ in range(50):
            ns.increment("c", delta=1, initial=0)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    barrier.set()
    for t in threads:
        t.join()
    # 8 threads * 50 increments, no lost updates.
    assert ns.get("c") == 400


# --------------------------------------------------------------------------- #
# §7.5/§7.7 transactions
# --------------------------------------------------------------------------- #

def test_transaction_commits_atomically(ns):
    with ns.transaction() as tx:
        tx.set("a", 1)
        tx.set("b", 2)
    assert ns.get("a") == 1 and ns.get("b") == 2


def test_transaction_rolls_back_on_error(ns):
    ns.set("keep", "yes")
    with pytest.raises(RuntimeError):
        with ns.transaction() as tx:
            tx.set("a", 1)
            raise RuntimeError("boom")
    assert ns.get("a", default="nope") == "nope"
    assert ns.get("keep") == "yes"  # pre-existing data untouched


# --------------------------------------------------------------------------- #
# iteration + namespaces
# --------------------------------------------------------------------------- #

def test_keys_and_items_are_snapshots(ns):
    ns.set("a", 1)
    ns.set("b", 2)
    assert sorted(ns.keys()) == ["a", "b"]
    assert dict(ns.items()) == {"a": 1, "b": 2}


def test_namespaces_are_isolated(store):
    a = store.namespace("a")
    b = store.namespace("b")
    a.set("k", 1)
    b.set("k", 2)
    assert a.get("k") == 1
    assert b.get("k") == 2


# --------------------------------------------------------------------------- #
# codecs (§7.4)
# --------------------------------------------------------------------------- #

def test_bytes_codec(store):
    ns = store.namespace("raw", codec=BytesCodec())
    ns.set("k", b"\x00\x01")
    assert ns.get("k") == b"\x00\x01"


def test_bytes_codec_rejects_non_bytes(store):
    ns = store.namespace("raw", codec=BytesCodec())
    with pytest.raises(CacheCodecError):
        ns.set("k", "not bytes")


def test_corrupt_blob_raises_codec_error(store):
    ns = store.namespace("bad")
    ns.set("k", {"x": 1})
    # Corrupt the stored blob directly.
    store._conn().execute(
        "UPDATE cache_entries SET value=? WHERE namespace='bad' AND key='k'",
        (b"not-json",))
    with pytest.raises(CacheCodecError):
        ns.get("k")


# --------------------------------------------------------------------------- #
# delete
# --------------------------------------------------------------------------- #

def test_delete_returns_bool(ns):
    ns.set("k", 1)
    assert ns.delete("k") is True
    assert ns.delete("k") is False  # already gone
    assert ns.contains("k") is False
