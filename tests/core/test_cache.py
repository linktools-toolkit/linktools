# -*- coding: utf-8 -*-
"""Tests for :class:`linktools.cache.FileCache` value semantics.

Covers the two correctness bugs the spec calls out for the cache layer and
lists in the first-batch checklist (§31): falsy values must round-trip (§7.5
CAC-003) and ``incr`` on a missing key must store ``initial + delta`` (§7.7
CAC-005). The full SQLite backend rewrite arrives in a later phase (§7); these
tests pin the value semantics so the rewrite can prove it preserves them.
"""
import pytest

from linktools.cache import FileCache


@pytest.fixture
def cache(tmp_path):
    return FileCache(str(tmp_path / "cache"))


# --------------------------------------------------------------------------- #
# §7.5 CAC-003 -- falsy values must be reliably stored and distinguished from
# "missing". Existence is decided by presence (contains), never by truthiness.
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("value", [None, False, 0, 0.0, "", [], {}])
def test_falsy_value_roundtrips(cache, value):
    cache.set("k", value)
    assert cache.contains("k") is True
    assert cache.get("k") == value


def test_missing_key_is_not_contained(cache):
    assert cache.contains("missing") is False
    # default must come back exactly when the key is absent
    assert cache.get("missing", default="sentinel") == "sentinel"


def test_stored_none_is_distinguishable_from_missing(cache):
    cache.set("k", None)
    assert cache.contains("k") is True
    # A stored None must not collide with the default-when-missing
    assert cache.get("k", default="sentinel") is None


def test_session_falsy_roundtrip(cache):
    with cache.session() as s:
        s.set("k", False)
        assert s.contains("k") is True
        assert s.get("k") is False


# --------------------------------------------------------------------------- #
# §7.7 CAC-005 -- increment must be atomic and add delta even when the key is
# missing (initial + delta), never "write default without adding delta".
# --------------------------------------------------------------------------- #

def test_incr_missing_key_adds_delta(cache):
    assert cache.incr("counter", delta=5, default=0) == 5
    assert cache.get("counter") == 5


def test_incr_missing_key_uses_initial_plus_delta(cache):
    assert cache.incr("counter", delta=3, default=10) == 13
    assert cache.get("counter") == 13


def test_incr_existing_key_adds_delta(cache):
    cache.set("counter", 7)
    assert cache.incr("counter", delta=5) == 12
    assert cache.get("counter") == 12


def test_session_incr_missing_key_adds_delta(cache):
    with cache.session() as s:
        assert s.incr("counter", delta=2, default=0) == 2


# --------------------------------------------------------------------------- #
# §7.4 CAC-004 -- TTL semantics (characterisation of current behaviour).
# --------------------------------------------------------------------------- #

def test_ttl_none_never_expires(cache):
    cache.set("k", "v", ttl=None)
    assert cache.get("k") == "v"
    assert cache.contains("k") is True


def test_ttl_zero_is_immediately_expired(cache):
    # spec §7.4: ttl=0 means immediately expired (not recommended to write).
    cache.set("k", "v", ttl=0)
    assert cache.contains("k") is False
    assert cache.get("k", default="gone") == "gone"
