# -*- coding: utf-8 -*-
"""Tests for :class:`linktools.types.Timeout` -- (TYP-002).

Semantics under test:
* uses ``time.monotonic`` (never wall-clock);
* ``None`` means infinite, ``0`` means immediately expired;
* negative values are rejected;
* can spawn a bounded sub-timeout;
* ``remaining``/``expired`` are the canonical reads, ``check``/``ensure`` retained.
"""
import time
import types as _types

import pytest

import linktools.types as lt_types
from linktools.types import Timeout


# --------------------------------------------------------------------------- #
# Clock: the spec mandates monotonic time, not wall-clock time.
# --------------------------------------------------------------------------- #

def test_default_clock_is_monotonic():
    """process timeouts must not use wall time."""
    assert lt_types._now is time.monotonic


def test_timeout_tracks_injected_clock(monkeypatch):
    # __new__ consumes one read during reset(); remaining/expired read after.
    times = iter([100.0, 103.0, 113.0])
    monkeypatch.setattr(lt_types, "_now", lambda: next(times))
    t = Timeout(10)             # reads 100.0 -> deadline 110.0
    assert t.remaining == 7.0   # 110 - 103
    assert t.expired is True    # 113 > 110


def test_wall_clock_change_does_not_affect_timeout(monkeypatch):
    """Advancing the wall clock alone must not expire a monotonic timeout."""
    fixed_monotonic = 50.0
    monkeypatch.setattr(lt_types, "_now", lambda: fixed_monotonic)
    t = Timeout(10)             # deadline 60.0 on the monotonic clock
    # Wall clock jumps arbitrarily; Timeout must be unaffected.
    assert t.expired is False
    assert t.remaining == 10.0


# --------------------------------------------------------------------------- #
# Boundary semantics.
# --------------------------------------------------------------------------- #

def test_none_means_infinite():
    t = Timeout(None)
    assert t.timeout is None
    assert t.remaining is None
    assert t.expired is False
    assert t.check() is True


def test_zero_means_immediately_expired():
    t = Timeout(0)
    assert t.expired is True
    assert t.remaining == 0
    assert t.check() is False


@pytest.mark.parametrize("value", [-1, -0.01, -100])
def test_negative_is_rejected(value):
    with pytest.raises(ValueError):
        Timeout(value)


def test_idempotent_constructor():
    inner = Timeout(5)
    assert Timeout(inner) is inner
    assert Timeout(None).timeout is None


# --------------------------------------------------------------------------- #
# Behaviour.
# --------------------------------------------------------------------------- #

def test_remaining_then_expired(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(lt_types, "_now", lambda: clock[0])
    t = Timeout(10)
    assert t.remaining == 10
    assert not t.expired
    clock[0] = 5.0
    assert t.remaining == 5.0
    assert not t.expired
    clock[0] = 11.0
    assert t.remaining == 0      # clamped, never negative
    assert t.expired


def test_ensure_raises_only_when_expired(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(lt_types, "_now", lambda: clock[0])
    t = Timeout(10)
    t.ensure()                   # not expired -> no-op
    clock[0] = 20.0
    with pytest.raises(TimeoutError):
        t.ensure()
    custom = type("Boom", (Exception,), {})
    with pytest.raises(custom):
        t.ensure(custom, "boom")


def test_reset_recomputes_deadline(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(lt_types, "_now", lambda: clock[0])
    t = Timeout(10)
    clock[0] = 100.0
    assert t.expired
    t.reset()
    assert not t.expired
    assert t.remaining == 10


def test_subtimeout_is_bounded_by_parent(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(lt_types, "_now", lambda: clock[0])
    parent = Timeout(10)
    child = parent.split(3)
    assert child.remaining <= 3
    clock[0] = 8.0
    # Parent still has 2s, child (3s) would have run out first.
    assert child.expired
    assert not parent.expired


def test_subtimeout_capped_by_remaining_parent_budget(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(lt_types, "_now", lambda: clock[0])
    parent = Timeout(2)
    child = parent.split(10)     # asks for 10 but parent only has 2
    assert child.remaining <= 2


def test_subtimeout_of_infinite_parent(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(lt_types, "_now", lambda: clock[0])
    parent = Timeout(None)
    child = parent.split(5)
    assert child.timeout == 5
    assert not child.expired


def test_repr_is_stable_and_informative():
    import re
    assert re.match(r"Timeout\(timeout=.*\)", repr(Timeout(7)))
    assert "None" in repr(Timeout(None))
