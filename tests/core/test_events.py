# -*- coding: utf-8 -*-
"""Tests for EventBus + EventHandlerMixin (spec §15.2)."""
import pytest

from linktools.runtime.events import (
    EventBus, EventHandlerMixin, LOG_AND_CONTINUE, RAISE_FIRST, COLLECT, STOP,
)


# §15.2 RUN-EVT-001 subscription ----------------------------------------------

def test_on_returns_cancellable_subscription():
    bus = EventBus()
    hits = []
    sub = bus.on("e", lambda *a: hits.append(a))
    bus.emit("e", 1)
    assert sub.cancelled is False
    sub.cancel()
    assert sub.cancelled is True
    bus.emit("e", 2)
    assert hits == [(1,)]  # second emit not delivered
    sub.cancel()  # idempotent


def test_off_removes_handler():
    bus = EventBus()
    hits = []
    cb = lambda *a: hits.append(a)
    bus.on("e", cb)
    bus.off("e", cb)
    bus.emit("e", 1)
    assert hits == []


# §15.2 RUN-EVT-003 once ------------------------------------------------------

def test_once_fires_only_once():
    bus = EventBus()
    hits = []
    bus.once("e", lambda *a: hits.append(a))
    bus.emit("e", 1)
    bus.emit("e", 2)
    assert hits == [(1,)]


def test_on_times_n():
    bus = EventBus()
    hits = []
    bus.on("e", lambda *a: hits.append(a), times=2)
    bus.emit("e", 1)
    bus.emit("e", 2)
    bus.emit("e", 3)
    assert hits == [(1,), (2,)]


# §15.2 RUN-EVT-002 exception policies ---------------------------------------

def test_policy_log_and_continue_keeps_going():
    bus = EventBus(exception_policy=LOG_AND_CONTINUE)
    order = []
    bus.on("e", lambda *a: order.append("a"))
    bus.on("e", lambda *a: (_ for _ in ()).throw(ValueError("boom")))  # noqa
    bus.on("e", lambda *a: order.append("c"))
    bus.emit("e")  # must not raise; c still runs
    assert order == ["a", "c"]


def test_policy_raise_first():
    bus = EventBus(exception_policy=RAISE_FIRST)
    bus.on("e", lambda *a: None)
    bus.on("e", lambda *a: 1 / 0)
    with pytest.raises(ZeroDivisionError):
        bus.emit("e")


def test_policy_stop_raises_and_halts():
    bus = EventBus(exception_policy=STOP)
    after = []
    bus.on("e", lambda *a: (_ for _ in ()).throw(RuntimeError("x")))  # noqa
    bus.on("e", lambda *a: after.append("ran"))
    with pytest.raises(RuntimeError):
        bus.emit("e")
    assert after == []  # later handler did not run


def test_policy_collect_raises_first_collected():
    bus = EventBus(exception_policy=COLLECT)
    bus.on("e", lambda *a: None)
    bus.on("e", lambda *a: (_ for _ in ()).throw(ValueError("v")))  # noqa
    bus.on("e", lambda *a: (_ for _ in ()).throw(KeyError("k")))  # noqa
    with pytest.raises(ValueError):
        bus.emit("e")


def test_unknown_policy_rejected():
    with pytest.raises(ValueError):
        EventBus(exception_policy="bogus")


# §15.2 RUN-EVT-004 thread-safety / self-cancel ------------------------------

def test_callback_can_cancel_self_during_emit():
    bus = EventBus()
    seen = []

    def cb(*a):
        seen.append(a)
        sub.cancel()

    sub = bus.on("e", cb)
    bus.emit("e", 1)
    bus.emit("e", 2)
    assert seen == [(1,)]  # cancelled itself; second emit not delivered


# EventHandlerMixin delegation ------------------------------------------------

def test_event_handler_mixin_delegates():
    class M(EventHandlerMixin):
        pass

    m = M()
    hits = []
    m.on("x", lambda *a, **k: hits.append((a, k)))
    m.trigger("x", 1, foo=2)
    assert hits == [((1,), {"foo": 2})]
