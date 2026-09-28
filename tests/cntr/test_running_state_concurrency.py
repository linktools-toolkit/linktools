#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Concurrent partial mark_started/mark_stopped calls must not lose each
other's write to RUNNING_CONTAINERS (review P1-10).

Before this, only remove() held the "cntr:settings" process lock;
mark_started/mark_stopped did a plain read-modify-write, so two racing
partial ups could each read the same starting set and one's write would
silently clobber the other's -- a lost update. RUNNING_CONTAINERS lives in
a SQLite-backed cache namespace where every get()/set() hits the DB
directly (no in-process stale-read cache), so this race can only be
demonstrated with genuine thread interleaving, not just sequential calls
from separate manager instances.
"""
import threading

from linktools.cntr.context import EventContext
from _harness import install_deterministic_interaction, _reset_global_config


def _make_manager(tmp_path, monkeypatch):
    storage = str(tmp_path)
    monkeypatch.setenv("LINKTOOLS_PATH", storage)
    monkeypatch.setenv("LINKTOOLS_DATA_PATH", storage + "/data")
    monkeypatch.setenv("LINKTOOLS_TEMP_PATH", storage + "/temp")

    from linktools.core._environ import Environ
    from linktools.cntr.manager import ContainerManager

    # A fresh Environ()/ConfigStore reads the on-disk state at construction
    # time -- exactly like a new `ct-cntr` process invocation.
    return ContainerManager(Environ(), name="aio")


def _two_managers_with_both_containers_installed(tmp_path, monkeypatch):
    install_deterministic_interaction()
    _reset_global_config()

    setup = _make_manager(tmp_path, monkeypatch)
    name_a, name_b = list(setup.containers.keys())[:2]
    setup.installed_state.add(name_a, name_b)

    # Both constructed AFTER the install, so each sees it fresh -- only
    # RUNNING_CONTAINERS (not INSTALLED_CONTAINERS) is raced on below.
    manager_a = _make_manager(tmp_path, monkeypatch)
    manager_b = _make_manager(tmp_path, monkeypatch)
    return manager_a, manager_b, name_a, name_b


def _partial_ctx(manager, name):
    ctx = EventContext()
    ctx.commands = ["up"]
    ctx.containers = manager.installed_state.get(resolve=True)
    ctx.target_containers = [c for c in ctx.containers if c.name == name]
    ctx.is_full_containers = False
    return ctx


def test_two_sequential_partial_ups_for_different_containers_both_persist(tmp_path, monkeypatch):
    manager_a, manager_b, name_a, name_b = _two_managers_with_both_containers_installed(tmp_path, monkeypatch)

    manager_a.running_state.mark_started(_partial_ctx(manager_a, name_a))
    manager_b.running_state.mark_started(_partial_ctx(manager_b, name_b))

    observer = _make_manager(tmp_path, monkeypatch)
    assert set(observer.running_state.get_persisted()) == {name_a, name_b}


def test_sequential_partial_up_then_partial_down_of_different_containers_both_persist(tmp_path, monkeypatch):
    manager_a, manager_b, name_a, name_b = _two_managers_with_both_containers_installed(tmp_path, monkeypatch)
    manager_a.running_state.mark_started(_partial_ctx(manager_a, name_a))
    manager_a.running_state.mark_started(_partial_ctx(manager_a, name_b))

    manager_a.running_state.mark_stopped(_partial_ctx(manager_a, name_a))
    manager_b.running_state.mark_started(_partial_ctx(manager_b, name_b))  # no-op, already started

    observer = _make_manager(tmp_path, monkeypatch)
    assert set(observer.running_state.get_persisted()) == {name_b}


def test_truly_concurrent_partial_ups_do_not_lose_either_update(fresh_manager, monkeypatch):
    """Force a genuine read/write interleave between two threads racing
    mark_started for different containers, via a real threading.Lock (not
    just sequential calls) -- this must fail if _mutate's process_lock is
    ever removed: thread A reads the (empty) running set, is paused before
    it writes; thread B reads the same empty set and writes its own target;
    thread A resumes and, unlocked, would overwrite B's write with a value
    computed before B's write ever happened.
    """
    name_a, name_b = list(fresh_manager.containers.keys())[:2]
    running_state = fresh_manager.running_state
    original_get = running_state._get

    a_read_done = threading.Event()
    release_a = threading.Event()
    call_count = {"n": 0}

    def get_with_delay():
        call_count["n"] += 1
        result = original_get()
        if call_count["n"] == 1:
            # Only the first call (thread A, inside the lock) pauses here --
            # if the lock is held, thread B's own _get() call blocks on the
            # lock itself and can't even start until A releases it.
            a_read_done.set()
            release_a.wait(timeout=5)
        return result

    monkeypatch.setattr(running_state, "_get", get_with_delay)

    ctx_a = _partial_ctx(fresh_manager, name_a)
    ctx_b = _partial_ctx(fresh_manager, name_b)

    thread_a = threading.Thread(target=running_state.mark_started, args=(ctx_a,))
    thread_a.start()
    assert a_read_done.wait(timeout=5), "thread A never reached its read"

    thread_b = threading.Thread(target=running_state.mark_started, args=(ctx_b,))
    thread_b.start()
    # Give thread B every chance to run if it were NOT blocked by the lock.
    thread_b.join(timeout=0.3)
    assert thread_b.is_alive(), "thread B must be blocked on the lock while thread A holds it"

    release_a.set()
    thread_a.join(timeout=5)
    thread_b.join(timeout=5)
    assert not thread_a.is_alive() and not thread_b.is_alive()

    assert set(running_state.get_persisted()) == {name_a, name_b}
