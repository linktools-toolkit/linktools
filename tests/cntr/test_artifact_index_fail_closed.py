#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ArtifactIndex.record() concurrency safety and Doctor/Plan's handling of
a corrupted index (review P1-12).
"""
import threading

import pytest

from linktools.cntr.artifacts import ArtifactIndexError
from linktools.cntr.doctor import ARTIFACT_INDEX_INVALID, Doctor, WARN
from linktools.cntr.execution.planner import ExecutionPlanner


def _corrupt_index(manager):
    import os
    path = manager.artifact_index.path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("not json")


def test_doctor_reports_warn_and_does_not_overwrite_corrupt_index(fresh_manager):
    _corrupt_index(fresh_manager)
    import os
    before = open(fresh_manager.artifact_index.path, "r", encoding="utf-8").read()

    doctor = Doctor(fresh_manager)
    findings = doctor.check_artifacts(list(fresh_manager.containers.values()))

    assert any(f.code == ARTIFACT_INDEX_INVALID and f.severity == WARN for f in findings)
    after = open(fresh_manager.artifact_index.path, "r", encoding="utf-8").read()
    assert after == before  # never overwritten


def test_doctor_run_does_not_crash_on_corrupt_index(fresh_manager):
    _corrupt_index(fresh_manager)
    doctor = Doctor(fresh_manager)

    findings = doctor.run()  # must not raise

    assert any(f.code == ARTIFACT_INDEX_INVALID for f in findings)


def test_plan_fails_explicitly_on_corrupt_index(fresh_manager):
    _corrupt_index(fresh_manager)
    planner = ExecutionPlanner(fresh_manager)

    with pytest.raises(ArtifactIndexError):
        planner.plan("up", names=["nginx"])


def test_record_holds_a_lock_serializing_concurrent_writers(fresh_manager, monkeypatch):
    """Two threads recording different artifacts at the same time must not
    lose either entry -- record()'s read-merge-write used to run unlocked,
    so a second writer starting its own read before the first writer's
    write landed would silently discard the first writer's entry. Proven
    here by forcing thread B to prove it's genuinely BLOCKED on the lock
    (not just racing and happening to succeed) while thread A holds it.
    """
    index = fresh_manager.artifact_index
    a_in_critical_section = threading.Event()
    release_a = threading.Event()
    original_load = index.load

    def slow_load():
        result = original_load()
        a_in_critical_section.set()
        release_a.wait(timeout=5)
        return result

    monkeypatch.setattr(index, "load", slow_load)

    thread_a = threading.Thread(target=index.record, args=(
        {"compose/a.yml": dict(kind="compose", container="a", sha256="1", source=None)},))
    thread_a.start()
    assert a_in_critical_section.wait(timeout=5), "thread A never reached its critical section"

    thread_b = threading.Thread(target=index.record, args=(
        {"compose/b.yml": dict(kind="compose", container="b", sha256="2", source=None)},))
    thread_b.start()
    thread_b.join(timeout=0.3)
    assert thread_b.is_alive(), "thread B must be blocked on the lock while thread A holds it"

    release_a.set()
    thread_a.join(timeout=5)
    thread_b.join(timeout=5)
    assert not thread_a.is_alive() and not thread_b.is_alive()

    monkeypatch.setattr(index, "load", original_load)
    artifacts = index.load()
    assert "compose/a.yml" in artifacts
    assert "compose/b.yml" in artifacts
