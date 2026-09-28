#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plan and Doctor are documented read-only (review P1-11): they must never
write a Dockerfile/compose file, mutate the Artifact Index, run a
container's on_prepare(), or touch any other persistent/generated state --
regardless of scan/render order. Every real write entry point is stubbed to
fail loudly if called at all.
"""
import pytest

from linktools.cntr.artifacts import ArtifactIndex
from linktools.cntr.doctor import Doctor
from linktools.cntr.execution.planner import ExecutionPlanner


class _WriteAttempted(AssertionError):
    pass


def _install_write_guards(manager, monkeypatch):
    def guard(name):
        def fail(*a, **k):
            raise _WriteAttempted(f"{name} must not be called by a read-only Plan/Doctor pass")
        return fail

    monkeypatch.setattr(manager.settings, "set", guard("ConfigStore.set"))
    monkeypatch.setattr(manager.settings, "save", guard("ConfigStore.save"))
    monkeypatch.setattr(manager.settings, "remove", guard("ConfigStore.remove"))
    monkeypatch.setattr(ArtifactIndex, "record", guard("ArtifactIndex.record"))

    from linktools.cntr import artifacts as artifacts_module
    monkeypatch.setattr(artifacts_module, "atomic_write_text_if_changed", guard("atomic_write_text_if_changed"))

    import os
    real_mkdir = os.makedirs

    def guarded_makedirs(path, *a, **k):
        # Only forbid mkdir under the *generated* data areas Plan/Doctor
        # must never populate -- not arbitrary directories used elsewhere
        # by the harness/test machinery itself.
        path_str = str(path)
        if any(part in path_str for part in ("/compose", "/dockerfile", "/generated")):
            raise _WriteAttempted(f"os.makedirs({path!r}) must not be called by a read-only Plan/Doctor pass")
        return real_mkdir(path, *a, **k)

    monkeypatch.setattr(os, "makedirs", guarded_makedirs)

    from linktools.cntr.lifecycle.dispatcher import LifecycleDispatcher
    monkeypatch.setattr(LifecycleDispatcher, "_invoke_callback", guard("lifecycle callback"))

    def container_on_prepare_guard(self, *a, **k):
        raise _WriteAttempted("BaseContainer.on_prepare must not be called by a read-only Plan/Doctor pass")

    from linktools.cntr.container import BaseContainer
    monkeypatch.setattr(BaseContainer, "on_prepare", container_on_prepare_guard)


def test_plan_up_triggers_zero_write_calls(fresh_manager, monkeypatch):
    _install_write_guards(fresh_manager, monkeypatch)
    planner = ExecutionPlanner(fresh_manager)

    plan = planner.plan("up", names=["nginx"])

    assert plan is not None


def test_plan_does_not_generate_dockerfile_or_compose_on_disk(fresh_manager, monkeypatch):
    import os
    planner = ExecutionPlanner(fresh_manager)
    planner.plan("up", names=["nginx"])

    dockerfile_dir = os.path.join(str(fresh_manager.data_path), "dockerfile")
    compose_dir = os.path.join(str(fresh_manager.data_path), "compose")
    assert not os.path.exists(dockerfile_dir)
    assert not os.path.exists(compose_dir)


def test_plan_does_not_write_artifact_index(fresh_manager):
    planner = ExecutionPlanner(fresh_manager)
    planner.plan("up", names=["nginx"])

    assert fresh_manager.artifact_index.load() == {}


def test_plan_does_not_call_on_prepare(fresh_manager, monkeypatch):
    _install_write_guards(fresh_manager, monkeypatch)
    planner = ExecutionPlanner(fresh_manager)

    # Must not raise -- if on_prepare were called, the guard would fire.
    planner.plan("down", names=["nginx"])
    planner.plan("restart", names=["nginx"])


def test_doctor_run_triggers_zero_write_calls(fresh_manager, monkeypatch):
    _install_write_guards(fresh_manager, monkeypatch)
    doctor = Doctor(fresh_manager)

    findings = doctor.run()

    assert isinstance(findings, list)


def test_doctor_does_not_generate_dockerfile_or_compose_on_disk(fresh_manager):
    import os
    doctor = Doctor(fresh_manager)
    doctor.run()

    dockerfile_dir = os.path.join(str(fresh_manager.data_path), "dockerfile")
    compose_dir = os.path.join(str(fresh_manager.data_path), "compose")
    assert not os.path.exists(dockerfile_dir)
    assert not os.path.exists(compose_dir)


def test_doctor_does_not_write_artifact_index(fresh_manager):
    doctor = Doctor(fresh_manager)
    doctor.run()

    assert fresh_manager.artifact_index.load() == {}


def test_real_up_still_generates_dockerfile_and_artifacts(fresh_manager):
    """Sanity check that the read-only guard tests above are actually
    meaningful -- real execution (not Plan/Doctor) still writes."""
    nginx = fresh_manager.containers["nginx"]
    nginx.get_docker_compose_file()

    assert "dockerfile/nginx.Dockerfile" in fresh_manager.artifact_index.load()
