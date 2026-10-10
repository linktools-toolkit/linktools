#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Exercise native Compose parsing without a Docker daemon or container execution."""
import json
import shutil
import subprocess
from types import SimpleNamespace

import pytest
import yaml

from linktools.cntr.runtime.compose import ComposeRunner
from test_service_application_order import Container


@pytest.fixture(scope="module")
def native_compose():
    standalone = shutil.which("docker-compose")
    docker = shutil.which("docker")
    command = [standalone] if standalone else [docker, "compose"] if docker else None
    if command is None:
        pytest.skip("Native Compose CLI is not installed")
    version = subprocess.run([*command, "version", "--short"], capture_output=True, text=True)
    if version.returncode:
        pytest.skip("Native Compose plugin is not available")
    return command


def test_native_profile_model_preserves_disabled_inputs_and_literal_values(native_compose, tmp_path, monkeypatch):
    (tmp_path / "compose").mkdir()
    services = {
        "app": {"image": "alpine", "environment": {"LITERAL": "$$HOME"}},
        "optional": {"image": "alpine", "profiles": ["debug"], "env_file": str(tmp_path / "absent.env")},
    }
    containers = tuple(Container(name, {name: spec}) for name, spec in services.items())
    content = yaml.safe_dump({"services": services})
    monkeypatch.setattr("linktools.cntr.artifacts.collect_candidates", lambda *args: {
        "fixture.yml": ("compose", "app", content)})
    calls = []

    def process(*args, **kwargs):
        assert args[0] == "compose" and "config" in args
        calls.append(args)
        return [*native_compose, *args[1:]]

    manager = SimpleNamespace(data_path=tmp_path, project_name="native-profile-fixture",
                              runtime=SimpleNamespace(create_docker_process=process),
                              structured_runner=SimpleNamespace(execute_json=lambda command, **kwargs: json.loads(
                                  subprocess.check_output(command, text=True))))
    runner = ComposeRunner(manager)
    context = SimpleNamespace(project_containers=containers, target_services=("app",), is_full_project=True)
    active = runner.final_model(context)
    complete = runner.final_model(context, preserve_disabled=True)
    assert set(active["services"]) == {"app"}
    assert set(complete["services"]) == {"app", "optional"}
    retained = tmp_path / "retained.yml"
    retained.write_text(yaml.safe_dump(complete))
    roundtrip = json.loads(subprocess.check_output([
        *native_compose, "--project-name", manager.project_name, "--file", str(retained), "config", "--format", "json"
    ], text=True))
    assert roundtrip == active
    assert len(calls) == 3


def test_native_explicit_service_enables_its_profile(native_compose, tmp_path):
    path = tmp_path / "compose.yml"
    path.write_text(yaml.safe_dump({"services": {
        "app": {"image": "alpine"},
        "optional": {"image": "alpine", "profiles": ["debug"]},
    }}))
    model = json.loads(subprocess.check_output([
        *native_compose, "--project-name", "native-profile-fixture", "--file", str(path),
        "config", "--format", "json", "optional",
    ], text=True))
    assert set(model["services"]) == {"optional"}


@pytest.mark.parametrize("observed", ["host", "service:db"])
def test_native_legacy_capture_pins_observed_namespace_before_interpolation(native_compose, tmp_path, monkeypatch, observed):
    (tmp_path / "compose").mkdir()
    monkeypatch.setenv("CNTR_REVIEW_MODE", "service:unrelated")
    services = {"worker": {"image": "alpine", "network_mode": "${CNTR_REVIEW_MODE}", "profiles": ["debug"]},
                "db": {"image": "alpine"}}
    manager = SimpleNamespace(data_path=tmp_path, project_name="native-recovery-fixture",
                              runtime=SimpleNamespace(create_docker_process=lambda *args, **kwargs: [*native_compose, *args[1:]]),
                              structured_runner=SimpleNamespace(execute_json=lambda command, **kwargs: json.loads(
                                  subprocess.check_output(command, text=True))))
    context = SimpleNamespace(
        project_containers=tuple(Container(name, {name: spec}) for name, spec in services.items()),
        service_models=SimpleNamespace(previous={}),
        previous_compose_contents={"old.yml": yaml.safe_dump({"services": services})},
        initial_runtime_state=SimpleNamespace(services=(SimpleNamespace(
            service="worker", namespace_bindings={"network_mode": observed}),)),
    )
    captured = yaml.safe_load(ComposeRunner(manager).saved_service_models(context, ("worker",))["worker"])
    assert captured["services"]["worker"]["network_mode"] == observed
    assert "unrelated" not in captured["services"]
