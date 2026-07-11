#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Read-only doctor checks.

scan_compose is a pure function over a rendered compose dict; Doctor.run ties the
checks together and must never modify the config store.
"""
from linktools.cntr.doctor import (
    ARTIFACT_STALE, COMPOSE_VALIDATION_FAILED, SECURITY_DOCKER_SOCKET_MOUNT, SECURITY_LATEST_IMAGE,
    SECURITY_TLS_DISABLED, Doctor, Finding, WARN, scan_compose,
)


def _messages(findings):
    return [f.message for f in findings]


def test_scan_reports_latest_tag():
    assert any("latest" in m for m in _messages(scan_compose("c", {"services": {"a": {"image": "x:latest"}}})))


def test_scan_reports_untagged_image_as_latest():
    assert any("latest" in m for m in _messages(scan_compose("c", {"services": {"a": {"image": "nginx"}}})))


def test_scan_ignores_pinned_tag():
    assert not any("latest" in m for m in _messages(scan_compose("c", {"services": {"a": {"image": "nginx:1.25"}}})))


def test_scan_keeps_registry_port_out_of_tag():
    # registry.example:5000/x:1.2 -> tag is 1.2, not "5000/x".
    assert not any("latest" in m for m in _messages(
        scan_compose("c", {"services": {"a": {"image": "registry.example:5000/x:1.2"}}})))


def test_scan_reports_docker_socket_mount():
    msgs = _messages(scan_compose("c", {"services": {"a": {
        "image": "x:1", "volumes": ["/var/run/docker.sock:/var/run/docker.sock"]}}}))
    assert any("docker socket" in m for m in msgs)


def test_scan_reports_tls_disabled_list_env():
    msgs = _messages(scan_compose("c", {"services": {"a": {
        "image": "x:1", "environment": ["NODE_TLS_REJECT_UNAUTHORIZED=0"]}}}))
    assert any("NODE_TLS_REJECT_UNAUTHORIZED" in m for m in msgs)


def test_scan_reports_tls_disabled_dict_env():
    msgs = _messages(scan_compose("c", {"services": {"a": {
        "image": "x:1", "environment": {"NODE_TLS_REJECT_UNAUTHORIZED": "0"}}}}))
    assert any("NODE_TLS_REJECT_UNAUTHORIZED" in m for m in msgs)


def test_scan_clean_compose_has_no_warnings():
    findings = [f for f in scan_compose("c", {"services": {"a": {
        "image": "x:1.2", "volumes": ["/data:/data"]}}}) if f.severity == WARN]
    assert findings == []


def test_lock_is_fully_removed(fresh_manager):
    """Deployment Lock (and its Doctor finding) is gone with no trace: no
    manager.lock_store, no Doctor.check_lock, no lock.* finding code."""
    assert not hasattr(fresh_manager, "lock_store")
    assert not hasattr(Doctor, "check_lock")
    findings = Doctor(fresh_manager).run()
    assert not any((f.code or "").startswith("lock.") for f in findings)


def test_doctor_runs_on_builtins(fresh_manager):
    findings = Doctor(fresh_manager).run()
    assert isinstance(findings, list)
    assert all(isinstance(f, Finding) for f in findings)
    assert findings  # at least the runtime finding


def test_doctor_does_not_write_to_config_store(fresh_manager, monkeypatch):
    store = fresh_manager.environ.config_store
    writes = []
    original_set = store.set

    def spy(*args, **kwargs):
        writes.append(args)
        return original_set(*args, **kwargs)

    monkeypatch.setattr(store, "set", spy)
    Doctor(fresh_manager).run()
    assert writes == [], f"doctor wrote to config store: {writes}"


# -- Finding stable codes --------------------------------------------------

def test_finding_is_frozen_with_code_component_details_defaults():
    finding = Finding(WARN, "message")
    assert finding.code is None
    assert finding.component is None
    assert finding.details == {}
    import pytest as _pytest
    with _pytest.raises(Exception):
        finding.severity = "OK"


def test_scan_compose_findings_carry_stable_codes():
    latest = scan_compose("c", {"services": {"a": {"image": "nginx"}}})
    assert any(f.code == SECURITY_LATEST_IMAGE for f in latest)

    socket_mount = scan_compose("c", {"services": {"a": {
        "image": "x:1", "volumes": ["/var/run/docker.sock:/var/run/docker.sock"]}}})
    assert any(f.code == SECURITY_DOCKER_SOCKET_MOUNT for f in socket_mount)

    tls = scan_compose("c", {"services": {"a": {
        "image": "x:1", "environment": ["NODE_TLS_REJECT_UNAUTHORIZED=0"]}}})
    assert any(f.code == SECURITY_TLS_DISABLED for f in tls)


# -- Artifact staleness (report-only) --------------------------------------

def test_check_artifacts_is_empty_when_index_is_empty(fresh_manager):
    assert Doctor(fresh_manager).check_artifacts(fresh_manager.containers.values()) == []


def test_check_artifacts_reports_stale_entry_not_deletes_it(fresh_manager):
    fresh_manager.artifact_index.record({
        "compose/ghost.yml": dict(kind="compose", container="ghost", sha256="abc", source=None),
    })
    findings = Doctor(fresh_manager).check_artifacts(fresh_manager.containers.values())
    assert any(f.code == ARTIFACT_STALE and "ghost" in f.message for f in findings)
    # Report-only: the index entry itself must still be present afterward.
    assert "compose/ghost.yml" in fresh_manager.artifact_index.load()


# -- CLI: --json / --check / --runtime ---------------------------------------

def test_doctor_json_output_is_stable_schema(fresh_manager, capsys, monkeypatch):
    import json
    import linktools.cntr.__main__ as cntr_main
    import linktools.cntr.commands._shared as cntr_shared
    monkeypatch.setattr(cntr_shared, "manager", fresh_manager)

    cntr_main.command.on_command_doctor(as_json=True)
    payload = json.loads(capsys.readouterr().out)
    assert payload["schema_version"] == 1
    assert payload["project"] == fresh_manager.project_name
    assert isinstance(payload["findings"], list)
    for entry in payload["findings"]:
        assert set(entry.keys()) == {"severity", "code", "component", "message", "details"}


def test_doctor_default_command_does_not_raise_regardless_of_findings(fresh_manager, monkeypatch):
    import linktools.cntr.__main__ as cntr_main
    import linktools.cntr.commands._shared as cntr_shared
    monkeypatch.setattr(cntr_shared, "manager", fresh_manager)
    # Must not raise even though this sandbox has no docker binary (WARN findings).
    cntr_main.command.on_command_doctor()


# -- Compose config validation (--runtime only) -------------------------------

def test_check_compose_validation_is_empty_without_docker_binary(fresh_manager):
    # This sandbox genuinely has no `docker` binary.
    findings = Doctor(fresh_manager).check_compose_validation(fresh_manager.containers.values())
    assert findings == []


def test_check_compose_validation_reports_failure_when_docker_present(fresh_manager, monkeypatch):
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/docker" if name == "docker" else None)

    class _FailedResult:
        succeeded = False
        stderr = "service \"nginx\" refers to undefined network"

    monkeypatch.setattr(
        fresh_manager.docker_inspector, "validate_compose",
        lambda containers, allow_sudo_prompt=False: _FailedResult(),
    )
    findings = Doctor(fresh_manager).check_compose_validation(fresh_manager.containers.values())
    assert any(f.code == COMPOSE_VALIDATION_FAILED for f in findings)


def test_check_compose_validation_is_empty_when_it_succeeds(fresh_manager, monkeypatch):
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/docker" if name == "docker" else None)

    class _OkResult:
        succeeded = True
        stderr = ""

    monkeypatch.setattr(
        fresh_manager.docker_inspector, "validate_compose",
        lambda containers, allow_sudo_prompt=False: _OkResult(),
    )
    findings = Doctor(fresh_manager).check_compose_validation(fresh_manager.containers.values())
    assert findings == []


def _stub_runtime_probes_and_docker_present(fresh_manager, monkeypatch):
    """Make check_runtime() see a `docker` binary without spawning a real
    subprocess for the engine/compose version probes."""
    from linktools.cntr.runtime.inspect import DockerEngineVersion
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/docker" if name == "docker" else None)
    monkeypatch.setattr(fresh_manager.docker_inspector, "get_engine_version",
                        lambda allow_sudo_prompt=False: DockerEngineVersion(client="1.0", server="1.0", api="1.0"))
    monkeypatch.setattr(fresh_manager.docker_inspector, "get_compose_version", lambda allow_sudo_prompt=False: "2.0")


def test_doctor_run_with_runtime_true_includes_compose_validation(fresh_manager, monkeypatch):
    _stub_runtime_probes_and_docker_present(fresh_manager, monkeypatch)

    class _FailedResult:
        succeeded = False
        stderr = "boom"

    monkeypatch.setattr(
        fresh_manager.docker_inspector, "validate_compose",
        lambda containers, allow_sudo_prompt=False: _FailedResult(),
    )
    findings = Doctor(fresh_manager).run(runtime=True)
    assert any(f.code == COMPOSE_VALIDATION_FAILED for f in findings)


def test_doctor_run_with_runtime_false_skips_compose_validation(fresh_manager, monkeypatch):
    _stub_runtime_probes_and_docker_present(fresh_manager, monkeypatch)
    called = []

    def fake_validate(containers, allow_sudo_prompt=False):
        called.append(1)
        raise AssertionError("must not be called when runtime=False")

    monkeypatch.setattr(fresh_manager.docker_inspector, "validate_compose", fake_validate)
    Doctor(fresh_manager).run(runtime=False)
    assert called == []


# -- --sudo-prompt opts into an interactive sudo prompt --------------------

def _stub_runtime_probes_recording_sudo_prompt(fresh_manager, monkeypatch, seen):
    from linktools.cntr.runtime.inspect import DockerEngineVersion
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/docker" if name == "docker" else None)

    def fake_engine_version(allow_sudo_prompt=False):
        seen.append(allow_sudo_prompt)
        return DockerEngineVersion(client=None, server=None, api=None)

    def fake_compose_version(allow_sudo_prompt=False):
        seen.append(allow_sudo_prompt)
        return None

    monkeypatch.setattr(fresh_manager.docker_inspector, "get_engine_version", fake_engine_version)
    monkeypatch.setattr(fresh_manager.docker_inspector, "get_compose_version", fake_compose_version)


def test_doctor_defaults_to_non_interactive_sudo_for_runtime_probes(fresh_manager, monkeypatch):
    seen = []
    _stub_runtime_probes_recording_sudo_prompt(fresh_manager, monkeypatch, seen)
    Doctor(fresh_manager).run()
    assert seen == [False, False]


def test_doctor_sudo_prompt_flag_allows_interactive_sudo_for_runtime_probes(fresh_manager, monkeypatch):
    seen = []
    _stub_runtime_probes_recording_sudo_prompt(fresh_manager, monkeypatch, seen)
    Doctor(fresh_manager).run(sudo_prompt=True)
    assert seen == [True, True]


def test_doctor_sudo_prompt_flag_threads_through_compose_validation(fresh_manager, monkeypatch):
    _stub_runtime_probes_and_docker_present(fresh_manager, monkeypatch)
    seen = []

    class _OkResult:
        succeeded = True
        stderr = ""

    monkeypatch.setattr(
        fresh_manager.docker_inspector, "validate_compose",
        lambda containers, allow_sudo_prompt=False: seen.append(allow_sudo_prompt) or _OkResult(),
    )
    Doctor(fresh_manager).run(runtime=True, sudo_prompt=True)
    assert seen == [True]


def test_doctor_cli_exposes_sudo_prompt_flag(fresh_manager, monkeypatch):
    import linktools.cntr.__main__ as cntr_main
    import linktools.cntr.commands._shared as cntr_shared
    monkeypatch.setattr(cntr_shared, "manager", fresh_manager)
    captured = {}

    def fake_run(self, runtime=False, sudo_prompt=False):
        captured["sudo_prompt"] = sudo_prompt
        return []

    monkeypatch.setattr(Doctor, "run", fake_run)
    cntr_main.command.on_command_doctor(sudo_prompt=True)
    assert captured["sudo_prompt"] is True


# -- Repo manifest unrecognized-requirement-key reporting -------------------

def test_check_repos_reports_unrecognized_manifest_requirement_as_warn(fresh_manager, tmp_path, monkeypatch):
    """An unrecognized requirement key now fails closed (see
    test_repo_manifest.py::test_unrecognized_requirement_key_now_blocks_fail_closed),
    so repo_store.add() itself would reject this manifest -- inject it
    directly into the repo store (simulating a repo that became
    incompatible after being installed) to exercise Doctor's reporting."""
    import json as json_module
    repo_dir = tmp_path / "repo_src"
    repo_dir.mkdir()
    (repo_dir / ".linktools.json").write_text(json_module.dumps({
        "schema_version": 1,
        "kind": "linktools-project",
        "components": {
            "cntr": {
                "schema_version": 1,
                "requires": {"some-other-tool": ">=1.0"},
                "config": {}, "metadata": {}, "extensions": {},
            },
        },
    }), encoding="utf-8")
    (repo_dir / "container.py").write_text(
        "from linktools.cntr.container import BaseContainer\n\n\n"
        "class Container(BaseContainer):\n    pass\n",
        encoding="utf-8",
    )
    repos = dict(fresh_manager.repo_store.get_all())
    repos[str(repo_dir)] = dict(type="local", repo_path=str(repo_dir), repo_name="repo_src")
    monkeypatch.setattr(fresh_manager.repo_store, "get_all", lambda: repos)

    findings = Doctor(fresh_manager).check_repos()
    assert any(
        f.severity == WARN and "some-other-tool" in f.message
        for f in findings
    )


def test_doctor_check_raises_when_warn_finding_present(fresh_manager, monkeypatch):
    import linktools.cntr.__main__ as cntr_main
    import linktools.cntr.commands._shared as cntr_shared
    from linktools.cntr.container import ContainerError
    monkeypatch.setattr(cntr_shared, "manager", fresh_manager)
    # No docker binary in this sandbox -> at least one WARN finding.
    import pytest as _pytest
    with _pytest.raises(ContainerError):
        cntr_main.command.on_command_doctor(check=True)
