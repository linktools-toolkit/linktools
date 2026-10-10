#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Legacy certificate copies remain readable without relaxing private modes."""

import os
import shutil
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from linktools.cntr.artifacts import AppliedServiceModels
from linktools.cntr.errors import ContainerError
from linktools.cntr.runtime.compose import ComposeRunner
from _harness import builtin_container_type
from test_lifecycle_rebuild import setup_case


@pytest.mark.parametrize("system, uid, docker_type", [
    ("linux", 1000, "docker"),
    ("darwin", 1000, "docker"),
    ("linux", 1000, "docker-rootless"),
    ("linux", 0, "docker"),
])
@pytest.mark.parametrize("existing_backup", [False, True])
def test_legacy_private_keys_are_owned_before_reading(
        fresh_manager, monkeypatch, system, uid, docker_type, existing_backup):
    nginx = fresh_manager.containers["nginx"]
    nginx.get_app_path().mkdir(parents=True, exist_ok=True)
    for name, value in (("system", system), ("uid", uid), ("gid", uid),
                        ("container_type", docker_type)):
        monkeypatch.setattr(type(fresh_manager), name, value)
    privileged_copy = uid != 0 and docker_type == "docker"
    pending_ownership = set()
    calls = []

    def seed(destination):
        destination.mkdir(parents=True, exist_ok=True)
        nested = destination / "private"
        nested.mkdir(mode=0o700)
        key = nested / "key.pem"
        key.write_bytes(b"synthetic-private-key")
        key.chmod(0o600)
        (destination / "key.pem").symlink_to("private/key.pem")
        if privileged_copy:
            pending_ownership.add(str(key))

    if existing_backup:
        for name in ("certs", "acme"):
            seed(nginx.get_app_path("migration-backup", name))

    def create_process(*args, **kwargs):
        calls.append((args, kwargs))

        def check_call():
            if args[:2] == ("docker", "cp"):
                assert kwargs["privilege"] == (docker_type == "docker")
                seed(Path(args[-1]))
            else:
                assert args[:4] == ("chown", "-R", "-h", "{}:{}".format(uid, uid))
                assert Path(args[4]).parent == nginx.get_app_path()
                assert Path(args[4]).name.startswith("migration-backup-")
                assert kwargs == {"privilege": True}
                pending_ownership.clear()
        return SimpleNamespace(check_call=check_call)

    monkeypatch.setattr(nginx.runtime, "create_process", create_process)
    monkeypatch.setattr(nginx.manager.structured_runner, "execute", lambda process, **kwargs:
                        process.check_call() or SimpleNamespace(succeeded=True))
    copy = shutil.copy2

    def copy_private_file(source, destination):
        # Model root-owned 0600 files even when the test runner itself is root.
        if pending_ownership:
            raise PermissionError("Private backup files still belong to root")
        return copy(source, destination)

    monkeypatch.setattr(shutil, "copy2", copy_private_file)
    backup = nginx._preserve_legacy_files()

    assert sum(args[0] == "chown" for args, _ in calls) == int(privileged_copy)
    assert sum(args[0] == "docker" for args, _ in calls) == 2
    for name in ("certs", "acme"):
        key = nginx.get_app_path(name, "private", "key.pem")
        assert key.read_bytes() == b"synthetic-private-key"
        assert key.stat().st_mode & 0o777 == 0o600
        assert os.readlink(str(nginx.get_app_path(name, "key.pem"))) == "private/key.pem"
        backup_key = backup / name / "private" / "key.pem"
        assert backup_key.stat().st_mode & 0o777 == 0o600

    # A retry must not replace files already installed by migration or renewal.
    key.write_bytes(b"newer-private-key")
    next_backup = nginx._preserve_legacy_files()
    assert next_backup != backup
    assert backup_key.read_bytes() == b"synthetic-private-key"
    assert key.read_bytes() == b"newer-private-key"


def test_legacy_retry_captures_latest_runtime_without_replacing_old_snapshot(fresh_manager, monkeypatch):
    nginx = fresh_manager.containers["nginx"]
    nginx.get_app_path().mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(type(fresh_manager), "uid", 0)
    current = {"value": b"initial-runtime"}

    def process(*args, **kwargs):
        def copy():
            destination = Path(args[-1])
            (destination / "current").write_bytes(current["value"])
        return SimpleNamespace(check_call=copy)

    monkeypatch.setattr(nginx.runtime, "create_docker_process", process)
    monkeypatch.setattr(nginx.manager.structured_runner, "execute", lambda process, **kwargs:
                        process.check_call() or SimpleNamespace(succeeded=True))
    first = nginx._preserve_legacy_files()
    current["value"] = b"renewed-runtime"
    second = nginx._preserve_legacy_files()
    assert first != second
    for name in ("certs", "acme"):
        assert (first / name / "current").read_bytes() == b"initial-runtime"
        assert (second / name / "current").read_bytes() == b"renewed-runtime"


@pytest.mark.parametrize("missing", [("certs", "acme"), ("certs",), ("acme",)])
def test_legacy_http_image_allows_absent_tls_directories(fresh_manager, monkeypatch, missing):
    nginx = fresh_manager.containers["nginx"]
    nginx.get_app_path().mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(type(fresh_manager), "uid", 0)
    monkeypatch.setattr(nginx.runtime, "create_docker_process", lambda *args, **kwargs: args)

    def execute(args, **kwargs):
        service, source = args[1].split(":", 1)
        destination = Path(args[2])
        if destination.name in missing:
            return SimpleNamespace(succeeded=False, stderr=(
                "Error response from daemon: Could not find the file {} in container {}\n".format(source, service)))
        (destination / "current").write_text("runtime state")
        return SimpleNamespace(succeeded=True)

    monkeypatch.setattr(nginx.manager.structured_runner, "execute", execute)
    backup = nginx._preserve_legacy_files()
    for name in ("certs", "acme"):
        assert (backup / name).is_dir()
        assert (backup / name / "current").exists() is (name not in missing)


@pytest.mark.parametrize("message", [
    "Cannot connect to the Docker daemon",
    "Error response from daemon: permission denied",
    "Error response from daemon: No such container: aio-nginx",
    "Error response from daemon: Could not find the file /another/path in container aio-nginx",
    "Error response from daemon: Could not find the file /etc/certs/. in container another-nginx",
])
def test_legacy_copy_does_not_hide_unrelated_failures(fresh_manager, monkeypatch, message):
    nginx = fresh_manager.containers["nginx"]
    nginx.get_app_path().mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(nginx.runtime, "create_docker_process", lambda *args, **kwargs: args)
    monkeypatch.setattr(nginx.manager.structured_runner, "execute", lambda *args, **kwargs:
                        SimpleNamespace(succeeded=False, stderr=message))
    with pytest.raises(ContainerError, match="Cannot preserve legacy nginx"):
        nginx._preserve_legacy_files()


def test_legacy_compose_preserves_explicit_mounts_and_other_owners(fresh_manager, tmp_path):
    nginx = fresh_manager.containers["nginx"]
    old = {"services": {"nginx": {"volumes": [
        "/custom/certs:/etc/certs:ro",
        {"type": "bind", "source": "/custom/acme", "target": "/root/.acme.sh"},
    ]}}}
    original = yaml.safe_dump(old)
    context = SimpleNamespace(previous_compose_contents={"nginx": original, "other": original},
                              compose_owners={"nginx": "nginx", "other": "other"})
    nginx._preserve_legacy_compose(context, tmp_path)
    assert context.previous_compose_contents == {"nginx": original, "other": original}


def test_legacy_start_archives_the_current_snapshot_account(fresh_manager, tmp_path, monkeypatch):
    nginx = fresh_manager.containers["nginx"]
    fresh_manager.env_config.set("NGINX_HTTPS_ENABLE", True)
    nginx.__dict__["cert_image_revision"] = "1234567890abcdef"
    nginx.__dict__["extend_configs"] = {}
    nginx.__dict__["_rendered_site_files"] = ({}, False)
    cached = nginx.get_app_path("acme")
    cached.mkdir(parents=True)
    (cached / "account.conf").write_text("old attempt")
    candidate = nginx.get_app_path("certs", nginx.cert_image_revision, "live", "acme")
    candidate.mkdir(parents=True)
    (candidate / "account.conf").write_text("failed candidate")
    snapshot = tmp_path / "latest-snapshot"
    (snapshot / "acme").mkdir(parents=True)
    (snapshot / "acme" / "account.conf").write_text("current runtime")
    (snapshot / "certs").mkdir()
    monkeypatch.setattr(nginx, "_preserve_legacy_files", lambda: snapshot)
    monkeypatch.setattr(nginx, "_render_site_template", lambda *args: "configuration")
    monkeypatch.setattr(nginx.runtime, "chmod", lambda *args: None)
    context = SimpleNamespace(
        initial_existing_services={"nginx"},
        initial_runtime_state=SimpleNamespace(services=(SimpleNamespace(service="nginx", labels={}),)),
        previous_compose_contents={"nginx": yaml.safe_dump({"services": {"nginx": {"image": "old"}}})},
        compose_owners={"nginx": "nginx"}, write_files=lambda *args: None,
    )
    nginx.on_starting(context)
    with tarfile.open(str(nginx.get_app_path("acme-build-account.tar"))) as archive:
        assert archive.extractfile("account.conf").read() == b"current runtime"
    assert (cached / "account.conf").read_text() == "old attempt"
    assert (candidate / "account.conf").read_text() == "failed candidate"
    volumes = yaml.safe_load(context.previous_compose_contents["nginx"])["services"]["nginx"]["volumes"]
    assert {item["source"] for item in volumes} == {str(snapshot / "certs"), str(snapshot / "acme")}


@pytest.mark.parametrize("action,names,failed", [
    ("up", ["nginx"], "nginx"),
    ("restart", None, "app"),
    ("restart", None, "nginx"),
])
def test_first_upgrade_failure_restores_copied_tls_with_old_image(tmp_path, action, names, failed):
    manager = setup_case(tmp_path, [
        ("app", {"app": {"image": "app:new"}}),
        ("nginx", {"nginx": {"image": "nginx:new", "volumes": [
            {"type": "bind", "source": str(tmp_path / "candidate"), "target": "/etc/nginx/managed"},
        ]}}),
    ], running=("app", "nginx"))
    for path in (tmp_path / "compose" / "applied" / "services").iterdir():
        path.unlink()
    old_mount = str(tmp_path / "conf.d") + ":/etc/nginx/conf.d"
    original = yaml.safe_dump({"services": {"nginx": {"image": "nginx:old", "volumes": [old_mount]}}})
    compose_file = tmp_path / "compose" / "nginx.yml"
    compose_file.write_text(original)
    (tmp_path / "compose" / "app.yml").write_text(
        yaml.safe_dump({"services": {"app": {"image": "app:old"}}}))
    backup = tmp_path / "migration-backup-current"
    for name in ("certs", "acme"):
        (backup / name).mkdir(parents=True)
        (backup / name / "current").write_text("renewed-runtime")
    nginx = manager.containers["nginx"]
    method = builtin_container_type("100-nginx")._preserve_legacy_compose
    nginx.on_starting = lambda context: method(nginx, context, backup)
    restored = []

    def resolve(args):
        result = {"services": {}}
        for index, arg in enumerate(args):
            if arg == "--file":
                data = yaml.safe_load(Path(args[index + 1]).read_text())
                for name, spec in data["services"].items():
                    result["services"].setdefault(name, {}).update(spec)
        return result

    def process(*args, **kwargs):
        def apply():
            assert "--force-recreate" in args
            restored.append(resolve(args)["services"][args[-1]])
        return SimpleNamespace(args=args, check_call=apply)

    manager.runtime = SimpleNamespace(create_docker_process=process)
    runner = ComposeRunner(manager)
    runner._resolved_model = lambda process: resolve(process.args)
    manager.compose_runner.saved_service_models = runner.saved_service_models
    manager.compose_runner.apply_saved_services = runner.apply_saved_services
    manager.compose_runner.fail = failed
    with pytest.raises(ContainerError, match="apply failed " + failed):
        getattr(manager.compose_operations, action)(names)
    old = next(spec for spec in restored if spec["image"] == "sha256:old-nginx")
    assert old["volumes"][0] == old_mount
    mounts = {item["target"]: Path(item["source"]) for item in old["volumes"] if isinstance(item, dict)}
    assert mounts == {"/etc/certs": backup / "certs", "/root/.acme.sh": backup / "acme"}
    assert all((source / "current").read_text() == "renewed-runtime" for source in mounts.values())
    assert compose_file.read_text() == original
    recorded = yaml.safe_load(AppliedServiceModels(manager, manager.model).previous["nginx"])
    assert recorded["services"]["nginx"]["volumes"] == old["volumes"]
