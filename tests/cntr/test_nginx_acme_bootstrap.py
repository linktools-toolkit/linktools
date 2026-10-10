#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build-time ACME inputs and offline certificate installation with the current script."""

import os
import shutil
import subprocess
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def certificate_case(fresh_manager, tmp_path):
    container = fresh_manager.containers["nginx"]
    settings = fresh_manager.env_config
    settings.set("NGINX_ROOT_DOMAIN", "example.test")
    settings.set("NGINX_HTTPS_ENABLE", True)
    settings.set("ACME_DNS_API", "dns_cf")
    settings.set("CF_Token", "fake-secret")
    container.__dict__["acme_ssl_domains"] = ["example.test", "*.example.test"]
    for name in ("docker_file", "docker_compose"):
        container.__dict__.pop(name, None)

    seed = tmp_path / "seed"
    (seed / "certs").mkdir(parents=True)
    (seed / "acme").mkdir()
    (seed / "acme" / "account.key").write_text("existing-account")
    revision = "1234567890abcdef"
    (seed / "revision").write_text(revision + "\n")
    (seed / "primary").write_text("example.test\n")
    (seed / "domains").write_text("example.test\n*.example.test\n")
    key = seed / "certs/example.test_key.pem"
    chain = seed / "certs/example.test_fullchain.pem"
    subprocess.run([
        "openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes",
        "-keyout", str(key), "-out", str(chain), "-days", "2",
        "-subj", "/CN=example.test",
        "-addext", "subjectAltName=DNS:example.test,DNS:*.example.test",
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    shutil.copyfile(str(chain), str(seed / "certs/example.test_cert.pem"))

    (tmp_path / "certs").mkdir()
    mock = tmp_path / "acme.sh"
    mock.write_text("""#!/bin/sh
set -eu
case " $* " in
  *" --issue "*) echo 'Unexpected runtime issuance' >&2; exit 41 ;;
esac
while test "$#" -gt 0; do
  case "$1" in
    --cert-file) cp "$MOCK_SEED/certs/example.test_cert.pem" "$2"; shift 2 ;;
    --key-file) cp "$MOCK_SEED/certs/example.test_key.pem" "$2"; shift 2 ;;
    --fullchain-file) cp "$MOCK_SEED/certs/example.test_fullchain.pem" "$2"; shift 2 ;;
    *) shift ;;
  esac
done
""")
    mock.chmod(0o755)

    script = container.get_source_path("nginx-certificates").read_text(encoding="utf-8")
    script = script.replace("/etc/certs", str(tmp_path / "certs"))
    script = script.replace("/opt/nginx-initial", str(seed))
    script = script.replace("/opt/acme/acme.sh", str(mock))
    script = script.replace("/usr/local/bin/nginx-acme", str(mock))
    path = tmp_path / "nginx-certificates"
    path.write_text(script, encoding="utf-8")
    path.chmod(0o755)
    env = dict(os.environ, MOCK_SEED=str(seed), NGINX_HTTPS_PORT="443")
    return container, path, tmp_path / "certs", seed, env, revision


def run_script(case, action):
    _, path, _, _, env, _ = case
    return subprocess.run([str(path), action], env=env, capture_output=True, text=True)


def test_acme_issuance_happens_in_image_build_only(certificate_case):
    container, _, _, _, _, _ = certificate_case
    dockerfile = container.docker_file
    assert "AS acme-build" not in dockerfile
    assert "--issue --force" in dockerfile
    assert "--mount=type=secret,id=cntr_acme_dns" in dockerfile
    assert "ARG CNTR_BUILD_REVISION" in dockerfile
    assert "ENV CF_Token" not in dockerfile
    assert "nginx-certificates renew" in dockerfile
    spec = container.docker_compose["services"]["nginx"]
    assert spec["build"]["secrets"] == ["cntr_acme_account", "cntr_acme_dns"]


def test_offline_install_preserves_baked_cert_and_account(certificate_case):
    _, _, base, seed, _, revision = certificate_case
    result = run_script(certificate_case, "install")
    assert result.returncode == 0, result.stderr
    current = base / revision / "live"
    assert current.is_symlink()
    assert (current / "example.test_fullchain.pem").read_bytes() == (
        seed / "certs/example.test_fullchain.pem").read_bytes()
    assert (current / "acme/account.key").read_text() == "existing-account"
    assert (current / "renewal/example.test_key.pem").is_file()


def test_repeated_install_keeps_same_certificate_generation(certificate_case):
    _, _, base, _, _, revision = certificate_case
    assert run_script(certificate_case, "install").returncode == 0
    current = base / revision / "live"
    original = os.readlink(str(current))
    assert run_script(certificate_case, "install").returncode == 0
    assert os.readlink(str(current)) == original


def test_invalid_baked_key_never_publishes_live_certificate(certificate_case):
    _, _, base, seed, _, revision = certificate_case
    (seed / "certs/example.test_key.pem").unlink()
    result = run_script(certificate_case, "install")
    assert result.returncode != 0
    assert not (base / revision / "live").exists()


def test_invalid_san_never_publishes_certificate(certificate_case):
    _, _, base, seed, _, revision = certificate_case
    (seed / "domains").write_text("example.test\n*.absent.test\n")
    result = run_script(certificate_case, "install")
    assert result.returncode != 0
    assert not (base / revision / "live").exists()


def test_check_uses_revision_specific_state(certificate_case):
    _, _, base, seed, _, revision = certificate_case
    assert run_script(certificate_case, "check").returncode == 0
    assert (base / revision / "live").is_symlink()
    assert (base / revision / "live/example.test_key.pem").is_file()
    assert (seed / "acme/account.key").read_text() == "existing-account"


def test_https_validation_uses_writable_disposable_certificate_tree(certificate_case, monkeypatch):
    container, _, base, _, _, _ = certificate_case
    assert any(mount.endswith(":/etc/certs")
               for mount in container.docker_compose["services"]["nginx"]["volumes"])
    generated = base.parent / "generated"
    credentials = base.parent / "dns.env"
    model = {"services": {"nginx": {"image": "nginx:target", "volumes": [
        {"type": "bind", "source": str(base), "target": "/etc/certs"},
        {"type": "bind", "source": str(generated), "target": "/etc/nginx/managed", "read_only": True},
        {"type": "bind", "source": str(credentials), "target": "/run/acme-secrets", "read_only": True},
    ]}}}
    observed = []

    def execute(args, **kwargs):
        mounts = [args[index + 1] for index, arg in enumerate(args) if arg == "--mount"]
        certs = next(mount for mount in mounts if "target=/etc/certs" in mount)
        assert not certs.endswith(",readonly")
        source = Path(certs.split("source=", 1)[1].split(",", 1)[0])
        assert source != base
        assert not any("source=" + str(base) + "," in mount for mount in mounts)
        assert (source / container.cert_image_revision / "previous").is_dir()
        assert "type=bind,source={},target=/etc/nginx/managed,readonly".format(generated) in mounts
        assert "type=bind,source={},target=/run/acme-secrets,readonly".format(credentials) in mounts
        assert "/usr/local/bin/nginx-certificates check && " in args[-1]
        observed.append(source)
        return SimpleNamespace(succeeded=True, stdout="", stderr="", returncode=0)

    monkeypatch.setattr(container.runtime, "create_docker_process", lambda *args, **kwargs: args)
    monkeypatch.setattr(container.manager.structured_runner, "execute", execute)
    container.on_check(SimpleNamespace(compose_model=model))
    assert len(observed) == 1
    assert not observed[0].exists()


def test_new_revision_does_not_modify_older_certificate(certificate_case):
    _, _, base, seed, _, revision = certificate_case
    assert run_script(certificate_case, "install").returncode == 0
    first = base / revision / "live"
    original_link = os.readlink(str(first))
    (seed / "revision").write_text("fedcba0987654321\n")
    assert run_script(certificate_case, "install").returncode == 0
    assert os.readlink(str(first)) == original_link
    assert (base / "fedcba0987654321/live").is_symlink()


@pytest.mark.parametrize("source", ["target", "running", "seed", "empty"])
def test_build_account_uses_current_storage_without_migrating_old_files(fresh_manager, monkeypatch, source):
    container = fresh_manager.containers["nginx"]
    fresh_manager.env_config.set("NGINX_HTTPS_ENABLE", True)
    revision, running_revision = "1234567890abcdef", "fedcba0987654321"
    container.__dict__.update(cert_image_revision=revision, extend_configs={}, _rendered_site_files=({}, False))
    accounts = {
        "target": container.get_app_path("certs", revision, "live", "acme"),
        "running": container.get_app_path("certs", running_revision, "live", "acme"),
        "seed": container.get_app_path("acme"),
    }
    available = {"target": ("target", "running", "seed"), "running": ("running", "seed"),
                 "seed": ("seed",), "empty": ()}
    for name in available[source]:
        if name in accounts:
            accounts[name].mkdir(parents=True, exist_ok=True)
            (accounts[name] / "account.conf").write_text(name)
    old_files = [container.get_app_path(name, "untouched")
                 for name in ("conf.d", "migration-backup-existing", "certs/live/acme")]
    for path in old_files:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("old layout")

    def fail_process(*args, **kwargs):
        raise AssertionError("Preparation must not copy from or change a container")

    monkeypatch.setattr(container.runtime, "create_docker_process", fail_process)
    monkeypatch.setattr(container.runtime, "chmod", lambda *args: None)
    monkeypatch.setattr(container, "_render_site_template", lambda *args: "configuration")
    context = SimpleNamespace(
        initial_runtime_state=SimpleNamespace(services=(SimpleNamespace(service="nginx", labels={
            "io.linktools.nginx.certificate-revision": running_revision,
        }),)),
        previous_compose_contents={"nginx": "original model"}, write_files=lambda *args: None,
    )
    container.on_starting(context)
    with tarfile.open(str(container.get_app_path("acme-build-account.tar"))) as archive:
        if source == "empty":
            assert archive.getnames() == []
        else:
            assert archive.extractfile("account.conf").read().decode() == source
    assert context.previous_compose_contents == {"nginx": "original model"}
    assert all(path.read_text() == "old layout" for path in old_files)
    assert list(container.get_app_path().glob("migration-backup-*")) == [old_files[1].parent]


def test_install_uses_image_seed_without_importing_flat_layout(certificate_case):
    _, _, base, seed, _, revision = certificate_case
    old = base / "old"
    shutil.copytree(str(seed / "certs"), str(old))
    shutil.copytree(str(seed / "acme"), str(old / "acme"))
    (old / "acme/account.key").write_text("old-layout-account")
    (old / "build-revision").write_text(revision)
    (base / "live").symlink_to("old", target_is_directory=True)
    result = run_script(certificate_case, "install")
    assert result.returncode == 0, result.stderr
    assert (base / revision / "live/acme/account.key").read_text() == "existing-account"
    assert (old / "acme/account.key").read_text() == "old-layout-account"
    assert os.readlink(str(base / "live")) == "old"
