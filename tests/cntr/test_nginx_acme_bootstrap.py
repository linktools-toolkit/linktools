#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build-time ACME issue and offline runtime promotion of baked certificates."""

import os
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest


def _issue_pair(folder, label, domains):
    key = folder / (label + ".key")
    pem = folder / (label + ".pem")
    subprocess.run([
        "openssl", "req", "-x509", "-nodes", "-newkey", "rsa:2048",
        "-days", "2", "-keyout", str(key), "-out", str(pem),
        "-subj", "/CN=example.test",
        "-addext", "subjectAltName=" + ",".join("DNS:" + d for d in domains),
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return pem, key


@pytest.fixture
def certificate_case(fresh_manager, tmp_path):
    container = fresh_manager.containers["nginx"]
    cfg = fresh_manager.env_config
    cfg.set("NGINX_ROOT_DOMAIN", "example.test")
    cfg.set("NGINX_HTTPS_ENABLE", True)
    cfg.set("ACME_DNS_API", "dns_cf")
    cfg.set("CF_Token", "fake-token")
    for key in ("docker_file", "docker_compose", "services", "extend_configs",
                "acme_ssl_domains"):
        container.__dict__.pop(key, None)

    certs = tmp_path / "certs"
    versions = certs / "versions"
    legacy = versions / "legacy"
    legacy.mkdir(parents=True)
    (tmp_path / "acme").mkdir()
    (tmp_path / "bin").mkdir()
    old, old_key = _issue_pair(tmp_path, "old", ("example.test", "*.example.test"))
    new, new_key = _issue_pair(tmp_path, "new", (
        "example.test", "*.example.test", "*.code.example.test"))
    for source, suffix in ((old, "fullchain"), (old_key, "key")):
        shutil.copyfile(str(source), str(legacy / ("example.test_" + suffix + ".pem")))
    (legacy / "primary").write_text("example.test\n")
    (legacy / "port").write_text("443\n")
    (legacy / "domains").write_text("example.test\n*.example.test\n")
    (certs / "live").symlink_to("versions/legacy")
    requested = versions / "pending"
    requested.mkdir()
    (requested / "primary").write_text("example.test\n")
    (requested / "port").write_text("443\n")
    (requested / "domains").write_text(
        "example.test\n*.example.test\n*.code.example.test\n")
    (tmp_path / "acme/account.key").write_text("existing-account-key")
    (legacy / "acme").mkdir()
    (legacy / "acme/account.key").write_text("existing-account-key")
    seed = tmp_path / "seed"
    (seed / "certs").mkdir(parents=True)
    (seed / "acme").mkdir()
    (seed / "acme/account.key").write_text("build-account-key")
    for source, suffix in ((new, "fullchain"), (new_key, "key"), (new, "cert")):
        shutil.copyfile(str(source), str(seed / "certs" / ("example.test_" + suffix + ".pem")))

    client = tmp_path / "bin/acme.sh"
    client.write_text("""#!/usr/bin/env python3
import os
import sys
import shutil
from pathlib import Path
args = sys.argv[1:]
if "--issue" in args:
    raise SystemExit("Runtime certificate preparation must never issue new certificates")
elif "--install-cert" in args:
    root = Path(os.environ["MOCK_CERTIFICATES"])
    for flag, name in (("--cert-file", "new.pem"),
                       ("--fullchain-file", "new.pem"),
                       ("--key-file", "new.key")):
        shutil.copyfile(str(root / name), args[args.index(flag) + 1])
elif "--cron" in args:
    if os.environ.get("MOCK_RENEW"):
        root = Path(os.environ["MOCK_CERTIFICATES"])
        target = Path(os.environ["MOCK_RENEWAL"])
        for suffix, source in (("cert", "new.pem"), ("fullchain", "new.pem"), ("key", "new.key")):
            shutil.copyfile(str(root / source), str(target / ("example.test_" + suffix + ".pem")))
else:
    raise SystemExit(3)
""")
    client.chmod(0o755)
    script = container.get_source_path("nginx-certificates").read_text()
    for before, after in (
        ("/etc/certs", str(certs)),
        ("/root/.acme.sh", str(tmp_path / "acme")),
        ("/opt/nginx-initial", str(seed)),
        ("/opt/acme/acme.sh", str(client)),
        ("/var/run/nginx.pid", str(tmp_path / "fake.pid")),
    ):
        script = script.replace(before, after)
    path = tmp_path / "bin/nginx-certificates"
    path.write_text(script)
    path.chmod(0o755)
    environment = dict(os.environ, MOCK_CERTIFICATES=str(tmp_path),
                       MOCK_RENEWAL=str(certs / ".renewal"),
                       MOCK_STATE=str(tmp_path / "issue.args"),
                       PATH=str(tmp_path / "bin") + os.pathsep + os.environ["PATH"])
    return container, tmp_path, path, environment


def _run(path, environ, *args):
    return subprocess.run([str(path), *args], env=environ,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          universal_newlines=True)


def test_acme_is_issued_during_build_and_rebuilt_for_new_domains(certificate_case):
    container, _, _, _ = certificate_case
    dockerfile = container.docker_file
    assert "RUN acme.sh" in dockerfile and "--issue" in dockerfile
    assert "COPY nginx-certificates nginx-reload" in dockerfile
    assert "/opt/nginx-initial/certs" in dockerfile
    assert "nginx-certificates renew" in dockerfile
    assert container.docker_compose["services"]["nginx"]["environment"]["CF_Token"] == "fake-token"
    assert container.docker_compose["services"]["nginx"]["build"]
    old_image = container.docker_compose["services"]["nginx"]["image"]
    container.__dict__["acme_ssl_domains"] = ["example.test", "*.example.test", "*.code.example.test"]
    container.__dict__.pop("cert_image_revision", None)
    container.__dict__.pop("docker_compose", None)
    assert container.docker_compose["services"]["nginx"]["image"] != old_image


def test_changed_san_list_stages_without_touching_live_certificate(certificate_case):
    _, root, script, env = certificate_case
    desired = root / "certs/versions/pending/domains"
    assert _run(script, env, "check", "example.test", str(desired)).returncode != 0
    old = (root / "certs/live/example.test_fullchain.pem").read_bytes()
    result = _run(script, env, "prepare", "pending", "example.test")
    assert result.returncode == 0, result.stderr
    assert (root / "certs/live").readlink() == Path("versions/legacy")
    assert (root / "certs/live/example.test_fullchain.pem").read_bytes() == old
    assert (root / "acme/account.key").read_text() == "existing-account-key"
    assert (root / "certs/versions/pending/example.test_key.pem").stat().st_mode & 0o777 == 0o600
    assert (root / "certs/versions/pending/acme/account.key").read_text() == "build-account-key"
    assert not (root / "issue.args").exists()
    assert _run(script, env, "activate", "pending").returncode == 0
    assert (root / "certs/live").readlink() == Path("versions/pending")
    assert _run(script, env, "check", "example.test", str(desired)).returncode == 0


def test_missing_baked_certificate_fails_without_runtime_issuance(certificate_case):
    _, root, script, env = certificate_case
    (root / "seed/certs/example.test_key.pem").unlink()
    result = _run(script, env, "prepare", "pending", "example.test")
    assert result.returncode != 0
    assert (root / "certs/live").readlink() == Path("versions/legacy")
    assert (root / "acme/account.key").read_text() == "existing-account-key"
    assert not (root / "issue.args").exists()


def test_renewal_promotes_only_validated_certificate_versions(certificate_case):
    _, root, script, env = certificate_case
    original = (root / "certs/live/example.test_fullchain.pem").read_bytes()
    assert _run(script, env, "prepare", "pending", "example.test").returncode == 0

    stage = root / "certs/.renewal"
    shutil.copyfile(str(root / "old.pem"), str(stage / "example.test_fullchain.pem"))
    shutil.copyfile(str(root / "old.key"), str(stage / "example.test_key.pem"))
    env = dict(env, MOCK_RENEW="1")
    result = _run(script, env, "renew")
    assert result.returncode == 0, result.stderr
    live = (root / "certs/live").readlink()
    assert str(live).startswith("versions/renew-")
    assert (root / "certs/live/example.test_fullchain.pem").read_bytes() != original
    assert (root / "certs/versions/legacy/example.test_fullchain.pem").read_bytes() == original


def test_renewal_never_promotes_stale_installed_certificates(certificate_case):
    _, root, script, env = certificate_case
    original = (root / "certs/live/example.test_fullchain.pem").read_bytes()
    assert _run(script, env, "prepare", "pending", "example.test").returncode == 0
    result = _run(script, env, "renew")
    assert result.returncode == 0, result.stderr
    assert (root / "certs/live").readlink() == Path("versions/legacy")
    assert (root / "certs/live/example.test_fullchain.pem").read_bytes() == original


def test_failed_nginx_reload_restores_old_certificate(certificate_case):
    _, root, script, env = certificate_case
    assert _run(script, env, "prepare", "pending", "example.test").returncode == 0
    (root / "fake.pid").write_text("123")
    nginx = root / "bin/nginx"
    nginx.write_text("#!/bin/sh\nexit 1\n")
    nginx.chmod(0o755)
    result = _run(script, env, "activate", "pending")
    assert result.returncode != 0
    assert (root / "certs/live").readlink() == Path("versions/legacy")


def test_http_does_not_resolve_dns_secrets(certificate_case, monkeypatch):
    container, _, _, _ = certificate_case
    container.env_config.set("NGINX_HTTPS_ENABLE", False)
    def reject(*args, **kwargs):
        raise AssertionError("HTTP-only configuration must not contact Docker")
    monkeypatch.setattr(container.manager.compose_runner, "validate_service", reject)
    container.on_prepare_config(SimpleNamespace(initial_services=()))
    container.__dict__.pop("docker_file", None)
    assert "nginx-certificates" not in container.docker_file


def test_preparation_reuses_matching_certificate_without_issuance(certificate_case, monkeypatch):
    container, root, _, _ = certificate_case
    monkeypatch.setattr(container, "get_app_path", lambda *parts: root.joinpath(*parts))
    calls = []

    def validate(context, service, command, **kwargs):
        calls.append((command[1], kwargs))
        return SimpleNamespace(succeeded=True)

    monkeypatch.setattr(container.manager.compose_runner, "validate_service", validate)
    container.on_prepare_config(SimpleNamespace(initial_services=()))

    assert container._certificate_version == "legacy"
    assert [name for name, _ in calls] == ["check", "configure"]
    assert (root / "certs/live").readlink() == Path("versions/legacy")


def test_preparation_can_enable_https_on_running_http_only_nginx(certificate_case, monkeypatch):
    container, root, _, _ = certificate_case
    (root / "certs/live").unlink()
    (root / "generated").mkdir()
    (root / "generated/current").symlink_to("http-only")
    monkeypatch.setattr(container, "get_app_path", lambda *parts: root.joinpath(*parts))

    def validate(context, service, command, **kwargs):
        return SimpleNamespace(succeeded=command[1] != "check")

    monkeypatch.setattr(container.manager.compose_runner, "validate_service", validate)
    container.on_prepare_config(SimpleNamespace(initial_services=("nginx",)))
    assert (root / "certs/live").readlink() == Path("versions") / container._certificate_version


def test_preparation_stages_added_names_without_publishing(certificate_case, monkeypatch):
    container, root, _, _ = certificate_case
    monkeypatch.setattr(container, "get_app_path", lambda *parts: root.joinpath(*parts))
    container.__dict__["acme_ssl_domains"] = ["example.test", "*.example.test", "*.code.example.test"]
    calls = []

    def validate(context, service, command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(succeeded=command[1] != "check")

    monkeypatch.setattr(container.manager.compose_runner, "validate_service", validate)
    container.on_prepare_config(SimpleNamespace(initial_services=()))

    assert container._certificate_version != "legacy"
    assert (root / "certs/live").readlink() == Path("versions/legacy")
    assert (root / "certs/versions" / container._certificate_version / "domains").read_text().splitlines() == [
        "example.test", "*.example.test", "*.code.example.test",
    ]
    assert [call[0][1] for call in calls] == ["check", "prepare"]
    assert not calls[1][1].get("network")
