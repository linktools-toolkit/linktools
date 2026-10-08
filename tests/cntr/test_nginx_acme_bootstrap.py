#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build-time issuance and offline certificate seeding with fake ACME only."""
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from _harness import builtin_consumer_type


_ACME_CLIENT = '''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--home")
parser.add_argument("--config-home", required=True)
parser.add_argument("--server", default="zerossl")
parser.add_argument("--accountemail", default="")
parser.add_argument("--issue", action="store_true")
parser.add_argument("--install-cert", action="store_true")
parser.add_argument("--domain", action="append")
parser.add_argument("--dns")
parser.add_argument("--cert-file")
parser.add_argument("--key-file")
parser.add_argument("--fullchain-file")
parser.add_argument("--reloadcmd")
args = parser.parse_args()
home = Path(args.config_home)
home.mkdir(parents=True, exist_ok=True)
account = home / "account.json"
state = json.loads(account.read_text()) if account.exists() else {}
if args.issue:
    if args.server == "zerossl" and not (args.accountemail or state.get("email")):
        parser.exit(1, "ZeroSSL account email is required\\n")
    state.setdefault("key", "initial-account-key")
    state.setdefault("email", args.accountemail)
    state.update(server=args.server, domains=args.domain, dns=args.dns)
    account.write_text(json.dumps(state))
    (home / "account.conf").write_text("SAVED_CF_Token='account-secret'\\n")
    domain = home / (args.domain[0] + "_ecc")
    domain.mkdir(exist_ok=True)
    (domain / (args.domain[0] + ".conf")).write_text(
        "Le_Domain='" + args.domain[0] + "'\\nLe_API='" + args.server + "'\\n"
        "CF_Token='domain-secret'\\nSAVED_CF_Token='saved-domain-secret'\\n"
        "ACMEDNS_PASSWORD='other-provider-secret'\\n")
elif args.install_cert:
    if state.get("server") != args.server or state.get("domains") != args.domain:
        parser.exit(1, "Certificate must be installed from its issuing CA\\n")
    for name in (args.cert_file, args.key_file, args.fullchain_file):
        Path(name).write_text("simulated-certificate")
else:
    parser.error("Expected issuance or installation")
'''

NginxGeneration = builtin_consumer_type("100-nginx")


def run_shell(script, binary=None):
    environment = dict(os.environ)
    environment["PATH"] = os.pathsep.join(filter(None, (
        str(binary) if binary else None, str(Path(sys.executable).parent), os.defpath)))
    return subprocess.run(["sh", "-ec", script], env=environment,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)


@pytest.fixture
def build_fixture(fresh_manager, tmp_path):
    container = fresh_manager.containers["nginx"]
    fresh_manager.env_config.set("NGINX_ROOT_DOMAIN", "example.test")
    fresh_manager.env_config.set("NGINX_HTTPS_ENABLE", True)
    fresh_manager.env_config.set("ACME_DNS_API", "dns_cf")
    fresh_manager.env_config.set("CF_Key", "fake-key")
    fresh_manager.env_config.set("CF_Token", "fake-secret")
    for name in ("docker_file", "docker_compose", "services", "extend_configs"):
        container.__dict__.pop(name, None)
    for name in ("_acme_ssl_domains", "acme_ssl_domains_args", "acme_ssl_certificate_args"):
        container.manager.integration_consumers["nginx"].__dict__.pop(name, None)
    root = tmp_path / "build state"
    root.mkdir()
    binary = tmp_path / "bin"
    binary.mkdir()
    client = binary / "acme.sh"
    client.write_text(_ACME_CLIENT, encoding="utf-8")
    client.chmod(0o755)
    return container, root, binary


def build_command(container, root):
    text = container.render_template(container.get_source_path("Dockerfile"))
    command = text.split("    . /run/secrets/nginx-acme &&", 1)[1].split("#", 1)[0]
    for source, dest in (("/root/.acme.sh", root / "acme"), ("/etc/certs", root / "certs"),
                         ("/opt/nginx-initial", root / "initial")):
        command = command.replace(source, shlex.quote(str(dest)))
    (root / "certs").mkdir(exist_ok=True)
    return command


def test_initial_issuance_and_installation_are_build_steps(build_fixture):
    container, root, binary = build_fixture
    result = run_shell(build_command(container, root), binary)
    assert result.returncode == 0, result.stderr
    state = json.loads((root / "initial/acme/account.json").read_text())
    assert state["server"] == "letsencrypt"
    assert state["domains"][:2] == ["example.test", "*.example.test"]
    assert (root / "initial/certs/example.test_fullchain.pem").read_text() == "simulated-certificate"
    assert "--mount=type=secret,id=nginx-acme,required=true" in container.docker_file
    assert "fake-secret" not in container.docker_file
    assert "ENV Ali_" not in container.docker_file
    assert "rm -f /root/.acme.sh/account.conf" in container.docker_file
    assert not (root / "initial/acme/account.conf").exists()
    domain = (root / "initial/acme/example.test_ecc/example.test.conf").read_text()
    assert "Le_Domain='example.test'" in domain
    assert "Le_API='letsencrypt'" in domain
    assert "secret" not in domain
    assert "CF_Token" not in domain and "ACMEDNS_PASSWORD" not in domain


def test_build_server_and_email_are_shell_quoted(build_fixture, fresh_manager):
    container, root, binary = build_fixture
    server = "https://ca.example.test/directory?tenant=o'reilly&profile=one"
    email = "ops+o'reilly@example.test"
    fresh_manager.env_config.set("ACME_SERVER", server)
    fresh_manager.env_config.set("ACME_ACCOUNT_EMAIL", email)
    result = run_shell(build_command(container, root), binary)
    assert result.returncode == 0, result.stderr
    state = json.loads((root / "initial/acme/account.json").read_text())
    assert (state["server"], state["email"]) == (server, email)


def test_failed_build_issuance_does_not_install_or_seed(build_fixture, fresh_manager):
    container, root, binary = build_fixture
    fresh_manager.env_config.set("ACME_SERVER", "zerossl")
    result = run_shell(build_command(container, root), binary)
    assert result.returncode != 0
    assert not list((root / "certs").iterdir())
    assert not (root / "initial").exists()


def test_build_secret_is_temporary_and_render_is_read_only(build_fixture):
    container, root, binary = build_fixture
    path = container.acme_build_secret_path
    model = container.docker_compose
    assert model["secrets"]["nginx-acme"]["file"] == str(path)
    assert model["services"]["nginx"]["build"]["secrets"] == [
        {"source": "nginx-acme", "target": "nginx-acme"}]
    assert not path.exists()
    container.get_docker_compose_file()
    assert path.stat().st_mode & 0o777 == 0o600
    assert "fake-secret" in path.read_text()
    assert container.get_docker_context_path() not in path.parents
    container._acme_build_secret_cleanup()
    assert not path.exists()


@pytest.fixture
def seeded_mounts(build_fixture):
    container, root, binary = build_fixture
    for name in ("certs", "acme"):
        (root / "initial" / name).mkdir(parents=True)
    (root / "initial/certs/example.test_key.pem").write_text("image-key")
    (root / "initial/certs/example.test_fullchain.pem").write_text("image-cert")
    (root / "initial/acme/example.test_ecc").mkdir()
    (root / "initial/acme/example.test_ecc/example.test.conf").write_text(
        "Le_Domain='example.test'\nLe_ReloadCmd='killall nginx'\n")
    (root / "initial/acme/account.key").write_text("image-account-key")
    text = container.get_source_path("nginx-init-certificates").read_text()
    for source, dest in (("/root/.acme.sh", root / "acme"), ("/etc/certs", root / "certs"),
                         ("/opt/nginx-initial", root / "initial")):
        text = text.replace(source, str(dest))
    # Paths above are substituted into both shell syntax and quoted literals.
    # Use a no-space execution root for this static container script.
    flat = root.parent / "mounts"
    root.rename(flat)
    return text.replace(str(root), str(flat)), flat


def test_empty_mounts_are_seeded_offline_and_idempotently(seeded_mounts):
    script, root = seeded_mounts
    for _ in range(2):
        result = run_shell(script)
        assert result.returncode == 0, result.stderr
    assert (root / "certs/example.test_key.pem").read_text() == "image-key"
    assert (root / "acme/account.key").read_text() == "image-account-key"
    config = (root / "acme/example.test_ecc/example.test.conf").read_text()
    assert config.count("Le_ReloadCmd=") == 1
    assert "nginx-reload" in config
    assert "--issue" not in script and "--install-cert" not in script


def test_existing_mounts_preserve_keys_certificates_and_account_settings(seeded_mounts):
    script, root = seeded_mounts
    (root / "certs").mkdir()
    (root / "acme").mkdir()
    (root / "certs/example.test_fullchain.pem").write_text("newer-certificate")
    (root / "certs/example.test_key.pem").write_text("existing-key")
    (root / "acme/account.key").write_text("existing-account-key")
    (root / "acme/account.conf").write_text("existing-settings")
    result = run_shell(script)
    assert result.returncode == 0, result.stderr
    assert (root / "certs/example.test_fullchain.pem").read_text() == "newer-certificate"
    assert (root / "certs/example.test_key.pem").read_text() == "existing-key"
    assert (root / "acme/account.key").read_text() == "existing-account-key"
    assert (root / "acme/account.conf").read_text() == "existing-settings"
    assert not (root / "acme/example.test_ecc").exists()


def test_prepare_only_seeds_and_validates_without_network(build_fixture, monkeypatch):
    container, root, binary = build_fixture
    calls = []
    monkeypatch.setattr(container.manager.compose_runner, "validate_service",
                        lambda *args, **kwargs: calls.append((args, kwargs)))
    container.manager.integration_consumers["nginx"].on_prepare(SimpleNamespace(initial_services=()))
    assert len(calls) == 1
    args, kwargs = calls[0]
    assert not kwargs.get("network")
    assert "nginx-init-certificates" in args[2][-1]
    assert "-checkend 0" in args[2][-1]
    assert "--issue" not in args[2][-1] and "--install-cert" not in args[2][-1]


def test_http_does_not_resolve_acme_or_create_secret(build_fixture, fresh_manager, monkeypatch):
    container, root, binary = build_fixture
    fresh_manager.env_config.set("NGINX_HTTPS_ENABLE", False)
    original = container.get_config
    def config(key, **kwargs):
        assert not str(key).startswith("ACME_")
        return original(key, **kwargs)
    monkeypatch.setattr(container, "get_config", config)
    container.manager.integration_consumers["nginx"].on_prepare(SimpleNamespace(initial_services=()))
    for name in ("docker_file", "docker_compose", "services"):
        container.__dict__.pop(name, None)
    assert "acme.sh" not in container.docker_file
    assert "secrets" not in container.docker_compose
    container.get_docker_compose_file()
    assert not container.acme_build_secret_path.exists()


@pytest.mark.parametrize("running,valid,expected", [(False, True, []), (True, True, ["-t", "reload"]), (True, False, ["-t"])])
def test_renewal_reload_checks_config_and_never_stops_nginx(build_fixture, running, valid, expected):
    container, root, binary = build_fixture
    pid = root / "nginx.pid"
    if running:
        pid.write_text("123")
    calls = root / "nginx-calls"
    nginx = binary / "nginx"
    nginx.write_text("#!/bin/sh\nprintf '%s\\n' \"$*\" >> " + shlex.quote(str(calls)) +
                     ("\nexit 0\n" if valid else "\nexit 1\n"))
    nginx.chmod(0o755)
    script = container.get_source_path("nginx-reload").read_text().replace(
        "/var/run/nginx.pid", shlex.quote(str(pid)))
    # Static script paths are quoted; the fake executable records arguments only.
    result = run_shell(script, binary)
    lines = calls.read_text().splitlines() if calls.exists() else []
    assert [line.split()[-1] for line in lines] == expected
    assert all("/etc/nginx/generated/current/nginx.conf" in line for line in lines)
    assert result.returncode == (1 if running and not valid else 0)
    assert "killall" not in script


def test_dns_secret_preserves_shell_and_compose_characters(build_fixture, fresh_manager):
    container, root, binary = build_fixture
    value = "fake'o$reilly\nsecond-line"
    fresh_manager.env_config.set("CF_Token", value)
    container.get_docker_compose_file()
    script = ". " + shlex.quote(str(container.acme_build_secret_path)) + " && printf '%s' \"$CF_Token\""
    result = run_shell(script)
    assert result.returncode == 0, result.stderr
    assert result.stdout == value
    assert container.docker_compose["services"]["nginx"]["environment"]["CF_Token"] == value.replace("$", "$$")
    assert value not in container.docker_file
