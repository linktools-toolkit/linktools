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

from linktools.cntr import ContainerError


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
    for name in ("docker_file", "docker_compose", "services", "extend_configs",
                 "acme_ssl_domains", "acme_ssl_domains_args", "acme_ssl_certificate_args"):
        container.__dict__.pop(name, None)
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
    command = "acme.sh" + text.split("RUN acme.sh", 1)[1].split("#", 1)[0]
    for source, dest in (("/root/.acme.sh", root / "acme"), ("/etc/certs", root / "certs"),
                         ("/opt/nginx-initial", root / "initial")):
        command = command.replace(source, shlex.quote(str(dest)))
    (root / "certs").mkdir(exist_ok=True)
    return command


def test_build_parameters_are_owned_by_container(build_fixture, monkeypatch):
    container, _, _ = build_fixture
    container.__dict__["sites"] = {
        "active": SimpleNamespace(enabled=True, https=True,
                                  cert_domains=("app.example.test", "example.test", "app.example.test")),
        "http": SimpleNamespace(enabled=True, https=False, cert_domains=("http.example.test",)),
        "disabled": SimpleNamespace(enabled=False, https=True, cert_domains=("disabled.example.test",)),
    }
    monkeypatch.setattr(type(container.manager), "generated_configs", property(
        lambda self: pytest.fail("Build templates must not access the generated config registry")))
    assert container.acme_ssl_domains == ["example.test", "*.example.test", "app.example.test"]
    assert "--domain app.example.test" in container.docker_file
    assert "--fullchain-file /etc/certs/example.test_fullchain.pem" in container.docker_file
    assert "secrets" not in container.docker_compose


def test_initial_issuance_and_installation_are_build_steps(build_fixture):
    container, root, binary = build_fixture
    result = run_shell(build_command(container, root), binary)
    assert result.returncode == 0, result.stderr
    state = json.loads((root / "initial/acme/account.json").read_text())
    assert state["server"] == "letsencrypt"
    assert state["domains"][:2] == ["example.test", "*.example.test"]
    assert (root / "initial/certs/example.test_fullchain.pem").read_text() == "simulated-certificate"
    assert 'ENV CF_Token="fake-secret"' in container.docker_file
    assert "--mount=" not in container.docker_file
    assert (root / "initial/acme/account.conf").read_text() == "SAVED_CF_Token='account-secret'\n"
    domain = (root / "initial/acme/example.test_ecc/example.test.conf").read_text()
    assert "Le_Domain='example.test'" in domain
    assert "Le_API='letsencrypt'" in domain
    assert "CF_Token='domain-secret'" in domain
    assert "ACMEDNS_PASSWORD='other-provider-secret'" in domain


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


def test_build_and_renewal_share_dns_environment_without_secret_files(build_fixture):
    container, root, binary = build_fixture
    model = container.docker_compose
    assert "secrets" not in model
    assert "secrets" not in model["services"]["nginx"]["build"]
    assert model["services"]["nginx"]["environment"]["CF_Token"] == "fake-secret"
    assert 'ENV CF_Token="fake-secret"' in container.docker_file
    assert "--cron --home /opt/acme --config-home /root/.acme.sh" in container.docker_file
    container.get_docker_compose_file()
    assert not container.get_temp_path("acme-build.env").exists()


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
    container.on_prepare_config(SimpleNamespace(initial_services=()))
    assert len(calls) == 1
    args, kwargs = calls[0]
    assert not kwargs.get("network")
    assert "nginx-init-certificates" in args[2][-1]
    assert "-checkend 0" in args[2][-1]
    assert "--issue" not in args[2][-1] and "--install-cert" not in args[2][-1]


def test_http_does_not_resolve_acme_or_create_dns_environment(build_fixture, fresh_manager, monkeypatch):
    container, root, binary = build_fixture
    fresh_manager.env_config.set("NGINX_HTTPS_ENABLE", False)
    original = container.get_config
    def config(key, **kwargs):
        assert not str(key).startswith("ACME_")
        return original(key, **kwargs)
    monkeypatch.setattr(container, "get_config", config)
    container.on_prepare_config(SimpleNamespace(initial_services=()))
    for name in ("docker_file", "docker_compose", "services"):
        container.__dict__.pop(name, None)
    assert "acme.sh" not in container.docker_file
    assert "secrets" not in container.docker_compose
    container.get_docker_compose_file()
    assert not container.get_temp_path("acme-build.env").exists()


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


def test_dns_environment_preserves_dockerfile_and_compose_characters(build_fixture, fresh_manager):
    container, root, binary = build_fixture
    value = "fake'o$reilly\\path\"suffix"
    fresh_manager.env_config.set("CF_Token", value)
    container.get_docker_compose_file()
    expected = value.replace("\\", "\\\\").replace('"', '\\"').replace("$", "\\$")
    assert 'ENV CF_Token="' + expected + '"' in container.docker_file
    assert container.docker_compose["services"]["nginx"]["environment"]["CF_Token"] == value.replace("$", "$$")


@pytest.mark.parametrize("value", ["first\nsecond", "first\rsecond"])
def test_multiline_dns_environment_fails_without_exposing_value(build_fixture, fresh_manager, value):
    container, root, binary = build_fixture
    fresh_manager.env_config.set("CF_Token", value)
    with pytest.raises(ContainerError, match="DNS credentials must be single-line") as caught:
        container.docker_file
    assert value not in str(caught.value)
