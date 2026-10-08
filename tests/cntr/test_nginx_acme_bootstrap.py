#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ACME bootstrap against an empty mounted config without Docker or a CA."""
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest

from linktools.cntr import ContainerError
from _harness import builtin_consumer_type

if TYPE_CHECKING:
    from linktools.cntr import ContainerManager


_ACME_CLIENT = '''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
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
elif args.install_cert:
    if state.get("server") != args.server or state.get("domains") != args.domain:
        parser.exit(1, "Certificate must be installed from its issuing CA\\n")
    for name in (args.cert_file, args.key_file, args.fullchain_file):
        Path(name).write_text("simulated-certificate")
else:
    parser.error("Expected issuance or installation")
'''


NginxGeneration = builtin_consumer_type("100-nginx")


@pytest.fixture
def acme_bootstrap(fresh_manager: "ContainerManager", tmp_path: Path,
                   monkeypatch: "pytest.MonkeyPatch") -> "tuple[NginxGeneration, Path, list[str]]":
    container = fresh_manager.containers["nginx"]
    root = tmp_path / "nginx state"
    binary = tmp_path / "bin"
    binary.mkdir()
    client = binary / "acme.sh"
    client.write_text(_ACME_CLIENT, encoding="utf-8")
    client.chmod(0o755)
    monkeypatch.setattr(container, "get_app_path", lambda *parts: root.joinpath(*parts))
    fresh_manager.env_config.set("NGINX_ROOT_DOMAIN", "example.test")
    fresh_manager.env_config.set("NGINX_HTTPS_ENABLE", True)
    fresh_manager.env_config.set("ACME_DNS_API", "dns_test")
    commands = []

    def validate(context: object, service: str, command: "tuple[str, ...]",
                 network: bool = False) -> None:
        assert service == "nginx"
        assert network
        shell = command[-1].replace("/root/.acme.sh", shlex.quote(str(root / "acme")))
        shell = shell.replace("/etc/certs", shlex.quote(str(root / "certs")))
        commands.append(shell)
        environment = dict(os.environ)
        environment["PATH"] = os.pathsep.join((str(binary), str(Path(sys.executable).parent), os.defpath))
        result = subprocess.run([*command[:-1], shell], env=environment,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        if result.returncode:
            raise ContainerError(result.stderr)

    monkeypatch.setattr(fresh_manager.compose_runner, "validate_service", validate)
    return NginxGeneration(container), root, commands


def test_empty_acme_mount_issues_with_explicit_default_ca(acme_bootstrap: "tuple") -> None:
    owner, root, commands = acme_bootstrap
    context = SimpleNamespace(initial_services=())
    owner.prepare(context)
    state = json.loads((root / "acme/account.json").read_text())
    assert state["server"] == "letsencrypt"
    assert state["email"] == ""
    assert state["domains"] == ["example.test", "*.example.test"]
    assert state["dns"] == "dns_test"
    assert (root / "certs/example.test_fullchain.pem").read_text() == "simulated-certificate"
    assert context.nginx_certificate_replaced
    assert len(commands) == 1


def test_runtime_ca_and_email_are_passed_without_shell_expansion(
        acme_bootstrap: "tuple", fresh_manager: "ContainerManager") -> None:
    owner, root, commands = acme_bootstrap
    server = "https://ca.example.test/directory?tenant=o'reilly&profile=one"
    email = "ops+o'reilly@example.test"
    fresh_manager.env_config.set("ACME_SERVER", server)
    fresh_manager.env_config.set("ACME_ACCOUNT_EMAIL", email)
    owner.prepare(SimpleNamespace(initial_services=()))
    state = json.loads((root / "acme/account.json").read_text())
    assert (state["server"], state["email"]) == (server, email)


def test_existing_acme_account_is_reused_without_rotating_key_or_email(acme_bootstrap: "tuple") -> None:
    owner, root, commands = acme_bootstrap
    home = root / "acme"
    home.mkdir(parents=True)
    (home / "account.json").write_text(json.dumps({"key": "existing-key", "email": "existing@example.test"}))
    owner.prepare(SimpleNamespace(initial_services=()))
    state = json.loads((home / "account.json").read_text())
    assert (state["key"], state["email"]) == ("existing-key", "existing@example.test")


def test_failed_first_issuance_does_not_install_certificates_or_mark_replacement(
        acme_bootstrap: "tuple", fresh_manager: "ContainerManager") -> None:
    owner, root, commands = acme_bootstrap
    fresh_manager.env_config.set("ACME_SERVER", "zerossl")
    context = SimpleNamespace(initial_services=())
    with pytest.raises(ContainerError, match="account email is required"):
        owner.prepare(context)
    assert not list((root / "certs").iterdir())
    assert not hasattr(context, "nginx_certificate_replaced")


def test_http_preparation_does_not_resolve_acme_account_settings(
        acme_bootstrap: "tuple", fresh_manager: "ContainerManager", monkeypatch: "pytest.MonkeyPatch") -> None:
    owner, root, commands = acme_bootstrap
    fresh_manager.env_config.set("NGINX_HTTPS_ENABLE", False)
    original = owner.container.get_config

    def config(key: str, **kwargs: object) -> object:
        assert not key.startswith("ACME_")
        return original(key, **kwargs)

    monkeypatch.setattr(owner.container, "get_config", config)
    owner.prepare(SimpleNamespace(initial_services=()))
    assert commands == []
