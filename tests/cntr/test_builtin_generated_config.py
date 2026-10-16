#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Builtin container contracts without Docker or persistent project state."""
from types import SimpleNamespace

import pytest
import yaml

from linktools.cntr import ContainerError, ContainerManager
from linktools.cntr.ext import Flare
from linktools.core import Config, ConfigSchema
from _harness import builtin_container_type
from _harness import builtin_module as builtin


NginxContainer = builtin_container_type("100-nginx")
LldapContainer = builtin_container_type("101-lldap")
AutheliaContainer = builtin_container_type("102-authelia")
FlareContainer = builtin_container_type("120-flare")


def instance(module, **manager):
    container = object.__new__(module.Container)
    manager.setdefault("env_config", Config(None, ConfigSchema(), []))
    container.manager = SimpleNamespace(**manager)
    return container


def flare_instance(module, entries):
    snapshot = {"flare": []}
    containers = {}
    for producer, key, link in entries:
        producer.env_config = Config(None, ConfigSchema(), [])
        containers[producer.name] = producer
        snapshot.setdefault(producer.name, []).append(link)
    manager = object.__new__(ContainerManager)
    manager.containers = dict(containers, flare=SimpleNamespace(name="flare", order=0))
    manager.integration_snapshot = {name: tuple(values) for name, values in snapshot.items()}
    container = object.__new__(module.Container)
    container.manager = manager
    return container


def test_flare_custom_categories_merge_by_name_without_title_deduplication():
    module = builtin("120-flare")
    first = Flare.category("team", "Team")("Same", "a", "One", "https://one.test")
    second = Flare.category("team", "Team")("Same", "b", "Two", "https://two.test")
    source = SimpleNamespace(name="source", order=1)
    container = flare_instance(module, [(source, "one", first), (source, "two", second)])
    result = container._navigation_files()
    bookmarks = yaml.safe_load(result["bookmarks.yml"])
    assert bookmarks["categories"] == [{"id": "team", "title": "Team"}]
    assert [entry["link"] for entry in bookmarks["links"]] == ["https://one.test", "https://two.test"]


def test_flare_rejects_conflicting_category_descriptions():
    module = builtin("120-flare")
    first = Flare.category("team", "Team")("One", "a", "", "https://one.test")
    second = Flare.category("team", "Different")("Two", "b", "", "https://two.test")
    source = SimpleNamespace(name="source", order=1)
    container = flare_instance(module, [(source, "one", first), (source, "two", second)])
    with pytest.raises(ContainerError, match="Conflicting description"):
        container._navigation_files()


def test_authelia_never_rotates_existing_secret_or_jwks(tmp_path, monkeypatch):
    module = builtin("102-authelia")
    secret = tmp_path / "secret"
    jwks = tmp_path / "jwks"
    secret.write_text("existing secret")
    jwks.write_text("existing jwks")
    def fail(*args, **kwargs):
        raise AssertionError("existing credentials must not be regenerated")
    monkeypatch.setattr(module.rsa, "newkeys", fail)
    monkeypatch.setattr(module.utils, "random_string", fail)
    AutheliaContainer._create_secret_file(secret)
    AutheliaContainer._create_pem_file(jwks)
    assert secret.read_text() == "existing secret"
    assert jwks.read_text() == "existing jwks"


def test_authelia_oidc_identity_and_redirects_are_acyclic_readonly(monkeypatch):
    module = builtin("102-authelia")
    from linktools.cntr.ext import Authelia
    declaration = Authelia.oidc(("https://app.test", "https://app.test"))
    container = instance(module, project_name="project")
    producer = SimpleNamespace(name="app", env_config=Config(None, ConfigSchema(), []))
    container.manager.iter_integrations = lambda consumer: iter(((producer, declaration),))
    monkeypatch.setitem(container.__dict__, "public_url", "https://sso.test")
    monkeypatch.setattr(container, "get_config", lambda key: "saved-secret")
    client = container.oidc_client
    assert client["client_secret"] == "saved-secret"
    assert client["redirect_uris"] == ("https://sso.test", "https://app.test")
    assert isinstance(client["scopes"], tuple)
    with pytest.raises(TypeError):
        client["client_id"] = "bad"


@pytest.mark.parametrize("portainer_auth", [True, False])
@pytest.mark.parametrize("global_auth", [True, False])
def test_portainer_callback_declaration_is_lazy_and_auth_independent(monkeypatch, portainer_auth, global_auth):
    from linktools.cntr.ext import Authelia, ResolvedSite

    module = builtin("110-portainer")
    container = instance(module)
    calls = []
    values = {"PORTAINER_AUTH_ENABLE": portainer_auth, "PORTAINER_DOMAIN": "portainer.test",
              "NGINX_AUTH_ENABLE": global_auth,
              "NGINX_HTTPS_ENABLE": True, "NGINX_HTTPS_PORT": 443}

    def config(key, **kwargs):
        calls.append(key)
        return values[key]

    monkeypatch.setattr(container, "get_config", config)
    container.env_config.get = config
    declarations = container.integrations
    declaration = next(value for value in declarations if isinstance(value, Authelia))
    assert calls == []
    container.manager.integration_snapshot = {"nginx": ()}
    container.manager.containers = {"nginx": SimpleNamespace(sites={
        declarations[0]: ResolvedSite(container, declarations[0], 0),
    })}
    assert declaration.resolve_redirect_uris(container.env_config) == ("https://portainer.test",)
    assert calls == ["PORTAINER_DOMAIN", "NGINX_HTTPS_ENABLE", "NGINX_HTTPS_PORT"]


def test_flare_navigation_orders_producers_and_preserves_link_values():
    module = builtin("120-flare")
    early = SimpleNamespace(name="early", order=10)
    late = SimpleNamespace(name="late", order=20)
    public = Flare.category("public", "Public", apps=True)
    tools = Flare.category("tools", "Tools")
    entries = [
        (late, "external", tools("External", "web", "", "https://external.test")),
        (early, "public", public("App", "apps", "Description", "https://app.test")),
        (early, "disabled", tools("Disabled", "off", "", "")),
        (early, "direct", tools("Direct", "lan", "", "http://host:1234")),
        (early, "custom", tools("Custom", "link", "", "custom://literal/{{port}}")),
    ]
    container = flare_instance(module, entries)
    result = container._navigation_files()
    assert yaml.safe_load(result["apps.yml"])["links"] == [
        {"name": "App", "icon": "apps", "desc": "Description", "link": "https://app.test"}]
    links = yaml.safe_load(result["bookmarks.yml"])["links"]
    assert [link["name"] for link in links] == ["Direct", "Custom", "External"]
    assert links[1]["link"] == "custom://literal/{{port}}"


def test_authelia_prepares_new_password_without_overwriting_legacy(tmp_path, monkeypatch):
    container = instance(builtin("102-authelia"),
                         runtime=SimpleNamespace(chmod=lambda *a, **k: None))
    legacy = tmp_path / "authentication_backend_ldap_password"
    legacy.write_text("old password")
    monkeypatch.setattr(container, "get_app_path", lambda *parts: tmp_path.joinpath(*parts))
    monkeypatch.setattr(container, "get_source_path", lambda *parts: tmp_path.joinpath(*parts))
    monkeypatch.setattr(container, "render_template", lambda path: "rendered " + path.name)
    monkeypatch.setattr(container, "get_config", lambda key: "new password")
    monkeypatch.setattr(container, "_create_secret_file", lambda path: None)
    monkeypatch.setattr(container, "_create_pem_file", lambda path: None)
    values = {}
    context = SimpleNamespace(target_services=("authelia",),
                              write_files=lambda owner, files: values.update(files))
    container.on_starting(context)
    assert values["authentication_backend_ldap_password"] == "new password"
    assert legacy.read_text() == "old password"
    commands = []
    container.manager.compose_runner = SimpleNamespace(
        validate_service=lambda context, service, command, **kwargs:
        commands.append(command) or SimpleNamespace(succeeded=True))
    container.on_check(context)
    assert commands == [["authelia", "config", "validate"] + [
        "--config=/generated/" + name for name in (
            "configuration.yml", "configuration.acl.yml", "configuration.2fa.yml", "configuration.oidc.yml")]]
    assert set(values) == {argument.removeprefix("--config=/generated/") for argument in commands[0][3:]} | {
        "authentication_backend_ldap_password"}


def test_lldap_prepares_without_modifying_persistent_inputs(tmp_path, monkeypatch):
    secrets = tmp_path / "secrets"
    data = tmp_path / "data"
    secrets.mkdir()
    data.mkdir()
    (secrets / "jwt_secret").write_text("existing-jwt")
    (data / "users.db").write_bytes(b"existing-database")
    container = instance(builtin("101-lldap"),
                         runtime=SimpleNamespace(chmod=lambda *a, **k: None))
    monkeypatch.setattr(container, "get_app_path", lambda *parts: tmp_path.joinpath(*parts))
    monkeypatch.setattr(container, "get_source_path", lambda *parts: tmp_path.joinpath(*parts))
    monkeypatch.setattr(container, "render_template", lambda path: "rendered " + path.name)
    monkeypatch.setattr(container, "get_config", lambda key: "new-password")
    files = {}
    container.on_starting(SimpleNamespace(write_files=lambda owner, output: files.update(output)))
    assert (secrets / "jwt_secret").read_text() == "existing-jwt"
    assert (data / "users.db").read_bytes() == b"existing-database"
    assert files["ldap_user_pass"] == "new-password"
