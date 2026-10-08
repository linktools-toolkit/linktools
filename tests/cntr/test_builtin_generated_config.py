#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Builtin container contracts without Docker or persistent project state."""
from types import SimpleNamespace

import pytest
import yaml

from linktools.cntr import ContainerError, Flare
from _harness import builtin_container_type
from _harness import builtin_module as builtin


NginxContainer = builtin_container_type("100-nginx")
LldapContainer = builtin_container_type("101-lldap")
AutheliaContainer = builtin_container_type("102-authelia")
FlareContainer = builtin_container_type("120-flare")


def instance(module, **manager):
    container = object.__new__(module.Container)
    container.manager = SimpleNamespace(**manager)
    return container


def flare_instance(module, entries):
    snapshot = {"flare": []}
    containers = {}
    for producer, local_id, link in entries:
        containers[producer.name] = producer
        snapshot.setdefault(producer.name, []).append(link)
    return instance(module, integration_snapshot={name: tuple(values) for name, values in snapshot.items()},
                    containers=containers, nginx_sites={})


def test_flare_custom_categories_merge_by_name_without_title_deduplication():
    module = builtin("120-flare")
    first = Flare.category("team", "Team")("Same", "a", "One", "https://one.test")
    second = Flare.category("team", "Team")("Same", "b", "Two", "https://two.test")
    source = SimpleNamespace(name="source", order=1)
    container = flare_instance(module, [(source, "one", first), (source, "two", second)])
    result = container.render_config("id")
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
        container.render_config("id")


@pytest.mark.parametrize("second_backup_exists", [False, True])
def test_flare_first_migration_failure_restores_original_files(tmp_path, monkeypatch, second_backup_exists):
    module = builtin("120-flare")
    def fail(*args, **kwargs):
        raise ContainerError("apply failed")
    runner = SimpleNamespace(apply_service=fail, wait_service_running=lambda *args: None)
    container = instance(module, compose_runner=runner)
    app = tmp_path / "app"
    app.mkdir()
    for name in ("apps.yml", "bookmarks.yml"):
        (app / name).write_text("original " + name)
    if second_backup_exists:
        (app / "bookmarks.yml.pre-cntr").write_text("older backup")
    monkeypatch.setattr(container, "get_app_path", lambda *parts: tmp_path.joinpath(*parts))
    with pytest.raises(ContainerError):
        container.apply_config(SimpleNamespace(), SimpleNamespace(changed=True), ("flare",))
    for name in ("apps.yml", "bookmarks.yml"):
        assert not (app / name).is_symlink()
        assert (app / name).read_text() == "original " + name


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


def test_authelia_ldap_password_is_candidate_data_and_does_not_overwrite_legacy(tmp_path, monkeypatch):
    module = builtin("102-authelia")
    container = instance(module)
    legacy = tmp_path / "authentication_backend_ldap_password"
    legacy.write_text("old password")
    monkeypatch.setattr(container, "get_config", lambda key: "new password")
    monkeypatch.setattr(container, "get_source_path", lambda *parts: tmp_path.joinpath(*parts))
    monkeypatch.setattr(container, "render_template", lambda path: "rendered " + path.name)
    result = container.render_config("new")
    assert result["authentication_backend_ldap_password"] == "new password"
    assert legacy.read_text() == "old password"


def test_authelia_admin_checks_own_compose_changes_without_acl_forced_restart():
    module = builtin("102-authelia")
    actions = []
    runner = SimpleNamespace(
        apply_service=lambda ctx, name, recreate=False: actions.append((name, recreate)),
        wait_service_healthy=lambda *args: None,
    )
    container = instance(module, compose_runner=runner)
    candidate = SimpleNamespace(changed=True, changed_files=("configuration.acl.yml",))
    container.apply_config(SimpleNamespace(), candidate, ("authelia", "authelia-admin"))
    assert ("authelia", True) in actions
    assert ("authelia-admin", False) in actions


def test_authelia_oidc_identity_and_redirects_are_acyclic_readonly(monkeypatch):
    module = builtin("102-authelia")
    site = SimpleNamespace(enabled=True, oidc_redirects=("https://app.test", "https://app.test"))
    container = instance(module, project_name="project", containers={"nginx": SimpleNamespace(sites={"app": site})})
    monkeypatch.setitem(container.__dict__, "public_url", "https://sso.test")
    monkeypatch.setattr(container, "get_config", lambda key: "saved-secret")
    client = container.oidc_client
    assert client["client_secret"] == "saved-secret"
    assert client["redirect_uris"] == ("https://sso.test", "https://app.test")
    assert isinstance(client["scopes"], tuple)
    with pytest.raises(TypeError):
        client["client_id"] = "bad"


def test_portainer_callback_declaration_is_lazy_and_auth_conditioned(monkeypatch):
    module = builtin("110-portainer")
    container = instance(module)
    calls = []
    def config(key):
        calls.append(key)
        return key == "PORTAINER_AUTH_ENABLE"
    monkeypatch.setattr(container, "get_config", config)
    monkeypatch.setattr(container, "get_config_later", lambda key: "portainer.test")
    declaration = next(value for value in container.integrations if value.consumer == "nginx")
    assert calls == ["PORTAINER_AUTH_ENABLE"]
    assert tuple(declaration.oidc_redirects) == ()
    assert calls == ["PORTAINER_AUTH_ENABLE", "PORTAINER_AUTH_ENABLE", "NGINX_AUTH_ENABLE"]


def test_flare_candidate_permissions_preserve_host_owner_and_service_read(tmp_path):
    import os
    for name in ("apps.yml", "bookmarks.yml"):
        (tmp_path / name).write_text("links: []\n")
        (tmp_path / name).chmod(0o600)
    container = instance(builtin("120-flare"))
    container.get_config = lambda key, **kwargs: os.getgid()
    container.validate_config(None, SimpleNamespace(path=str(tmp_path)))
    for name in ("apps.yml", "bookmarks.yml"):
        assert (tmp_path / name).stat().st_mode & 0o777 == 0o640
        assert (tmp_path / name).stat().st_uid == os.getuid()


def test_lldap_preparation_preserves_active_config_and_persistent_data(tmp_path):
    secrets = tmp_path / "secrets"
    data = tmp_path / "data"
    secrets.mkdir()
    data.mkdir()
    (secrets / "jwt_secret").write_text("existing-jwt")
    (secrets / "ldap_user_pass").write_text("old-password")
    (data / "lldap_config.toml").write_text("old-config")
    (data / "users.db").write_bytes(b"existing-database")
    container = instance(builtin("101-lldap"), runtime=SimpleNamespace(chmod=lambda *args, **kwargs: None))
    container.get_app_path = lambda *parts: tmp_path.joinpath(*parts)
    container.on_prepare_config(None)
    assert (secrets / "jwt_secret").read_text() == "existing-jwt"
    assert (secrets / "ldap_user_pass").read_text() == "old-password"
    assert (data / "lldap_config.toml").read_text() == "old-config"
    assert (data / "users.db").read_bytes() == b"existing-database"


def test_lldap_derived_password_is_only_in_candidate():
    container = instance(builtin("101-lldap"))
    container.get_source_path = lambda *parts: parts
    container.render_template = lambda source: 'database_url = "sqlite:///data/users.db?mode=rwc"'
    container.get_config = lambda key: "new-password"
    assert container.render_config("generation")["ldap_user_pass"] == "new-password"


def test_stopped_legacy_nginx_preserves_certificates_before_migration(tmp_path, monkeypatch):
    copied = []
    def preserve():
        copied.append(True)
        (tmp_path / "certs" / "example.test_fullchain.pem").write_text("certificate")
        (tmp_path / "certs" / "example.test_key.pem").write_text("key")
    values = {"NGINX_HTTPS_ENABLE": True, "NGINX_ROOT_DOMAIN": "example.test",
              "NGINX_HTTPS_PORT": 443, "NGINX_TAG": "stable-alpine",
              "ACME_SERVER": "letsencrypt", "ACME_DNS_API": "dns_cf",
              "ACME_ACCOUNT_EMAIL": ""}
    container = instance(builtin("100-nginx"),
                         compose_runner=SimpleNamespace(
                             validate_service=lambda *args, **kwargs: SimpleNamespace(succeeded=True)))
    container.get_app_path = lambda *parts: tmp_path.joinpath(*parts)
    container.get_config = lambda key, **kwargs: values[key]
    container.__dict__["acme_ssl_domains"] = ["example.test", "*.example.test"]
    owner = container
    monkeypatch.setattr(owner, "_preserve_legacy_files", preserve)
    context = SimpleNamespace(initial_services={"nginx"}, initial_running=set())
    owner.on_starting(context)
    owner.on_prepare_config(context)
    assert copied == [True]


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
    result = container.render_config("id")
    assert yaml.safe_load(result["apps.yml"])["links"] == [
        {"name": "App", "icon": "apps", "desc": "Description", "link": "https://app.test"}]
    links = yaml.safe_load(result["bookmarks.yml"])["links"]
    assert [link["name"] for link in links] == ["Direct", "Custom", "External"]
    assert links[1]["link"] == "custom://literal/{{port}}"


@pytest.mark.parametrize("owner_type", [NginxContainer, LldapContainer, AutheliaContainer, FlareContainer])
def test_generation_owner_empty_apply_scope_does_not_touch_stopped_services(owner_type):
    def unexpected(*args, **kwargs):
        raise AssertionError("Stopped services must not be touched")
    runner = SimpleNamespace(
        apply_service=unexpected, exec_service=unexpected,
        wait_service_healthy=unexpected, wait_service_running=unexpected,
        is_generation_current=unexpected,
    )
    container = object.__new__(owner_type)
    container.manager = SimpleNamespace(compose_runner=runner)
    container.get_app_path = unexpected
    container.apply_config(SimpleNamespace(), SimpleNamespace(changed=True), ())


@pytest.mark.parametrize("services, expected", [
    (("authelia",), [("apply", "authelia", True), ("healthy", "authelia")]),
    (("authelia-admin",), [("apply", "authelia-admin", True)]),
    (("authelia-admin", "authelia", "authelia-redis"), [
        ("apply", "authelia-redis", False), ("apply", "authelia", True),
        ("healthy", "authelia"), ("apply", "authelia-admin", True),
    ]),
])
def test_authelia_apply_scope_preserves_stopped_sibling(services, expected):
    actions = []
    runner = SimpleNamespace(
        apply_service=lambda ctx, name, recreate=False: actions.append(("apply", name, recreate)),
        wait_service_healthy=lambda ctx, name: actions.append(("healthy", name)),
    )
    container = instance(builtin("102-authelia"), compose_runner=runner)
    candidate = SimpleNamespace(changed=True, changed_files=("configuration.yml",))
    container.apply_config(SimpleNamespace(), candidate, iter(services))
    assert actions == expected


def test_authelia_redis_only_scope_reconciles_without_starting_siblings(fresh_manager, monkeypatch):
    actions = []
    runner = fresh_manager.compose_runner
    context = SimpleNamespace()
    monkeypatch.setattr(runner, "apply_service", lambda ctx, name, recreate=False:
                        actions.append((ctx, name, recreate)))
    def unexpected(*args, **kwargs):
        raise AssertionError("Unselected Authelia services must not be checked or started")
    monkeypatch.setattr(runner, "wait_service_healthy", unexpected)
    monkeypatch.setattr(runner, "is_generation_current", unexpected)
    fresh_manager.generated_configs["authelia"].apply_config(
        context, SimpleNamespace(changed=False), ("authelia-redis",))
    assert actions == [(context, "authelia-redis", False)]


def test_native_owner_validators_use_candidate_paths_and_password_file(tmp_path):
    calls = []
    def validate_service(*args, **kwargs):
        calls.append((args, kwargs))
        return SimpleNamespace(succeeded=True, stdout="", stderr="", returncode=0)
    runner = SimpleNamespace(validate_service=validate_service)
    context = SimpleNamespace()
    candidate = SimpleNamespace(generation_id="candidate", path=str(tmp_path))
    instance(builtin("100-nginx"), compose_runner=runner).validate_config(context, candidate)
    instance(builtin("102-authelia"), compose_runner=runner).validate_config(context, candidate)
    assert calls[0] == ((context, "nginx", (
        "nginx", "-p", "/etc/nginx/", "-c", "/etc/nginx/generated/candidate/nginx.conf", "-t")), {"check": False})
    assert calls[1][0][0:2] == (context, "authelia")
    assert calls[1][0][2] == [
        "authelia", "config", "validate", "--config=/generated/candidate/configuration.yml",
        "--config=/generated/candidate/configuration.acl.yml",
        "--config=/generated/candidate/configuration.2fa.yml",
        "--config=/generated/candidate/configuration.oidc.yml",
    ]
    assert calls[1][1] == {"environment": {
        "AUTHELIA_AUTHENTICATION_BACKEND_LDAP_PASSWORD_FILE":
        "/generated/candidate/authentication_backend_ldap_password"}, "check": False}
