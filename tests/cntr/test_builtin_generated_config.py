#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Builtin consumer contracts without Docker or persistent project state."""
import importlib.util
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from linktools.cntr import ContainerError, ExposeCategory


ASSETS = Path(__file__).resolve().parents[2] / "linktools-cntr/src/linktools/assets/containers"


@lru_cache(maxsize=None)
def builtin(name):
    spec = importlib.util.spec_from_file_location("test_generated_" + name.replace("-", "_"), str(ASSETS / name / "container.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def instance(module, **manager):
    container = object.__new__(module.Container)
    container.manager = SimpleNamespace(**manager)
    return container


def test_flare_custom_categories_merge_by_name_without_title_deduplication():
    module = builtin("120-flare")
    first = ExposeCategory("team", "Team")("Same", "a", "One", "https://one.test")
    second = ExposeCategory("team", "Team")("Same", "b", "Two", "https://two.test")
    source = SimpleNamespace(order=1, exposes=[first, second])
    container = instance(module, installed_state=SimpleNamespace(get=lambda: [source]))
    result = container.render_generated_config("id")
    bookmarks = yaml.safe_load(result["bookmarks.yml"])
    assert bookmarks["categories"] == [{"id": "team", "title": "Team"}]
    assert [entry["link"] for entry in bookmarks["links"]] == ["https://one.test", "https://two.test"]


def test_flare_rejects_conflicting_category_descriptions():
    module = builtin("120-flare")
    first = ExposeCategory("team", "Team")("One", "a", "", "https://one.test")
    second = ExposeCategory("team", "Different")("Two", "b", "", "https://two.test")
    source = SimpleNamespace(order=1, exposes=[first, second])
    container = instance(module, installed_state=SimpleNamespace(get=lambda: [source]))
    with pytest.raises(ContainerError, match="Conflicting description"):
        container.render_generated_config("id")


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
        container.apply_generated_config(SimpleNamespace(changed=True), SimpleNamespace())
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
    module.Container._create_secret_file(secret)
    module.Container._create_pem_file(jwks)
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
    result = container.render_generated_config("new")
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
    container.apply_generated_config(candidate, SimpleNamespace())
    assert ("authelia", True) in actions
    assert ("authelia-admin", False) in actions


def test_authelia_oidc_identity_and_redirects_are_acyclic_readonly(monkeypatch):
    module = builtin("102-authelia")
    site = SimpleNamespace(enabled=True, oidc_redirects=("https://app.test", "https://app.test"))
    container = instance(module, project_name="project", containers={"nginx": SimpleNamespace(sites={"app": site})})
    monkeypatch.setattr(container, "load_nginx_url", lambda local_id: "https://sso.test")
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
    declaration = container.integrations["nginx"]["web"]
    assert calls == ["PORTAINER_AUTH_ENABLE"]
    assert tuple(declaration.oidc_redirects) == ()
    assert calls == ["PORTAINER_AUTH_ENABLE", "PORTAINER_AUTH_ENABLE", "NGINX_AUTH_ENABLE"]


def test_flare_candidate_permissions_preserve_host_owner_and_service_read(tmp_path):
    import os
    module = builtin("120-flare")
    for name in ("apps.yml", "bookmarks.yml"):
        (tmp_path / name).write_text("links: []\n")
        (tmp_path / name).chmod(0o600)
    container = SimpleNamespace(get_config=lambda key, **kwargs: os.getgid())
    module.Container.validate_generated_config(container, SimpleNamespace(path=str(tmp_path)), None)
    for name in ("apps.yml", "bookmarks.yml"):
        assert (tmp_path / name).stat().st_mode & 0o777 == 0o640
        assert (tmp_path / name).stat().st_uid == os.getuid()


def test_lldap_preparation_preserves_active_config_and_persistent_data(tmp_path):
    module = builtin("101-lldap")
    secrets = tmp_path / "secrets"
    data = tmp_path / "data"
    secrets.mkdir()
    data.mkdir()
    (secrets / "jwt_secret").write_text("existing-jwt")
    (secrets / "ldap_user_pass").write_text("old-password")
    (data / "lldap_config.toml").write_text("old-config")
    (data / "users.db").write_bytes(b"existing-database")
    container = SimpleNamespace(
        get_app_path=lambda *parts: tmp_path.joinpath(*parts),
        runtime=SimpleNamespace(chmod=lambda *args, **kwargs: None),
        _create_secret_file=module.Container._create_secret_file,
    )
    module.Container.prepare_generated_config(container, None)
    assert (secrets / "jwt_secret").read_text() == "existing-jwt"
    assert (secrets / "ldap_user_pass").read_text() == "old-password"
    assert (data / "lldap_config.toml").read_text() == "old-config"
    assert (data / "users.db").read_bytes() == b"existing-database"


def test_lldap_derived_password_is_only_in_candidate():
    module = builtin("101-lldap")
    container = SimpleNamespace(
        get_source_path=lambda *parts: parts,
        render_template=lambda source: 'database_url = "sqlite:///data/users.db?mode=rwc"',
        get_config=lambda key: "new-password",
    )
    assert module.Container.render_generated_config(container, "generation")["ldap_user_pass"] == "new-password"


def test_stopped_legacy_nginx_preserves_certificates_before_migration(tmp_path):
    module = builtin("100-nginx")
    copied = []
    def preserve():
        copied.append(True)
        (tmp_path / "certs" / "example.test_fullchain.pem").write_text("certificate")
        (tmp_path / "certs" / "example.test_key.pem").write_text("key")
    values = {"NGINX_HTTPS_ENABLE": True, "NGINX_ROOT_DOMAIN": "example.test"}
    container = SimpleNamespace(
        get_app_path=lambda *parts: tmp_path.joinpath(*parts),
        get_config=lambda key, **kwargs: values[key],
        _preserve_legacy_files=preserve,
        _acme_ssl_domains=("example.test",),
        manager=SimpleNamespace(compose_runner=SimpleNamespace(validate_service=lambda *args, **kwargs: None)),
    )
    module.Container.prepare_generated_config(
        container, SimpleNamespace(initial_services={"nginx"}, initial_running=set()))
    assert copied == [True]
