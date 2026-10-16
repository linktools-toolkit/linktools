#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Configuration-only consumers preserve persistent startup inputs."""
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from _harness import builtin_container_type
from linktools.cntr import ContainerError, OperationContext
from linktools.cntr.runtime.inspect import ProjectRuntimeState, ServiceRuntimeState


REVISION = "1234567890abcdef"
OLD_REVISION = "fedcba0987654321"


def _unexpected(*args, **kwargs):
    raise AssertionError("Configuration-only preparation must not initialize persistent data")


def _container(name, monkeypatch):
    container = object.__new__(builtin_container_type(name))
    container.manager = SimpleNamespace(
        runtime=SimpleNamespace(chmod=_unexpected, chown=_unexpected), user="test")
    monkeypatch.setattr(container, "get_app_path", _unexpected)
    monkeypatch.setattr(container, "get_source_path", lambda *parts: Path(*parts))
    return container


def _context(container, explicit=False, services=("app",), state=None):
    context = OperationContext(
        target_containers=[container] if explicit else [], target_services=services,
        initial_runtime_state=state)
    written = []
    context.write_files = lambda owner, files, **kwargs: written.append((owner, files, kwargs))
    return context, written


def _state(*revisions, oneoff_revisions=()):
    instances = []
    for oneoff, values in ((False, revisions), (True, oneoff_revisions)):
        for revision in values:
            labels = {} if revision is None else {"io.linktools.nginx.certificate-revision": revision}
            if oneoff:
                labels["com.docker.compose.oneoff"] = "True"
            instances.append(ServiceRuntimeState(
                logical_containers=("nginx",), service="nginx", runtime_name="test-nginx",
                state="running", health="healthy", image="nginx:" + REVISION,
                exit_code=0, labels=labels))
    return ProjectRuntimeState("test", tuple(instances), "docker")


def test_nginx_consumer_renders_without_dns_credentials_or_account_preparation(monkeypatch):
    container = _container("100-nginx", monkeypatch)
    container.__dict__.update(cert_image_revision=REVISION,
                              _rendered_site_files=({"sites/generated.conf": "site configuration"}, True))
    monkeypatch.setattr(container, "get_config", _unexpected)
    rendered = []
    monkeypatch.setattr(container, "_render_site_template",
                        lambda owner, path, site: rendered.append(site.template_vars) or "nginx configuration")
    from dataclasses import replace
    state = _state(REVISION, REVISION, oneoff_revisions=(OLD_REVISION,))
    stopped = replace(_state(OLD_REVISION).services[0], state="exited")
    state = replace(state, services=state.services + (stopped,))
    context, written = _context(container, state=state)

    container.on_integrating(context)

    assert rendered == [{"waf": True, "site_files": ("sites/generated.conf",)}]
    assert written == [(container, {"sites/generated.conf": "site configuration",
                                    "nginx.conf": "nginx configuration"}, {})]
    assert context.target_containers == []
    assert context.target_services == ("app",)


@pytest.mark.parametrize("state", [
    None, _state(), _state(None), _state(OLD_REVISION), _state(REVISION, OLD_REVISION),
    _state(oneoff_revisions=(REVISION,)),
])
def test_nginx_consumer_requires_explicit_selection_for_unverified_certificate_revision(monkeypatch, state):
    container = _container("100-nginx", monkeypatch)
    container.__dict__["cert_image_revision"] = REVISION
    monkeypatch.setattr(container, "get_config", _unexpected)
    monkeypatch.setattr(container, "_render_site_template", _unexpected)
    context, written = _context(container, state=state)

    with pytest.raises(ContainerError, match="explicitly select nginx"):
        container.on_integrating(context)

    assert written == []


def test_nginx_explicit_selection_keeps_dns_and_account_preparation(tmp_path, monkeypatch):
    container = _container("100-nginx", monkeypatch)
    container.__dict__.update(cert_image_revision=REVISION, extra_fields={"DNS_TOKEN": "DNS_TOKEN"},
                              _rendered_site_files=({}, False))
    monkeypatch.setattr(container, "get_app_path", lambda *parts: tmp_path.joinpath(*parts))
    monkeypatch.setattr(container, "get_config", lambda key, **kwargs: {
        "NGINX_HTTPS_ENABLE": True, "DNS_TOKEN": "test-token",
    }[key])
    monkeypatch.setattr(container, "_render_site_template", lambda *args: "nginx configuration")
    permissions = []
    container.runtime.chmod = lambda *args: permissions.append(args)
    account = tmp_path / "certs" / OLD_REVISION / "live" / "acme"
    account.mkdir(parents=True)
    (account / "account.conf").write_text("existing account", encoding="utf-8")
    context, written = _context(container, explicit=True, services=("nginx",), state=_state(OLD_REVISION))

    container.on_starting(context)
    assert written == []
    monkeypatch.setattr(container, "get_app_path", _unexpected)
    container.on_integrating(context)

    assert (tmp_path / "acme-secrets" / "dns.env").read_text(encoding="utf-8") == "export DNS_TOKEN=test-token\n"
    with tarfile.open(str(tmp_path / "acme-build-account.tar")) as archive:
        assert archive.extractfile("account.conf").read() == b"existing account"
    assert permissions == [(tmp_path / "acme-secrets", 0o700), (tmp_path / "certs" / ".acme.lock", 0o600)]
    assert written == [(container, {"nginx.conf": "nginx configuration"}, {})]


@pytest.mark.parametrize("services", [None, (), ("app",), ("authelia-redis",)])
def test_authelia_consumer_renders_without_creating_secrets(monkeypatch, services):
    container = _container("102-authelia", monkeypatch)
    monkeypatch.setattr(container, "_create_secret_file", _unexpected)
    monkeypatch.setattr(container, "_create_pem_file", _unexpected)
    monkeypatch.setattr(container, "get_config", lambda key: "current LDAP password")
    monkeypatch.setattr(container, "render_template", lambda path: "rendered " + path.name)
    context, written = _context(container, services=services)

    container.on_integrating(context)

    expected = {name: "rendered " + name for name in container.config_files}
    expected["authentication_backend_ldap_password"] = "current LDAP password"
    assert written == [(container, expected, {})]
    assert context.target_services == services
    assert context.target_containers == []


@pytest.mark.parametrize("services", [("authelia",), ("authelia-admin",), None])
def test_authelia_explicit_selection_keeps_persistent_initialization(tmp_path, monkeypatch, services):
    container = _container("102-authelia", monkeypatch)
    monkeypatch.setattr(container, "get_app_path", lambda *parts: tmp_path.joinpath(*parts))
    monkeypatch.setattr(container, "get_config", lambda key: "LDAP password")
    monkeypatch.setattr(container, "render_template", lambda path: "rendered " + path.name)
    created = []
    monkeypatch.setattr(container, "_create_secret_file", lambda path: created.append(path.name))
    monkeypatch.setattr(container, "_create_pem_file", lambda path: created.append(path.name))
    permissions = []
    container.runtime.chmod = lambda *args, **kwargs: permissions.append((args, kwargs))
    context, written = _context(container, explicit=True, services=services)

    container.on_starting(context)
    assert written == []
    monkeypatch.setattr(container, "get_app_path", _unexpected)
    container.on_integrating(context)

    assert (tmp_path / "secrets").is_dir()
    assert (tmp_path / "config").is_dir()
    assert created == ["jwt_secret", "session_secret", "storage_encryption_key", "oidc_hmac_secret",
                       "identity_providers_oidc_jwks"]
    assert permissions == [((tmp_path / "secrets", 0o700), {"recursive": True})]
    assert len(written) == 1


def test_authelia_explicit_redis_selection_still_skips_generated_configuration(monkeypatch):
    container = _container("102-authelia", monkeypatch)
    monkeypatch.setattr(container, "render_template", _unexpected)
    context, written = _context(container, explicit=True, services=("authelia-redis",))

    container.on_starting(context)
    container.on_integrating(context)

    assert written == []


def test_flare_consumer_renders_without_runtime_app_migration(monkeypatch):
    container = _container("120-flare", monkeypatch)
    navigation = {"apps.yml": "links: []", "bookmarks.yml": "categories: []\nlinks: []"}
    monkeypatch.setattr(container, "_navigation_files", lambda: navigation)
    monkeypatch.setattr(container, "get_config", lambda key, **kwargs: 123)
    context, written = _context(container)

    container.on_integrating(context)

    assert written == [(container, navigation, {"mode": 0o640, "group": 123})]
    assert context.target_containers == []
    assert context.target_services == ("app",)


def test_flare_explicit_selection_keeps_runtime_app_migration(tmp_path, monkeypatch):
    container = _container("120-flare", monkeypatch)
    monkeypatch.setattr(container, "get_app_path", lambda *parts: tmp_path.joinpath(*parts))
    monkeypatch.setattr(container, "_navigation_files", lambda: {"apps.yml": "current applications"})
    monkeypatch.setattr(container, "get_config", lambda key, **kwargs: 123)
    permissions = []
    container.runtime.chown = lambda path, user, **kwargs: permissions.append(("chown", user, kwargs))
    container.runtime.chmod = lambda path, mode: permissions.append(("chmod", mode))
    legacy = tmp_path / "app"
    (legacy / "icons").mkdir(parents=True)
    (legacy / "icons" / "app.svg").write_text("icon", encoding="utf-8")
    (legacy / "settings.yml").write_text("settings", encoding="utf-8")
    (legacy / "link").symlink_to("settings.yml")
    for name in ("apps.yml", "bookmarks.yml"):
        (legacy / name).write_text("old generated data", encoding="utf-8")
    context, written = _context(container, explicit=True, services=("flare",))

    container.on_starting(context)
    assert written == []
    monkeypatch.setattr(container, "get_app_path", _unexpected)
    container.on_integrating(context)

    app = tmp_path / "runtime-app"
    assert (app / "icons" / "app.svg").read_text(encoding="utf-8") == "icon"
    assert (app / "settings.yml").read_text(encoding="utf-8") == "settings"
    assert (app / "link").is_symlink()
    assert not (app / "apps.yml").exists()
    assert not (app / "bookmarks.yml").exists()
    assert (legacy / "apps.yml").read_text(encoding="utf-8") == "old generated data"
    assert permissions == [("chown", "test", {"recursive": True}), ("chmod", 0o750)]
    assert written == [(container, {"apps.yml": "current applications"}, {"mode": 0o640, "group": 123})]


def test_authelia_admin_only_validation_requires_explicit_container(monkeypatch):
    container = _container("102-authelia", monkeypatch)
    monkeypatch.setattr(container, "get_config", _unexpected)
    container.manager.compose_runner = SimpleNamespace(validate_service=_unexpected)
    context, _ = _context(container, services=("app", "authelia-admin"))
    with pytest.raises(ContainerError, match="explicitly select the authelia"):
        container.on_check(context)


@pytest.mark.parametrize('access', ['owner', 'group', 'supplementary', 'unreadable'])
def test_flare_checks_generated_files_against_effective_service_user(tmp_path, monkeypatch, access):
    import os
    container = _container('120-flare', monkeypatch)
    for name in ('apps.yml', 'bookmarks.yml'):
        path = tmp_path / name
        path.write_text('links: []\n', encoding='utf-8')
        path.chmod(0o640)
    owner, group = os.getuid(), os.getgid()
    user = owner if access == 'owner' else owner + 12345
    gid = group if access == 'group' else group + 12345
    context = OperationContext(compose_model={'services': {'flare': {
        'user': '{}:{}'.format(user, gid), 'group_add': [str(group)] if access == 'supplementary' else []}}})
    context.file_path = lambda owner, name: tmp_path / name
    if access == 'unreadable':
        with pytest.raises(ContainerError, match='explicitly select the flare'):
            container.on_check(context)
    else:
        container.on_check(context)
