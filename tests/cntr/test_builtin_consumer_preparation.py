#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Configuration-only consumers preserve persistent startup inputs."""
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from _harness import builtin_container_type, builtin_module
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
    container._name = name.split("-", 1)[1]
    container._prepared = []
    monkeypatch.setattr(builtin_module(name), "prepare_files", lambda owner, files, **kwargs: (
        container._prepared.append((owner, files, kwargs)) or Path("/generated") / REVISION))
    monkeypatch.setattr(container, "get_app_path", _unexpected)
    monkeypatch.setattr(container, "get_source_path", lambda *parts: Path(*parts))
    return container


def _context(container, explicit=False, services=("app",), state=None):
    if state is None and not explicit and container.name == "authelia":
        state = SimpleNamespace(running_services=frozenset(("authelia",)))
    context = OperationContext(
        target_containers=[container] if explicit else [], target_services=services,
        initial_runtime_state=state, compose_model={"services": {"nginx": {"build": {"labels": {
            "io.linktools.cntr.build-revision": REVISION}}}}})
    return context, container._prepared


def _state(*revisions, oneoff_revisions=(), label="io.linktools.cntr.build-revision"):
    instances = []
    for oneoff, values in ((False, revisions), (True, oneoff_revisions)):
        for revision in values:
            labels = {} if revision is None else {label: revision}
            if oneoff:
                labels["com.docker.compose.oneoff"] = "True"
            instances.append(ServiceRuntimeState(
                logical_containers=("nginx",), service="nginx", runtime_name="test-nginx",
                state="running", health="healthy", image="nginx:" + REVISION,
                exit_code=0, labels=labels))
    return ProjectRuntimeState("test", tuple(instances), "docker")


def test_nginx_consumer_renders_without_dns_credentials_or_account_preparation(monkeypatch):
    container = _container("100-nginx", monkeypatch)
    container.__dict__.update(
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

    assert container.config_path == Path("/generated") / REVISION

    assert rendered == [{"waf": True, "site_files": ("sites/generated.conf",)}]
    assert written == [(container, {"sites/generated.conf": "site configuration",
                                    "nginx.conf": "nginx configuration"}, {})]
    assert context.target_containers == []
    assert context.target_services == ("app",)


def test_nginx_consumer_checks_compatible_certificates_without_reading_dns_secrets(monkeypatch):
    from dataclasses import replace
    container = _container("100-nginx", monkeypatch)
    container.__dict__.update( sites={}, config_path=Path("/generated") / REVISION)
    monkeypatch.setattr(container, "get_config", lambda key, **kwargs: False)
    calls = []
    container.manager.compose_runner = SimpleNamespace(validate_service=lambda *args, **kwargs:
        calls.append(args) or SimpleNamespace(succeeded=True, stdout="", stderr=""))
    state = _state(REVISION, REVISION, oneoff_revisions=(OLD_REVISION,))
    stopped = replace(_state(OLD_REVISION).services[0], state="exited")
    state = replace(state, services=state.services + (stopped,))
    context, _ = _context(container, services=("app", "nginx"), state=state)

    container.on_check(context)

    assert len(calls) == 1


def test_nginx_explicit_selection_keeps_dns_and_account_preparation(tmp_path, monkeypatch):
    container = _container("100-nginx", monkeypatch)
    container.__dict__.update(extra_fields={"DNS_TOKEN": "DNS_TOKEN"},
                              _rendered_site_files=({}, False))
    monkeypatch.setattr(container, "get_app_path", lambda *parts: tmp_path.joinpath(*parts))
    monkeypatch.setattr(container, "get_config", lambda key, **kwargs: {
        "NGINX_HTTPS_ENABLE": True, "DNS_TOKEN": "test-token",
    }[key])
    monkeypatch.setattr(container, "_render_site_template", lambda *args: "nginx configuration")
    permissions = []
    container.runtime.chmod = lambda *args: permissions.append(args)
    account = tmp_path / "certs" / "live" / "acme"
    account.mkdir(parents=True)
    (account / "account.conf").write_text("existing account", encoding="utf-8")
    context, written = _context(container, explicit=True, services=("nginx",), state=_state(OLD_REVISION))

    container.on_starting(context)
    assert written == []
    monkeypatch.setattr(container, "get_app_path", _unexpected)
    assert container.config_path == Path("/generated") / REVISION

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

    assert container.config_path == Path("/generated") / REVISION

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
    assert container.config_path == Path("/generated") / REVISION

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
    container.on_check(context)

    assert "config_path" not in container.__dict__
    assert written == []


def test_flare_consumer_renders_without_runtime_app_migration(monkeypatch):
    container = _container("120-flare", monkeypatch)
    navigation = {"apps.yml": "links: []", "bookmarks.yml": "categories: []\nlinks: []"}
    monkeypatch.setattr(container, "_navigation_files", lambda: navigation)
    monkeypatch.setattr(container, "get_config", lambda key, **kwargs: 123)
    context, written = _context(container)

    assert container.config_path == Path("/generated") / REVISION

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
    assert container.config_path == Path("/generated") / REVISION

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


def test_flare_yaml_validation_does_not_infer_container_permissions_from_host_ids(tmp_path, monkeypatch):
    container = _container('120-flare', monkeypatch)
    for name in ('apps.yml', 'bookmarks.yml'):
        path = tmp_path / name
        path.write_text('links: []\n', encoding='utf-8')
        path.chmod(0o640)
    context = OperationContext(compose_model={'services': {'flare': {'user': '12345:23456'}}})
    container.__dict__["config_path"] = tmp_path
    container.on_check(context)


@pytest.mark.parametrize("name", ["100-nginx", "101-lldap", "102-authelia", "120-flare"])
def test_builtin_config_path_is_fixed_for_the_integration_snapshot(monkeypatch, name):
    container = _container(name, monkeypatch)
    monkeypatch.setattr(container, "get_config", lambda *args, **kwargs: "configured value")
    if name == "100-nginx":
        container.__dict__["_rendered_site_files"] = ({}, False)
        monkeypatch.setattr(container, "_render_site_template", lambda *args: "nginx configuration")
    elif name == "120-flare":
        monkeypatch.setattr(container, "_navigation_files", lambda: {"apps.yml": "links: []"})
    else:
        monkeypatch.setattr(container, "render_template", lambda path: "rendered " + path.name)

    expected = container.config_path
    assert len(container._prepared) == 1
    monkeypatch.setattr(builtin_module(name), "prepare_files", _unexpected)
    monkeypatch.setattr(container, "get_config", _unexpected)
    container.on_prepare()
    assert container.config_path == expected
    assert "on_integrating" not in type(container).__dict__
    assert "get_config_revision" not in type(container).__dict__
    assert "_config_revision" not in container.__dict__


def test_lldap_keeps_persistent_initialization_separate_from_prepared_files(tmp_path, monkeypatch):
    container = _container("101-lldap", monkeypatch)
    monkeypatch.setattr(container, "get_app_path", lambda *parts: tmp_path.joinpath(*parts))
    monkeypatch.setattr(container, "render_template", lambda path: "rendered " + path.name)
    monkeypatch.setattr(container, "get_config", lambda key: "LDAP password")
    created, permissions = [], []
    monkeypatch.setattr(container, "_create_secret_file", lambda path, **kwargs: created.append((path.name, kwargs)))
    container.runtime.chmod = lambda *args, **kwargs: permissions.append((args, kwargs))
    context, written = _context(container, explicit=True, services=("lldap",))

    container.on_starting(context)
    assert written == []
    assert (tmp_path / "secrets").is_dir()
    assert (tmp_path / "data").is_dir()
    assert created == [("jwt_secret", {"length": 64})]
    assert permissions == [((tmp_path / "secrets", 0o700), {"recursive": True})]
    monkeypatch.setattr(container, "get_app_path", _unexpected)
    assert container.config_path == Path("/generated") / REVISION
    assert written == [(container, {
        "lldap_config.toml": "rendered lldap_config.toml", "ldap_user_pass": "LDAP password",
    }, {})]


@pytest.mark.parametrize("https", [False, True])
def test_nginx_validation_explicitly_mounts_candidate_without_changing_deployment(tmp_path, monkeypatch, https):
    from copy import deepcopy
    container = _container("100-nginx", monkeypatch)
    container.__dict__.update(sites={},  config_path=tmp_path / "candidate")
    values = {"NGINX_WILDCARD_DOMAIN": False, "NGINX_WAF_ENABLE": False,
              "NGINX_AUTH_ENABLE": False, "NGINX_HTTPS_ENABLE": https, "NGINX_ROOT_DOMAIN": "example.test"}
    monkeypatch.setattr(container, "get_config", lambda key, **kwargs: values[key])
    monkeypatch.setattr(container, "get_app_path", lambda *parts: tmp_path.joinpath("persistent", *parts))
    calls = []
    def validate(context, service, command, **kwargs):
        mounts = kwargs["mount_overrides"]
        assert mounts["/etc/nginx/managed"] == str(tmp_path / "candidate")
        if https:
            certificates = Path(mounts["/etc/certs"])
            assert (certificates / "previous").is_dir()
        calls.append((service, command, kwargs))
        return SimpleNamespace(succeeded=True, stdout="", stderr="", returncode=0)
    container.manager.compose_runner = SimpleNamespace(validate_service=validate)
    context = OperationContext(target_containers=[container], target_services=("nginx",),
                               compose_model={"services": {"nginx": {
                                   "volumes": [{"type": "bind", "source": "/generated/previous-hash",
                                                "target": "/etc/nginx/managed", "read_only": True}],
                               }}})
    original = deepcopy(context.compose_model)

    container.on_check(context)

    assert len(calls) == 1
    assert calls[0][0] == "nginx"
    assert calls[0][2]["check"] is False
    assert context.compose_model == original
    if https:
        assert not Path(calls[0][2]["mount_overrides"]["/etc/certs"]).exists()


def test_authelia_validation_explicitly_mounts_all_candidate_files(tmp_path, monkeypatch):
    container = _container("102-authelia", monkeypatch)
    monkeypatch.setattr(container, "get_config", lambda key: True)
    calls = []
    container.manager.compose_runner = SimpleNamespace(validate_service=lambda *args, **kwargs:
        calls.append((args, kwargs)) or SimpleNamespace(succeeded=True))
    container.__dict__["config_path"] = tmp_path
    context = OperationContext(target_containers=[container], target_services=("authelia", "authelia-admin"))

    container.on_check(context)

    assert calls[0][0][1] == "authelia"
    assert calls[0][1]["mount_overrides"] == {
        "/generated/" + name: str(tmp_path / name)
        for name in (*container.config_files, "authentication_backend_ldap_password")
    }


@pytest.mark.parametrize("state,selected,expected", [
    (None, ("nginx",), False),
    ("running", ("nginx",), True),
    ("restarting", ("nginx",), False),
    ("exited", ("nginx",), False),
    (None, ("nginx", "authelia", "safeline-mgt", "safeline-tengine"), True),
])
def test_selected_nginx_requires_unselected_auth_and_waf_services_running(monkeypatch, state, selected, expected):
    container = _container("100-nginx", monkeypatch)
    container.__dict__["sites"] = {object(): SimpleNamespace(enabled=True, auth=True, waf=True)}
    container.__dict__["config_path"] = Path("/candidate")
    container.manager.containers = {
        "authelia": SimpleNamespace(enable=True),
        "safeline": SimpleNamespace(enable=True, services={"safeline-mgt": {}, "safeline-tengine": {}}),
    }
    monkeypatch.setattr(container, "get_config", lambda key, **kwargs: False)
    calls = []
    container.manager.compose_runner = SimpleNamespace(validate_service=lambda *args, **kwargs:
        calls.append(args) or SimpleNamespace(succeeded=True, stdout="", stderr=""))
    from dataclasses import replace
    instances = tuple(replace(_state(REVISION).services[0], service=name, state=state)
                      for name in ("authelia", "safeline-mgt", "safeline-tengine")) if state else ()
    context = OperationContext(target_containers=[container], target_services=selected,
                               initial_runtime_state=ProjectRuntimeState("test", instances, "docker"))

    if expected:
        container.on_check(context)
        assert len(calls) == 1
    else:
        with pytest.raises(ContainerError, match="requires running integration services"):
            container.on_check(context)
        assert calls == []


def test_unselected_nginx_has_no_implicit_runtime_gate(monkeypatch):
    container = _container("100-nginx", monkeypatch)
    monkeypatch.setattr(container, "get_config", _unexpected)
    container.on_check(OperationContext(target_services=("app",)))


@pytest.mark.parametrize("name", ["100-nginx", "101-lldap", "102-authelia", "120-flare"])
def test_builtin_config_path_defers_materialization_to_start_hook(tmp_path, monkeypatch, name):
    import os
    import stat
    from linktools.cntr.artifacts import prepare_files
    from linktools.cntr.lifecycle.hooks import HookPhase

    container = _container(name, monkeypatch)
    monkeypatch.setattr(builtin_module(name), "prepare_files", prepare_files)
    monkeypatch.setattr(container, "get_app_path", lambda *parts: tmp_path.joinpath(*parts))
    monkeypatch.setattr(container, "get_config", lambda key, **kwargs:
                        os.getgid() if key == "DOCKER_GID" else "saved-password")
    container.manager.data_path = tmp_path
    container.manager.artifact_index = SimpleNamespace(record=lambda entries: None)
    container.runtime.chmod = lambda path, mode: path.chmod(mode)
    if name == "100-nginx":
        container.__dict__["_rendered_site_files"] = ({"sites/generated.conf": "site configuration"}, False)
        monkeypatch.setattr(container, "_render_site_template", lambda *args: "nginx configuration")
        expected = {"sites/generated.conf": "site configuration", "nginx.conf": "nginx configuration"}
    elif name == "120-flare":
        expected = {"apps.yml": "links: []\n", "bookmarks.yml": "categories: []\nlinks: []\n"}
        monkeypatch.setattr(container, "_navigation_files", lambda: expected)
    else:
        monkeypatch.setattr(container, "render_template", lambda path: "rendered " + path.name)
        if name == "101-lldap":
            expected = {"lldap_config.toml": "rendered lldap_config.toml", "ldap_user_pass": "saved-password"}
        else:
            expected = {filename: "rendered " + filename for filename in container.config_files}
            expected["authentication_backend_ldap_password"] = "saved-password"

    destination = container.config_path
    assert destination.parent == tmp_path / "generated"
    assert len(destination.name) == 64
    assert not destination.parent.exists()
    assert container.config_path == destination
    assert len(tuple(container.hooks.iter_phase(HookPhase.BEFORE_START))) == 1
    context = OperationContext()
    container.hooks.call(HookPhase.BEFORE_START, context)

    assert context.file_path(container) == destination
    assert {path.relative_to(destination).as_posix(): path.read_text(encoding="utf-8")
            for path in destination.rglob("*") if path.is_file()} == expected
    for filename in expected:
        info = (destination / filename).stat()
        assert stat.S_IMODE(info.st_mode) == (0o640 if name == "120-flare" else 0o600)
        if name == "120-flare":
            assert info.st_gid == os.getgid()
    assert not (destination.parent / "current").exists()
    assert not container.get_app_path("secrets").exists()
    assert not container.get_app_path("runtime-app").exists()


@pytest.mark.parametrize("name,services,targets", [
    ("nginx", ("nginx",), ("/etc/nginx/managed",)),
    ("lldap", ("lldap",), ("/generated/lldap_config.toml", "/generated/ldap_user_pass")),
    ("authelia", ("authelia", "authelia-admin"), ("/generated/configuration.yml",)),
    ("flare", ("flare",), ("/app/apps.yml", "/app/bookmarks.yml")),
])
def test_builtin_compose_binds_the_immutable_config_path(fresh_manager, tmp_path, name, services, targets):
    container = fresh_manager.containers[name]
    destination = tmp_path / "generated" / ("a" * 64)
    container.__dict__["config_path"] = destination
    specs = container.docker_compose["services"]

    for service in services:
        volumes = specs[service]["volumes"]
        for target in targets:
            mount = next(volume for volume in volumes if volume.endswith(":" + target + ":ro"))
            expected = destination if name == "nginx" else destination / Path(target).name
            assert mount == str(expected) + ":" + target + ":ro"
    assert not destination.exists()
