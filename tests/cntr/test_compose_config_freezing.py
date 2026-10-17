#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Installed Compose consumers freeze configuration before start hooks."""
import pytest

from _harness import configure_builtin_snapshot
from linktools.cntr import OperationContext
from linktools.cntr.lifecycle import HookPhase


@pytest.mark.parametrize("persisted", [False, True])
def test_installed_authelia_compose_freezes_real_templates_once(fresh_manager, monkeypatch, persisted):
    from linktools import utils

    configure_builtin_snapshot(fresh_manager)
    container = fresh_manager.containers["authelia"]
    key = "AUTHELIA_OIDC_CLIENT_SECRET"
    fresh_manager.config.unset(key)
    fresh_manager.config.remove(key)
    calls = []
    monkeypatch.setattr(utils, "random_string", lambda length: calls.append(length) or "fixture-new-secret")
    if persisted:
        fresh_manager.config.persist(key, "fixture-saved-secret")
    expected_secret = "fixture-saved-secret" if persisted else "fixture-new-secret"
    assert "config_path" not in container.__dict__
    assert key not in fresh_manager.config.persisted_keys() or persisted

    selection = fresh_manager.compose_operations.select(["portainer"], for_start=True)
    assert container in selection.project_containers
    assert container not in selection.target_containers
    compose = container.docker_compose
    destination = container.config_path
    assert calls == ([] if persisted else [20])
    assert fresh_manager.config.get(key) == expected_secret
    assert key in fresh_manager.config.persisted_keys()
    assert not destination.exists()
    assert any(str(destination / "configuration.oidc.yml") in mount
               for mount in compose["services"]["authelia"]["volumes"])

    fresh_manager.config.set(key, "fixture-later-secret")
    monkeypatch.setattr(container, "render_template", lambda *args, **kwargs: pytest.fail("second render"))
    monkeypatch.setattr(container, "get_config", lambda *args, **kwargs: pytest.fail("second config resolution"))
    monkeypatch.setattr(container.runtime, "chmod", lambda path, mode: path.chmod(mode))
    context = OperationContext()
    container.hooks.call(HookPhase.BEFORE_START, context)
    assert container.config_path == destination
    assert context.file_path(container) == destination
    content = (destination / "configuration.oidc.yml").read_text(encoding="utf-8")
    assert expected_secret in content
    assert "fixture-later-secret" not in content
    assert calls == ([] if persisted else [20])
    assert not container.get_app_path("secrets", "identity_providers_oidc_jwks").exists()


def test_uninstalled_consumers_are_not_evaluated_when_constructing_compose(fresh_manager, monkeypatch):
    portainer = fresh_manager.containers["portainer"]
    monkeypatch.setattr(fresh_manager.installed_state, "get", lambda resolve=False: [portainer])
    for name in ("nginx", "lldap", "authelia", "flare"):
        container = fresh_manager.containers[name]
        monkeypatch.setattr(type(container), "config_path", property(
            lambda self: pytest.fail("uninstalled consumer config evaluated")))
        monkeypatch.setattr(container, "get_config", lambda *args, **kwargs:
                            pytest.fail("uninstalled consumer config resolved"))
    selection = fresh_manager.compose_operations.select(["portainer"], for_start=True)
    assert selection.project_containers == (portainer,)
    assert {name for owner in selection.project_containers for name in owner.services} == {"portainer"}
    assert all(not fresh_manager.containers[name].enable for name in ("nginx", "lldap", "authelia", "flare"))


def test_read_only_authelia_render_rejects_missing_secret_without_writes(fresh_manager, monkeypatch):
    from linktools import utils
    from linktools.core import Config
    from linktools.errors import ConfigError

    configure_builtin_snapshot(fresh_manager)
    container = fresh_manager.containers["authelia"]
    key = "AUTHELIA_OIDC_CLIENT_SECRET"
    fresh_manager.config.unset(key)
    fresh_manager.config.remove(key)
    before = fresh_manager.config.persisted_keys()

    def forbidden(*args, **kwargs):
        pytest.fail("read-only rendering must not generate or persist credentials")

    monkeypatch.setattr(utils, "random_string", forbidden)
    monkeypatch.setattr(fresh_manager.settings, "set", forbidden)
    monkeypatch.setattr(fresh_manager.settings, "save", forbidden)
    monkeypatch.setattr(fresh_manager.artifact_index, "record", forbidden)
    with Config.read_only_resolution(), pytest.raises(ConfigError, match="read-only resolution cannot create"):
        _ = container.docker_compose
    assert fresh_manager.config.persisted_keys() == before
    assert key not in before
    assert "config_path" not in container.__dict__
    assert not container.get_app_path("generated").exists()
