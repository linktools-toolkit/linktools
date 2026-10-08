#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Declarative integrations do not depend on navigation registration."""

from linktools.cntr import NginxSite


def test_portainer_site_is_independent_of_navigation(fresh_manager):
    portainer = fresh_manager.containers["portainer"]
    baseline = len(portainer.start_hooks)

    first = portainer.integrations["nginx"]["web"]
    assert isinstance(first, NginxSite)
    assert first.proxy == "http://portainer:9000"
    assert first.auth_bypass == (r"\.(css|js)$",)
    assert first is portainer.integrations["nginx"]["web"]

    portainer.load_nginx_url("web")
    portainer.load_nginx_url("web", "settings")
    assert len(portainer.start_hooks) == baseline


def test_nginx_consumes_sites_without_exposure_side_effects(fresh_manager):
    portainer = fresh_manager.containers["portainer"]
    original_hooks = len(portainer.start_hooks)

    entries = list(fresh_manager.iter_integrations("nginx"))
    matches = [(producer, site_id, site) for producer, site_id, site in entries
               if producer is portainer and site_id == "web"]
    assert len(matches) == 1
    assert matches[0][2] is portainer.integrations["nginx"]["web"]
    assert len(portainer.start_hooks) == original_hooks


def test_navigation_is_declared_without_resolving_lazy_urls(fresh_manager, monkeypatch):
    from linktools.cntr import BaseContainer, ExposeLink

    def fail(*args, **kwargs):
        raise AssertionError("navigation URLs must stay lazy")

    monkeypatch.setattr(fresh_manager, "nginx_sites", {})
    for name in ("lldap", "authelia", "safeline", "portainer", "flare"):
        container = fresh_manager.containers[name]
        original = container.get_config
        monkeypatch.setattr(container, "get_config", lambda key, *args, _get=original, **kwargs:
                            _get(key, *args, **kwargs) if key.endswith("AUTH_ENABLE") else fail())
        links = container.integrations["flare"]
        assert links
        assert all(isinstance(link, ExposeLink) for link in links.values())
    assert not hasattr(BaseContainer, "exposes")


def test_preparation_does_not_access_navigation(fresh_manager, monkeypatch):
    from linktools.cntr import BaseContainer

    def fail(self):
        raise AssertionError("prepare must not read navigation")

    monkeypatch.setattr(BaseContainer, "exposes", property(fail), raising=False)
    fresh_manager.prepare_installed_containers()


def test_absent_flare_does_not_consume_navigation(fresh_manager, monkeypatch):
    installed = [c for c in fresh_manager.installed_state.get(resolve=True) if c.name != "flare"]
    monkeypatch.setattr(fresh_manager.installed_state, "get", lambda resolve=False: installed)
    assert list(fresh_manager.iter_integrations("flare")) == []
    assert fresh_manager.integration_snapshot["portainer"]["flare"]
    selection = fresh_manager.compose_operations.select(["portainer"], metadata_only=True, for_start=True)
    assert "flare" not in [c.name for c in fresh_manager.compose_operations.start_selection(selection).target_containers]


def test_partial_nginx_selection_refreshes_full_navigation_snapshot(fresh_manager):
    import yaml

    fresh_manager.env_config.set("NGINX_ROOT_DOMAIN", "example.test")
    fresh_manager.env_config.set("NGINX_WILDCARD_DOMAIN", True)
    fresh_manager.env_config.set("NGINX_HTTPS_PORT", 9443)
    operations = fresh_manager.compose_operations
    selection = operations.select(["nginx"], metadata_only=True, for_start=True)
    synchronized = operations.sync_selection(selection)
    flare = fresh_manager.containers["flare"]
    assert flare in synchronized
    result = fresh_manager.generated_configs["flare"].render("candidate")
    links = yaml.safe_load(result["apps.yml"])["links"]
    portainer = next(link for link in links if link["name"] == "Portainer")
    assert ":9443" in portainer["link"]
    assert any(link["name"] == "Authelia" for link in links)


def test_container_authoring_has_one_integration_entry():
    from linktools.cntr import BaseContainer

    assert hasattr(BaseContainer, "integrations")
    for name in ("exposes", "config_sources", "integration_requires_start", "generated_config_path",
                 "prepare_generated_config", "render_generated_config", "validate_generated_config",
                 "apply_generated_config", "render_nginx_template"):
        assert not hasattr(BaseContainer, name)


def test_proxy_runtime_dependency_does_not_select_authelia_admin(fresh_manager):
    operations = fresh_manager.compose_operations
    explicit = operations.select(["portainer"], metadata_only=True, for_start=True)
    selected = operations.start_selection(explicit)
    assert {"authelia", "authelia-redis", "lldap"} <= set(selected.services)
    assert "authelia-admin" not in selected.services
