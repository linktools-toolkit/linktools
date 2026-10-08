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
        links = dict(container.integrations.get("flare", {}))
        links.update({local_id: site.expose for local_id, site in
                      container.integrations.get("nginx", {}).items() if site.expose is not None})
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


def test_integrations_public_type_is_open_nested_mapping():
    from typing import Mapping
    from linktools.cntr import Integrations

    assert Integrations == Mapping[str, Mapping[str, object]]


def _site_navigation_manager(declarations, links=None, nginx=True):
    from types import SimpleNamespace
    from linktools.cntr._nginx import ResolvedSite

    reads = []
    def config(key, **kwargs):
        reads.append(key)
        return {"NGINX_HTTPS_ENABLE": True, "NGINX_HTTPS_PORT": 9443}[key]
    manager = SimpleNamespace()
    producer = SimpleNamespace(name="app", order=10, manager=manager, get_config=config)
    manager.containers = {"app": producer}
    manager.integration_snapshot = {"flare": {}, "app": {"nginx": declarations, "flare": links or {}}}
    if nginx:
        manager.integration_snapshot["nginx"] = {}
    manager.nginx_sites = {(producer.name, key): ResolvedSite(producer, key, value)
                           for key, value in declarations.items()}
    return manager, reads


def _render_navigation(manager):
    from types import SimpleNamespace
    from linktools.cntr.generation import FlareGeneration
    import yaml

    return {key: yaml.safe_load(value) for key, value in
            FlareGeneration(SimpleNamespace(manager=manager)).render("candidate").items()}


def test_site_navigation_inherits_only_omitted_url_lazily():
    from linktools.cntr import ExposeCategory

    public = ExposeCategory("public", "Public")
    omitted = public("Inherited", "web", "")
    manager, reads = _site_navigation_manager({
        "inherited": NginxSite("app.test", expose=omitted),
        "empty": NginxSite("empty.test", expose=public("Empty", "web", "", "")),
        "none": NginxSite("none.test", expose=public("None", "web", "", None)),
        "explicit": NginxSite("explicit.test", expose=public("Explicit", "web", "", "custom://{{port}}")),
        "silent": NginxSite("silent.test"),
    }, {"omitted": public("Unbound", "web", "")})
    inherited = manager.nginx_sites[("app", "inherited")].expose
    assert reads == []
    assert omitted.url is None
    assert inherited is not omitted
    result = _render_navigation(manager)
    assert result["apps.yml"]["links"] == [
        {"name": "Inherited", "icon": "web", "desc": "Inherited", "link": "https://app.test:9443"},
        {"name": "Explicit", "icon": "web", "desc": "Explicit", "link": "custom://{{port}}"},
    ]
    assert reads == ["NGINX_HTTPS_ENABLE", "NGINX_HTTPS_PORT"]
    assert omitted.url is None


def test_disabled_or_uninstalled_sites_do_not_resolve_navigation_defaults():
    from linktools.cntr import ExposeCategory
    from linktools.runtime import lazy_load

    public = ExposeCategory("public", "Public")
    def fail():
        raise AssertionError("disabled URL must not resolve")
    for server_name, nginx in (("", True), ("app.test", False)):
        manager, reads = _site_navigation_manager({"web": NginxSite(
            server_name, url=lazy_load(fail), expose=public("App", "web", ""),
        )}, nginx=nginx)
        assert _render_navigation(manager)["apps.yml"]["links"] == []
        assert reads == []


def test_site_navigation_category_does_not_change_auth_and_paths_are_independent():
    from linktools.cntr import ExposeCategory

    category = ExposeCategory("team", "Team")
    declaration = NginxSite("app.test", auth=True, expose=category("Root", "web", ""))
    manager, reads = _site_navigation_manager({"web": declaration}, {
        "one": category("Path", "web", "", "https://app.test/one"),
        "two": category("Path", "web", "", "https://app.test/two?q=1"),
    })
    result = _render_navigation(manager)
    assert result["apps.yml"]["links"] == []
    assert [link["link"] for link in result["bookmarks.yml"]["links"]] == [
        "https://app.test:9443", "https://app.test/one", "https://app.test/two?q=1"]
    assert declaration.auth is True
    assert "NGINX_AUTH_ENABLE" not in reads


def test_attached_navigation_rejects_invalid_values_and_category_conflicts():
    import pytest
    from linktools.cntr import ContainerError, ExposeCategory

    manager, _ = _site_navigation_manager({"web": NginxSite("app.test", expose=True)})
    with pytest.raises(ContainerError, match="expose must be an ExposeLink"):
        _render_navigation(manager)
    manager, _ = _site_navigation_manager({"web": NginxSite(
        "app.test", expose=ExposeCategory("team", "Team")("Root", "web", ""),
    )}, {"other": ExposeCategory("team", "Different")("Other", "web", "", "https://other.test")})
    with pytest.raises(ContainerError, match="Conflicting description"):
        _render_navigation(manager)


def test_builtin_navigation_matches_complete_output_baseline(fresh_manager):
    import json
    from pathlib import Path

    for key, value in {
        "HOST": "host.example.test", "NGINX_ROOT_DOMAIN": "example.test",
        "NGINX_WILDCARD_DOMAIN": True, "NGINX_HTTPS_ENABLE": True,
        "NGINX_HTTPS_PORT": 9443, "LLDAP_WEB_PORT": 17170,
        "SAFELINE_PORT": 9200, "PORTAINER_PORT": 9000, "FLARE_PORT": 5000,
    }.items():
        fresh_manager.env_config.set(key, value)
    expected = json.loads((Path(__file__).parent / "snapshots/navigation.json").read_text(encoding="utf-8"))
    assert _render_navigation(fresh_manager) == expected
    assert sum(len(data["links"]) for data in expected.values()) == 7


def test_navigation_merge_preserves_producer_ties_and_local_id_collisions():
    from types import SimpleNamespace
    from linktools.cntr import ExposeCategory
    from linktools.cntr._nginx import ResolvedSite

    public = ExposeCategory("public", "Public")
    manager, _ = _site_navigation_manager({
        "web": NginxSite("one.test", expose=public("One", "web", "")),
        "two": NginxSite("two.test", expose=public("Two", "web", "")),
    }, {"web": public("Path", "web", "", "https://one.test/path")})
    producer = SimpleNamespace(name="other", order=10, manager=manager)
    manager.containers["other"] = producer
    manager.integration_snapshot["other"] = {"nginx": {"web": NginxSite(
        "other.test", expose=public("Other", "web", "", "https://other.test"),
    )}}
    manager.nginx_sites[("other", "web")] = ResolvedSite(
        producer, "web", manager.integration_snapshot["other"]["nginx"]["web"])
    assert [entry["name"] for entry in _render_navigation(manager)["apps.yml"]["links"]] == [
        "One", "Two", "Path", "Other"]


def test_absent_flare_does_not_resolve_attached_navigation():
    from linktools.cntr import ExposeCategory
    from linktools.runtime import lazy_load

    def fail():
        raise AssertionError("absent consumer must not resolve URL")
    manager, reads = _site_navigation_manager({"web": NginxSite(
        "app.test", expose=ExposeCategory("public", "Public")("App", "web", "", lazy_load(fail)),
    )})
    del manager.integration_snapshot["flare"]
    assert _render_navigation(manager)["apps.yml"]["links"] == []
    assert reads == []
