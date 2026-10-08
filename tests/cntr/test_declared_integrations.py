#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Declarative integrations do not depend on navigation registration."""

from linktools.cntr import NginxSite
from linktools.cntr.urls import load_nginx_url


def test_portainer_site_is_independent_of_navigation(fresh_manager):
    portainer = fresh_manager.containers["portainer"]
    baseline = len(portainer.start_hooks)

    first = portainer.integrations["nginx"]["web"]
    assert isinstance(first, NginxSite)
    assert first.proxy == "http://portainer:9000"
    assert first.auth_bypass == (r"\.(css|js)$",)
    assert first is portainer.integrations["nginx"]["web"]

    load_nginx_url(portainer, "web")
    load_nginx_url(portainer, "web", "settings")
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
        links = list(container.integrations.get("flare", ()))
        links.extend(site.expose for site in container.integrations.get("nginx", {}).values()
                     if site.expose is not None)
        assert links
        assert all(isinstance(link, ExposeLink) for link in links)
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
                 "apply_generated_config", "render_nginx_template", "expose_public",
                 "expose_private", "expose_container", "expose_other", "load_config_url",
                 "load_port_url", "load_nginx_url"):
        assert not hasattr(BaseContainer, name)


def test_proxy_runtime_dependency_does_not_select_authelia_admin(fresh_manager):
    operations = fresh_manager.compose_operations
    explicit = operations.select(["portainer"], metadata_only=True, for_start=True)
    selected = operations.start_selection(explicit)
    assert {"authelia", "authelia-redis", "lldap"} <= set(selected.services)
    assert "authelia-admin" not in selected.services


def test_integrations_public_type_accepts_named_or_anonymous_declarations():
    from typing import Iterable, Mapping, Union
    from linktools.cntr import Integration, Integrations

    assert Integrations == Mapping[str, Union[Mapping[str, Integration], Iterable[Integration]]]


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


def test_declarations_have_canonical_public_identity_and_nominal_marker():
    import importlib.util
    from linktools import cntr
    from linktools.cntr import container, integration

    for name in ("Integration", "Integrations", "NginxSite", "ExposeCategory", "ExposeLink"):
        assert getattr(cntr, name) is getattr(integration, name)
    assert issubclass(cntr.NginxSite, cntr.Integration)
    assert issubclass(cntr.ExposeLink, cntr.Integration)
    assert not issubclass(cntr.ExposeCategory, cntr.Integration)
    for name in ("ExposeMixin", "NginxMixin", "ExposeCategory", "ExposeLink", "Integrations"):
        assert not hasattr(container, name)
    assert importlib.util.find_spec("linktools.cntr._container.expose") is None
    for name, description in (("public", "Public"), ("private", "Private"),
                              ("container", "Internal"), ("other", "Tools")):
        category = getattr(cntr.ExposeLink, name)
        assert isinstance(category, cntr.ExposeCategory)
        assert (category.name, category.desc) == (name, description)
        link = category("App", "icon", "")
        assert isinstance(link, cntr.Integration)
        assert link.category is category
        assert link.desc == "App"
        assert link.url is None


def test_flare_iterable_preserves_attached_then_standalone_order():
    from linktools.cntr import ExposeLink

    manager, _ = _site_navigation_manager({
        "web": NginxSite("app.test", expose=ExposeLink.public("Attached", "web", "")),
    }, (ExposeLink.public("First", "web", "", "https://one.test"),
        ExposeLink.public("Second", "web", "", "https://two.test")))
    assert [link["name"] for link in _render_navigation(manager)["apps.yml"]["links"]] == [
        "Attached", "First", "Second"]


def test_navigation_standard_categories_precede_first_seen_custom_categories():
    from linktools.cntr import ExposeCategory, ExposeLink

    team = ExposeCategory("team", "Team")
    tools = ExposeCategory("tools", "Custom tools")
    manager, _ = _site_navigation_manager({}, [
        ExposeLink.container("Internal one", "web", "", "https://internal-one.test"),
        team("Team one", "web", "", "https://team-one.test"),
        ExposeLink.other("Other", "web", "", "https://other.test"),
        ExposeLink.public("App one", "web", "", "https://app-one.test"),
        ExposeLink.private("Private", "web", "", "https://private.test"),
        tools("Tool", "web", "", "https://tool.test"),
        ExposeLink.container("Internal two", "web", "", "https://internal-two.test"),
        ExposeLink.public("App two", "web", "", "https://app-two.test"),
        team("Team two", "web", "", "https://team-two.test"),
    ])
    result = _render_navigation(manager)
    assert [category["id"] for category in result["bookmarks.yml"]["categories"]] == [
        "private", "container", "other", "team", "tools"]
    assert [link["name"] for link in result["bookmarks.yml"]["links"]] == [
        "Private", "Internal one", "Internal two", "Other", "Team one", "Team two", "Tool"]
    assert [link["name"] for link in result["apps.yml"]["links"]] == ["App one", "App two"]


def test_authelia_admin_link_does_not_change_oidc_issuer(fresh_manager):
    fresh_manager.env_config.set("NGINX_ROOT_DOMAIN", "example.test")
    fresh_manager.env_config.set("NGINX_WILDCARD_DOMAIN", True)
    fresh_manager.env_config.set("NGINX_HTTPS_ENABLE", True)
    fresh_manager.env_config.set("NGINX_HTTPS_PORT", 9443)
    authelia = fresh_manager.containers["authelia"]
    site = fresh_manager.nginx_sites[("authelia", "web")]
    assert site.expose.url == "https://sso.example.test:9443/auth-admin"
    assert authelia.oidc_client["issuer_url"] == "https://sso.example.test:9443"
    assert authelia.oidc_client["authorization_url"] == "https://sso.example.test:9443/api/oidc/authorization"


def test_general_templates_expose_url_functions(fresh_manager, tmp_path):
    container = fresh_manager.containers["portainer"]
    fresh_manager.env_config.set("NGINX_ROOT_DOMAIN", "example.test")
    fresh_manager.env_config.set("NGINX_WILDCARD_DOMAIN", True)
    fresh_manager.env_config.set("NGINX_HTTPS_ENABLE", True)
    fresh_manager.env_config.set("NGINX_HTTPS_PORT", 9443)
    template = tmp_path / "docker-compose.yml"
    template.write_text('{{ urls.load_nginx_url(container, "web", "settings", queries={"mode": "a b"}) }}')
    assert container.render_template(template) == "https://portainer.example.test:9443/settings?mode=a+b"
