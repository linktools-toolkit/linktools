#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Declarative integrations do not depend on navigation registration."""

from linktools.cntr import Nginx
from linktools.cntr.ext import load_nginx_url
from linktools.cntr.lifecycle import HookPhase


def test_portainer_site_is_independent_of_navigation(fresh_manager):
    portainer = fresh_manager.containers["portainer"]
    baseline = len(list(portainer.hooks.iter_phase(HookPhase.BEFORE_START)))

    first = next(value for value in portainer.integrations if isinstance(value, Nginx))
    assert isinstance(first, Nginx)
    assert first.proxy == "http://portainer:9000"
    assert first.auth_bypass == (r"\.(css|js)$",)
    assert first is next(value for value in portainer.integrations if isinstance(value, Nginx))
    assert first.local_id == "web"

    load_nginx_url(portainer, "web")
    load_nginx_url(portainer, "web", "settings")
    assert len(list(portainer.hooks.iter_phase(HookPhase.BEFORE_START))) == baseline


def test_nginx_consumes_sites_without_exposure_side_effects(fresh_manager):
    portainer = fresh_manager.containers["portainer"]
    original_hooks = len(list(portainer.hooks.iter_phase(HookPhase.BEFORE_START)))

    entries = list(fresh_manager.iter_integrations("nginx"))
    matches = [(producer, site) for producer, site in entries
               if producer is portainer and site.local_id == "web"]
    assert len(matches) == 1
    assert matches[0][1] in portainer.integrations
    assert len(list(portainer.hooks.iter_phase(HookPhase.BEFORE_START))) == original_hooks


def test_nginx_owns_one_site_snapshot_shared_by_all_consumers(fresh_manager, monkeypatch) -> None:
    import pytest
    from linktools.cntr import ContainerManager
    from linktools.cntr.ext import ResolvedSite

    calls = []
    collect = ResolvedSite.collect

    def counted(cls: "type[ResolvedSite]", manager: "ContainerManager") -> "object":
        calls.append(manager)
        return collect(manager)

    monkeypatch.setattr(ResolvedSite, "collect", classmethod(counted))
    nginx = fresh_manager.containers["nginx"]
    sites = nginx.sites
    authelia = fresh_manager.containers["authelia"]
    assert str(load_nginx_url(authelia, "web")) == authelia.public_url
    fresh_manager.containers["flare"]._navigation_files()
    assert nginx.sites is sites
    assert calls == [fresh_manager]
    assert not hasattr(ContainerManager, "nginx_sites")
    assert not hasattr(fresh_manager, "nginx_sites")
    with pytest.raises(TypeError):
        sites[("app", "web")] = None


def test_navigation_is_declared_without_resolving_lazy_urls(fresh_manager, monkeypatch):
    from linktools.cntr import BaseContainer, Flare

    def fail(*args, **kwargs):
        raise AssertionError("navigation URLs must stay lazy")

    monkeypatch.setattr(fresh_manager.containers["nginx"], "sites", {})
    for name in ("lldap", "authelia", "safeline", "portainer", "flare"):
        container = fresh_manager.containers[name]
        original = container.get_config
        monkeypatch.setattr(container, "get_config", lambda key, *args, _get=original, **kwargs:
                            _get(key, *args, **kwargs) if key.endswith("AUTH_ENABLE") else fail())
        links = [value for value in container.integrations if isinstance(value, Flare)]
        links.extend(site.link for site in container.integrations
                     if isinstance(site, Nginx) and site.link is not None)
        assert links
        assert all(isinstance(link, Flare) for link in links)
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
    assert any(value.consumer == "flare" for value in fresh_manager.integration_snapshot["portainer"])
    selection = fresh_manager.compose_operations.select(["portainer"], for_start=True)
    assert "flare" not in [c.name for c in fresh_manager.compose_operations.start_selection(selection).target_containers]


def test_partial_nginx_selection_refreshes_full_navigation_snapshot(fresh_manager):
    import yaml

    fresh_manager.env_config.set("NGINX_ROOT_DOMAIN", "example.test")
    fresh_manager.env_config.set("NGINX_WILDCARD_DOMAIN", True)
    fresh_manager.env_config.set("NGINX_HTTPS_PORT", 9443)
    operations = fresh_manager.compose_operations
    selection = operations.select(["nginx"], for_start=True)
    synchronized = selection.project_containers
    flare = fresh_manager.containers["flare"]
    assert flare in synchronized
    result = fresh_manager.containers["flare"]._navigation_files()
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
                 "load_port_url", "load_nginx_url", "get_nginx_domain"):
        assert not hasattr(BaseContainer, name)


def test_proxy_runtime_dependency_does_not_select_authelia_admin(fresh_manager):
    operations = fresh_manager.compose_operations
    explicit = operations.select(["portainer"], for_start=True)
    selected = operations.start_selection(explicit)
    assert {"authelia", "authelia-redis", "lldap"} <= set(selected.services)
    assert "authelia-admin" not in selected.services


def test_integrations_public_type_is_a_flat_iterable() -> None:
    from typing import Iterable
    from linktools.cntr import Integration, Integrations

    assert Integrations == Iterable[Integration]


def _site_navigation_manager(declarations, links=None, nginx=True):
    from collections.abc import Mapping
    from types import SimpleNamespace
    from linktools.cntr.ext import ResolvedSite

    reads = []
    def config(key, **kwargs):
        reads.append(key)
        return {"NGINX_HTTPS_ENABLE": True, "NGINX_HTTPS_PORT": 9443}[key]
    manager = SimpleNamespace()
    producer = SimpleNamespace(name="app", order=10, manager=manager, get_config=config)
    manager.containers = {"app": producer, "nginx": SimpleNamespace(sites={})}
    for local_id, site in declarations.items():
        site.local_id = local_id
    links = links.values() if isinstance(links, Mapping) else (links or ())
    manager.integration_snapshot = {"flare": (), "app": tuple(declarations.values()) + tuple(links)}
    if nginx:
        manager.integration_snapshot["nginx"] = ()
    manager.containers["nginx"].sites = {(producer.name, key): ResolvedSite(producer, key, value)
                           for key, value in declarations.items()}
    return manager, reads


def _render_navigation(manager):
    from _harness import builtin_container_type
    FlareContainer = builtin_container_type("120-flare")
    import yaml

    container = object.__new__(FlareContainer)
    container.manager = manager
    return {key: yaml.safe_load(value) for key, value in
            container._navigation_files().items()}


def test_site_navigation_inherits_only_omitted_url_lazily():
    from linktools.cntr import Flare

    public = Flare.category("public", "Public", apps=True)
    omitted = public("Inherited", "web", "")
    manager, reads = _site_navigation_manager({
        "inherited": Nginx.site("app.test", link=omitted),
        "empty": Nginx.site("empty.test", link=public("Empty", "web", "", "")),
        "none": Nginx.site("none.test", link=public("None", "web", "", None)),
        "explicit": Nginx.site("explicit.test", link=public("Explicit", "web", "", "custom://{{port}}")),
        "silent": Nginx.site("silent.test"),
    }, {"omitted": public("Unbound", "web", "")})
    inherited = manager.containers["nginx"].sites[("app", "inherited")].link
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
    from linktools.cntr import Flare
    from linktools.runtime import lazy_load

    public = Flare.category("public", "Public", apps=True)
    def fail():
        raise AssertionError("disabled URL must not resolve")
    for server_name, nginx in (("", True), ("app.test", False)):
        manager, reads = _site_navigation_manager({"web": Nginx.site(
            server_name, public_url=lazy_load(fail), link=public("App", "web", ""),
        )}, nginx=nginx)
        assert _render_navigation(manager)["apps.yml"]["links"] == []
        assert reads == []


def test_site_navigation_category_does_not_change_auth_and_paths_are_independent():
    from linktools.cntr import Flare

    category = Flare.category("team", "Team")
    declaration = Nginx.site("app.test", auth=True, link=category("Root", "web", ""))
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
    from linktools.cntr import ContainerError, Flare

    manager, _ = _site_navigation_manager({"web": Nginx.site("app.test", link=True)})
    with pytest.raises(ContainerError, match="link must be a Flare"):
        _render_navigation(manager)
    manager, _ = _site_navigation_manager({"web": Nginx.site(
        "app.test", link=Flare.category("team", "Team")("Root", "web", ""),
    )}, {"other": Flare.category("team", "Different")("Other", "web", "", "https://other.test")})
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
    from linktools.cntr import Flare
    from linktools.cntr.ext import ResolvedSite

    public = Flare.category("public", "Public", apps=True)
    manager, _ = _site_navigation_manager({
        "web": Nginx.site("one.test", link=public("One", "web", "")),
        "two": Nginx.site("two.test", link=public("Two", "web", "")),
    }, {"web": public("Path", "web", "", "https://one.test/path")})
    producer = SimpleNamespace(name="other", order=10, manager=manager)
    manager.containers["other"] = producer
    site = Nginx.site(
        "other.test", link=public("Other", "web", "", "https://other.test"),
    )
    manager.integration_snapshot["other"] = (site,)
    manager.containers["nginx"].sites[("other", "web")] = ResolvedSite(
        producer, "web", site)
    assert [entry["name"] for entry in _render_navigation(manager)["apps.yml"]["links"]] == [
        "One", "Two", "Path", "Other"]


def test_absent_flare_does_not_resolve_attached_navigation(fresh_manager, monkeypatch):
    from linktools.cntr import Flare
    from linktools.runtime import lazy_load

    def fail():
        raise AssertionError("absent consumer must not resolve URL")

    installed = [c for c in fresh_manager.installed_state.get(resolve=True) if c.name != "flare"]
    monkeypatch.setattr(fresh_manager.installed_state, "get", lambda resolve=False: installed)
    monkeypatch.setattr(fresh_manager.containers["portainer"], "integrations", [Nginx.site(
        "app.test", link=Flare.category("public", "Public", apps=True)("App", "web", "", lazy_load(fail)),
    )])
    assert "flare" not in {c.name for c in fresh_manager.installed_state.get(resolve=True)}
    assert list(fresh_manager.iter_integrations("flare")) == []


def test_declarations_have_canonical_public_identity_and_nominal_marker():
    import importlib.util
    from linktools import cntr
    from linktools.cntr import container, ext

    for name in ("Integration", "Integrations", "Nginx", "Flare", "Authelia"):
        assert getattr(cntr, name) is getattr(ext, name)
    assert issubclass(cntr.Nginx, cntr.Integration)
    assert issubclass(cntr.Flare, cntr.Integration)
    for name in ("ExposeCategory", "ExposeLink", "FlareCategory", "FlareLink", "NginxSite"):
        assert not hasattr(cntr, name)
        assert not hasattr(ext, name)
    for name in ("ExposeMixin", "NginxMixin", "FlareCategory", "FlareLink", "Integrations"):
        assert not hasattr(container, name)
    assert importlib.util.find_spec("linktools.cntr._container.expose") is None
    for name, description in (("public", "Public"), ("private", "Private"),
                              ("container", "Internal"), ("other", "Tools")):
        link = (cntr.Flare.public("App", "icon", "") if name == "public"
                else cntr.Flare.bookmark("App", "icon", category=name))
        category = link.display_category
        assert not isinstance(category, cntr.Integration)
        assert (category.name, category.desc) == (name, description)
        assert category.apps is (name == "public")
        assert category.order == {"public": 100, "private": 10, "container": 20, "other": 30}[name]
        assert isinstance(link, cntr.Flare)
        assert link.desc == "App"
        assert link.url is None


def test_flare_iterable_preserves_attached_then_standalone_order():
    from linktools.cntr import Flare

    manager, _ = _site_navigation_manager({
        "web": Nginx.site("app.test", link=Flare.public("Attached", "web", "")),
    }, (Flare.public("First", "web", "", "https://one.test"),
        Flare.public("Second", "web", "", "https://two.test")))
    assert [link["name"] for link in _render_navigation(manager)["apps.yml"]["links"]] == [
        "Attached", "First", "Second"]


def test_navigation_standard_categories_precede_first_seen_custom_categories():
    from linktools.cntr import Flare

    team = Flare.category("team", "Team")
    tools = Flare.category("tools", "Custom tools")
    manager, _ = _site_navigation_manager({}, [
        Flare.container("Internal one", "web", "https://internal-one.test"),
        team("Team one", "web", "", "https://team-one.test"),
        Flare.bookmark("Other", "web", "https://other.test", category="other"),
        Flare.public("App one", "web", "", "https://app-one.test"),
        Flare.bookmark("Private", "web", "https://private.test", category="private"),
        tools("Tool", "web", "", "https://tool.test"),
        Flare.bookmark("Internal two", "web", "https://internal-two.test", category="container"),
        Flare.public("App two", "web", "", "https://app-two.test"),
        team("Team two", "web", "", "https://team-two.test"),
    ])
    result = _render_navigation(manager)
    assert [category["id"] for category in result["bookmarks.yml"]["categories"]] == [
        "private", "container", "other", "team", "tools"]
    assert [link["name"] for link in result["bookmarks.yml"]["links"]] == [
        "Private", "Internal one", "Internal two", "Other", "Team one", "Team two", "Tool"]
    assert [link["name"] for link in result["apps.yml"]["links"]] == ["App one", "App two"]


def test_flare_category_output_area_is_independent_of_name() -> None:
    from linktools.cntr import Flare

    dashboard = Flare.category("dashboard", "Dashboard", apps=True)
    favorites = Flare.category("favorites", "Favorites", apps=True)
    public = Flare.category("public", "Public bookmarks")
    manager, _ = _site_navigation_manager({
        "web": Nginx.site("app.test", link=dashboard("Attached", "web", "Application")),
    }, [
        public("Bookmark", "web", "", "https://bookmark.test"),
        favorites("Favorite", "web", "", "https://favorite.test"),
        dashboard("Custom app", "web", "Custom", "https://custom.test"),
    ])
    result = _render_navigation(manager)
    assert [link["name"] for link in result["apps.yml"]["links"]] == ["Attached", "Favorite", "Custom app"]
    assert [link["desc"] for link in result["apps.yml"]["links"]] == ["Application", "Favorite", "Custom"]
    assert result["bookmarks.yml"] == {
        "categories": [{"id": "public", "title": "Public bookmarks"}],
        "links": [{"category": "public", "name": "Bookmark", "icon": "web", "link": "https://bookmark.test"}],
    }


def test_flare_bookmark_order_is_explicit_and_ties_keep_first_seen_order() -> None:
    from linktools.cntr import Flare

    team = Flare.category("team", "Team", order=5)
    docs = Flare.category("docs", "Documentation", order=5)
    manager, _ = _site_navigation_manager({}, [
        Flare.bookmark("Internal", "web", "https://internal.test", category="container"),
        docs("Docs one", "web", "", "https://docs.test/one"),
        team("Team", "web", "", "https://team.test"),
        Flare.category("docs", "Documentation", order=5)("Docs two", "web", "", "https://docs.test/two"),
    ])
    result = _render_navigation(manager)["bookmarks.yml"]
    assert [category["id"] for category in result["categories"]] == ["docs", "team", "container"]
    assert [link["name"] for link in result["links"]] == ["Docs one", "Docs two", "Team", "Internal"]


def test_flare_rejects_conflicting_category_output_areas_and_orders() -> None:
    import pytest
    from linktools.cntr import ContainerError, Flare

    for options, message in (({"apps": True}, "output area"), ({"order": 5}, "order")):
        manager, _ = _site_navigation_manager({}, [
            Flare.category("team", "Team")("One", "web", "", "https://one.test"),
            Flare.category("team", "Team", **options)("Two", "web", "", "https://two.test"),
        ])
        with pytest.raises(ContainerError, match="Conflicting " + message):
            _render_navigation(manager)


def test_flare_bookmarks_accept_custom_category_ids_and_inherit_site_urls() -> None:
    from linktools.cntr import Flare

    manager, reads = _site_navigation_manager({
        "web": Nginx.site("app.test", link=Flare.bookmark("Attached", "web", category="tool")),
    }, [
        Flare.public("App", "apps", "Application description", "https://app.test"),
        Flare.bookmark("Standalone", "book", "https://docs.test", category="tool"),
        Flare.bookmark("Unbound", "off", category="empty"),
        Flare.bookmark("Disabled", "off", None, category="empty"),
    ])
    assert reads == []
    result = _render_navigation(manager)
    assert result["apps.yml"]["links"] == [
        {"name": "App", "icon": "apps", "desc": "Application description", "link": "https://app.test"}]
    assert result["bookmarks.yml"] == {
        "categories": [{"id": "tool", "title": "tool"}],
        "links": [
            {"category": "tool", "name": "Attached", "icon": "web", "link": "https://app.test:9443"},
            {"category": "tool", "name": "Standalone", "icon": "book", "link": "https://docs.test"},
        ],
    }


def test_flare_container_factory_preserves_lazy_urls() -> None:
    from linktools.cntr import Flare
    from linktools.runtime import lazy_load

    reads = []
    def load_url():
        reads.append("url")
        return "https://internal.test"

    link = Flare.container("Internal", "web", lazy_load(load_url))
    assert isinstance(link, Flare)
    assert link.display_category is Flare.category("container")
    assert link.desc == "Internal"
    assert link.with_default_url("https://fallback.test") is link
    manager, _ = _site_navigation_manager({}, [link])
    assert reads == []
    result = _render_navigation(manager)
    assert result["apps.yml"]["links"] == []
    assert result["bookmarks.yml"] == {
        "categories": [{"id": "container", "title": "Internal"}],
        "links": [{"category": "container", "name": "Internal", "icon": "web",
                   "link": "https://internal.test"}],
    }
    assert reads == ["url"]


def test_flare_container_factory_inherits_only_omitted_site_urls() -> None:
    from linktools.cntr import Flare

    omitted = Flare.container("Inherited", "web")
    empty = Flare.container("Empty", "web", "")
    disabled = Flare.container("Disabled", "web", None)
    assert empty.with_default_url("https://fallback.test") is empty
    assert disabled.with_default_url("https://fallback.test") is disabled
    manager, reads = _site_navigation_manager({
        "inherited": Nginx.site("app.test", link=omitted),
        "empty": Nginx.site("empty.test", link=empty),
        "disabled": Nginx.site("disabled.test", link=disabled),
    }, [Flare.container("Unbound", "web")])
    assert reads == []
    result = _render_navigation(manager)
    assert result["bookmarks.yml"]["links"] == [
        {"category": "container", "name": "Inherited", "icon": "web",
         "link": "https://app.test:9443"},
    ]
    assert reads == ["NGINX_HTTPS_ENABLE", "NGINX_HTTPS_PORT"]
    assert omitted.url is None


def test_flare_bookmark_factories_preserve_category_description_and_url_contract() -> None:
    from linktools.cntr import Flare
    from linktools.runtime import lazy_load
    from linktools.types import MISSING

    def fail():
        raise AssertionError("bookmark URL must stay lazy")

    custom = Flare.category("team", "Team", order=5)
    for category in ("container", "other", custom):
        group = Flare.category(category) if isinstance(category, str) else category
        for desc in (None, "", "Detailed description"):
            for url in (MISSING, None, "", "https://app.test", lazy_load(fail)):
                original = group("App", "web", desc, url)
                links = [Flare.bookmark("App", "web", url, category=category, desc=desc)]
                if category == "container":
                    links.append(Flare.container("App", "web", url, desc=desc))
                for link in links:
                    assert link.display_category is original.display_category
                    assert (link.name, link.icon, link.desc) == (original.name, original.icon, original.desc)
                    assert link._url is original._url


def test_flare_bookmarks_accept_category_titles_and_orders() -> None:
    from linktools.cntr import Flare

    tools = Flare.category("tool", "Tools", order=5)
    manager, _ = _site_navigation_manager({}, [
        Flare.bookmark("Internal", "web", "https://internal.test", category="container"),
        Flare.bookmark("Custom", "tool", "https://tool.test", category=tools),
        Flare.bookmark("Other", "web", "https://other.test"),
        Flare.bookmark("Private", "web", "https://private.test", category="private"),
    ])
    result = _render_navigation(manager)["bookmarks.yml"]
    assert result["categories"] == [
        {"id": "tool", "title": "Tools"},
        {"id": "private", "title": "Private"},
        {"id": "container", "title": "Internal"},
        {"id": "other", "title": "Tools"},
    ]
    assert [link["name"] for link in result["links"]] == ["Custom", "Private", "Internal", "Other"]


def test_flare_bookmarks_reject_application_categories() -> None:
    import pytest
    from linktools.cntr import Flare

    with pytest.raises(ValueError, match="bookmarks output area"):
        Flare.bookmark("App", "web", category=Flare.category("public", "Public", apps=True))
    with pytest.raises(TypeError, match="string or _FlareCategory"):
        Flare.bookmark("Invalid", "web", category=None)


def test_authelia_admin_link_does_not_change_oidc_issuer(fresh_manager):
    fresh_manager.env_config.set("NGINX_ROOT_DOMAIN", "example.test")
    fresh_manager.env_config.set("NGINX_WILDCARD_DOMAIN", True)
    fresh_manager.env_config.set("NGINX_HTTPS_ENABLE", True)
    fresh_manager.env_config.set("NGINX_HTTPS_PORT", 9443)
    authelia = fresh_manager.containers["authelia"]
    site = fresh_manager.containers["nginx"].sites[("authelia", "web")]
    assert site.link.url == "https://sso.example.test:9443/auth-admin"
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


def test_namespace_factories_preserve_typed_constructor_and_mixed_list():
    import inspect
    from linktools.cntr import Flare

    assert "server_name" in inspect.signature(Nginx.site).parameters
    declarations = [Nginx.site("app.example.com"),
                    Flare.bookmark("Tools", "web", "https://tools.example.com", category="tool")]
    assert isinstance(declarations[0], Nginx)
    assert isinstance(declarations[1], Flare)
    assert declarations[0].server_name == "app.example.com"
    assert declarations[0].local_id == "web"


def test_installed_container_metadata_reuses_actual_instances(fresh_manager):
    from linktools.cntr import BaseContainer

    assert not hasattr(BaseContainer, "integration_consumer")
    assert not hasattr(fresh_manager, "integration_consumers")
    installed = fresh_manager.load_installed_config_metadata()
    nginx = fresh_manager.containers["nginx"]
    assert nginx in installed
    assert all(container is fresh_manager.containers[container.name] for container in installed)


def test_integration_containers_preserve_dependencies(fresh_manager):
    assert tuple(fresh_manager.containers["authelia"].dependencies) == ("nginx", "lldap")
    assert tuple(fresh_manager.containers["safeline"].dependencies) == ("nginx",)
    installed = fresh_manager.resolver.resolve_dependencies([fresh_manager.containers["authelia"]])
    assert {container.name for container in installed} == {"nginx", "lldap", "authelia"}
    selection = fresh_manager.compose_operations.select(["portainer"], for_start=True)
    started = fresh_manager.compose_operations.start_selection(selection)
    assert "lldap" in started.services


def test_flare_display_category_does_not_shadow_factory():
    from linktools.cntr.ext import Flare

    category = Flare.category("team", "Team")
    link = category("Docs", "web", "Docs")
    assert link.display_category is category
    assert callable(link.category)
    assert link.with_default_url("https://docs.test").display_category is category
