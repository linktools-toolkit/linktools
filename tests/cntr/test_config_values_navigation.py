#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Navigation and access consumers resolve declarations in their producer's config."""
from types import MappingProxyType, SimpleNamespace
from typing import TYPE_CHECKING

import pytest
import yaml

from linktools.cntr import ContainerError, ContainerManager
from linktools.cntr.ext import Authelia, Flare, Integration, Nginx, ResolvedSite, endpoint
from linktools.core import Config, ConfigFormat, ConfigLazy, ConfigRef, ConfigSchema, RuntimeOverrideSource
from linktools.runtime import lazy_load
from linktools.types import MISSING

from _harness import builtin_container_type

if TYPE_CHECKING:
    from typing import Any


class _Config(Config):
    def __init__(self, **values: "Any") -> None:
        super().__init__(None, ConfigSchema(), [RuntimeOverrideSource()])
        defaults = {"NGINX_AUTH_ENABLE": True, "NGINX_HTTPS_ENABLE": True,
                    "NGINX_HTTPS_PORT": 443, "AUTHELIA_MIN_AUTH_LEVEL": 2}
        defaults.update(values)
        for key, value in defaults.items():
            self.set(key, value)
        self.reads = []

    def get(self, key: "Any", type: "Any" = None, default: "Any" = MISSING) -> "Any":
        self.reads.append(key)
        return super().get(key, type=type, default=default)


def _fail(config: "Config") -> None:
    raise AssertionError("unused declaration must not resolve")


def _manager(*entries: "tuple[str, Config, Any]", nginx: bool = True) -> "ContainerManager":
    manager = object.__new__(ContainerManager)
    manager.env_config = _Config()
    manager.containers = {}
    manager.integration_snapshot = {"flare": (), "authelia": ()}
    for name, builtin in (("flare", "120-flare"), ("authelia", "102-authelia")):
        consumer = object.__new__(builtin_container_type(builtin))
        consumer._name = name
        consumer.manager = manager
        consumer.get_config = manager.env_config.get
        manager.containers[name] = consumer
    manager.containers["authelia"].__dict__["public_url"] = "https://sso.test"
    manager.containers["nginx"] = SimpleNamespace(sites={})
    if nginx:
        manager.integration_snapshot["nginx"] = ()
    for order, (name, config, declarations) in enumerate(entries):
        producer = SimpleNamespace(name=name, order=order, env_config=config,
                                   manager=manager, get_config=config.get)
        manager.containers[name] = producer
        values = declarations(producer) if callable(declarations) else declarations
        manager.integration_snapshot[name] = tuple(Integration.iter_declarations(values))
    manager.containers["nginx"].sites = ResolvedSite.collect(manager)
    return manager


def _navigation(manager: "ContainerManager", name: str = "apps.yml") -> "dict[str, Any]":
    return yaml.safe_load(manager.containers["flare"]._navigation_files()[name])


def test_config_values_navigation_construction_and_scan_are_lazy() -> None:
    value = ConfigLazy(_fail)
    link = Flare.public(value, value, value, value)
    access = Authelia.access(value)
    callbacks = Authelia.oidc(value)
    declarations = endpoint(object(), domain=value, extensions=[link, access, callbacks])
    assert type(declarations) is tuple
    assert tuple(Integration.iter_declarations(declarations)) == declarations
    assert declarations[0].server_name is value
    assert declarations[1]._url is link._url is value
    assert declarations[2].site is declarations[0]
    assert declarations[3]._redirect_uris is callbacks._redirect_uris is value
    assert access.site is None
    assert repr(declarations)


@pytest.mark.parametrize("builtin", ["120-flare", "102-authelia", "103-safeline", "110-portainer"])
def test_builtin_navigation_declarations_do_not_read_auth_configuration(builtin: str) -> None:
    container = object.__new__(builtin_container_type(builtin))
    container.get_config = lambda *args, **kwargs: _fail(None)
    container.get_source_path = lambda *args: "/templates/nginx.conf"
    declarations = tuple(Integration.iter_declarations(container.integrations))
    assert isinstance(declarations[0].server_name, ConfigRef)
    assert isinstance(declarations[0].auth, ConfigLazy)


def test_flare_shared_declaration_uses_each_producer_config() -> None:
    link = Flare.public(ConfigRef("NAME"), ConfigRef("ICON"), ConfigRef("DESC"),
                        ConfigFormat("https://{domain}/{{literal}}", domain=ConfigRef("DOMAIN")))
    first = _Config(NAME="First", ICON="one", DESC="", DOMAIN="first.test")
    second = _Config(NAME="Second", ICON="two", DESC="Details", DOMAIN="second.test")
    manager = _manager(("first", first, [link]), ("second", second, [link]))
    assert first.reads == second.reads == []
    assert manager.integration_snapshot["first"][0] is manager.integration_snapshot["second"][0] is link
    assert _navigation(manager)["links"] == [
        {"name": "First", "icon": "one", "desc": "First", "link": "https://first.test/{literal}"},
        {"name": "Second", "icon": "two", "desc": "Details", "link": "https://second.test/{literal}"},
    ]
    assert first.reads == second.reads == ["DOMAIN", "NAME", "ICON", "DESC"]


@pytest.mark.parametrize("url", [None, "", False, 0, MISSING])
def test_flare_empty_resolved_url_skips_metadata_and_base(url: "Any") -> None:
    config = _Config()
    link = Flare.public(ConfigLazy(_fail), ConfigLazy(_fail), ConfigLazy(_fail),
                        ConfigLazy(lambda current: url)).with_base_url(ConfigLazy(_fail))
    manager = _manager(("app", config, [link]))
    assert _navigation(manager)["links"] == []
    assert config.reads == []


def test_flare_bookmark_does_not_resolve_unused_description() -> None:
    link = Flare.bookmark(ConfigRef("NAME"), "web", "https://docs.test", desc=ConfigLazy(_fail))
    manager = _manager(("app", _Config(NAME="Docs"), [link]))
    assert _navigation(manager, "bookmarks.yml")["links"] == [
        {"name": "Docs", "icon": "web", "link": "https://docs.test", "category": "other"},
    ]


@pytest.mark.parametrize("reference,expected", [
    ("/ui?q=a%20b#settings", "https://app.test/ui?q=a%20b#settings"),
    ("{url}/ui", "https://app.test/ui"),
    ("relative", "https://app.test/relative"),
    ("https://external.test/cb?q=%23", "https://external.test/cb?q=%23"),
])
def test_flare_endpoint_resolves_config_url_before_relative_composition(reference: str, expected: str) -> None:
    config = _Config(DOMAIN="app.test", URL=reference)
    raw_url = ConfigRef("URL")
    link = Flare.public("App", "web", url=raw_url)
    manager = _manager(("app", config, lambda producer: endpoint(
        producer, domain=ConfigRef("DOMAIN"), extensions=[link])))
    declaration = manager.integration_snapshot["app"][1]
    assert declaration._url is raw_url
    assert config.reads == []
    assert _navigation(manager)["links"][0]["link"] == expected
    assert config.reads[0] == "URL"
    if reference.startswith("https:"):
        assert config.reads == ["URL"]


def test_flare_default_endpoint_url_resolves_config_domain() -> None:
    config = _Config(DOMAIN="app.test")
    manager = _manager(("app", config, lambda producer: endpoint(
        producer, domain=ConfigRef("DOMAIN"), extensions=[Flare.public("App", "web")])))
    assert config.reads == []
    assert _navigation(manager)["links"][0]["link"] == "https://app.test"


@pytest.mark.parametrize("domain,nginx", [("", True), ("app.test", False)])
def test_flare_relative_config_url_skips_metadata_when_site_is_disabled(domain: str, nginx: bool) -> None:
    config = _Config(DOMAIN=domain, URL="/ui")
    link = Flare.public(ConfigLazy(_fail), ConfigLazy(_fail), ConfigLazy(_fail), ConfigRef("URL"))
    manager = _manager(("app", config, lambda producer: endpoint(
        producer, domain=ConfigRef("DOMAIN"), extensions=[link])), nginx=nginx)
    assert _navigation(manager)["links"] == []
    assert config.reads == (["URL", "DOMAIN"] if nginx else ["URL"])


@pytest.mark.parametrize("field", ["name", "icon"])
def test_flare_required_metadata_is_checked_after_config_resolution(field: str) -> None:
    fields = {"name": "App", "icon": "web", field: ConfigLazy(lambda config: MISSING)}
    declaration = Flare.public(url="https://app.test", **fields)
    with pytest.raises(ValueError, match="requires an? " + field):
        declaration.resolve_metadata(_Config())


def test_flare_bound_config_values_recompute_without_cross_config_cache() -> None:
    calls = []
    link = Flare.public("App", "web", url=ConfigLazy(
        lambda config: calls.append(config) or ConfigRef("PATH"))).with_base_url(ConfigRef("BASE"))
    first = _Config(PATH="first", BASE="https://one.test/")
    second = _Config(PATH="second", BASE="https://two.test/")
    assert link.resolve_url(first) == "https://one.test/first"
    assert link.resolve_url(second) == "https://two.test/second"
    assert link.resolve_url(first) == "https://one.test/first"
    assert calls == [first, second, first]


def test_authelia_mapping_rule_resolves_nested_fields_for_each_producer() -> None:
    native = MappingProxyType({"policy": ConfigRef("POLICY"),
                               "subject": [ConfigFormat("group:{group}", group=ConfigRef("GROUP"))],
                               "query": [{"key": "mode", "value": ConfigRef("MODE")}],
                               "methods": ("GET", "POST")})
    declaration = Authelia.access(ConfigLazy(lambda config: native))
    first = _Config(DOMAIN="one.test", POLICY="one_factor", GROUP="staff", MODE="one")
    second = _Config(DOMAIN="two.test", POLICY="deny", GROUP="admins", MODE="two")
    manager = _manager(("first", first, [Nginx.site(ConfigRef("DOMAIN")), declaration]),
                       ("second", second, [Nginx.site(ConfigRef("DOMAIN")), declaration]))
    assert first.reads == second.reads == []
    assert manager.containers["authelia"].acl_rules == [
        {"policy": "one_factor", "subject": ["group:staff"], "query": [{"key": "mode", "value": "one"}],
         "methods": ("GET", "POST"), "domain": "one.test"},
        {"policy": "deny", "subject": ["group:admins"], "query": [{"key": "mode", "value": "two"}],
         "methods": ("GET", "POST"), "domain": "two.test"},
    ]
    assert declaration.site is None and "domain" not in native
    assert isinstance(native["policy"], ConfigRef)
    resolved = declaration.resolve_rule(first)
    with pytest.raises(TypeError):
        resolved["policy"] = "bypass"


@pytest.mark.parametrize("domain,auth,nginx", [("", True, True), ("app.test", False, True),
                                              ("app.test", True, False)])
def test_authelia_disabled_access_skips_config_rule(domain: str, auth: bool, nginx: bool) -> None:
    config = _Config(DOMAIN=domain, AUTH=auth)
    site = Nginx.site(ConfigRef("DOMAIN"), auth=ConfigRef("AUTH"))
    declaration = Authelia.access(ConfigLazy(_fail))
    manager = _manager(("app", config, [site, declaration]), nginx=nginx)
    assert manager.containers["authelia"].acl_rules == []
    if not nginx:
        assert config.reads == []
    elif not domain:
        assert config.reads == ["DOMAIN"]
    else:
        assert config.reads == ["DOMAIN", "AUTH"]


def test_authelia_callbacks_resolve_nested_values_with_exact_deduplication() -> None:
    declaration = Authelia.oidc(ConfigLazy(lambda config: [
        ConfigRef("URL"), ConfigLazy(lambda current: ConfigRef("URL")),
        ConfigFormat("custom:{name}", name=ConfigRef("NAME")), "",
    ]))
    first = _Config(URL="https://one.test/callback?q=%23", NAME="one")
    second = _Config(URL="https://two.test/callback", NAME="two")
    manager = _manager(("first", first, [declaration]), ("second", second, [declaration]))
    assert first.reads == second.reads == []
    assert manager.containers["authelia"].oidc_redirects == (
        "https://sso.test", "https://one.test/callback?q=%23", "custom:one",
        "https://two.test/callback", "custom:two",
    )


def test_authelia_endpoint_does_not_complete_callbacks_after_config_values() -> None:
    config = _Config(DOMAIN="app.test", CALLBACK="/callback?q=%2F", RETURN="{url}/return")
    callbacks = Authelia.oidc((ConfigRef("CALLBACK"), ConfigRef("RETURN"), ConfigRef("CALLBACK")))
    manager = _manager(("app", config, lambda producer: endpoint(
        producer, domain=ConfigRef("DOMAIN"), extensions=[callbacks])))
    assert manager.containers["authelia"].oidc_redirects == (
        "https://sso.test",
    )
    assert "DOMAIN" not in config.reads


@pytest.mark.parametrize("url", ["https://app.test/callback#fragment",
                                "https://app.test/{{callback}}", 42, None,
                                ["https://nested.test/callback"]])
def test_authelia_validates_callbacks_after_config_resolution(url: "Any") -> None:
    declaration = Authelia.oidc((ConfigLazy(lambda config: url),))
    with pytest.raises(ContainerError, match="Authelia redirect URI"):
        declaration.resolve_redirect_uris(_Config())


@pytest.mark.parametrize("url", ["//other.test/callback", "/callback#fragment"])
def test_authelia_incomplete_callbacks_are_filtered_after_config_resolution(url: str) -> None:
    declaration = Authelia.oidc((ConfigRef("URL"),))
    assert declaration.resolve_redirect_uris(_Config(URL=url)) == ()


def test_authelia_callback_sequence_and_rule_types_are_checked_after_resolution() -> None:
    with pytest.raises(ContainerError, match="must be a sequence"):
        Authelia.oidc(ConfigRef("URL")).resolve_redirect_uris(_Config(URL="https://app.test"))
    with pytest.raises(ContainerError, match="rule must be a mapping"):
        Authelia.access(ConfigLazy(lambda config: [])).resolve_rule(_Config())


def test_authelia_empty_and_absolute_callbacks_need_no_site() -> None:
    declaration = Authelia.oidc((ConfigRef("EMPTY"), ConfigRef("URL")))
    assert declaration.resolve_redirect_uris(_Config(EMPTY="", URL="custom:callback")) == ("custom:callback",)


def test_navigation_legacy_proxy_results_remain_success_cached() -> None:
    calls = []
    url = lazy_load(lambda: calls.append("url") or "relative")
    base = lazy_load(lambda: calls.append("base") or "https://app.test/")
    link = Flare.public("App", "web", url=url).with_base_url(base)
    rule = Authelia.access(lazy_load(lambda: calls.append("rule") or {"policy": "one_factor"}))
    callbacks = Authelia.oidc(lazy_load(lambda: calls.append("callbacks") or ("https://app.test/cb",)))
    assert calls == []
    for config in (_Config(), _Config()):
        assert link.resolve_url(config) == link.url == "https://app.test/relative"
        assert rule.resolve_rule(config) == rule.rule == {"policy": "one_factor"}
        assert callbacks.resolve_redirect_uris(config) == callbacks.redirect_uris == ("https://app.test/cb",)
    assert calls == ["url", "base", "rule", "callbacks"]


def test_authelia_native_consumer_keeps_command_snapshot_cache() -> None:
    config = _Config(POLICY="one_factor")
    declaration = Authelia.access(policy=ConfigRef("POLICY"))
    manager = _manager(("app", config, [Nginx.site("app.test"), declaration]))
    consumer = manager.containers["authelia"]
    rules = consumer.acl_rules
    config.set("POLICY", "deny")
    assert consumer.acl_rules is rules
    assert consumer.acl_rules[0]["policy"] == "one_factor"
    assert declaration.resolve_rule(config)["policy"] == "deny"
