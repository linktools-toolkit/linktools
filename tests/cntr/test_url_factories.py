#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Public URL factories retain lazy config, port and site-reference semantics."""
from linktools.core import ConfigLazy
from linktools.cntr.ext import ResolvedSite
from linktools.core import ConfigRef
from types import SimpleNamespace
from collections import UserList

import pytest

from linktools.cntr import BaseContainer
from linktools.errors import ConfigError
from linktools.cntr.ext import Nginx, load_config_url, load_nginx_url, load_port_url
from linktools.core import Config, ConfigSchema


class Container:
    def __init__(self, value):
        self.value = value
        self.reads = []
        self.config = Config(None, ConfigSchema(), [])
        self.config.get = self.get_config

    def get_config(self, key, **kwargs):
        self.reads.append((key, kwargs))
        return "host.test" if key == "HOST" else self.value


def test_config_factory_is_lazy_and_preserves_path_queries():
    container = Container("https://config.test/base")
    url = load_config_url(ConfigRef("URL", type=str, default=None), "settings", queries={"mode": "a b"})
    assert isinstance(url, ConfigLazy)
    assert container.reads == []
    assert container.config.resolve(url) == "https://config.test/base/settings?mode=a+b"
    assert container.config.resolve(url) == "https://config.test/base/settings?mode=a+b"
    assert container.reads == [("URL", {"type": str, "default": None})] * 2


@pytest.mark.parametrize("value", [None, ""])
def test_empty_config_url_remains_empty(value):
    container = Container(value)
    assert container.config.resolve(load_config_url(ConfigRef("URL", type=str, default=None), "must-not-appear")) == ""


@pytest.mark.parametrize("port", [-1, 0, 65536])
def test_invalid_port_does_not_resolve_host(port):
    container = Container(port)
    url = load_port_url(ConfigRef("PORT", type=int, default=0), "must-not-appear")
    assert container.reads == []
    assert container.config.resolve(url) == ""
    assert container.reads == [("PORT", {"type": int, "default": 0})]


@pytest.mark.parametrize("port", [1, 443, 65534, 65535])
def test_valid_port_resolves_host_lazily_and_preserves_boundaries(port):
    container = Container(port)
    url = load_port_url(ConfigRef("PORT", type=int, default=0), "ui", queries={"q": "a b"})
    assert container.reads == []
    expected_port = "" if port == 443 else ":%s" % port
    assert container.config.resolve(url) == "https://host.test%s/ui?q=a+b" % expected_port
    assert container.reads == [("PORT", {"type": int, "default": 0}), ("HOST", {"type": str})]


def test_http_port_factory_retains_literal_template_path():
    container = Container(80)
    assert container.config.resolve(load_port_url(ConfigRef("PORT", type=int, default=0), "{{path}}", https=False)) == "http://host.test/{{path}}"


@pytest.mark.parametrize("reference", ["", 0, False, ("web",), {}])
def test_invalid_nginx_reference_is_rejected_when_consumed(reference):
    container = SimpleNamespace(name="app", config=Config(None, ConfigSchema(), []), manager=SimpleNamespace(
        containers={"nginx": SimpleNamespace(sites={})}))
    url = load_nginx_url(ResolvedSite.find(container, reference))
    with pytest.raises(ConfigError, match="ConfigLazy callback failed"):
        container.config.resolve(url)


def test_nginx_object_lookup_is_deferred_and_uses_producer_identity():
    reads = []
    declaration = Nginx.site("app.test")
    foreign = Nginx.site("other.test")
    container = SimpleNamespace(name="app", config=Config(None, ConfigSchema(), []))
    other = SimpleNamespace(name="other", config=container.config)
    app_site = ResolvedSite(container, declaration, 0)
    app_site.get_url = lambda default="": "https://app.test"
    foreign_site = ResolvedSite(other, foreign, 1)
    foreign_site.get_url = lambda default="": "https://other.test"

    class Consumer:
        @property
        def sites(self):
            reads.append("sites")
            return {declaration: app_site, foreign: foreign_site}

    container.manager = SimpleNamespace(containers={"nginx": Consumer()})
    url = load_nginx_url(ResolvedSite.find(container, declaration), "ui")
    assert reads == []
    assert container.config.resolve(url) == "https://app.test/ui"
    assert reads == ["sites"]
    for reference in (foreign, Nginx.site("app.test")):
        missing = load_nginx_url(ResolvedSite.find(container, reference))
        before = len(reads)
        with pytest.raises(ConfigError, match="ConfigLazy callback failed"):
            container.config.resolve(missing)
        assert len(reads) == before + 1


def test_url_factories_do_not_extend_base_container():
    for name in ("load_config_url", "load_port_url", "load_nginx_url", "expose_public",
                 "expose_private", "expose_container", "expose_other"):
        assert not hasattr(BaseContainer, name)


def test_nginx_url_accepts_resolved_site_and_resolves_query_in_consuming_config():
    from linktools.core import ConfigRef, DictSource
    producer = SimpleNamespace(name="app")
    site = ResolvedSite(producer, Nginx.site("app.test"), 0)
    site.get_url = lambda default="": "https://app.test"
    value = load_nginx_url(site, ConfigLazy(lambda config: "ui"), queries={"q": ConfigRef("QUERY")})
    first = Config(None, ConfigSchema(), [DictSource({"QUERY": "one"})])
    second = Config(None, ConfigSchema(), [DictSource({"QUERY": "two"})])
    assert isinstance(value, ConfigLazy)
    assert first.resolve(value) == "https://app.test/ui?q=one"
    assert second.resolve(value) == "https://app.test/ui?q=two"
    assert first.resolve(value) == "https://app.test/ui?q=one"


def test_nginx_url_does_not_resolve_query_without_public_base():
    from linktools.core import ConfigRef
    site = ResolvedSite(SimpleNamespace(name="app"), Nginx.site(""), 0)
    site.get_url = lambda default="": ""
    value = load_nginx_url(site, ConfigRef("MISSING_PATH"), queries={"q": ConfigRef("MISSING")})
    assert Config(None, ConfigSchema(), []).resolve(value) == ""


def test_nginx_url_resolves_site_reference_in_each_consuming_config():
    from linktools.core import ConfigRef, DictSource
    producer = SimpleNamespace(name="app")
    one = ResolvedSite(producer, Nginx.site("one.test"), 0)
    two = ResolvedSite(producer, Nginx.site("two.test"), 1)
    one.get_url = lambda default="": "https://one.test"
    two.get_url = lambda default="": "https://two.test"
    value = load_nginx_url(ConfigLazy(lambda config: config.get("SITE")), "ui", queries={"q": ConfigRef("QUERY")})
    first = Config(None, ConfigSchema(), [DictSource({"SITE": one, "QUERY": "first"})])
    second = Config(None, ConfigSchema(), [DictSource({"SITE": two, "QUERY": "second"})])
    assert first.resolve(value) == "https://one.test/ui?q=first"
    assert second.resolve(value) == "https://two.test/ui?q=second"
    assert first.resolve(value) == "https://one.test/ui?q=first"


@pytest.mark.parametrize("reference", ["web", Nginx.site("app.test"), (Nginx.site("app.test"),)])
def test_nginx_url_does_not_implicitly_select_a_site(reference):
    value = load_nginx_url(reference)
    with pytest.raises(ConfigError, match="ConfigLazy callback failed"):
        Config(None, ConfigSchema(), []).resolve(value)


def test_nginx_url_unbound_endpoint_reference_fails_without_reading_dependents():
    reads = []
    path = ConfigLazy(lambda config: reads.append("path") or "/ui")
    query = ConfigLazy(lambda config: reads.append("secret") or {"secret": "fixture"})
    value = load_nginx_url(path=path, queries=query)
    assert reads == []
    with pytest.raises(ConfigError):
        Config(None, ConfigSchema(), []).resolve(value)
    assert reads == []


@pytest.mark.parametrize("path", ["/ui/#/setup", "", ConfigLazy(lambda config: "/ui/#/setup")])
def test_nginx_url_keyword_path_matches_positional_path(path):
    producer = SimpleNamespace(name="app")
    site = ResolvedSite(producer, Nginx.site("app.test"), 0)
    site.get_url = lambda default="": "https://app.test"
    config = Config(None, ConfigSchema(), [])
    query = {"q": "a b"}
    assert config.resolve(load_nginx_url(site, path=path, queries=query)) == config.resolve(
        load_nginx_url(site, path, queries=query))


@pytest.mark.parametrize("factory", [load_config_url, load_port_url, load_nginx_url])
def test_url_helpers_reject_multiple_positional_paths(factory):
    with pytest.raises(TypeError):
        factory(None, "ui", "settings")


@pytest.mark.parametrize("factory", [load_config_url, load_port_url, load_nginx_url])
@pytest.mark.parametrize("path,expected", [
    ("settings", "/settings"),
    (["ui", "settings"], "/ui/settings"),
    (("ui", "settings"), "/ui/settings"),
    (ConfigLazy(lambda config: "settings"), "/settings"),
    (ConfigLazy(lambda config: ["ui", "settings"]), "/ui/settings"),
    (ConfigLazy(lambda config: ("ui", "settings")), "/ui/settings"),
    ([ConfigRef("PATH"), ConfigLazy(lambda config: "settings")], "/ui/settings"),
    ((ConfigRef("PATH"), ConfigLazy(lambda config: "settings")), "/ui/settings"),
    (ConfigRef("PATHS"), "/ui/settings"),
    (UserList([ConfigRef("PATH"), "settings"]), "/ui/settings"),
    (ConfigLazy(lambda config: UserList([ConfigRef("PATH"), "settings"])), "/ui/settings"),
    (None, ""),
    ("", ""),
    ([], ""),
    ((), ""),
    (ConfigLazy(lambda config: None), ""),
])
def test_url_helpers_single_path_contract(factory, path, expected):
    from linktools.core import DictSource
    config = Config(None, ConfigSchema(), [DictSource({
        "HOST": "app.test", "PATH": "ui", "PATHS": ["ui", "settings"],
    })])
    site = ResolvedSite(SimpleNamespace(name="app"), Nginx.site("app.test"), 0)
    site.get_url = lambda default="": "https://app.test"
    base = site if factory is load_nginx_url else 443 if factory is load_port_url else "https://app.test"
    for helper in (factory(base, path, queries={"q": "a b"}),
                   factory(base, path=path, queries={"q": "a b"})):
        assert config.resolve(helper) == "https://app.test%s?q=a+b" % expected


@pytest.mark.parametrize("factory", [load_config_url, load_port_url, load_nginx_url])
def test_url_helpers_invalid_base_skips_whole_path(factory):
    reads = []
    path = ConfigLazy(lambda config: reads.append("path") or [ConfigRef("MISSING")])
    query = ConfigLazy(lambda config: reads.append("query") or {"secret": ConfigRef("MISSING")})
    site = ResolvedSite(SimpleNamespace(name="app"), Nginx.site(""), 0)
    site.get_url = lambda default="": ""
    base = site if factory is load_nginx_url else 0 if factory is load_port_url else ""
    assert Config(None, ConfigSchema(), []).resolve(factory(base, path=path, queries=query)) == ""
    assert reads == []
