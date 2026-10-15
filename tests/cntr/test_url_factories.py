#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Public URL factories retain lazy config, port and site-reference semantics."""
from types import SimpleNamespace

import pytest

from linktools.cntr import BaseContainer, ContainerError
from linktools.cntr.ext import Nginx, load_config_url, load_nginx_url, load_port_url
from linktools.core import Config, ConfigSchema
from linktools.runtime import Proxy


class Container:
    def __init__(self, value):
        self.value = value
        self.reads = []
        self.env_config = Config(None, ConfigSchema(), [])
        self.env_config.get = self.get_config

    def get_config(self, key, **kwargs):
        self.reads.append((key, kwargs))
        return self.value

    @property
    def host(self):
        self.reads.append("host")
        return "host.test"


def test_config_factory_is_lazy_and_preserves_path_queries():
    container = Container("https://config.test/base")
    url = load_config_url(container, "URL", "settings", queries={"mode": "a b"})
    assert isinstance(url, Proxy)
    assert container.reads == []
    assert str(url) == "https://config.test/base/settings?mode=a+b"
    assert str(url) == "https://config.test/base/settings?mode=a+b"
    assert container.reads == [("URL", {"type": str, "default": None})]


@pytest.mark.parametrize("value", [None, ""])
def test_empty_config_url_remains_empty(value):
    container = Container(value)
    assert str(load_config_url(container, "URL", "must-not-appear")) == ""


@pytest.mark.parametrize("port", [-1, 0, 65536])
def test_invalid_port_does_not_resolve_host(port):
    container = Container(port)
    url = load_port_url(container, "PORT", "must-not-appear")
    assert container.reads == []
    assert str(url) == ""
    assert container.reads == [("PORT", {"type": int, "default": 0})]


@pytest.mark.parametrize("port", [1, 443, 65534, 65535])
def test_valid_port_resolves_host_lazily_and_preserves_boundaries(port):
    container = Container(port)
    url = load_port_url(container, "PORT", "ui", queries={"q": "a b"})
    assert container.reads == []
    expected_port = "" if port == 443 else ":%s" % port
    assert str(url) == "https://host.test%s/ui?q=a+b" % expected_port
    assert container.reads == [("PORT", {"type": int, "default": 0}), "host"]


def test_http_port_factory_retains_literal_template_path():
    container = Container(80)
    assert str(load_port_url(container, "PORT", "{{path}}", https=False)) == "http://host.test/{{path}}"


@pytest.mark.parametrize("reference", ["", 0, False, ("web",), {}])
def test_invalid_nginx_reference_is_rejected_when_consumed(reference):
    container = SimpleNamespace(name="app", manager=SimpleNamespace(
        containers={"nginx": SimpleNamespace(sites={})}))
    url = load_nginx_url(container, reference)
    with pytest.raises(ContainerError, match="reference|declaration"):
        str(url)


def test_nginx_object_lookup_is_deferred_and_uses_producer_identity():
    reads = []
    declaration = Nginx.site("app.test")
    foreign = Nginx.site("other.test")
    container = SimpleNamespace(name="app", env_config=Config(None, ConfigSchema(), []))
    other = SimpleNamespace(name="other")

    class Consumer:
        @property
        def sites(self):
            reads.append("sites")
            return {declaration: SimpleNamespace(producer=container, get_url=lambda default="": "https://app.test"),
                    foreign: SimpleNamespace(producer=other, get_url=lambda default="": "https://other.test")}

    container.manager = SimpleNamespace(containers={"nginx": Consumer()})
    url = load_nginx_url(container, declaration, "ui")
    assert reads == []
    assert str(url) == "https://app.test/ui"
    assert reads == ["sites"]
    for reference in (foreign, Nginx.site("app.test")):
        missing = load_nginx_url(container, reference)
        before = len(reads)
        with pytest.raises(ContainerError, match="reference the original endpoint"):
            str(missing)
        assert len(reads) == before + 1


def test_url_factories_do_not_extend_base_container():
    for name in ("load_config_url", "load_port_url", "load_nginx_url", "expose_public",
                 "expose_private", "expose_container", "expose_other"):
        assert not hasattr(BaseContainer, name)
