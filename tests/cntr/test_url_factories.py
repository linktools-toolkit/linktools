#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Module URL factories retain lazy config, port and site-reference semantics."""
from types import SimpleNamespace

import pytest

from linktools.cntr import BaseContainer, ContainerError
from linktools.cntr.urls import load_config_url, load_nginx_url, load_port_url
from linktools.runtime import Proxy


class Container:
    def __init__(self, value):
        self.value = value
        self.reads = []

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


@pytest.mark.parametrize("port", [-1, 0, 65535, 65536])
def test_invalid_port_does_not_resolve_host(port):
    container = Container(port)
    url = load_port_url(container, "PORT", "must-not-appear")
    assert container.reads == []
    assert str(url) == ""
    assert container.reads == [("PORT", {"type": int, "default": 0})]


@pytest.mark.parametrize("port", [1, 443, 65534])
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


@pytest.mark.parametrize("local_id", [None, "", 0, False, ("web",)])
def test_nginx_local_id_is_validated_before_manager_access(local_id):
    with pytest.raises(ContainerError, match="nonempty string"):
        load_nginx_url(object(), local_id)


def test_nginx_tuple_lookup_is_deferred_and_uses_producer_identity():
    reads = []

    class Manager:
        @property
        def nginx_sites(self):
            reads.append("sites")
            return {("app", "web"): SimpleNamespace(url="https://app.test"),
                    ("other", "web"): SimpleNamespace(url="https://other.test")}

    container = SimpleNamespace(name="app", manager=Manager())
    url = load_nginx_url(container, "web", "ui")
    assert reads == []
    assert str(url) == "https://app.test/ui"
    assert reads == ["sites"]
    missing = load_nginx_url(container, "missing")
    assert reads == ["sites"]
    with pytest.raises(ContainerError, match="Unknown nginx site app/missing"):
        str(missing)


def test_url_factories_do_not_extend_base_container():
    for name in ("load_config_url", "load_port_url", "load_nginx_url", "expose_public",
                 "expose_private", "expose_container", "expose_other"):
        assert not hasattr(BaseContainer, name)
