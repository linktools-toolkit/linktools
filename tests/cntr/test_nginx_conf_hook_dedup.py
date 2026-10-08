#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Navigation never owns proxy registration or mutates declaration identities."""
from types import SimpleNamespace

from linktools.cntr import BaseContainer, Nginx
from linktools.cntr.integration import load_nginx_url
from linktools.cntr.lifecycle import HookRegistry


def test_same_site_can_have_many_navigation_links_without_hooks():
    class Links:
        name = "app"
        hooks = HookRegistry()
        manager = SimpleNamespace(nginx_sites={
            ("app", "web"): SimpleNamespace(url="https://app.example.com"),
        })
    container = Links()
    first = load_nginx_url(container, "web")
    second = load_nginx_url(container, "web", "admin")
    assert str(first) == "https://app.example.com"
    assert str(second) == "https://app.example.com/admin"
    assert container.hooks.describe() == []


def test_sites_share_backends_without_sharing_identity():
    one = Nginx.site("app.example.com", proxy="http://backend:8080")
    two = Nginx.site("admin.example.com", proxy="http://backend:8080")
    assert one is not two
    assert one.proxy == two.proxy


def test_legacy_mutating_entrypoints_are_absent():
    assert not hasattr(BaseContainer, "load_exist_nginx_url")
    assert not hasattr(BaseContainer, "write_nginx_conf")
