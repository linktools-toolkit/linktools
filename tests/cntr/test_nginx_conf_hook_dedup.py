#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Navigation never owns proxy registration or mutates declaration identities."""
from linktools.cntr.ext import ResolvedSite
from types import SimpleNamespace

from linktools.cntr import BaseContainer
from linktools.cntr.ext import Nginx, load_nginx_url
from linktools.cntr.lifecycle import HookRegistry
from linktools.core import Config, ConfigSchema, DictSource


def test_same_site_can_have_many_navigation_links_without_hooks():
    class Links:
        name = "app"
        hooks = HookRegistry()
        env_config = Config(None, ConfigSchema(), [DictSource({})])
    container = Links()
    declaration = Nginx.site("app.example.com")
    site = ResolvedSite(container, declaration, 0)
    site.get_url = lambda default="": "https://app.example.com"
    container.manager = SimpleNamespace(containers={"nginx": SimpleNamespace(sites={
        declaration: site,
    })})
    first = load_nginx_url(ResolvedSite.find(container, None))
    second = load_nginx_url(ResolvedSite.find(container, None), "admin")
    assert container.env_config.resolve(first) == "https://app.example.com"
    assert container.env_config.resolve(second) == "https://app.example.com/admin"
    assert container.hooks.describe() == []


def test_sites_share_backends_without_sharing_identity():
    one = Nginx.site("app.example.com", proxy="http://backend:8080")
    two = Nginx.site("admin.example.com", proxy="http://backend:8080")
    assert one is not two
    assert one.proxy == two.proxy


def test_legacy_mutating_entrypoints_are_absent():
    assert not hasattr(BaseContainer, "load_exist_nginx_url")
    assert not hasattr(BaseContainer, "write_nginx_conf")
