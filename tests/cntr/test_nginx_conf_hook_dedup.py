#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""load_nginx_url's start-hook dedup key must cover every parameter that
shapes the generated nginx conf, not just proxy_conf/proxy_url -- otherwise
two different domains proxying to the same backend collapse into one hook
and only the first domain's conf is ever written.
"""
from linktools.core import ConfigField


def test_same_backend_with_different_domains_registers_two_hooks(fresh_manager, monkeypatch):
    container = fresh_manager.containers["portainer"]
    baseline = len(container.start_hooks)

    written = []
    monkeypatch.setattr(container, "write_nginx_conf", lambda **kwargs: written.append(kwargs))

    container.load_nginx_url(
        ConfigField(name="TEST_APP_DOMAIN", default="app.example.com"),
        proxy_url="http://backend:8080",
    )
    container.load_nginx_url(
        ConfigField(name="TEST_ADMIN_DOMAIN", default="admin.example.com"),
        proxy_url="http://backend:8080",
    )

    added = container.start_hooks[baseline:]
    assert len(added) == 2

    for hook in added:
        hook()

    assert {call["domain"] for call in written} == {"app.example.com", "admin.example.com"}


def test_same_nginx_exposure_evaluated_twice_is_deduplicated(fresh_manager, monkeypatch):
    container = fresh_manager.containers["portainer"]
    baseline = len(container.start_hooks)

    monkeypatch.setattr(container, "write_nginx_conf", lambda **kwargs: None)

    field = ConfigField(name="TEST_SAME_DOMAIN", default="same.example.com")
    container.load_nginx_url(field, proxy_url="http://backend:9090")
    container.load_nginx_url(field, proxy_url="http://backend:9090")

    assert len(container.start_hooks) - baseline == 1
