#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Verifies load_nginx_url's start-hook identity and idempotent registration.

The dedup key must include every parameter that shapes the generated nginx
conf, so two different domains proxying to the same backend each get their
own conf written, while re-evaluating the same exposure still registers only
one hook.
"""
from linktools.core import ConfigField
from linktools.cntr._container.expose import _freeze


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


def test_freeze_sorts_sets_into_a_canonical_order():
    # set iteration order is a function of insertion/deletion history, not
    # just content, so _freeze must not rely on it: two sets built
    # differently but holding the same elements must freeze identically.
    assert _freeze({3, 1, 2}) == _freeze({1, 2, 3}) == ("set", (1, 2, 3))


def test_freeze_does_not_conflate_list_and_set_of_same_elements():
    assert _freeze(["a", "b"]) != _freeze({"a", "b"})


def test_same_backend_with_equivalent_but_differently_built_auth_extra_sets_is_deduplicated(
        fresh_manager, monkeypatch):
    container = fresh_manager.containers["portainer"]
    baseline = len(container.start_hooks)

    monkeypatch.setattr(container, "write_nginx_conf", lambda **kwargs: None)

    uris_a = {"/oauth/callback", "/login/callback"}
    uris_b = set()
    for item in ("/login/callback", "/oauth/callback", "/tmp1", "/tmp2"):
        uris_b.add(item)
    for item in ("/tmp1", "/tmp2"):
        uris_b.discard(item)

    field = ConfigField(name="TEST_AUTH_EXTRA_DOMAIN", default="auth.example.com")
    container.load_nginx_url(field, proxy_url="http://backend:9091", auth_extra={"uris": uris_a})
    container.load_nginx_url(field, proxy_url="http://backend:9091", auth_extra={"uris": uris_b})

    assert len(container.start_hooks) - baseline == 1
