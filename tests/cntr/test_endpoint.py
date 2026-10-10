#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""The endpoint recipe returns ordinary independent lazy declarations."""
from types import SimpleNamespace

import pytest

from linktools.cntr import ContainerError
from linktools.cntr.ext import Authelia, Flare, Nginx, ResolvedSite, endpoint
from linktools.runtime import Proxy, lazy_load


def fail() -> None:
    raise AssertionError("endpoint construction must not resolve values or register state")


def test_endpoint_constructs_plain_leaves_without_accessing_producer_state() -> None:
    class Producer:
        def __getattr__(self, name: str) -> None:
            return fail()

    domain = lazy_load(fail)
    proxy = lazy_load(fail)
    auth = lazy_load(fail)
    bypass = (r"^/assets/",)
    declarations = endpoint(
        Producer(), "web", name="App", icon="web", desc="Application", domain=domain,
        proxy=proxy, auth=auth, auth_bypass=bypass, waf_bypass=bypass,
        path="ui", direct_port="APP_PORT", oidc_paths=("callback", "return"),
        oidc_enabled=lazy_load(fail),
    )
    assert isinstance(declarations, tuple)
    assert len(declarations) == 4
    named, public, direct, oidc = declarations
    assert isinstance(named, dict) and tuple(named) == ("web",)
    site = named["web"]
    assert isinstance(site, Nginx)
    assert site.server_name is domain and site.proxy is proxy and site.auth is auth
    assert site.auth_bypass is bypass and site.waf_bypass is bypass
    assert not hasattr(site, "key") and not hasattr(site, "link")
    assert isinstance(public, Flare) and isinstance(direct, Flare)
    assert public.display_category.apps is True
    assert direct.display_category is Flare.category("container")
    assert (public.name, public.icon, public.desc) == ("App", "web", "Application")
    assert (direct.name, direct.icon, direct.desc) == ("App", "web", "Application")
    assert isinstance(public._url, Proxy) and isinstance(direct._url, Proxy)
    assert isinstance(oidc, Authelia)
    assert all(isinstance(value, Proxy) for value in oidc._redirect_uris)


@pytest.mark.parametrize("direct,callbacks,count", [
    (None, (), 2), ("APP_PORT", (), 3), (None, ("callback",), 3),
    ("APP_PORT", ("callback",), 4),
])
def test_endpoint_optional_leaves_are_explicit(direct: "str | None", callbacks: "tuple[str, ...]",
                                             count: int) -> None:
    declarations = endpoint(object(), "web", name="App", icon="web", desc="Application",
                            domain="app.test", proxy="http://app", direct_port=direct,
                            oidc_paths=callbacks)
    assert len(declarations) == count
    assert sum(isinstance(value, Flare) for value in declarations) == (2 if direct else 1)
    assert sum(isinstance(value, Authelia) for value in declarations) == bool(callbacks)


def test_endpoint_resolves_public_direct_and_oidc_paths_independently() -> None:
    reads = []
    values = {"NGINX_HTTPS_ENABLE": True, "NGINX_HTTPS_PORT": 9443, "APP_PORT": 8080}

    def config(key: str, **kwargs: object) -> int:
        reads.append(key)
        return values[key]

    producer = SimpleNamespace(name="app", get_config=config, host="host.test")
    declarations = endpoint(producer, "portal", name="App", icon="web", desc="Application",
                            domain="app.test", proxy="http://app:8080", path="ui",
                            direct_port="APP_PORT", oidc_paths=("callback", "callback", "return"))
    site = ResolvedSite(producer, "portal", declarations[0]["portal"])
    producer.manager = SimpleNamespace(integration_snapshot={"nginx": ()},
                                      containers={"nginx": SimpleNamespace(sites={("app", "portal"): site})})
    assert reads == []
    assert declarations[2].url == "http://host.test:8080/ui"
    assert reads == ["APP_PORT"]
    assert declarations[1].url == "https://app.test:9443/ui"
    assert declarations[3].redirect_uris == ("https://app.test:9443/callback", "https://app.test:9443/return")
    assert reads == ["APP_PORT", "NGINX_HTTPS_ENABLE"] + ["NGINX_HTTPS_PORT"] * 4
    assert declarations[0]["portal"] is site._declaration


def test_endpoint_disabled_oidc_does_not_resolve_site_or_public_navigation() -> None:
    declarations = endpoint(object(), "web", name="App", icon="web", desc="Application",
                            domain=lazy_load(fail), proxy=lazy_load(fail),
                            oidc_paths=("callback",), oidc_enabled=False)
    assert declarations[-1].redirect_uris == ()


@pytest.mark.parametrize("key", [None, "", 0, False])
def test_endpoint_rejects_invalid_reference_keys_without_state_access(key: object) -> None:
    with pytest.raises(ContainerError, match="nonempty string"):
        endpoint(object(), key, name="App", icon="web", desc="Application",
                 domain="app.test", proxy="http://app")


def test_endpoint_producers_keep_their_own_named_entries() -> None:
    first = endpoint(object(), "web", name="One", icon="web", desc="One",
                     domain="one.test", proxy="http://one")
    second = endpoint(object(), "web", name="Two", icon="web", desc="Two",
                      domain="two.test", proxy="http://two")
    assert first is not second and first[0] is not second[0]
    assert first[0]["web"] is not second[0]["web"]
    assert first[0]["web"].server_name == "one.test"
    assert second[0]["web"].server_name == "two.test"
