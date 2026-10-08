#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Lazy declaration and URL contracts, independent of Docker and templates."""
from types import SimpleNamespace

import pytest

from linktools.cntr import ContainerError, ContainerManager, NginxSite
from linktools.cntr.urls import load_nginx_url
from linktools.cntr._nginx import ResolvedSite
from linktools.runtime import lazy_load


class Producer:
    def __init__(self, site, installed=("nginx", "authelia", "safeline"), **config):
        self.name = "app"
        self.config = dict(NGINX_HTTPS_ENABLE=True, NGINX_AUTH_ENABLE=True,
                           NGINX_WAF_ENABLE=True, NGINX_HTTP_PORT=80, NGINX_HTTPS_PORT=443)
        self.config.update(config)
        self.manager = SimpleNamespace(integration_snapshot={name: {} for name in installed})
        self.manager.nginx_sites = {(self.name, "web"): ResolvedSite(self, "web", site)}

    def get_config(self, key, **kwargs):
        return self.config[key]

    @property
    def site(self):
        return self.manager.nginx_sites[(self.name, "web")]


def fail():
    raise AssertionError("unrelated lazy value evaluated")


def test_url_only_resolves_required_values():
    producer = Producer(NginxSite("app.example.com", proxy=lazy_load(fail),
                                 auth_headers=lazy_load(fail), oidc_redirects=lazy_load(fail)))
    assert str(load_nginx_url(producer, "web")) == "https://app.example.com"
    assert str(load_nginx_url(producer, "web", "ui", queries={"a": "b"})) == "https://app.example.com/ui?a=b"


@pytest.mark.parametrize("installed,domain", [((), None), (("nginx",), "")])
def test_disabled_sites_do_not_resolve_unrelated_fields(installed, domain):
    domain = lazy_load(fail) if domain is None else domain
    producer = Producer(NginxSite(domain, proxy=lazy_load(fail), template=lazy_load(fail),
                                 oidc_redirects=lazy_load(fail)), installed=installed)
    assert str(load_nginx_url(producer, "web")) == ""
    assert producer.site.resolve() is producer.site
    with pytest.raises(ContainerError, match="Unknown nginx site"):
        str(load_nginx_url(producer, "missing"))


@pytest.mark.parametrize("field", ["https", "auth", "waf"])
def test_explicit_capability_cannot_silently_downgrade(field):
    producer = Producer(NginxSite("a.test", proxy="http://app", **{field: True}),
                        **{"NGINX_%s_ENABLE" % field.upper(): False})
    with pytest.raises(ContainerError, match="explicitly requires"):
        getattr(producer.site, field)


@pytest.mark.parametrize("field,provider", [("auth", "authelia"), ("waf", "safeline")])
def test_inherited_capability_requires_provider(field, provider):
    producer = Producer(NginxSite("a.test", proxy="http://app"), installed=("nginx",))
    with pytest.raises(ContainerError, match=provider):
        getattr(producer.site, field)


def test_http_auth_rejected_and_disabled_auth_fields_unread():
    producer = Producer(NginxSite("a.test", proxy="http://app", https=False))
    with pytest.raises(ContainerError, match="requires HTTPS"):
        producer.site.auth
    producer = Producer(NginxSite("a.test", proxy="http://app", auth=False,
                                 auth_rule=lazy_load(fail), auth_headers=lazy_load(fail),
                                 auth_bypass=lazy_load(fail)))
    assert producer.site.auth_rule is None
    assert dict(producer.site.auth_headers) == {}
    assert producer.site.auth_bypass == ()


@pytest.mark.parametrize("domain", ["_", "*.test", "a.test b.test", "~^app\\.test$", "a.test\tb.test"])
def test_nonliteral_domain_requires_explicit_url(domain):
    producer = Producer(NginxSite(domain, proxy="http://app"))
    with pytest.raises(ContainerError, match="explicit public URL"):
        producer.site.url


def test_literal_template_url_is_not_executed_or_relative_oidc_base():
    producer = Producer(NginxSite("~^app", proxy="http://app", url="https://app:{{port}}", oidc_redirects=("/callback",)))
    assert producer.site.url == "https://app:{{port}}"
    with pytest.raises(ContainerError, match="concrete public URL"):
        producer.site.oidc_redirects


def test_oidc_preserves_empty_callback_and_stably_deduplicates():
    producer = Producer(NginxSite("a.test", proxy="http://app", url="https://a.test/base?x=1",
                                 oidc_redirects=lazy_load(lambda: ("", "/callback?q=2", "custom:callback", ""))))
    assert producer.site.oidc_redirects == ("https://a.test/base?x=1", "https://a.test/callback?q=2", "custom:callback")


@pytest.mark.parametrize("value", ["//evil.test/path", "/cb#", "https://a.test/#bad", "callback"])
def test_invalid_oidc_redirects_fail(value):
    producer = Producer(NginxSite("a.test", proxy="http://app", oidc_redirects=(value,)))
    with pytest.raises(ContainerError, match="OIDC"):
        producer.site.oidc_redirects


def test_auth_rule_preserves_native_fields_and_is_read_only():
    rule = {"policy": "two_factor", "networks": ["10.0.0.0/8"]}
    producer = Producer(NginxSite("a.test", proxy="http://app", auth_rule=rule))
    assert dict(producer.site.auth_rule) == dict(rule, domain="a.test")
    assert "domain" not in rule
    with pytest.raises(TypeError):
        producer.site.auth_rule["policy"] = "bypass"


def test_identity_encoding_has_no_separator_collisions():
    producer = Producer(NginxSite("a.test", proxy="http://app"))
    first = ResolvedSite(producer, "a/b", NginxSite("a.test"))
    second = ResolvedSite(producer, "a_b", NginxSite("a.test"))
    assert first.file_id != second.file_id
    assert first.identity == ("app", "a/b")


def test_old_proxy_declaration_arguments_are_removed():
    producer = Producer(NginxSite("a.test", proxy="http://app"))
    with pytest.raises(TypeError):
        load_nginx_url(producer, "web", proxy_url="http://app")
    assert not hasattr(producer, "load_exist_nginx_url")


def manager_with(containers, installed):
    manager = object.__new__(ContainerManager)
    manager.__dict__["containers"] = containers
    manager.__dict__["installed_state"] = SimpleNamespace(get=lambda resolve: [containers[name] for name in installed])
    return manager


def test_snapshot_is_once_and_optional_consumer_is_not_consumed():
    class Counted:
        name = "app"
        calls = 0
        @property
        def integrations(self):
            self.calls += 1
            return {"nginx": {"web": NginxSite("a.test")}}
    app = Counted()
    manager = manager_with({"app": app, "nginx": SimpleNamespace(name="nginx")}, ["app"])
    assert list(manager.iter_integrations("nginx")) == []
    assert ("app", "web") in manager.nginx_sites
    assert app.calls == 1


def test_explicit_false_does_not_read_unneeded_global_switches():
    producer = Producer(NginxSite("a.test", proxy="http://app", https=False, auth=False, waf=False))
    del producer.config["NGINX_HTTPS_ENABLE"]
    del producer.config["NGINX_AUTH_ENABLE"]
    del producer.config["NGINX_WAF_ENABLE"]
    assert producer.site.url == "http://a.test"
    assert producer.site.resolve() is producer.site


def test_snapshot_freezes_named_and_iterable_inputs_once_without_url_resolution():
    from collections import OrderedDict
    from linktools.cntr import ExposeLink, Integration

    class CustomIntegration(Integration):
        pass

    named = OrderedDict((("second", CustomIntegration()), ("first", CustomIntegration())))
    anonymous = [ExposeLink.public("App", "web", "", lazy_load(fail)), CustomIntegration()]
    iterations = []

    def declarations():
        for value in anonymous:
            iterations.append(value)
            yield value

    class Producer:
        name = "app"
        calls = 0

        @property
        def integrations(self):
            self.calls += 1
            return {"named": named, "anonymous": declarations()}

    app = Producer()
    containers = {"app": app, "named": SimpleNamespace(name="named", integrations={}),
                  "anonymous": SimpleNamespace(name="anonymous", integrations={})}
    manager = manager_with(containers, ["app", "named", "anonymous"])
    snapshot = manager.integration_snapshot
    assert list(snapshot["app"]["named"]) == ["second", "first"]
    assert snapshot["app"]["anonymous"] == tuple(anonymous)
    assert iterations == anonymous
    named.clear()
    anonymous.clear()
    assert [key for _, key, _ in manager.iter_integrations("named")] == ["second", "first"]
    assert [key for _, key, _ in manager.iter_integrations("anonymous")] == [None, None]
    assert manager.integration_snapshot is snapshot
    assert app.calls == 1
    assert len(iterations) == 2
    for target, key in ((snapshot, "app"), (snapshot["app"], "named"),
                        (snapshot["app"]["named"], "second")):
        with pytest.raises(TypeError):
            target[key] = {}
    with pytest.raises(TypeError):
        snapshot["app"]["anonymous"][0] = None


@pytest.mark.parametrize("values", [[], (), iter(())])
def test_nginx_rejects_anonymous_declarations_even_when_empty(values):
    app = SimpleNamespace(name="app", integrations={"nginx": values})
    manager = manager_with({"app": app, "nginx": SimpleNamespace(integrations={})}, ["app"])
    with pytest.raises(ContainerError, match="require a named mapping"):
        manager.integration_snapshot


@pytest.mark.parametrize("declarations", ["text", b"text", 42, None, [object()], {"id": object()}])
def test_snapshot_rejects_invalid_declaration_shapes_or_non_markers(declarations):
    app = SimpleNamespace(name="app", integrations={"custom": declarations})
    manager = manager_with({"app": app, "custom": SimpleNamespace(integrations={})}, ["app"])
    with pytest.raises(ContainerError, match="Invalid custom integration"):
        manager.integration_snapshot


@pytest.mark.parametrize("local_id", ["", 0, None])
def test_snapshot_rejects_invalid_named_ids(local_id):
    from linktools.cntr import Integration

    app = SimpleNamespace(name="app", integrations={"custom": {local_id: Integration()}})
    manager = manager_with({"app": app, "custom": SimpleNamespace(integrations={})}, ["app"])
    with pytest.raises(ContainerError, match="integration ID"):
        manager.integration_snapshot


def test_unknown_consumer_rejected_and_empty_installed_consumer_is_empty():
    app = SimpleNamespace(name="app", integrations={"missing": []})
    manager = manager_with({"app": app}, ["app"])
    with pytest.raises(ContainerError, match="Unknown integration consumer"):
        manager.integration_snapshot
    app.integrations = {}
    assert list(manager.iter_integrations("app")) == []
    with pytest.raises(ContainerError, match="Unknown integration consumer"):
        list(manager.iter_integrations("missing"))
