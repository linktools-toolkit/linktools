#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Lazy declaration and URL contracts, independent of Docker and templates."""
from types import SimpleNamespace

import pytest

from linktools.cntr import ContainerError, ContainerManager, Integration
from linktools.cntr.ext import Authelia, Flare, Nginx, load_nginx_url
from linktools.cntr.ext import ResolvedSite
from linktools.runtime import lazy_load


class Producer:
    def __init__(self, site, installed=("nginx", "authelia", "safeline"), **config):
        self.name = "app"
        self.config = dict(NGINX_HTTPS_ENABLE=True, NGINX_AUTH_ENABLE=True,
                           NGINX_WAF_ENABLE=True, NGINX_HTTP_PORT=80, NGINX_HTTPS_PORT=443)
        self.config.update(config)
        self.manager = SimpleNamespace(
            integration_snapshot={name: () for name in installed},
            containers={"nginx": SimpleNamespace(sites={})},
        )
        self.manager.containers["nginx"].sites = {(self.name, "web"): ResolvedSite(self, "web", site)}

    def get_config(self, key, **kwargs):
        return self.config[key]

    @property
    def site(self):
        return self.manager.containers["nginx"].sites[(self.name, "web")]


def fail():
    raise AssertionError("unrelated lazy value evaluated")


def test_url_only_resolves_required_values():
    producer = Producer(Nginx.site("app.example.com", proxy=lazy_load(fail),
                                 auth_headers=lazy_load(fail)))
    assert str(load_nginx_url(producer, "web")) == "https://app.example.com"
    assert str(load_nginx_url(producer, "web", "ui", queries={"a": "b"})) == "https://app.example.com/ui?a=b"


@pytest.mark.parametrize("installed,domain", [((), None), (("nginx",), "")])
def test_disabled_sites_do_not_resolve_unrelated_fields(installed, domain):
    domain = lazy_load(fail) if domain is None else domain
    producer = Producer(Nginx.site(domain, proxy=lazy_load(fail), template=lazy_load(fail)), installed=installed)
    assert str(load_nginx_url(producer, "web")) == ""
    assert producer.site.resolve() is producer.site
    with pytest.raises(ContainerError, match="Unknown nginx site"):
        str(load_nginx_url(producer, "missing"))


@pytest.mark.parametrize("field", ["https", "auth", "waf"])
def test_explicit_capability_cannot_silently_downgrade(field):
    producer = Producer(Nginx.site("a.test", proxy="http://app", **{field: True}),
                        **{"NGINX_%s_ENABLE" % field.upper(): False})
    with pytest.raises(ContainerError, match="explicitly requires"):
        getattr(producer.site, field)


@pytest.mark.parametrize("field,provider", [("auth", "authelia"), ("waf", "safeline")])
def test_inherited_capability_requires_provider(field, provider):
    producer = Producer(Nginx.site("a.test", proxy="http://app"), installed=("nginx",))
    with pytest.raises(ContainerError, match=provider):
        getattr(producer.site, field)


def test_http_auth_rejected_and_disabled_auth_fields_unread():
    producer = Producer(Nginx.site("a.test", proxy="http://app", https=False))
    with pytest.raises(ContainerError, match="requires HTTPS"):
        producer.site.auth
    producer = Producer(Nginx.site("a.test", proxy="http://app", auth=False,
                                 auth_rule=lazy_load(fail), auth_headers=lazy_load(fail),
                                 auth_bypass=lazy_load(fail)))
    assert producer.site.auth_rule is None
    assert dict(producer.site.auth_headers) == {}
    assert producer.site.auth_bypass == ()


def test_migrated_http_site_must_disable_inherited_auth_explicitly():
    producer = Producer(Nginx.site("public.example.test", proxy="http://app",
                                  https=False, auth=False))
    assert producer.site.resolve() is producer.site
    assert producer.site.auth is False
    assert producer.site.public_url == "http://public.example.test"


@pytest.mark.parametrize("domain", ["_", "*.test", "a.test b.test", "~^app\\.test$", "a.test\tb.test"])
def test_nonliteral_domain_requires_explicit_url(domain):
    producer = Producer(Nginx.site(domain, proxy="http://app"))
    with pytest.raises(ContainerError, match="explicit public URL"):
        producer.site.public_url


def test_placeholder_domain_skips_navigation_without_oidc_coupling():
    producer = Producer(Nginx.site(
        "_", proxy="http://app", link=Flare.public("App", "app", "Application"),
    ))
    assert producer.site.link.url is None
    assert str(load_nginx_url(producer, "web")) == ""
    assert producer.site.resolve() is producer.site


def test_literal_template_url_is_not_executed():
    producer = Producer(Nginx.site("~^app", proxy="http://app", public_url="https://app:{{port}}"))
    assert producer.site.public_url == "https://app:{{port}}"


def test_auth_rule_preserves_native_fields_and_is_read_only():
    rule = {"policy": "two_factor", "networks": ["10.0.0.0/8"]}
    producer = Producer(Nginx.site("a.test", proxy="http://app", auth_rule=rule))
    assert dict(producer.site.auth_rule) == dict(rule, domain="a.test")
    assert "domain" not in rule
    with pytest.raises(TypeError):
        producer.site.auth_rule["policy"] = "bypass"


def test_identity_encoding_has_no_separator_collisions():
    producer = Producer(Nginx.site("a.test", proxy="http://app"))
    first = ResolvedSite(producer, "a/b", Nginx.site("a.test"))
    second = ResolvedSite(producer, "a_b", Nginx.site("a.test"))
    assert first.file_id != second.file_id
    assert first.identity == ("app", "a/b")


def test_old_proxy_declaration_arguments_are_removed():
    producer = Producer(Nginx.site("a.test", proxy="http://app"))
    with pytest.raises(TypeError):
        load_nginx_url(producer, "web", proxy_url="http://app")
    assert not hasattr(producer, "load_exist_nginx_url")


def manager_with(containers, installed):
    from _harness import builtin_container_type

    manager = object.__new__(ContainerManager)
    if "nginx" in containers:
        nginx = object.__new__(builtin_container_type("100-nginx"))
        nginx._name = "nginx"
        nginx.manager = manager
        containers["nginx"] = nginx
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
            return [Nginx.site(lazy_load(fail))]
    app = Counted()
    manager = manager_with({"app": app, "nginx": SimpleNamespace(name="nginx")}, ["app"])
    app.manager = manager
    assert list(manager.iter_integrations("nginx")) == []
    assert ("app", "web") in manager.containers["nginx"].sites
    assert str(load_nginx_url(app, "web")) == ""
    assert app.calls == 1


def test_explicit_false_does_not_read_unneeded_global_switches():
    producer = Producer(Nginx.site("a.test", proxy="http://app", https=False, auth=False, waf=False))
    del producer.config["NGINX_HTTPS_ENABLE"]
    del producer.config["NGINX_AUTH_ENABLE"]
    del producer.config["NGINX_WAF_ENABLE"]
    assert producer.site.public_url == "http://a.test"
    assert producer.site.resolve() is producer.site


def test_snapshot_freezes_mixed_declarations_once_without_url_resolution() -> None:
    from linktools.cntr.ext import Flare

    class CustomIntegration(Integration):
        consumer = "custom"

        def __init__(self, label: "str | None" = None) -> None:
            self.label = label

    inputs = [
        CustomIntegration("second"),
        Flare.bookmark("Tool", "web", lazy_load(fail), category="tool"),
        Nginx.site(lazy_load(fail), local_id="web"),
        CustomIntegration("first"),
        CustomIntegration(),
    ]
    iterations = []

    def declarations():
        for value in inputs:
            iterations.append(value)
            yield value

    class Producer:
        name = "app"
        calls = 0

        @property
        def integrations(self):
            self.calls += 1
            return declarations()

    app = Producer()
    containers = {"app": app}
    containers.update({name: SimpleNamespace(name=name, integrations=[])
                       for name in ("custom", "flare", "nginx")})
    manager = manager_with(containers, list(containers))
    snapshot = manager.integration_snapshot
    assert snapshot["app"] == tuple(inputs)
    assert iterations == inputs
    inputs.clear()
    assert [item.label for _, item in manager.iter_integrations("custom")] == ["second", "first", None]
    assert list(manager.iter_integrations("flare")) == [(app, snapshot["app"][1])]
    assert [item.local_id for _, item in manager.iter_integrations("nginx")] == ["web"]
    assert manager.integration_snapshot is snapshot
    assert app.calls == 1
    assert len(iterations) == 5
    with pytest.raises(TypeError):
        snapshot["app"] = ()
    with pytest.raises(TypeError):
        snapshot["app"][0] = None


@pytest.mark.parametrize("values", [[], (), iter(())])
def test_empty_integration_sequences_are_valid(values: object) -> None:
    app = SimpleNamespace(name="app", integrations=values)
    manager = manager_with({"app": app, "nginx": SimpleNamespace(integrations=[])}, ["app"])
    assert manager.integration_snapshot["app"] == ()
    assert not manager.containers["nginx"].sites


@pytest.mark.parametrize("declarations", ["text", b"text", 42, None, [object()], {}, {"nginx": []}])
def test_snapshot_rejects_invalid_declaration_shapes_or_non_markers(declarations: object) -> None:
    app = SimpleNamespace(name="app", integrations=declarations)
    manager = manager_with({"app": app, "custom": SimpleNamespace(integrations=[])}, ["app"])
    with pytest.raises(ContainerError, match="Invalid integration"):
        manager.integration_snapshot


@pytest.mark.parametrize("local_id", ["", 0, None])
def test_nginx_rejects_invalid_ids_before_resolving_fields(local_id: "str | int | None") -> None:
    app = SimpleNamespace(name="app", integrations=[Nginx.site(lazy_load(fail), local_id=local_id)])
    manager = manager_with({"app": app, "nginx": SimpleNamespace(integrations=[])}, ["app"])
    with pytest.raises(ContainerError, match="integration ID"):
        manager.containers["nginx"].sites


def test_generic_integrations_have_no_identity_contract() -> None:
    class CustomIntegration(Integration):
        consumer = "custom"

    declarations = [CustomIntegration(), CustomIntegration()]
    flare = Flare.public("App", "web", "Application", lazy_load(fail))
    authelia = Authelia.oidc(lazy_load(fail))
    app = SimpleNamespace(name="app", integrations=declarations)
    custom = SimpleNamespace(name="custom", integrations=())
    declarations.extend((flare, authelia))
    containers = {"app": app, "custom": custom}
    containers.update({name: SimpleNamespace(name=name, integrations=()) for name in ("flare", "authelia")})
    manager = manager_with(containers, list(containers))
    assert manager.integration_snapshot["app"] == tuple(declarations)
    assert list(manager.iter_integrations("custom")) == [(app, item) for item in declarations[:2]]
    assert list(manager.iter_integrations("flare")) == [(app, flare)]
    assert list(manager.iter_integrations("authelia")) == [(app, authelia)]
    assert not hasattr(Integration, "local_id")
    assert not hasattr(Integration, "requires_local_id")
    assert not hasattr(flare, "local_id")
    assert not hasattr(authelia, "local_id")


def test_site_factory_and_resolved_view_use_explicit_field_names() -> None:
    declaration = Nginx.site(
        "~^app", proxy="http://app", public_url="https://public.example.test",
        template_vars={"title": "Application"}, link=Flare.public("App", "web", "Application"),
        default_server=True,
    )
    site = Producer(declaration).site
    assert site.public_url == declaration.public_url == "https://public.example.test"
    assert dict(site.template_vars) == declaration.template_vars == {"title": "Application"}
    assert site.link.url == site.public_url
    assert site.default_server is declaration.default_server is True
    assert declaration.local_id == "web"
    for old in ("url", "vars", "expose", "default"):
        assert not hasattr(declaration, old)
        assert not hasattr(site, old)
        with pytest.raises(TypeError, match="unexpected keyword argument"):
            Nginx.site("app.test", **{old: None})


def test_nginx_rejects_duplicate_ids_before_resolving_fields() -> None:
    app = SimpleNamespace(name="app", integrations=[Nginx.site(lazy_load(fail)), Nginx.site(lazy_load(fail))])
    manager = manager_with({"app": app, "nginx": SimpleNamespace(integrations=[])}, ["app"])
    with pytest.raises(ContainerError, match="Duplicate nginx integration ID 'web' in app"):
        manager.containers["nginx"].sites


def test_site_ids_are_scoped_to_their_producer() -> None:
    app = SimpleNamespace(name="app", integrations=[Nginx.site("a.test"), Nginx.site("api.test", local_id="api")])
    other = SimpleNamespace(name="other", integrations=[Nginx.site("b.test")])
    manager = manager_with({"app": app, "other": other, "nginx": SimpleNamespace(integrations=[])},
                           ["app", "other"])
    assert tuple(manager.containers["nginx"].sites) == (("app", "web"), ("app", "api"), ("other", "web"))


def test_nginx_rejects_declarations_of_another_type() -> None:
    declaration = Integration()
    declaration.consumer = "nginx"
    app = SimpleNamespace(name="app", integrations=[declaration])
    manager = manager_with({"app": app, "nginx": SimpleNamespace(integrations=[])}, ["app"])
    with pytest.raises(ContainerError, match="expected Nginx"):
        manager.containers["nginx"].sites


def test_unknown_consumer_rejected_and_empty_installed_consumer_is_empty() -> None:
    declaration = Integration()
    declaration.consumer = "missing"
    app = SimpleNamespace(name="app", integrations=[declaration])
    manager = manager_with({"app": app}, ["app"])
    with pytest.raises(ContainerError, match="Unknown integration consumer"):
        manager.integration_snapshot
    app.integrations = []
    assert list(manager.iter_integrations("app")) == []
    with pytest.raises(ContainerError, match="Unknown integration consumer"):
        list(manager.iter_integrations("missing"))


@pytest.mark.parametrize("server_name,default", [("_", False), ("app.example.com", True)])
def test_default_server_is_independent_of_domain(server_name: str, default: bool) -> None:
    producer = Producer(Nginx.site(server_name, proxy="http://app", default_server=default))
    assert producer.site.default_server is default
    assert producer.site.server_name == server_name
    if default:
        assert producer.site.public_url == "https://app.example.com"


def test_disabled_default_is_lazy() -> None:
    producer = Producer(Nginx.site("", default_server=lazy_load(fail)))
    assert producer.site.default_server is False
    assert producer.site.resolve() is producer.site
