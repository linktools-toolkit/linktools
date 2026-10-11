#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Lazy declaration and URL contracts, independent of Docker and templates."""
from types import SimpleNamespace

import pytest

from linktools.cntr import ContainerError, ContainerManager, Integration
from linktools.cntr.ext import Authelia, Flare, Nginx, load_nginx_url
from linktools.cntr.ext import ResolvedSite, endpoint
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
                                 auth_headers=lazy_load(fail),
                                 auth_bypass=lazy_load(fail)))
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
        "_", proxy="http://app",
    ))
    link = Flare.public("App", "app", "Application", load_nginx_url(producer, "web"))
    assert link.url is None
    assert str(load_nginx_url(producer, "web")) == ""
    assert producer.site.resolve() is producer.site


def test_literal_template_url_is_not_executed():
    producer = Producer(Nginx.site("~^app", proxy="http://app", public_url="https://app:{{port}}"))
    assert producer.site.public_url == "https://app:{{port}}"


def test_access_rules_are_not_nginx_declaration_or_resolution_fields() -> None:
    producer = Producer(Nginx.site("a.test", proxy="http://app"))
    assert not hasattr(producer.site, "auth_rule")
    assert not hasattr(producer.site._declaration, "auth_rule")
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        Nginx.site("a.test", auth_rule=lazy_load(fail))


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
            return {"web": Nginx.site(lazy_load(fail))}
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
        {"web": Nginx.site(lazy_load(fail))},
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
    assert snapshot["app"] == tuple(
        ("web", item["web"]) if isinstance(item, dict) else (None, item) for item in inputs)
    assert iterations == inputs
    inputs.clear()
    assert [item.label for _, item in manager.iter_integrations("custom")] == ["second", "first", None]
    assert list(manager.iter_integrations("flare")) == [(app, snapshot["app"][1][1])]
    assert list(manager.iter_integrations("nginx")) == [(app, snapshot["app"][2][1])]
    assert tuple(manager.containers["nginx"].sites) == (("app", "web"),)
    assert manager.integration_snapshot is snapshot
    assert app.calls == 1
    assert len(iterations) == 5
    with pytest.raises(TypeError):
        snapshot["app"] = ()
    with pytest.raises(TypeError):
        snapshot["app"][0] = None


@pytest.mark.parametrize("values", [[], (), iter(()), {}])
def test_empty_integration_sequences_are_valid(values: object) -> None:
    app = SimpleNamespace(name="app", integrations=values)
    manager = manager_with({"app": app, "nginx": SimpleNamespace(integrations=[])}, ["app"])
    assert manager.integration_snapshot["app"] == ()
    assert not manager.containers["nginx"].sites


@pytest.mark.parametrize("declarations", [
    "text", b"text", 42, None, [object()], {"nginx": []},
    [iter(())], [("web", Nginx.site("app.test"))], [{"web": [Nginx.site("app.test")]}],
])
def test_snapshot_rejects_invalid_declaration_shapes_or_non_markers(declarations: object) -> None:
    app = SimpleNamespace(name="app", integrations=declarations)
    manager = manager_with({"app": app, "custom": SimpleNamespace(integrations=[])}, ["app"])
    with pytest.raises(ContainerError, match="Invalid integration"):
        manager.integration_snapshot


@pytest.mark.parametrize("key", ["", 0, None, False, ("web",)])
def test_snapshot_rejects_invalid_keys_before_resolving_fields(key: object) -> None:
    app = SimpleNamespace(name="app", integrations={key: Nginx.site(lazy_load(fail))})
    manager = manager_with({"app": app, "nginx": SimpleNamespace(integrations=[])}, ["app"])
    with pytest.raises(ContainerError, match="Invalid integration key in app"):
        manager.integration_snapshot


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
    assert manager.integration_snapshot["app"] == tuple((None, value) for value in declarations)
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
        template_vars={"title": "Application"},
        default_server=True,
    )
    site = Producer(declaration).site
    assert site.public_url == declaration.public_url == "https://public.example.test"
    assert dict(site.template_vars) == declaration.template_vars == {"title": "Application"}
    assert site.default_server is declaration.default_server is True
    assert site.key == "web"
    assert not hasattr(declaration, "key")
    for old in ("url", "vars", "expose", "default", "local_id", "link", "auth_rule"):
        assert not hasattr(declaration, old)
        assert not hasattr(site, old)
        with pytest.raises(TypeError, match="unexpected keyword argument"):
            Nginx.site("app.test", **{old: None})


def test_nginx_rejects_duplicate_keys_before_resolving_fields() -> None:
    app = SimpleNamespace(name="app", integrations=[
        {"web": Nginx.site(lazy_load(fail))}, {"web": Nginx.site(lazy_load(fail))},
    ])
    manager = manager_with({"app": app, "nginx": SimpleNamespace(integrations=[])}, ["app"])
    assert len(manager.integration_snapshot["app"]) == 2
    with pytest.raises(ContainerError, match="Duplicate nginx site key 'web' in app"):
        manager.containers["nginx"].sites


def test_site_keys_are_scoped_to_their_producer() -> None:
    app = SimpleNamespace(name="app", integrations={"web": Nginx.site("a.test"), "api": Nginx.site("api.test")})
    other = SimpleNamespace(name="other", integrations={"web": Nginx.site("b.test")})
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


@pytest.mark.parametrize("wrapper", [lambda value: value, lambda value: [value],
                                      lambda value: iter([value]), lambda value: [([value],)]])
def test_named_mapping_shapes_freeze_keys_without_mutating_declarations(wrapper) -> None:
    site = Nginx.site(lazy_load(fail))
    link = Flare.bookmark("Docs", "web", lazy_load(fail))
    mapping = {"web": site, "docs": link}
    app = SimpleNamespace(name="app", integrations=wrapper(mapping))
    manager = manager_with({"app": app, "nginx": SimpleNamespace(integrations=()),
                            "flare": SimpleNamespace(name="flare", integrations=())}, ["app", "nginx", "flare"])
    snapshot = manager.integration_snapshot
    mapping.clear()
    assert snapshot["app"] == (("web", site), ("docs", link))
    assert tuple(manager.containers["nginx"].sites) == (("app", "web"),)
    assert list(manager.iter_integrations("flare")) == [(app, link)]
    assert not hasattr(site, "key")
    assert not hasattr(site, "local_id")
    assert not hasattr(link, "key")


def test_keys_are_local_to_consumer_and_generic_consumers_may_ignore_duplicates() -> None:
    site = Nginx.site(lazy_load(fail))
    first = Flare.bookmark("First", "web", lazy_load(fail))
    second = Flare.bookmark("Second", "web", lazy_load(fail))
    app = SimpleNamespace(name="app", integrations=[{"web": site}, {"web": first}, {"web": second}])
    manager = manager_with({"app": app, "nginx": SimpleNamespace(integrations=()),
                            "flare": SimpleNamespace(name="flare", integrations=())}, ["app", "nginx", "flare"])
    assert manager.integration_snapshot["app"] == (("web", site), ("web", first), ("web", second))
    assert tuple(manager.containers["nginx"].sites) == (("app", "web"),)
    assert list(manager.iter_integrations("flare")) == [(app, first), (app, second)]


def test_nginx_requires_named_entry_without_resolving_anonymous_site_fields() -> None:
    app = SimpleNamespace(name="app", integrations=[Nginx.site(lazy_load(fail))])
    manager = manager_with({"app": app, "nginx": SimpleNamespace(integrations=())}, ["app"])
    assert manager.integration_snapshot["app"][0][0] is None
    with pytest.raises(ContainerError, match="Invalid nginx site key in app"):
        manager.containers["nginx"].sites


@pytest.mark.parametrize("key", ["", 0, None, False])
def test_named_non_nginx_integrations_validate_key_shape(key: object) -> None:
    app = SimpleNamespace(name="app", integrations=[{key: Flare.bookmark("Docs", "web", lazy_load(fail))}])
    manager = manager_with({"app": app, "flare": SimpleNamespace(name="flare", integrations=())}, ["app"])
    with pytest.raises(ContainerError, match="Invalid integration key in app"):
        manager.integration_snapshot


def test_nested_composition_freezes_order_and_structure_without_resolving_leaves() -> None:
    access = Authelia.access(lazy_load(fail))
    first = Flare.public("First", "web", url=lazy_load(fail))
    second = Flare.bookmark("Second", "web", lazy_load(fail))
    web = Nginx.site(lazy_load(fail))
    nested = [first, ({"web": web}, [access, {"docs": second}])]
    app = SimpleNamespace(name="app", integrations=nested)
    consumers = {name: SimpleNamespace(name=name, integrations=())
                 for name in ("nginx", "flare", "authelia")}
    manager = manager_with(dict(consumers, app=app), ["app"] + list(consumers))
    assert list(Integration.iter_declarations(nested)) == [(None, first), ("web", web), (None, access), ("docs", second)]
    snapshot = manager.integration_snapshot["app"]
    nested[1][1].clear()
    nested.clear()
    assert snapshot == ((None, first), ("web", web), (None, access), ("docs", second))
    assert list(manager.iter_integrations("flare")) == [(app, first), (app, second)]
    assert list(manager.iter_integrations("authelia")) == [(app, access)]
    assert tuple(manager.containers["nginx"].sites) == (("app", "web"),)


@pytest.mark.parametrize("unpack", [False, True])
def test_endpoint_is_a_nested_array_element_or_can_be_expanded(unpack: bool) -> None:
    app = SimpleNamespace(name="app")
    recipe = endpoint(app, domain=lazy_load(fail), extensions=[Authelia.access(lazy_load(fail))])
    other = {"api": Nginx.site(lazy_load(fail))}
    app.integrations = [*recipe, other] if unpack else [recipe, other]
    consumers = {name: SimpleNamespace(name=name, integrations=()) for name in ("nginx", "authelia")}
    manager = manager_with(dict(consumers, app=app), ["app"] + list(consumers))
    assert manager.integration_snapshot["app"] == (("web", recipe[0]["web"]), (None, recipe[1]),
                                                   ("api", other["api"]))
    assert tuple(manager.containers["nginx"].sites) == (("app", "web"), ("app", "api"))


@pytest.mark.parametrize("wrapped", [False, True])
def test_single_endpoint_can_be_returned_directly_or_as_an_array_element(wrapped: bool) -> None:
    app = SimpleNamespace(name="app")
    access = Authelia.access(lazy_load(fail))
    recipe = endpoint(app, domain=lazy_load(fail), extensions=[access])
    app.integrations = [recipe] if wrapped else recipe
    consumers = {name: SimpleNamespace(name=name, integrations=()) for name in ("nginx", "authelia")}
    manager = manager_with(dict(consumers, app=app), ["app"] + list(consumers))
    assert manager.integration_snapshot["app"] == (("web", recipe[0]["web"]), (None, access))
    assert list(manager.iter_integrations("authelia")) == [(app, access)]


def test_multiple_endpoints_in_one_producer_reject_duplicate_web_before_resolution() -> None:
    app = SimpleNamespace(name="app")
    app.integrations = [endpoint(app, domain=lazy_load(fail)), endpoint(app, domain=lazy_load(fail))]
    manager = manager_with({"app": app, "nginx": SimpleNamespace(integrations=())}, ["app"])
    assert len(manager.integration_snapshot["app"]) == 2
    with pytest.raises(ContainerError, match="Duplicate nginx site key 'web' in app"):
        manager.containers["nginx"].sites


def test_endpoint_web_keys_remain_local_to_each_producer() -> None:
    first = SimpleNamespace(name="first")
    second = SimpleNamespace(name="second")
    first.integrations = [endpoint(first, domain=lazy_load(fail))]
    second.integrations = [endpoint(second, domain=lazy_load(fail))]
    manager = manager_with({"first": first, "second": second, "nginx": SimpleNamespace(integrations=())},
                           ["first", "second"])
    assert tuple(manager.containers["nginx"].sites) == (("first", "web"), ("second", "web"))
