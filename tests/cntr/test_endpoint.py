#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""The endpoint recipe returns ordinary independent lazy declarations."""
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest

from linktools.cntr import ContainerError
from linktools.cntr.ext import Authelia, Flare, Integration, Nginx, ResolvedSite, endpoint
from linktools.cntr.ext import load_nginx_url, load_port_url
from linktools.runtime import Proxy, lazy_load
from linktools.types import MISSING

if TYPE_CHECKING:
    from typing import Iterator


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
    headers = lazy_load(fail)
    variables = lazy_load(fail)
    template = lazy_load(fail)
    producer = Producer()
    public, direct = Flare.links(
        Flare.public(), Flare.container(url=load_port_url(producer, "APP_PORT", https=False)),
        name="App", icon="web", desc="Application",
    )
    oidc = Authelia.oidc((load_nginx_url(producer, None, "callback"),), enabled=lazy_load(fail))
    access = Authelia.access(lazy_load(fail))
    declarations = endpoint(
        producer, domain=domain, proxy=proxy, auth=auth, auth_bypass=bypass,
        waf_bypass=bypass, auth_headers=headers, template_vars=variables, template=template,
        extensions=((public, direct), [access, oidc]),
    )
    assert type(declarations) is tuple
    assert len(declarations) == 5
    site, public_copy, direct_copy, access_copy, oidc_copy = declarations
    assert isinstance(site, Nginx)
    assert site.server_name is domain and site.proxy is proxy and site.auth is auth
    assert site.auth_bypass is bypass and site.waf_bypass is bypass
    assert site.auth_headers is headers and site.template_vars is variables and site.template is template
    assert not hasattr(site, "key") and not hasattr(site, "link") and not hasattr(site, "auth_rule")
    assert public_copy is not public and direct_copy is not direct
    assert public_copy.display_category.apps is True
    assert direct_copy.display_category is Flare.category("container")
    assert (public_copy.name, public_copy.icon, public_copy.desc) == ("App", "web", "Application")
    assert (direct_copy.name, direct_copy.icon, direct_copy.desc) == ("App", "web", "Application")
    assert isinstance(public_copy._url, Proxy) and isinstance(direct_copy._url, Proxy)
    assert public._url is MISSING and direct_copy._url is direct._url
    assert access_copy is not access and access_copy.site is site
    assert access.site is None
    assert oidc_copy is oidc and oidc.site is MISSING
    assert all(isinstance(value, Proxy) for value in oidc._redirect_uris)


@pytest.mark.parametrize("extensions", [(), []])
def test_endpoint_has_no_implicit_navigation_or_authelia_declarations(extensions: object) -> None:
    declarations = endpoint(object(), domain="mcp.test", proxy="http://mcp", extensions=extensions)
    assert len(declarations) == 1
    assert isinstance(declarations[0], Nginx)
    assert declarations[0].auth is None
    assert declarations[0].https is None
    assert declarations[0].waf is None


@pytest.mark.parametrize("direct,callbacks,count", [
    (False, False, 2), (True, False, 3), (False, True, 3), (True, True, 4),
])
def test_endpoint_optional_leaves_are_explicit(direct: bool, callbacks: bool, count: int) -> None:
    extensions = [Flare.public("App", "web", "Application")]
    if direct:
        extensions.append(Flare.container("App", "web", "http://host.test:8080"))
    if callbacks:
        extensions.append(Authelia.oidc(("https://app.test/callback",)))
    declarations = endpoint(object(), domain="app.test", proxy="http://app", extensions=extensions)
    assert len(declarations) == count
    assert sum(isinstance(value, Flare) for value in declarations) == (2 if direct else 1)
    assert sum(isinstance(value, Authelia) for value in declarations) == callbacks


def test_endpoint_resolves_public_direct_and_oidc_paths_independently() -> None:
    reads = []
    values = {"NGINX_HTTPS_ENABLE": True, "NGINX_HTTPS_PORT": 9443, "APP_PORT": 8080}

    def config(key: str, **kwargs: object) -> int:
        reads.append(key)
        return values[key]

    producer = SimpleNamespace(name="app", get_config=config, host="host.test")
    declarations = endpoint(producer, domain="app.test", proxy="http://app:8080", extensions=[
        Flare.public("App", "web", "Application", load_nginx_url(producer, None, "ui")),
        Flare.container("App", "web", load_port_url(producer, "APP_PORT", "ui", https=False)),
        Authelia.oidc(tuple(load_nginx_url(producer, None, path)
                            for path in ("callback", "callback", "return"))),
    ])
    site = ResolvedSite(producer, declarations[0], 0)
    producer.manager = SimpleNamespace(integration_snapshot={"nginx": ()},
                                      containers={"nginx": SimpleNamespace(sites={declarations[0]: site})})
    assert reads == []
    assert declarations[2].url == "http://host.test:8080/ui"
    assert reads == ["APP_PORT"]
    assert declarations[1].url == "https://app.test:9443/ui"
    assert declarations[3].redirect_uris == ("https://app.test:9443/callback", "https://app.test:9443/return")
    assert reads == ["APP_PORT", "NGINX_HTTPS_ENABLE"] + ["NGINX_HTTPS_PORT"] * 4
    assert declarations[0] is site._declaration


def test_endpoint_disabled_oidc_does_not_resolve_site_or_public_navigation() -> None:
    declarations = endpoint(object(), domain=lazy_load(fail), proxy=lazy_load(fail), extensions=[
        Flare.public("App", "web"), Authelia.oidc(lazy_load(fail), enabled=False),
    ])
    assert declarations[-1].redirect_uris == ()


@pytest.mark.parametrize("key", ["web", "portal", None, "", 0, False])
def test_endpoint_rejects_positional_site_keys_without_state_access(key: object) -> None:
    with pytest.raises(TypeError):
        endpoint(object(), key, domain=lazy_load(fail), proxy=lazy_load(fail))


@pytest.mark.parametrize("field", ["key", "name", "icon", "desc", "path", "direct_port", "direct_https",
                                   "oidc_paths", "oidc_enabled", "auth_rule"])
def test_endpoint_has_no_display_callback_or_acl_shortcuts(field: str) -> None:
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        endpoint(object(), domain=lazy_load(fail), **{field: lazy_load(fail)})


def test_endpoint_producers_keep_their_own_declaration_objects() -> None:
    public = Flare.public("App", "web")
    first = endpoint(object(), domain="one.test", proxy="http://one", extensions=[public])
    second = endpoint(object(), domain="two.test", proxy="http://two", extensions=[public])
    assert first is not second and first[0] is not second[0]
    assert first[0].server_name == "one.test"
    assert second[0].server_name == "two.test"
    assert first[1] is not second[1] and public._url is MISSING


def test_endpoint_nested_extensions_keep_generic_leaves_and_order() -> None:
    custom = Integration()
    custom.consumer = "custom"
    access = Authelia.access(policy="one_factor")
    api = Nginx.site(lazy_load(fail))
    application = Flare.public("App", "web")
    declarations = endpoint(object(), domain=lazy_load(fail), extensions=[
        (custom, [api, (access,)]), application,
    ])
    assert isinstance(declarations[0], Nginx)
    assert declarations[1] is custom
    assert declarations[2] is api
    assert declarations[3] is not access and declarations[3].site is declarations[0]
    assert access.site is None
    assert declarations[4] is not application
    assert isinstance(declarations[4]._url, Proxy)
    assert application._url is MISSING


def test_endpoint_extension_generator_is_consumed_once_at_the_outer_boundary() -> None:
    seen = []
    access = Authelia.access(lazy_load(fail))

    def extensions() -> "Iterator[object]":
        seen.append("start")
        yield [access]
        seen.append("end")

    declarations = endpoint(object(), domain=lazy_load(fail), extensions=extensions())
    assert declarations[1] is not access and declarations[1].site is declarations[0]
    assert access.site is None
    assert seen == ["start", "end"]


@pytest.mark.parametrize("extensions", ["text", b"text", None, [object()], {}, {"web": Nginx.site(lazy_load(fail))},
                                        [{"navigation": Flare.public("App", "web")}], {"web": []},
                                        [iter(())], [("web", Nginx.site("app.test"))]])
def test_endpoint_extensions_use_the_common_declaration_shape_validation(extensions: object) -> None:
    with pytest.raises(ValueError, match="Invalid integration"):
        endpoint(object(), domain=lazy_load(fail), extensions=extensions)


def test_endpoint_inherits_only_omitted_application_urls_and_preserves_site_fields() -> None:
    producer = SimpleNamespace(name="app", get_config=fail)
    dashboard = Flare.category("dashboard", apps=True)
    explicit = lazy_load(fail)
    originals = [Flare.public("App", "web"), dashboard("Dashboard", "web"),
                 Flare.public("None", "web", url=None), Flare.public("Empty", "web", url=""),
                 Flare.public("Explicit", "web", url=explicit), Flare.container("Internal", "web")]
    declarations = endpoint(producer, domain="~^app", public_url="https://public.test/base",
                            proxy="http://app", https=False, auth=False, waf=False,
                            cert_domains=("app.test",), default_server=True,
                            extensions=originals)
    site = ResolvedSite(producer, declarations[0], 0)
    producer.manager = SimpleNamespace(integration_snapshot={"nginx": ()},
                                      containers={"nginx": SimpleNamespace(sites={declarations[0]: site})})
    assert site.public_url == "https://public.test/base"
    assert site.cert_domains == ("app.test",) and site.default_server is True
    assert site.https is site.auth is site.waf is False
    assert [value.url for value in declarations[1:5]] == ["https://public.test/base"] * 2 + [None, None]
    assert declarations[5]._url is explicit
    assert declarations[6]._url is MISSING
    assert originals[0]._url is originals[1]._url is MISSING


def test_endpoint_preserves_explicit_access_references_without_mutating_them() -> None:
    site = Nginx.site(lazy_load(fail))
    access = Authelia.access(site=site, policy="deny")
    declarations = endpoint(object(), domain=lazy_load(fail), extensions=[access])
    assert declarations[1] is access
    assert access.site is site


def test_endpoint_binds_reused_access_and_public_links_to_each_actual_site() -> None:
    values = {"NGINX_HTTPS_ENABLE": True, "NGINX_HTTPS_PORT": 443}
    producer = SimpleNamespace(name="app", get_config=lambda key, **kwargs: values[key])
    access = Authelia.access(policy="one_factor")
    link = Flare.public("App", "web")
    first = endpoint(producer, domain="one.test", extensions=[access, link])
    second = endpoint(producer, domain="two.test", extensions=[access, link])
    producer.manager = SimpleNamespace(
        integration_snapshot={"nginx": ()},
        containers={"nginx": SimpleNamespace(sites={
            first[0]: ResolvedSite(producer, first[0], 0),
            second[0]: ResolvedSite(producer, second[0], 1),
        })},
    )
    assert first[1].site is first[0] and second[1].site is second[0]
    assert access.site is None and link._url is MISSING
    assert first[2].url == "https://one.test"
    assert second[2].url == "https://two.test"
    assert str(load_nginx_url(producer, first)) == "https://one.test"
    assert str(load_nginx_url(producer, second)) == "https://two.test"
    with pytest.raises(ContainerError, match="Expected one declared nginx site"):
        str(load_nginx_url(producer))


@pytest.mark.parametrize("active,auth_enabled", [(True, True), (True, False), (False, True), (False, False)])
def test_extension_factory_binds_lazy_url_and_auth_without_early_reads(active, auth_enabled):
    from urllib.parse import urlsplit

    reads = []
    calls = []
    values = {"NGINX_HTTPS_ENABLE": True, "NGINX_HTTPS_PORT": 9443,
              "NGINX_AUTH_ENABLE": auth_enabled}
    producer = SimpleNamespace(name="app", get_config=lambda key, **kwargs: values[key])

    def query_value():
        reads.append("query")
        return "public-fixture-value"

    def extensions(url, auth):
        calls.append("factory")
        return [
            Flare.links(
                Flare.public(url=url("ui", "metacubexd", "#", "setup", queries={"secret": lazy_load(query_value)})),
                name="Mihomo", icon="server",
            ),
            Authelia.oidc((url(), "http://10.10.10.1:9000"), enabled=auth),
        ]

    declarations = endpoint(producer, domain="app.test" if active else "", extensions=extensions)
    assert calls == ["factory"] and reads == []
    producer.manager = SimpleNamespace(
        integration_snapshot={"nginx": (), "authelia": ()},
        containers={"nginx": SimpleNamespace(sites={declarations[0]: ResolvedSite(producer, declarations[0], 0)})},
    )
    url = declarations[1].url
    if active:
        parsed = urlsplit(url)
        assert parsed.path == "/ui/metacubexd/" and parsed.query == ""
        assert parsed.fragment == "/setup?secret=public-fixture-value"
        assert reads == ["query"]
    else:
        assert url is None and reads == []
    assert declarations[2].redirect_uris == (
        ("https://app.test:9443", "http://10.10.10.1:9000") if active and auth_enabled else ()
    )
    assert calls == ["factory"]


def test_extension_factory_reuse_and_generator_keep_original_site_bindings():
    calls = []
    producer = SimpleNamespace(name="app", get_config=fail)

    def factory(url, auth):
        calls.append("start")
        yield Flare.public("App", "web", url=url())
        calls.append("end")

    first = endpoint(producer, domain="one.test", public_url="https://one.test/exact/", extensions=factory)
    second = endpoint(producer, domain="two.test", public_url="https://two.test/other?x=1", extensions=factory)
    producer.manager = SimpleNamespace(integration_snapshot={"nginx": ()}, containers={"nginx": SimpleNamespace(sites={
        first[0]: ResolvedSite(producer, first[0], 0), second[0]: ResolvedSite(producer, second[0], 1),
    })})
    assert first[1].url == "https://one.test/exact/"
    assert second[1].url == "https://two.test/other?x=1"
    assert calls == ["start", "end", "start", "end"]


def test_extension_factory_does_not_reinterpret_existing_lazy_declarations_or_none_enable():
    declarations = endpoint(object(), domain="app.test", extensions=lazy_load(lambda: [
        Flare.public("App", "web", url=None),
        Flare.public("Empty", "web", url=""),
        Authelia.oidc((lazy_load(fail),), enabled=None),
    ]))
    assert declarations[1].url is None and declarations[2].url is None
    assert declarations[3].redirect_uris == ()


def test_extension_factory_errors_propagate_without_retry():
    calls = []

    def extensions(url, auth):
        calls.append(1)
        raise ValueError("broken extension factory")

    with pytest.raises(ValueError, match="broken extension factory"):
        endpoint(object(), domain="app.test", extensions=extensions)
    assert calls == [1]


def test_extension_factory_uninstalled_nginx_does_not_read_domain_queries_or_auth():
    producer = SimpleNamespace(name="app", get_config=fail)
    declarations = endpoint(producer, domain=lazy_load(fail), extensions=lambda url, auth: [
        Flare.public("App", "web", url=url("ui", queries={"secret": lazy_load(fail)})),
        Authelia.oidc((url(), lazy_load(fail)), enabled=auth),
    ])
    producer.manager = SimpleNamespace(integration_snapshot={}, containers={"nginx": SimpleNamespace(sites={
        declarations[0]: ResolvedSite(producer, declarations[0], 0),
    })})
    assert declarations[1].url is None
    assert declarations[2].redirect_uris == ()


def test_extension_factory_preserves_callable_integration_leaf():
    class CallableIntegration(Integration):
        consumer = "custom"

        def __call__(self, *args):
            raise AssertionError("a declaration is not an extension factory")

    leaf = CallableIntegration()
    assert endpoint(object(), domain="app.test", extensions=leaf)[1] is leaf
