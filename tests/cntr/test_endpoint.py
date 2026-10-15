#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""The endpoint recipe returns ordinary independent lazy declarations."""
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest

from linktools.cntr import ContainerError
from linktools.cntr.ext import Authelia, Flare, Integration, Nginx, ResolvedSite, endpoint
from linktools.cntr.ext import load_nginx_url, load_port_url
from linktools.core import Config, ConfigSchema
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
    oidc = Authelia.oidc((load_nginx_url(producer, None, "callback"),))
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
    assert public._url is MISSING and direct._url is not MISSING
    assert access_copy is not access and access_copy.site is site
    assert access.site is None
    assert oidc_copy is not oidc and oidc.site is MISSING
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
    producer.env_config = Config(None, ConfigSchema(), [])
    producer.env_config.get = producer.get_config
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


def test_endpoint_independent_oidc_does_not_resolve_site_or_public_navigation() -> None:
    declarations = endpoint(object(), domain=lazy_load(fail), proxy=lazy_load(fail), extensions=[
        Flare.public("App", "web"), Authelia.oidc(("https://fixed.test/callback",)),
    ])
    assert declarations[-1].redirect_uris == ("https://fixed.test/callback",)


@pytest.mark.parametrize("key", ["web", "portal", None, "", 0, False])
def test_endpoint_rejects_positional_site_keys_without_state_access(key: object) -> None:
    with pytest.raises(TypeError):
        endpoint(object(), key, domain=lazy_load(fail), proxy=lazy_load(fail))


@pytest.mark.parametrize("field", ["key", "name", "icon", "desc", "path", "direct_port", "direct_https",
                                   "oidc_paths", "oidc_enabled", "auth_rule", "public_url"])
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
    producer = SimpleNamespace(name="app", get_config=lambda key, **kwargs: 8080)
    producer.env_config = Config(None, ConfigSchema(), [])
    producer.env_config.get = producer.get_config
    dashboard = Flare.category("dashboard", apps=True)
    explicit = lazy_load(fail)
    originals = [Flare.public("App", "web"), dashboard("Dashboard", "web"),
                 Flare.public("None", "web", url=None), Flare.public("Empty", "web", url=""),
                 Flare.public("Explicit", "web", url=explicit), Flare.container("Internal", "web")]
    declarations = endpoint(producer, domain="app.test",
                            proxy="http://app", https=False, auth=False, waf=False,
                            cert_domains=("app.test",), default_server=True,
                            extensions=originals)
    site = ResolvedSite(producer, declarations[0], 0)
    producer.manager = SimpleNamespace(integration_snapshot={"nginx": ()},
                                      containers={"nginx": SimpleNamespace(sites={declarations[0]: site})})
    assert site.public_url == "http://app.test:8080"
    assert site.cert_domains == ("app.test",) and site.default_server is True
    assert site.https is site.auth is site.waf is False
    assert [value.url for value in declarations[1:5]] == ["http://app.test:8080"] * 2 + [None, None]
    assert originals[4]._url is explicit
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
    producer.env_config = Config(None, ConfigSchema(), [])
    producer.env_config.get = producer.get_config
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




def test_endpoint_rejects_extension_factories_but_accepts_callable_declaration():
    def factory(url):
        raise AssertionError("extension factories must not run")

    with pytest.raises(ValueError, match="Invalid integrations"):
        endpoint(object(), domain="app.test", extensions=factory)

    class CallableIntegration(Integration):
        consumer = "custom"

        def __call__(self):
            raise AssertionError("a declaration is not a factory")

    leaf = CallableIntegration()
    assert endpoint(object(), domain="app.test", extensions=leaf)[1] is leaf


@pytest.mark.parametrize("reference,expected", [
    ("{url}", "https://app.test"), ("/", "https://app.test/"),
    ("/callback?x=1", "https://app.test/callback?x=1"),
    ("/ui/#/setup?a=b", "https://app.test/ui/#/setup?a=b"),
    ("//other.test/path", "https://other.test/path"),
    ("https://other.test/exact/?x=1#fragment", "https://other.test/exact/?x=1#fragment"),
    ("https://other.test/{{port}}", "https://other.test/{{port}}"),
    (None, None), ("", None),
])
def test_endpoint_resolves_only_declared_url_fields(reference, expected):
    producer = SimpleNamespace(name="app", get_config=lambda key, **kw: True if key.endswith("ENABLE") else 443)
    producer.env_config = Config(None, ConfigSchema(), [])
    producer.env_config.get = producer.get_config
    original = Flare.public("literal {url}", "web", "{{metadata}}", reference)
    declarations = endpoint(producer, domain="app.test", extensions=[original])
    producer.manager = SimpleNamespace(integration_snapshot={"nginx": ()}, containers={"nginx": SimpleNamespace(sites={
        declarations[0]: ResolvedSite(producer, declarations[0], 0),
    })})
    assert declarations[1].url == expected
    assert declarations[1].name == "literal {url}" and declarations[1].desc == "{{metadata}}"
    assert original._url is reference


@pytest.mark.parametrize("active", [True, False])
@pytest.mark.parametrize("auth", [True, False, None])
def test_declarative_urls_are_lazy_and_oidc_does_not_follow_auth(active, auth):
    from linktools import utils
    from urllib.parse import urlsplit
    reads = []
    producer = SimpleNamespace(name="app", get_config=lambda key, **kw: True if key.endswith("ENABLE") else 443)
    producer.env_config = Config(None, ConfigSchema(), [])
    producer.env_config.get = producer.get_config
    relative = lazy_load(utils.join_url, "/ui/metacubexd/#/setup", queries={
        "secret": lazy_load(lambda: reads.append("query") or "public-fixture-value"),
    })
    declarations = endpoint(producer, domain="app.test" if active else "", auth=auth, extensions=[
        Flare.public("Mihomo", "server", url=relative),
        Authelia.oidc(("{url}", "/callback", "http://local.test:9000")),
    ])
    assert reads == []
    producer.manager = SimpleNamespace(integration_snapshot={"nginx": ()}, containers={"nginx": SimpleNamespace(sites={
        declarations[0]: ResolvedSite(producer, declarations[0], 0),
    })})
    actual = declarations[1].url
    if active:
        parsed = urlsplit(actual)
        assert parsed.query == "" and parsed.fragment == "/setup?secret=public-fixture-value"
    else:
        assert actual is None
    assert reads == ["query"]
    assert declarations[2].redirect_uris == (("https://app.test", "https://app.test/callback", "http://local.test:9000")
                                            if active else ("http://local.test:9000",))


def test_rebound_oidc_copies_cached_results_and_preserves_none_validation():
    original = Authelia.oidc(("{url}", "/callback", "https://absolute.test/"))
    first = original.with_base_url("https://one.test")
    assert first.redirect_uris == ("https://one.test", "https://one.test/callback", "https://absolute.test/")
    second = first.with_base_url("https://two.test")
    assert second.redirect_uris == ("https://two.test", "https://two.test/callback", "https://absolute.test/")
    assert first.redirect_uris[0] == "https://one.test"
    for value in (None, "//other.test/callback", " //other.test/callback", "\t//other.test/callback"):
        with pytest.raises(ContainerError, match="Authelia redirect URI"):
            Authelia.oidc((value,)).with_base_url("https://app.test").redirect_uris
