#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Config-dependent integration fields resolve only at their consumption boundary."""
from collections import UserDict
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest

from linktools.cntr import ContainerError
from linktools.cntr.ext import Nginx, ResolvedSite, endpoint, load_config_url, load_nginx_url, load_port_url
from linktools.core import Config, ConfigField, ConfigFormat, ConfigLazy, ConfigRef, ConfigSchema, RuntimeOverrideSource
from linktools.runtime import Proxy, lazy_load
from linktools.types import MISSING

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Any
    from linktools.cntr import BaseContainer, ContainerManager


def make_config(**values: "Any") -> Config:
    source = RuntimeOverrideSource()
    for key, value in values.items():
        source.set(key, value)
    return Config(None, ConfigSchema(), [source])


def make_site(config: Config, declaration: Nginx, installed: "tuple[str, ...]" = ("nginx",)) -> ResolvedSite:
    producer = SimpleNamespace(name="app", env_config=config, get_config=config.get)
    site = ResolvedSite(producer, declaration, 0)
    producer.manager = SimpleNamespace(
        integration_snapshot={name: () for name in installed},
        containers={"nginx": SimpleNamespace(sites={declaration: site})},
    )
    return site


def fail(config: Config) -> None:
    raise AssertionError("unused configuration was resolved")


def test_config_values_site_uses_producer_config_without_binding_the_declaration() -> None:
    domain = ConfigRef("DOMAIN")
    declaration = Nginx.site(domain, proxy=ConfigFormat("http://{host}:8080", host=ConfigRef("HOST")),
                             https=False, auth=False, waf=False,
                             cert_domains=(ConfigFormat("*.{domain}", domain=domain),))
    first = make_site(make_config(DOMAIN="one.test", HOST="one", NGINX_HTTP_PORT=80), declaration)
    second = make_site(make_config(DOMAIN="two.test", HOST="two", NGINX_HTTP_PORT=8080), declaration)
    assert first.resolve().public_url == "http://one.test"
    assert second.resolve().public_url == "http://two.test:8080"
    assert first.proxy == "http://one:8080" and second.proxy == "http://two:8080"
    assert first.cert_domains == ("*.one.test",) and second.cert_domains == ("*.two.test",)
    assert declaration.server_name is domain
    first.producer.env_config.set("DOMAIN", "changed.test")
    assert first.server_name == "one.test"
    assert make_site(first.producer.env_config, declaration).server_name == "changed.test"


@pytest.mark.parametrize("installed,domain", [((), ConfigLazy(fail)), (("nginx",), ConfigRef("DOMAIN"))])
def test_config_values_disabled_site_short_circuits_all_other_fields(installed: tuple, domain: object) -> None:
    unused = ConfigLazy(fail)
    declaration = Nginx.site(domain, proxy=unused, template=unused, default_server=unused,
                             https=unused, auth=unused, waf=unused, cert_domains=unused,
                             auth_headers=unused, auth_bypass=unused, waf_bypass=unused,
                             template_vars={"unused": unused})
    site = make_site(make_config(DOMAIN=""), declaration, installed)
    assert site.resolve() is site
    assert site.public_url == "" and not site.default_server
    assert site.proxy is site.template is None
    assert site.cert_domains == ()
    assert dict(site.template_vars) == {}


def test_config_values_disabled_auth_waf_do_not_read_globals_or_fields() -> None:
    config = make_config(ENABLE=False)
    declaration = Nginx.site("app.test", proxy="http://app", https=False,
                             auth=ConfigRef("ENABLE"), waf=ConfigRef("ENABLE"),
                             auth_headers=ConfigLazy(fail), auth_bypass=ConfigLazy(fail),
                             waf_bypass=ConfigLazy(fail))
    site = make_site(config, declaration)
    assert site.resolve() is site
    assert site.auth is site.waf is False
    assert dict(site.auth_headers) == {} and site.auth_bypass == site.waf_bypass == ()


def test_config_values_site_mapping_and_sequence_normalize_before_validation() -> None:
    config = make_config(ENABLE=True, TOKEN="token", BYPASS=r"^/public/",
                         NGINX_HTTPS_ENABLE=True, NGINX_AUTH_ENABLE=True)
    declaration = Nginx.site("app.test", proxy=ConfigRef("OPTIONAL", default=None),
                             template=ConfigLazy(lambda c: "/tmp/app.conf"),
                             https=ConfigRef("ENABLE"), auth=ConfigRef("ENABLE"), waf=False,
                             auth_headers=UserDict({"Authorization": ConfigFormat("Bearer {token}", token=ConfigRef("TOKEN"))}),
                             auth_bypass=(ConfigRef("BYPASS"),))
    site = make_site(config, declaration, ("nginx", "authelia"))
    assert site.resolve() is site
    assert dict(site.auth_headers) == {"Authorization": "Bearer token"}
    assert site.auth_bypass == (r"^/public/",)
    assert site.proxy is None and site.template == "/tmp/app.conf"
    with pytest.raises(TypeError):
        site.auth_headers["Authorization"] = "changed"


@pytest.mark.parametrize("value", [0, "false", [], {}])
def test_config_values_site_retains_capability_type_validation(value: object) -> None:
    site = make_site(make_config(VALUE=value), Nginx.site("app.test", auth=ConfigRef("VALUE")))
    with pytest.raises(ContainerError, match="must be None or a boolean"):
        site.auth


def test_config_values_endpoint_keeps_original_reference_and_legacy_url_proxy() -> None:
    config = make_config(DOMAIN="app.test", NGINX_HTTP_PORT=80, QUERY="a b")
    producer = SimpleNamespace(name="app", env_config=config, get_config=config.get)
    declarations = endpoint(producer, domain=ConfigRef("DOMAIN"), proxy="http://app", https=False)
    site = ResolvedSite(producer, declarations[0], 0)
    producer.manager = SimpleNamespace(integration_snapshot={"nginx": ()},
                                      containers={"nginx": SimpleNamespace(sites={declarations[0]: site})})
    url = load_nginx_url(producer, declarations, "ui", queries={"q": ConfigRef("QUERY")})
    assert type(declarations) is tuple
    assert type(url) is Proxy
    assert ResolvedSite.find(producer, declarations) is site
    assert str(url) == "http://app.test/ui?q=a+b"
    config.set("QUERY", "changed")
    assert str(url) == "http://app.test/ui?q=a+b"
    with pytest.raises(ContainerError, match="original endpoint"):
        ResolvedSite.find(producer, endpoint(producer, domain=ConfigRef("DOMAIN")))


@pytest.mark.parametrize("factory,key,disabled", [(load_config_url, "URL", ""), (load_port_url, "PORT", 0),
                                                  (load_port_url, "PORT", 65535)])
def test_config_values_url_helpers_skip_query_for_empty_or_invalid_base(factory: object, key: str, disabled: object) -> None:
    config = make_config(**{key: disabled})
    producer = SimpleNamespace(env_config=config, get_config=config.get, host=lazy_load(lambda: fail(config)))
    value = factory(producer, key, queries={"secret": ConfigLazy(fail)})
    assert type(value) is Proxy and str(value) == ""


@pytest.mark.parametrize("factory,key,value,expected", [
    (load_config_url, "URL", "https://app.test/#/setup", "https://app.test/#/setup?secret=a%2Bb+%26c"),
    (load_port_url, "PORT", 8443, "https://host.test:8443?secret=a%2Bb+%26c"),
])
def test_config_values_url_helpers_encode_nested_query_and_cache_success(factory: object, key: str,
                                                                        value: object, expected: str) -> None:
    config = make_config(SECRET="a+b &c", **{key: value})
    producer = SimpleNamespace(env_config=config, get_config=config.get, host="host.test")
    url = factory(producer, key, queries={"secret": ConfigRef("SECRET")})
    assert str(url) == expected
    config.set("SECRET", "new")
    assert str(url) == expected


def test_config_values_compose_template_resolves_only_consumed_kwargs(fresh_manager: "ContainerManager", tmp_path: "Path") -> None:
    container = fresh_manager.containers["portainer"]
    container.env_config.set("TITLE", "{{not_jinja}} {not_format}")
    source = tmp_path / "values.j2"
    source.write_text("{{ title }}|{{ data['items'][0] }}|{{ flag is false }}|{{ empty is none }}", encoding="utf-8")
    rendered = container.render_template(
        source, title=ConfigRef("TITLE"), data=ConfigLazy(lambda config: {"items": [ConfigRef("TITLE")]}),
        unused=ConfigLazy(fail), flag=False, empty=None)
    assert rendered == "{{not_jinja}} {not_format}|{{not_jinja}} {not_format}|True|True"


@pytest.mark.parametrize("variable,key", [("TEMPLATE_VALUE", "TEMPLATE_VALUE"), ("docker_user", "DOCKER_USER")])
def test_config_values_compose_template_context_reads_lazily_once_per_render(
        fresh_manager: "ContainerManager", tmp_path: "Path", monkeypatch: "pytest.MonkeyPatch",
        variable: str, key: str) -> None:
    container = fresh_manager.containers["portainer"]
    config = make_config(**{key: "first", "UNUSED": ConfigLazy(fail)})
    config.define(ConfigField(key))
    config.define(ConfigField("UNUSED"))
    monkeypatch.setattr(fresh_manager, "env_config", config)
    reads = []
    get = config.get

    def read(name: str, type: "Any" = None, default: "Any" = MISSING) -> "Any":
        reads.append(name)
        return get(name, type=type, default=default)

    monkeypatch.setattr(config, "get", read)
    source = tmp_path / "context.j2"
    source.write_text("{{ " + variable + " }}|{{ " + variable + " }}"
                      "{% if false %}{{ UNUSED }}{% endif %}", encoding="utf-8")
    assert container.render_template(source) == "first|first"
    assert reads == ["DEBUG", key]
    config.set(key, "updated")
    assert container.render_template(source) == "updated|updated"
    assert reads == ["DEBUG", key, "DEBUG", key]
    other = make_config(**{key: "other"})
    other.define(ConfigField(key))
    monkeypatch.setattr(fresh_manager, "env_config", other)
    assert container.render_template(source) == "other|other"
    assert reads == ["DEBUG", key, "DEBUG", key]


def test_config_values_compose_template_kwargs_override_deferred_context_without_reading_it(
        fresh_manager: "ContainerManager", tmp_path: "Path", monkeypatch: "pytest.MonkeyPatch") -> None:
    container = fresh_manager.containers["portainer"]
    config = make_config(OVERRIDDEN=ConfigLazy(fail), DOCKER_USER=ConfigLazy(fail))
    config.define(ConfigField("OVERRIDDEN"))
    config.define(ConfigField("DOCKER_USER"))
    monkeypatch.setattr(fresh_manager, "env_config", config)
    source = tmp_path / "override.j2"
    source.write_text("{{ OVERRIDDEN is none }}|{{ docker_user is false }}", encoding="utf-8")
    assert container.render_template(source, OVERRIDDEN=None, docker_user=False) == "True|True"


def test_config_values_native_template_reads_each_variable_in_its_producer_context(
        fresh_manager: "ContainerManager", tmp_path: "Path") -> None:
    nginx = fresh_manager.containers["nginx"]
    config = make_config(VALUE="{{literal}} $host", EMPTY=None, FLAG=False, ZERO=0)
    declaration = Nginx.site("app.test", template_vars={
        "value": ConfigRef("VALUE"), "unused": ConfigLazy(fail),
        "empty": ConfigRef("EMPTY"), "flag": ConfigRef("FLAG"), "zero": ConfigRef("ZERO"),
    })
    site = make_site(config, declaration)
    source = tmp_path / "native.conf"
    source.write_text("{{ template_vars.value }}|{{ template_vars.empty is none }}|"
                      "{{ template_vars.flag is false }}|{{ template_vars.zero }}", encoding="utf-8")
    assert nginx._render_site_template(site.producer, source, site) == "{{literal}} $host|True|True|0"


def test_config_values_unused_native_variables_remain_unread_in_full_generation(
        fresh_manager: "ContainerManager", tmp_path: "Path") -> None:
    nginx = fresh_manager.containers["nginx"]
    declarations = [Nginx.site("app.test", proxy="http://app", https=False, waf=False, auth=False,
                              default_server=index == 0, template_vars={"unused": ConfigLazy(fail)})
                    for index in range(2)]
    sites = [ResolvedSite(nginx, declaration, index) for index, declaration in enumerate(declarations)]
    nginx.__dict__["sites"] = dict(zip(declarations, sites))
    files, waf = nginx._rendered_site_files
    assert not waf and "proxy_pass" in files["sites/generated.conf"]


@pytest.mark.parametrize("second,compatible", [("10m", True), ("20m", False)])
def test_config_values_shared_used_variables_compare_resolved_values_in_each_producer(
        fresh_manager: "ContainerManager", tmp_path: "Path", second: str, compatible: bool) -> None:
    nginx = fresh_manager.containers["nginx"]
    sites = []
    for index, limit in enumerate(("10m", second)):
        config = make_config(LIMIT=limit, NGINX_HTTP_PORT=80)
        source = tmp_path / (str(index) + ".conf")
        source.write_text("location /route" + str(index) + " { return 204; }", encoding="utf-8")
        declaration = Nginx.site("app.test", template=source, https=False, waf=False, auth=False,
                                 default_server=index == 0,
                                 template_vars={"client_max_body_size": ConfigRef("LIMIT")})
        site = make_site(config, declaration)
        site.producer.name = "app" + str(index)
        site.var_name = "s_" + str(index)
        sites.append(site)
    nginx.__dict__["sites"] = {site._declaration: site for site in sites}
    if compatible:
        files, _ = nginx._rendered_site_files
        assert 'client_max_body_size "10m";' in files["sites/generated.conf"]
    else:
        with pytest.raises(ContainerError, match="Conflicting nginx template variable") as exc:
            nginx._rendered_site_files
        assert "10m" not in str(exc.value) and "20m" not in str(exc.value)


def test_config_values_shared_custom_key_conflicts_when_business_template_reads_it(
        fresh_manager: "ContainerManager", tmp_path: "Path") -> None:
    nginx = fresh_manager.containers["nginx"]
    sites = []
    for index, title in enumerate(("first", "second")):
        config = make_config(TITLE=title, NGINX_HTTP_PORT=80)
        source = tmp_path / (str(index) + ".conf")
        source.write_text("# {{ template_vars.title }}\nlocation /route" + str(index) + " { return 204; }", encoding="utf-8")
        declaration = Nginx.site("app.test", template=source, https=False, waf=False, auth=False,
                                 default_server=index == 0, template_vars={"title": ConfigRef("TITLE")})
        sites.append(make_site(config, declaration))
    nginx.__dict__["sites"] = {site._declaration: site for site in sites}
    with pytest.raises(ContainerError, match="Conflicting nginx template variable title"):
        nginx._rendered_site_files


def test_config_values_plain_template_kwargs_keep_their_native_types(
        fresh_manager: "ContainerManager", tmp_path: "Path") -> None:
    container = fresh_manager.containers["portainer"]
    source = tmp_path / "plain.j2"
    source.write_text("{{ data | tojson }}|{{ items | tojson }}|{{ pair | tojson }}", encoding="utf-8")
    assert container.render_template(source, data={"enabled": False}, items=[1, 2], pair=(3, 4)) == (
        '{"enabled": false}|[1, 2]|[3, 4]')
