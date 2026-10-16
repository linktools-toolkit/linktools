#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Config-dependent integration fields resolve only at their consumption boundary."""
from linktools.core import ConfigLazy
from linktools.cntr.ext import ResolvedSite
from collections import UserDict
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest

from linktools.errors import ConfigError

from linktools.cntr import ContainerError
from linktools.cntr.ext import Nginx, endpoint, load_config_url, load_nginx_url, load_port_url
from linktools.core import Config, ConfigField, ConfigFormat, ConfigRef, ConfigSchema, RuntimeOverrideSource
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


def test_config_values_endpoint_keeps_original_reference_and_resolves_current_query() -> None:
    config = make_config(DOMAIN="app.test", NGINX_HTTP_PORT=80, QUERY="a b")
    producer = SimpleNamespace(name="app", env_config=config, get_config=config.get)
    declarations = endpoint(producer, domain=ConfigRef("DOMAIN"), proxy="http://app", https=False)
    site = ResolvedSite(producer, declarations[0], 0)
    producer.manager = SimpleNamespace(integration_snapshot={"nginx": ()},
                                      containers={"nginx": SimpleNamespace(sites={declarations[0]: site})})
    url = load_nginx_url(ResolvedSite.find(producer, declarations), "ui", queries={"q": ConfigRef("QUERY")})
    assert type(declarations) is tuple
    assert type(url) is ConfigLazy
    assert producer.env_config.resolve(ResolvedSite.find(producer, declarations)) is site
    assert config.resolve(url) == "http://app.test/ui?q=a+b"
    config.set("QUERY", "changed")
    assert config.resolve(url) == "http://app.test/ui?q=changed"
    with pytest.raises(ConfigError, match="ConfigLazy callback failed"):
        producer.env_config.resolve(ResolvedSite.find(producer, endpoint(producer, domain=ConfigRef("DOMAIN"))))


@pytest.mark.parametrize("factory,key,disabled", [(load_config_url, "URL", ""), (load_port_url, "PORT", 0),
                                                  (load_port_url, "PORT", 65536)])
def test_config_values_url_helpers_skip_query_for_empty_or_invalid_base(factory: object, key: str, disabled: object) -> None:
    config = make_config(HOST=ConfigLazy(fail), **{key: disabled})
    value = factory(ConfigRef(key, type=str if factory is load_config_url else int,
                              default=None if factory is load_config_url else 0), queries={"secret": ConfigLazy(fail)})
    assert type(value) is ConfigLazy and config.resolve(value) == ""


@pytest.mark.parametrize("factory,key,value,expected", [
    (load_config_url, "URL", "https://app.test/#/setup", "https://app.test/#/setup?secret=a%2Bb+%26c"),
    (load_port_url, "PORT", 8443, "https://host.test:8443?secret=a%2Bb+%26c"),
])
def test_config_values_url_helpers_encode_nested_query_in_current_config(factory: object, key: str,
                                                                        value: object, expected: str) -> None:
    config = make_config(HOST="host.test", SECRET="a+b &c", **{key: value})
    url = factory(ConfigRef(key, type=str if factory is load_config_url else int,
                              default=None if factory is load_config_url else 0), queries={"secret": ConfigRef("SECRET")})
    assert config.resolve(url) == expected
    config.set("SECRET", "new")
    assert config.resolve(url) == expected.replace("a%2Bb+%26c", "new")


def test_config_values_compose_template_resolves_only_consumed_kwargs(fresh_manager: "ContainerManager", tmp_path: "Path") -> None:
    container = fresh_manager.containers["portainer"]
    container.env_config.set("TITLE", "{{not_jinja}} {not_format}")
    source = tmp_path / "values.j2"
    source.write_text("{{ title }}|{{ data['items'][0] }}|{{ flag is false }}|{{ empty is none }}", encoding="utf-8")
    rendered = container.render_template(
        source, title=ConfigRef("TITLE"), data=ConfigLazy(lambda config: {"items": [ConfigRef("TITLE")]}),
        unused=ConfigLazy(fail), flag=False, empty=None)
    assert rendered == "{{not_jinja}} {not_format}|{{not_jinja}} {not_format}|True|True"


def test_config_values_compose_template_consumes_inline_field_without_registration(
        fresh_manager: "ContainerManager", tmp_path: "Path") -> None:
    container = fresh_manager.containers["portainer"]
    config = container.env_config
    field = ConfigField("INLINE_PORT", default="8080", cast=int)
    reference = ConfigRef(field)
    revision = config.revision
    source = tmp_path / "inline.j2"
    source.write_text("{{ port + 1 }}|{{ port }}", encoding="utf-8")
    assert container.render_template(source, port=reference) == "8081|8080"
    assert config.schema.get("INLINE_PORT") is None
    assert config.revision == revision
    assert config.get("INLINE_PORT", default="absent") == "absent"
    field.default = "9090"
    assert container.render_template(source, port=reference) == "9091|9090"


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


@pytest.mark.parametrize("factory,key,first,second,expected", [
    (load_config_url, "URL", "https://one.test", "https://two.test", "https://two.test/ui?q=second"),
    (load_port_url, "PORT", 8443, 65535, "https://two.test:65535/ui?q=second"),
])
def test_url_helpers_reuse_inline_fields_and_query_refs_across_configs(
        factory: object, key: str, first: object, second: object, expected: str) -> None:
    field = ConfigField(key, default=first)
    query = ConfigRef(ConfigField("QUERY", default="first"))
    value = factory(ConfigRef(field, type=str if factory is load_config_url else int,
                                   default=None if factory is load_config_url else 0), "ui", queries={"q": query})
    one = make_config(HOST="one.test")
    two = make_config(HOST="two.test", QUERY="second", **{key: second})
    first_result = one.resolve(value)
    assert "one.test" in first_result and first_result.endswith("?q=first")
    assert two.resolve(value) == expected
    assert one.resolve(value) == first_result
    assert one.schema.get(key) is None and two.schema.get(key) is None
    assert one.schema.get("QUERY") is None and two.schema.get("QUERY") is None


@pytest.mark.parametrize("deferred", [False, True])
def test_url_helpers_accept_values_and_resolve_path_https_and_query(deferred: bool) -> None:
    config = make_config(HOST="host.test", PATH="ui", HTTPS=False, QUERY="a b")
    url = ConfigLazy(lambda config: "https://literal.test/base") if deferred else "https://literal.test/base"
    port = ConfigLazy(lambda config: 8080) if deferred else 8080
    path = ConfigRef("PATH")
    query = {"q": ConfigRef("QUERY")}
    assert config.resolve(load_config_url(url, path, queries=query)) == "https://literal.test/base/ui?q=a+b"
    value = load_port_url(port, path, https=ConfigRef("HTTPS"), queries=query)
    assert config.resolve(value) == "http://host.test:8080/ui?q=a+b"
    config.set("HTTPS", True)
    config.set("PATH", "other")
    assert config.resolve(value) == "https://host.test:8080/other?q=a+b"


@pytest.mark.parametrize("factory,value", [(load_config_url, None), (load_config_url, ""),
                                           (load_port_url, 0), (load_port_url, -1), (load_port_url, 65536)])
def test_disabled_url_values_skip_all_dependent_values(factory: object, value: object) -> None:
    config = make_config(HOST=ConfigLazy(fail))
    kwargs = {"https": ConfigLazy(fail)} if factory is load_port_url else {}
    helper = factory(value, ConfigLazy(fail), queries=ConfigLazy(fail), **kwargs)
    assert config.resolve(helper) == ""


@pytest.mark.parametrize("raw,expected", [("8080", "https://host.test:8080/ui"), ("bad", ""), (0, "")])
def test_port_url_explicit_reference_owns_cast_and_default(raw: object, expected: str) -> None:
    config = make_config(HOST="host.test", PORT=raw)
    value = load_port_url(ConfigRef("PORT", type=int, default=0), "ui")
    assert config.resolve(value) == expected
    assert make_config().resolve(load_port_url(ConfigRef("ABSENT", type=int, default=0), ConfigLazy(fail))) == ""


def test_config_url_explicit_reference_owns_missing_default_and_string_cast() -> None:
    config = make_config(URL=123)
    assert config.resolve(load_config_url(ConfigRef("URL", type=str, default=None))) == "123"
    assert make_config().resolve(load_config_url(ConfigRef("ABSENT", type=str, default=None), ConfigLazy(fail))) == ""


def test_config_url_literal_string_is_not_a_configuration_key() -> None:
    config = make_config(URL="https://configured.test")
    assert config.resolve(load_config_url("URL")) == "URL"
