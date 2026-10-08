#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Single-pass template namespaces and complete proxy header ownership."""

from typing import TYPE_CHECKING

import pytest

from linktools.cntr import ContainerError, Nginx
from linktools.cntr.container import ContainerTemplateError


if TYPE_CHECKING:
    from typing import Any, Mapping, Optional
    from linktools.cntr import ContainerManager
    from linktools.cntr.container import BaseContainer
    from linktools.cntr.integration import ResolvedSite



def make_site(**kwargs):
    values = dict(server_name="app.example.test", https=True, waf=True, auth=True,
                  proxy="http://app:8080", auth_headers={"Authorization": 'Bearer "$host"'})
    values.update(kwargs)
    site = Nginx.site(**values)
    site.local_id = "web"
    site.file_id = site.var_name = "s_test"
    return site


def test_header_macros_merge_case_insensitively_and_preserve_native_values(fresh_manager, tmp_path):
    nginx = fresh_manager.containers["nginx"]
    source = tmp_path / "business.conf"
    source.write_text('''{% from "nginx/headers.j2" import proxy_headers, grpc_headers with context %}
{{ proxy_headers({"host": "$native_host", "Origin": "https://$original_host/a\\\\b"}) }}
{{ grpc_headers({"Forwarded": "for=$original_client_ip"}) }}''')
    rendered = nginx._render_site_template(nginx, source, make_site())
    assert rendered.count('proxy_set_header "host" ') == 1
    assert 'proxy_set_header "Host" ' not in rendered
    assert 'proxy_set_header "host" "$native_host";' in rendered
    assert 'grpc_set_header "Forwarded" "for=$original_client_ip";' in rendered
    for name in ("Scheme", "Host", "URI", "Method", "Client-IP"):
        assert 'proxy_set_header "X-Proxy-Original-' + name + '" "";' in rendered
    assert 'proxy_set_header "X-Forwarded-For" "$original_client_ip";' in rendered
    assert 'proxy_set_header "Forwarded" "";' in rendered


@pytest.mark.parametrize("overrides", [
    {"x-proxy-original-host": "bad"}, {"x-auth-user": "bad"}, {"authorization": "bad"},
    {"Host": "one", "host": "two"}, {"X-Test": "secret\nvalue"}, {"Bad\rName": "x"},
])
def test_header_overrides_reject_conflicts_and_controls(fresh_manager, overrides):
    with pytest.raises(ContainerError) as error:
        fresh_manager.containers["nginx"].header_items(make_site(), overrides)
    assert "secret" not in str(error.value)


def test_auth_maps_require_exact_success_without_regex_capture_side_effects(fresh_manager):
    nginx = fresh_manager.containers["nginx"]
    rendered = nginx.security_maps(make_site(auth_bypass=(r"^/public",)))
    gate = rendered[rendered.index('map "$auth_status_'):]
    assert '"200:0:1" 1;' in gate and '"299:0:1" 1;' in gate
    assert '"300:0:1:1" 1;' not in gate and '"204:1:1" 1;' not in gate
    assert "~" not in gate
    assert "default ${http_authorization};" in gate
    assert "${literal_dollar}host" in gate


def test_header_names_are_quoted_and_auth_data_is_not_reinterpreted(fresh_manager, tmp_path):
    nginx = fresh_manager.containers["nginx"]
    site = make_site(auth_headers={"#Odd": "{{not_jinja}} $value"})
    source = tmp_path / "business.conf"
    source.write_text('{% from "nginx/headers.j2" import proxy_headers with context %}{{ proxy_headers() }}')
    assert 'proxy_set_header "#Odd" ' in nginx._render_site_template(nginx, source, site)
    maps = nginx.security_maps(site)
    assert "{{not_jinja}} ${literal_dollar}value" in maps


def test_namespaced_single_pass_includes_comments_raw_and_literal_values(fresh_manager, tmp_path):
    nginx = fresh_manager.containers["nginx"]
    (tmp_path / "headers.j2").write_text("local header body")
    (tmp_path / "child.conf").write_text("{{ vars.literal }}")
    source = tmp_path / "business.conf"
    source.write_text('''# {{ vars.comment }}
{% include "local/headers.j2" %}
{% include "local/child.conf" %}
{% from "nginx/headers.j2" import proxy_headers with context %}{{ proxy_headers() }}
{% raw %}{{port}}{% endraw %}''')
    rendered = nginx._render_site_template(nginx, source, make_site(vars={
        "comment": "evaluated", "literal": "{{unexpanded}} $native",
    }))
    assert "# evaluated" in rendered
    assert "local header body" in rendered
    assert "{{unexpanded}} $native" in rendered
    assert "{{port}}" in rendered
    assert 'proxy_set_header "Host"' in rendered


@pytest.mark.parametrize("content", ["{{ missing }}", '{% include "child.conf" %}',
                                     '{% include "local/missing.conf" %}'])
def test_template_errors_identify_owner_and_entrypoint(fresh_manager, tmp_path, content):
    nginx = fresh_manager.containers["nginx"]
    source = tmp_path / "business.conf"
    source.write_text(content)
    with pytest.raises(ContainerTemplateError) as error:
        nginx._render_site_template(nginx, source, make_site())
    assert "nginx/web" in str(error.value)
    assert str(source) in str(error.value)


def test_business_file_is_rendered_once_across_generation_markers(fresh_manager, tmp_path):
    nginx = fresh_manager.containers["nginx"]
    calls = []
    source = tmp_path / "business.conf"
    source.write_text("{{ vars.record() }}\nlocation / { return 204; }")
    site = make_site(server_name="_", default=True, waf=False, auth=False, template=source,
                     vars={"record": lambda: calls.append("render") or "# business"})
    site.producer = nginx
    site.enabled = True
    site.resolve = lambda: site
    nginx.__dict__["sites"] = {("nginx", "web"): site}
    owner = nginx
    first = owner.render_config("first")
    second = owner.render_config("second")
    assert calls == ["render"]
    assert set(first) == {"nginx.conf", "sites/s_test.conf"}
    assert "include sites/" not in first["sites/s_test.conf"]
    assert first["sites/s_test.conf"] == second["sites/s_test.conf"]
    assert 'return 200 "first"' in first["nginx.conf"]
    assert 'return 200 "second"' in second["nginx.conf"]
    assert "/current/" not in "\n".join(first.values())
    assert "NGINX_ROOT_site" not in "\n".join(first.values())
    assert "default_server" in first["sites/s_test.conf"]


@pytest.mark.parametrize("server_name", ["_", "app.example.test", "*.example.test", "~^app\\.example\\.test$"])
@pytest.mark.parametrize("default", [False, True])
def test_default_listeners_are_explicit_and_independent_of_server_name(
        fresh_manager: "ContainerManager", server_name: str, default: bool) -> None:
    nginx = fresh_manager.containers["nginx"]
    site = make_site(server_name=server_name, default=default)
    rendered = nginx._render_site_template(
        nginx, nginx.get_source_path("templates", "server.conf"), site)
    assert rendered.count("default_server") == (3 if default else 0)
    assert rendered.count("server_name " + server_name + ";") == 3
    assert "auth_request /_internal/auth;" in rendered
    assert "proxy_pass $waf_target;" in rendered
    assert "ssl_certificate " in rendered


def generation_site(nginx: "BaseContainer", local_id: str = "web",
                    ports: "Optional[Mapping[str, int]]" = None, **kwargs: "Any") -> "ResolvedSite":
    from types import SimpleNamespace
    from linktools.cntr.integration import ResolvedSite

    def get_config(key: str, **options: "Any") -> "Any":
        return ports[key] if ports and key in ports else nginx.get_config(key, **options)

    producer = SimpleNamespace(name=local_id, manager=nginx.manager,
                               get_config=get_config, env_config=SimpleNamespace(get=get_config))
    values = dict(server_name="app.example.test", proxy="http://app:8080",
                  https=False, waf=False, auth=False)
    values.update(kwargs)
    return ResolvedSite(producer, local_id, Nginx.site(**values))


@pytest.mark.parametrize("default", [False, True])
def test_fallback_depends_on_explicit_default_not_underscore(fresh_manager: "ContainerManager", default: bool) -> None:
    nginx = fresh_manager.containers["nginx"]
    site = generation_site(nginx, server_name="_", default=default)
    nginx.__dict__["sites"] = {site.identity: site}
    files = nginx.render_config("test")
    assert ("sites/default.conf" in files) is not default
    assert ("default_server" in files["sites/" + site.file_id + ".conf"]) is default
    if not default:
        assert "default_server" in files["sites/default.conf"]
        assert 'server_name "";' in files["sites/default.conf"]
        assert "server_name _;" not in files["sites/default.conf"]
        assert "server_name _;" in files["sites/" + site.file_id + ".conf"]


def test_disabled_default_keeps_fallback_without_resolving_other_fields(fresh_manager: "ContainerManager") -> None:
    from linktools.runtime import lazy_load

    def fail() -> None:
        raise AssertionError("disabled site evaluated")

    nginx = fresh_manager.containers["nginx"]
    site = generation_site(nginx, server_name="", default=lazy_load(fail), proxy=lazy_load(fail))
    nginx.__dict__["sites"] = {site.identity: site}
    files = nginx.render_config("test")
    assert "sites/default.conf" in files
    assert "sites/" + site.file_id + ".conf" not in files


@pytest.mark.parametrize("shared", [True, False])
def test_explicit_defaults_only_conflict_on_shared_listeners(fresh_manager: "ContainerManager", shared: bool) -> None:
    nginx = fresh_manager.containers["nginx"]
    first = generation_site(nginx, "first", ports={"NGINX_HTTP_PORT": 8080}, default=True)
    second = generation_site(nginx, "second", ports={"NGINX_HTTP_PORT": 8080 if shared else 8081}, default=True)
    nginx.__dict__["sites"] = {site.identity: site for site in (first, second)}
    owner = nginx
    if shared:
        with pytest.raises(ContainerError, match="[Dd]efault"):
            owner.render_config("test")
    else:
        files = owner.render_config("test")
        assert "listen 8080 default_server;" in files["sites/" + first.file_id + ".conf"]
        assert "listen 8081 default_server;" in files["sites/" + second.file_id + ".conf"]
        assert "sites/default.conf" not in files


@pytest.mark.parametrize("capability,port", [("https", "NGINX_HTTPS_PORT"), ("waf", "NGINX_WAF_PORT")])
def test_defaults_detect_collisions_on_optional_listeners(
        fresh_manager: "ContainerManager", capability: str, port: str) -> None:
    nginx = fresh_manager.containers["nginx"]
    first = generation_site(nginx, "first", ports={"NGINX_HTTP_PORT": 8080, port: 8443},
                            default=True, **{capability: True})
    second = generation_site(nginx, "second", ports={"NGINX_HTTP_PORT": 8081, port: 8443},
                             default=True, **{capability: True})
    nginx.__dict__["sites"] = {site.identity: site for site in (first, second)}
    with pytest.raises(ContainerError, match="[Dd]efault"):
        nginx.render_config("test")


def test_shared_hostname_routes_merge_with_independent_auth_maps(fresh_manager, tmp_path):
    nginx = fresh_manager.containers["nginx"]
    api_template = tmp_path / "api.conf"
    web_template = tmp_path / "web.conf"
    api_template.write_text("location /api { return 204; }")
    web_template.write_text("location /web { return 204; }")
    api = generation_site(nginx, "api", template=api_template, https=True,
                          auth=True, auth_bypass=(r"^/public/",))
    web = generation_site(nginx, "web", template=web_template, https=True,
                          auth=True, auth_bypass=(r"^/public/",))
    nginx.__dict__["sites"] = {site.identity: site for site in (api, web)}
    files = nginx.render_config("generation")
    servers = [value for name, value in files.items() if name.startswith("sites/") and name != "sites/default.conf"]
    assert len(servers) == 1
    server = servers[0]
    assert server.count("server_name app.example.test;") == 2
    assert "location /api { return 204; }" in server
    assert "location /web { return 204; }" in server
    for site in (api, web):
        assert "# site {}/{}".format(site.producer.name, site.local_id) in server
        assert "$auth_verified_" + site.var_name in server
        assert "auth_request_set $auth_status_" + site.var_name in server


def test_shared_hostname_rejects_incompatible_security_policies(fresh_manager):
    nginx = fresh_manager.containers["nginx"]
    api = generation_site(nginx, "api", auth=True, https=True,
                          auth_bypass=(r"^/api/public/",))
    web = generation_site(nginx, "web", auth=True, https=True,
                          auth_bypass=(r"^/web/public/",))
    nginx.__dict__["sites"] = {site.identity: site for site in (api, web)}
    with pytest.raises(ContainerError, match="Incompatible nginx routing policies"):
        nginx.render_config("generation")


def test_duplicate_locations_remain_visible_to_native_validator(fresh_manager):
    nginx = fresh_manager.containers["nginx"]
    first = generation_site(nginx, "first")
    second = generation_site(nginx, "second")
    nginx.__dict__["sites"] = {site.identity: site for site in (first, second)}
    files = nginx.render_config("generation")
    combined = files["sites/" + first.file_id + ".conf"]
    assert combined.count("location / {") == 2


def test_header_overrides_suppress_headers_without_losing_same_level_defaults(fresh_manager):
    nginx = fresh_manager.containers["nginx"]
    headers = dict(nginx.header_items(make_site(auth_headers={}), {
        "host": "$host", "Authorization": "$http_authorization", "X-Forwarded-For": None,
    }))
    assert headers["host"] == '"$host"'
    assert headers["Authorization"] == '"$http_authorization"'
    assert headers["X-Forwarded-For"] == '""'
    assert headers["X-Real-IP"] == '"$original_client_ip"'
    assert headers["X-Proxy-Original-Host"] == '""'
    assert "Host" not in headers


def test_generated_sites_are_self_contained_and_internal_names_are_purpose_specific(fresh_manager):
    nginx = fresh_manager.containers["nginx"]
    site = make_site(default=True)
    site.producer = nginx
    site.enabled = True
    site.resolve = lambda: site
    nginx.__dict__["sites"] = {("nginx", "web"): site}
    files = nginx.render_config("example")
    assert set(files) == {"nginx.conf", "sites/s_test.conf"}
    rendered = files["sites/s_test.conf"]
    assert "location = /_internal/auth" in rendered
    assert "location / {" in rendered
    assert "include sites/" not in rendered
    assert "cntr" not in "\n".join(files.values()).lower()
    assert "map $server_port $original_uri" in files["nginx.conf"]
    assert "map $server_port $request_uri" not in files["nginx.conf"]


def test_embedded_business_preserves_multiline_quoted_values(fresh_manager, tmp_path):
    nginx = fresh_manager.containers["nginx"]
    source = tmp_path / "business.conf"
    content = 'location / { return 200 "first\nsecond"; }'
    source.write_text(content)
    site = make_site(default=True, auth=False, waf=False, template=source)
    site.producer = nginx
    site.enabled = True
    site.resolve = lambda: site
    nginx.__dict__["sites"] = {("nginx", "web"): site}
    rendered = nginx.render_config("example")["sites/s_test.conf"]
    assert content in rendered
