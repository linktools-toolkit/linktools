#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Single-pass template namespaces and complete proxy header ownership."""

import pytest

from linktools.cntr import ContainerError, Nginx
from _harness import builtin_consumer_type
from linktools.cntr.container import ContainerTemplateError


NginxGeneration = builtin_consumer_type("100-nginx")


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
{{ proxy_headers({"host": "$native_host", "Origin": "https://$cntr_host/a\\\\b"}) }}
{{ grpc_headers({"Forwarded": "for=$cntr_client_ip"}) }}''')
    rendered = NginxGeneration(nginx).render_template(nginx, source, make_site())
    assert rendered.count('proxy_set_header "host" ') == 1
    assert 'proxy_set_header "Host" ' not in rendered
    assert 'proxy_set_header "host" "$native_host";' in rendered
    assert 'grpc_set_header "Forwarded" "for=$cntr_client_ip";' in rendered
    for name in ("Scheme", "Host", "URI", "Method", "Client-IP"):
        assert 'proxy_set_header "X-Cntr-' + name + '" "";' in rendered
    assert 'proxy_set_header "X-Forwarded-For" "$cntr_client_ip";' in rendered
    assert 'proxy_set_header "Forwarded" "";' in rendered


@pytest.mark.parametrize("overrides", [
    {"x-cntr-host": "bad"}, {"x-auth-user": "bad"}, {"authorization": "bad"},
    {"Host": "one", "host": "two"}, {"X-Test": "secret\nvalue"}, {"Bad\rName": "x"},
])
def test_header_overrides_reject_conflicts_and_controls(fresh_manager, overrides):
    with pytest.raises(ContainerError) as error:
        fresh_manager.containers["nginx"].header_items(make_site(), overrides)
    assert "secret" not in str(error.value)


def test_auth_maps_require_exact_success_without_regex_capture_side_effects(fresh_manager):
    nginx = fresh_manager.containers["nginx"]
    rendered = nginx.security_maps(make_site(auth_bypass=(r"^/public",)))
    gate = rendered[rendered.index('map "$cntr_auth_status_'):]
    assert '"200:0:1" 1;' in gate and '"299:0:1" 1;' in gate
    assert '"300:0:1:1" 1;' not in gate and '"204:1:1" 1;' not in gate
    assert "~" not in gate
    assert "default ${http_authorization};" in gate
    assert "${cntr_dollar}host" in gate


def test_header_names_are_quoted_and_auth_data_is_not_reinterpreted(fresh_manager, tmp_path):
    nginx = fresh_manager.containers["nginx"]
    site = make_site(auth_headers={"#Odd": "{{not_jinja}} $value"})
    source = tmp_path / "business.conf"
    source.write_text('{% from "nginx/headers.j2" import proxy_headers with context %}{{ proxy_headers() }}')
    assert 'proxy_set_header "#Odd" ' in NginxGeneration(nginx).render_template(nginx, source, site)
    maps = nginx.security_maps(site)
    assert "{{not_jinja}} ${cntr_dollar}value" in maps


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
    rendered = NginxGeneration(nginx).render_template(nginx, source, make_site(vars={
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
        NginxGeneration(nginx).render_template(nginx, source, make_site())
    assert "nginx/web" in str(error.value)
    assert str(source) in str(error.value)


def test_business_file_is_rendered_once_across_generation_markers(fresh_manager, tmp_path):
    nginx = fresh_manager.containers["nginx"]
    calls = []
    source = tmp_path / "business.conf"
    source.write_text("{{ vars.record() }}\nlocation / { return 204; }")
    site = make_site(server_name="_", waf=False, auth=False, template=source,
                     vars={"record": lambda: calls.append("render") or "# business"})
    site.producer = nginx
    site.enabled = True
    site.resolve = lambda: site
    nginx.__dict__["sites"] = {("nginx", "web"): site}
    owner = NginxGeneration(nginx)
    first = owner.render("first")
    second = owner.render("second")
    assert calls == ["render"]
    assert first["sites/s_test/business.conf"] == second["sites/s_test/business.conf"]
    assert 'return 200 "first"' in first["nginx.conf"]
    assert 'return 200 "second"' in second["nginx.conf"]
    assert "/current/" not in "\n".join(first.values())
    assert "NGINX_ROOT_site" not in "\n".join(first.values())
    assert "default_server" in first["sites/s_test.conf"]
