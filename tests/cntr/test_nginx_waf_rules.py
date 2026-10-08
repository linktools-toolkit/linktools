#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Nginx bypass rendering and sensitive proxy-header boundaries."""

import pytest

from linktools.cntr import ContainerError, NginxSite
from linktools.cntr.generation import NginxGeneration


def _render_site(nginx, waf, patterns):
    site = NginxSite(
        server_name="app.example.test", https=False, waf=waf, auth=False,
        waf_bypass=patterns,
    )
    site.file_id = "site_123"
    site.var_name = "123"
    return NginxGeneration(nginx).render_template(
        nginx, nginx.get_source_path("templates", "server.conf"), site,
    )


def test_site_waf_bypass_routes_through_named_location(fresh_manager):
    nginx = fresh_manager.containers["nginx"]
    rendered = _render_site(nginx, True, (r"^/health$", r"\.(css|js)$"))
    assert "map $uri $cntr_waf_skip_123" in rendered
    assert r'"~*^/health$" 1;' in rendered
    assert r'"~*\\.(css|js)$" 1;' in rendered
    assert "if ($cntr_waf_skip_123 = 0)" in rendered
    assert "error_page 418 = @cntr_waf" in rendered
    assert "location @cntr_waf" in rendered
    assert "include sites/site_123/business.conf;" in rendered
    assert "real_ip_header X-Cntr-Client-IP;" in rendered


def test_disabled_waf_has_no_bypass_or_internal_origin(fresh_manager):
    nginx = fresh_manager.containers["nginx"]
    rendered = _render_site(nginx, False, ())
    assert "$cntr_waf_skip_" not in rendered
    assert "@cntr_waf" not in rendered


def test_nginx_literal_preserves_data_not_template_expression(fresh_manager):
    nginx = fresh_manager.containers["nginx"]
    value = nginx._nginx_literal('Bearer "path\\$host"')
    assert value.startswith('"') and value.endswith('"')
    assert r'\"' in value
    assert r'\\' in value
    assert ('$' + '{cntr_dollar}host') in value
    assert r'\$host' not in value


@pytest.mark.parametrize("value", ["secret\rx", "secret\nx", "secret\x00x"])
def test_nginx_header_rejects_control_characters_without_leaking(fresh_manager, value):
    nginx = fresh_manager.containers["nginx"]
    with pytest.raises(ContainerError, match="control character") as exc:
        nginx._nginx_literal(value)
    assert "secret" not in str(exc.value)


def test_nginx_auth_header_dedup_and_reserved_names(fresh_manager):
    nginx = fresh_manager.containers["nginx"]
    assert nginx._validated_auth_headers({"Authorization": "value"}) == {"Authorization": "value"}
    for headers in (
        {"X-Cntr-Method": "bad"},
        {"X-Auth-User": "bad"},
        {"x-auth-user": "bad"},
        {"X-Original-URL": "bad"},
        {"My-Key": "x", "my-key": "y"},
        {"Bad\nKey": "x"},
    ):
        with pytest.raises(ContainerError):
            nginx._validated_auth_headers(headers)
