#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Nginx bypass rendering and sensitive proxy-header boundaries."""

import re
import shlex
from typing import TYPE_CHECKING

import pytest

from linktools.cntr import ContainerError
from linktools.cntr.ext import Nginx

if TYPE_CHECKING:
    from linktools.cntr import ContainerManager


def _render_site(nginx, waf, patterns):
    site = Nginx.site(
        server_name="app.example.test", https=False, waf=waf, auth=False,
        waf_bypass=patterns,
    )
    site.file_id = "site_123"
    site.var_name = "123"
    return nginx._render_site_template(
        nginx, nginx.get_source_path("templates", "server.conf"), site, business="location / { return 204; }",
    )


def test_site_waf_bypass_routes_through_named_location(fresh_manager):
    nginx = fresh_manager.containers["nginx"]
    rendered = _render_site(nginx, True, (r"^/health$", r"\.(css|js)$"))
    assert "map $uri $waf_skip_123" in rendered
    assert r'"~*^/health$" 1;' in rendered
    assert r'"~*\\.(css|js)$" 1;' in rendered
    assert "if ($waf_skip_123 = 0)" in rendered
    assert "error_page 418 = @waf" in rendered
    assert "location @waf" in rendered
    assert "location / { return 204; }" in rendered
    assert "include sites/" not in rendered
    assert "real_ip_header X-Proxy-Original-Client-IP;" in rendered


def test_disabled_waf_has_no_bypass_or_internal_origin(fresh_manager):
    nginx = fresh_manager.containers["nginx"]
    rendered = _render_site(nginx, False, ())
    assert "$waf_skip_" not in rendered
    assert "@waf" not in rendered


@pytest.mark.parametrize("waf_enabled", [True, False])
def test_authelia_waf_bypass_preserves_auth_and_other_sites(
        fresh_manager: "ContainerManager", waf_enabled: bool) -> None:
    for key, value in (("NGINX_ROOT_DOMAIN", "example.test"), ("NGINX_WILDCARD_DOMAIN", True),
                       ("NGINX_WAF_ENABLE", waf_enabled), ("NGINX_HTTPS_ENABLE", True),
                       ("NGINX_AUTH_ENABLE", True), ("AUTHELIA_ADMIN_AUTH_ENABLE", True)):
        fresh_manager.env_config.set(key, value)
    nginx = fresh_manager.containers["nginx"]
    site = nginx.sites[("authelia", "web")]
    files, _ = nginx._rendered_site_files
    rendered = files["sites/" + site.file_id + ".conf"]
    assert site.waf is waf_enabled
    assert site.auth is True
    assert site.auth_bypass == (r"\.(css|js)$",)
    assert site.auth_rule["subject"] == ["group:lldap_admin"]
    assert "auth_request /_internal/auth;" in rendered
    assert "location /auth-admin {" in rendered
    if not waf_enabled:
        assert "$waf_skip_" not in rendered
        assert "@waf" not in rendered
        return

    waf_map = rendered.split("map $uri $waf_skip_" + site.var_name + " {", 1)[1].split("}", 1)[0]
    assert "default 0;" in waf_map
    patterns = [shlex.split(line)[0][2:] for line in waf_map.splitlines() if '"~*' in line]
    assert patterns == [r"^/api/", r"^/\.well-known/", r"^/jwks\.json$"]
    for uri in ("/api/", "/api/foo", "/.well-known/", "/.well-known/openid-configuration",
                "/jwks.json", "/jwks.json?x=1", "/API/foo"):
        assert any(re.search(pattern, uri.partition("?")[0], re.IGNORECASE) for pattern in patterns), uri
    for uri in ("/api", "/apix", "/other/.well-known/", "/.well-known", "/.well-knownx/",
                "/jwks.json.evil", "/jwks.json/", "/jwksXjson", "/", "/auth-admin", "/app.js"):
        assert not any(re.search(pattern, uri, re.IGNORECASE) for pattern in patterns), uri
    assert "if ($waf_skip_" + site.var_name + " = 0) { return 418; }" in rendered
    assert "location @waf" in rendered
    for other in nginx.sites.values():
        if other.enabled and other.identity != site.identity:
            assert other.waf is True
            assert other.waf_bypass == ()
            assert '"~*^/api/" 1;' not in files["sites/" + other.file_id + ".conf"]


def test_waf_hop_keeps_common_proxy_limits_and_timeouts(fresh_manager):
    nginx = fresh_manager.containers["nginx"]
    rendered = _render_site(nginx, True, ())
    waf_location = rendered.split("location @waf {", 1)[1].split("}", 1)[0]
    parameters = nginx.get_source_path("templates", "params.conf").read_text().splitlines()
    for directive in parameters:
        if directive.strip():
            assert waf_location.count(directive.strip()) == 1
    assert "proxy_http_version 1.1;" in waf_location
    assert "proxy_set_header Upgrade $http_upgrade;" in waf_location
    assert "proxy_set_header Connection $connection_upgrade;" in waf_location
    assert "proxy_set_header X-Proxy-Original-Client-IP $original_client_ip;" in waf_location


def test_nginx_literal_preserves_data_not_template_expression(fresh_manager):
    nginx = fresh_manager.containers["nginx"]
    value = nginx.quote('Bearer "path\\$host"')
    assert value.startswith('"') and value.endswith('"')
    assert r'\"' in value
    assert r'\\' in value
    assert ('$' + '{literal_dollar}host') in value
    assert r'\$host' not in value


@pytest.mark.parametrize("value", ["secret\rx", "secret\nx", "secret\x00x"])
def test_nginx_header_rejects_control_characters_without_leaking(fresh_manager, value):
    nginx = fresh_manager.containers["nginx"]
    with pytest.raises(ContainerError, match="control character") as exc:
        nginx.quote(value)
    assert "secret" not in str(exc.value)


def test_nginx_auth_header_dedup_and_reserved_names(fresh_manager):
    nginx = fresh_manager.containers["nginx"]
    assert nginx.validated_auth_headers({"Authorization": "value"}) == {"Authorization": "value"}
    for headers in (
        {"X-Proxy-Original-Method": "bad"},
        {"X-Auth-User": "bad"},
        {"x-auth-user": "bad"},
        {"X-Original-URL": "bad"},
        {"My-Key": "x", "my-key": "y"},
        {"Bad\nKey": "x"},
    ):
        with pytest.raises(ContainerError):
            nginx.validated_auth_headers(headers)
