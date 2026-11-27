#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Nginx bypass rendering and sensitive proxy-header boundaries."""

import re
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest

from linktools.cntr import ContainerError
from linktools.errors import ConfigError
from linktools.cntr.ext import Nginx

if TYPE_CHECKING:
    from linktools.cntr import ContainerManager


def _render_site(nginx, waf, patterns):
    site = Nginx.site(
        server_name="app.example.test", https=False, waf=waf, auth=False,
        waf_bypass=patterns,
    )
    site.var_name = "123"
    return nginx._render_site_template(
        nginx, nginx.get_source_path("templates", "server.conf"), site, business="location / { return 204; }",
    )


def test_site_waf_bypass_uses_private_origin_socket(fresh_manager):
    nginx = fresh_manager.containers["nginx"]
    rendered = _render_site(nginx, True, (r"^/health$", r"\.(css|js)$"))
    assert 'location ~* "^/health$"' in rendered
    assert r'location ~* "\\.(css|js)$"' in rendered
    assert "proxy_pass http://172.22.222.254:8000;" in rendered
    assert "proxy_pass http://unix:/run/nginx-waf-123.sock:;" in rendered
    assert "listen unix:/run/nginx-waf-123.sock;" in rendered
    assert "@waf" not in rendered and "418" not in rendered
    assert rendered.count("location / { return 204; }") == 1
    assert "include sites/" not in rendered
    assert "real_ip_header X-Forwarded-For;" in rendered

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
        fresh_manager.config.set(key, value)
    nginx = fresh_manager.containers["nginx"]
    site = next(site for site in nginx.sites.values() if site.producer.name == "authelia")
    files, _ = nginx._rendered_site_files
    rendered = files["sites/generated.conf"]
    assert site.waf is waf_enabled
    assert site.auth is True
    assert site.auth_bypass == (r"\.(css|js)$",)
    acl = fresh_manager.containers["authelia"].acl_rules
    rule = next(rule for rule in acl if rule.get("domain") == site.literal_domain)
    assert rule["subject"] == ["group:lldap_admin"]
    assert "auth_request /_internal/auth;" in rendered
    assert "location /auth-admin {" in rendered
    if not waf_enabled:
        assert "$waf_skip_" not in rendered
        assert "@waf" not in rendered
        return

    patterns = (r"^/api/", r"^/\.well-known/", r"^/jwks\.json$")
    assert site.waf_bypass == patterns
    for pattern in patterns:
        assert "location ~* " + nginx.complex_value(pattern) + " {" in rendered
    for uri in ("/api/", "/api/foo", "/.well-known/", "/.well-known/openid-configuration",
                "/jwks.json", "/jwks.json?x=1", "/API/foo"):
        assert any(re.search(pattern, uri.partition("?")[0], re.IGNORECASE) for pattern in patterns), uri
    for uri in ("/api", "/apix", "/other/.well-known/", "/.well-known", "/.well-knownx/",
                "/jwks.json.evil", "/jwks.json/", "/jwksXjson", "/", "/auth-admin", "/app.js"):
        assert not any(re.search(pattern, uri, re.IGNORECASE) for pattern in patterns), uri
    assert "proxy_pass http://unix:/run/nginx-waf-" + site.var_name + ".sock:;" in rendered
    for other in nginx.sites.values():
        if other.enabled and other is not site:
            assert other.waf is True
            assert other.waf_bypass == ()
            assert "listen unix:/run/nginx-waf-" + other.var_name + ".sock;" not in rendered


@pytest.mark.parametrize("https", (False, True))
def test_safeline_private_dns_suffix_is_http_only(fresh_manager: "ContainerManager", https: bool) -> None:
    fresh_manager.installed_state.remove(
        *(name for name in fresh_manager.installed_state.load_names() if name not in ("nginx", "safeline")),
        force=True,
    )
    for key, value in (("NGINX_ROOT_DOMAIN", "lan"), ("NGINX_WILDCARD_DOMAIN", True),
                       ("NGINX_WAF_ENABLE", True), ("NGINX_HTTPS_ENABLE", https),
                       ("NGINX_AUTH_ENABLE", False)):
        fresh_manager.config.set(key, value)
    nginx = fresh_manager.containers["nginx"]
    if https:
        with pytest.raises(ConfigError, match="NGINX_ROOT_DOMAIN with a DNS domain for HTTPS"):
            nginx.acme_ssl_domains
        return

    files, waf_enabled = nginx._rendered_site_files
    rendered = files["sites/generated.conf"]
    assert waf_enabled
    assert "server_name safeline.lan;" in rendered
    assert "ssl_certificate " not in rendered
    assert "listen 8000" in rendered
    assert 'proxy_pass http://172.22.222.254:8000;' in rendered


def test_waf_hop_keeps_common_proxy_limits_and_timeouts(fresh_manager):
    nginx = fresh_manager.containers["nginx"]
    rendered = _render_site(nginx, True, ())
    waf_location = rendered.split("location / {", 1)[1].split("}", 1)[0]
    parameters = nginx._render_site_template(
        nginx, nginx.get_source_path("templates", "params.conf"),
        SimpleNamespace(template_vars={})).splitlines()
    for directive in parameters:
        if directive.strip():
            assert directive.strip() not in waf_location
            assert rendered.count(directive.strip()) == 2
    assert "proxy_http_version 1.1;" in waf_location
    assert "proxy_set_header Upgrade $http_upgrade;" in waf_location
    assert "proxy_set_header Connection $connection_upgrade;" in waf_location
    assert "proxy_set_header X-Forwarded-For $original_client_ip;" in waf_location


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
