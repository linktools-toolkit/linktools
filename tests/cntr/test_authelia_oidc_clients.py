#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Authelia client metadata and derived redirects never mutate saved state."""

import pytest

from linktools.cntr import Nginx
from linktools.cntr.ext import Authelia, ResolvedSite


@pytest.fixture(autouse=True)
def configure_https_identity(fresh_manager):
    fresh_manager.env_config.set("NGINX_ROOT_DOMAIN", "example.com")
    fresh_manager.env_config.set("AUTHELIA_DOMAIN", "sso.example.com")
    fresh_manager.env_config.set("NGINX_HTTPS_ENABLE", True)
    fresh_manager.env_config.set("NGINX_AUTH_ENABLE", True)


def test_oidc_client_is_read_only_and_uses_saved_secret(fresh_manager):
    authelia = fresh_manager.containers["authelia"]
    client = authelia.oidc_client
    assert client["client_id"] == fresh_manager.project_name + "-web-client"
    assert client["client_secret"] == authelia.get_config("AUTHELIA_OIDC_CLIENT_SECRET")
    assert client["issuer_url"].startswith("https://")
    assert isinstance(client["scopes"], tuple)
    with pytest.raises(TypeError):
        client["client_id"] = "overwritten"


def test_oidc_redirects_derive_only_from_current_declarations(fresh_manager, monkeypatch):
    authelia = fresh_manager.containers["authelia"]
    producer = fresh_manager.containers["portainer"]
    declaration = Authelia.oidc(("https://service.example.com", "https://service.example.com/callback",
                                 "https://external.example.com/callback", "https://service.example.com/callback"))
    monkeypatch.setattr(fresh_manager, "iter_integrations", lambda consumer: iter(((producer, declaration),)))
    assert authelia.oidc_redirects == (
        authelia.oidc_client["issuer_url"], "https://service.example.com",
        "https://service.example.com/callback", "https://external.example.com/callback",
    )
    assert isinstance(authelia.oidc_client["redirect_uris"], tuple)


def test_acl_supports_native_optional_fields(fresh_manager, monkeypatch):
    authelia = fresh_manager.containers["authelia"]
    producer = fresh_manager.containers["portainer"]
    site = Nginx.site(
        server_name="secure.example.com",
        proxy="http://app:8080",
        auth_rule={"policy": "one_factor", "networks": ["10.0.0.0/8"]},
    )
    monkeypatch.setattr(fresh_manager.containers["nginx"], "sites", {
        (producer.name, "web"): ResolvedSite(producer, "web", site),
    })
    assert authelia.acl_rules == [
        {"policy": "one_factor", "networks": ["10.0.0.0/8"],
         "domain": "secure.example.com"}
    ]
    assert "secure.example.com" in authelia.acl_config



@pytest.mark.parametrize("port,authority", [(443, "sso.example.com"), (8443, "sso.example.com:8443")])
def test_session_oidc_and_admin_reuse_declared_public_url(fresh_manager, port, authority):
    import yaml
    fresh_manager.env_config.set("NGINX_HTTPS_PORT", port)
    authelia = fresh_manager.containers["authelia"]
    expected = "https://" + authority
    assert authelia.public_url == expected
    assert authelia.public_authority == authority
    assert authelia.oidc_client["issuer_url"] == expected
    assert authelia.oidc_client["authorization_url"] == expected + "/api/oidc/authorization"
    config = yaml.safe_load(authelia.render_template(authelia.get_source_path("templates", "configuration.yml")))
    cookie = config["session"]["cookies"][0]
    assert cookie == {"domain": "example.com", "authelia_url": expected,
                      "default_redirection_url": expected + "/settings"}
    environment = authelia.services["authelia-admin"]["environment"]
    assert "TRUSTED_ORIGINS=" + expected in environment
    assert "AAD_AUTHELIA_DOMAIN=" + authority in environment



def test_explicit_public_base_path_keeps_origin_and_cookie_domain_separate(fresh_manager, monkeypatch):
    import yaml
    authelia = fresh_manager.containers["authelia"]
    site = ResolvedSite(authelia, "web", Nginx.site(
        server_name="sso.example.com", public_url="https://login.example.com:8443/auth",
        proxy="http://authelia:9091", auth=False, waf=False))
    sites = dict(fresh_manager.containers["nginx"].sites)
    sites[("authelia", "web")] = site
    monkeypatch.setattr(fresh_manager.containers["nginx"], "sites", sites)
    assert authelia.oidc_client["issuer_url"] == "https://login.example.com:8443/auth"
    config = yaml.safe_load(authelia.render_template(authelia.get_source_path("templates", "configuration.yml")))
    cookie = config["session"]["cookies"][0]
    assert cookie == {"domain": "example.com", "authelia_url": "https://login.example.com:8443/auth",
                      "default_redirection_url": "https://login.example.com:8443/auth/settings"}
    environment = authelia.services["authelia-admin"]["environment"]
    assert "TRUSTED_ORIGINS=https://login.example.com:8443" in environment
    assert "AAD_AUTHELIA_DOMAIN=login.example.com:8443" in environment


def test_unconfigured_public_identity_remains_metadata_only(fresh_manager):
    from linktools.cntr import ContainerError
    fresh_manager.env_config.set("AUTHELIA_DOMAIN", "_")
    authelia = fresh_manager.containers["authelia"]
    assert authelia.public_url == ""
    assert "TRUSTED_ORIGINS=" in authelia.services["authelia-admin"]["environment"]
    with pytest.raises(ContainerError, match="concrete HTTPS"):
        _ = authelia.oidc_client



@pytest.mark.parametrize("server_name", ["_", "~^auth\\.example\\.com$"])
def test_optional_site_url_only_defaults_for_absent_concrete_identity(fresh_manager, server_name):
    from linktools.cntr import ContainerError
    authelia = fresh_manager.containers["authelia"]
    site = ResolvedSite(authelia, "web", Nginx.site(server_name=server_name, proxy="http://app"))
    assert site.get_url(default="") == ""
    with pytest.raises(ContainerError, match="explicit public URL"):
        _ = site.public_url
    explicit = ResolvedSite(authelia, "web", Nginx.site(
        server_name=server_name, public_url="https://login.example.com/auth", proxy="http://app"))
    assert explicit.get_url(default="") == explicit.public_url == "https://login.example.com/auth"
    invalid = ResolvedSite(authelia, "web", Nginx.site(server_name=server_name, public_url=123, proxy="http://app"))
    with pytest.raises(ContainerError, match="public_url must be a string"):
        invalid.get_url(default="")


def test_optional_site_url_does_not_swallow_provider_configuration_errors(fresh_manager):
    from linktools.cntr import ContainerError
    authelia = fresh_manager.containers["authelia"]
    fresh_manager.env_config.set("NGINX_HTTPS_ENABLE", False)
    site = ResolvedSite(authelia, "web", Nginx.site(
        server_name="sso.example.com", https=True, proxy="http://app"))
    with pytest.raises(ContainerError, match="disabled global capability"):
        site.get_url(default="")
