#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Authelia client metadata and derived redirects never mutate saved state."""

import pytest

from linktools.cntr import NginxSite
from linktools.cntr._nginx import ResolvedSite


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


def test_oidc_redirects_derive_only_from_current_sites(fresh_manager, monkeypatch):
    authelia = fresh_manager.containers["authelia"]
    producer = fresh_manager.containers["portainer"]
    site = NginxSite(
        server_name="service.example.com",
        proxy="http://app:8080",
        oidc_redirects=("", "/callback", "https://external.example.com/callback", "/callback"),
    )
    monkeypatch.setattr(fresh_manager.containers["nginx"], "sites", {
        (producer.name, "web"): ResolvedSite(producer, "web", site),
    })
    url = "https://service.example.com"
    assert authelia.oidc_redirects == (
        authelia.oidc_client["issuer_url"],
        url,
        url + "/callback",
        "https://external.example.com/callback",
    )
    assert isinstance(authelia.oidc_client["redirect_uris"], tuple)


def test_acl_supports_native_optional_fields(fresh_manager, monkeypatch):
    authelia = fresh_manager.containers["authelia"]
    producer = fresh_manager.containers["portainer"]
    site = NginxSite(
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
