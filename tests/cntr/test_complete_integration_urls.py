#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Consumers omit incomplete resolved URLs before emitting configuration."""
import pytest
import yaml

from linktools.cntr.ext import Authelia, Flare
from linktools.core import ConfigLazy, ConfigRef
from test_config_values_navigation import _Config, _manager, _navigation, _fail


@pytest.mark.parametrize("url", ["", "/relative", "{url}/callback", "//host.test/path", "https:///path",
                                "https://:443/path", "https://user@/path", "https://host.test:bad/path",
                                "https://host.test:65536/path", "https://host.test:0/path", "https://[invalid"])
def test_flare_omits_incomplete_resolved_urls_before_metadata(url: str) -> None:
    config = _Config(URL=url)
    link = Flare.public(ConfigLazy(_fail), ConfigLazy(_fail), url=ConfigRef("URL"))
    manager = _manager(("app", config, [link]), nginx=False)
    assert config.reads == []
    assert _navigation(manager)["links"] == []
    assert config.reads == ["URL"]


@pytest.mark.parametrize("url", ["http://private.test:65535/ui#section", "https://external.test/path?q=%2F", "custom:callback"])
def test_flare_keeps_complete_independent_urls_without_nginx(url: str) -> None:
    config = _Config(URL=url)
    manager = _manager(("app", config, [Flare.public("App", "web", url=ConfigRef("URL"))]), nginx=False)
    assert _navigation(manager)["links"][0]["link"] == url


def test_authelia_written_config_uses_only_complete_resolved_callbacks(fresh_manager, monkeypatch) -> None:
    provider = fresh_manager.containers["authelia"]
    producer = fresh_manager.containers["portainer"]
    provider.public_url = "https://sso.test"
    producer.env_config.set("CALLBACK_URL", "http://private.test:65535/cb?q=%2F")
    callbacks = Authelia.oidc(("/callback", "{url}", "//host.test/cb", "https:///cb", "",
                              ConfigRef("CALLBACK_URL"), "custom:callback"))
    monkeypatch.setattr(fresh_manager, "iter_integrations", lambda consumer: iter(((producer, callbacks),)))
    expected = ("https://sso.test", "http://private.test:65535/cb?q=%2F", "custom:callback")
    assert provider.oidc_redirects == expected
    rendered = provider.render_template(provider.get_source_path("templates", "configuration.oidc.yml"))
    # Authelia evaluates its native secret expression after the Jinja render.
    rendered = "\n".join(line for line in rendered.splitlines() if "- key: {{ secret" not in line)
    written = yaml.safe_load(rendered)
    assert tuple(written["identity_providers"]["oidc"]["clients"][0]["redirect_uris"]) == expected


def test_authelia_accepted_redirects_do_not_read_client_secret(fresh_manager, monkeypatch) -> None:
    provider = fresh_manager.containers["authelia"]
    provider.public_url = "https://sso.test"
    original = provider.get_config

    def get_config(key: object, **kwargs: object) -> object:
        assert key != "AUTHELIA_OIDC_CLIENT_SECRET"
        return original(key, **kwargs)

    monkeypatch.setattr(provider, "get_config", get_config)
    assert provider.oidc_redirects[0] == "https://sso.test"
    assert "_oidc_identity" not in provider.__dict__


@pytest.mark.parametrize("url", ["https://:443/cb", "https://user@/cb", "https://host.test:bad/cb",
                                "https://host.test:65536/cb", "https://host.test:0/cb", "https://[invalid"])
def test_authelia_omits_incomplete_http_targets(url: str) -> None:
    config = _Config(URL=url)
    callbacks = Authelia.oidc((ConfigRef("URL"), "https://host.test:65535/cb", "custom:callback"))
    assert callbacks.resolve_redirect_uris(config) == ("https://host.test:65535/cb", "custom:callback")


@pytest.mark.parametrize("issuer", ["https://:443", "https://sso.test:65536"])
def test_authelia_issuer_cannot_bypass_complete_url_filter(fresh_manager, monkeypatch, issuer: str) -> None:
    from linktools.cntr import ContainerError
    provider = fresh_manager.containers["authelia"]
    provider.public_url = issuer
    monkeypatch.setattr(fresh_manager, "iter_integrations", lambda consumer: iter(()))
    assert provider.oidc_redirects == ()
    with pytest.raises(ContainerError, match="concrete HTTPS"):
        _ = provider.oidc_client
