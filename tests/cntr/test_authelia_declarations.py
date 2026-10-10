#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""OIDC callbacks are independent lazy declarations for the existing client."""
import pytest

from linktools.cntr.ext import Authelia, Integration
from linktools.cntr.errors import ContainerError
from linktools.runtime import lazy_load


def test_oidc_factory_is_lazy_and_preserves_exact_callback_identity():
    calls = []
    declaration = Authelia.oidc(lazy_load(lambda: calls.append(True) or (
        "https://app.test/base?x=1", "https://app.test/callback?q=2", "custom:callback",
        "https://app.test/base?x=1", "", "http://local.test/", "https://app.test",
    )))
    assert isinstance(declaration, Integration)
    assert declaration.consumer == "authelia"
    assert calls == []
    assert declaration.redirect_uris == (
        "https://app.test/base?x=1", "https://app.test/callback?q=2", "custom:callback",
        "http://local.test/", "https://app.test",
    )
    assert calls == [True]
    assert declaration.redirect_uris[-1] == "https://app.test"
    assert calls == [True]


def test_disabled_oidc_declaration_never_resolves_callbacks():
    def fail():
        raise AssertionError("disabled callback evaluated")
    declaration = Authelia.oidc(lazy_load(fail), enabled=lazy_load(lambda: False))
    assert declaration.redirect_uris == ()


@pytest.mark.parametrize("value", ["//evil.test/path", "/callback", "/cb#", "https://a.test/#bad",
                                   "callback", "https://app:{{port}}/callback", "https:///callback"])
def test_invalid_oidc_callbacks_fail_without_nginx(value):
    with pytest.raises(ContainerError, match="Authelia redirect URI"):
        Authelia.oidc((value,)).redirect_uris


def test_independent_oidc_declarations_keep_separate_inputs():
    values = ["https://first.test/callback"]
    first = Authelia.oidc(values)
    values.append("https://later.test")
    second = Authelia.oidc(("https://second.test/callback",))
    assert first.redirect_uris == ("https://first.test/callback",)
    assert second.redirect_uris == ("https://second.test/callback",)
