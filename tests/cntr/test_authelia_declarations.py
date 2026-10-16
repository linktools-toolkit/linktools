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
        "https://app.test/base?x=1", "https://app.test/callback?q=2", "custom://app.test/callback",
        "https://app.test/base?x=1", "", "http://local.test/", "https://app.test",
    )))
    assert isinstance(declaration, Integration)
    assert declaration.consumer == "authelia"
    assert calls == []
    assert declaration.redirect_uris == (
        "https://app.test/base?x=1", "https://app.test/callback?q=2", "custom://app.test/callback",
        "http://local.test/", "https://app.test",
    )
    assert calls == [True]
    assert declaration.redirect_uris[-1] == "https://app.test"
    assert calls == [True]


def test_oidc_has_no_enable_switch():
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        Authelia.oidc(("https://app.test/callback",), enabled=False)


@pytest.mark.parametrize("value", ["https://a.test/#bad"])
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


@pytest.mark.parametrize("value", ["", "//evil.test/path", "/callback", "/cb#", "callback", "{url}", "https:///callback"])
def test_incomplete_oidc_callbacks_are_omitted_without_nginx(value: str) -> None:
    assert Authelia.oidc((value, "http://local.test/callback")).redirect_uris == ("http://local.test/callback",)
