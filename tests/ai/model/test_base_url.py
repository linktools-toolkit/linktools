#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Spec §19.6: base_url literal pass-through. The legacy auto-append /v1 is
opt-in via base_url_mode=append_v1_if_missing; custom gateway paths and an
explicit /v1 are never corrupted."""

import pytest

from linktools.ai.model.registry import ModelClientUnavailable, RuntimeModelConfig, _resolve_base_url


def _cfg(base_url: str, mode: str = "literal") -> RuntimeModelConfig:
    return RuntimeModelConfig(
        model_type="t", protocol="openai", model="m", base_url=base_url,
        api_key="k", auth_token=None, timeout_seconds=30, raw={},
        base_url_mode=mode,
    )


def test_default_mode_is_literal():
    assert _cfg("https://g.example.com").base_url_mode == "literal"


def test_literal_keeps_explicit_v1():
    assert _resolve_base_url(_cfg("https://gateway.example.com/v1")) == "https://gateway.example.com/v1"


def test_literal_keeps_custom_path():
    assert _resolve_base_url(_cfg("https://gateway.example.com/custom/openai")) == "https://gateway.example.com/custom/openai"


def test_literal_does_not_append_v1():
    assert _resolve_base_url(_cfg("https://gateway.example.com")) == "https://gateway.example.com"


def test_literal_does_not_strip_trailing_slash():
    # No normalization at all in literal mode -- caller owns the exact URL.
    assert _resolve_base_url(_cfg("https://gateway.example.com/v1/")) == "https://gateway.example.com/v1/"


def test_append_v1_is_opt_in():
    url = _resolve_base_url(_cfg("https://gateway.example.com", mode="append_v1_if_missing"))
    assert url == "https://gateway.example.com/v1"


def test_append_v1_idempotent_when_v1_present():
    url = _resolve_base_url(_cfg("https://gateway.example.com/v1", mode="append_v1_if_missing"))
    assert url == "https://gateway.example.com/v1"


def test_append_v1_normalizes_trailing_slash():
    url = _resolve_base_url(_cfg("https://gateway.example.com/", mode="append_v1_if_missing"))
    assert url == "https://gateway.example.com/v1"


def test_missing_base_url_raises():
    with pytest.raises(ModelClientUnavailable, match="requires base_url"):
        _resolve_base_url(_cfg(""))


def test_invalid_mode_raises():
    with pytest.raises(ModelClientUnavailable, match="invalid base_url_mode"):
        _resolve_base_url(RuntimeModelConfig(
            model_type="t", protocol="openai", model="m", base_url="https://x.example.com",
            api_key="k", auth_token=None, timeout_seconds=30, raw={},
            base_url_mode="bogus",  # type: ignore[arg-type]
        ))
