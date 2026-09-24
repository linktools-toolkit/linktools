#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Model binding must not expose credential material."""

import pytest

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.model import ModelRegistry
from linktools.ai.model._openai import _OpenAIModelBinding, _resolved_connection


def test_model_binding_model_digest_excludes_secret_material() -> None:
    registry = ModelRegistry.openai(model="gpt-test", api_key="secret")
    binding = registry.snapshot().resolve("default")

    assert "secret" not in repr(binding)
    assert "secret" not in binding.model_digest
    assert binding.model_identity == "openai:gpt-test"


def test_openai_connection_resolver_is_lazy_and_overrides_defaults() -> None:
    calls: list[str] = []

    def resolve() -> dict[str, object]:
        calls.append("called")
        return {"api_key": "resolved", "max_retries": 4}

    binding = _OpenAIModelBinding(
        "route",
        "gpt-test",
        api_key="default",
        connection_resolver=resolve,
    )

    assert calls == []
    values = _resolved_connection(binding)
    assert values["api_key"] == "resolved"
    assert values["max_retries"] == 4
    assert calls == ["called"]


def test_openai_connection_resolver_rejects_model_fields() -> None:
    binding = _OpenAIModelBinding(
        "route",
        "gpt-test",
        connection_resolver=lambda: {"model": "other"},
    )

    with pytest.raises(AIError) as raised:
        _resolved_connection(binding)
    assert raised.value.code is ErrorCode.MODEL_CONFIG_INVALID


def test_openai_connection_resolver_normalizes_connection_values() -> None:
    binding = _OpenAIModelBinding(
        "route",
        "gpt-test",
        connection_resolver=lambda: {
            "base_url": "  https://EXAMPLE.com/v1/  ",
            "api_key": "   ",
        },
    )
    values = _resolved_connection(binding)
    assert values["base_url"] == "https://example.com/v1"
    assert values["api_key"] is None


def test_openai_connection_resolver_invalid_values_keep_a_cause() -> None:
    binding = _OpenAIModelBinding(
        "route",
        "gpt-test",
        connection_resolver=lambda: {"max_retries": None},
    )

    with pytest.raises(AIError) as raised:
        _resolved_connection(binding)
    assert raised.value.code is ErrorCode.MODEL_CONFIG_INVALID
    assert raised.value.__cause__ is not None


def test_model_alias_resolves_to_an_immutable_binding() -> None:
    registry = ModelRegistry()
    registry.register_openai("target", model="gpt-old")
    registry.register_alias("alias", "target")
    registry.register_alias("alias-2", "alias")
    registry.register_openai("target", model="gpt-new")

    snapshot = registry.snapshot()
    assert snapshot.resolve("alias").model_identity == "openai:gpt-old"
    assert snapshot.resolve("alias-2").model_identity == "openai:gpt-old"
    assert snapshot.resolve("target").model_identity == "openai:gpt-new"
    with pytest.raises(AttributeError):
        setattr(snapshot.resolve("alias"), "_target", snapshot.resolve("target"))


def test_model_alias_requires_an_existing_target() -> None:
    with pytest.raises(AIError) as raised:
        ModelRegistry().register_alias("alias", "missing")
    assert raised.value.code is ErrorCode.MODEL_CONNECTION_NOT_FOUND
