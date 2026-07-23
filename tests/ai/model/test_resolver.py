#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ModelResolver: resolve a ModelPolicy to a ResolvedModel.

Covers the candidate-walk contract (single vs FallbackModel, unregistered-skip,
no-candidate error), the revision contract (secret-excluding, order-sensitive,
request_retries-sensitive), and the request_retries wiring (rebuilt into the
provider HTTP client only for config-backed OpenAI models; prebuilt models are
reused as-is)."""
from typing import Any
from unittest.mock import MagicMock

import pytest
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel

from linktools.ai.errors import ModelRoutingError
from linktools.ai.model import resolver as resolver_module
from linktools.ai.model.policy import ModelPolicy
from linktools.ai.model.registry import ModelRegistry, RuntimeModelConfig
from linktools.ai.model.resolver import ModelResolver, ResolvedModel


def _fn(text: str = "ok"):
    def _f(messages, info: AgentInfo):
        from pydantic_ai.messages import ModelResponse, TextPart

        return ModelResponse(parts=[TextPart(content=text)])

    return _f


def _prebuilt_registry(*model_types: str) -> ModelRegistry:
    registry = ModelRegistry()
    for i, mt in enumerate(model_types):
        registry.register(mt, model=FunctionModel(_fn(str(i))))
    return registry


def _config(model_type: str, *, base_url: str, api_key: str) -> RuntimeModelConfig:
    return RuntimeModelConfig(
        model_type=model_type,
        protocol="openai",
        model=model_type,
        base_url=base_url,
        api_key=api_key,
        auth_token=None,
        timeout_seconds=300,
        raw={},
    )


def test_single_registered_candidate_used_directly():
    registry = _prebuilt_registry("only")
    resolved = ModelResolver(registry=registry).resolve(ModelPolicy(primary="only"))
    assert isinstance(resolved, ResolvedModel)
    # A single candidate is injected directly, not wrapped in FallbackModel.
    assert isinstance(resolved.model, FunctionModel)
    assert resolved.model is registry.get("only").model


def test_multiple_candidates_wrap_in_fallback_model():
    from pydantic_ai.models.fallback import FallbackModel

    registry = _prebuilt_registry("primary", "fb1", "fb2")
    resolved = ModelResolver(registry=registry).resolve(
        ModelPolicy(primary="primary", fallbacks=("fb1", "fb2"))
    )
    assert isinstance(resolved.model, FallbackModel)


def test_unregistered_primary_falls_through_to_registered_fallback():
    registry = _prebuilt_registry("fb")
    resolved = ModelResolver(registry=registry).resolve(
        ModelPolicy(primary="missing", fallbacks=("fb",))
    )
    assert isinstance(resolved.model, FunctionModel)
    assert resolved.model is registry.get("fb").model


def test_all_candidates_unregistered_raises_routing_error():
    registry = _prebuilt_registry("unrelated")
    with pytest.raises(ModelRoutingError):
        ModelResolver(registry=registry).resolve(
            ModelPolicy(primary="missing", fallbacks=("also-missing",))
        )


def test_revision_excludes_api_key():
    registry = ModelRegistry()
    registry.register("m", config=_config("m", base_url="http://x", api_key="secret-a"))
    rev_a = ModelResolver(registry=registry).resolve(
        ModelPolicy(primary="m")
    ).revision
    registry.register("m", config=_config("m", base_url="http://x", api_key="secret-b"))
    rev_b = ModelResolver(registry=registry).resolve(
        ModelPolicy(primary="m")
    ).revision
    # Rotating the api key is NOT a revision change.
    assert rev_a == rev_b


def test_revision_changes_on_endpoint_change():
    registry = ModelRegistry()
    registry.register("m", config=_config("m", base_url="http://x", api_key="k"))
    rev_a = ModelResolver(registry=registry).resolve(ModelPolicy(primary="m")).revision
    registry.register("m", config=_config("m", base_url="http://y", api_key="k"))
    rev_b = ModelResolver(registry=registry).resolve(ModelPolicy(primary="m")).revision
    assert rev_a != rev_b


def test_revision_changes_on_candidate_reorder():
    registry = _prebuilt_registry("a", "b")
    rev_ab = ModelResolver(registry=registry).resolve(
        ModelPolicy(primary="a", fallbacks=("b",))
    ).revision
    rev_ba = ModelResolver(registry=registry).resolve(
        ModelPolicy(primary="b", fallbacks=("a",))
    ).revision
    assert rev_ab != rev_ba


def test_revision_changes_on_request_retries():
    registry = _prebuilt_registry("m")
    rev_zero = ModelResolver(registry=registry).resolve(
        ModelPolicy(primary="m", request_retries=0)
    ).revision
    rev_three = ModelResolver(registry=registry).resolve(
        ModelPolicy(primary="m", request_retries=3)
    ).revision
    assert rev_zero != rev_three


def test_request_retries_rebuilds_config_backed_model(monkeypatch):
    # A config-backed (openai-protocol) model with request_retries>0 is rebuilt
    # so the retry count is wired into the provider HTTP client.
    registry = ModelRegistry()
    registry.register("m", config=_config("m", base_url="http://x", api_key="k"))
    captured: "list[int]" = []
    real = resolver_module.__dict__  # noqa: F841 - for clarity below

    import linktools.ai.model.registry as registry_module

    original = registry_module._bundle_from_config

    def _spy(config: RuntimeModelConfig, *, request_retries: int = 0) -> Any:
        captured.append(request_retries)
        return original(config, request_retries=request_retries)

    monkeypatch.setattr(registry_module, "_bundle_from_config", _spy)
    ModelResolver(registry=registry).resolve(
        ModelPolicy(primary="m", request_retries=4)
    )
    assert 4 in captured


def test_zero_request_retries_reuses_bundle_model():
    # request_retries==0 matches the registration-time build, so the existing
    # bundle.model is reused without a rebuild.
    registry = ModelRegistry()
    registry.register("m", config=_config("m", base_url="http://x", api_key="k"))
    bundle_model = registry.get("m").model
    resolved = ModelResolver(registry=registry).resolve(
        ModelPolicy(primary="m", request_retries=0)
    )
    assert resolved.model is bundle_model


def test_prebuilt_model_reused_regardless_of_request_retries():
    # A prebuilt model has no HTTP client to configure, so request_retries is
    # ignored and the registered model is reused as-is.
    registry = _prebuilt_registry("m")
    bundle_model = registry.get("m").model
    resolved = ModelResolver(registry=registry).resolve(
        ModelPolicy(primary="m", request_retries=9)
    )
    assert resolved.model is bundle_model
