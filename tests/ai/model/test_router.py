#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests/ai/model/test_router.py"""
import pytest

from linktools.ai.model.registry import ModelBundle, ModelRegistry, RuntimeModelConfig
from linktools.ai.errors import ModelRoutingError
from linktools.ai.model.policy import ModelPolicy
from linktools.ai.model.router import ModelRouter


def _config(model_type: str) -> RuntimeModelConfig:
    return RuntimeModelConfig(
        model_type=model_type, protocol="openai", model="gpt-4", base_url="http://localhost",
        api_key="test-key", auth_token=None, timeout_seconds=30, raw={},
    )


@pytest.mark.asyncio
async def test_resolve_primary_when_available_async():
    registry = ModelRegistry()
    registry.register("primary-model", config=_config("primary-model"))
    router = ModelRouter(registry=registry)
    bundle = await router.resolve(ModelPolicy(primary="primary-model"))
    assert isinstance(bundle, ModelBundle)
    assert bundle.config.model_type == "primary-model"


@pytest.mark.asyncio
async def test_resolve_falls_back_when_primary_missing():
    registry = ModelRegistry()
    registry.register("fallback-model", config=_config("fallback-model"))
    router = ModelRouter(registry=registry)
    bundle = await router.resolve(ModelPolicy(primary="missing-model", fallbacks=("fallback-model",)))
    assert bundle.config.model_type == "fallback-model"


@pytest.mark.asyncio
async def test_resolve_raises_when_all_exhausted():
    registry = ModelRegistry()
    router = ModelRouter(registry=registry)
    with pytest.raises(ModelRoutingError):
        await router.resolve(ModelPolicy(primary="missing-model", fallbacks=("also-missing",)))


def test_model_policy_defaults():
    policy = ModelPolicy(primary="m")
    assert policy.fallbacks == ()
    assert policy.max_retries == 0
    assert policy.timeout_seconds is None
    assert policy.max_tokens is None
    assert policy.budget is None
