#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests/ai/model/test_router.py"""
import pytest

from linktools.ai.model.registry import (
    ModelBundle,
    ModelClientUnavailable,
    ModelRegistry,
    RuntimeModelConfig,
)
from linktools.ai.errors import ModelRoutingError
from linktools.ai.model.policy import ModelPolicy
from linktools.ai.model.router import ModelRouter


def _config(model_type: str) -> RuntimeModelConfig:
    return RuntimeModelConfig(
        model_type=model_type, protocol="openai", model="gpt-4", base_url="http://localhost",
        api_key="test-key", auth_token=None, timeout_seconds=30, raw={},
    )


class _FlakyRegistry:
    """Wraps a ModelRegistry; ``get()`` raises ModelClientUnavailable the first
    ``fail_times[model_type]`` times for that model_type, then delegates. Records
    per-model_type call counts so tests can assert retry behavior."""

    def __init__(self, inner: ModelRegistry, fail_times: "dict[str, int]") -> None:
        self._inner = inner
        self._remaining = dict(fail_times)
        self.call_counts: "dict[str, int]" = {}

    def get(self, model_type: str) -> ModelBundle:
        self.call_counts[model_type] = self.call_counts.get(model_type, 0) + 1
        remaining = self._remaining.get(model_type, 0)
        if remaining > 0:
            self._remaining[model_type] = remaining - 1
            raise ModelClientUnavailable(f"flaky failure for {model_type}")
        return self._inner.get(model_type)


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


# --- max_retries retry behavior ---------------------------------------------

@pytest.mark.asyncio
async def test_resolve_retries_within_max_retries_then_succeeds():
    """Primary fails twice (transient), succeeds on the 3rd attempt. With
    max_retries=2 the router retries in-place and resolves to the primary."""
    inner = ModelRegistry()
    inner.register("primary-model", config=_config("primary-model"))
    flaky = _FlakyRegistry(inner, fail_times={"primary-model": 2})
    router = ModelRouter(registry=flaky)

    bundle = await router.resolve(ModelPolicy(primary="primary-model", max_retries=2))

    assert bundle.config.model_type == "primary-model"
    # 2 failed attempts + 1 successful attempt = 3 total.
    assert flaky.call_counts["primary-model"] == 3


@pytest.mark.asyncio
async def test_resolve_exhausts_retries_for_each_model_then_uses_fallback():
    """Primary is permanently unavailable; fallback is healthy. With
    max_retries=1 the router attempts primary (1 + 1 retry) = 2 times, then
    fallback once, and resolves to the fallback."""
    inner = ModelRegistry()
    inner.register("fallback-model", config=_config("fallback-model"))
    flaky = _FlakyRegistry(inner, fail_times={"primary-model": 99})
    router = ModelRouter(registry=flaky)

    bundle = await router.resolve(ModelPolicy(
        primary="primary-model", fallbacks=("fallback-model",), max_retries=1,
    ))

    assert bundle.config.model_type == "fallback-model"
    # primary tried (1 + max_retries=1) = 2 times before giving up.
    assert flaky.call_counts["primary-model"] == 2
    # fallback tried once and succeeded.
    assert flaky.call_counts["fallback-model"] == 1


@pytest.mark.asyncio
async def test_resolve_raises_after_retrying_every_model_type():
    """All model_types permanently unavailable; max_retries=2 -> each is tried
    3 times (1 + 2 retries) before the router raises ModelRoutingError."""
    inner = ModelRegistry()
    flaky = _FlakyRegistry(inner, fail_times={"primary-model": 99, "fb-model": 99})
    router = ModelRouter(registry=flaky)

    with pytest.raises(ModelRoutingError):
        await router.resolve(ModelPolicy(
            primary="primary-model", fallbacks=("fb-model",), max_retries=2,
        ))

    assert flaky.call_counts["primary-model"] == 3
    assert flaky.call_counts["fb-model"] == 3


@pytest.mark.asyncio
async def test_resolve_max_retries_zero_preserves_current_behavior():
    """max_retries=0 (default) means exactly one attempt per model_type -- no
    retries. Confirms the new code path preserves the baseline behavior."""
    inner = ModelRegistry()
    inner.register("fallback-model", config=_config("fallback-model"))
    flaky = _FlakyRegistry(inner, fail_times={"primary-model": 1})
    router = ModelRouter(registry=flaky)

    bundle = await router.resolve(ModelPolicy(
        primary="primary-model", fallbacks=("fallback-model",), max_retries=0,
    ))

    # primary attempted once (failed), fallback once (succeeded).
    assert bundle.config.model_type == "fallback-model"
    assert flaky.call_counts["primary-model"] == 1
    assert flaky.call_counts["fallback-model"] == 1
