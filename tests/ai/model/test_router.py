#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ModelResolver (single-type config resolution) and ModelGateway (retry +
fallback policy) -- the two concerns separates."""

import pytest

from linktools.ai.model.registry import (
    ModelBundle,
    ModelClientUnavailable,
    ModelRegistry,
    RuntimeModelConfig,
)
from linktools.ai.errors import ModelRoutingError
from linktools.ai.model.policy import ModelPolicy
from linktools.ai.model.router import ModelGateway, ModelResolver


def _config(model_type: str) -> RuntimeModelConfig:
    return RuntimeModelConfig(
        model_type=model_type,
        protocol="openai",
        model="gpt-4",
        base_url="http://localhost",
        api_key="test-key",
        auth_token=None,
        timeout_seconds=30,
        raw={},
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


# --- ModelResolver: single-type config resolution, no retry/fallback --------


@pytest.mark.asyncio
async def test_resolver_resolves_a_registered_model_type():
    registry = ModelRegistry()
    registry.register("primary-model", config=_config("primary-model"))
    resolver = ModelResolver(registry=registry)
    bundle = await resolver.resolve("primary-model")
    assert isinstance(bundle, ModelBundle)
    assert bundle.config.model_type == "primary-model"


@pytest.mark.asyncio
async def test_resolver_raises_when_type_unregistered_no_fallback():
    """ModelResolver resolves ONE type only: it does not walk fallbacks (that is
    the gateway's job) -- an unregistered type raises ModelClientUnavailable."""
    registry = ModelRegistry()
    registry.register("fallback-model", config=_config("fallback-model"))
    resolver = ModelResolver(registry=registry)
    with pytest.raises(ModelClientUnavailable):
        await resolver.resolve("missing-model")


# --- ModelGateway: primary/fallback walk + per-candidate retry --------------


@pytest.mark.asyncio
async def test_gateway_resolves_primary_when_available():
    registry = ModelRegistry()
    registry.register("primary-model", config=_config("primary-model"))
    gateway = ModelGateway(ModelResolver(registry=registry))
    bundle = await gateway.resolve(ModelPolicy(primary="primary-model"))
    assert bundle.config.model_type == "primary-model"


@pytest.mark.asyncio
async def test_gateway_falls_back_when_primary_missing():
    registry = ModelRegistry()
    registry.register("fallback-model", config=_config("fallback-model"))
    gateway = ModelGateway(ModelResolver(registry=registry))
    bundle = await gateway.resolve(
        ModelPolicy(primary="missing-model", fallbacks=("fallback-model",))
    )
    assert bundle.config.model_type == "fallback-model"


@pytest.mark.asyncio
async def test_gateway_raises_when_all_exhausted():
    registry = ModelRegistry()
    gateway = ModelGateway(ModelResolver(registry=registry))
    with pytest.raises(ModelRoutingError):
        await gateway.resolve(
            ModelPolicy(primary="missing-model", fallbacks=("also-missing",))
        )


@pytest.mark.asyncio
async def test_gateway_retries_within_max_retries_then_succeeds():
    """Primary fails twice (transient), succeeds on the 3rd attempt. With
    max_retries=2 the gateway retries in-place and resolves to the primary."""
    inner = ModelRegistry()
    inner.register("primary-model", config=_config("primary-model"))
    flaky = _FlakyRegistry(inner, fail_times={"primary-model": 2})
    gateway = ModelGateway(ModelResolver(registry=flaky))

    bundle = await gateway.resolve(ModelPolicy(primary="primary-model", max_retries=2))

    assert bundle.config.model_type == "primary-model"
    # 2 failed attempts + 1 successful attempt = 3 total.
    assert flaky.call_counts["primary-model"] == 3


@pytest.mark.asyncio
async def test_gateway_exhausts_retries_for_each_model_then_uses_fallback():
    """Primary permanently unavailable; fallback healthy. With max_retries=1 the
    gateway attempts primary (1 + 1 retry) = 2 times, then fallback once."""
    inner = ModelRegistry()
    inner.register("fallback-model", config=_config("fallback-model"))
    flaky = _FlakyRegistry(inner, fail_times={"primary-model": 99})
    gateway = ModelGateway(ModelResolver(registry=flaky))

    bundle = await gateway.resolve(
        ModelPolicy(
            primary="primary-model",
            fallbacks=("fallback-model",),
            max_retries=1,
        )
    )

    assert bundle.config.model_type == "fallback-model"
    assert flaky.call_counts["primary-model"] == 2
    assert flaky.call_counts["fallback-model"] == 1


@pytest.mark.asyncio
async def test_gateway_raises_after_retrying_every_model_type():
    """All model_types permanently unavailable; max_retries=2 -> each is tried
    3 times (1 + 2 retries) before the gateway raises ModelRoutingError."""
    inner = ModelRegistry()
    flaky = _FlakyRegistry(inner, fail_times={"primary-model": 99, "fb-model": 99})
    gateway = ModelGateway(ModelResolver(registry=flaky))

    with pytest.raises(ModelRoutingError):
        await gateway.resolve(
            ModelPolicy(
                primary="primary-model",
                fallbacks=("fb-model",),
                max_retries=2,
            )
        )

    assert flaky.call_counts["primary-model"] == 3
    assert flaky.call_counts["fb-model"] == 3


@pytest.mark.asyncio
async def test_gateway_max_retries_zero_is_one_attempt_per_candidate():
    """max_retries=0 (default) means exactly one attempt per candidate -- no
    retries, matching the pre-split baseline."""
    inner = ModelRegistry()
    inner.register("fallback-model", config=_config("fallback-model"))
    flaky = _FlakyRegistry(inner, fail_times={"primary-model": 1})
    gateway = ModelGateway(ModelResolver(registry=flaky))

    bundle = await gateway.resolve(
        ModelPolicy(
            primary="primary-model",
            fallbacks=("fallback-model",),
            max_retries=0,
        )
    )

    assert bundle.config.model_type == "fallback-model"
    assert flaky.call_counts["primary-model"] == 1
    assert flaky.call_counts["fallback-model"] == 1


def test_model_policy_defaults():
    policy = ModelPolicy(primary="m")
    assert policy.fallbacks == ()
    assert policy.max_retries == 0
    assert policy.timeout_seconds is None
    assert policy.max_tokens is None
    assert policy.budget is None
