#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.model import ModelResolver, create_model
from linktools.ai.runtime import RuntimeFeatures


@pytest.mark.asyncio
async def test_model_capture_is_lazy_isolated_and_success_is_shared() -> None:
    calls = []
    binding = create_model(protocol="openai", model="fixture")
    def load():
        calls.append("load")
        return binding
    features = RuntimeFeatures()
    features.model("assistant", load)
    captured = await features.capture()
    features.model("assistant", create_model(protocol="anthropic", model="replacement"))
    assert calls == []
    resolved = captured.models.resolve("assistant")
    assert calls == []
    with ThreadPoolExecutor(max_workers=4) as pool:
        identities = list(pool.map(lambda _: resolved.model_identity, range(8)))
    assert identities == ["openai:fixture"] * 8
    assert calls == ["load"]
    assert dict(resolved.contract) == dict(binding.contract)
    assert (await features.capture()).models.resolve("assistant").model_identity == "anthropic:replacement"
    assert calls == ["load"]


@pytest.mark.asyncio
async def test_unused_model_loader_is_not_called() -> None:
    features = RuntimeFeatures()
    def fail():
        pytest.fail("unused loader called")
    features.model("unused", fail)
    await features.capture()


def test_lazy_model_failure_can_be_retried_and_async_result_is_rejected() -> None:
    attempts = []
    def load():
        attempts.append(1)
        if len(attempts) == 1:
            raise AIError(ErrorCode.MODEL_CONFIG_INVALID, "fixture configuration missing")
        return create_model(protocol="openai", model="fixture")
    binding = ModelResolver({"model": load}).resolve("model")
    with pytest.raises(AIError):
        _ = binding.contract
    assert binding.model_identity == "openai:fixture"
    assert len(attempts) == 2
    async def async_load():
        return create_model(protocol="openai", model="fixture")
    with pytest.raises(TypeError):
        ModelResolver({"async": async_load})
    hidden = ModelResolver({"async": lambda: async_load()}).resolve("async")
    with pytest.raises(TypeError):
        _ = hidden.contract


def test_lazy_model_restore_checks_frozen_contract_and_missing_route() -> None:
    binding = create_model(protocol="openai", model="fixture")
    resolver = ModelResolver({"assistant": lambda: binding})
    assert resolver.restore(binding.contract, route_id="assistant").model_identity == binding.model_identity
    with pytest.raises(AIError) as error:
        resolver.restore(dict(binding.contract, model_identity="openai:different"), route_id="assistant")
    assert error.value.code == ErrorCode.AGENT_BINDING_UNAVAILABLE
    with pytest.raises(AIError) as error:
        resolver.resolve("missing")
    assert error.value.code == ErrorCode.MODEL_CONNECTION_NOT_FOUND


def test_recursive_loader_fails_without_deadlocking() -> None:
    resolver = ModelResolver({"loop": lambda: resolver.resolve("loop").contract})
    with pytest.raises(AIError):
        _ = resolver.resolve("loop").contract

@pytest.mark.asyncio
async def test_runtime_open_compiles_routes_without_eager_model_loading() -> None:
    from linktools.ai.runtime import Runtime, RuntimeStorage
    calls = []
    def load():
        calls.append(1)
        return create_model(protocol="openai", model="fixture")
    features = RuntimeFeatures()
    features.model("default", load)
    features.model("unused", lambda: pytest.fail("unused model loaded"))
    features.agent("assistant", model="default")
    async with Runtime.open("lazy-model-test", storage=RuntimeStorage.in_memory(), features=features) as runtime:
        assert calls == []
        binding = runtime.agents.get("assistant").model
        assert calls == []
        assert binding.model_identity == "openai:fixture"
        assert calls == [1]


@pytest.mark.asyncio
async def test_callable_custom_binding_is_not_mistaken_for_a_loader() -> None:
    reads = []
    class CallableBinding:
        @property
        def provider(self):
            reads.append("provider")
            return "custom"
        @property
        def model_identity(self):
            reads.append("identity")
            return "custom:fixture"
        @property
        def contract(self):
            reads.append("contract")
            return {"provider": "custom", "model_identity": "custom:fixture"}
        def materialize(self):
            pytest.fail("client created during capture")
        def __call__(self):
            pytest.fail("binding called as loader")
    binding = CallableBinding()
    features = RuntimeFeatures()
    features.model("custom", binding)
    capture = await features.capture()
    assert reads == []
    assert capture.models.resolve("custom") is binding
    assert capture.models.resolve("custom").contract["provider"] == "custom"
    assert reads == ["contract"]


def test_async_callable_loader_result_is_closed_and_rejected() -> None:
    class Loader:
        async def __call__(self):
            return create_model(protocol="openai", model="fixture")
    binding = ModelResolver({"async": Loader()}).resolve("async")
    with pytest.raises(TypeError):
        _ = binding.contract


@pytest.mark.asyncio
async def test_lazy_semantics_are_cached_but_connections_refresh_per_materialization(monkeypatch) -> None:
    import httpx2
    from linktools.ai.model import _openai
    clients, loads, connections = [], [], []
    def client(**kwargs):
        value = httpx2.AsyncClient(transport=httpx2.MockTransport(lambda request: pytest.fail("unexpected request")), **kwargs)
        clients.append(value)
        return value
    monkeypatch.setattr(_openai, "DefaultAsyncHttpxClient", client)
    def connection():
        connections.append(1)
        return {"api_key": f"fixture-secret-{len(connections)}"}
    def load():
        loads.append(1)
        return create_model(protocol="openai", model="fixture", connection_resolver=connection)
    features = RuntimeFeatures()
    features.model("assistant", load)
    capture = await features.capture()
    binding = capture.models.resolve("assistant")
    contract = dict(binding.contract)
    assert loads == [1] and connections == [] and clients == []
    for _ in range(2):
        async with binding.materialize():
            assert len(loads) == 1
    assert connections == [1, 1]
    assert len(clients) == 2 and all(client.is_closed for client in clients)
    assert dict(binding.contract) == contract
    assert "fixture-secret" not in repr(binding.contract)


def test_cancelled_loader_does_not_poison_first_success() -> None:
    import asyncio
    calls = []
    def load():
        calls.append(1)
        if len(calls) == 1:
            raise asyncio.CancelledError
        return create_model(protocol="openai", model="fixture")
    binding = ModelResolver({"assistant": load}).resolve("assistant")
    with pytest.raises(asyncio.CancelledError):
        _ = binding.contract
    assert binding.model_identity == "openai:fixture"
    assert len(calls) == 2
