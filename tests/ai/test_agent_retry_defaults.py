#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Retry defaults resolve once into the durable Agent binding."""
from ._runtime_test_helpers import model_features
from linktools.ai.model import create_model

from collections.abc import Mapping
from dataclasses import replace

import pytest

from linktools.ai.agent import AgentBindingContract, AgentCompiler
from linktools.ai.runtime import RuntimeFeatures
from linktools.ai.core import ImmutableJsonMapping, JsonValue
from linktools.ai.errors import AIError, ErrorCode

from linktools.ai.spec import DEFAULT_AGENT_RETRIES, AgentSpec, AgentSpecCodec


def _compiler(*, defaults: Mapping[str, JsonValue] | None = None) -> AgentCompiler:
    return AgentCompiler(
        models={"default": create_model(protocol='openai', model="gpt-test")},
        candidates=(), agents={}, defaults=defaults,
    )


@pytest.mark.parametrize("fields", ({}, {"tool_retries": None, "output_retries": None}))
def test_omitted_and_none_retry_declarations_inherit(fields: dict[str, object]) -> None:
    spec = AgentSpecCodec().from_payload({"version": 1, "id": "agent", **fields})
    assert spec.tool_retries is None and spec.output_retries is None
    assert AgentSpecCodec().decode(AgentSpecCodec().encode(spec)) == spec
    compiler = _compiler(defaults={"agent": {"tool_retries": 7, "output_retries": 5}})
    binding = compiler.bind(compiler.compile(spec))
    assert binding.binding_contract.agent_spec.tool_retries == 7
    assert binding.binding_contract.agent_spec.output_retries == 5


@pytest.mark.parametrize("value", (0, 2))
def test_explicit_retry_limits_override_runtime_defaults(value: int) -> None:
    spec = AgentSpec("agent", tool_retries=value, output_retries=value)
    compiled = _compiler(defaults={"agent": {"tool_retries": 7, "output_retries": 5}}).compile(spec)
    assert (compiled.spec.tool_retries, compiled.spec.output_retries) == (value, value)


def test_direct_compilation_resolves_default_retry_limits() -> None:
    compiler = AgentCompiler(
        models={"default": create_model(protocol='openai', model="gpt-test")}, candidates=(), agents={},
    )
    compiled = compiler.compile(AgentSpec("agent"))
    assert (compiled.spec.tool_retries, compiled.spec.output_retries) == (3, 3)


@pytest.mark.asyncio
async def test_feature_capture_freezes_retry_defaults() -> None:
    agent_defaults = {"tool_retries": 7, "output_retries": 5}
    features = RuntimeFeatures("retries", defaults={"agent": agent_defaults})
    declared = features.agent("agent")
    capture = await features.capture()
    assert declared.tool_retries is None and declared.output_retries is None
    agent_defaults.update(tool_retries=9, output_retries=8)
    assert isinstance(capture.defaults, ImmutableJsonMapping)
    assert capture.defaults["agent"] == {"tool_retries": 7, "output_retries": 5}
    compiler = _compiler(defaults=capture.defaults)
    saved = compiler.bind(compiler.compile(declared)).binding_contract
    restored_wire = AgentBindingContract.from_payload(saved.to_payload())
    reconfigured = _compiler(defaults={"agent": {"tool_retries": 9, "output_retries": 8}})
    restored = reconfigured.restore(restored_wire)
    assert (restored.compiled_agent.spec.tool_retries, restored.compiled_agent.spec.output_retries) == (7, 5)
    assert restored.binding_contract.to_payload() == saved.to_payload()
    fresh = reconfigured.compile(declared)
    assert (fresh.spec.tool_retries, fresh.spec.output_retries) == (9, 8)


def test_unresolved_retry_limits_cannot_enter_durable_binding() -> None:
    compiler = _compiler()
    saved = compiler.bind(compiler.compile(AgentSpec("agent"))).binding_contract
    with pytest.raises(AIError) as caught:
        replace(saved, agent_spec=AgentSpec("agent"))
    assert caught.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


@pytest.mark.asyncio
async def test_runtime_reconfiguration_does_not_change_restored_retry_limits() -> None:
    from linktools.ai.runtime import Runtime, RuntimeStorage

    first = RuntimeFeatures("retries", defaults={"agent": {"tool_retries": 6, "output_retries": 4}})
    first.agent("agent")
    first.agent("disabled", tool_retries=0, output_retries=0)
    models = create_model(protocol='openai', model="gpt-test")
    async with Runtime.open(
        "retry-binding", storage=RuntimeStorage.in_memory(), features=model_features(models, first),
    ) as runtime:
        compiled = runtime._catalog.root_agent("agent")
        assert (compiled.spec.tool_retries, compiled.spec.output_retries) == (6, 4)
        disabled = runtime._catalog.root_agent("disabled")
        assert (disabled.spec.tool_retries, disabled.spec.output_retries) == (0, 0)
        payload = runtime._compiler.bind(compiled).binding_contract.to_payload()
    second = RuntimeFeatures("retries", defaults={"agent": {"tool_retries": 9, "output_retries": 8}})
    second.agent("agent")
    async with Runtime.open(
        "retry-binding", storage=RuntimeStorage.in_memory(), features=model_features(models, second),
    ) as runtime:
        fresh = runtime._catalog.root_agent("agent")
        assert (fresh.spec.tool_retries, fresh.spec.output_retries) == (9, 8)
        restored = runtime._compiler.restore(AgentBindingContract.from_payload(payload))
        assert (restored.compiled_agent.spec.tool_retries, restored.compiled_agent.spec.output_retries) == (6, 4)
        assert restored.binding_contract.to_payload() == payload


@pytest.mark.asyncio
async def test_runtime_and_direct_agent_resolution_share_one_default() -> None:
    assert DEFAULT_AGENT_RETRIES == 3
    features = RuntimeFeatures("shared-default")
    capture = await features.capture()
    compiler = AgentCompiler(
        models={"default": create_model(protocol='openai', model="gpt-test")}, candidates=(), agents={},
    )
    compiled = compiler.compile(AgentSpec("agent"))
    assert capture.defaults == {}
    assert compiled.spec.tool_retries == compiled.spec.output_retries == DEFAULT_AGENT_RETRIES
