#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Namespaced defaults freeze at declaration and resolve before durable binding."""
from linktools.ai.model import create_model

from dataclasses import dataclass, replace

import pytest
from pydantic_ai.capabilities import AbstractCapability

from linktools.ai.agent import AgentCompiler
from linktools.ai.runtime import RuntimeFeatures

from linktools.ai.spec import AgentSpec


def _compiler(defaults=None, candidates=()):
    return AgentCompiler(
        models={"default": create_model(protocol='openai', model="gpt-test")},
        candidates=candidates, agents={}, defaults=defaults,
    )


@pytest.mark.asyncio
async def test_nested_defaults_are_detached_from_inputs_and_reads():
    original = {"agent": {"tool_retries": 5}, "custom.search": {"options": [1, 2]}}
    features = RuntimeFeatures("defaults", defaults=original)
    original["agent"]["tool_retries"] = 99
    original["custom.search"]["options"].append(3)
    features.defaults["agent"]["tool_retries"] = 88
    capture = await features.capture()
    capture.defaults["custom.search"]["options"].append(4)
    assert dict(capture.defaults) == {
        "agent": {"tool_retries": 5}, "custom.search": {"options": [1, 2]},
    }
    compiled = _compiler(capture.defaults).compile(AgentSpec("agent", output_retries=0))
    assert (compiled.spec.tool_retries, compiled.spec.output_retries) == (5, 0)
    with pytest.raises(AttributeError):
        features.defaults = {}
    assert replace(capture, defaults={"agent": {"tool_retries": 9}}).defaults != capture.defaults


@pytest.mark.parametrize("value", (None, True, -1, 1.5, "5"))
def test_agent_consumer_rejects_invalid_retry_defaults(value):
    with pytest.raises((TypeError, ValueError)):
        _compiler({"agent": {"tool_retries": value}})


def test_agent_consumer_rejects_unknown_keys_without_restricting_custom_namespaces():
    with pytest.raises(TypeError):
        _compiler({"agent": {"tool_retry": 5}})
    compiled = _compiler({"custom.search": {"tool_retry": None}}).compile(AgentSpec("agent"))
    assert compiled.spec.tool_retries == 3


@pytest.mark.parametrize("value", (None, 2, [], "settings"))
def test_namespaces_must_be_objects(value):
    with pytest.raises(TypeError):
        RuntimeFeatures("defaults", defaults={"custom.search": value})


@pytest.mark.asyncio
async def test_custom_consumer_resolves_same_named_setting_into_existing_contract():
    @dataclass(frozen=True)
    class SearchSettings:
        tool_retries: int

    class SearchCapability(AbstractCapability):
        id = "custom.search"

        def __init__(self, settings):
            self.settings = settings

    features = RuntimeFeatures("custom", defaults={
        "agent": {"tool_retries": 5}, "custom.search": {"tool_retries": 10},
    })
    settings = SearchSettings(**features.defaults["custom.search"])
    features.capability(SearchCapability(settings), scope="agent", config={"tool_retries": settings.tool_retries})
    capture = await features.capture()
    compiler = _compiler(capture.defaults, capture.contributions)
    saved = compiler.bind(compiler.compile(AgentSpec("agent"))).binding_contract
    assert saved.agent_spec.tool_retries == 5
    assert capture.contributions[0].value.settings.tool_retries == 10
    assert capture.contributions[0].contract["config"] == {"tool_retries": 10}
    restored = _compiler({"agent": {"tool_retries": 9}}, capture.contributions).restore(saved)
    assert restored.compiled_agent.spec.tool_retries == 5


@pytest.mark.parametrize("name", ("tool_retries", "output_retries"))
def test_removed_retry_keywords_have_no_alias(name):
    with pytest.raises(TypeError):
        RuntimeFeatures("defaults", **{name: 3})
    with pytest.raises(TypeError):
        AgentCompiler(models={"default": create_model(protocol='openai', model="gpt-test")},
                      candidates=(), agents={}, **{name: 3})
