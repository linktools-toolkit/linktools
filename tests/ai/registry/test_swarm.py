#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for registry/swarm.py: SwarmRegistry resolves SwarmSpec from {name}.yaml
files via SpecLoader, revision-cached."""

import asyncio

import pytest

from linktools.ai.errors import (
    InvalidSpecError,
    RegistryNotFoundError,
    RegistryParseError,
)
from linktools.ai.registry.parser import SpecLoader
from linktools.ai.registry.swarm import SwarmRegistry, parse_swarm_spec
from linktools.ai.swarm.aggregation import AggregationMode
from linktools.ai.swarm.limits import DEFAULT_SWARM_LIMITS
from linktools.ai.swarm.spec import (
    AgentRef,
    SwarmContextPolicy,
    SwarmSpec,
    SwarmStrategySpec,
)


def _write_research(tmp_path) -> None:
    """Write swarms/research.yaml — full fixture exercising every field."""
    swarms = tmp_path / "swarms"
    swarms.mkdir()
    (swarms / "research.yaml").write_text(
        "name: research\n"
        "agents:\n"
        "  - agent_id: searcher\n"
        "  - agent_id: writer\n"
        "    role: synthesizer\n"
        "coordinator:\n"
        "  agent_id: planner\n"
        "strategy:\n"
        "  kind: parallel_fan_out\n"
        "  config:\n"
        "    task_count: 2\n"
        "limits:\n"
        "  max_rounds: 5\n"
        "  max_tasks: 20\n"
        "aggregation: concat\n",
        encoding="utf-8",
    )


def _write_minimal(tmp_path) -> None:
    """Write swarms/minimal.yaml — only agents + coordinator, exercising defaults."""
    swarms = tmp_path / "swarms"
    swarms.mkdir()
    (swarms / "minimal.yaml").write_text(
        "agents:\n"
        "  - agent_id: worker\n"
        "coordinator:\n"
        "  agent_id: boss\n",
        encoding="utf-8",
    )


# 1. get() parses a full YAML into a SwarmSpec with all fields populated.
def test_get_returns_swarm_spec_from_yaml(tmp_path):
    _write_research(tmp_path)
    registry = SwarmRegistry(SpecLoader.from_filesystem(tmp_path / "swarms"))

    async def run():
        return await registry.get("research")

    spec = asyncio.run(run())
    assert isinstance(spec, SwarmSpec)
    assert spec.id == "research"
    assert spec.name == "research"
    # agents
    assert len(spec.agents) == 2
    assert all(isinstance(a, AgentRef) for a in spec.agents)
    assert spec.agents[0].agent_id == "searcher"
    assert spec.agents[0].role is None
    assert spec.agents[1].agent_id == "writer"
    assert spec.agents[1].role == "synthesizer"
    # coordinator
    assert isinstance(spec.coordinator, AgentRef)
    assert spec.coordinator.agent_id == "planner"
    # strategy
    assert isinstance(spec.strategy, SwarmStrategySpec)
    assert spec.strategy.kind == "parallel_fan_out"
    assert dict(spec.strategy.config) == {"task_count": 2}
    # limits
    assert spec.limits.max_rounds == 5
    assert spec.limits.max_tasks == 20
    # aggregation
    assert spec.aggregation.mode == AggregationMode.CONCAT


# 2. Defaults: agents + coordinator only → coordinator_delegation, DEFAULT_SWARM_LIMITS,
#    CONCAT aggregation, default SwarmContextPolicy.
def test_get_applies_defaults_when_minimal(tmp_path):
    _write_minimal(tmp_path)
    registry = SwarmRegistry(SpecLoader.from_filesystem(tmp_path / "swarms"))

    async def run():
        return await registry.get("minimal")

    spec = asyncio.run(run())
    assert spec.name == "minimal"  # name defaults to the swarm_id
    assert spec.strategy.kind == "coordinator_delegation"
    assert dict(spec.strategy.config) == {}
    assert spec.limits == DEFAULT_SWARM_LIMITS
    assert spec.aggregation.mode == AggregationMode.CONCAT
    assert isinstance(spec.context_policy, SwarmContextPolicy)
    assert spec.context_policy == SwarmContextPolicy()
    assert spec.middleware == ()
    assert dict(spec.metadata) == {}


# 3. list_ids() returns every swarm id under the loader root.
def test_list_ids_returns_all_swarm_ids(tmp_path):
    _write_research(tmp_path)
    registry = SwarmRegistry(SpecLoader.from_filesystem(tmp_path / "swarms"))

    async def run():
        return await registry.list_ids()

    ids = asyncio.run(run())
    assert "research" in ids


# 4a. Missing 'agents' → InvalidSpecError.
def test_get_missing_agents_raises_invalid_spec(tmp_path):
    swarms = tmp_path / "swarms"
    swarms.mkdir()
    (swarms / "noagents.yaml").write_text(
        "name: noagents\ncoordinator:\n  agent_id: boss\n",
        encoding="utf-8",
    )
    registry = SwarmRegistry(SpecLoader.from_filesystem(swarms))

    async def run():
        await registry.get("noagents")

    with pytest.raises(InvalidSpecError):
        asyncio.run(run())


# 4b. Missing 'coordinator' → InvalidSpecError.
def test_get_missing_coordinator_raises_invalid_spec(tmp_path):
    swarms = tmp_path / "swarms"
    swarms.mkdir()
    (swarms / "nocoord.yaml").write_text(
        "name: nocoord\nagents:\n  - agent_id: worker\n",
        encoding="utf-8",
    )
    registry = SwarmRegistry(SpecLoader.from_filesystem(swarms))

    async def run():
        await registry.get("nocoord")

    with pytest.raises(InvalidSpecError):
        asyncio.run(run())


# 5a. Missing swarm file → RegistryNotFoundError (propagated from the loader).
def test_get_missing_swarm_raises_not_found(tmp_path):
    _write_research(tmp_path)
    registry = SwarmRegistry(SpecLoader.from_filesystem(tmp_path / "swarms"))

    async def run():
        await registry.get("nope")

    with pytest.raises(RegistryNotFoundError):
        asyncio.run(run())


# 5b. Malformed YAML → RegistryParseError.
def test_get_malformed_yaml_raises_parse_error(tmp_path):
    swarms = tmp_path / "swarms"
    swarms.mkdir()
    (swarms / "broken.yaml").write_text(
        "name: broken\nagents: [unterminated\n",
        encoding="utf-8",
    )
    registry = SwarmRegistry(SpecLoader.from_filesystem(swarms))

    async def run():
        await registry.get("broken")

    with pytest.raises(RegistryParseError):
        asyncio.run(run())


# 6. Cache hit: second get() returns the same object for an unchanged revision.
def test_get_caches_spec_per_revision(tmp_path):
    _write_research(tmp_path)
    registry = SwarmRegistry(SpecLoader.from_filesystem(tmp_path / "swarms"))

    async def run():
        a = await registry.get("research")
        b = await registry.get("research")
        return a, b

    a, b = asyncio.run(run())
    assert a is b


# 7. parse_swarm_spec accepts string-form agents/coordinator and dict middleware.
def test_parse_swarm_spec_supports_string_refs_and_middleware():
    payload = {
        "name": "fast",
        "agents": ["a1", {"agent_id": "a2", "role": "checker"}],
        "coordinator": "lead",
        "middleware": [{"name": "budget", "config": {"limit": 5}}],
    }
    spec = parse_swarm_spec("fast", payload)
    assert spec.agents[0].agent_id == "a1"
    assert spec.agents[0].role is None
    assert spec.agents[1].agent_id == "a2"
    assert spec.agents[1].role == "checker"
    assert spec.coordinator.agent_id == "lead"
    assert len(spec.middleware) == 1
    assert spec.middleware[0].name == "budget"
    assert dict(spec.middleware[0].config) == {"limit": 5}
