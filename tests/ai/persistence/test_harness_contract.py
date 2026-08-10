#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""V23 checks for the public Harness persistence boundary."""

import asyncio
from importlib.metadata import version
from inspect import signature
from pathlib import Path

import pytest
from linktools.ai.adapter import DurableFilesystemStepStore
from linktools.ai.agent import (
    AgentCatalogItem,
    AgentCatalogSnapshot,
    AgentRunScope,
    compose_platform_capabilities,
)
from linktools.ai.core import step_conversation_id, step_run_id
from pydantic_ai import Agent
from pydantic_ai.exceptions import ModelRetry, UnexpectedModelBehavior
from pydantic_ai.messages import ModelRequest, ModelResponse
from pydantic_ai.models.test import TestModel
from pydantic_ai_harness.compaction import DeduplicateFileReads
from pydantic_ai_harness.conversation_search import ConversationSearch
from pydantic_ai_harness.dynamic_workflow import DynamicWorkflow
from pydantic_ai_harness.memory import InMemoryStore, Memory
from pydantic_ai_harness.planning import Planning
from pydantic_ai_harness.step_persistence import (
    InMemoryStepStore,
    StepPersistence,
    StepStore,
    continue_run,
    fork_run,
)
from pydantic_ai_harness.subagents import SubAgent, SubAgents


def test_harness_versions_and_public_step_store() -> None:
    assert version("pydantic-ai-harness") == "0.13.0"
    assert version("pydantic-ai-slim") == "2.27.0"
    assert isinstance(InMemoryStepStore(), StepStore)
    assert isinstance(DurableFilesystemStepStore.__new__(DurableFilesystemStepStore), StepStore)


def test_harness_public_signatures_are_the_locked_contract() -> None:
    run_parameters = signature(Agent.run_stream_events).parameters
    assert "conversation_id" in run_parameters and "capabilities" in run_parameters and "run_id" in run_parameters
    assert "run_id" in ModelRequest.__dataclass_fields__ and "run_id" in ModelResponse.__dataclass_fields__
    persistence_parameters = signature(StepPersistence).parameters
    assert {"run_id", "agent_name", "parent_run_id", "metadata"} <= set(persistence_parameters)
    assert "include_interrupted" in signature(StepStore.latest_snapshot).parameters
    assert {"parent_run_id", "conversation_id"} <= set(signature(StepStore.list_runs).parameters)
    for helper in (continue_run, fork_run):
        assert {"store", "run_id"} <= set(signature(helper).parameters)
        assert "include_interrupted" in signature(helper).parameters


@pytest.mark.asyncio
async def test_continue_and_fork_require_a_provider_valid_snapshot() -> None:
    store = InMemoryStepStore()
    with pytest.raises(LookupError):
        await continue_run(store, run_id="missing")
    with pytest.raises(LookupError):
        await fork_run(store, run_id="missing")


@pytest.mark.asyncio
async def test_workspace_composition_uses_harness_capabilities_directly() -> None:
    catalog = AgentCatalogSnapshot(
        (
            AgentCatalogItem("agent-z", "agent_z", "z", "execute z", None),
            AgentCatalogItem("agent-a", "agent_a", "a", "execute a", None),
        )
    )
    capabilities = await compose_platform_capabilities(
        AgentRunScope(
            root=Path("."),
            agent_name="parent",
            conversation_id="conversation",
            step_run_id="run",
            segment_sequence=1,
            memory_namespace="namespace",
            step_store=InMemoryStepStore(),
            inherited_capabilities=(),
            agent_catalog=catalog,
            memory_store=InMemoryStore(),
        ),
        model_factory=lambda value: TestModel(call_tools=[]),
        parent_model=TestModel(call_tools=[]),
    )

    assert isinstance(capabilities[0], StepPersistence)
    assert isinstance(capabilities[1], Memory)
    assert isinstance(capabilities[2], Planning)
    assert capabilities[3].__class__ is ConversationSearch
    assert capabilities[4].__class__ is SubAgents
    assert capabilities[5].__class__ is DynamicWorkflow
    assert capabilities[6].__class__ is DeduplicateFileReads
    assert all(isinstance(item, SubAgent) for item in capabilities[4].agents)
    assert all(isinstance(item, Agent) for item in capabilities[5].agents)


@pytest.mark.asyncio
async def test_tool_retry_closes_step_effect() -> None:
    store = InMemoryStepStore()
    capabilities = await compose_platform_capabilities(
        AgentRunScope(
            root=Path("."),
            agent_name="agent",
            conversation_id=None,
            step_run_id="run",
            segment_sequence=1,
            memory_namespace=None,
            step_store=store,
            inherited_capabilities=(),
            agent_catalog=AgentCatalogSnapshot(()),
            memory_store=None,
        ),
        model_factory=lambda value: TestModel(call_tools=[]),
        parent_model=TestModel(call_tools=[]),
    )
    agent = Agent(TestModel(call_tools=["fail_tool"]), capabilities=capabilities)

    @agent.tool_plain
    async def fail_tool() -> str:
        raise ModelRetry("retry")

    with pytest.raises(UnexpectedModelBehavior):
        await agent.run("run", run_id="run", conversation_id="conversation")

    assert await store.list_unresolved_tool_effects(run_id="run") == []
    effect = await store.get_tool_effect(run_id="run", tool_call_id="pyd_ai_tool_call_id__fail_tool")
    assert effect is not None and effect.status == "failed"


def test_step_ids_are_scoped_and_fixed_width() -> None:
    conversation = step_conversation_id(namespace="ns", tenant_id="tenant", execution_id="execution")
    run = step_run_id(namespace="ns", tenant_id="tenant", execution_id="execution", segment_sequence=1)
    assert conversation.startswith("c-") and len(conversation) == 66
    assert run.startswith("r-") and len(run) == 66
    assert conversation == step_conversation_id(namespace="ns", tenant_id="tenant", execution_id="execution")
    assert run != step_run_id(namespace="ns", tenant_id="tenant", execution_id="execution", segment_sequence=2)


def test_filesystem_step_store_uses_digest_only_paths(tmp_path: Path) -> None:
    async def run() -> list[Path]:
        store = DurableFilesystemStepStore(tmp_path, "tenant/unsafe")
        await store.initialize()
        paths = list((tmp_path / "steps").rglob("*"))
        await store.close()
        return paths

    paths = asyncio.run(run())
    assert all(path.name not in {"tenant", "unsafe"} for path in paths)
