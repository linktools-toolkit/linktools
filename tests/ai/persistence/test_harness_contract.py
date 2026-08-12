#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""V23 checks for the public Harness persistence boundary."""

import asyncio
from importlib.metadata import version
from inspect import signature
from pathlib import Path

import pytest
from linktools.ai.adapter import DurableFilesystemStepStore
from linktools.ai.core import step_conversation_id, step_run_id
from pydantic_ai import Agent
from pydantic_ai.messages import ModelRequest, ModelResponse
from pydantic_ai_harness.step_persistence import InMemoryStepStore, StepPersistence, StepStore, continue_run, fork_run


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
        paths = list((tmp_path / "step").rglob("*"))
        await store.close()
        return paths

    paths = asyncio.run(run())
    assert all(path.name not in {"tenant", "unsafe"} for path in paths)
