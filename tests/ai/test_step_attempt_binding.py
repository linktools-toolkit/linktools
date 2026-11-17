#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run-local checkpoint binding preserves declarations and custom hooks."""

import pytest
from pydantic_ai_harness.step_persistence import InMemoryStepStore, StepPersistence

from linktools.ai.capability.steps import (
    AtomicStepPersistence, StepRunRegistration, bind_step_persistence,
)


def test_native_step_binding_keeps_declaration_and_uses_attempt_resources() -> None:
    source_store, attempt_store = InMemoryStepStore(), InMemoryStepStore()
    declaration = StepPersistence(
        store=source_store, id="checkpoint", description="Keep frontier", defer_loading=True,
        run_id="declared-run", parent_run_id="declared-parent", agent_name="declared-agent",
        metadata={"source": "declaration"}, capture_frontier=True,
    )
    adapted = AtomicStepPersistence.adapt(declaration)
    metadata = {"source": "declaration", "attempt": "current"}
    registration = StepRunRegistration(
        run_id="attempt", parent_run_id="parent", agent_name="agent", metadata=metadata,
    )
    bound = bind_step_persistence(
        adapted, store=attempt_store, run_id="attempt", parent_run_id="parent",
        agent_name="agent", metadata=metadata, run_registration=registration,
    )
    assert isinstance(bound, AtomicStepPersistence)
    assert bound.run_registration is registration
    assert bound.store is attempt_store
    assert (bound.run_id, bound.parent_run_id, bound.agent_name) == ("attempt", "parent", "agent")
    assert bound.metadata == metadata
    assert (bound.id, bound.description, bound.defer_loading, bound.capture_frontier) == (
        "checkpoint", "Keep frontier", True, True,
    )
    assert declaration.store is source_store
    assert (declaration.run_id, declaration.parent_run_id, declaration.agent_name) == (
        "declared-run", "declared-parent", "declared-agent",
    )
    assert declaration.metadata == {"source": "declaration"}


@pytest.mark.asyncio
async def test_custom_step_attempt_retains_its_tool_hook() -> None:
    class CustomStep(StepPersistence):
        async def before_tool_execute(self, ctx, *, call, tool_def, args):
            return {**args, "custom": True}

    declaration = CustomStep(store=InMemoryStepStore(), capture_frontier=True)
    bound = bind_step_persistence(
        AtomicStepPersistence.adapt(declaration), store=declaration.store,
        run_id="attempt", parent_run_id=None, agent_name="agent", metadata={},
    )
    assert type(bound) is CustomStep
    assert await bound.before_tool_execute(None, call=None, tool_def=None, args={"value": 1}) == {
        "value": 1, "custom": True,
    }
    assert declaration.run_id is None
    assert bound.capture_frontier


def test_atomic_binding_without_new_registration_keeps_existing_owner() -> None:
    registration = StepRunRegistration(run_id="old", parent_run_id=None, agent_name=None, metadata={})
    declaration = AtomicStepPersistence(store=InMemoryStepStore(), run_registration=registration)
    bound = bind_step_persistence(
        AtomicStepPersistence.adapt(declaration), store=declaration.store,
        run_id="attempt", parent_run_id=None, agent_name="agent", metadata={},
    )
    assert bound.run_registration is registration
    assert bound is not declaration
