#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SwarmEngine.execute(): the outcome-returning swarm loop, added alongside
legacy run(). Proves execute() drives the strategy + manages SwarmRun but
does NOT create/transition the driving RunRecord or write the parent-session
aggregate (RunCoordinator owns those, mirroring the agent path). The driving
Run + session write are the Coordinator's job."""

import asyncio

import pytest

from linktools.ai.agent.spec import AgentSpec, PromptSpec
from linktools.ai.errors import SwarmConflictError
from linktools.ai.model.policy import ModelPolicy
from linktools.ai.run.cancellation import CancellationToken
from linktools.ai.run.models import RunInput, RunnableType, RunStatus
from linktools.ai.swarm.engine import SwarmEngine
from linktools.ai.swarm.models import SwarmCompleted, SwarmFailed, SwarmStatus
from tests.ai.swarm.test_engine import (
    AgentRef,
    _agent_spec,
    _build_compiler,
    _driving_context,
    _limits,
    _spec,
    _Stores,
)


def test_execute_returns_swarm_completed_without_creating_driving_run(tmp_path):
    compiler = _build_compiler("coord-out", "alpha-out", "beta-out")
    stores = _Stores(tmp_path)
    stores.seed_shared_session("shared-session")
    runner = SwarmEngine(
        dispatcher=stores.agent_runner,
        compiler=compiler,
        swarm_commit_coordinator=stores.swarm_commit_coordinator,
    )
    spec = _spec(
        kind="parallel_fan_out",
        limits=_limits(max_concurrency=4),
        agents=(AgentRef("coord"), AgentRef("worker-a"), AgentRef("worker-b")),
        coordinator=AgentRef("coord"),
        config={"task_count": 2},
    )
    agents = {
        "coord": _agent_spec("coord", "model-0"),
        "worker-a": _agent_spec("worker-a", "model-1"),
        "worker-b": _agent_spec("worker-b", "model-2"),
    }
    context = _driving_context("drive-exec-1", "shared-session")

    async def _run():
        return await runner.execute(
            spec,
            RunInput(prompt="do the work"),
            context,
            agents=agents,
            cancellation=CancellationToken(),
        )

    outcome = asyncio.run(_run())

    assert isinstance(outcome, SwarmCompleted)
    assert "alpha-out" in str(outcome.result.output)
    assert "beta-out" in str(outcome.result.output)
    # execute() builds the aggregate messages for the Coordinator to persist --
    # it does NOT write them itself.
    assert len(outcome.aggregate_messages) == 1
    assert outcome.usage.input_tokens >= 0

    async def _verify():
        # No driving RunRecord -- execute() does not own it (RunCoordinator does).
        driving = await stores.run_store.get(context.run_id)
        children = await stores.run_store.list_children(context.run_id)
        messages = await stores.session_store.list_messages(context.session_id)
        return driving, children, messages

    driving, children, messages = asyncio.run(_verify())
    assert driving is None, "execute() must not create the driving RunRecord"
    # Child runs ARE created (worker dispatch goes through the dispatcher).
    assert len(children) == 2
    # The parent session was NOT written by execute() (no new aggregate message).
    assert messages == ()


def test_execute_returns_swarm_failed_on_strategy_error(tmp_path):
    compiler = _build_compiler("coord-out")
    stores = _Stores(tmp_path)
    stores.seed_shared_session("shared-session")
    runner = SwarmEngine(
        dispatcher=stores.agent_runner,
        compiler=compiler,
        swarm_commit_coordinator=stores.swarm_commit_coordinator,
    )
    # A single-worker swarm whose model is not registered -> resolution fails.
    spec = _spec(
        kind="parallel_fan_out",
        limits=_limits(max_concurrency=1),
        agents=(AgentRef("coord"), AgentRef("worker-x")),
        coordinator=AgentRef("coord"),
        config={"task_count": 1},
    )
    agents = {
        "coord": _agent_spec("coord", "model-0"),
        "worker-x": _agent_spec("worker-x", "model-missing"),
    }
    context = _driving_context("drive-exec-2", "shared-session")

    async def _run():
        return await runner.execute(
            spec,
            RunInput(prompt="go"),
            context,
            agents=agents,
            cancellation=CancellationToken(),
        )

    outcome = asyncio.run(_run())
    assert isinstance(outcome, SwarmFailed)
    assert outcome.error.error_type


def test_execute_propagates_swarm_conflict_error_instead_of_swarm_failed(tmp_path):
    """A conflict/invariant swarm error must propagate -- it is NOT an expected
    per-run failure, so it must not be swallowed into a SwarmFailed outcome
    (which would mislabel a swarm that actually completed). Simulates the
    reviewer's scenario: a version conflict on the SUCCEEDED transition raised
    AFTER the strategy already produced a successful result."""
    compiler = _build_compiler("coord-out", "alpha-out")
    stores = _Stores(tmp_path)
    stores.seed_shared_session("shared-session")
    runner = SwarmEngine(
        dispatcher=stores.agent_runner,
        compiler=compiler,
        swarm_commit_coordinator=stores.swarm_commit_coordinator,
    )
    spec = _spec(
        kind="parallel_fan_out",
        limits=_limits(max_concurrency=1),
        agents=(AgentRef("coord"), AgentRef("worker-a")),
        coordinator=AgentRef("coord"),
        config={"task_count": 1},
    )
    agents = {
        "coord": _agent_spec("coord", "model-0"),
        "worker-a": _agent_spec("worker-a", "model-1"),
    }
    context = _driving_context("drive-exec-3", "shared-session")

    # Sabotage ONLY the SUCCEEDED transition: the strategy completes, then the
    # swarm_store rejects the SUCCEEDED update as a version conflict.
    original_update_run = stores.swarm_store.update_run

    async def _conflict_on_succeeded(swarm_run_id, **kwargs):
        if kwargs.get("status") is SwarmStatus.SUCCEEDED:
            raise SwarmConflictError(
                f"simulated version conflict on SUCCEEDED for {swarm_run_id}"
            )
        return await original_update_run(swarm_run_id, **kwargs)

    stores.swarm_store.update_run = _conflict_on_succeeded  # type: ignore[assignment]

    async def _run():
        return await runner.execute(
            spec,
            RunInput(prompt="go"),
            context,
            agents=agents,
            cancellation=CancellationToken(),
        )

    with pytest.raises(SwarmConflictError):
        asyncio.run(_run())


def test_execute_child_run_runnable_id_is_the_agent_ref_key_not_spec_id(tmp_path):
    """The child RunRecord.runnable_id is the AgentRef key (assigned_agent_id),
    not compiled.spec.id -- the two differ when a swarm references an agent by
    an alias distinct from the spec's own id. Guards the metadata threading:
    runnable_id must travel on RunDispatchRequest.metadata (where dispatch_child
    reads it), not on open_child's metadata (which only feeds session_id_format
    and is discarded)."""
    compiler = _build_compiler("coord-out", "worker-out")
    stores = _Stores(tmp_path)
    stores.seed_shared_session("shared-session")
    runner = SwarmEngine(
        dispatcher=stores.agent_runner,
        compiler=compiler,
        swarm_commit_coordinator=stores.swarm_commit_coordinator,
    )
    spec = _spec(
        kind="parallel_fan_out",
        limits=_limits(max_concurrency=1),
        agents=(AgentRef("coord"), AgentRef("worker-alias")),
        coordinator=AgentRef("coord"),
        config={"task_count": 1},
    )
    # Key the worker by an alias; the referenced spec's own id differs.
    agents = {
        "coord": _agent_spec("coord", "model-0"),
        "worker-alias": _agent_spec("worker-actual", "model-1"),
    }
    context = _driving_context("drive-exec-runnable", "shared-session")

    async def _run():
        return await runner.execute(
            spec,
            RunInput(prompt="go"),
            context,
            agents=agents,
            cancellation=CancellationToken(),
        )

    outcome = asyncio.run(_run())
    assert isinstance(outcome, SwarmCompleted)

    children = asyncio.run(stores.run_store.list_children(context.run_id))
    assert len(children) == 1
    # runnable_id is the AgentRef key (assigned_agent_id), NOT spec.id.
    assert children[0].runnable_id == "worker-alias", children[0].runnable_id


def test_execute_emits_swarm_lifecycle_events_via_the_injected_sink(tmp_path):
    """execute() emits SwarmStarted + SwarmCompleted through the commit
    coordinator's event_store (the events are persisted as part of the
    swarm's commit lifecycle, not appended by the engine directly)."""
    compiler = _build_compiler("coord-out", "alpha-out")
    stores = _Stores(tmp_path)
    stores.seed_shared_session("shared-session")
    runner = SwarmEngine(
        dispatcher=stores.agent_runner,
        compiler=compiler,
        swarm_commit_coordinator=stores.swarm_commit_coordinator,
    )
    spec = _spec(
        kind="parallel_fan_out",
        limits=_limits(max_concurrency=1),
        agents=(AgentRef("coord"), AgentRef("worker-a")),
        coordinator=AgentRef("coord"),
        config={"task_count": 1},
    )
    agents = {
        "coord": _agent_spec("coord", "model-0"),
        "worker-a": _agent_spec("worker-a", "model-1"),
    }
    context = _driving_context("drive-exec-sink", "shared-session")

    async def _run():
        return await runner.execute(
            spec,
            RunInput(prompt="go"),
            context,
            agents=agents,
            cancellation=CancellationToken(),
        )

    outcome = asyncio.run(_run())
    assert isinstance(outcome, SwarmCompleted)
    # Events were persisted through the coordinator's event_store.
    events = asyncio.run(stores.event_store.list(context.run_id))
    emitted_types = [type(e.payload).__name__ for e in events.items]
    assert "SwarmStarted" in emitted_types
    assert "SwarmCompleted" in emitted_types


def test_execute_emits_swarm_failed_via_the_injected_sink(tmp_path):
    """On an expected strategy failure, execute() emits SwarmFailed through the
    commit coordinator's event_store."""
    compiler = _build_compiler("coord-out")
    stores = _Stores(tmp_path)
    stores.seed_shared_session("shared-session")
    runner = SwarmEngine(
        dispatcher=stores.agent_runner,
        compiler=compiler,
        swarm_commit_coordinator=stores.swarm_commit_coordinator,
    )
    # A worker whose model is not registered -> ModelRoutingError -> SwarmFailed.
    spec = _spec(
        kind="parallel_fan_out",
        limits=_limits(max_concurrency=1),
        agents=(AgentRef("coord"), AgentRef("worker-x")),
        coordinator=AgentRef("coord"),
        config={"task_count": 1},
    )
    agents = {
        "coord": _agent_spec("coord", "model-0"),
        "worker-x": _agent_spec("worker-x", "model-missing"),
    }
    context = _driving_context("drive-exec-sink-fail", "shared-session")

    async def _run():
        return await runner.execute(
            spec,
            RunInput(prompt="go"),
            context,
            agents=agents,
            cancellation=CancellationToken(),
        )

    outcome = asyncio.run(_run())
    assert isinstance(outcome, SwarmFailed)
    # The model-resolution failure surfaces during _compile_members, BEFORE the
    # swarm run is created -- so no swarm lifecycle events are persisted (the
    # SwarmFailed outcome is returned directly without a coordinator commit).
    events = asyncio.run(stores.event_store.list(context.run_id))
    emitted_types = [type(e.payload).__name__ for e in events.items]
    assert "SwarmStarted" not in emitted_types
