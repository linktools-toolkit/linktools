#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import asyncio
from datetime import datetime, timezone

import pytest
from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from linktools.ai.agent.spec import AgentSpec, PromptSpec
from linktools.ai.model.policy import ModelPolicy
from linktools.ai.runtime import Runtime, build_runtime
from linktools.ai.storage.facade import FilesystemStorage
from linktools.ai.storage.filesystem.commit import FilesystemRunCommitCoordinator


def _model_fn(messages, info: AgentInfo) -> ModelResponse:
    # AgentCompiler defaults output_type to `dict` when the spec has no
    # output_schema. pydantic-ai's dict validator expects a {"response": {...}}
    # wrapper and unwraps the inner dict as result.output. Carrying the
    # assertion string inside the inner dict keeps the dict parse succeeding.
    return ModelResponse(
        parts=[TextPart(content='{"response": {"message": "hello from runtime"}}')]
    )


def _registry():
    from linktools.ai.model.registry import ModelRegistry

    registry = ModelRegistry()
    registry.register("test-model", model=FunctionModel(_model_fn))
    return registry


def test_runtime_build_hides_internal_components(tmp_path):
    from linktools.ai.model.resolver import ModelResolver

    storage = FilesystemStorage(root=tmp_path)
    runtime = build_runtime(
        storage=storage,
        model_resolver=ModelResolver(registry=_registry()),
        commit_coordinator=FilesystemRunCommitCoordinator.from_storage(storage),
    )
    assert not hasattr(runtime, "storage")
    assert not hasattr(runtime, "runner")
    assert not hasattr(runtime, "compiler")


def test_runtime_build_no_longer_accepts_workdir(tmp_path):
    """scenario (actionable-fix-contract): build_runtime() must not depend
    on a bare `workdir: Path` param -- the caller builds its own
    Sandbox and passes it via `sandbox=`."""
    from linktools.ai.model.resolver import ModelResolver

    storage = FilesystemStorage(root=tmp_path)
    with pytest.raises(TypeError):
        build_runtime(
            storage=storage,
            model_resolver=ModelResolver(registry=_registry()),
            commit_coordinator=FilesystemRunCommitCoordinator.from_storage(storage),
            workdir=tmp_path,
        )


def test_runtime_build_wires_execution_backend_for_builtin_tools(tmp_path):
    """`sandbox=` (scenario's replacement for `workdir=`) gives the
    compiled agent builtin file/terminal tools via the SAME Sandbox
    machinery `workdir=` used to construct internally."""
    from linktools.ai.sandbox.local import LocalSandbox
    from linktools.ai.model.resolver import ModelResolver

    storage = FilesystemStorage(root=tmp_path)
    backend = LocalSandbox(runtime_dir=tmp_path / "workdir")
    runtime = build_runtime(
        storage=storage,
        model_resolver=ModelResolver(registry=_registry()),
        sandbox=backend,
        commit_coordinator=FilesystemRunCommitCoordinator.from_storage(storage),
    )
    assert runtime._components.sandbox is backend


def test_runtime_run_creates_session_when_none_given_and_returns_result(tmp_path):
    from linktools.ai.model.resolver import ModelResolver

    storage = FilesystemStorage(root=tmp_path)
    runtime = build_runtime(
        storage=storage,
        model_resolver=ModelResolver(registry=_registry()),
        commit_coordinator=FilesystemRunCommitCoordinator.from_storage(storage),
    )
    spec = AgentSpec(
        id="agent-1",
        name="rt-agent",
        model=ModelPolicy(primary="test-model"),
        instructions=PromptSpec(instructions="hi"),
    )

    async def _run():
        return await runtime.run(spec, "say hello")

    result = asyncio.run(_run())
    assert "hello from runtime" in str(result.output)

    async def _verify():
        sessions_dir = tmp_path / "sessions"
        if not sessions_dir.exists():
            return None
        children = [p for p in sessions_dir.iterdir() if p.is_dir()]
        return len(children)

    session_count = asyncio.run(_verify())
    assert session_count is not None and session_count >= 1


def test_runtime_run_with_explicit_session_reuses_it(tmp_path):
    from linktools.ai.model.resolver import ModelResolver
    from linktools.ai.session.models import SessionRecord, SessionStatus

    storage = FilesystemStorage(root=tmp_path)
    runtime = build_runtime(
        storage=storage,
        model_resolver=ModelResolver(registry=_registry()),
        commit_coordinator=FilesystemRunCommitCoordinator.from_storage(storage),
    )
    fixed_session_id = "fixed-session-7"
    now = datetime.now(timezone.utc)

    async def _setup():
        await storage.sessions.create(
            SessionRecord(
                id=fixed_session_id,
                parent_id=None,
                status=SessionStatus.ACTIVE,
                version=1,
                created_at=now,
                updated_at=now,
            )
        )

    asyncio.run(_setup())
    spec = AgentSpec(
        id="agent-2",
        name="rt-agent-2",
        model=ModelPolicy(primary="test-model"),
        instructions=PromptSpec(instructions="hi"),
    )

    async def _run():
        return await runtime.run(
            spec, "hello", session_id=fixed_session_id, run_id="fixed-run-7"
        )

    asyncio.run(_run())

    async def _verify():
        run = await storage.runs.get("fixed-run-7")
        messages = await storage.sessions.list_messages(fixed_session_id)
        return run, messages

    run, messages = asyncio.run(_verify())
    assert run is not None
    assert any("hello from runtime" in str(m.content) for m in messages)


def test_runtime_run_dispatches_swarm_spec_and_marks_driving_run_succeeded(tmp_path):
    from linktools.ai.model.registry import ModelRegistry
    from linktools.ai.model.resolver import ModelResolver
    from linktools.ai.run.models import RunnableType, RunStatus
    from linktools.ai.swarm.aggregation import AggregationPolicy
    from linktools.ai.swarm.limits import SwarmLimits
    from linktools.ai.swarm.models import AgentRef
    from linktools.ai.swarm.spec import (
        SwarmContextPolicy,
        SwarmSpec,
        SwarmStrategySpec,
    )

    def _worker_fn(messages, info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart(content="worker-output")])

    registry = ModelRegistry()
    registry.register("test-model", model=FunctionModel(_worker_fn))
    storage = FilesystemStorage(root=tmp_path)
    runtime = build_runtime(
        storage=storage,
        model_resolver=ModelResolver(registry=registry),
        commit_coordinator=FilesystemRunCommitCoordinator.from_storage(storage),
    )

    def _agent_spec(agent_id: str) -> AgentSpec:
        return AgentSpec(
            id=agent_id,
            name=agent_id,
            model=ModelPolicy(primary="test-model"),
            instructions=PromptSpec(instructions=f"you are {agent_id}"),
            output_schema=str,
        )

    swarm_spec = SwarmSpec(
        id="swarm-1",
        name="rt-swarm",
        agents=(AgentRef("coord"), AgentRef("worker-a")),
        coordinator=AgentRef("coord"),
        strategy=SwarmStrategySpec(kind="parallel_fan_out", config={"task_count": 1}),
        limits=SwarmLimits(
            max_rounds=10,
            max_tasks=50,
            max_delegations=20,
            max_depth=5,
            max_concurrency=4,
            max_total_tokens=None,
            max_total_cost=None,
            timeout_seconds=None,
        ),
        context_policy=SwarmContextPolicy(),
        aggregation=AggregationPolicy(),
    )
    agents = {
        "coord": _agent_spec("coord"),
        "worker-a": _agent_spec("worker-a"),
    }

    async def _run():
        return await runtime.run(
            swarm_spec,
            "do the work",
            run_id="drive-run-swarm-1",
            agents=agents,
        )

    result = asyncio.run(_run())
    assert "worker-output" in str(result.output)

    async def _verify():
        driving = await storage.runs.get("drive-run-swarm-1")
        return driving

    driving = asyncio.run(_verify())
    assert driving is not None
    assert driving.status is RunStatus.SUCCEEDED
    assert driving.runnable_type is RunnableType.SWARM


def test_runtime_run_swarm_spec_without_agents_raises(tmp_path):
    from linktools.ai.errors import SwarmError
    from linktools.ai.model.resolver import ModelResolver
    from linktools.ai.swarm.aggregation import AggregationPolicy
    from linktools.ai.swarm.limits import SwarmLimits
    from linktools.ai.swarm.models import AgentRef
    from linktools.ai.swarm.spec import (
        SwarmContextPolicy,
        SwarmSpec,
        SwarmStrategySpec,
    )

    storage = FilesystemStorage(root=tmp_path)
    runtime = build_runtime(
        storage=storage,
        model_resolver=ModelResolver(registry=_registry()),
        commit_coordinator=FilesystemRunCommitCoordinator.from_storage(storage),
    )
    swarm_spec = SwarmSpec(
        id="swarm-2",
        name="rt-swarm-2",
        agents=(AgentRef("coord"),),
        coordinator=AgentRef("coord"),
        strategy=SwarmStrategySpec(kind="parallel_fan_out", config={}),
        limits=SwarmLimits(
            max_rounds=10,
            max_tasks=50,
            max_delegations=20,
            max_depth=5,
            max_concurrency=4,
            max_total_tokens=None,
            max_total_cost=None,
            timeout_seconds=None,
        ),
        context_policy=SwarmContextPolicy(),
        aggregation=AggregationPolicy(),
    )

    async def _run():
        await runtime.run(swarm_spec, "do the work")

    with pytest.raises(SwarmError):
        asyncio.run(_run())


# -- Memory is on-by-default via storage.memories ----------------------------


def _echo_model_fn(messages, info: AgentInfo) -> ModelResponse:
    # FunctionModel sees the full prompt pydantic-ai was called with as a
    # UserPromptPart inside the last ModelRequest.parts. Echo it back wrapped
    # for pydantic-ai's default dict output validator.
    import json as _json

    prompt_text = "no-prompt-captured"
    for msg in reversed(messages):
        for part in reversed(getattr(msg, "parts", ()) or ()):
            content = getattr(part, "content", None)
            if isinstance(content, str) and content:
                prompt_text = content
                break
        else:
            continue
        break
    return ModelResponse(
        parts=[
            TextPart(content='{"response": {"echo": ' + _json.dumps(prompt_text) + "}}")
        ]
    )


def _echo_registry():
    from linktools.ai.model.registry import ModelRegistry

    registry = ModelRegistry()
    registry.register("test-model", model=FunctionModel(_echo_model_fn))
    return registry


def test_runtime_build_threads_storage_memories_into_runner(tmp_path):
    from linktools.ai.model.resolver import ModelResolver
    from linktools.ai.storage.filesystem.memory import FilesystemMemoryStore

    storage = FilesystemStorage(root=tmp_path)
    runtime = build_runtime(
        storage=storage,
        model_resolver=ModelResolver(registry=_echo_registry()),
        commit_coordinator=FilesystemRunCommitCoordinator.from_storage(storage),
    )
    # Memory is on-by-default: the runner's memory_store is the facade's memories.
    assert isinstance(runtime._components.runner._memory_store, FilesystemMemoryStore)
    assert runtime._components.runner._memory_store is storage.memories


def test_runtime_run_surfaces_seeded_memory_in_output(tmp_path):
    from linktools.ai.memory.models import MemoryRecord
    from linktools.ai.model.resolver import ModelResolver
    from linktools.ai.session.models import SessionRecord, SessionStatus

    storage = FilesystemStorage(root=tmp_path)
    runtime = build_runtime(
        storage=storage,
        model_resolver=ModelResolver(registry=_echo_registry()),
        commit_coordinator=FilesystemRunCommitCoordinator.from_storage(storage),
    )
    now = datetime.now(timezone.utc)

    async def _seed():
        # Runtime.run with an explicit session_id requires the session to exist.
        # The session is created under the same tenant the run will carry, so
        # resolve_session's strict (tenant_id) equality check passes.
        await storage.sessions.create(
            SessionRecord(
                id="rt-session-mem",
                parent_id=None,
                status=SessionStatus.ACTIVE,
                version=1,
                created_at=now,
                updated_at=now,
                tenant_id="rt-tenant",
            )
        )
        # Memory is tenant-scoped: seed it under the same tenant the run will
        # carry, so the DefaultMemoryPolicy's tenant-bound search finds it.
        # Content includes the query keyword ("hello") because
        # FilesystemMemoryStore.search is keyword-substring based.
        await storage.memories.remember(
            MemoryRecord(
                id="mem-rt-1",
                tenant_id="rt-tenant",
                owner_id="rt-session-mem",
                content="hello context: runtime-memory-token",
                category=None,
                confidence=None,
                version=1,
                created_at=now,
                updated_at=now,
                metadata={},
            )
        )

    asyncio.run(_seed())
    spec = AgentSpec(
        id="agent-mem",
        name="rt-mem-agent",
        model=ModelPolicy(primary="test-model"),
        instructions=PromptSpec(instructions="hi"),
    )

    async def _run():
        return await runtime.run(
            spec,
            "hello",
            session_id="rt-session-mem",
            run_id="rt-run-mem-1",
            tenant_id="rt-tenant",
        )

    result = asyncio.run(_run())
    assert "runtime-memory-token" in str(result.output)
    assert "## Memory" in str(result.output)


def test_runtime_applies_session_window_policy(tmp_path):
    """CapabilityRuntimeOptions.session_window_policy is applied to session
    history before the prompt is built (contract)."""
    from linktools.ai.capability import CapabilityRuntimeOptions
    from linktools.ai.model.resolver import ModelResolver

    seen = {"called": False, "count": None}

    class _Recording:
        async def select_messages(self, messages, model_policy):
            seen["called"] = True
            seen["count"] = len(messages)
            return list(messages)

    storage = FilesystemStorage(root=tmp_path)
    runtime = build_runtime(
        storage=storage,
        model_resolver=ModelResolver(registry=_registry()),
        options=CapabilityRuntimeOptions(session_window_policy=_Recording()),
        commit_coordinator=FilesystemRunCommitCoordinator.from_storage(storage),
    )
    spec = AgentSpec(
        id="agent-1",
        name="rt-agent",
        model=ModelPolicy(primary="test-model"),
        instructions=PromptSpec(instructions="hi"),
    )
    asyncio.run(runtime.run(spec, "go"))
    assert seen["called"] is True
    assert seen["count"] == 0  # fresh session has no prior messages
