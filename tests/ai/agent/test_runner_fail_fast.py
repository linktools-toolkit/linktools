#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AgentEngine missing-dependency fail-fast.

A spec that needs tools must fail eagerly when the resolver or the managed
executor is missing -- before any capability resolution work. ``tools=()`` is
a model-only run and never raises."""

import asyncio
from datetime import datetime, timezone

import pytest
from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from linktools.ai.agent.compiler import AgentCompiler
from linktools.ai.agent.engine import AgentEngine
from linktools.ai.agent.spec import AgentSpec, PromptSpec, ToolRef
from linktools.ai.capability.resolver import CapabilityResolver
from linktools.ai.errors import RuntimeInitializationError
from linktools.ai.model.policy import ModelPolicy
from linktools.ai.model.registry import ModelRegistry
from linktools.ai.model.resolver import ModelResolver
from linktools.ai.run.context import RunContext
from linktools.ai.run.models import RunInput, RunnableType
from linktools.ai.session.models import SessionRecord, SessionStatus
from linktools.ai.storage.filesystem.checkpoint import FilesystemCheckpointStore
from linktools.ai.storage.filesystem.event import FilesystemEventStore
from linktools.ai.storage.filesystem.run import FilesystemRunStore
from linktools.ai.storage.filesystem.session import FilesystemSessionStore
from linktools.ai.governance.policy.engine import PolicyEngine
from linktools.ai.tool.executor import GovernedToolInvoker


def _model_fn(messages, info: AgentInfo) -> ModelResponse:
    return ModelResponse(parts=[TextPart(content='{"response": {"answer": 42}}')])


def _registry():
    registry = ModelRegistry()
    registry.register("test-model", model=FunctionModel(_model_fn))
    return registry


def _context() -> RunContext:
    return RunContext(
        run_id="run-1",
        root_run_id="run-1",
        parent_run_id=None,
        session_id="session-1",
        runnable_id="agent-1",
        runnable_type=RunnableType.AGENT,
        user_id=None,
        tenant_id=None,
        workspace=None,
    )


def _seed(store, session_id) -> None:
    now = datetime.now(timezone.utc)
    asyncio.run(
        store.create(
            SessionRecord(
                id=session_id,
                parent_id=None,
                status=SessionStatus.ACTIVE,
                version=1,
                created_at=now,
                updated_at=now,
            )
        )
    )


def _compiled_spec_with_tools():
    compiler = AgentCompiler(
        tool_executor=GovernedToolInvoker(policy=PolicyEngine(rules=())),
        model_resolver=ModelResolver(registry=_registry()),
    )
    spec = AgentSpec(
        id="agent-1",
        name="a",
        model=ModelPolicy(primary="test-model"),
        instructions=PromptSpec(instructions="hi"),
        tools=(ToolRef(kind="builtin", name="file"),),
    )
    return asyncio.run(compiler.compile(spec))


def _make_runner(
    tmp_path, *, capability_resolver, managed_tool_executor
) -> AgentEngine:
    from linktools.ai.storage.filesystem.approval import FilesystemApprovalStore
    from linktools.ai.storage.filesystem.commit import FilesystemRunCommitCoordinator

    run_store = FilesystemRunStore(root=tmp_path / "runs")
    session_store = FilesystemSessionStore(root=tmp_path / "sessions")
    event_store = FilesystemEventStore(root=tmp_path / "events")
    checkpoint_store = FilesystemCheckpointStore(root=tmp_path / "checkpoints")
    return AgentEngine(
        run_store=run_store,
        session_store=session_store,
        event_store=event_store,
        checkpoint_store=checkpoint_store,
        capability_resolver=capability_resolver,
        managed_tool_executor=managed_tool_executor,
        commit_coordinator=FilesystemRunCommitCoordinator(
            approval_store=FilesystemApprovalStore(root=tmp_path / "approvals"),
            checkpoint_store=checkpoint_store,
            run_store=run_store,
            session_store=session_store,
            event_store=event_store,
        ),
    )


@pytest.mark.parametrize(
    "resolver,executor,match",
    [
        (None, None, "CapabilityResolver"),
        (CapabilityResolver({}), None, "GovernedToolInvoker"),
    ],
    ids=["no-resolver", "no-executor"],
)
def test_eager_fail_fast_when_tools_declared(tmp_path, resolver, executor, match):
    """Declaring tools with a missing resolver or executor must fail eagerly
    (before any capability resolution work)."""
    runner = _make_runner(
        tmp_path,
        capability_resolver=resolver,
        managed_tool_executor=executor,
    )
    _seed(runner._session_store, "session-1")
    compiled = _compiled_spec_with_tools()

    with pytest.raises(RuntimeInitializationError, match=match):
        asyncio.run(runner.run(compiled, RunInput(prompt="hi"), _context()))


def test_empty_tools_never_raises_even_without_assembler_or_executor(tmp_path):
    """tools=() is a model-only run and does not require tool wiring."""
    runner = _make_runner(
        tmp_path,
        capability_resolver=None,
        managed_tool_executor=None,
    )
    _seed(runner._session_store, "session-1")
    compiler = AgentCompiler(
        tool_executor=GovernedToolInvoker(policy=PolicyEngine(rules=())),
        model_resolver=ModelResolver(registry=_registry()),
    )
    spec = AgentSpec(
        id="agent-1",
        name="a",
        model=ModelPolicy(primary="test-model"),
        instructions=PromptSpec(instructions="hi"),
        tools=(),
    )
    compiled = asyncio.run(compiler.compile(spec))
    result = asyncio.run(runner.run(compiled, RunInput(prompt="hi"), _context()))
    assert result is not None
