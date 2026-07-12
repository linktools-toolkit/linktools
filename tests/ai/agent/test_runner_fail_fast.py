#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""§8.2 AgentRunner missing-dependency fail-fast (parameterized, spec §17.3).

A spec that needs tools must fail eagerly when the assembler or the managed
executor is missing -- before any capability resolution work. ``tools=()`` is
a model-only run and never raises."""

import asyncio
from datetime import datetime, timezone

import pytest
from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from linktools.ai.agent.compiler import AgentCompiler
from linktools.ai.agent.runner import AgentRunner
from linktools.ai.agent.spec import AgentSpec, PromptSpec, ToolRef
from linktools.ai.capability.assembler import CapabilityAssembler
from linktools.ai.errors import RuntimeInitializationError
from linktools.ai.model.policy import ModelPolicy
from linktools.ai.model.registry import ModelRegistry
from linktools.ai.model.router import ModelRouter
from linktools.ai.run.context import RunContext
from linktools.ai.run.models import RunInput, RunnableType
from linktools.ai.session.models import SessionRecord, SessionStatus
from linktools.ai.storage.file.checkpoint import FileCheckpointStore
from linktools.ai.storage.file.event import FileEventStore
from linktools.ai.storage.file.run import FileRunStore
from linktools.ai.storage.file.session import FileSessionStore


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
    compiler = AgentCompiler(model_router=ModelRouter(registry=_registry()))
    spec = AgentSpec(
        id="agent-1",
        name="a",
        model=ModelPolicy(primary="test-model"),
        instructions=PromptSpec(instructions="hi"),
        tools=(ToolRef(kind="builtin", name="file"),),
    )
    return asyncio.run(compiler.compile(spec))


def _make_runner(
    tmp_path, *, capability_assembler, managed_tool_executor
) -> AgentRunner:
    return AgentRunner(
        run_store=FileRunStore(root=tmp_path / "runs"),
        session_store=FileSessionStore(root=tmp_path / "sessions"),
        event_store=FileEventStore(root=tmp_path / "events"),
        checkpoint_store=FileCheckpointStore(root=tmp_path / "checkpoints"),
        capability_assembler=capability_assembler,
        managed_tool_executor=managed_tool_executor,
    )


@pytest.mark.parametrize(
    "assembler,executor,match",
    [
        (None, None, "CapabilityAssembler"),
        (CapabilityAssembler({}), None, "ToolExecutor"),
    ],
    ids=["no-assembler", "no-executor"],
)
def test_eager_fail_fast_when_tools_declared(tmp_path, assembler, executor, match):
    """Declaring tools with a missing assembler or executor must fail eagerly
    (before any capability resolution work)."""
    runner = _make_runner(
        tmp_path,
        capability_assembler=assembler,
        managed_tool_executor=executor,
    )
    _seed(runner._session_store, "session-1")
    compiled = _compiled_spec_with_tools()

    with pytest.raises(RuntimeInitializationError, match=match):
        asyncio.run(runner.run(compiled, RunInput(prompt="hi"), _context()))


def test_empty_tools_never_raises_even_without_assembler_or_executor(tmp_path):
    """tools=() is a model-only run: no assembler, no executor, no raise."""
    runner = _make_runner(
        tmp_path,
        capability_assembler=None,
        managed_tool_executor=None,
    )
    _seed(runner._session_store, "session-1")
    compiler = AgentCompiler(model_router=ModelRouter(registry=_registry()))
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
