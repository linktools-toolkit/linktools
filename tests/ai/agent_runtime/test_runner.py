#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import asyncio
from datetime import datetime, timezone

import pytest
from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from linktools.ai.agent_runtime.compiler import AgentCompiler
from linktools.ai.agent_runtime.runner import AgentRunner
from linktools.ai.agent_runtime.spec import AgentSpec, PromptSpec
from linktools.ai.core.model_runtime import ModelRegistry
from linktools.ai.middleware.base import Middleware
from linktools.ai.middleware.pipeline import MiddlewarePipeline
from linktools.ai.model.policy import ModelPolicy
from linktools.ai.model.router import ModelRouter
from linktools.ai.run.context import RunContext
from linktools.ai.run.models import RunInput, RunnableType, RunStatus
from linktools.ai.session.models import SessionRecord, SessionStatus
from linktools.ai.storage.file.checkpoint import FileCheckpointStore
from linktools.ai.storage.file.event import FileEventStore
from linktools.ai.storage.file.run import FileRunStore
from linktools.ai.storage.file.session import FileSessionStore


def _model_fn(text: str = '{"response": {"answer": 42}}'):
    def _fn(messages, info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart(content=text)])
    return _fn

def _registry(model_fn):
    registry = ModelRegistry()
    registry.register("test-model", model=FunctionModel(model_fn))
    return registry

def _run_context(run_id="run-1", session_id="session-1") -> RunContext:
    return RunContext(run_id=run_id, root_run_id=run_id, parent_run_id=None, session_id=session_id,
                     runnable_id="agent-1", runnable_type=RunnableType.AGENT, user_id=None, tenant_id=None, workspace=None)

def _seed_session(store, session_id) -> None:
    now = datetime.now(timezone.utc)
    asyncio.run(store.create(SessionRecord(id=session_id, parent_id=None, status=SessionStatus.ACTIVE, version=1, created_at=now, updated_at=now)))

def _make_runner(tmp_path, pipeline=None):
    return AgentRunner(run_store=FileRunStore(root=tmp_path / "runs"),
                       session_store=FileSessionStore(root=tmp_path / "sessions"),
                       event_store=FileEventStore(root=tmp_path / "events"),
                       checkpoint_store=FileCheckpointStore(root=tmp_path / "checkpoints"),
                       middleware_pipeline=pipeline)


def test_run_succeeds_persists_session_run_events_and_checkpoint(tmp_path):
    compiler = AgentCompiler(model_router=ModelRouter(registry=_registry(_model_fn())))
    compiled = asyncio.run(compiler.compile(AgentSpec(id="agent-1", name="a", model=ModelPolicy(primary="test-model"), instructions=PromptSpec(instructions="hi"))))
    runner = _make_runner(tmp_path)
    _seed_session(runner._session_store, "session-1")
    async def _run():
        return await runner.run(compiled, RunInput(prompt="what is the answer?"), _run_context())
    result = asyncio.run(_run())
    assert "42" in str(result.output)
    async def _verify():
        run_record = await runner._run_store.get("run-1")
        messages = await runner._session_store.list_messages("session-1")
        events = await runner._event_store.list("run-1")
        checkpoint = await runner._checkpoint_store.latest("run-1")
        return run_record, messages, events, checkpoint
    run_record, messages, events, checkpoint = asyncio.run(_verify())
    assert run_record.status == RunStatus.SUCCEEDED
    assert any("42" in str(m.content) for m in messages)
    assert len(events.items) >= 2
    assert checkpoint is not None and checkpoint.run_id == "run-1"


def test_run_transitions_to_failed_and_appends_run_failed_on_model_error(tmp_path):
    def _boom(messages, info: AgentInfo) -> ModelResponse:
        raise RuntimeError("model exploded")
    compiler = AgentCompiler(model_router=ModelRouter(registry=_registry(_boom)))
    compiled = asyncio.run(compiler.compile(AgentSpec(id="agent-2", name="a", model=ModelPolicy(primary="test-model"), instructions=PromptSpec(instructions="hi"))))
    runner = _make_runner(tmp_path)
    _seed_session(runner._session_store, "session-2")
    async def _run():
        await runner.run(compiled, RunInput(prompt="hi"), RunContext(run_id="run-2", root_run_id="run-2", parent_run_id=None, session_id="session-2", runnable_id="agent-2", runnable_type=RunnableType.AGENT, user_id=None, tenant_id=None, workspace=None))
    with pytest.raises(Exception):
        asyncio.run(_run())
    async def _verify():
        run_record = await runner._run_store.get("run-2")
        events = await runner._event_store.list("run-2")
        return run_record, events
    run_record, events = asyncio.run(_verify())
    assert run_record.status == RunStatus.FAILED
    assert any(type(e.payload).__name__ == "RunFailed" for e in events.items)


class _RecordingMiddleware(Middleware):
    def __init__(self, log: list) -> None:
        self.log = log
    async def before_run(self, context):
        self.log.append("before_run")
    async def after_run(self, context, result):
        self.log.append("after_run")
        return result
    async def on_error(self, context, error):
        self.log.append("on_error")


def test_middleware_runner_hooks_fire_in_order_on_success(tmp_path):
    log: "list[str]" = []
    pipeline = MiddlewarePipeline(middlewares=(_RecordingMiddleware(log),))
    compiler = AgentCompiler(model_router=ModelRouter(registry=_registry(_model_fn())), middleware_pipeline=pipeline)
    compiled = asyncio.run(compiler.compile(AgentSpec(id="agent-3", name="a", model=ModelPolicy(primary="test-model"), instructions=PromptSpec(instructions="hi"))))
    runner = _make_runner(tmp_path, pipeline=pipeline)
    _seed_session(runner._session_store, "session-3")
    async def _run():
        await runner.run(compiled, RunInput(prompt="hi"), RunContext(run_id="run-3", root_run_id="run-3", parent_run_id=None, session_id="session-3", runnable_id="agent-3", runnable_type=RunnableType.AGENT, user_id=None, tenant_id=None, workspace=None))
    asyncio.run(_run())
    assert log == ["before_run", "after_run"]


def test_capabilities_current_context_set_during_run_then_cleared(tmp_path):
    compiler = AgentCompiler(model_router=ModelRouter(registry=_registry(_model_fn())))
    compiled = asyncio.run(compiler.compile(AgentSpec(id="agent-4", name="a", model=ModelPolicy(primary="test-model"), instructions=PromptSpec(instructions="hi"))))
    assert compiled.policy_capability.current_context is None
    runner = _make_runner(tmp_path)
    _seed_session(runner._session_store, "session-4")
    async def _run():
        await runner.run(compiled, RunInput(prompt="hi"), RunContext(run_id="run-4", root_run_id="run-4", parent_run_id=None, session_id="session-4", runnable_id="agent-4", runnable_type=RunnableType.AGENT, user_id=None, tenant_id=None, workspace=None))
    asyncio.run(_run())
    assert compiled.policy_capability.current_context is None
