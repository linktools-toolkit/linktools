#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import asyncio
from datetime import datetime, timezone

from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from linktools.ai.agent_runtime.spec import AgentSpec, PromptSpec
from linktools.ai.model.policy import ModelPolicy
from linktools.ai.runtime import Runtime
from linktools.ai.storage.facade import FileStorage


def _model_fn(messages, info: AgentInfo) -> ModelResponse:
    # AgentCompiler defaults output_type to `dict` when the spec has no
    # output_schema. pydantic-ai's dict validator expects a {"response": {...}}
    # wrapper and unwraps the inner dict as result.output. Carrying the
    # assertion string inside the inner dict keeps the dict parse succeeding.
    return ModelResponse(parts=[TextPart(content='{"response": {"message": "hello from runtime"}}')])

def _registry():
    from linktools.ai.core.model_runtime import ModelRegistry
    registry = ModelRegistry()
    registry.register("test-model", model=FunctionModel(_model_fn))
    return registry


def test_runtime_build_assembles_storage_compiler_runner(tmp_path):
    from linktools.ai.model.router import ModelRouter
    storage = FileStorage(root=tmp_path)
    runtime = Runtime.build(storage=storage, model_router=ModelRouter(registry=_registry()))
    assert runtime.storage is storage
    assert runtime.runner is not None
    assert runtime.compiler is not None


def test_runtime_run_creates_session_when_none_given_and_returns_result(tmp_path):
    from linktools.ai.model.router import ModelRouter
    storage = FileStorage(root=tmp_path)
    runtime = Runtime.build(storage=storage, model_router=ModelRouter(registry=_registry()))
    spec = AgentSpec(id="agent-1", name="rt-agent", model=ModelPolicy(primary="test-model"), instructions=PromptSpec(instructions="hi"))
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
    from linktools.ai.model.router import ModelRouter
    from linktools.ai.session.models import SessionRecord, SessionStatus
    storage = FileStorage(root=tmp_path)
    runtime = Runtime.build(storage=storage, model_router=ModelRouter(registry=_registry()))
    fixed_session_id = "fixed-session-7"
    now = datetime.now(timezone.utc)
    async def _setup():
        await storage.sessions.create(SessionRecord(id=fixed_session_id, parent_id=None, status=SessionStatus.ACTIVE, version=1, created_at=now, updated_at=now))
    asyncio.run(_setup())
    spec = AgentSpec(id="agent-2", name="rt-agent-2", model=ModelPolicy(primary="test-model"), instructions=PromptSpec(instructions="hi"))
    async def _run():
        return await runtime.run(spec, "hello", session_id=fixed_session_id, run_id="fixed-run-7")
    asyncio.run(_run())
    async def _verify():
        run = await storage.runs.get("fixed-run-7")
        messages = await storage.sessions.list_messages(fixed_session_id)
        return run, messages
    run, messages = asyncio.run(_verify())
    assert run is not None
    assert any("hello from runtime" in str(m.content) for m in messages)
