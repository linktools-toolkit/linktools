#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""(v5 guide ): the USER session message must be exactly the caller's
original prompt -- not the model prompt (which folds in prior history, memory,
and knowledge). Persisting the model prompt stored internal runtime context as
the user's words, recursively re-injecting it each turn and leaking context.

The test seeds a prior USER/ASSISTANT turn, then runs a new turn whose prompt
is ``"hello"``. The new USER message must equal ``"hello"`` and contain no trace
of the prior history. It fails before the fix (the new USER message was the
concatenated ``history + prompt``)."""

import asyncio
from datetime import datetime, timezone

from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from linktools.ai.agent.compiler import AgentCompiler
from linktools.ai.agent.engine import AgentEngine
from linktools.ai.agent.spec import AgentSpec, PromptSpec
from linktools.ai.model.policy import ModelPolicy
from linktools.ai.model.registry import ModelRegistry
from linktools.ai.model.router import ModelGateway, ModelResolver
from linktools.ai.governance.policy.engine import PolicyEngine
from linktools.ai.run.context import RunContext
from linktools.ai.run.models import RunInput, RunnableType
from linktools.ai.session.models import (
    MessageRole,
    NewSessionMessage,
    SessionRecord,
    SessionStatus,
)
from linktools.ai.storage.filesystem.approval import FilesystemApprovalStore
from linktools.ai.storage.filesystem.checkpoint import FilesystemCheckpointStore
from linktools.ai.storage.filesystem.commit import FilesystemRunCommitCoordinator
from linktools.ai.storage.filesystem.event import FilesystemEventStore
from linktools.ai.storage.filesystem.run import FilesystemRunStore
from linktools.ai.storage.filesystem.session import FilesystemSessionStore
from linktools.ai.tool.executor import GovernedToolInvoker


def _model_fn(messages, info: AgentInfo) -> ModelResponse:  # noqa: ARG001
    return ModelResponse(parts=[TextPart(content='{"response": {"answer": 42}}')])


def _run_context():
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


def _build(tmp_path):
    run_store = FilesystemRunStore(root=tmp_path / "runs")
    session_store = FilesystemSessionStore(root=tmp_path / "sessions")
    event_store = FilesystemEventStore(root=tmp_path / "events")
    checkpoint_store = FilesystemCheckpointStore(root=tmp_path / "checkpoints")
    return (
        AgentEngine(
            run_store=run_store,
            session_store=session_store,
            event_store=event_store,
            checkpoint_store=checkpoint_store,
            commit_coordinator=FilesystemRunCommitCoordinator(
                approval_store=FilesystemApprovalStore(root=tmp_path / "approvals"),
                checkpoint_store=checkpoint_store,
                run_store=run_store,
                session_store=session_store,
                event_store=event_store,
            ),
        ),
        session_store,
    )


def _seed_with_history(session_store):
    now = datetime.now(timezone.utc)
    asyncio.run(
        session_store.create(
            SessionRecord(
                id="session-1",
                parent_id=None,
                status=SessionStatus.ACTIVE,
                version=1,
                created_at=now,
                updated_at=now,
            )
        )
    )
    asyncio.run(
        session_store.append_messages(
            "session-1",
            (
                NewSessionMessage(
                    role=MessageRole.USER, content="earlier-question", run_id="run-0"
                ),
                NewSessionMessage(
                    role=MessageRole.ASSISTANT,
                    content="earlier-answer",
                    run_id="run-0",
                ),
            ),
        )
    )


def test_user_message_is_original_prompt_not_history(tmp_path):
    runner, session_store = _build(tmp_path)
    _seed_with_history(session_store)

    registry = ModelRegistry()
    registry.register("test-model", model=FunctionModel(_model_fn))
    compiler = AgentCompiler(
        tool_executor=GovernedToolInvoker(policy=PolicyEngine(rules=())),
        model_router=ModelGateway(ModelResolver(registry=registry)),
    )
    compiled = asyncio.run(
        compiler.compile(
            AgentSpec(
                id="agent-1",
                name="a",
                model=ModelPolicy(primary="test-model"),
                instructions=PromptSpec(instructions="hi"),
                output_schema=str,
            )
        )
    )

    asyncio.run(runner.run(compiled, RunInput(prompt="hello"), _run_context()))

    messages = asyncio.run(session_store.list_messages("session-1"))
    user_messages = [m for m in messages if m.role is MessageRole.USER]
    # The new turn's USER message is exactly the original prompt -- no prior
    # history, no role-prefixed blob.
    assert user_messages[-1].content == "hello"
    assert "earlier" not in user_messages[-1].content
