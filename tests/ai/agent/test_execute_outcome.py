#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests/ai/agent/test_execute_outcome.py

WP9 step 2 (strangler-fig increment): ``AgentEngine.execute_outcome()`` is
the new section-12.4 signature (``*, context, agent, input, cancellation``)
returning a single ``AgentExecutionOutcome`` instead of the legacy async-
generator-of-dict-events shape. It is an adapter over the EXISTING
``execute()`` generator (unchanged), so these tests cover the four outcome
statuses end-to-end against the same fixtures the generator's own tests use."""

import asyncio
from datetime import datetime, timezone

import pytest
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from linktools.ai.agent.compiler import AgentCompiler
from linktools.ai.agent.engine import AgentEngine
from linktools.ai.agent.models import AgentExecutionStatus, AgentInput
from linktools.ai.agent.spec import AgentSpec, PromptSpec, ToolRef
from linktools.ai.capability.models import CapabilityBundle
from linktools.ai.capability.provider import CapabilityProvider
from linktools.ai.capability.resolver import CapabilityResolver
from linktools.ai.governance.policy.approval import ApprovalRule
from linktools.ai.governance.policy.engine import PolicyEngine
from linktools.ai.model.policy import ModelPolicy
from linktools.ai.model.registry import ModelRegistry
from linktools.ai.model.resolver import ModelResolver
from linktools.ai.run.context import RunContext
from linktools.ai.run.models import RunnableType, RunStatus
from linktools.ai.session.models import SessionRecord, SessionStatus
from linktools.ai.storage.filesystem.approval import FilesystemApprovalStore
from linktools.ai.storage.filesystem.checkpoint import FilesystemCheckpointStore
from linktools.ai.storage.filesystem.commit import FilesystemRunCommitCoordinator
from linktools.ai.storage.filesystem.event import FilesystemEventStore
from linktools.ai.storage.filesystem.run import FilesystemRunStore
from linktools.ai.storage.filesystem.session import FilesystemSessionStore
from linktools.ai.tool.executor import GovernedToolInvoker
from linktools.ai.tool.models import (
    ManagedToolDefinition,
    ToolContribution,
    ToolDescriptor,
)

TOOL_NAME = "risky"


def _run_context(run_id="run-o1", session_id="session-o1") -> RunContext:
    return RunContext(
        run_id=run_id,
        root_run_id=run_id,
        parent_run_id=None,
        session_id=session_id,
        runnable_id="agent-1",
        runnable_type=RunnableType.AGENT,
        user_id=None,
        tenant_id=None,
        workspace=None,
    )


def _seed_session(store, session_id) -> None:
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


def _make_runner(tmp_path, **kw) -> AgentEngine:
    run_store = FilesystemRunStore(root=tmp_path / "runs")
    session_store = FilesystemSessionStore(root=tmp_path / "sessions")
    event_store = FilesystemEventStore(root=tmp_path / "events")
    checkpoint_store = FilesystemCheckpointStore(root=tmp_path / "checkpoints")
    approval_store = kw.pop("approval_store", None) or FilesystemApprovalStore(
        root=tmp_path / "approvals"
    )
    return AgentEngine(
        run_store=run_store,
        session_store=session_store,
        event_store=event_store,
        commit_coordinator=FilesystemRunCommitCoordinator(
            approval_store=approval_store,
            checkpoint_store=checkpoint_store,
            run_store=run_store,
            session_store=session_store,
            event_store=event_store,
        ),
        **kw,
    )


def _model_fn(text: str = '{"response": {"answer": 42}}'):
    def _fn(messages, info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart(content=text)])

    return _fn


def _registry(model_fn):
    registry = ModelRegistry()
    registry.register("test-model", model=FunctionModel(model_fn))
    return registry


def _compile_simple(tmp_path, model_fn):
    compiler = AgentCompiler(
        tool_executor=GovernedToolInvoker(policy=PolicyEngine(rules=())),
        model_resolver=ModelResolver(registry=_registry(model_fn)),
    )
    return asyncio.run(
        compiler.compile(
            AgentSpec(
                id="agent-1",
                name="a",
                model=ModelPolicy(primary="test-model"),
                instructions=PromptSpec(instructions="hi"),
            )
        )
    )


def test_execute_outcome_completed_carries_result_and_usage(tmp_path):
    compiled = _compile_simple(tmp_path, _model_fn())
    runner = _make_runner(tmp_path)
    _seed_session(runner._session_store, "session-o1")

    outcome = asyncio.run(
        runner.execute_outcome(
            context=_run_context(),
            agent=compiled,
            input=AgentInput(prompt="what is the answer?"),
        )
    )
    assert outcome.status is AgentExecutionStatus.COMPLETED
    assert outcome.result is not None
    assert "42" in str(outcome.result.output)
    assert outcome.usage is not None
    assert outcome.pause_request is None
    assert outcome.error is None

    run_record = asyncio.run(runner._run_store.get("run-o1"))
    assert run_record.status is RunStatus.SUCCEEDED


def test_execute_outcome_failed_carries_error_info(tmp_path):
    def _boom(messages, info: AgentInfo) -> ModelResponse:
        raise RuntimeError("model exploded")

    compiled = _compile_simple(tmp_path, _boom)
    runner = _make_runner(tmp_path)
    _seed_session(runner._session_store, "session-o1")

    outcome = asyncio.run(
        runner.execute_outcome(
            context=_run_context(),
            agent=compiled,
            input=AgentInput(prompt="boom"),
        )
    )
    assert outcome.status is AgentExecutionStatus.FAILED
    assert outcome.error is not None
    assert outcome.error.error_type == "RuntimeError"
    assert outcome.result is None
    assert outcome.pause_request is None

    run_record = asyncio.run(runner._run_store.get("run-o1"))
    assert run_record.status is RunStatus.FAILED


class _RiskyProvider(CapabilityProvider):
    supported_kinds = ("test",)

    async def resolve(self, ref, context):
        async def risky(x: int) -> int:
            return x * 2

        return CapabilityBundle(
            tool_contributions=(
                ToolContribution(
                    tools=(
                        ManagedToolDefinition(
                            descriptor=ToolDescriptor(
                                name=TOOL_NAME,
                                source="test",
                                category="discovery",
                                risk="high",
                                mutating=False,
                            ),
                            handler=risky,
                        ),
                    )
                ),
            )
        )


def _tool_call_model_fn(messages, info: AgentInfo) -> ModelResponse:
    return ModelResponse(parts=[ToolCallPart(tool_name=TOOL_NAME, args={"x": 1})])


def test_execute_outcome_paused_carries_pause_request(tmp_path):
    approval_store = FilesystemApprovalStore(root=tmp_path / "approvals")
    executor = GovernedToolInvoker(
        policy=PolicyEngine(rules=(ApprovalRule(require_for=frozenset({TOOL_NAME})),)),
        approval_store=approval_store,
    )
    compiler = AgentCompiler(
        model_resolver=ModelResolver(registry=_registry(_tool_call_model_fn)),
        tool_executor=executor,
    )
    compiled = asyncio.run(
        compiler.compile(
            AgentSpec(
                id="agent-1",
                name="a",
                model=ModelPolicy(primary="test-model"),
                instructions=PromptSpec(instructions="hi"),
                output_schema=str,
                tools=(ToolRef(kind="test", name=TOOL_NAME),),
            )
        )
    )
    runner = _make_runner(
        tmp_path,
        approval_store=approval_store,
        capability_resolver=CapabilityResolver({"test": _RiskyProvider()}),
        managed_tool_executor=executor,
    )
    _seed_session(runner._session_store, "session-o1")

    outcome = asyncio.run(
        runner.execute_outcome(
            context=_run_context(),
            agent=compiled,
            input=AgentInput(prompt="call the risky tool"),
        )
    )
    assert outcome.status is AgentExecutionStatus.PAUSED
    assert outcome.pause_request is not None
    assert outcome.pause_request.approval_id
    assert outcome.result is None
    assert outcome.error is None

    run_record = asyncio.run(runner._run_store.get("run-o1"))
    assert run_record.status is RunStatus.WAITING_APPROVAL


def test_execute_outcome_cancelled_on_cancelled_error(tmp_path):
    def _hangs(messages, info: AgentInfo) -> ModelResponse:
        raise asyncio.CancelledError()

    compiled = _compile_simple(tmp_path, _hangs)
    runner = _make_runner(tmp_path)
    _seed_session(runner._session_store, "session-o1")

    async def _run():
        return await runner.execute_outcome(
            context=_run_context(),
            agent=compiled,
            input=AgentInput(prompt="cancel me"),
        )

    outcome = asyncio.run(_run())
    assert outcome.status is AgentExecutionStatus.CANCELLED
    assert outcome.result is None
    assert outcome.error is None
    assert outcome.pause_request is None
