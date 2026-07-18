#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import asyncio
from datetime import datetime, timezone

import pytest
from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from linktools.ai.agent.compiler import AgentCompiler
from linktools.ai.agent.runner import AgentRunner
from linktools.ai.agent.spec import AgentSpec, PromptSpec
from linktools.ai.model.registry import ModelRegistry
from linktools.ai.middleware.base import Middleware
from linktools.ai.middleware.pipeline import MiddlewarePipeline
from linktools.ai.model.policy import ModelPolicy
from linktools.ai.model.router import ModelRouter
from linktools.ai.run.context import RunContext
from linktools.ai.run.models import RunInput, RunnableType, RunStatus
from linktools.ai.session.models import SessionRecord, SessionStatus
from linktools.ai.storage.file.approval import FileApprovalStore
from linktools.ai.storage.file.checkpoint import FileCheckpointStore
from linktools.ai.storage.file.commit import FileRunCommitCoordinator
from linktools.ai.storage.file.event import FileEventStore
from linktools.ai.storage.file.run import FileRunStore
from linktools.ai.storage.file.session import FileSessionStore
from linktools.ai.policy.engine import PolicyEngine
from linktools.ai.tool.executor import ToolExecutor


def _model_fn(text: str = '{"response": {"answer": 42}}'):
    def _fn(messages, info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart(content=text)])

    return _fn


def _registry(model_fn):
    registry = ModelRegistry()
    registry.register("test-model", model=FunctionModel(model_fn))
    return registry


def _run_context(run_id="run-1", session_id="session-1", tenant_id=None) -> RunContext:
    return RunContext(
        run_id=run_id,
        root_run_id=run_id,
        parent_run_id=None,
        session_id=session_id,
        runnable_id="agent-1",
        runnable_type=RunnableType.AGENT,
        user_id=None,
        tenant_id=tenant_id,
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


def _make_runner(tmp_path, pipeline=None):
    from linktools.ai.storage.file.approval import FileApprovalStore
    from linktools.ai.storage.file.commit import FileRunCommitCoordinator

    run_store = FileRunStore(root=tmp_path / "runs")
    session_store = FileSessionStore(root=tmp_path / "sessions")
    event_store = FileEventStore(root=tmp_path / "events")
    checkpoint_store = FileCheckpointStore(root=tmp_path / "checkpoints")
    return AgentRunner(
        run_store=run_store,
        session_store=session_store,
        event_store=event_store,
        checkpoint_store=checkpoint_store,
        middleware_pipeline=pipeline,
        commit_coordinator=FileRunCommitCoordinator(
            approval_store=FileApprovalStore(root=tmp_path / "approvals"),
            checkpoint_store=checkpoint_store,
            run_store=run_store,
            session_store=session_store,
            event_store=event_store,
        ),
    )


def test_run_succeeds_persists_session_run_events_and_checkpoint(tmp_path):
    compiler = AgentCompiler(
        tool_executor=ToolExecutor(policy=PolicyEngine(rules=())),
        model_router=ModelRouter(registry=_registry(_model_fn())),
    )
    compiled = asyncio.run(
        compiler.compile(
            AgentSpec(
                id="agent-1",
                name="a",
                model=ModelPolicy(primary="test-model"),
                instructions=PromptSpec(instructions="hi"),
            )
        )
    )
    runner = _make_runner(tmp_path)
    _seed_session(runner._session_store, "session-1")

    async def _run():
        return await runner.run(
            compiled, RunInput(prompt="what is the answer?"), _run_context()
        )

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

    compiler = AgentCompiler(
        tool_executor=ToolExecutor(policy=PolicyEngine(rules=())),
        model_router=ModelRouter(registry=_registry(_boom)),
    )
    compiled = asyncio.run(
        compiler.compile(
            AgentSpec(
                id="agent-2",
                name="a",
                model=ModelPolicy(primary="test-model"),
                instructions=PromptSpec(instructions="hi"),
            )
        )
    )
    runner = _make_runner(tmp_path)
    _seed_session(runner._session_store, "session-2")

    async def _run():
        await runner.run(
            compiled,
            RunInput(prompt="hi"),
            RunContext(
                run_id="run-2",
                root_run_id="run-2",
                parent_run_id=None,
                session_id="session-2",
                runnable_id="agent-2",
                runnable_type=RunnableType.AGENT,
                user_id=None,
                tenant_id=None,
                workspace=None,
            ),
        )

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
    compiler = AgentCompiler(
        tool_executor=ToolExecutor(policy=PolicyEngine(rules=())),
        model_router=ModelRouter(registry=_registry(_model_fn())),
        middleware_pipeline=pipeline,
    )
    compiled = asyncio.run(
        compiler.compile(
            AgentSpec(
                id="agent-3",
                name="a",
                model=ModelPolicy(primary="test-model"),
                instructions=PromptSpec(instructions="hi"),
            )
        )
    )
    runner = _make_runner(tmp_path, pipeline=pipeline)
    _seed_session(runner._session_store, "session-3")

    async def _run():
        await runner.run(
            compiled,
            RunInput(prompt="hi"),
            RunContext(
                run_id="run-3",
                root_run_id="run-3",
                parent_run_id=None,
                session_id="session-3",
                runnable_id="agent-3",
                runnable_type=RunnableType.AGENT,
                user_id=None,
                tenant_id=None,
                workspace=None,
            ),
        )

    asyncio.run(_run())
    assert log == ["before_run", "after_run"]


def test_capabilities_have_no_mutable_state_before_or_after_run(tmp_path):
    # PolicyCapability / MiddlewareCapability
    # carry no mutable per-Run field at all -- the per-Run ToolContext reaches
    # them via pydantic-ai DI (ctx.deps.tool_context). A run leaves the
    # CompiledAgent byte-for-byte unchanged (the concurrency-safety invariant).
    compiler = AgentCompiler(
        tool_executor=ToolExecutor(policy=PolicyEngine(rules=())),
        model_router=ModelRouter(registry=_registry(_model_fn())),
    )
    compiled = asyncio.run(
        compiler.compile(
            AgentSpec(
                id="agent-4",
                name="a",
                model=ModelPolicy(primary="test-model"),
                instructions=PromptSpec(instructions="hi"),
            )
        )
    )
    assert not hasattr(compiled.policy_capability, "current_context")
    runner = _make_runner(tmp_path)
    _seed_session(runner._session_store, "session-4")

    async def _run():
        await runner.run(
            compiled,
            RunInput(prompt="hi"),
            RunContext(
                run_id="run-4",
                root_run_id="run-4",
                parent_run_id=None,
                session_id="session-4",
                runnable_id="agent-4",
                runnable_type=RunnableType.AGENT,
                user_id=None,
                tenant_id=None,
                workspace=None,
            ),
        )

    asyncio.run(_run())
    assert not hasattr(compiled.policy_capability, "current_context")


# -- Memory + Knowledge prompt injection ------------------------------------
# FunctionModel sees the FULL prompt pydantic-ai was called with as a
# UserPromptPart inside the last ModelRequest.parts. An echo model returns that
# text (wrapped for pydantic-ai's default dict output validator) so the test can
# assert what was injected without poking at private runner state.

import json as _json  # noqa: E402


def _echo_model_fn(text_when_missing: str = "no-prompt-captured"):
    def _fn(messages, info: AgentInfo) -> ModelResponse:
        prompt_text = text_when_missing
        for msg in reversed(messages):
            for part in reversed(getattr(msg, "parts", ()) or ()):
                content = getattr(part, "content", None)
                if isinstance(content, str) and content:
                    prompt_text = content
                    break
            if prompt_text != text_when_missing:
                break
        # Wrap with json.dumps so newlines/quotes in the prompt survive as a
        # valid JSON string for pydantic-ai's default dict output validator.
        return ModelResponse(
            parts=[
                TextPart(
                    content='{"response": {"echo": ' + _json.dumps(prompt_text) + "}}"
                )
            ]
        )

    return _fn


def _seed_memory(
    store,
    memory_id: str,
    content: str,
    owner_id: str = "session-1",
    tenant_id: str = "t1",
) -> None:
    from linktools.ai.memory.models import MemoryRecord

    now = datetime.now(timezone.utc)
    asyncio.run(
        store.remember(
            MemoryRecord(
                id=memory_id,
                tenant_id=tenant_id,
                owner_id=owner_id,
                content=content,
                category=None,
                confidence=None,
                version=1,
                created_at=now,
                updated_at=now,
                metadata={},
            )
        )
    )


def _make_runner_with_memory(tmp_path):
    from linktools.ai.storage.file.approval import FileApprovalStore
    from linktools.ai.storage.file.commit import FileRunCommitCoordinator
    from linktools.ai.storage.file.memory import FileMemoryStore

    run_store = FileRunStore(root=tmp_path / "runs")
    session_store = FileSessionStore(root=tmp_path / "sessions")
    event_store = FileEventStore(root=tmp_path / "events")
    checkpoint_store = FileCheckpointStore(root=tmp_path / "checkpoints")
    return AgentRunner(
        run_store=run_store,
        session_store=session_store,
        event_store=event_store,
        checkpoint_store=checkpoint_store,
        memory_store=FileMemoryStore(root=tmp_path / "memories"),
        commit_coordinator=FileRunCommitCoordinator(
            approval_store=FileApprovalStore(root=tmp_path / "approvals"),
            checkpoint_store=checkpoint_store,
            run_store=run_store,
            session_store=session_store,
            event_store=event_store,
        ),
    )


def test_memory_store_injection_prepends_memory_section_to_prompt(tmp_path):
    compiler = AgentCompiler(
        tool_executor=ToolExecutor(policy=PolicyEngine(rules=())),
        model_router=ModelRouter(registry=_registry(_echo_model_fn())),
    )
    compiled = asyncio.run(
        compiler.compile(
            AgentSpec(
                id="agent-mem",
                name="a",
                model=ModelPolicy(primary="test-model"),
                instructions=PromptSpec(instructions="hi"),
            )
        )
    )
    runner = _make_runner_with_memory(tmp_path)
    _seed_session(runner._session_store, "session-1")
    # FileMemoryStore.search is keyword-substring based, so the content must
    # contain the query ("hello") for the memory to match and be injected. The
    # memory is seeded under tenant "t1"; the run context carries the same
    # tenant so the DefaultMemoryPolicy's tenant-bound search finds it.
    _seed_memory(
        runner._memory_store,
        "mem-1",
        "hello: prefers terse answers (token user-secret-token-xyz)",
        owner_id="session-1",
        tenant_id="t1",
    )

    async def _run():
        return await runner.run(
            compiled, RunInput(prompt="hello"), _run_context(tenant_id="t1")
        )

    result = asyncio.run(_run())
    # Seeded under tenant "t1" with a matching run context, the memory matches
    # and is injected as a `## Memory` section.
    assert "user-secret-token-xyz" in str(result.output)
    assert "## Memory" in str(result.output)


def test_memory_store_none_default_leaves_prompt_unchanged(tmp_path):
    # Default runner (memory_store=None) must not inject anything: the echoed
    # prompt is exactly the user prompt (no history seeded -> no history text).
    compiler = AgentCompiler(
        tool_executor=ToolExecutor(policy=PolicyEngine(rules=())),
        model_router=ModelRouter(registry=_registry(_echo_model_fn())),
    )
    compiled = asyncio.run(
        compiler.compile(
            AgentSpec(
                id="agent-nomem",
                name="a",
                model=ModelPolicy(primary="test-model"),
                instructions=PromptSpec(instructions="hi"),
            )
        )
    )
    runner = _make_runner(tmp_path)
    _seed_session(runner._session_store, "session-1")

    async def _run():
        return await runner.run(
            compiled, RunInput(prompt="plain-prompt-token"), _run_context()
        )

    result = asyncio.run(_run())
    assert "## Memory" not in str(result.output)
    assert "## Knowledge" not in str(result.output)
    assert "plain-prompt-token" in str(result.output)


def test_retriever_injection_prepends_knowledge_section_to_prompt(tmp_path):
    from linktools.ai.knowledge.document import Document

    class _StubRetriever:
        async def search(self, query, *, scope, limit=10):
            return (
                Document(
                    id="doc-1",
                    content="known-fact-alpha",
                    score=None,
                    source="stub",
                    metadata={},
                ),
            )

    runner = AgentRunner(
        run_store=FileRunStore(root=tmp_path / "runs"),
        session_store=FileSessionStore(root=tmp_path / "sessions"),
        event_store=FileEventStore(root=tmp_path / "events"),
        checkpoint_store=FileCheckpointStore(root=tmp_path / "checkpoints"),
        retriever=_StubRetriever(),
        commit_coordinator=FileRunCommitCoordinator(
            approval_store=FileApprovalStore(root=tmp_path / "approvals"),
            checkpoint_store=FileCheckpointStore(root=tmp_path / "checkpoints"),
            run_store=FileRunStore(root=tmp_path / "runs"),
            session_store=FileSessionStore(root=tmp_path / "sessions"),
            event_store=FileEventStore(root=tmp_path / "events"),
        ),
    )
    compiler = AgentCompiler(
        tool_executor=ToolExecutor(policy=PolicyEngine(rules=())),
        model_router=ModelRouter(registry=_registry(_echo_model_fn())),
    )
    compiled = asyncio.run(
        compiler.compile(
            AgentSpec(
                id="agent-kn",
                name="a",
                model=ModelPolicy(primary="test-model"),
                instructions=PromptSpec(instructions="hi"),
            )
        )
    )
    _seed_session(runner._session_store, "session-1")

    async def _run():
        return await runner.run(
            compiled, RunInput(prompt="question"), _run_context(tenant_id="t1")
        )

    result = asyncio.run(_run())
    assert "known-fact-alpha" in str(result.output)
    assert "## Knowledge" in str(result.output)


def test_empty_memory_store_injects_no_memory_section(tmp_path):
    # Memory store is wired but has no matching records -> format_memory returns
    # "" -> no `## Memory` section added -> output unchanged from the no-memory
    # baseline.
    compiler = AgentCompiler(
        tool_executor=ToolExecutor(policy=PolicyEngine(rules=())),
        model_router=ModelRouter(registry=_registry(_echo_model_fn())),
    )
    compiled = asyncio.run(
        compiler.compile(
            AgentSpec(
                id="agent-empty",
                name="a",
                model=ModelPolicy(primary="test-model"),
                instructions=PromptSpec(instructions="hi"),
            )
        )
    )
    runner = _make_runner_with_memory(tmp_path)
    _seed_session(runner._session_store, "session-1")

    async def _run():
        return await runner.run(
            compiled, RunInput(prompt="unmatched-query-token"), _run_context()
        )

    result = asyncio.run(_run())
    assert "## Memory" not in str(result.output)
    assert "unmatched-query-token" in str(result.output)


# --- ModelPolicy.timeout_seconds + max_tokens enforcement -------------------


def test_run_model_timeout_transitions_run_to_failed(tmp_path):
    """ModelPolicy.timeout_seconds wraps agent.run in asyncio.wait_for; a model
    that sleeps past the timeout -> run FAILED with a descriptive 'model timeout'
    message (the asyncio.TimeoutError is translated before the FAILED handler)."""

    async def _slow_fn(messages, info: AgentInfo) -> ModelResponse:
        await asyncio.sleep(10)
        return ModelResponse(parts=[TextPart(content="done")])

    registry = ModelRegistry()
    registry.register("test-model", model=FunctionModel(_slow_fn))
    compiler = AgentCompiler(
        tool_executor=ToolExecutor(policy=PolicyEngine(rules=())),
        model_router=ModelRouter(registry=registry),
    )
    compiled = asyncio.run(
        compiler.compile(
            AgentSpec(
                id="agent-to",
                name="a",
                model=ModelPolicy(primary="test-model", timeout_seconds=0.05),
                instructions=PromptSpec(instructions="hi"),
            )
        )
    )
    runner = _make_runner(tmp_path)
    _seed_session(runner._session_store, "session-to")

    async def _run():
        await runner.run(
            compiled,
            RunInput(prompt="hi"),
            RunContext(
                run_id="run-to",
                root_run_id="run-to",
                parent_run_id=None,
                session_id="session-to",
                runnable_id="agent-to",
                runnable_type=RunnableType.AGENT,
                user_id=None,
                tenant_id=None,
                workspace=None,
            ),
        )

    with pytest.raises(Exception):
        asyncio.run(_run())

    rec = asyncio.run(runner._run_store.get("run-to"))
    assert rec.status == RunStatus.FAILED
    assert rec.error is not None
    assert "model timeout" in rec.error.message


def test_run_max_tokens_exceeded_transitions_run_to_failed(tmp_path):
    """ModelPolicy.max_tokens: when the model returns usage whose
    input+output > max_tokens, the run is transitioned to FAILED before the
    SUCCEEDED transition and the error is re-raised."""
    from pydantic_ai.usage import RunUsage

    def _heavy_fn(messages, info: AgentInfo) -> ModelResponse:
        return ModelResponse(
            parts=[TextPart(content='{"response": {"answer": 1}}')],
            usage=RunUsage(input_tokens=1000, output_tokens=1000),
        )

    registry = ModelRegistry()
    registry.register("test-model", model=FunctionModel(_heavy_fn))
    compiler = AgentCompiler(
        tool_executor=ToolExecutor(policy=PolicyEngine(rules=())),
        model_router=ModelRouter(registry=registry),
    )
    compiled = asyncio.run(
        compiler.compile(
            AgentSpec(
                id="agent-mt",
                name="a",
                model=ModelPolicy(primary="test-model", max_tokens=50),
                instructions=PromptSpec(instructions="hi"),
            )
        )
    )
    runner = _make_runner(tmp_path)
    _seed_session(runner._session_store, "session-mt")

    async def _run():
        await runner.run(
            compiled,
            RunInput(prompt="hi"),
            RunContext(
                run_id="run-mt",
                root_run_id="run-mt",
                parent_run_id=None,
                session_id="session-mt",
                runnable_id="agent-mt",
                runnable_type=RunnableType.AGENT,
                user_id=None,
                tenant_id=None,
                workspace=None,
            ),
        )

    with pytest.raises(Exception):
        asyncio.run(_run())

    rec = asyncio.run(runner._run_store.get("run-mt"))
    assert rec.status == RunStatus.FAILED
    assert rec.error is not None
    assert "max_tokens" in rec.error.message


def test_run_under_max_tokens_succeeds_and_records_usage(tmp_path):
    """When usage fits under max_tokens, the run SUCCEEDS and the returned
    RunResult carries the model's token usage (so the swarm can accumulate)."""
    from pydantic_ai.usage import RunUsage

    def _light_fn(messages, info: AgentInfo) -> ModelResponse:
        return ModelResponse(
            parts=[TextPart(content='{"response": {"answer": 42}}')],
            usage=RunUsage(input_tokens=10, output_tokens=5),
        )

    registry = ModelRegistry()
    registry.register("test-model", model=FunctionModel(_light_fn))
    compiler = AgentCompiler(
        tool_executor=ToolExecutor(policy=PolicyEngine(rules=())),
        model_router=ModelRouter(registry=registry),
    )
    compiled = asyncio.run(
        compiler.compile(
            AgentSpec(
                id="agent-ok",
                name="a",
                model=ModelPolicy(primary="test-model", max_tokens=100),
                instructions=PromptSpec(instructions="hi"),
            )
        )
    )
    runner = _make_runner(tmp_path)
    _seed_session(runner._session_store, "session-ok")

    async def _run():
        return await runner.run(
            compiled,
            RunInput(prompt="hi"),
            RunContext(
                run_id="run-ok",
                root_run_id="run-ok",
                parent_run_id=None,
                session_id="session-ok",
                runnable_id="agent-ok",
                runnable_type=RunnableType.AGENT,
                user_id=None,
                tenant_id=None,
                workspace=None,
            ),
        )

    result = asyncio.run(_run())

    rec = asyncio.run(runner._run_store.get("run-ok"))
    assert rec.status == RunStatus.SUCCEEDED
    # token_usage is now populated from run_result.usage (the accounting hook).
    assert result.token_usage.get("input_tokens") == 10
    assert result.token_usage.get("output_tokens") == 5


def test_run_without_timeout_or_max_tokens_preserves_current_behavior(tmp_path):
    """Defaults (timeout_seconds=None, max_tokens=None) must reproduce the
    baseline lifecycle byte-for-byte -- no wait_for wrapper, no usage check."""
    compiler = AgentCompiler(
        tool_executor=ToolExecutor(policy=PolicyEngine(rules=())),
        model_router=ModelRouter(registry=_registry(_model_fn())),
    )
    compiled = asyncio.run(
        compiler.compile(
            AgentSpec(
                id="agent-def",
                name="a",
                model=ModelPolicy(primary="test-model"),
                instructions=PromptSpec(instructions="hi"),
            )
        )
    )
    runner = _make_runner(tmp_path)
    _seed_session(runner._session_store, "session-def")

    async def _run():
        return await runner.run(
            compiled,
            RunInput(prompt="hi"),
            RunContext(
                run_id="run-def",
                root_run_id="run-def",
                parent_run_id=None,
                session_id="session-def",
                runnable_id="agent-def",
                runnable_type=RunnableType.AGENT,
                user_id=None,
                tenant_id=None,
                workspace=None,
            ),
        )

    result = asyncio.run(_run())
    assert "42" in str(result.output)
    rec = asyncio.run(runner._run_store.get("run-def"))
    assert rec.status == RunStatus.SUCCEEDED


def test_cost_budget_exceeded_raises(tmp_path):
    """WP-14 §18: ModelPolicy.budget is enforced via a ModelPricingProvider --
    a response whose token cost exceeds the Decimal budget raises
    ModelPolicyExceededError (the budget field is no longer inert)."""
    from decimal import Decimal

    from pydantic_ai.usage import RunUsage

    from linktools.ai.errors import ModelPolicyExceededError
    from linktools.ai.model.pricing import ModelPricing, StaticModelPricingProvider

    def _usage_model(messages, info):
        return ModelResponse(
            parts=[TextPart(content='{"response": "x"}')],
            usage=RunUsage(input_tokens=1000, output_tokens=1000),
        )

    registry = ModelRegistry()
    registry.register("test-model", model=FunctionModel(_usage_model))
    runner = _make_runner(tmp_path)
    runner._pricing_provider = StaticModelPricingProvider(
        {
            "test-model": ModelPricing(
                model_id="test-model",
                input_cost_per_token=Decimal("0.001"),
                output_cost_per_token=Decimal("0.001"),
            )
        }
    )
    spec = AgentSpec(
        id="b",
        name="b",
        model=ModelPolicy(
            primary="test-model", budget=Decimal("0.5")
        ),  # 2000 tokens * 0.001 = 2.0 > 0.5
        instructions=PromptSpec(instructions="hi"),
    )
    compiler = AgentCompiler(
        tool_executor=ToolExecutor(policy=PolicyEngine(rules=())),
        model_router=ModelRouter(registry=registry),
    )
    compiled = asyncio.run(compiler.compile(spec))

    async def _run():
        await runner.run(
            compiled, RunInput(prompt="hi"), _run_context("run-b", "session-1")
        )

    with pytest.raises(ModelPolicyExceededError):
        asyncio.run(_run())


def test_budget_without_pricing_fails_closed(tmp_path):
    """WP-14 §18.6: a ModelPolicy.budget set with no pricing provider is a
    configuration error -- the run refuses to start (fail-closed)."""
    from decimal import Decimal

    from linktools.ai.errors import ModelPolicyExceededError

    registry = ModelRegistry()
    registry.register("test-model", model=FunctionModel(_model_fn()))
    runner = _make_runner(tmp_path)
    # No pricing_provider wired.
    spec = AgentSpec(
        id="c",
        name="c",
        model=ModelPolicy(primary="test-model", budget=Decimal("1")),
        instructions=PromptSpec(instructions="hi"),
    )
    compiler = AgentCompiler(
        tool_executor=ToolExecutor(policy=PolicyEngine(rules=())),
        model_router=ModelRouter(registry=registry),
    )
    compiled = asyncio.run(compiler.compile(spec))

    async def _run():
        await runner.run(
            compiled, RunInput(prompt="hi"), _run_context("run-c", "session-1")
        )

    with pytest.raises(ModelPolicyExceededError):
        asyncio.run(_run())
