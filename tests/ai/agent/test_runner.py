#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AgentEngine core execution tests via the Store-free ``execute_pure``.

These tests drive ``AgentEngine.execute_pure`` directly -- the engine's sole
public lifecycle entry point after FS-29 deleted the legacy ``run()`` /
``run_stream()`` / ``execute()`` generators. ``execute_pure`` touches NO
run_store / session_store / event_store / checkpoint_store / commit_coordinator
(RunCoordinator owns Run-record creation + transition, checkpoint/session
persistence, and event publication); the assertions here are therefore on the
returned ``AgentExecutionOutcome`` discriminated union, never on Store state.

The Run-lifecycle persistence invariants the deleted tests used to cover
(session/run/event/checkpoint writes on both success and model failure) are now
owned by RunCoordinator and exercised at the runtime/integration level
(``tests/ai/test_runtime.py``), not by the engine.

FS-29: expected provider/model/tool failures (ModelRoutingError,
ModelPolicyExceededError, ...) converge to an ``AgentFailed`` outcome --
execute_pure returns them rather than re-raising -- so the timeout / max_tokens
/ budget cases assert ``isinstance(outcome, AgentFailed)`` instead of
``pytest.raises``.
"""

import asyncio
import json
from datetime import datetime, timezone

from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from linktools.ai.agent.compiler import AgentCompiler
from linktools.ai.agent.engine import AgentEngine
from linktools.ai.agent.models import AgentCompleted, AgentFailed, AgentInput
from linktools.ai.agent.spec import AgentSpec, PromptSpec
from linktools.ai.governance.policy.engine import PolicyEngine
from linktools.ai.middleware.base import Middleware
from linktools.ai.middleware.pipeline import MiddlewarePipeline
from linktools.ai.model.policy import ModelPolicy
from linktools.ai.model.registry import ModelRegistry
from linktools.ai.model.resolver import ModelResolver
from linktools.ai.run.cancellation import CancellationToken
from linktools.ai.run.context import RunContext
from linktools.ai.run.live_events import NullRunLiveEventSink, NullSecurityEventSink
from linktools.ai.run.models import RunnableType
from linktools.ai.tool.executor import GovernedToolInvoker


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


def _execute(runner, compiled, prompt="hi", **ctx_kwargs):
    """Drive execute_pure with the Store-free null sinks it requires. FS-29:
    execute_pure touches no run/session/event/checkpoint Store, so nothing
    outside the returned outcome is observable. ``ctx_kwargs`` forward to
    ``_run_context`` (e.g. ``tenant_id=`` for the memory/retrieval cases)."""
    return asyncio.run(
        runner.execute_pure(
            compiled,
            AgentInput(prompt=prompt),
            _run_context(**ctx_kwargs),
            cancellation=CancellationToken(),
            live_events=NullRunLiveEventSink(),
            security_events=NullSecurityEventSink(),
        )
    )


def _make_runner(pipeline=None) -> AgentEngine:
    # FS-29: AgentEngine takes only its pure-execution dependencies -- no
    # run/session/event/checkpoint Store, no commit_coordinator. The middleware
    # + model drive under test lives entirely inside execute_pure.
    return AgentEngine(middleware_pipeline=pipeline)


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


def test_middleware_runner_hooks_fire_in_order_on_success():
    log: "list[str]" = []
    pipeline = MiddlewarePipeline(middlewares=(_RecordingMiddleware(log),))
    compiler = AgentCompiler(
        tool_executor=GovernedToolInvoker(policy=PolicyEngine(rules=())),
        model_resolver=ModelResolver(registry=_registry(_model_fn())),
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
    runner = _make_runner(pipeline=pipeline)

    outcome = _execute(runner, compiled, prompt="hi")

    assert isinstance(outcome, AgentCompleted)
    # execute_pure runs before_run/after_run on a new (non-resuming) run.
    assert log == ["before_run", "after_run"]


def test_capabilities_have_no_mutable_state_before_or_after_run():
    # PolicyCapability / MiddlewareCapability carry no mutable per-Run field --
    # the per-Run ToolContext reaches them via pydantic-ai DI
    # (ctx.deps.tool_context). A run leaves the CompiledAgent byte-for-byte
    # unchanged (the concurrency-safety invariant).
    compiler = AgentCompiler(
        tool_executor=GovernedToolInvoker(policy=PolicyEngine(rules=())),
        model_resolver=ModelResolver(registry=_registry(_model_fn())),
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
    runner = _make_runner()

    outcome = _execute(runner, compiled, prompt="hi")

    assert isinstance(outcome, AgentCompleted)
    assert not hasattr(compiled.policy_capability, "current_context")


# -- Memory + Knowledge prompt injection ------------------------------------
# FunctionModel sees the FULL prompt pydantic-ai was called with as a
# UserPromptPart inside the last ModelRequest.parts. An echo model returns that
# text (wrapped for pydantic-ai's default dict output validator) so the test can
# assert what was injected without poking at private runner state.


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
                    content='{"response": {"echo": ' + json.dumps(prompt_text) + "}}"
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


def _make_runner_with_memory(tmp_path) -> AgentEngine:
    from linktools.ai.memory.persistence.filesystem import FilesystemMemoryStore

    # FS-29: only the memory_store pure-execution dep is wired -- no run/
    # session/event/checkpoint Store, no commit_coordinator. execute_pure
    # searches the memory store via DefaultMemoryPolicy but writes nowhere.
    return AgentEngine(memory_store=FilesystemMemoryStore(root=tmp_path / "memories"))


def test_memory_store_injection_prepends_memory_section_to_prompt(tmp_path):
    compiler = AgentCompiler(
        tool_executor=GovernedToolInvoker(policy=PolicyEngine(rules=())),
        model_resolver=ModelResolver(registry=_registry(_echo_model_fn())),
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
    # FilesystemMemoryStore.search is keyword-substring based, so the content
    # must contain the query ("hello") for the memory to match and be injected.
    # The memory is seeded under tenant "t1"; the run context carries the same
    # tenant so DefaultMemoryPolicy's tenant-bound search finds it.
    _seed_memory(
        runner._memory_store,
        "mem-1",
        "hello: prefers terse answers (token user-secret-token-xyz)",
        owner_id="session-1",
        tenant_id="t1",
    )

    outcome = _execute(runner, compiled, prompt="hello", tenant_id="t1")

    assert isinstance(outcome, AgentCompleted)
    # Seeded under tenant "t1" with a matching run context, the memory matches
    # and is injected as a `## Memory` section that the echo model returns.
    assert "user-secret-token-xyz" in str(outcome.result.output)
    assert "## Memory" in str(outcome.result.output)


def test_memory_store_none_default_leaves_prompt_unchanged():
    # Default runner (memory_store=None) must not inject anything: the echoed
    # prompt is exactly the user prompt (no history -> no history text).
    compiler = AgentCompiler(
        tool_executor=GovernedToolInvoker(policy=PolicyEngine(rules=())),
        model_resolver=ModelResolver(registry=_registry(_echo_model_fn())),
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
    runner = _make_runner()

    outcome = _execute(runner, compiled, prompt="plain-prompt-token")

    assert isinstance(outcome, AgentCompleted)
    assert "## Memory" not in str(outcome.result.output)
    assert "## Knowledge" not in str(outcome.result.output)
    assert "plain-prompt-token" in str(outcome.result.output)


def test_retriever_injection_prepends_knowledge_section_to_prompt():
    from linktools.ai.retrieval.document import Document

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

    # FS-29: AgentEngine takes only the retriever pure-execution dep -- no
    # Stores, no commit_coordinator. execute_pure runs the DefaultRetrievalPolicy
    # against the stub and folds the result into a `## Knowledge` section.
    runner = AgentEngine(retriever=_StubRetriever())
    compiler = AgentCompiler(
        tool_executor=GovernedToolInvoker(policy=PolicyEngine(rules=())),
        model_resolver=ModelResolver(registry=_registry(_echo_model_fn())),
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

    outcome = _execute(runner, compiled, prompt="question", tenant_id="t1")

    assert isinstance(outcome, AgentCompleted)
    assert "known-fact-alpha" in str(outcome.result.output)
    assert "## Knowledge" in str(outcome.result.output)


def test_empty_memory_store_injects_no_memory_section(tmp_path):
    # Memory store is wired but has no matching records -> format_memory returns
    # "" -> no `## Memory` section added -> output unchanged from the no-memory
    # baseline.
    compiler = AgentCompiler(
        tool_executor=GovernedToolInvoker(policy=PolicyEngine(rules=())),
        model_resolver=ModelResolver(registry=_registry(_echo_model_fn())),
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

    outcome = _execute(runner, compiled, prompt="unmatched-query-token")

    assert isinstance(outcome, AgentCompleted)
    assert "## Memory" not in str(outcome.result.output)
    assert "unmatched-query-token" in str(outcome.result.output)


# --- ModelPolicy.timeout_seconds + max_tokens enforcement -------------------


def test_execute_pure_model_timeout_returns_agent_failed():
    """ModelPolicy.timeout_seconds wraps the model drive in asyncio.wait_for;
    a model that sleeps past the timeout converges to AgentFailed with a
    descriptive 'model timeout' message (ModelRoutingError is an expected
    failure -- execute_pure returns it rather than re-raising)."""

    async def _slow_fn(messages, info: AgentInfo) -> ModelResponse:
        # Only needs to outlive ModelPolicy.timeout_seconds (0.05s) to trigger
        # the timeout path; ~4x the timeout is the wait floor when the model
        # runs to completion.
        await asyncio.sleep(0.2)
        return ModelResponse(parts=[TextPart(content="done")])

    registry = ModelRegistry()
    registry.register("test-model", model=FunctionModel(_slow_fn))
    compiler = AgentCompiler(
        tool_executor=GovernedToolInvoker(policy=PolicyEngine(rules=())),
        model_resolver=ModelResolver(registry=registry),
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
    runner = _make_runner()

    outcome = _execute(runner, compiled, prompt="hi")

    assert isinstance(outcome, AgentFailed)
    assert "model timeout" in outcome.error.message


def test_execute_pure_max_tokens_exceeded_returns_agent_failed():
    """ModelPolicy.max_tokens: when the model returns usage whose
    input+output > max_tokens, execute_pure converges to AgentFailed (the
    ModelPolicyExceededError is an expected failure, not re-raised)."""
    from pydantic_ai.usage import RunUsage

    def _heavy_fn(messages, info: AgentInfo) -> ModelResponse:
        return ModelResponse(
            parts=[TextPart(content='{"response": {"answer": 1}}')],
            usage=RunUsage(input_tokens=1000, output_tokens=1000),
        )

    registry = ModelRegistry()
    registry.register("test-model", model=FunctionModel(_heavy_fn))
    compiler = AgentCompiler(
        tool_executor=GovernedToolInvoker(policy=PolicyEngine(rules=())),
        model_resolver=ModelResolver(registry=registry),
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
    runner = _make_runner()

    outcome = _execute(runner, compiled, prompt="hi")

    assert isinstance(outcome, AgentFailed)
    assert "max_tokens" in outcome.error.message


def test_execute_pure_under_max_tokens_succeeds_and_records_usage():
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
        tool_executor=GovernedToolInvoker(policy=PolicyEngine(rules=())),
        model_resolver=ModelResolver(registry=registry),
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
    runner = _make_runner()

    outcome = _execute(runner, compiled, prompt="hi")

    assert isinstance(outcome, AgentCompleted)
    # token_usage is populated from the model's reported usage.
    assert outcome.result.token_usage.get("input_tokens") == 10
    assert outcome.result.token_usage.get("output_tokens") == 5


def test_execute_pure_without_timeout_or_max_tokens_succeeds():
    """Defaults (timeout_seconds=None, max_tokens=None) reproduce the baseline
    behavior -- no wait_for wrapper, no usage check, just a completed run."""
    compiler = AgentCompiler(
        tool_executor=GovernedToolInvoker(policy=PolicyEngine(rules=())),
        model_resolver=ModelResolver(registry=_registry(_model_fn())),
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
    runner = _make_runner()

    outcome = _execute(runner, compiled, prompt="hi")

    assert isinstance(outcome, AgentCompleted)
    assert "42" in str(outcome.result.output)


def test_execute_pure_cost_budget_exceeded_returns_agent_failed():
    """ModelPolicy.budget is enforced via a ModelPricingProvider -- a response
    whose token cost exceeds the Decimal budget converges to AgentFailed (the
    ModelPolicyExceededError is an expected failure, not re-raised)."""
    from decimal import Decimal

    from pydantic_ai.usage import RunUsage

    from linktools.ai.model.pricing import ModelPricing, StaticModelPricingProvider

    def _usage_model(messages, info):
        return ModelResponse(
            parts=[TextPart(content='{"response": "x"}')],
            usage=RunUsage(input_tokens=1000, output_tokens=1000),
        )

    registry = ModelRegistry()
    registry.register("test-model", model=FunctionModel(_usage_model))
    runner = _make_runner()
    # pricing_provider is a valid AgentEngine __init__ kwarg; setting it as a
    # plain attribute after construction still works.
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
        tool_executor=GovernedToolInvoker(policy=PolicyEngine(rules=())),
        model_resolver=ModelResolver(registry=registry),
    )
    compiled = asyncio.run(compiler.compile(spec))

    outcome = _execute(runner, compiled, prompt="hi")

    assert isinstance(outcome, AgentFailed)


def test_execute_pure_budget_without_pricing_returns_agent_failed():
    """A ModelPolicy.budget set with no pricing provider is a configuration
    error -- execute_pure fails closed, converging to AgentFailed rather than
    running without a cost limit."""
    from decimal import Decimal

    registry = ModelRegistry()
    registry.register("test-model", model=FunctionModel(_model_fn()))
    runner = _make_runner()
    # No pricing_provider wired.
    spec = AgentSpec(
        id="c",
        name="c",
        model=ModelPolicy(primary="test-model", budget=Decimal("1")),
        instructions=PromptSpec(instructions="hi"),
    )
    compiler = AgentCompiler(
        tool_executor=GovernedToolInvoker(policy=PolicyEngine(rules=())),
        model_resolver=ModelResolver(registry=registry),
    )
    compiled = asyncio.run(compiler.compile(spec))

    outcome = _execute(runner, compiled, prompt="hi")

    assert isinstance(outcome, AgentFailed)
    assert "pricing" in outcome.error.message.lower()
