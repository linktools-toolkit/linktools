#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AgentEngine.execute_pure: the target pure execution loop, added
alongside the legacy generator/collectors as an additive increment (WP9c,
round 1). Proves the new method never touches run_store/session_store/
commit_coordinator/run_controller (all wired to ``None`` here), drives live
events through the injected sink instead of a constructor-held event bus,
and classifies exceptions per the spec's exception table instead of one
generic except-Exception-return-FAILED catch-all."""

import asyncio

import pytest
from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from linktools.ai.agent.compiler import AgentCompiler
from linktools.ai.agent.engine import AgentEngine
from linktools.ai.agent.models import (
    AgentCancelled,
    AgentCompleted,
    AgentFailed,
    AgentInput,
    AgentPaused,
)
from linktools.ai.agent.spec import AgentSpec, PromptSpec, ToolRef
from linktools.ai.capability.resolver import CapabilityResolver
from linktools.ai.errors import RunInvariantError, RuntimeInitializationError
from linktools.ai.governance.policy.engine import PolicyEngine
from linktools.ai.model.policy import ModelPolicy
from linktools.ai.model.registry import ModelRegistry
from linktools.ai.model.resolver import ModelResolver
from linktools.ai.run.cancellation import CancellationToken
from linktools.ai.run.context import RunContext
from linktools.ai.run.live_events import NullRunLiveEventSink, NullSecurityEventSink
from linktools.ai.run.models import RunnableType
from linktools.ai.tool.executor import GovernedToolInvoker


def _model_fn(text='{"response": {"answer": 42}}'):
    def _fn(messages, info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart(content=text)])

    return _fn


def _registry(model_fn):
    registry = ModelRegistry()
    registry.register("test-model", model=FunctionModel(model_fn))
    return registry


def _context(run_id="run-1", session_id="session-1") -> RunContext:
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


def _pure_engine(**overrides) -> AgentEngine:
    """Every Run-lifecycle Store dependency is ``None`` -- execute_pure()
    must never dereference them."""
    kwargs = dict(
        # FS-29: AgentEngine no longer accepts any Run-lifecycle Store --
        # run_store / session_store / event_store / commit_coordinator /
        # run_controller were removed. execute_pure must never dereference a
        # Store, and the engine constructed with bare defaults has none.
    )
    kwargs.update(overrides)
    return AgentEngine(**kwargs)


def _compiled(model_fn=None, tools=None, capability_resolver=None, managed_tool_executor=None):
    compiler = AgentCompiler(
        tool_executor=GovernedToolInvoker(policy=PolicyEngine(rules=())),
        model_resolver=ModelResolver(registry=_registry(model_fn or _model_fn())),
    )
    spec = AgentSpec(
        id="agent-1",
        name="a",
        model=ModelPolicy(primary="test-model"),
        instructions=PromptSpec(instructions="hi"),
        **({"tools": tools} if tools is not None else {}),
    )
    return asyncio.run(compiler.compile(spec))


class _CollectingLiveEvents:
    def __init__(self) -> None:
        self.events: "list[dict]" = []

    async def publish(self, event: dict) -> None:
        self.events.append(event)


def test_execute_pure_success_returns_agent_completed_with_no_store_access():
    engine = _pure_engine()
    compiled = _compiled()
    live = _CollectingLiveEvents()

    async def _run():
        return await engine.execute_pure(
            compiled,
            AgentInput(prompt="what is the answer?"),
            _context(),
            cancellation=CancellationToken(),
            live_events=live,
            security_events=NullSecurityEventSink(),
        )

    outcome = asyncio.run(_run())
    assert isinstance(outcome, AgentCompleted)
    assert "42" in str(outcome.result.output)
    assert [m.role.value for m in outcome.messages] == ["user", "assistant"]
    assert outcome.messages[0].content == "what is the answer?"
    assert outcome.checkpoint_payload
    # A FunctionModel without a stream_function returns a complete response
    # (no per-delta text events) -- the same non-streaming behavior as the
    # legacy generator. Output + messages + checkpoint are the load-bearing
    # assertions; live text deltas are covered by the streaming-model
    # tests/ai/agent/test_runner_stream.py path.


def test_execute_pure_resuming_true_skips_prompt_and_feeds_message_history():
    from linktools.ai.session.reader import _to_model_message
    from linktools.ai.session.models import MessageRole, SessionMessage
    from datetime import datetime, timezone

    engine = _pure_engine()
    compiled = _compiled()
    prior = _to_model_message(
        SessionMessage(
            id="m1",
            session_id="session-1",
            sequence=1,
            role=MessageRole.USER,
            content="what is the answer?",
            run_id="run-1",
            created_at=datetime.now(timezone.utc),
        )
    )

    async def _run():
        return await engine.execute_pure(
            compiled,
            AgentInput(prompt="", message_history=(prior,), resuming=True),
            _context(),
            cancellation=CancellationToken(),
            live_events=NullRunLiveEventSink(),
            security_events=NullSecurityEventSink(),
        )

    outcome = asyncio.run(_run())
    assert isinstance(outcome, AgentCompleted)
    assert "42" in str(outcome.result.output)
    # Resuming with an empty user_prompt appends only the ASSISTANT turn --
    # matching SessionRecorder.completed_messages' documented behavior.
    assert [m.role.value for m in outcome.messages] == ["assistant"]


def test_execute_pure_pause_returns_agent_paused_with_no_messages():
    from linktools.ai.capability.models import CapabilityBundle
    from linktools.ai.capability.provider import CapabilityProvider
    from linktools.ai.tool.models import (
        ManagedToolDefinition,
        ToolContribution,
        ToolDescriptor,
    )
    from linktools.ai.errors import RunPaused

    class _PauseProvider(CapabilityProvider):
        supported_kinds = ("test",)

        async def resolve(self, ref, context) -> CapabilityBundle:
            async def risky(x: int) -> int:
                raise RunPaused(
                    run_id=context.run_id,
                    approval_id="appr-1",
                    tool_name="risky",
                    reason="needs approval",
                )

            return CapabilityBundle(
                tool_contributions=(
                    ToolContribution(
                        tools=(
                            ManagedToolDefinition(
                                descriptor=ToolDescriptor(
                                    name="risky",
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

    engine = _pure_engine(
        capability_resolver=CapabilityResolver({"test": _PauseProvider()}),
        managed_tool_executor=GovernedToolInvoker(policy=PolicyEngine(rules=())),
    )
    # A model fn that emits a ToolCallPart for "risky" (turn 1) then TextPart
    # "done" once the tool has returned (turn 2). The risky handler raises
    # RunPaused on the first call, so we never reach turn 2.
    from pydantic_ai.messages import ToolCallPart

    def _call_risky(messages, info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[ToolCallPart(tool_name="risky", args={"x": 1})])

    compiled = _compiled(model_fn=_call_risky, tools=(ToolRef(kind="test", name="risky"),))
    live = _CollectingLiveEvents()

    async def _run():
        return await engine.execute_pure(
            compiled,
            AgentInput(prompt="call risky"),
            _context(),
            cancellation=CancellationToken(),
            live_events=live,
            security_events=NullSecurityEventSink(),
        )

    outcome = asyncio.run(_run())
    assert isinstance(outcome, AgentPaused)
    assert outcome.request.approval_id == "appr-1"
    assert outcome.messages == ()
    assert outcome.checkpoint_payload
    # the state-event split spec state-event split: the engine publishes ONLY process events
    # (text/tool/model_progress); it must NOT publish the "paused" state
    # event -- that is the RunCoordinator's job, emitted only AFTER the
    # durable pause commit succeeds. The outcome carries the pause
    # descriptor up to the Coordinator, which commits and then publishes.
    assert not any(e["type"] == "paused" for e in live.events)


def test_execute_pure_explicit_cancellation_returns_agent_cancelled():
    def _slow(messages, info: AgentInfo) -> ModelResponse:
        raise asyncio.CancelledError()

    engine = _pure_engine()
    compiled = _compiled(model_fn=_slow)
    token = CancellationToken()
    token.cancel()

    async def _run():
        return await engine.execute_pure(
            compiled,
            AgentInput(prompt="hi"),
            _context(),
            cancellation=token,
            live_events=NullRunLiveEventSink(),
            security_events=NullSecurityEventSink(),
        )

    outcome = asyncio.run(_run())
    assert isinstance(outcome, AgentCancelled)


def test_execute_pure_external_cancelled_error_propagates_unconverted():
    """A CancelledError that surfaces while the token was NEVER cancelled is
    an external cancel (e.g. asyncio shutdown) -- it must re-raise, not
    become AgentCancelled."""

    def _boom(messages, info: AgentInfo) -> ModelResponse:
        raise asyncio.CancelledError()

    engine = _pure_engine()
    compiled = _compiled(model_fn=_boom)

    async def _run():
        return await engine.execute_pure(
            compiled,
            AgentInput(prompt="hi"),
            _context(),
            cancellation=CancellationToken(),
            live_events=NullRunLiveEventSink(),
            security_events=NullSecurityEventSink(),
        )

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(_run())


def test_execute_pure_runtime_initialization_error_propagates_not_agent_failed():
    engine = _pure_engine(capability_resolver=None, managed_tool_executor=None)
    compiled = _compiled(tools=(ToolRef(kind="builtin", name="file"),))

    async def _run():
        return await engine.execute_pure(
            compiled,
            AgentInput(prompt="hi"),
            _context(),
            cancellation=CancellationToken(),
            live_events=NullRunLiveEventSink(),
            security_events=NullSecurityEventSink(),
        )

    with pytest.raises(RuntimeInitializationError):
        asyncio.run(_run())


def test_execute_pure_expected_model_failure_returns_agent_failed():
    """A TYPED expected provider/model failure (ModelRoutingError -- the model
    could not be reached/timed out) becomes AgentFailed. This is the
    allowlisted expected-failure row; contrast with the unknown-error and
    config-violation rows below, which must propagate."""
    from linktools.ai.errors import ModelRoutingError

    def _boom(messages, info: AgentInfo) -> ModelResponse:
        raise ModelRoutingError("model unreachable")

    engine = _pure_engine()
    compiled = _compiled(model_fn=_boom)

    async def _run():
        return await engine.execute_pure(
            compiled,
            AgentInput(prompt="hi"),
            _context(),
            cancellation=CancellationToken(),
            live_events=NullRunLiveEventSink(),
            security_events=NullSecurityEventSink(),
        )

    outcome = asyncio.run(_run())
    assert isinstance(outcome, AgentFailed)
    assert outcome.error.error_type == "ModelRoutingError"
    assert "model unreachable" in outcome.error.message


def test_execute_pure_unknown_programming_error_propagates_not_agent_failed():
    """An UNKNOWN programming error (TypeError -- the canonical 'real bug')
    must propagate as a raised exception, NOT be swallowed into a clean
    AgentFailed outcome. This is the direct assertion for the spec's
    'unknown programming error -> re-raise' row; a catch-all
    except-Exception-return-AgentFailed would hide the bug behind a
    'model failure' outcome instead."""

    def _bug(messages, info: AgentInfo) -> ModelResponse:
        raise TypeError("NoneType has no attribute split")

    engine = _pure_engine()
    compiled = _compiled(model_fn=_bug)

    async def _run():
        return await engine.execute_pure(
            compiled,
            AgentInput(prompt="hi"),
            _context(),
            cancellation=CancellationToken(),
            live_events=NullRunLiveEventSink(),
            security_events=NullSecurityEventSink(),
        )

    with pytest.raises(TypeError, match="split"):
        asyncio.run(_run())


def test_execute_pure_run_invariant_violation_propagates_not_agent_failed():
    """A configuration/invariant/protocol violation (RunInvariantError) must
    propagate, not become AgentFailed -- the run did not 'fail' in the
    expected-model-failure sense; the runtime contract was broken."""

    def _bug(messages, info: AgentInfo) -> ModelResponse:
        raise RunInvariantError("run completed without a pending commit")

    engine = _pure_engine()
    compiled = _compiled(model_fn=_bug)

    async def _run():
        return await engine.execute_pure(
            compiled,
            AgentInput(prompt="hi"),
            _context(),
            cancellation=CancellationToken(),
            live_events=NullRunLiveEventSink(),
            security_events=NullSecurityEventSink(),
        )

    with pytest.raises(RunInvariantError):
        asyncio.run(_run())


def test_execute_pure_max_tokens_exceeded_returns_agent_failed():
    engine = _pure_engine()
    compiler = AgentCompiler(
        tool_executor=GovernedToolInvoker(policy=PolicyEngine(rules=())),
        model_resolver=ModelResolver(registry=_registry(_model_fn())),
    )
    spec = AgentSpec(
        id="agent-1",
        name="a",
        model=ModelPolicy(primary="test-model", max_tokens=1),
        instructions=PromptSpec(instructions="hi"),
    )
    compiled = asyncio.run(compiler.compile(spec))

    async def _run():
        return await engine.execute_pure(
            compiled,
            AgentInput(prompt="hi"),
            _context(),
            cancellation=CancellationToken(),
            live_events=NullRunLiveEventSink(),
            security_events=NullSecurityEventSink(),
        )

    outcome = asyncio.run(_run())
    assert isinstance(outcome, AgentFailed)
    assert outcome.error.error_type == "ModelPolicyExceededError"
