#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for AgentEngine observability integration (via the Store-free
execute_pure).

Verifies that when ``observability`` and ``metrics`` are wired into AgentEngine:
- a successful run opens exactly one outer "agent.run" span and one nested
  "agent.model" span (model span parented to run span), records
  counter("agent.run.completed") and histogram("agent.run.duration_ms");
- an expected model failure (ModelRoutingError) records
  counter("agent.run.failed") with error_type, both spans still end, and the
  outcome is AgentFailed;
- with both kwargs default-None, execute_pure behaves exactly as before (no
  sink is consulted) -- the regression guard.

FS-29: the legacy ``run()`` is gone; these now drive ``execute_pure`` directly.
execute_pure records agent.run.failed only for the narrow
``_EXPECTED_RUN_FAILURES`` set (a bare programming error propagates without
that counter), so the failure case uses ``ModelRoutingError`` -- the canonical
model-layer failure an expected-failure counter is meant to track."""

import asyncio
from datetime import datetime, timezone
from typing import Any, Mapping

from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from linktools.ai.agent.compiler import AgentCompiler
from linktools.ai.agent.engine import AgentEngine
from linktools.ai.agent.models import AgentCompleted, AgentFailed, AgentInput
from linktools.ai.agent.spec import AgentSpec, PromptSpec
from linktools.ai.errors import ModelRoutingError
from linktools.ai.model.registry import ModelRegistry
from linktools.ai.model.policy import ModelPolicy
from linktools.ai.model.resolver import ModelResolver
from linktools.ai.observability.tracing import Span
from linktools.ai.run.cancellation import CancellationToken
from linktools.ai.run.context import RunContext
from linktools.ai.run.live_events import NullRunLiveEventSink, NullSecurityEventSink
from linktools.ai.run.models import RunnableType
from linktools.ai.governance.policy.engine import PolicyEngine
from linktools.ai.tool.executor import GovernedToolInvoker


class _RecordingSink:
    """Fake ObservabilitySink that records every start/end/event call so tests
    can assert on span nesting and end-on-exception behaviour."""

    def __init__(self) -> None:
        self.started: "list[Span]" = []
        self.events: "list[tuple[str, dict]]" = []
        self.ended: "list[Span]" = []

    def start_span(
        self,
        name: str,
        *,
        attributes: "Mapping[str, Any] | None" = None,
        parent: "Span | None" = None,
    ) -> Span:
        span = Span(
            name=name,
            span_id=f"id-{len(self.started)}",
            parent_id=parent.span_id if parent is not None else None,
            started_at=datetime.now(timezone.utc),
            attributes=dict(attributes or {}),
        )
        self.started.append(span)
        return span

    def record_event(
        self, name: str, *, attributes: "Mapping[str, Any] | None" = None
    ) -> None:
        self.events.append((name, dict(attributes or {})))

    def end_span(self, span: Span) -> None:
        self.ended.append(span)


class _RecordingMetrics:
    """Fake ObservabilityMetrics that records every counter/histogram/gauge."""

    def __init__(self) -> None:
        self.counters: "list[tuple[str, int, dict]]" = []
        self.histograms: "list[tuple[str, float, dict]]" = []
        self.gauges: "list[tuple[str, float, dict]]" = []

    def counter(
        self,
        name: str,
        *,
        value: int = 1,
        attributes: "Mapping[str, Any] | None" = None,
    ) -> None:
        self.counters.append((name, value, dict(attributes or {})))

    def histogram(
        self, name: str, *, value: float, attributes: "Mapping[str, Any] | None" = None
    ) -> None:
        self.histograms.append((name, value, dict(attributes or {})))

    def gauge(
        self, name: str, *, value: float, attributes: "Mapping[str, Any] | None" = None
    ) -> None:
        self.gauges.append((name, value, dict(attributes or {})))


def _model_fn(text: str = '{"response": {"answer": 42}}'):
    def _fn(messages, info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart(content=text)])

    return _fn


def _boom_model_fn():
    def _fn(messages, info: AgentInfo) -> ModelResponse:
        raise ModelRoutingError("model exploded")

    return _fn


def _registry(model_fn):
    registry = ModelRegistry()
    registry.register("test-model", model=FunctionModel(model_fn))
    return registry


def _run_context(run_id="run-1", session_id="session-1") -> RunContext:
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


def _compile(model_fn):
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


def _execute_pure(runner, compiled, prompt="hi"):
    return asyncio.run(
        runner.execute_pure(
            compiled,
            AgentInput(prompt=prompt),
            _run_context(),
            cancellation=CancellationToken(),
            live_events=NullRunLiveEventSink(),
            security_events=NullSecurityEventSink(),
        )
    )


def _make_runner(*, sink=None, metrics=None):
    # FS-29: AgentEngine takes only its pure-execution dependencies -- no
    # run/session/event/checkpoint Store, no commit_coordinator. observability
    # + metrics are the only wiring these tests exercise.
    return AgentEngine(observability=sink, metrics=metrics)


def test_observability_records_run_and_model_spans_with_metrics_on_success(tmp_path):
    sink = _RecordingSink()
    metrics = _RecordingMetrics()
    compiled = _compile(_model_fn())
    runner = _make_runner(sink=sink, metrics=metrics)

    outcome = _execute_pure(runner, compiled, prompt="what is the answer?")
    assert isinstance(outcome, AgentCompleted)
    assert "42" in str(outcome.result.output)

    # Exactly one outer "agent.run" span and one nested "agent.model" span.
    assert [s.name for s in sink.started] == ["agent.run", "agent.model"]
    # LIFO end order: model span ends before run span.
    assert [s.name for s in sink.ended] == ["agent.model", "agent.run"]
    run_span = sink.started[0]
    model_span = sink.started[1]
    # Inner model span is parented to the outer run span.
    assert model_span.parent_id == run_span.span_id
    # Outer run span carries run_id + session_id attributes.
    assert run_span.attributes.get("run_id") == "run-1"
    assert run_span.attributes.get("session_id") == "session-1"

    # Metrics: exactly one completed counter + one duration histogram.
    completed = [c for c in metrics.counters if c[0] == "agent.run.completed"]
    failed = [c for c in metrics.counters if c[0] == "agent.run.failed"]
    assert len(completed) == 1
    assert len(failed) == 0
    assert completed[0][2].get("run_id") == "run-1"
    durations = [h for h in metrics.histograms if h[0] == "agent.run.duration_ms"]
    assert len(durations) == 1
    assert durations[0][1] >= 0.0  # value is ms, non-negative
    assert durations[0][2].get("run_id") == "run-1"


def test_observability_records_failed_counter_and_ends_span_on_model_error(tmp_path):
    sink = _RecordingSink()
    metrics = _RecordingMetrics()
    compiled = _compile(_boom_model_fn())
    runner = _make_runner(sink=sink, metrics=metrics)

    outcome = _execute_pure(runner, compiled)

    # An expected model failure (ModelRoutingError) converges to AgentFailed --
    # not re-raised -- and both spans still started+ended (use_span ends on
    # exception, LIFO order).
    assert isinstance(outcome, AgentFailed)
    assert [s.name for s in sink.started] == ["agent.run", "agent.model"]
    assert [s.name for s in sink.ended] == ["agent.model", "agent.run"]
    # Failed counter recorded exactly once; no completed counter, no histogram.
    completed = [c for c in metrics.counters if c[0] == "agent.run.completed"]
    failed = [c for c in metrics.counters if c[0] == "agent.run.failed"]
    assert len(completed) == 0
    assert len(failed) == 1
    assert failed[0][2].get("run_id") == "run-1"
    assert "error_type" in failed[0][2]
    durations = [h for h in metrics.histograms if h[0] == "agent.run.duration_ms"]
    assert len(durations) == 0


def test_default_none_observability_is_a_noop(tmp_path):
    # Default runner (observability=None, metrics=None): execute_pure succeeds
    # exactly as today; no sink is consulted, no metrics recorded. This is the
    # regression guard.
    compiled = _compile(_model_fn())
    runner = _make_runner()  # no sink/metrics -> defaults to None
    assert runner._observability is None
    assert runner._metrics is None

    outcome = _execute_pure(runner, compiled)
    assert isinstance(outcome, AgentCompleted)
    assert "42" in str(outcome.result.output)
