import inspect
import pytest

from linktools.ai.errors import ToolDeniedError, ToolSecurityAuditError
from linktools.ai.tool.models import ToolDescriptor
from linktools.ai.governance.security.pipeline import (
    PipelineAction,
    PipelineDecision,
    SecurityPipeline,
)
from linktools.ai.tool.managed import ManagedToolAdapter


class _Executor:
    async def is_approved(self, run_id, call_id):
        return False

    async def execute(self, request, context, handler, **kwargs):
        result = await handler(**request.arguments)
        processor = kwargs.get("result_processor")
        return result if processor is None else await processor(result)


def _executor():
    return _Executor()


class _RecordingEmitter:
    """Captures which channel (security vs observability) each event is sent to.
    The adapter must pick the channel explicitly at the call site; this records
    the choice so a misrouted event is detectable."""

    def __init__(self) -> None:
        self.security: "list" = []
        self.observability: "list" = []

    async def emit_security(self, event) -> None:
        self.security.append(event)

    async def emit_observability(self, event) -> None:
        self.observability.append(event)


def _descriptor(**kw) -> ToolDescriptor:
    base = dict(name="t", source="test", category="c", risk="low", mutating=False)
    base.update(kw)
    return ToolDescriptor(**base)


def _types(events) -> "list[str]":
    return [type(e).__name__ for e in events]


@pytest.mark.asyncio
async def test_policy_resolved_routes_to_security_channel():
    async def handler(x: str = "d") -> str:
        return f"ok:{x}"

    em = _RecordingEmitter()
    adapter = ManagedToolAdapter(
        descriptor=_descriptor(),
        handler=handler,
        tool_executor=_executor(),
        security_event_emitter=em,
    )
    await adapter.invoke(x="hi")

    assert "ToolPolicyResolved" in _types(em.security)
    assert "ToolPolicyResolved" not in _types(em.observability)


@pytest.mark.asyncio
async def test_lifecycle_events_route_to_observability_channel():
    async def handler(x: str = "d") -> str:
        return f"ok:{x}"

    em = _RecordingEmitter()
    adapter = ManagedToolAdapter(
        descriptor=_descriptor(),
        handler=handler,
        tool_executor=_executor(),
        security_event_emitter=em,
    )
    await adapter.invoke(x="hi")

    obs = _types(em.observability)
    assert "ToolStarted" in obs
    assert "ToolCompleted" in obs
    # ToolCompleted is observability even though it records the call outcome --
    # it must never land in the security fail-closed channel.
    assert "ToolCompleted" not in _types(em.security)
    assert "ToolStarted" not in _types(em.security)


@pytest.mark.asyncio
async def test_pipeline_decision_routes_to_security_channel():
    class _AllowPipeline(SecurityPipeline):
        async def before_tool(self, e):
            return PipelineDecision(action=PipelineAction.ALLOW)

        async def after_tool(self, e):
            return PipelineDecision(action=PipelineAction.ALLOW)

    async def handler(x: str = "d") -> str:
        return "ok"

    em = _RecordingEmitter()
    adapter = ManagedToolAdapter(
        descriptor=_descriptor(),
        handler=handler,
        tool_executor=_executor(),
        security_pipeline=_AllowPipeline(),
        security_event_emitter=em,
    )
    await adapter.invoke(x="hi")

    assert "ToolPipelineDecision" in _types(em.security)
    # The before/after markers are observability; only the decision is security.
    assert "ToolPipelineBefore" in _types(em.observability)
    assert "ToolPipelineAfter" in _types(em.observability)
    assert "ToolPipelineBefore" not in _types(em.security)


@pytest.mark.asyncio
async def test_security_degraded_routes_to_security_channel():
    # A failing policy provider triggers the degraded path, which must emit
    # SecurityDegraded through the security channel (the audit-relevant one),
    # not the observability channel.
    from types import SimpleNamespace

    class _BoomProvider:
        async def resolve(self, descriptor, context):
            raise RuntimeError("provider down")

    async def handler(x: str = "d") -> str:
        return "ok"

    em = _RecordingEmitter()
    adapter = ManagedToolAdapter(
        descriptor=_descriptor(),
        handler=handler,
        tool_executor=_executor(),
        policy_provider=_BoomProvider(),
        security_event_emitter=em,
        run_context=SimpleNamespace(run_id="r1"),
    )
    with pytest.raises(ToolDeniedError):
        await adapter.invoke(x="hi")

    assert "SecurityDegraded" in _types(em.security)
    assert "SecurityDegraded" not in _types(em.observability)


def test_managed_adapter_does_not_route_events_by_class_name():
    # Structural guard: routing must be explicit per call site, never inferred
    # from type(event).__name__ in a set -- that approach silently misroutes any
    # new event type whose name was not added to the set.
    source = inspect.getsource(ManagedToolAdapter)
    assert "__name__ in security_events" not in source
    assert "security_events = {" not in source
    assert "_emit_security" in source
    assert "_emit_observability" in source


@pytest.mark.asyncio
async def test_security_audit_failure_is_fail_closed_via_store():
    # A security event whose persistence fails must fail closed; an
    # observability event is always best-effort. The store-backed emitter
    # wraps the persistence failure into ToolSecurityAuditError; the adapter
    # propagates it so an unrecorded security decision blocks the call.
    from linktools.ai.governance.security.emitter import EventStoreSecurityEventEmitter

    class _FailingStore:
        async def append(self, **kw):
            raise RuntimeError("disk full")

    async def handler(x: str = "d") -> str:
        return "ok"

    adapter = ManagedToolAdapter(
        descriptor=_descriptor(),
        handler=handler,
        tool_executor=_executor(),
        security_audit_failure_mode="fail_closed",
        security_event_emitter=EventStoreSecurityEventEmitter(
            _FailingStore(), failure_mode="fail_closed"
        ),
    )
    # The first event is the ToolPolicyResolved security audit; fail_closed
    # turns the store failure into ToolSecurityAuditError before execution.
    with pytest.raises(ToolSecurityAuditError):
        await adapter.invoke(x="hi")


@pytest.mark.asyncio
async def test_security_event_sink_emitter_fail_closed_vs_best_effort():
    """The production SecurityEventSinkEmitter bridge (used by execute_pure)
    must propagate a sink persistence failure on emit_security (fail-closed:
    an unrecorded security decision blocks the call) but swallow it on
    emit_observability (best-effort: an observability record never masks the
    decision being audited). This is the bridge the legacy
    EventStoreSecurityEventEmitter used to provide directly."""
    from linktools.ai.run.live_events import SecurityEventSinkEmitter

    class _FailingSink:
        async def emit(self, event):
            raise RuntimeError("sink down")

    emitter = SecurityEventSinkEmitter(_FailingSink())

    # emit_security propagates the failure.
    with pytest.raises(RuntimeError, match="sink down"):
        await emitter.emit_security(object())

    # emit_observability swallows it (no raise).
    await emitter.emit_observability(object())
