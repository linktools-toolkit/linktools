#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""before_tool exception audit.

When the before_tool pipeline fails (raises OR returns an after-only action),
the adapter must: emit the right audit event, raise ToolDeniedError, and never
invoke the handler.
"""

import asyncio

import pytest

from linktools.ai.errors import ToolDeniedError
from linktools.ai.events.payloads import SecurityDegraded, ToolPipelineDecision
from linktools.ai.security.emitter import CollectingSecurityEventEmitter
from linktools.ai.security.emitter import DefaultSecurityEventSanitizer
from linktools.ai.tool.models import ToolDescriptor
from linktools.ai.security.pipeline import PipelineAction, PipelineDecision
from linktools.ai.tool.managed import ManagedToolAdapter
from linktools.ai.tool.executor import ToolExecutor
from linktools.ai.policy.engine import PolicyEngine
from linktools.ai.tool.policy import ResolvedToolPolicy


def _descriptor() -> ToolDescriptor:
    return ToolDescriptor(
        name="t", source="test", category="misc", risk="low", mutating=False
    )


def _adapter(
    *, pipeline, handler
) -> tuple[ManagedToolAdapter, CollectingSecurityEventEmitter]:
    collector = CollectingSecurityEventEmitter(
        sanitizer=DefaultSecurityEventSanitizer()
    )
    adapter = ManagedToolAdapter(
        descriptor=_descriptor(),
        handler=handler,
        tool_executor=ToolExecutor(policy=PolicyEngine(rules=())),
        policy_provider=None,
        security_pipeline=pipeline,
        baseline_policy=ResolvedToolPolicy(),
        security_event_emitter=collector,
    )
    return adapter, collector


class _RaisingBeforePipeline:
    async def before_tool(self, event):
        raise RuntimeError("boom")

    async def after_tool(self, event):
        return PipelineDecision(action=PipelineAction.ALLOW)


class _AfterOnlyBeforePipeline:
    async def before_tool(self, event):
        # DENY_RESULT is an after-stage action; invalid at before_tool.
        return PipelineDecision(
            action=PipelineAction.DENY_RESULT, reason="result blocked"
        )

    async def after_tool(self, event):
        return PipelineDecision(action=PipelineAction.ALLOW)


def test_before_tool_raising_emits_security_degraded_and_denies():
    called = {"count": 0}

    async def handler(**kw):
        called["count"] += 1
        return "ran"

    adapter, collector = _adapter(pipeline=_RaisingBeforePipeline(), handler=handler)
    with pytest.raises(ToolDeniedError):
        asyncio.run(adapter.invoke())
    # SecurityDegraded was emitted for the pipeline failure.
    degraded = [e for e in collector.security_events if isinstance(e, SecurityDegraded)]
    assert degraded, "expected a SecurityDegraded event for the before_tool failure"
    assert "before" in degraded[0].component or "pipeline" in degraded[0].component
    # The handler never ran.
    assert called["count"] == 0


def test_before_tool_after_only_action_emits_deny_decision_and_denies():
    called = {"count": 0}

    async def handler(**kw):
        called["count"] += 1
        return "ran"

    adapter, collector = _adapter(pipeline=_AfterOnlyBeforePipeline(), handler=handler)
    with pytest.raises(ToolDeniedError):
        asyncio.run(adapter.invoke())
    decisions = [
        e for e in collector.security_events if isinstance(e, ToolPipelineDecision)
    ]
    assert decisions, "expected a ToolPipelineDecision for the after-only before-action"
    assert decisions[0].action == PipelineAction.DENY.value
    assert decisions[0].stage == "before"
    # The handler never ran.
    assert called["count"] == 0
