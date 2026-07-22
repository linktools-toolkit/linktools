#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""EventCriticality classification."""

from linktools.ai.events.criticality import EventCriticality, classify_event
from linktools.ai.events.payloads import (
    ApprovalRequested,
    RunCompleted,
    RunPaused,
    SecurityDegraded,
    ToolCompleted,
    ToolExposureDenied,
    ToolPipelineDecision,
    ToolPolicyResolved,
)


def test_state_critical_events():
    assert (
        classify_event(RunPaused(run_id="r", reason="x"))
        is EventCriticality.STATE_CRITICAL
    )
    assert (
        classify_event(ApprovalRequested(approval_id="a", tool_name="t", reason="x"))
        is EventCriticality.STATE_CRITICAL
    )


def test_security_critical_events():
    assert (
        classify_event(SecurityDegraded(run_id="r", component="c", reason="x"))
        is EventCriticality.SECURITY_CRITICAL
    )
    assert (
        classify_event(ToolExposureDenied(agent_id="a", reason="x"))
        is EventCriticality.SECURITY_CRITICAL
    )
    assert (
        classify_event(
            ToolPolicyResolved(
                run_id="r",
                tool_name="t",
                enabled=True,
                timeout_seconds=None,
                max_retries=0,
                idempotent=False,
                require_approval=False,
                risk="low",
            )
        )
        is EventCriticality.SECURITY_CRITICAL
    )
    assert (
        classify_event(
            ToolPipelineDecision(
                run_id="r",
                tool_name="t",
                call_id="c",
                action="allow",
                reason="",
                stage="before",
            )
        )
        is EventCriticality.SECURITY_CRITICAL
    )


def test_observability_events():
    assert classify_event(RunCompleted(run_id="r")) is EventCriticality.OBSERVABILITY
    assert (
        classify_event(
            ToolCompleted(
                tool_name="t",
                tool_call_id="c",
                success=True,
                execution_success=True,
                result_action="returned",
            )
        )
        is EventCriticality.OBSERVABILITY
    )


def test_unknown_event_defaults_to_observability():
    class _Custom:
        pass

    assert classify_event(_Custom()) is EventCriticality.OBSERVABILITY
