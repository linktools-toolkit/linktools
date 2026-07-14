#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Event criticality classification (WP-17). Every event payload maps to one of
three criticality levels that govern its failure policy: security/state-critical
events fail closed (a persistence failure blocks the operation); observability
events are best-effort (a failure is logged but does not block)."""

from enum import Enum
from typing import Any


class EventCriticality(str, Enum):
    STATE_CRITICAL = "state_critical"
    SECURITY_CRITICAL = "security_critical"
    OBSERVABILITY = "observability"


# Events whose persistence is bound to run/approval state -- a missing one
# means the state itself is inconsistent.
_STATE_CRITICAL: "frozenset[str]" = frozenset(
    {
        "ApprovalRequested",
        "RunPaused",
    }
)

# Security decisions (deny/expose/pipeline/degradation) -- must be auditable.
_SECURITY_CRITICAL: "frozenset[str]" = frozenset(
    {
        "SecurityDegraded",
        "ToolExposureDenied",
        "ToolPolicyResolved",
        "ToolPipelineBefore",
        "ToolPipelineAfter",
        "ToolPipelineDecision",
    }
)

# Everything else is observability (lifecycle markers, metrics, spans).
_DEFAULT = EventCriticality.OBSERVABILITY


def classify_event(payload: Any) -> EventCriticality:
    """Map an event payload (by its class name) to its criticality level."""
    name = type(payload).__name__
    if name in _STATE_CRITICAL:
        return EventCriticality.STATE_CRITICAL
    if name in _SECURITY_CRITICAL:
        return EventCriticality.SECURITY_CRITICAL
    return _DEFAULT
