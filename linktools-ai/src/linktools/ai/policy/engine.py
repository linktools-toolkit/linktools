#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PolicyEngine: composes PolicyRules into one ALLOW/DENY/REQUIRE_APPROVAL
decision. Middleware never makes authorization decisions itself -- only
PolicyEngine does.

The decision types and the PolicyRule Protocol live in rule.py and are
re-exported here so existing imports from policy.engine keep working."""

from .rule import (
    PolicyDecision,
    PolicyDecisionKind,
    PolicyRule,
    ToolContext,
    ToolRequest,
)

__all__ = [
    "PolicyDecision",
    "PolicyDecisionKind",
    "PolicyEngine",
    "PolicyRule",
    "ToolContext",
    "ToolRequest",
]


class PolicyEngine:
    def __init__(self, *, rules: "tuple[PolicyRule, ...]") -> None:
        self._rules = rules

    async def evaluate(self, request: ToolRequest, context: ToolContext) -> PolicyDecision:
        approval_decision: "PolicyDecision | None" = None
        for rule in self._rules:
            decision = await rule.evaluate(request, context)
            if decision.kind == PolicyDecisionKind.DENY:
                return decision
            if decision.kind == PolicyDecisionKind.REQUIRE_APPROVAL and approval_decision is None:
                approval_decision = decision
        if approval_decision is not None:
            return approval_decision
        return PolicyDecision(kind=PolicyDecisionKind.ALLOW, rule_id="default", reason=None)
