#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PolicyEngine: composes PolicyRules into one ALLOW/DENY/REQUIRE_APPROVAL
decision. Middleware never makes authorization decisions itself -- only
PolicyEngine does, per spec section 25."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Protocol, runtime_checkable


class PolicyDecisionKind(str, Enum):
    ALLOW = "allow"
    DENY = "deny"
    REQUIRE_APPROVAL = "require_approval"


@dataclass(frozen=True, slots=True)
class PolicyDecision:
    kind: PolicyDecisionKind
    rule_id: str
    reason: "str | None"
    metadata: "Mapping[str, Any]" = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ToolRequest:
    tool_name: str
    arguments: "Mapping[str, Any]"


@dataclass(frozen=True, slots=True)
class ToolContext:
    run_id: str
    session_id: str


@runtime_checkable
class PolicyRule(Protocol):
    async def evaluate(self, request: ToolRequest, context: ToolContext) -> PolicyDecision:
        ...


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
