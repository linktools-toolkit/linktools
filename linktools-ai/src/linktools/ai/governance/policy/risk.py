#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RiskRule + UsageLimitRule.

RiskRule: denies a tool whose declared RiskLevel exceeds the configured cap.
UsageLimitRule: denies a call once a usage counter in ToolContext.metadata
exceeds a configured ceiling. Reads these context.metadata keys:

  - "tokens_used"   compared against limits["max_tokens"]
  - "calls_used"    compared against limits["max_calls"]

Both default to 0 when absent, so an empty/missing counter never trips the rule.
Other keys in `limits` are accepted but ignored at present (extensibility)."""

from typing import Mapping

from .rule import (
    PolicyDecision,
    PolicyDecisionKind,
    RiskLevel,
    ToolContext,
    ToolPolicyMetadata,
    ToolRequest,
)

_LIMIT_TO_USAGE_KEY: "Mapping[str, str]" = {
    "max_tokens": "tokens_used",
    "max_calls": "calls_used",
}


class RiskRule:
    def __init__(
        self,
        *,
        max_allowed: RiskLevel,
        tool_metadata: "Mapping[str, ToolPolicyMetadata]",
    ) -> None:
        self._max_allowed = max_allowed
        self._tool_metadata = tool_metadata

    async def evaluate(
        self, request: ToolRequest, context: ToolContext
    ) -> PolicyDecision:
        meta = self._tool_metadata.get(request.tool_name)
        if meta is None:
            return PolicyDecision(
                kind=PolicyDecisionKind.ALLOW, rule_id="risk-rule", reason=None
            )
        if meta.risk > self._max_allowed:
            return PolicyDecision(
                kind=PolicyDecisionKind.DENY,
                rule_id="risk-rule",
                reason=f"risk {meta.risk.name} exceeds max {self._max_allowed.name}",
            )
        return PolicyDecision(
            kind=PolicyDecisionKind.ALLOW, rule_id="risk-rule", reason=None
        )


class UsageLimitRule:
    def __init__(self, *, limits: "Mapping[str, int]") -> None:
        self._limits = limits

    async def evaluate(
        self, request: ToolRequest, context: ToolContext
    ) -> PolicyDecision:
        for limit_key, ceiling in self._limits.items():
            usage_key = _LIMIT_TO_USAGE_KEY.get(limit_key)
            if usage_key is None:
                continue
            used = int(context.metadata.get(usage_key, 0))
            if used > ceiling:
                return PolicyDecision(
                    kind=PolicyDecisionKind.DENY,
                    rule_id="asset-limit-rule",
                    reason=f"{usage_key} {used} exceeds {limit_key} {ceiling}",
                )
        return PolicyDecision(
            kind=PolicyDecisionKind.ALLOW, rule_id="asset-limit-rule", reason=None
        )
