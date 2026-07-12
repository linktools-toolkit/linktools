#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ApprovalRule: escalates a tool call to REQUIRE_APPROVAL when it matches an
explicit name-based allowlist (`require_for`) OR when its declared SideEffectKind
is at/above `require_side_effect` (default DESTRUCTIVE). SideEffectKind is a
str enum (no ordinal), so rank comparison goes through _SIDE_EFFECT_RANK."""

from typing import Mapping

from .rule import (
    PolicyDecision,
    PolicyDecisionKind,
    SideEffectKind,
    ToolContext,
    ToolPolicyMetadata,
    ToolRequest,
)

_SIDE_EFFECT_RANK: "dict[SideEffectKind, int]" = {
    SideEffectKind.NONE: 0,
    SideEffectKind.READ_ONLY: 1,
    SideEffectKind.NAMESPACE_MUTATING: 2,
    SideEffectKind.DESTRUCTIVE: 3,
}


class ApprovalRule:
    def __init__(
        self,
        *,
        require_for: "frozenset[str]" = frozenset(),
        require_side_effect: SideEffectKind = SideEffectKind.DESTRUCTIVE,
        tool_metadata: "Mapping[str, ToolPolicyMetadata] | None" = None,
    ) -> None:
        self._require_for = require_for
        self._require_side_effect = require_side_effect
        self._tool_metadata: "Mapping[str, ToolPolicyMetadata]" = tool_metadata or {}

    async def evaluate(
        self, request: ToolRequest, context: ToolContext
    ) -> PolicyDecision:
        if request.tool_name in self._require_for:
            return PolicyDecision(
                kind=PolicyDecisionKind.REQUIRE_APPROVAL,
                rule_id="approval-rule",
                reason=f"tool {request.tool_name} requires approval",
            )
        meta = self._tool_metadata.get(request.tool_name)
        if meta is not None:
            if (
                _SIDE_EFFECT_RANK[meta.side_effect]
                >= _SIDE_EFFECT_RANK[self._require_side_effect]
            ):
                return PolicyDecision(
                    kind=PolicyDecisionKind.REQUIRE_APPROVAL,
                    rule_id="approval-rule",
                    reason=f"tool side_effect {meta.side_effect.value} requires approval",
                )
        return PolicyDecision(
            kind=PolicyDecisionKind.ALLOW, rule_id="approval-rule", reason=None
        )
