#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PermissionRule: denies a tool call when the tool's declared Permission set
asks for anything the runner has not granted. Tools with no ToolPolicyMetadata
are unrestricted (default ALLOW) -- only declared tools are policed."""

from typing import Mapping

from .rule import (
    Permission,
    PolicyDecision,
    PolicyDecisionKind,
    ToolContext,
    ToolPolicyMetadata,
    ToolRequest,
)


class PermissionRule:
    def __init__(
        self,
        *,
        allowed: "frozenset[Permission]",
        tool_metadata: "Mapping[str, ToolPolicyMetadata] | None" = None,
    ) -> None:
        self._allowed = allowed
        self._tool_metadata: "Mapping[str, ToolPolicyMetadata]" = tool_metadata or {}

    async def evaluate(self, request: ToolRequest, context: ToolContext) -> PolicyDecision:
        meta = self._tool_metadata.get(request.tool_name)
        if meta is None:
            return PolicyDecision(kind=PolicyDecisionKind.ALLOW, rule_id="permission-rule", reason=None)
        disallowed = meta.permissions - self._allowed
        if disallowed:
            return PolicyDecision(
                kind=PolicyDecisionKind.DENY,
                rule_id="permission-rule",
                reason=f"tool requires {disallowed}",
            )
        return PolicyDecision(kind=PolicyDecisionKind.ALLOW, rule_id="permission-rule", reason=None)
