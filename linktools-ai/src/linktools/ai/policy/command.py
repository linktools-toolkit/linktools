#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CommandRule: denies terminal commands matching a configurable pattern list.
Replaces security/hook.py's SecurityCapability's hardcoded regex blocklist with
a PolicyRule -- same default patterns, now configurable."""

import re

from .engine import PolicyDecision, PolicyDecisionKind, ToolContext, ToolRequest

DEFAULT_DENIED_COMMAND_PATTERNS: "tuple[str, ...]" = (
    r"rm\s+-rf\s+/(\s|$)",
    r"dd\s+.*of=/dev/",
    r":\(\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;",
    r"mkfs\.",
    r">\s*/dev/sd[a-z]",
)


class CommandRule:
    def __init__(
        self, *, denied_patterns: "tuple[str, ...]" = DEFAULT_DENIED_COMMAND_PATTERNS
    ) -> None:
        self._compiled = tuple(re.compile(pattern) for pattern in denied_patterns)

    async def evaluate(
        self, request: ToolRequest, context: ToolContext
    ) -> PolicyDecision:
        # Category-based: a tool is subject to the command denylist because its
        # descriptor declares category="terminal", not because of what it
        # happens to be named -- renaming the terminal tool cannot evade this
        # rule as long as its descriptor still says "terminal".
        if request.category is not None:
            if request.category != "terminal":
                return PolicyDecision(
                    kind=PolicyDecisionKind.ALLOW, rule_id="command-rule", reason=None
                )
        elif request.tool_name not in ("bash", "terminal"):
            # Compat fallback: request.category is only populated by callers
            # that resolved a ToolDescriptor (ManagedToolAdapter, or
            # PolicyCapability with a descriptor lookup wired). A caller that
            # never threads descriptor info through falls back to the prior
            # name check rather than silently losing command-injection
            # protection.
            return PolicyDecision(
                kind=PolicyDecisionKind.ALLOW, rule_id="command-rule", reason=None
            )
        command = str(request.arguments.get("command", ""))
        for pattern in self._compiled:
            if pattern.search(command):
                return PolicyDecision(
                    kind=PolicyDecisionKind.DENY,
                    rule_id="command-rule",
                    reason=f"command matches denied pattern: {pattern.pattern}",
                )
        return PolicyDecision(
            kind=PolicyDecisionKind.ALLOW, rule_id="command-rule", reason=None
        )
