#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PathRule: denies filesystem-touching tool calls whose target path escapes
the configured allowed_roots. Pulls the candidate path from request.arguments
under `path_argument` (default "path"). For the terminal tool, also walks the
shell command (`arguments["command"]`) and checks any token that looks like a
path (absolute, ./, ../, ~/)."""

import shlex
from pathlib import Path

from .rule import (
    PolicyDecision,
    PolicyDecisionKind,
    ToolContext,
    ToolRequest,
)


def _looks_like_path(token: str) -> bool:
    return (
        token.startswith("/")
        or token.startswith("./")
        or token.startswith("../")
        or token.startswith("~/")
        or "/" in token
    )


class PathRule:
    def __init__(
        self,
        *,
        allowed_roots: "tuple[Path, ...]",
        path_argument: str = "path",
    ) -> None:
        self._allowed_roots = tuple(root.resolve() for root in allowed_roots)
        self._path_argument = path_argument

    async def evaluate(
        self, request: ToolRequest, context: ToolContext
    ) -> PolicyDecision:
        candidates: list[str] = []
        path_value = request.arguments.get(self._path_argument)
        if isinstance(path_value, str):
            candidates.append(path_value)
        elif path_value is not None:
            candidates.append(str(path_value))

        if request.tool_name == "terminal":
            command = request.arguments.get("command")
            if isinstance(command, str):
                for token in shlex.split(command):
                    if _looks_like_path(token):
                        candidates.append(token)

        if not candidates:
            return PolicyDecision(
                kind=PolicyDecisionKind.ALLOW, rule_id="path-rule", reason=None
            )

        for candidate in candidates:
            resolved = Path(candidate).expanduser().resolve()
            if not any(resolved.is_relative_to(root) for root in self._allowed_roots):
                return PolicyDecision(
                    kind=PolicyDecisionKind.DENY,
                    rule_id="path-rule",
                    reason="path escapes allowed roots",
                )
        return PolicyDecision(
            kind=PolicyDecisionKind.ALLOW, rule_id="path-rule", reason=None
        )
