#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""LoopGuardMiddleware: blocks a tool call whose (tool_name, arguments) signature
has failed max_repeats times in a row. Ports stuck_loop/capability.py's
StuckLoopCapability logic to the new Middleware Protocol (that pre-vNext module
is untouched by this plan)."""

import json
from typing import Any

from ..errors import ToolDeniedError
from .base import Middleware


def _signature(request: Any) -> str:
    return json.dumps(
        {"tool_name": request.tool_name, "arguments": dict(request.arguments)},
        sort_keys=True,
    )


class LoopGuardMiddleware(Middleware):
    def __init__(self, *, max_repeats: int = 3) -> None:
        self._max_repeats = max_repeats
        self._failure_counts: "dict[str, int]" = {}

    async def before_tool(self, context: Any, request: Any) -> Any:
        signature = _signature(request)
        if self._failure_counts.get(signature, 0) >= self._max_repeats:
            raise ToolDeniedError(
                f"tool call blocked after {self._max_repeats} consecutive failures: {request.tool_name}"
            )
        return request

    async def after_tool(self, context: Any, request: Any, result: Any) -> Any:
        signature = _signature(request)
        if isinstance(result, dict) and "error" in result:
            self._failure_counts[signature] = self._failure_counts.get(signature, 0) + 1
        else:
            self._failure_counts.pop(signature, None)
        return result
