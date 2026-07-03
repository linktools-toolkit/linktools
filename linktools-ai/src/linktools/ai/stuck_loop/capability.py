#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""StuckLoopCapability: interrupt a model that keeps retrying an identically-failing
tool call, instead of letting it burn turns/budget on a call that will never succeed."""

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.exceptions import SkipToolExecution

if TYPE_CHECKING:
    from pydantic_ai import RunContext
    from pydantic_ai.messages import ToolCallPart
    from pydantic_ai.tools import ToolDefinition


def _signature(tool_name: str, args: Any) -> str:
    try:
        return f"{tool_name}:{json.dumps(args, sort_keys=True, default=str)}"
    except TypeError:
        return f"{tool_name}:{args!r}"


def _is_error_result(result: Any) -> bool:
    return isinstance(result, dict) and "error" in result


@dataclass
class StuckLoopCapability(AbstractCapability[None]):
    max_repeats: int = 3
    _failure_counts: "dict[str, int]" = field(default_factory=dict, repr=False, compare=False)

    async def before_tool_execute(
        self,
        ctx: "RunContext[Any]",
        *,
        call: "ToolCallPart",
        tool_def: "ToolDefinition",
        args: Any,
    ) -> Any:
        sig = _signature(tool_def.name, args)
        if self._failure_counts.get(sig, 0) >= self.max_repeats:
            raise SkipToolExecution({
                "error": (
                    f"blocked: '{tool_def.name}' has failed {self._failure_counts[sig]} times "
                    "in a row with these exact arguments — stopping to avoid a stuck loop"
                ),
            })
        return args

    async def after_tool_execute(
        self,
        ctx: "RunContext[Any]",
        *,
        call: "ToolCallPart",
        tool_def: "ToolDefinition",
        args: Any,
        result: Any,
    ) -> Any:
        sig = _signature(tool_def.name, args)
        if _is_error_result(result):
            self._failure_counts[sig] = self._failure_counts.get(sig, 0) + 1
        else:
            self._failure_counts.pop(sig, None)
        return result

    async def on_tool_execute_error(
        self,
        ctx: "RunContext[Any]",
        *,
        call: "ToolCallPart",
        tool_def: "ToolDefinition",
        args: Any,
        error: Exception,
    ) -> Any:
        sig = _signature(tool_def.name, args)
        self._failure_counts[sig] = self._failure_counts.get(sig, 0) + 1
        raise error
