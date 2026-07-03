#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SkillCapability: skill_view as an independent AgentCapability (Section: skill/subagent/MCP -> AbstractCapability)."""

import time
from dataclasses import dataclass
from typing import Any, Callable

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.toolsets import FunctionToolset


@dataclass
class SkillCapability(AbstractCapability[None]):
    skill_view_fn: "Callable[[dict[str, Any]], dict[str, Any]]" = None  # type: ignore[assignment]
    kernel: Any = None
    trace_id: str = ""
    parent_call_id: "str | None" = None

    def get_toolset(self) -> FunctionToolset:
        toolset: FunctionToolset = FunctionToolset()
        skill_view_fn = self.skill_view_fn

        async def skill_view(skill_id: "str | None" = None, file_path: "str | None" = None) -> "dict[str, Any]":
            """Load a skill's instructions and linked files; optionally read one file."""
            return skill_view_fn({"skill_id": skill_id, "file_path": file_path})

        toolset.add_function(skill_view)
        return toolset

    async def wrap_tool_execute(self, ctx: Any, *, call: Any, tool_def: Any, args: Any, handler: Any) -> Any:
        t = time.monotonic()
        success = True
        error: "str | None" = None
        result: Any = None
        if self.kernel:
            self.kernel.trigger(
                "mcp_call_start",
                trace_id=self.trace_id,
                server="builtin",
                tool_name=tool_def.name,
                arguments=args,
                call_id=call.tool_call_id,
                parent_call_id=self.parent_call_id,
            )
        try:
            result = await handler(args)
            return result
        except Exception as exc:
            success = False
            error = str(exc)
            raise
        finally:
            if self.kernel:
                self.kernel.trigger(
                    "post_mcp_call",
                    trace_id=self.trace_id,
                    server="builtin",
                    tool_name=tool_def.name,
                    duration_ms=round((time.monotonic() - t) * 1000, 2),
                    success=success,
                    data_gaps=[] if success else [f"builtin_tool_failed: {tool_def.name}"],
                    result=result,
                    error=error,
                    call_id=call.tool_call_id,
                    parent_call_id=self.parent_call_id,
                    tool_use_id=call.tool_call_id or tool_def.name,
                    source="builtin",
                )
