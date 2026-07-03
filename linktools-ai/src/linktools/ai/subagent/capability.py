#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SubagentCapability: call_subagent as an independent AgentCapability."""

import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from pydantic_ai import RunContext
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.toolsets import FunctionToolset


@dataclass
class SubagentCapability(AbstractCapability[None]):
    run_subagent_fn: "Callable[..., Awaitable[dict[str, Any]]]" = None  # type: ignore[assignment]
    allowed_subagents: "set[str]" = field(default_factory=set)
    kernel: Any = None
    trace_id: str = ""
    parent_call_id: "str | None" = None

    def get_toolset(self) -> FunctionToolset:
        toolset: FunctionToolset = FunctionToolset()
        run_subagent_fn = self.run_subagent_fn
        allowed_subagents = self.allowed_subagents

        async def call_subagent(ctx: "RunContext[Any]", subagent_id: str, input: Any = None) -> "dict[str, Any]":
            """Invoke a declared subagent by ID and return its result."""
            if subagent_id not in allowed_subagents:
                return {"error": f"subagent_not_available: {subagent_id}"}
            return await run_subagent_fn(subagent_id, input, call_id=ctx.tool_call_id or subagent_id)

        toolset.add_function(call_subagent)
        return toolset

    async def wrap_tool_execute(self, ctx: Any, *, call: Any, tool_def: Any, args: Any, handler: Any) -> Any:
        t = time.monotonic()
        success = True
        subagent_id = args.get("subagent_id") if isinstance(args, dict) else None
        if self.kernel:
            self.kernel.trigger(
                "subagent_start",
                trace_id=self.trace_id,
                subagent_id=subagent_id,
                call_id=call.tool_call_id,
                parent_call_id=self.parent_call_id,
            )
        try:
            result = await handler(args)
            return result
        except Exception:
            success = False
            raise
        finally:
            if self.kernel:
                self.kernel.trigger(
                    "subagent_end",
                    trace_id=self.trace_id,
                    subagent_id=subagent_id,
                    duration_ms=round((time.monotonic() - t) * 1000, 2),
                    status="completed" if success else "failed",
                    call_id=call.tool_call_id,
                    parent_call_id=self.parent_call_id,
                )
