#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SubagentCapability: call_subagent as an independent AgentCapability."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from pydantic_ai import RunContext
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.toolsets import FunctionToolset

from ..support.hooks import HookEvent


@dataclass
class SubagentCapability(AbstractCapability[None]):
    run_subagent_fn: Callable[..., Awaitable[dict[str, Any]]] = None  # type: ignore[assignment]
    allowed_subagents: set[str] = field(default_factory=set)
    hooks: Any = None
    trace_id: str = ""
    parent_call_id: str | None = None

    def get_toolset(self) -> FunctionToolset:
        toolset: FunctionToolset = FunctionToolset()
        run_subagent_fn = self.run_subagent_fn
        allowed_subagents = self.allowed_subagents

        async def call_subagent(ctx: RunContext[Any], subagent_id: str, input: Any = None) -> dict[str, Any]:
            """Invoke a declared subagent by ID and return its result."""
            if subagent_id not in allowed_subagents:
                return {"error": f"subagent_not_available: {subagent_id}"}
            return await run_subagent_fn(subagent_id, input, call_id=ctx.tool_call_id or subagent_id)

        toolset.add_function(call_subagent)
        return toolset

    async def wrap_tool_execute(self, ctx: Any, *, call: Any, tool_def: Any, args: Any, handler: Any) -> Any:
        t = time.monotonic()
        success = True
        error: str | None = None
        result: Any = None
        if self.hooks:
            self.hooks.fire(
                HookEvent.MCP_CALL_START,
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
            if self.hooks:
                self.hooks.fire(
                    HookEvent.POST_MCP_CALL,
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
