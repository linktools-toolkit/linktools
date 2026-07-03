#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""HookedMCPCapability: per-MCP-server AgentCapability wrapping pydantic_ai.capabilities.MCP.

Each MCP server spec gets its own capability instance (its own `self.server_name`), so
hook attribution doesn't need toolset-origin identity from ToolDefinition/RunContext —
identity lives on `self`, resolving the limitation that made a single shared
wrap_tool_execute infeasible for the combined builtin+MCP toolset case.
"""

import json
import time
from dataclasses import dataclass, field
from typing import Any

from pydantic_ai.capabilities import WrapperCapability


def _normalize_mcp_payload(server: str, raw: Any) -> Any:
    """Inject the legacy server/status/data_gaps envelope onto MCP results."""
    if isinstance(raw, dict):
        inner = raw.get("result") if isinstance(raw.get("result"), dict) else raw
        inner.setdefault("server", server)
        inner.setdefault("status", "ok")
        inner.setdefault("data_gaps", [])
        return inner
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict):
            parsed.setdefault("server", server)
            parsed.setdefault("status", "ok")
            parsed.setdefault("data_gaps", [])
            return parsed
    return {"server": server, "status": "ok", "data_gaps": [], "data": raw}


@dataclass
class HookedMCPCapability(WrapperCapability):
    server_name: str = ""
    kernel: Any = None
    context: "dict[str, Any]" = field(default_factory=dict)
    parent_call_id: "str | None" = None

    async def wrap_tool_execute(self, ctx: Any, *, call: Any, tool_def: Any, args: Any, handler: Any) -> Any:
        t = time.monotonic()
        success = True
        error: "str | None" = None
        result: Any = None
        data_gaps: "list[Any]" = []
        if self.kernel:
            self.kernel.trigger(
                "mcp_call_start",
                **self.context,
                server=self.server_name,
                tool_name=tool_def.name,
                arguments=args,
                call_id=call.tool_call_id,
                parent_call_id=self.parent_call_id,
            )
        try:
            raw = await handler(args)
            result = _normalize_mcp_payload(self.server_name, raw)
            if isinstance(result, dict):
                data_gaps = result.get("data_gaps") or []
            return result
        except Exception as exc:
            success = False
            error = str(exc)
            raise
        finally:
            if self.kernel:
                self.kernel.trigger(
                    "post_mcp_call",
                    **self.context,
                    server=self.server_name,
                    tool_name=tool_def.name,
                    duration_ms=round((time.monotonic() - t) * 1000, 2),
                    success=success,
                    data_gaps=data_gaps,
                    result=result,
                    error=error,
                    call_id=call.tool_call_id,
                    parent_call_id=self.parent_call_id,
                    tool_use_id=call.tool_call_id or tool_def.name,
                    source="mcp",
                )
