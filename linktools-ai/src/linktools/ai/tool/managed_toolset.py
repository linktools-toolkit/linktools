#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ManagedToolsetWrapper: wraps any pydantic-ai AbstractToolset so that every
call_tool invocation passes through the security governance chain. Used for
toolsets that don't expose individual handlers (e.g. MCPToolset), where
per-function ManagedToolAdapter wrapping is not possible.

Integrates with ToolExecutor.check() for policy/approval when a ToolExecutor
is wired, so MCP and opaque toolset calls go through the same PolicyEngine +
approval flow as builtin tools."""

import uuid
from typing import TYPE_CHECKING, Any

from pydantic_ai.toolsets import WrapperToolset

from ..errors import RunPaused, ToolDeniedError
from ..security.descriptor import ToolDescriptor
from ..security.pipeline import (
    PipelineAction,
    PipelineDecision,
    SecurityPipeline,
    ToolInvocationEvent,
    ToolResultEvent,
)

if TYPE_CHECKING:
    from .policy import ResolvedToolPolicy, ToolPolicyProvider
    from ..run.context import RunContext


class ManagedToolsetWrapper(WrapperToolset):
    """Wraps an AbstractToolset (e.g. MCPToolset) so every call_tool goes
    through SecurityPipeline before_tool/after_tool + ToolExecutor.check when
    wired."""

    def __init__(
        self,
        wrapped: Any,
        *,
        descriptor: ToolDescriptor,
        security_pipeline: "SecurityPipeline | None" = None,
        tool_executor: Any = None,
        run_context: "RunContext | None" = None,
    ) -> None:
        super().__init__(wrapped)
        self._descriptor = descriptor
        self._pipeline = security_pipeline
        self._tool_executor = tool_executor
        self._run_context = run_context

    async def call_tool(self, name, tool_args, ctx, tool):
        run_id = getattr(self._run_context, "run_id", None) if self._run_context else None
        call_id = str(uuid.uuid4())

        # 1. Pipeline before_tool.
        if self._pipeline is not None:
            event = ToolInvocationEvent(
                tool_name=name, arguments=tool_args, run_id=run_id,
            )
            try:
                decision = await self._pipeline.before_tool(event)
            except Exception:
                raise ToolDeniedError(f"pipeline before_tool failed for {name!r}")
            if decision.action == PipelineAction.DENY:
                raise ToolDeniedError(
                    decision.reason or f"tool {name!r} denied by pipeline")
            if decision.action == PipelineAction.REQUIRE_APPROVAL:
                raise RunPaused(
                    run_id=run_id or "", approval_id=str(uuid.uuid4()),
                    tool_call_id=call_id, tool_name=name,
                    reason=decision.reason or "pipeline requires approval",
                    arguments=dict(tool_args) if isinstance(tool_args, dict) else {},
                )
            if decision.action == PipelineAction.MODIFY and decision.modified_payload:
                tool_args = decision.modified_payload

        # 2. ToolExecutor.check (policy engine + approval).
        if self._tool_executor is not None:
            from ..policy.engine import ToolContext, ToolRequest
            tc = ToolContext(
                run_id=run_id or "",
                session_id=getattr(self._run_context, "session_id", None),
                tool_call_id=call_id,
            )
            request = ToolRequest(tool_name=name, arguments=tool_args)
            await self._tool_executor.check(request, tc)

        # 3. Execute (after_tool fails closed on error).
        success = True
        result = None
        try:
            result = await self.wrapped.call_tool(name, tool_args, ctx, tool)
        except Exception:
            success = False
            raise
        finally:
            # 4. Pipeline after_tool (fail closed on error).
            if self._pipeline is not None:
                result_event = ToolResultEvent(
                    tool_name=name, result=result if success else None,
                    success=success, run_id=run_id,
                )
                try:
                    after = await self._pipeline.after_tool(result_event)
                    if after.action == PipelineAction.DENY and success:
                        raise ToolDeniedError(
                            f"tool {name!r} result denied by after_tool pipeline")
                except ToolDeniedError:
                    raise
                except Exception:
                    raise ToolDeniedError(
                        f"after_tool pipeline failed for {name!r}")

        return result
