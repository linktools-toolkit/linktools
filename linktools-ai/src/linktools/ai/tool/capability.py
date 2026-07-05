#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PolicyCapability: adapts ToolExecutor into a real pydantic-ai
AbstractCapability, converting ToolDeniedError/ToolApprovalRequiredError into
SkipToolExecution so a denied call surfaces as a tool result the model can
see, matching security/hook.py's SecurityCapability's existing error-surfacing
pattern.

current_context is a MUTABLE field, not constructor-baked -- a CompiledAgent
is compiled once and reused across many real Runs, each with its own
run_id/session_id. AgentRunner (Task 11) sets current_context immediately
before each agent.pydantic_agent.run(...) call and clears it afterward.
Concurrent Runs sharing the same CompiledAgent would race on this field --
that's a known, explicitly out-of-scope limitation of this phase, not
something this design hides."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.exceptions import SkipToolExecution

from ..errors import ToolApprovalRequiredError, ToolDeniedError
from ..policy.engine import ToolContext, ToolRequest
from .executor import ToolExecutor

if TYPE_CHECKING:
    from pydantic_ai import RunContext
    from pydantic_ai.messages import ToolCallPart
    from pydantic_ai.tools import ToolDefinition


@dataclass
class PolicyCapability(AbstractCapability[None]):
    executor: ToolExecutor
    current_context: "ToolContext | None" = None

    async def before_tool_execute(
        self,
        ctx: "RunContext[Any]",
        *,
        call: "ToolCallPart",
        tool_def: "ToolDefinition",
        args: Any,
    ) -> Any:
        request = ToolRequest(tool_name=tool_def.name, arguments=args)
        context = self.current_context or ToolContext(run_id="unknown", session_id="unknown")
        try:
            await self.executor.check(request, context)
        except ToolDeniedError as exc:
            raise SkipToolExecution({"error": str(exc)}) from exc
        except ToolApprovalRequiredError as exc:
            raise SkipToolExecution({"error": str(exc)}) from exc
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
        return result

    async def on_tool_execute_error(
        self,
        ctx: "RunContext[Any]",
        *,
        call: "ToolCallPart",
        tool_def: "ToolDefinition",
        args: Any,
        error: BaseException,
    ) -> Any:
        raise SkipToolExecution({"error": f"{type(error).__name__}: {error}"}) from error


def build_policy_capability(executor: ToolExecutor) -> PolicyCapability:
    return PolicyCapability(executor=executor)
