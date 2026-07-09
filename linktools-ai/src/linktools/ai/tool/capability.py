#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PolicyCapability: adapts ToolExecutor into a real pydantic-ai
AbstractCapability, converting ToolDeniedError/ToolApprovalRequiredError into
SkipToolExecution so a denied call surfaces as a tool result the model can
see, matching security/hook.py's SecurityCapability's existing error-surfacing
pattern.

The per-Run ToolContext arrives via pydantic-ai dependency injection:
``AgentRunner`` constructs an ``AgentDependencies(tool_context=...)`` and
passes it as ``deps=`` to ``agent.pydantic_agent.run()`` / ``.iter()``; the
capability reads it off ``ctx.deps.tool_context``. No mutable per-Run field on
the capability itself, so a single ``CompiledAgent`` (and thus a single
``PolicyCapability``) is safe to reuse across many concurrent Runs."""

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.exceptions import SkipToolExecution

from ..errors import ToolApprovalRequiredError, ToolDeniedError
from ..policy.engine import ToolRequest
from .executor import ToolExecutor

if TYPE_CHECKING:
    from pydantic_ai import RunContext
    from pydantic_ai.messages import ToolCallPart
    from pydantic_ai.tools import ToolDefinition


@dataclass
class PolicyCapability(AbstractCapability[None]):
    executor: ToolExecutor

    async def before_tool_execute(
        self,
        ctx: "RunContext[Any]",
        *,
        call: "ToolCallPart",
        tool_def: "ToolDefinition",
        args: Any,
    ) -> Any:
        request = ToolRequest(tool_name=tool_def.name, arguments=args)
        # Thread ToolCallPart.tool_call_id through ToolContext so the executor
        # keys ApprovalRequest.tool_call_id on the SAME id pydantic-ai's message
        # history uses -- the linchpin of resume (a re-driven call after
        # approve() must find the matching approval). ctx.deps.tool_context is
        # the per-Run base; copy-on-write via replace() rather than mutate it.
        base = ctx.deps.tool_context
        context = replace(base, tool_call_id=call.tool_call_id)
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
