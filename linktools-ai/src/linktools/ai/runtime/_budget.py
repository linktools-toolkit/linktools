#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime admission at actual model and tool invocation boundaries."""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from pydantic_ai.capabilities import AbstractCapability, CapabilityOrdering, ValidatedToolArgs
from pydantic_ai.messages import ModelResponse, ToolCallPart
from pydantic_ai.tools import RunContext, ToolDefinition

from ..capability import AgentContext
from ..core import canonical_sha256
from .state._contracts import BudgetRepository
from ._tool_metrics import TOOL_METRICS_MANAGED_METADATA_KEY


@dataclass(frozen=True, slots=True)
class RunBudgetContext:
    repository: BudgetRepository
    scope_id: str
    execution_id: str
    agent_run_id: str

    async def run_model(
        self, request_id: str, handler: Callable[[], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        await self.repository.admit_model(self.scope_id, request_id)
        try:
            response = await handler()
        except BaseException:
            await self.repository.settle_model(self.scope_id, request_id, None)
            raise
        await self.repository.settle_model(
            self.scope_id, request_id,
            response.usage.input_tokens + response.usage.output_tokens,
        )
        return response

    async def admit_tool(
        self, call_id: str, *, operation_id: str | None = None, fence: int | None = None,
    ) -> None:
        identity = canonical_sha256({
            "execution_id": self.execution_id,
            "agent_run_id": self.agent_run_id,
            "tool_call_id": call_id,
            "operation_id": operation_id,
            "fence": fence,
        })
        await self.repository.admit_tool(self.scope_id, identity)


class RunBudgetCapability(AbstractCapability[AgentContext[object]]):
    """Gate capability tools; managed toolsets gate their actual uncached leaves."""

    def __init__(self, budget: RunBudgetContext) -> None:
        self.id = "linktools.ai.run-budget"
        self._budget = budget

    def get_ordering(self) -> CapabilityOrdering:
        return CapabilityOrdering(position="innermost")

    async def wrap_tool_execute(
        self, ctx: RunContext[AgentContext[object]], *, call: ToolCallPart,
        tool_def: ToolDefinition, args: ValidatedToolArgs,
        handler: Callable[[dict[str, Any]], Awaitable[Any]],
    ) -> Any:
        if not (tool_def.metadata is not None
                and tool_def.metadata.get(TOOL_METRICS_MANAGED_METADATA_KEY) is True):
            await self._budget.admit_tool(call.tool_call_id)
        return await handler(args)


__all__ = ["RunBudgetContext", "RunBudgetCapability"]
