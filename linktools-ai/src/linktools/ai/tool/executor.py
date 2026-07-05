#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ToolExecutor: consults PolicyEngine before a tool executes, translating
its decision into the corresponding domain error."""

from typing import Any, Awaitable, Callable

from ..errors import ToolApprovalRequiredError, ToolDeniedError
from ..policy.engine import PolicyDecisionKind, PolicyEngine, ToolContext, ToolRequest


class ToolExecutor:
    def __init__(self, *, policy: PolicyEngine) -> None:
        self._policy = policy

    async def check(self, request: ToolRequest, context: ToolContext) -> None:
        decision = await self._policy.evaluate(request, context)
        if decision.kind == PolicyDecisionKind.DENY:
            raise ToolDeniedError(decision.reason or f"tool denied: {request.tool_name}")
        if decision.kind == PolicyDecisionKind.REQUIRE_APPROVAL:
            raise ToolApprovalRequiredError(decision.reason or f"tool requires approval: {request.tool_name}")

    async def execute(
        self,
        request: ToolRequest,
        context: ToolContext,
        handler: "Callable[..., Awaitable[Any]]",
    ) -> Any:
        await self.check(request, context)
        return await handler(**dict(request.arguments))
