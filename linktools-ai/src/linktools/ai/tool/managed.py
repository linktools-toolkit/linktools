#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ManagedToolAdapter: the single entry point through which every model-driven
tool call must pass. Wraps a raw handler with the full governance chain:

    descriptor -> ToolPolicyProvider -> SecurityBaseline merge ->
    SecurityPipeline.before_tool -> ToolExecutor.check (policy/approval) ->
    handler (timeout + retry) -> SecurityPipeline.after_tool -> stable result

Providers supply raw handlers + descriptors; the adapter handles ALL generic
governance. When a ToolExecutor is wired, the adapter delegates policy/approval
to it (reusing the existing PolicyEngine + approval store) so there is no
duplicate governance. When no executor is wired, the adapter applies its own
timeout/retry from the merged ResolvedToolPolicy."""

import asyncio
import logging
import uuid
from typing import TYPE_CHECKING, Any, Awaitable, Callable

from ..errors import RunPaused, ToolDeniedError, ToolTimeoutError, TransientToolError
from ..security.descriptor import ToolDescriptor
from ..security.pipeline import (
    PipelineAction,
    PipelineDecision,
    SecurityPipeline,
    ToolInvocationEvent,
    ToolResultEvent,
)
from .policy import ResolvedToolPolicy, ToolInvocationContext, ToolPolicyProvider, merge_policies

if TYPE_CHECKING:
    from ..run.context import RunContext

_LOGGER = logging.getLogger(__name__)


class ManagedToolAdapter:
    """Wraps a raw tool handler with unified security governance. When a
    ToolExecutor is wired, delegates policy/approval/idempotency to it; the
    adapter adds pipeline before/after and policy resolution on top."""

    def __init__(
        self,
        *,
        descriptor: ToolDescriptor,
        handler: "Callable[..., Awaitable[Any]]",
        tool_executor: Any = None,
        policy_provider: "ToolPolicyProvider | None" = None,
        security_pipeline: "SecurityPipeline | None" = None,
        baseline_policy: "ResolvedToolPolicy | None" = None,
        run_context: "RunContext | None" = None,
    ) -> None:
        self._descriptor = descriptor
        self._handler = handler
        self._tool_executor = tool_executor
        self._policy_provider = policy_provider
        self._pipeline = security_pipeline
        self._baseline = baseline_policy
        self._run_context = run_context

    async def invoke(self, **arguments: Any) -> Any:
        ctx = self._run_context
        call_id = str(uuid.uuid4())

        # 1. Resolve + merge policy.
        provider_policy = None
        if self._policy_provider is not None and ctx is not None:
            try:
                provider_policy = await self._policy_provider.resolve(self._descriptor, ctx)
            except Exception:
                raise ToolDeniedError(
                    f"tool policy resolution failed for {self._descriptor.name!r}")
        policy = merge_policies(None, self._baseline, provider_policy)
        if not policy.enabled:
            raise ToolDeniedError(f"tool {self._descriptor.name!r} is disabled by policy")

        # 2. Pipeline before_tool.
        if self._pipeline is not None:
            event = ToolInvocationEvent(
                tool_name=self._descriptor.name, arguments=arguments,
                run_id=getattr(ctx, "run_id", None) if ctx else None,
            )
            try:
                decision = await self._pipeline.before_tool(event)
            except Exception:
                raise ToolDeniedError(
                    f"pipeline before_tool failed for {self._descriptor.name!r}")
            if decision.action == PipelineAction.DENY:
                raise ToolDeniedError(
                    decision.reason or f"tool {self._descriptor.name!r} denied by pipeline")
            if decision.action == PipelineAction.REQUIRE_APPROVAL:
                raise RunPaused(
                    run_id=getattr(ctx, "run_id", "") or "",
                    approval_id=str(uuid.uuid4()),
                    tool_call_id=call_id,
                    tool_name=self._descriptor.name,
                    reason=decision.reason or "pipeline requires approval",
                    arguments=dict(arguments),
                )
            if decision.action == PipelineAction.MODIFY and decision.modified_payload:
                arguments = dict(decision.modified_payload)

        # 3. ToolExecutor.check (policy engine + approval flow).
        if self._tool_executor is not None:
            from ..policy.engine import ToolContext, ToolRequest
            tc = ToolContext(
                run_id=getattr(ctx, "run_id", "") or "",
                session_id=getattr(ctx, "session_id", None),
                tool_call_id=call_id,
            )
            request = ToolRequest(
                tool_name=self._descriptor.name,
                arguments=arguments,
            )
            await self._tool_executor.check(request, tc)

        # 4. Require approval from policy (if set and no executor handled it).
        if policy.require_approval and self._tool_executor is None:
            raise RunPaused(
                run_id=getattr(ctx, "run_id", "") or "",
                approval_id=str(uuid.uuid4()),
                tool_call_id=call_id,
                tool_name=self._descriptor.name,
                reason=f"policy requires approval for {self._descriptor.name!r}",
                arguments=dict(arguments),
            )

        # 5. Execute handler (timeout + retry from merged policy).
        max_attempts = 1 + policy.max_retries
        can_retry = not (self._descriptor.mutating and not policy.idempotent)
        result = None
        for attempt in range(1, max_attempts + 1):
            try:
                if policy.timeout_seconds is not None:
                    result = await asyncio.wait_for(
                        self._handler(**arguments), timeout=policy.timeout_seconds)
                else:
                    result = await self._handler(**arguments)
                break
            except asyncio.TimeoutError:
                raise ToolTimeoutError(
                    f"tool {self._descriptor.name!r} timed out after {policy.timeout_seconds}s")
            except TransientToolError:
                if attempt < max_attempts and can_retry:
                    continue
                raise

        # 5. Pipeline after_tool.
        if self._pipeline is not None:
            result_event = ToolResultEvent(
                tool_name=self._descriptor.name, result=result,
                success=True,
                run_id=getattr(ctx, "run_id", None) if ctx else None,
            )
            try:
                after_decision = await self._pipeline.after_tool(result_event)
            except Exception:
                _LOGGER.warning("after_tool pipeline error for %r (fail closed)",
                                self._descriptor.name)
                raise ToolDeniedError(
                    f"after_tool pipeline failed for {self._descriptor.name!r}")
            if after_decision.action == PipelineAction.DENY:
                raise ToolDeniedError(
                    f"tool {self._descriptor.name!r} result denied by after_tool pipeline")

        return result
