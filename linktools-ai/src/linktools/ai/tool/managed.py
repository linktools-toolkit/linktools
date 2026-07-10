#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ManagedToolAdapter: the single entry point through which every model-driven
tool call must pass. Wraps a raw handler with the full governance chain:

    descriptor -> ToolPolicyProvider -> SecurityBaseline ->
    SecurityPipeline.before_tool -> approval -> handler execution ->
    SecurityPipeline.after_tool -> events -> stable result/error

Providers supply raw handlers + descriptors; the adapter handles ALL generic
governance. No provider should call ToolExecutor/PolicyEngine/Pipeline directly."""

import asyncio
import logging
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Mapping

from ..errors import ToolDeniedError, ToolTimeoutError
from ..security.descriptor import ToolDescriptor
from ..security.pipeline import (
    PipelineAction,
    PipelineDecision,
    SecurityPipeline,
    ToolInvocationEvent,
    ToolResultEvent,
)
from .policy import ResolvedToolPolicy, ToolInvocationContext, ToolPolicyProvider

if TYPE_CHECKING:
    from ..run.context import RunContext

_LOGGER = logging.getLogger(__name__)


class ManagedToolAdapter:
    """Wraps a raw tool handler with unified security governance."""

    def __init__(
        self,
        *,
        descriptor: ToolDescriptor,
        handler: "Callable[..., Awaitable[Any]]",
        policy_provider: "ToolPolicyProvider | None" = None,
        security_pipeline: "SecurityPipeline | None" = None,
        baseline_policy: "ResolvedToolPolicy | None" = None,
        run_context: "RunContext | None" = None,
    ) -> None:
        self._descriptor = descriptor
        self._handler = handler
        self._policy_provider = policy_provider
        self._pipeline = security_pipeline
        self._baseline = baseline_policy
        self._run_context = run_context

    async def invoke(self, **arguments: Any) -> Any:
        """Execute the tool through the full governance chain."""
        from .policy import merge_policies

        ctx = self._run_context
        call_id = str(id(arguments))  # simplified; production uses ToolContext

        # 1. Resolve policy.
        provider_policy = None
        if self._policy_provider is not None and ctx is not None:
            try:
                provider_policy = await self._policy_provider.resolve(self._descriptor, ctx)
            except Exception:
                # Fail closed: do not execute if policy resolution errors.
                raise ToolDeniedError(
                    f"tool policy resolution failed for {self._descriptor.name!r}")
        policy = merge_policies(None, self._baseline, provider_policy)

        if not policy.enabled:
            raise ToolDeniedError(f"tool {self._descriptor.name!r} is disabled by policy")

        # 2. Build invocation context.
        inv_ctx = ToolInvocationContext(
            descriptor=self._descriptor,
            arguments=arguments,
            run_context=ctx,
            policy=policy,
            call_id=call_id,
        )

        # 3. before_tool pipeline.
        if self._pipeline is not None:
            event = ToolInvocationEvent(
                tool_name=self._descriptor.name,
                arguments=arguments,
                run_id=getattr(ctx, "run_id", None) if ctx else None,
            )
            try:
                decision = await self._pipeline.before_tool(event)
            except Exception:
                raise ToolDeniedError(
                    f"security pipeline before_tool failed for {self._descriptor.name!r}")
            if decision.action == PipelineAction.DENY:
                raise ToolDeniedError(decision.reason or f"tool {self._descriptor.name!r} denied by pipeline")
            if decision.action == PipelineAction.REQUIRE_APPROVAL:
                # In a full implementation this raises RunPaused; here we deny
                # since the adapter is not yet wired into the approval flow.
                raise ToolDeniedError(
                    f"tool {self._descriptor.name!r} requires approval (pipeline)")
            if decision.action == PipelineAction.MODIFY and decision.modified_payload:
                arguments = dict(decision.modified_payload)

        # 4. Execute handler (with timeout + retry per policy).
        from ..errors import TransientToolError
        max_attempts = 1 + policy.max_retries
        can_retry = not (self._descriptor.mutating and not policy.idempotent)
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
            except Exception:
                raise

        # 5. after_tool pipeline.
        if self._pipeline is not None:
            result_event = ToolResultEvent(
                tool_name=self._descriptor.name,
                result=result,
                success=True,
                run_id=getattr(ctx, "run_id", None) if ctx else None,
            )
            try:
                after_decision = await self._pipeline.after_tool(result_event)
            except Exception:
                _LOGGER.warning("after_tool pipeline error for %r", self._descriptor.name)
                after_decision = PipelineDecision(action=PipelineAction.AUDIT_ONLY)
            if after_decision.action == PipelineAction.DENY:
                raise ToolDeniedError(
                    f"tool {self._descriptor.name!r} result denied by after_tool pipeline")

        return result


def build_managed_toolset(
    contributions: "tuple[Any, ...]",
    *,
    policy_provider: "ToolPolicyProvider | None" = None,
    security_pipeline: "SecurityPipeline | None" = None,
    baseline_policy: "ResolvedToolPolicy | None" = None,
    run_context: "RunContext | None" = None,
) -> "tuple[ManagedToolAdapter, ...]":
    """Wrap a tuple of ToolContributions into ManagedToolAdapters. Each
    contribution's handler is extracted from its toolset."""
    from pydantic_ai.toolsets import FunctionToolset
    from .contribution import ToolContribution

    adapters: "list[ManagedToolAdapter]" = []
    for contrib in contributions:
        if isinstance(contrib, ToolContribution):
            toolset = contrib.toolset
            for desc in contrib.descriptors:
                # Extract the raw handler from the FunctionToolset by name.
                handler = None
                tools = getattr(toolset, "tools", {})
                if isinstance(tools, dict) and desc.name in tools:
                    handler = tools[desc.name].function
                if handler is None:
                    continue
                adapters.append(ManagedToolAdapter(
                    descriptor=desc, handler=handler,
                    policy_provider=policy_provider,
                    security_pipeline=security_pipeline,
                    baseline_policy=baseline_policy,
                    run_context=run_context,
                ))
    return tuple(adapters)
