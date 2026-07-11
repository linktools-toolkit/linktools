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
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Mapping

from ..errors import RunPaused, ToolDeniedError, ToolTimeoutError, TransientToolError
from ..security.descriptor import ToolDescriptor
from ..security.redact import redact_for_audit
from ..tool.schema_validate import validate_arguments
from ..security.pipeline import (
    PipelineAction,
    PipelineDecision,
    SecurityPipeline,
    ToolInvocationEvent,
    ToolResultEvent,
)
from .policy import (
    EffectiveToolPolicy,
    ResolvedToolPolicy,
    ToolInvocationContext,
    ToolPolicyProvider,
    finalize_policy,
    merge_policies,
)

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
        event_store: Any = None,
    ) -> None:
        self._descriptor = descriptor
        self._handler = handler
        self._tool_executor = tool_executor
        self._policy_provider = policy_provider
        self._pipeline = security_pipeline
        self._baseline = baseline_policy
        self._run_context = run_context
        self._event_store = event_store

    async def _emit_degraded(self, component: str, reason: str) -> None:
        """Best-effort SecurityDegraded event when a security component fails
        and the call falls back to a fail-closed posture. Never raises -- a
        degraded-path event-emission failure must not mask the original deny."""
        from ..events.payloads import SecurityDegraded
        await self._emit(SecurityDegraded(
            run_id=self._run_id_for_events(), component=component, reason=reason))

    async def _emit(self, payload: Any) -> None:
        """Best-effort governance-event append when an event_store is wired.
        Observability only -- a failure here is logged and swallowed so it
        never masks the tool decision being audited."""
        if self._event_store is None:
            return
        ctx = self._run_context
        run_id = getattr(ctx, "run_id", None) if ctx else None
        root = (getattr(ctx, "root_run_id", None) or run_id) if ctx else run_id
        try:
            await self._event_store.append(
                stream_id=run_id or "", run_id=run_id, root_run_id=root,
                parent_run_id=getattr(ctx, "parent_run_id", None) if ctx else None,
                session_id=getattr(ctx, "session_id", None) if ctx else None,
                runnable_id=getattr(ctx, "runnable_id", None) if ctx else None,
                payload=payload,
            )
        except Exception:  # noqa: BLE001 - best-effort observability
            _LOGGER.debug("failed to emit %r for %r", type(payload).__name__, self._descriptor.name)

    def _run_id_for_events(self) -> "str | None":
        ctx = self._run_context
        return getattr(ctx, "run_id", None) if ctx else None

    async def invoke(
        self,
        *,
        tool_call_id: "str | None" = None,
        parameter_schema: "Mapping[str, Any] | None" = None,
        **arguments: Any,
    ) -> Any:
        ctx = self._run_context
        # Prefer the REAL pydantic-ai tool_call_id (threaded in from the
        # wrapper's call_tool via ctx.tool_call_id) so a pause keys the
        # ApprovalRequest on the same id the message history uses -- the
        # linchpin of resume. Fall back to a fresh uuid only when no real id is
        # available (standalone adapter use, e.g. tests).
        call_id = tool_call_id or str(uuid.uuid4())
        run_id = getattr(ctx, "run_id", None) if ctx else None

        async def _already_approved() -> bool:
            """Resume gate for managed-path approval: a re-driven call after
            external approval must NOT re-raise RunPaused (a stateless pipeline
            otherwise would, looping the run forever)."""
            if self._tool_executor is None:
                return False
            return await self._tool_executor.is_approved(run_id, call_id)

        # 1. Resolve + merge policy, then collapse the tri-state result to
        # concrete values. A layer that never declared a field (e.g. no
        # ToolPolicyProvider wired) must not be mistaken for an explicit
        # 0/False -- finalize_policy() is the single place that decision is made.
        provider_policy = None
        if self._policy_provider is not None and ctx is not None:
            try:
                provider_policy = await self._policy_provider.resolve(self._descriptor, ctx)
            except Exception as exc:
                # Provider failure -> fail closed (deny) AND emit a security
                # degradation event so the silent fallback is observable.
                await self._emit_degraded(
                    "tool-policy-provider", f"{type(exc).__name__}: {exc}")
                raise ToolDeniedError(
                    f"tool policy resolution failed for {self._descriptor.name!r}")
        policy: EffectiveToolPolicy = finalize_policy(
            merge_policies(None, self._baseline, provider_policy))
        # Audit: the finalized policy governing this call -- emitted BEFORE the
        # enabled check so a policy-disabled tool is still auditable.
        from ..events.payloads import ToolPolicyResolved
        await self._emit(ToolPolicyResolved(
            run_id=run_id, tool_name=self._descriptor.name,
            enabled=policy.enabled, timeout_seconds=policy.timeout_seconds,
            max_retries=policy.max_retries, idempotent=policy.idempotent,
            require_approval=policy.require_approval, risk=policy.risk,
        ))
        if not policy.enabled:
            raise ToolDeniedError(f"tool {self._descriptor.name!r} is disabled by policy")

        # 2. Pipeline before_tool.
        if self._pipeline is not None:
            event = ToolInvocationEvent(
                tool_name=self._descriptor.name,
                arguments=arguments,
                run_id=run_id,
                call_id=call_id,
                root_run_id=getattr(ctx, "root_run_id", None) if ctx else None,
                parent_run_id=getattr(ctx, "parent_run_id", None) if ctx else None,
                session_id=getattr(ctx, "session_id", None) if ctx else None,
                agent_id=getattr(ctx, "runnable_id", None) if ctx else None,
                user_id=getattr(ctx, "user_id", None) if ctx else None,
                tenant_id=getattr(ctx, "tenant_id", None) if ctx else None,
                workspace=getattr(ctx, "workspace", None) if ctx else None,
                capability_kind=self._descriptor.capability_kind or None,
                capability_name=self._descriptor.capability_name or None,
                risk=policy.risk,
                mutating=self._descriptor.mutating,
                parameter_schema=parameter_schema,
            )
            from ..events.payloads import ToolPipelineBefore, ToolPipelineDecision
            await self._emit(ToolPipelineBefore(
                run_id=run_id, tool_name=self._descriptor.name, call_id=call_id))
            try:
                decision = await self._pipeline.before_tool(event)
            except Exception:
                raise ToolDeniedError(
                    f"pipeline before_tool failed for {self._descriptor.name!r}")
            await self._emit(ToolPipelineDecision(
                run_id=run_id, tool_name=self._descriptor.name, call_id=call_id,
                action=decision.action.value, reason=decision.reason or ""))
            if decision.action == PipelineAction.DENY:
                raise ToolDeniedError(
                    decision.reason or f"tool {self._descriptor.name!r} denied by pipeline")
            if decision.action == PipelineAction.REQUIRE_APPROVAL and not await _already_approved():
                raise RunPaused(
                    run_id=run_id or "",
                    approval_id=str(uuid.uuid4()),
                    tool_call_id=call_id,
                    tool_name=self._descriptor.name,
                    reason=decision.reason or "pipeline requires approval",
                    # Audit copy only: resume re-emits real args from history,
                    # so masking secrets here never affects execution.
                    arguments=redact_for_audit(arguments),
                )
            if decision.action == PipelineAction.MODIFY and decision.modified_payload:
                arguments = dict(decision.modified_payload)
                # Re-validate the MODIFY'd arguments against the tool's parameter
                # schema so a pipeline cannot inject a payload the tool cannot
                # safely accept. Fails closed (deny) on a mismatch.
                validate_arguments(
                    arguments, parameter_schema, tool_name=self._descriptor.name)

        # 3. Require approval from the resolved policy. Checked regardless of
        # whether a ToolExecutor is wired -- a ToolPolicyProvider/baseline
        # declaring require_approval=True is a distinct signal from the
        # executor's own PolicyEngine decision and must not be silently
        # dropped just because an executor happens to be present. The
        # already-approved gate makes a resume drive skip the re-pause.
        if policy.require_approval and not await _already_approved():
            raise RunPaused(
                run_id=run_id or "",
                approval_id=str(uuid.uuid4()),
                tool_call_id=call_id,
                tool_name=self._descriptor.name,
                reason=f"policy requires approval for {self._descriptor.name!r}",
                arguments=redact_for_audit(arguments),
            )

        # 4. Execute. can_retry mirrors the mutating/idempotent safety rule:
        # 4. Execute. can_retry mirrors the mutating/idempotent safety rule:
        # a mutating, non-idempotent tool is never retried even if a positive
        # max_retries was declared upstream.
        from ..events.payloads import ToolCompleted, ToolFailed, ToolStarted
        await self._emit(ToolStarted(
            tool_name=self._descriptor.name, tool_call_id=call_id))
        can_retry = not (self._descriptor.mutating and not policy.idempotent)
        effective_retries = policy.max_retries if can_retry else 0
        try:
            if self._tool_executor is not None:
                # ToolExecutor.execute is the single real execution entry point:
                # it runs PolicyEngine.evaluate() (deny/approval) THEN the handler
                # with timeout/retry -- never call handler directly when an
                # executor is wired.
                from ..policy.engine import ToolContext, ToolRequest
                tc = ToolContext(
                    run_id=run_id or "",
                    session_id=getattr(ctx, "session_id", None) if ctx else None,
                    tool_call_id=call_id,
                )
                request = ToolRequest(
                    tool_name=self._descriptor.name, arguments=arguments,
                    category=self._descriptor.category, risk=self._descriptor.risk,
                    mutating=self._descriptor.mutating,
                )
                try:
                    result = await self._tool_executor.execute(
                        request, tc, self._handler,
                        timeout=policy.timeout_seconds,
                        max_retries=effective_retries,
                    )
                except asyncio.TimeoutError:
                    raise ToolTimeoutError(
                        f"tool {self._descriptor.name!r} timed out after {policy.timeout_seconds}s")
            else:
                # No executor wired (e.g. a standalone adapter in tests): apply
                # timeout/retry from the resolved policy directly.
                max_attempts = 1 + effective_retries
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
                        if attempt < max_attempts:
                            continue
                        raise
        except Exception as exec_exc:
            # Tool execution failed (timeout/transient-exhausted/handler error).
            # Emit the execution-error audit event, then re-raise so the caller
            # sees the stable error.
            await self._emit(ToolFailed(
                tool_name=self._descriptor.name, tool_call_id=call_id,
                error_message=f"{type(exec_exc).__name__}: {exec_exc}"))
            raise
        await self._emit(ToolCompleted(
            tool_name=self._descriptor.name, tool_call_id=call_id, success=True))

        # 5. Pipeline after_tool.
        if self._pipeline is not None:
            result_event = ToolResultEvent(
                tool_name=self._descriptor.name, result=result,
                success=True,
                run_id=run_id, call_id=call_id,
            )
            from ..events.payloads import ToolPipelineAfter
            await self._emit(ToolPipelineAfter(
                run_id=run_id, tool_name=self._descriptor.name,
                call_id=call_id, success=True))
            try:
                after_decision = await self._pipeline.after_tool(result_event)
            except Exception:
                _LOGGER.warning("after_tool pipeline error for %r (fail closed)",
                                self._descriptor.name)
                raise ToolDeniedError(
                    f"after_tool pipeline failed for {self._descriptor.name!r}")
            # after_tool distinct actions: DENY_RESULT (or legacy DENY) discards
            # the result; MODIFY_RESULT (or legacy MODIFY) replaces it. The
            # result-level decision is never confused with a before_tool
            # call-level DENY/MODIFY.
            if after_decision.action in (PipelineAction.DENY, PipelineAction.DENY_RESULT):
                raise ToolDeniedError(
                    f"tool {self._descriptor.name!r} result denied by after_tool pipeline")
            if after_decision.action in (PipelineAction.MODIFY, PipelineAction.MODIFY_RESULT):
                result = after_decision.modified_payload

        return result
