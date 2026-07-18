#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ManagedToolAdapter: the single entry point through which every model-driven
tool call must pass. Wraps a handler with the full governance chain:

    descriptor -> ToolPolicyProvider -> SecurityBaseline merge ->
    SecurityPipeline.before_tool -> ToolExecutor.check (policy/approval) ->
    handler (timeout + retry) -> SecurityPipeline.after_tool -> stable result

Providers supply handlers and descriptors; execution and governance are owned by
ToolExecutor."""

import asyncio
import logging
import uuid
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Mapping

from ..errors import (
    RunPaused,
    ToolDeniedError,
    ToolTimeoutError,
    ToolSecurityAuditError,
    ToolResultDeniedError,
    RuntimeInitializationError,
)
from ..tool.models import ToolDescriptor
from ..security.redact import redact_for_audit, redact_exception
from ..tool.schema import validate_arguments
from ..security.pipeline import (
    PipelineAction,
    SecurityPipeline,
    ToolInvocationEvent,
    ToolResultEvent,
    validate_tool_decision,
)
from .policy import (
    EffectiveToolPolicy,
    ResolvedToolPolicy,
    ToolPolicyProvider,
    finalize_policy,
    merge_policies,
)

if TYPE_CHECKING:
    from ..run.context import RunContext

_LOGGER = logging.getLogger(__name__)


class ManagedToolAdapter:
    """Wraps a tool handler with unified security governance. When a
    ToolExecutor is wired, delegates policy/approval/idempotency to it; the
    adapter adds pipeline before/after and policy resolution on top."""

    def __init__(
        self,
        *,
        descriptor: ToolDescriptor,
        handler: "Callable[..., Awaitable[Any]]",
        tool_executor: Any,
        policy_provider: "ToolPolicyProvider | None" = None,
        security_pipeline: "SecurityPipeline | None" = None,
        baseline_policy: "ResolvedToolPolicy | None" = None,
        run_context: "RunContext | None" = None,
        event_store: Any = None,
        security_audit_failure_mode: Any = "fail_closed",
        security_event_emitter: Any = None,
    ) -> None:
        if tool_executor is None:
            raise RuntimeInitializationError("ManagedToolAdapter requires ToolExecutor")
        self._descriptor = descriptor
        self._handler = handler
        self._tool_executor = tool_executor
        self._policy_provider = policy_provider
        self._pipeline = security_pipeline
        self._baseline = baseline_policy
        self._run_context = run_context
        self._event_store = event_store
        self._security_audit_failure_mode = getattr(
            security_audit_failure_mode, "value", security_audit_failure_mode
        )
        if security_event_emitter is None and event_store is not None:
            from ..security.emitter import EventStoreSecurityEventEmitter

            security_event_emitter = EventStoreSecurityEventEmitter(
                event_store,
                context=run_context,
                failure_mode=security_audit_failure_mode,
            )
        self._security_event_emitter = security_event_emitter

    async def _emit_degraded(self, component: str, reason: str) -> None:
        """Best-effort SecurityDegraded event when a security component fails
        and the call falls back to a fail-closed posture. Never raises -- a
        degraded-path event-emission failure must not mask the original deny."""
        from ..events.payloads import SecurityDegraded

        await self._emit_security(
            SecurityDegraded(
                run_id=self._run_id_for_events(), component=component, reason=reason
            )
        )

    async def _emit_security(self, payload: Any) -> None:
        """Persist a security-critical audit event (policy decision, pipeline
        decision, degradation). Classification is explicit at the call site, not
        inferred from the payload class, so adding a new audit event can never be
        silently misrouted to the observability channel. A failure to persist is
        governed by the security failure mode -- fail_closed re-raises so an
        unrecorded security decision blocks the call."""
        if self._security_event_emitter is not None:
            await self._security_event_emitter.emit_security(payload)
            return
        await self._append_to_store(payload, security=True)

    async def _emit_observability(self, payload: Any) -> None:
        """Persist an observability event (tool started/completed, pipeline
        before/after markers). Failure is always best-effort -- an observability
        record must never mask the tool decision being audited."""
        if self._security_event_emitter is not None:
            await self._security_event_emitter.emit_observability(payload)
            return
        await self._append_to_store(payload, security=False)

    async def _append_to_store(self, payload: Any, *, security: bool) -> None:
        """Direct-to-EventStore fallback used when no SecurityEventEmitter is
        wired (e.g. a standalone adapter in tests). ``security`` selects the
        failure handling: a security event respects the configured failure mode;
        an observability event is always best-effort."""
        if self._event_store is None:
            return
        ctx = self._run_context
        run_id = getattr(ctx, "run_id", None) if ctx else None
        root = (getattr(ctx, "root_run_id", None) or run_id) if ctx else run_id
        from ..events.context import EventContext, append_event

        try:
            await append_event(
                self._event_store,
                EventContext(
                    stream_id=run_id or "",
                    run_id=run_id,
                    root_run_id=root,
                    parent_run_id=getattr(ctx, "parent_run_id", None) if ctx else None,
                    session_id=getattr(ctx, "session_id", None) if ctx else None,
                    runnable_id=getattr(ctx, "runnable_id", None) if ctx else None,
                ),
                payload,
            )
        except Exception as exc:  # noqa: BLE001 - security events can fail closed
            _LOGGER.debug(
                "failed to emit %r for %r",
                type(payload).__name__,
                self._descriptor.name,
            )
            if security and self._security_audit_failure_mode != "best_effort":
                raise ToolSecurityAuditError(
                    f"failed to persist security audit event {type(payload).__name__}"
                ) from exc

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
            from ..agent.approval import compute_arguments_hash
            metadata = getattr(ctx, "metadata", {}) if ctx is not None else {}
            binding = {"tool_name": self._descriptor.name,
                       "arguments_hash": compute_arguments_hash(self._descriptor.name, arguments)}
            binding.update({key: metadata.get(key) for key in (
                "descriptor_fingerprint", "handler_revision", "provider_revision",
                "policy_revision", "capability_revision", "result_processor_revision")})
            return await self._tool_executor._is_approved_binding(run_id, call_id, binding=binding)

        # 1. Resolve + merge policy, then collapse the tri-state result to
        # concrete values. A layer that never declared a field (e.g. no
        # ToolPolicyProvider wired) must not be mistaken for an explicit
        # 0/False -- finalize_policy() is the single place that decision is made.
        provider_policy = None
        if self._policy_provider is not None and ctx is not None:
            try:
                provider_policy = await self._policy_provider.resolve(
                    self._descriptor, ctx
                )
            except Exception as exc:
                # Provider failure -> fail closed (deny) AND emit a security
                # degradation event so the silent fallback is observable.
                await self._emit_degraded(
                    "tool-policy-provider", f"{type(exc).__name__}: {exc}"
                )
                raise ToolDeniedError(
                    f"tool policy resolution failed for {self._descriptor.name!r}"
                )
        policy: EffectiveToolPolicy = finalize_policy(
            merge_policies(None, self._baseline, provider_policy)
        )
        # Audit: the finalized policy governing this call -- emitted BEFORE the
        # enabled check so a policy-disabled tool is still auditable.
        from ..events.payloads import ToolPolicyResolved

        await self._emit_security(
            ToolPolicyResolved(
                run_id=run_id,
                tool_name=self._descriptor.name,
                enabled=policy.enabled,
                timeout_seconds=policy.timeout_seconds,
                max_retries=policy.max_retries,
                idempotent=policy.idempotent,
                require_approval=policy.require_approval,
                risk=policy.risk,
            )
        )
        if not policy.enabled:
            raise ToolDeniedError(
                f"tool {self._descriptor.name!r} is disabled by policy"
            )

        # Validate the original arguments against the tool's parameter schema.
        # pydantic-ai already validates this for signature-derived tools, but a
        # **kwargs handler registered via Tool.from_schema skips validation, so
        # this is the gate that catches a bad original payload uniformly.
        validate_arguments(arguments, parameter_schema, tool_name=self._descriptor.name)

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

            await self._emit_observability(
                ToolPipelineBefore(
                    run_id=run_id, tool_name=self._descriptor.name, call_id=call_id
                )
            )
            try:
                decision = await self._pipeline.before_tool(event)
            except Exception as exc:
                # before_tool raised: surface the degradation so the silent
                # fallback to deny is observable, then fail closed. The handler
                # never runs -- this raise precedes the execution section.
                await self._emit_degraded(
                    "security-pipeline-before-tool", f"{type(exc).__name__}: {exc}"
                )
                raise ToolDeniedError(
                    f"pipeline before_tool failed for {self._descriptor.name!r}"
                ) from exc
            try:
                validate_tool_decision(decision, stage="before")
            except Exception as exc:
                # before_tool returned an after-only/illegal action: treat it as
                # an explicit deny, audit the decision, then fail closed. The
                # handler never runs.
                await self._emit_security(
                    ToolPipelineDecision(
                        run_id=run_id,
                        tool_name=self._descriptor.name,
                        call_id=call_id,
                        action=PipelineAction.DENY.value,
                        reason=redact_exception(exc) or "invalid before-action",
                        stage="before",
                    )
                )
                raise ToolDeniedError(str(exc)) from exc
            await self._emit_security(
                ToolPipelineDecision(
                    run_id=run_id,
                    tool_name=self._descriptor.name,
                    call_id=call_id,
                    action=decision.action.value,
                    reason=decision.reason or "",
                    stage="before",
                )
            )
            if decision.action == PipelineAction.DENY:
                raise ToolDeniedError(
                    decision.reason
                    or f"tool {self._descriptor.name!r} denied by pipeline"
                )
            if (
                decision.action == PipelineAction.REQUIRE_APPROVAL
                and not await _already_approved()
            ):
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
            if decision.action == PipelineAction.MODIFY:
                arguments = dict(decision.modified_payload or {})
                # Re-validate the MODIFY'd arguments against the tool's parameter
                # schema so a pipeline cannot inject a payload the tool cannot
                # safely accept. Fails closed (deny) on a mismatch.
                validate_arguments(
                    arguments, parameter_schema, tool_name=self._descriptor.name
                )

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
                tenant_id=getattr(ctx, "tenant_id", None) if ctx else None,
                tool_name=self._descriptor.name,
                reason=f"policy requires approval for {self._descriptor.name!r}",
                arguments=redact_for_audit(arguments),
            )

        # 4. Execute. can_retry mirrors the mutating/idempotent safety rule:
        # a mutating, non-idempotent tool is never retried even if a positive
        # max_retries was declared upstream.
        from ..events.payloads import ToolCompleted, ToolFailed, ToolStarted

        await self._emit_observability(
            ToolStarted(tool_name=self._descriptor.name, tool_call_id=call_id)
        )
        can_retry = not (self._descriptor.mutating and not policy.idempotent)
        effective_retries = policy.max_retries if can_retry else 0
        try:
            # ToolExecutor.execute is the single real execution entry point.
            from ..policy.engine import ToolContext, ToolRequest

            tc = ToolContext(
                run_id=run_id or "",
                session_id=getattr(ctx, "session_id", None) if ctx else None,
                tool_call_id=call_id,
            )
            request = ToolRequest(
                tool_name=self._descriptor.name,
                arguments=arguments,
                category=self._descriptor.category,
                risk=self._descriptor.risk,
                mutating=self._descriptor.mutating,
            )
            # When the policy declares the tool idempotent, derive a stable
            # idempotency key (run + tool + canonical args + schema_version)
            # so replays return the cached result; pass schema_version so a
            # contract change is a fresh idempotency record. The executor
            # fail-closes if no IdempotencyStore is wired.
            idempotency_key = None
            if policy.idempotent:
                from .idempotency import DefaultIdempotencyKeyBuilder

                idempotency_key = DefaultIdempotencyKeyBuilder().build(
                    descriptor=self._descriptor,
                    arguments=arguments,
                    run_context=ctx,
                    schema_version=policy.schema_version,
                    policy=policy,
                )
            try:
                result = await self._tool_executor.execute(
                    request,
                    tc,
                    self._handler,
                    timeout=policy.timeout_seconds,
                    max_retries=effective_retries,
                    descriptor=self._descriptor,
                    effective_policy=policy,
                    idempotency_key=idempotency_key,
                    schema_version=policy.schema_version,
                    idempotency_scope=(
                        f"{getattr(ctx, 'tenant_id', '')}|{getattr(ctx, 'workspace', '')}|"
                        f"{self._descriptor.name}|{policy.schema_version}"
                        if getattr(policy, "idempotency_strategy", None).value
                        == "business_key"
                        else None
                    ),
                )
            except asyncio.TimeoutError:
                raise ToolTimeoutError(
                    f"tool {self._descriptor.name!r} timed out after {policy.timeout_seconds}s"
                )
        except Exception as exec_exc:
            # Tool execution failed (timeout/transient-exhausted/handler error).
            # Emit the execution-error audit event, then re-raise so the caller
            # sees the stable error.
            await self._emit_observability(
                ToolFailed(
                    tool_name=self._descriptor.name,
                    tool_call_id=call_id,
                    error_message=f"{type(exec_exc).__name__}: {redact_exception(exec_exc)}",
                )
            )
            raise
        # 5. Pipeline after_tool.
        result_action = "returned"
        if self._pipeline is not None:
            result_event = ToolResultEvent(
                tool_name=self._descriptor.name,
                result=result,
                success=True,
                run_id=run_id,
                call_id=call_id,
            )
            from ..events.payloads import ToolPipelineAfter

            try:
                after_decision = await self._pipeline.after_tool(result_event)
            except Exception:
                _LOGGER.warning(
                    "after_tool pipeline error for %r (fail closed)",
                    self._descriptor.name,
                )
                await self._emit_security(
                    ToolPipelineDecision(
                        run_id=run_id,
                        tool_name=self._descriptor.name,
                        call_id=call_id,
                        action=PipelineAction.DENY_RESULT.value,
                        reason="after_tool pipeline failed",
                        stage="after",
                    )
                )
                await self._emit_observability(
                    ToolCompleted(
                        tool_name=self._descriptor.name,
                        tool_call_id=call_id,
                        success=True,
                        execution_success=True,
                        result_action="denied",
                    )
                )
                raise ToolResultDeniedError(
                    f"after_tool pipeline failed for {self._descriptor.name!r}"
                )
            try:
                validate_tool_decision(after_decision, stage="after")
            except Exception as exc:
                safe_error = redact_exception(exc)
                await self._emit_security(
                    ToolPipelineDecision(
                        run_id=run_id,
                        tool_name=self._descriptor.name,
                        call_id=call_id,
                        action=PipelineAction.DENY_RESULT.value,
                        reason=safe_error,
                        stage="after",
                    )
                )
                await self._emit_observability(
                    ToolCompleted(
                        tool_name=self._descriptor.name,
                        tool_call_id=call_id,
                        success=True,
                        execution_success=True,
                        result_action="denied",
                    )
                )
                raise ToolResultDeniedError(
                    "after_tool returned an invalid decision"
                ) from exc
            await self._emit_security(
                ToolPipelineDecision(
                    run_id=run_id,
                    tool_name=self._descriptor.name,
                    call_id=call_id,
                    action=after_decision.action.value,
                    reason=after_decision.reason or "",
                    stage="after",
                )
            )
            await self._emit_observability(
                ToolPipelineAfter(
                    run_id=run_id,
                    tool_name=self._descriptor.name,
                    call_id=call_id,
                    success=True,
                )
            )
            if after_decision.action is PipelineAction.DENY_RESULT:
                result_action = "denied"
                await self._emit_observability(
                    ToolCompleted(
                        tool_name=self._descriptor.name,
                        tool_call_id=call_id,
                        success=True,
                        execution_success=True,
                        result_action=result_action,
                    )
                )
                raise ToolResultDeniedError(
                    f"tool {self._descriptor.name!r} result denied by after_tool pipeline"
                )
            if after_decision.action is PipelineAction.MODIFY_RESULT:
                result = after_decision.modified_payload
                result_action = "modified"

        await self._emit_observability(
            ToolCompleted(
                tool_name=self._descriptor.name,
                tool_call_id=call_id,
                success=True,
                execution_success=True,
                result_action=result_action,
            )
        )

        return result
