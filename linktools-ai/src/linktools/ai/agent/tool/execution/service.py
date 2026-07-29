"""The sole entry point for model-driven tool calls."""

import asyncio
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Protocol
from uuid import uuid4

from ....errors import (
    RunPaused,
    RuntimeInitializationError,
    ToolIdempotencyConflictError,
    ToolDeniedError,
    ToolResultDeniedError,
    ToolTimeoutError,
)
from ....json import JsonValue, canonical_json_bytes, normalize_json
from ....execution.trace_models import ToolResultTrace
from ....governance.policy.engine import PolicyEngine
from ....governance.policy.rule import (
    PolicyDecisionKind,
    ToolContext,
    ToolRequest,
)
from ....governance.security.pipeline import (
    PipelineAction,
    SecurityPipeline,
    ToolInvocationEvent,
    ToolResultEvent,
    validate_tool_decision,
)
from ..policy.resolver import (
    EffectiveToolPolicy,
    IdempotencyStrategy,
    ToolPolicyResolver,
    finalize_policy,
)
from ..state.models import ToolOperation, ToolOperationStatus
from ..state.store import ToolStateStore
from .binding import ToolExecutionBinding, ToolRevisionSet
from .idempotency import hash_tool_arguments, operation_id
from .models import ExecuteTool
from .retry import DefaultRetryPolicy, RetryPolicy, backoff_delay
from .schema import validate_arguments


class ToolExecutionHook(Protocol):
    async def before(self, request: ExecuteTool) -> None: ...

    async def after(
        self,
        request: ExecuteTool,
        result: JsonValue,
    ) -> JsonValue: ...


def normalize_tool_error(error: BaseException) -> JsonValue:
    return {
        "type": type(error).__name__,
        "message": str(error),
    }


@dataclass(slots=True)
class ToolExecutionService:
    """Govern and durably execute one tool call in the mandated order."""

    state: ToolStateStore | None = None
    policy: ToolPolicyResolver | None = None
    policy_engine: PolicyEngine | None = None
    security: SecurityPipeline | None = None
    policy_revision: str = "default"
    hooks: tuple[ToolExecutionHook, ...] = ()
    retry: RetryPolicy = DefaultRetryPolicy()
    lease_seconds: float = 300.0

    async def execute(self, request: ExecuteTool) -> JsonValue:
        started_at = datetime.now(timezone.utc)
        try:
            result, replayed = await self._execute(request)
        except RunPaused:
            raise
        except asyncio.CancelledError as error:
            stored = None
            if self.state is not None and request.context.tool_call_id:
                stored = await self.state.get(
                    operation_id(
                        request.context.execution_id,
                        request.context.tool_call_id,
                    )
                )
            if stored is None or stored.status is ToolOperationStatus.PREPARED:
                raise
            status = (
                "indeterminate"
                if stored.status is ToolOperationStatus.CLAIMED
                else stored.status.value
            )
            await self._record_trace(
                request,
                result=None,
                status=status,
                error=error,
                replayed=False,
                started_at=started_at,
            )
            raise
        except BaseException as error:
            await self._record_trace(
                request,
                result=None,
                status=(
                    "result_denied"
                    if isinstance(error, ToolResultDeniedError)
                    else "denied"
                    if isinstance(error, ToolDeniedError)
                    else "failed"
                ),
                error=error,
                replayed=False,
                started_at=started_at,
            )
            raise
        await self._record_trace(
            request,
            result=result,
            status="completed",
            error=None,
            replayed=replayed,
            started_at=started_at,
        )
        return result

    async def _execute(
        self, request: ExecuteTool
    ) -> tuple[JsonValue, bool]:
        definition = request.definition
        descriptor = definition.descriptor
        arguments = normalize_json(dict(request.arguments))
        if not isinstance(arguments, dict):
            raise TypeError("tool arguments must normalize to an object")
        validate_arguments(
            arguments,
            definition.input_schema,
            tool_name=descriptor.name,
        )

        resolved = None
        if self.policy is not None:
            if request.context.run_context is None:
                raise RuntimeInitializationError(
                    "tool policy resolution requires a RunContext"
                )
            resolved = await self.policy.resolve(
                descriptor,
                request.context.run_context,
            )
        effective = finalize_policy(resolved)
        if not effective.enabled:
            raise ToolDeniedError(f"tool {descriptor.name!r} is disabled")

        policy_approval_reason: str | None = None
        if self.policy_engine is not None:
            decision = await self.policy_engine.evaluate(
                ToolRequest(
                    tool_name=descriptor.name,
                    arguments=arguments,
                    category=descriptor.category.value,
                    risk=descriptor.risk.name.lower(),
                    mutating=descriptor.mutating,
                    metadata=descriptor.metadata,
                ),
                request.context.dependencies.tool_context,
            )
            if decision.kind is PolicyDecisionKind.DENY:
                raise ToolDeniedError(
                    decision.reason or f"tool {descriptor.name!r} denied"
                )
            if decision.kind is PolicyDecisionKind.REQUIRE_APPROVAL:
                policy_approval_reason = decision.reason or "policy requires approval"

        security_approval_reason: str | None = None
        if self.security is not None:
            run = request.context.run_context
            security_decision = await self.security.before_tool(
                ToolInvocationEvent(
                    tool_name=descriptor.name,
                    arguments=arguments,
                    run_id=request.context.execution_id,
                    call_id=request.context.tool_call_id,
                    root_execution_id=(
                        run.root_execution_id if run is not None else None
                    ),
                    parent_execution_id=(
                        run.parent_execution_id if run is not None else None
                    ),
                    session_id=run.session_id if run is not None else None,
                    agent_id=run.runnable_id if run is not None else None,
                    user_id=run.user_id if run is not None else None,
                    tenant_id=run.tenant_id if run is not None else None,
                    workspace=run.workspace if run is not None else None,
                    feature_kind=descriptor.feature.kind,
                    feature_name=descriptor.feature.name,
                    risk=descriptor.risk.name.lower(),
                    mutating=descriptor.mutating,
                    parameter_schema=definition.input_schema,
                )
            )
            validate_tool_decision(security_decision, stage="before")
            if security_decision.action is PipelineAction.DENY:
                raise ToolDeniedError(
                    security_decision.reason
                    or f"tool {descriptor.name!r} denied"
                )
            if security_decision.action is PipelineAction.REQUIRE_APPROVAL:
                security_approval_reason = (
                    security_decision.reason or "security requires approval"
                )
            if security_decision.action is PipelineAction.MODIFY:
                arguments = normalize_json(security_decision.modified_payload)
                if not isinstance(arguments, dict):
                    raise ToolDeniedError(
                        "security pipeline produced non-object arguments"
                    )
                validate_arguments(
                    arguments,
                    definition.input_schema,
                    tool_name=descriptor.name,
                )

        for hook in self.hooks:
            await hook.before(request)

        arguments_hash = hash_tool_arguments(descriptor.name, arguments)
        binding = self._binding(request, arguments_hash, effective)
        approved = (
            request.context.approved_tool_call_id
            == request.context.tool_call_id
            and request.context.approved_binding_fingerprint is not None
        )
        if approved and (
            request.context.approved_binding_fingerprint
            != binding.fingerprint()
        ):
            raise ToolIdempotencyConflictError(
                "approved tool binding no longer matches the current call"
            )
        approval_reason = (
            policy_approval_reason
            or security_approval_reason
            or (
                "tool policy requires approval"
                if effective.require_approval
                else None
            )
        )
        if approval_reason is not None and not approved:
            raise RunPaused(
                request.context.execution_id,
                uuid4().hex,
                tool_call_id=request.context.tool_call_id,
                tool_name=descriptor.name,
                reason=approval_reason,
                arguments=dict(arguments),
                idempotency_key=self._idempotency_key(
                    request,
                    arguments,
                    arguments_hash,
                    effective,
                ),
                binding={
                    **normalize_json(asdict(binding)),
                    "fingerprint": binding.fingerprint(),
                },
            )

        if self.state is None:
            raise RuntimeInitializationError(
                "ToolExecutionService requires ToolStateStore"
            )
        if not request.context.tool_call_id:
            raise RuntimeInitializationError("tool call id is required")

        op_id = operation_id(
            request.context.execution_id,
            request.context.tool_call_id,
        )
        prepared = await self.state.prepare(
            ToolOperation(
                id=op_id,
                tenant_id=request.context.run_context.tenant_id
                if request.context.run_context is not None
                else None,
                execution_id=request.context.execution_id,
                tool_call_id=request.context.tool_call_id,
                idempotency_key=self._idempotency_key(
                    request,
                    arguments,
                    arguments_hash,
                    effective,
                ),
                tool_name=descriptor.name,
                arguments_hash=arguments_hash,
                binding_fingerprint=binding.fingerprint(),
                status=ToolOperationStatus.PREPARED,
                replay_safe=effective.idempotent or not descriptor.mutating,
            )
        )
        if prepared.status is ToolOperationStatus.COMPLETED:
            return normalize_json(prepared.result), True
        if prepared.status is ToolOperationStatus.INDETERMINATE:
            raise RuntimeError("tool operation outcome is indeterminate")

        # Deliver cancellation before acquiring a lease. A prepared operation
        # is replayable; once claimed, cancellation may make a mutating
        # handler's outcome unknowable.
        await asyncio.sleep(0)
        owner = uuid4().hex
        try:
            claimed = await self.state.claim(
                op_id,
                owner=owner,
                duration=self._lease_duration(),
            )
        except asyncio.CancelledError:
            current = await self.state.get(op_id)
            if (
                current is not None
                and current.status is ToolOperationStatus.CLAIMED
                and current.owner == owner
            ):
                if current.replay_safe:
                    await self.state.fail(
                        op_id,
                        owner=owner,
                        fence=current.fence,
                        error=normalize_tool_error(asyncio.CancelledError()),
                    )
                else:
                    await self.state.mark_indeterminate(
                        op_id,
                        owner=owner,
                        fence=current.fence,
                        error=normalize_tool_error(asyncio.CancelledError()),
                    )
            raise
        if claimed.status is ToolOperationStatus.COMPLETED:
            return normalize_json(claimed.result), True
        if claimed.status is ToolOperationStatus.INDETERMINATE:
            raise RuntimeError("tool operation outcome is indeterminate")

        try:
            result = await self._invoke(request, arguments, effective)
            if self.security is not None:
                security_result = await self.security.after_tool(
                    ToolResultEvent(
                        tool_name=descriptor.name,
                        result=result,
                        run_id=request.context.execution_id,
                        call_id=request.context.tool_call_id,
                    )
                )
                validate_tool_decision(security_result, stage="after")
                if security_result.action is PipelineAction.DENY_RESULT:
                    raise ToolResultDeniedError(
                        security_result.reason or "tool result denied"
                    )
                if security_result.action is PipelineAction.MODIFY_RESULT:
                    result = normalize_json(security_result.modified_payload)
            for hook in reversed(self.hooks):
                result = normalize_json(await hook.after(request, result))
            committed = await self.state.complete(
                op_id,
                owner=owner,
                fence=claimed.fence,
                result=result,
            )
            return normalize_json(committed.result), False
        except asyncio.CancelledError as exc:
            error = normalize_tool_error(exc)
            if claimed.replay_safe:
                await self.state.fail(
                    op_id,
                    owner=owner,
                    fence=claimed.fence,
                    error=error,
                )
            else:
                await self.state.mark_indeterminate(
                    op_id,
                    owner=owner,
                    fence=claimed.fence,
                    error=error,
                )
            raise
        except BaseException as exc:
            await self.state.fail(
                op_id,
                owner=owner,
                fence=claimed.fence,
                error=normalize_tool_error(exc),
            )
            raise

    async def _record_trace(
        self,
        request: ExecuteTool,
        *,
        result: JsonValue | None,
        status: str,
        error: BaseException | None,
        replayed: bool,
        started_at: datetime,
    ) -> None:
        sink = request.context.trace_sink
        if sink is None:
            return
        trace = ToolResultTrace(
            tool_call_id=request.context.tool_call_id,
            tool_name=request.definition.descriptor.name,
            operation_id=operation_id(
                request.context.execution_id,
                request.context.tool_call_id,
            ),
            status=status,
            result=result,
            error=(
                normalize_tool_error(error)
                if error is not None
                else None
            ),
            replayed=replayed,
            started_at=started_at,
            completed_at=datetime.now(timezone.utc),
        )
        await sink.tool_result(normalize_json(asdict(trace)))

    async def _invoke(
        self,
        request: ExecuteTool,
        arguments: dict[str, JsonValue],
        policy: EffectiveToolPolicy,
    ) -> JsonValue:
        attempt = 0
        while True:
            attempt += 1
            try:
                invocation = request.definition.handler(**arguments)
                if policy.timeout_seconds is None:
                    raw_result = await invocation
                else:
                    raw_result = await asyncio.wait_for(
                        invocation,
                        timeout=policy.timeout_seconds,
                    )
                return normalize_json(raw_result)
            except asyncio.TimeoutError as exc:
                error: BaseException = ToolTimeoutError(
                    f"tool {request.definition.descriptor.name!r} timed out"
                )
                error.__cause__ = exc
            except asyncio.CancelledError:
                raise
            except BaseException as exc:
                error = exc
            if attempt > policy.max_retries or not self.retry.should_retry(
                error=error,
                attempt=attempt,
                policy=policy,
                descriptor=request.definition.descriptor,
            ):
                raise error
            await asyncio.sleep(backoff_delay(attempt))

    def _binding(
        self,
        request: ExecuteTool,
        arguments_hash: str,
        policy: EffectiveToolPolicy,
    ) -> ToolExecutionBinding:
        definition = request.definition
        descriptor = definition.descriptor
        return ToolExecutionBinding(
            schema_version=int(policy.schema_version),
            tool_name=descriptor.name,
            arguments_hash=arguments_hash,
            revisions=ToolRevisionSet(
                descriptor=descriptor.fingerprint(),
                handler=definition.handler_revision,
                provider=definition.provider_revision,
                policy=self.policy_revision,
                feature=descriptor.feature.fingerprint(),
                result_processor="normalize_json.v1",
            ),
        )

    @staticmethod
    def _idempotency_key(
        request: ExecuteTool,
        arguments: dict[str, JsonValue],
        arguments_hash: str,
        policy: EffectiveToolPolicy,
    ) -> str:
        if policy.idempotency_strategy is IdempotencyStrategy.BUSINESS_KEY:
            field = policy.idempotency_key_field
            if not field or field not in arguments:
                raise ValueError(
                    f"missing business idempotency key field: {field!r}"
                )
            payload: JsonValue = {
                "tenant_id": (
                    request.context.run_context.tenant_id
                    if request.context.run_context is not None
                    else None
                ),
                "tool_name": request.definition.descriptor.name,
                "business_key": arguments[field],
                "schema_version": policy.schema_version,
            }
        else:
            payload = {
                "execution_id": request.context.execution_id,
                "tool_name": request.definition.descriptor.name,
                "arguments_hash": arguments_hash,
                "schema_version": policy.schema_version,
            }
        from hashlib import sha256

        return sha256(canonical_json_bytes(payload)).hexdigest()

    def _lease_duration(self):
        from datetime import timedelta

        return timedelta(seconds=self.lease_seconds)

    async def deny(self, request: ExecuteTool, reason: str) -> None:
        raise ToolDeniedError(
            f"tool {request.definition.descriptor.name!r} denied: {reason}"
        )


__all__ = [
    "ToolExecutionHook",
    "ToolExecutionService",
    "normalize_tool_error",
]
