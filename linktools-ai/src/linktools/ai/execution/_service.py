#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared execution settlement over caller-supplied persistence operations."""
from __future__ import annotations

import asyncio
import uuid
from collections.abc import Callable, Mapping
from contextlib import AbstractContextManager, asynccontextmanager
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from functools import wraps
from typing import TYPE_CHECKING, Protocol

from linktools.core import environ
from ..agent import restore_output
from ..core import (AuthorizationAction, AuthorizationPolicy, BudgetUsage, CorrelationData,
    ExecutionEventType, ExecutionLineageKind, ExecutionStatus, HandoffGate, HandoffState,
    IdempotencyStatus, JsonValue, Principal, ResourceKind, ResourceRef, StopReason, UsageMetrics,
    canonical_json_bytes, canonical_sha256, normalize_json_value, overlay_correlation,
    principal_identity_payload, idempotency_key_digest as compute_idempotency_key_digest)
from ..errors import AIError, ErrorCode
from ..storage import PayloadPolicy, StoredPayload, payload_fits_inline
from ..task import TaskBindingContract, TaskEffectResolution
from ._api import CancelExecutionResult, ExecutionHandle, ExecutionResult, ExecutionView, project_execution_view
from ._management import (
    ExecutionManagementPort,
    ExecutionManager,
    execution_recovery_error as _execution_recovery_error,
)
from ._records import (ExecutionRecord, ExecutionStartClaim, ExecutionStartReservation,
    ExecutionTerminalCommit, IdempotencyRecord, IdempotencyTerminalUpdate, ResultRecord, StoredUserInput)

if TYPE_CHECKING:
    from linktools.ai.persistence.contracts import ExecutionRepositories, SessionRepository, TaskAdmissionRepository

_logger = environ.get_logger("ai.execution")

def overlay_execution_correlation(
    base: Mapping[str, object],
    overlay: Mapping[str, object],
) -> CorrelationData:
    try:
        return overlay_correlation(base, overlay)
    except (TypeError, ValueError) as error:
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID) from error


def consumed_terminal_query(method: "Callable[..., object]") -> "Callable[..., object]":
    @wraps(method)
    async def wrapped(
        self: "TaskExecutionService",
        execution_id: str,
        *args: object,
        principal: Principal,
        **kwargs: object,
    ) -> object:
        async with self.execution_consumer(execution_id, principal.tenant_id):
            with self.source_reads():
                result = await method(
                    self, execution_id, *args, principal=principal, **kwargs
                )
            return result

    return wrapped


def observed_query(method: "Callable[..., object]") -> "Callable[..., object]":
    @wraps(method)
    async def wrapped(
        self: "TaskExecutionService",
        execution_id: str,
        *args: object,
        principal: Principal,
        **kwargs: object,
    ) -> object:
        async with self.execution_consumer(execution_id, principal.tenant_id):
            return await method(
                self, execution_id, *args, principal=principal, **kwargs
            )

    return wrapped


class ExecutionReleaseCallback(Protocol):
    async def __call__(
        self, execution_id: str, *, tenant_id: str, terminal_commit: bool = False,
    ) -> bool: ...


class LocalExecutionWaiter(Protocol):
    def owns_execution(self, execution_id: str, *, tenant_id: str) -> bool: ...

    async def wait_terminal(self, execution_id: str, *, tenant_id: str) -> None: ...


async def _no_release_terminal(
    execution_id: str, *, tenant_id: str, terminal_commit: bool = False,
) -> bool:
    del execution_id, tenant_id, terminal_commit
    return True


def _require_task_attempt(
    execution: ExecutionRecord,
    attempt: ExecutionView | None,
) -> None:
    if attempt is None:
        return
    if (
        attempt.execution_id != execution.execution_id
        or attempt.event_seq != execution.event_seq
        or attempt.task_attempt != execution.task_attempt
        or attempt.status is not execution.status
        or attempt.task_deadline_at != execution.task_deadline_at
        or attempt.task_next_attempt_at != execution.task_next_attempt_at
    ):
        raise AIError(ErrorCode.TASK_FENCE_STALE)


def _validate_task_binding_output(
    binding: TaskBindingContract,
    output: JsonValue,
) -> None:
    contract = dict(binding.output_contract)
    if contract == {"kind": "json"}:
        return
    if set(contract) != {"mode", "schema"}:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    try:
        restored = restore_output(contract["mode"], contract["schema"])
    except AIError as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
    try:
        restored.validate_payload(output)
    except AIError as error:
        if error.code is ErrorCode.OUTPUT_VALIDATION_FAILED:
            raise AIError(
                ErrorCode.OUTPUT_CONTRACT_INVALID,
                retryable=False,
            ) from error
        raise


class TaskExecutionService:
    """Authorize and settle callable executions without constructing an Agent host."""

    def __init__(
        self,
        state: ExecutionRepositories,
        authorization: AuthorizationPolicy,
        *,
        port: ExecutionManagementPort,
        sessions: SessionRepository | None = None,
        task_admissions: TaskAdmissionRepository | None = None,
        operation_ids: Callable[[], str] | None = None,
        release_terminal: ExecutionReleaseCallback | None = None,
        payload_policy: PayloadPolicy | None = None,
    ) -> None:
        self._repositories = state
        self._policy = authorization
        self._session_repository = sessions
        self._admissions = task_admissions
        self._new_operation_id = operation_ids or (lambda: uuid.uuid4().hex)
        self._release_terminal = release_terminal or _no_release_terminal
        self._output_policy = payload_policy
        self._handoff = HandoffGate[tuple[str, str], bool]()
        self._port = port
        self._manager = ExecutionManager(port, authorization)

    @property
    def manager(self) -> ExecutionManager:
        return self._manager

    def source_reads(self) -> AbstractContextManager[None]:
        return self._port.source_reads()

    @property
    def local_waiter(self) -> LocalExecutionWaiter | None:
        return None

    async def load_execution(self, execution_id: str, *, tenant_id: str) -> ExecutionRecord | None:
        return await self.load_source_execution(execution_id, tenant_id=tenant_id)

    async def load_observed_authorized(
        self, execution_id: str, principal: Principal, action: AuthorizationAction,
    ) -> ExecutionRecord:
        return await self.load_authorized(execution_id, principal, action)

    @observed_query
    async def inspect(self, execution_id: str, *, principal: Principal) -> ExecutionView:
        execution = await self.load_authorized(execution_id, principal, AuthorizationAction.EXECUTION_READ)
        return project_execution_view(execution)

    @observed_query
    async def list_children(self, execution_id: str, *, principal: Principal) -> tuple[ExecutionView, ...]:
        await self.load_authorized(execution_id, principal, AuthorizationAction.EXECUTION_READ)
        children = await self._repositories.executions.list_children(execution_id, tenant_id=principal.tenant_id)
        return tuple(project_execution_view(child) for child in children)

    @consumed_terminal_query
    async def wait(
        self, execution_id: str, *, principal: Principal, timeout_seconds: float | None = None,
    ) -> ExecutionResult:
        if timeout_seconds is not None and timeout_seconds < 0:
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)

        async def wait_once() -> ExecutionResult:
            backoff = 1.0
            while True:
                execution = await self.load_authorized(execution_id, principal, AuthorizationAction.EXECUTION_READ)
                if execution.status is ExecutionStatus.RECOVERY_REQUIRED:
                    raise _execution_recovery_error(execution)
                if execution.status in {ExecutionStatus.SUCCEEDED, ExecutionStatus.FAILED, ExecutionStatus.CANCELLED}:
                    return await self.consume_result_from_execution(execution, tenant_id=principal.tenant_id)
                await asyncio.sleep(backoff)
                backoff = min(30.0, backoff * 2)

        try:
            return await asyncio.wait_for(wait_once(), timeout_seconds)
        except asyncio.TimeoutError as error:
            raise AIError(ErrorCode.EXECUTION_WAIT_TIMEOUT) from error

    async def acquire_dependency_hold(
        self,
        execution_id: str,
        *,
        tenant_id: str,
        hold_id: str,
    ) -> bool:
        created = await self._repositories.executions.acquire_dependency_hold(
            execution_id,
            tenant_id=tenant_id,
            hold_id=hold_id,
        )
        try:
            await self._handoff.acquire_hold(
                (tenant_id, execution_id),
                hold_id,
            )
        except BaseException:
            if created:
                await self._repositories.executions.release_dependency_hold(
                    execution_id,
                    tenant_id=tenant_id,
                    hold_id=hold_id,
                )
            raise
        return created

    async def release_dependency_hold(
        self,
        execution_id: str,
        *,
        tenant_id: str,
        hold_id: str,
    ) -> None:
        key = tenant_id, execution_id
        state, owner = await self._handoff.release_hold(key, hold_id)
        if owner and state is not None:
            await self._run_handoff_cleanup(execution_id, tenant_id, state)
        await self._repositories.executions.release_dependency_hold(
            execution_id,
            tenant_id=tenant_id,
            hold_id=hold_id,
        )
        await self.request_handoff_if_terminal(execution_id, tenant_id)

    async def request_terminal_handoff(
        self, execution_id: str, *, tenant_id: str, terminal_commit: bool = False,
    ) -> None:
        key = tenant_id, execution_id
        state, owner = await self._handoff.request_release(key, value=terminal_commit)
        if owner:
            await self._run_handoff_cleanup(execution_id, tenant_id, state)

    async def resume_terminal_handoff(
        self, execution_id: str, *, tenant_id: str,
    ) -> None:
        state = await self._handoff.claim_requested_release((tenant_id, execution_id))
        if state is not None:
            await self._run_handoff_cleanup(execution_id, tenant_id, state)

    async def request_handoff_if_terminal(
        self, execution_id: str, tenant_id: str
    ) -> None:
        current = await self.load_execution(execution_id, tenant_id=tenant_id)
        if current is not None and current.status in {
            ExecutionStatus.SUCCEEDED,
            ExecutionStatus.FAILED,
            ExecutionStatus.CANCELLED,
        }:
            await self.request_terminal_handoff(
                execution_id, tenant_id=tenant_id, terminal_commit=current.conversation,
            )

    async def _run_handoff_cleanup(
        self,
        execution_id: str,
        tenant_id: str,
        state: HandoffState[bool],
    ) -> None:
        key = tenant_id, execution_id
        waiter = self.local_waiter
        if waiter is not None and waiter.owns_execution(execution_id, tenant_id=tenant_id):
            await self._handoff.finish_release(key, state, succeeded=False)
            if not waiter.owns_execution(execution_id, tenant_id=tenant_id):
                await self.resume_terminal_handoff(execution_id, tenant_id=tenant_id)
            return
        try:
            if state.release_value:
                released = await self._release_terminal(
                    execution_id, tenant_id=tenant_id, terminal_commit=True,
                )
            else:
                released = await self._release_terminal(execution_id, tenant_id=tenant_id)
        except BaseException as error:
            await self._handoff.finish_release(key, state, succeeded=False)
            if not isinstance(error, Exception):
                raise
            _logger.error(
                "execution handoff cleanup failed: execution=%s",
                execution_id,
                exc_info=environ.debug,
            )
            return
        await self._handoff.finish_release(
            key,
            state,
            succeeded=released,
        )

    @asynccontextmanager
    async def execution_consumer(self, execution_id: str, tenant_id: str):
        key = tenant_id, execution_id
        state = await self._handoff.enter(key)
        try:
            yield
        finally:
            owner = await self._handoff.leave(key, state)
            if owner:
                await self._run_handoff_cleanup(execution_id, tenant_id, state)

    async def start_task(
        self,
        binding: TaskBindingContract,
        *,
        principal: Principal,
        input: Mapping[str, JsonValue],
        idempotency_key: str,
        correlation: Mapping[str, str | int],
        requires_task_invocation_capture: bool = False,
        budget_scope_id: str | None = None,
    ) -> ExecutionHandle:
        if not isinstance(requires_task_invocation_capture, bool):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        if not isinstance(binding, TaskBindingContract):
            raise TypeError("binding must be TaskBindingContract")
        if budget_scope_id is not None and (
            not isinstance(budget_scope_id, str) or not budget_scope_id.strip()
        ):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        normalized_input = normalize_json_value(dict(input))
        if not isinstance(normalized_input, dict):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        normalized_correlation = overlay_execution_correlation({}, correlation)
        execution_id = self._new_operation_id()
        await self._policy.authorize(
            principal,
            AuthorizationAction.EXECUTION_RUN,
            ResourceRef(ResourceKind.EXECUTION, execution_id, principal.tenant_id),
        )
        if budget_scope_id is not None:
            await self.authorize_budget_scope(budget_scope_id, principal)
        stored_input = StoredUserInput(
            "task-input-v1",
            StoredPayload.inline_json(normalized_input),
            {
                "version": 1,
                "kind": "task",
                "task_id": binding.id,
                "task_revision": binding.revision,
            },
        )
        scope = "execution.task"
        key_digest = compute_idempotency_key_digest(idempotency_key)
        request_digest = canonical_sha256(
            {
                "contract": "task-execution-request-v1",
                "principal": principal_identity_payload(principal),
                "binding_digest": binding.binding_digest,
                "input_digest": stored_input.digest,
                **({"budget_scope_id": budget_scope_id} if budget_scope_id is not None else {}),
                **({"requires_task_invocation_capture": True} if requires_task_invocation_capture else {}),
            }
        )
        now = datetime.now(timezone.utc)
        execution = ExecutionRecord(
            execution_id=execution_id,
            session_id=None,
            parent_execution_id=None,
            root_execution_id=execution_id,
            previous_execution_id=None,
            fork_base_execution_id=None,
            lineage_kind=ExecutionLineageKind.RUN,
            status=ExecutionStatus.PENDING_START,
            revision=0,
            event_seq=0,
            agent_run_seq=0,
            error_code=None,
            safe_error_details={},
            created_at=now,
            updated_at=now,
            mode=None,
            planning=None,
            thinking=None,
            binding=binding,
            principal_id=principal.principal_id,
            principal_kind=principal.kind,
            stored_user_input=stored_input,
            correlation=normalized_correlation,
            requires_task_invocation_capture=requires_task_invocation_capture,
            budget_scope_id=budget_scope_id,
            restart_checkpoints=True,
            durable_trace=True,
            conversation=False,
        )
        reservation = await self._repositories.executions.reserve_start(
            ExecutionStartReservation(
                execution,
                IdempotencyRecord(
                    scope=scope,
                    idempotency_key_digest=key_digest,
                    request_digest=request_digest,
                    resource_kind=ResourceKind.EXECUTION,
                    resource_id=execution_id,
                    status=IdempotencyStatus.RESERVED,
                    result_digest=None,
                    error_code=None,
                    created_at=now,
                    updated_at=now,
                ),
            )
        )
        current = reservation.execution
        if (
            current.requires_task_invocation_capture is not requires_task_invocation_capture
            or current.budget_scope_id != budget_scope_id
            or current.binding_digest != binding.binding_digest
            or current.binding != binding
            or current.stored_user_input.digest != stored_input.digest
            or current.principal_id != principal.principal_id
            or current.principal_kind != principal.kind
        ):
            raise AIError(ErrorCode.IDEMPOTENCY_CONFLICT)
        if reservation.idempotency.request_digest != request_digest:
            raise AIError(ErrorCode.IDEMPOTENCY_CONFLICT)
        if current.status is ExecutionStatus.PENDING_START:
            current = await self._repositories.executions.claim_start(
                ExecutionStartClaim(
                    current.execution_id,
                    current.revision,
                    current.event_seq,
                    scope,
                    key_digest,
                    request_digest,
                    now,
                )
            )
        elif current.status is ExecutionStatus.START_UNKNOWN:
            raise AIError(ErrorCode.EXECUTION_START_UNKNOWN)
        elif current.status not in {
            ExecutionStatus.STARTED,
            ExecutionStatus.WAITING_RETRY,
            ExecutionStatus.WAITING_DEFERRED,
            ExecutionStatus.RECOVERY_REQUIRED,
            ExecutionStatus.SUCCEEDED,
            ExecutionStatus.FAILED,
            ExecutionStatus.CANCELLED,
        }:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return ExecutionHandle(current.execution_id)

    async def claim_task_attempt(
        self,
        execution_id: str,
        *,
        principal: Principal,
    ) -> ExecutionView:
        current = await self.load_authorized(
            execution_id,
            principal,
            AuthorizationAction.EXECUTION_RUN,
        )
        if not isinstance(current.binding, TaskBindingContract):
            raise AIError(ErrorCode.RUNTIME_SERVICE_MISMATCH)
        if current.status in {
            ExecutionStatus.SUCCEEDED,
            ExecutionStatus.FAILED,
            ExecutionStatus.CANCELLED,
            ExecutionStatus.RECOVERY_REQUIRED,
        }:
            return project_execution_view(current)
        now = datetime.now(timezone.utc)
        if current.status is ExecutionStatus.WAITING_RETRY:
            if current.task_next_attempt_at is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if current.task_next_attempt_at > now:
                return project_execution_view(current)
        elif current.status is not ExecutionStatus.STARTED:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if current.task_deadline_at is not None and current.task_deadline_at <= now:
            await self.fail_task(
                execution_id,
                principal=principal,
                error=AIError(
                    ErrorCode.EXECUTION_WAIT_TIMEOUT,
                    safe_details={"reason": "deadline_exceeded"},
                ),
                attempt=project_execution_view(current),
            )
            current = await self.load_authorized(
                execution_id,
                principal,
                AuthorizationAction.EXECUTION_RUN,
            )
            return project_execution_view(current)
        if current.task_attempt >= current.binding.max_attempts:
            await self.fail_task(
                execution_id,
                principal=principal,
                error=AIError(
                    ErrorCode.TASK_NODE_FAILED,
                    safe_details={"reason": "attempts_exhausted"},
                ),
                attempt=project_execution_view(current),
            )
            current = await self.load_authorized(
                execution_id,
                principal,
                AuthorizationAction.EXECUTION_RUN,
            )
            return project_execution_view(current)
        deadline = current.task_deadline_at
        if deadline is None and current.binding.timeout_seconds is not None:
            deadline = now + timedelta(seconds=current.binding.timeout_seconds)
        updated = await self._repositories.executions.transition_task_execution(
            execution_id,
            tenant_id=self._repositories.executions.tenant_id,
            expected_revision=current.revision,
            expected_event_seq=current.event_seq,
            expected_status=current.status,
            next_status=ExecutionStatus.STARTED,
            task_attempt=current.task_attempt + 1,
            task_deadline_at=deadline,
            task_next_attempt_at=None,
            error_code=None,
            safe_error_details={},
            event_type=ExecutionEventType.EXECUTION_RESUMED.value,
            payload={"attempt": current.task_attempt + 1},
            occurred_at=now,
        )
        return project_execution_view(updated)

    async def schedule_task_retry(
        self,
        execution_id: str,
        *,
        principal: Principal,
        error_code: str,
        attempt: ExecutionView | None = None,
    ) -> ExecutionView:
        current = await self.load_authorized(
            execution_id,
            principal,
            AuthorizationAction.EXECUTION_RUN,
        )
        if (
            not isinstance(current.binding, TaskBindingContract)
            or current.status is not ExecutionStatus.STARTED
        ):
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        _require_task_attempt(current, attempt)
        if current.task_attempt >= current.binding.max_attempts:
            await self.fail_task(
                execution_id,
                principal=principal,
                error=AIError(
                    ErrorCode.TASK_NODE_FAILED,
                    safe_details={"reason": "attempts_exhausted"},
                ),
                attempt=attempt,
            )
            return await self.inspect(execution_id, principal=principal)
        now = datetime.now(timezone.utc)
        retry_at = now + timedelta(seconds=current.binding.retry_delay_seconds)
        if current.task_deadline_at is not None and retry_at >= current.task_deadline_at:
            await self.fail_task(
                execution_id,
                principal=principal,
                error=AIError(
                    ErrorCode.EXECUTION_WAIT_TIMEOUT,
                    safe_details={"reason": "deadline_exceeded"},
                ),
                attempt=attempt,
            )
            return await self.inspect(execution_id, principal=principal)
        updated = await self._repositories.executions.transition_task_execution(
            execution_id,
            tenant_id=self._repositories.executions.tenant_id,
            expected_revision=current.revision,
            expected_event_seq=current.event_seq,
            expected_status=ExecutionStatus.STARTED,
            next_status=ExecutionStatus.WAITING_RETRY,
            task_attempt=current.task_attempt,
            task_deadline_at=current.task_deadline_at,
            task_next_attempt_at=retry_at,
            error_code=error_code,
            safe_error_details={},
            event_type=ExecutionEventType.EXECUTION_RETRY_SCHEDULED.value,
            payload={"attempt": current.task_attempt, "retry_at": retry_at.isoformat()},
            occurred_at=now,
        )
        return project_execution_view(updated)

    async def require_task_recovery(
        self,
        execution_id: str,
        *,
        principal: Principal,
        error_code: str,
        attempt: ExecutionView | None = None,
    ) -> ExecutionView:
        current = await self.load_authorized(
            execution_id,
            principal,
            AuthorizationAction.EXECUTION_RUN,
        )
        if not isinstance(current.binding, TaskBindingContract):
            raise AIError(ErrorCode.RUNTIME_SERVICE_MISMATCH)
        _require_task_attempt(current, attempt)
        if current.status is ExecutionStatus.RECOVERY_REQUIRED:
            return project_execution_view(current)
        if current.status is not ExecutionStatus.STARTED:
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        now = datetime.now(timezone.utc)
        updated = await self._repositories.executions.transition_task_execution(
            execution_id,
            tenant_id=self._repositories.executions.tenant_id,
            expected_revision=current.revision,
            expected_event_seq=current.event_seq,
            expected_status=ExecutionStatus.STARTED,
            next_status=ExecutionStatus.RECOVERY_REQUIRED,
            task_attempt=current.task_attempt,
            task_deadline_at=current.task_deadline_at,
            task_next_attempt_at=None,
            error_code=error_code,
            safe_error_details={"effect": "unknown"},
            event_type=ExecutionEventType.EXECUTION_RECOVERY_REQUIRED.value,
            payload={"error_code": error_code, "safe_error_details": {"effect": "unknown"}},
            occurred_at=now,
        )
        return project_execution_view(updated)

    async def defer_task_input(
        self,
        execution_id: str,
        *,
        principal: Principal,
        wait_id: str,
    ) -> ExecutionView:
        current = await self.load_authorized(
            execution_id,
            principal,
            AuthorizationAction.EXECUTION_RUN,
        )
        if (
            not isinstance(current.binding, TaskBindingContract)
            or not isinstance(wait_id, str)
            or not wait_id.strip()
        ):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        if current.status is ExecutionStatus.WAITING_DEFERRED:
            return project_execution_view(current)
        if current.status is not ExecutionStatus.STARTED:
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        now = datetime.now(timezone.utc)
        updated = await self._repositories.executions.transition_task_execution(
            execution_id,
            tenant_id=self._repositories.executions.tenant_id,
            expected_revision=current.revision,
            expected_event_seq=current.event_seq,
            expected_status=ExecutionStatus.STARTED,
            next_status=ExecutionStatus.WAITING_DEFERRED,
            task_attempt=current.task_attempt,
            task_deadline_at=current.task_deadline_at,
            task_next_attempt_at=None,
            error_code=None,
            safe_error_details={},
            event_type=ExecutionEventType.EXTERNAL_REQUESTED.value,
            payload={"wait_id": wait_id},
            occurred_at=now,
        )
        return project_execution_view(updated)

    async def supply_task_input(
        self,
        execution_id: str,
        *,
        principal: Principal,
        value: JsonValue,
    ) -> ExecutionView:
        current = await self.load_authorized(
            execution_id,
            principal,
            AuthorizationAction.EXECUTION_RUN,
        )
        if not isinstance(current.binding, TaskBindingContract):
            raise AIError(ErrorCode.RUNTIME_SERVICE_MISMATCH)
        normalized = normalize_json_value(value)
        try:
            _validate_task_binding_output(current.binding, normalized)
        except AIError as error:
            if error.code is ErrorCode.STORAGE_INTEGRITY_ERROR:
                raise
            raise AIError(
                ErrorCode.OUTPUT_CONTRACT_INVALID,
                retryable=False,
            ) from error
        value_digest = canonical_sha256(normalized)
        if current.status is ExecutionStatus.SUCCEEDED:
            result = await self.result(execution_id, principal=principal)
            if canonical_sha256(result.output) != value_digest:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            return project_execution_view(current)
        if current.status is not ExecutionStatus.WAITING_DEFERRED:
            raise AIError(ErrorCode.TASK_NOT_READY)
        try:
            await self._complete_task_output(
                current,
                principal=principal,
                output=normalized,
                terminal_event_payload={
                    "task_input": "supplied",
                    "task_input_value_digest": value_digest,
                },
            )
        except AIError as error:
            if error.code not in {ErrorCode.EXECUTION_RESULT_CONFLICT, ErrorCode.STORAGE_CONFLICT}:
                raise
            latest = await self.inspect(execution_id, principal=principal)
            if latest.status is not ExecutionStatus.SUCCEEDED:
                raise
            result = await self.result(execution_id, principal=principal)
            if canonical_sha256(result.output) != value_digest:
                raise
            return latest
        return await self.inspect(execution_id, principal=principal)

    async def resume_task_not_applied(
        self,
        execution_id: str,
        *,
        principal: Principal,
    ) -> ExecutionView:
        current = await self.load_authorized(
            execution_id,
            principal,
            AuthorizationAction.EXECUTION_RUN,
        )
        if (
            not isinstance(current.binding, TaskBindingContract)
            or current.status is not ExecutionStatus.RECOVERY_REQUIRED
        ):
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        now = datetime.now(timezone.utc)
        if (
            current.task_deadline_at is not None
            and current.task_deadline_at <= now
        ):
            await self.fail_task(
                execution_id,
                principal=principal,
                error=AIError(
                    ErrorCode.EXECUTION_WAIT_TIMEOUT,
                    safe_details={
                        "task_effect": "not_applied",
                        "reason": "deadline_exceeded",
                    },
                ),
            )
            return await self.inspect(execution_id, principal=principal)
        if current.task_attempt >= current.binding.max_attempts:
            await self.fail_task(
                execution_id,
                principal=principal,
                error=AIError(
                    ErrorCode.TASK_NODE_FAILED,
                    safe_details={
                        "task_effect": "not_applied",
                        "reason": "attempts_exhausted",
                    },
                ),
            )
            return await self.inspect(execution_id, principal=principal)
        updated = await self._repositories.executions.transition_task_execution(
            execution_id,
            tenant_id=self._repositories.executions.tenant_id,
            expected_revision=current.revision,
            expected_event_seq=current.event_seq,
            expected_status=ExecutionStatus.RECOVERY_REQUIRED,
            next_status=ExecutionStatus.WAITING_RETRY,
            task_attempt=current.task_attempt,
            task_deadline_at=current.task_deadline_at,
            task_next_attempt_at=now,
            error_code=None,
            safe_error_details={},
            event_type=ExecutionEventType.EXECUTION_RESUMED.value,
            payload={"reason": "task_effect_not_applied"},
            occurred_at=now,
        )
        return project_execution_view(updated)

    async def resolve_task_effect(
        self,
        execution_id: str,
        *,
        principal: Principal,
        resolution: TaskEffectResolution,
    ) -> ExecutionView:
        if not isinstance(resolution, TaskEffectResolution):
            raise TypeError("resolution must be TaskEffectResolution")
        current = await self.load_authorized(
            execution_id,
            principal,
            AuthorizationAction.EXECUTION_RUN,
        )
        binding = current.binding
        if (
            not isinstance(binding, TaskBindingContract)
            or binding.effect_policy != "non_replay_safe"
        ):
            raise AIError(ErrorCode.TASK_NOT_READY)

        applied_output = (
            None
            if resolution.kind != "applied"
            else normalize_json_value(resolution.value)
        )
        applied_digest = (
            None
            if resolution.kind != "applied"
            else canonical_sha256(applied_output)
        )

        if current.status is ExecutionStatus.WAITING_RETRY:
            if resolution.kind != "not_applied":
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            return project_execution_view(current)
        if current.status is ExecutionStatus.SUCCEEDED:
            if resolution.kind != "applied" or applied_digest is None:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            result = await self.result(execution_id, principal=principal)
            if canonical_sha256(result.output) != applied_digest:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            return project_execution_view(current)
        if current.status is ExecutionStatus.FAILED:
            if (
                resolution.kind != "applied"
                or applied_digest is None
                or current.error_code != ErrorCode.OUTPUT_CONTRACT_INVALID.value
                or current.safe_error_details.get("task_effect") != "applied"
                or current.safe_error_details.get("task_effect_value_digest")
                != applied_digest
            ):
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            return project_execution_view(current)
        if current.status is ExecutionStatus.CANCELLED:
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        if current.status is not ExecutionStatus.RECOVERY_REQUIRED:
            raise AIError(ErrorCode.TASK_NOT_READY)
        if resolution.kind == "unknown":
            return project_execution_view(current)

        if resolution.kind == "not_applied":
            now = datetime.now(timezone.utc)
            if (
                current.task_deadline_at is not None
                and current.task_deadline_at <= now
            ):
                await self.fail_task(
                    execution_id,
                    principal=principal,
                    error=AIError(
                        ErrorCode.EXECUTION_WAIT_TIMEOUT,
                        retryable=False,
                        safe_details={
                            "task_effect": "not_applied",
                            "reason": "deadline_exceeded",
                        },
                    ),
                )
                return await self.inspect(execution_id, principal=principal)
            if current.task_attempt >= binding.max_attempts:
                await self.fail_task(
                    execution_id,
                    principal=principal,
                    error=AIError(
                        ErrorCode.TASK_NODE_FAILED,
                        retryable=False,
                        safe_details={
                            "task_effect": "not_applied",
                            "reason": "attempts_exhausted",
                        },
                    ),
                )
                return await self.inspect(execution_id, principal=principal)
            return await self.resume_task_not_applied(
                execution_id,
                principal=principal,
            )

        if applied_digest is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        try:
            _validate_task_binding_output(binding, applied_output)
        except AIError as error:
            if error.code is ErrorCode.STORAGE_INTEGRITY_ERROR:
                raise
            await self.fail_task(
                execution_id,
                principal=principal,
                error=AIError(
                    ErrorCode.OUTPUT_CONTRACT_INVALID,
                    retryable=False,
                    safe_details={
                        "task_effect": "applied",
                        "task_effect_value_digest": applied_digest,
                    },
                ),
            )
            return await self.inspect(execution_id, principal=principal)

        await self._complete_task_output(
            current,
            principal=principal,
            output=applied_output,
            terminal_event_payload={
                "task_effect": "applied",
                "task_effect_value_digest": applied_digest,
            },
        )
        return await self.inspect(execution_id, principal=principal)

    async def cancel_task(
        self,
        execution_id: str,
        *,
        principal: Principal,
    ) -> CancelExecutionResult:
        current = await self.load_authorized(
            execution_id,
            principal,
            AuthorizationAction.EXECUTION_RUN,
        )
        if not isinstance(current.binding, TaskBindingContract):
            raise AIError(ErrorCode.RUNTIME_SERVICE_MISMATCH)
        if current.status is ExecutionStatus.CANCELLED:
            return CancelExecutionResult(execution_id, True)
        if current.status in {ExecutionStatus.SUCCEEDED, ExecutionStatus.FAILED}:
            return CancelExecutionResult(execution_id, False)
        if current.status is ExecutionStatus.RECOVERY_REQUIRED:
            raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)
        if (
            current.binding.effect_policy == "non_replay_safe"
            and current.task_attempt > 0
            and current.status is ExecutionStatus.STARTED
        ):
            await self.require_task_recovery(
                execution_id,
                principal=principal,
                error_code=ErrorCode.TASK_EFFECT_UNKNOWN.value,
            )
            raise AIError(ErrorCode.TASK_EFFECT_UNKNOWN)
        now = datetime.now(timezone.utc)
        result = ResultRecord(
            None,
            StopReason.CANCELLED,
            UsageMetrics(),
            now,
        )
        terminal = replace(
            current,
            status=ExecutionStatus.CANCELLED,
            error_code=ErrorCode.EXECUTION_CANCELLED.value,
            safe_error_details={},
            error_diagnostics=None,
            task_next_attempt_at=None,
            updated_at=now,
        )
        idempotency = await self._task_terminal_idempotency(
            current,
            result,
            next_status=IdempotencyStatus.CANCELLED,
            error_code=ErrorCode.EXECUTION_CANCELLED.value,
        )
        await self._repositories.executions.commit_terminal(
            ExecutionTerminalCommit(
                current.revision,
                current.event_seq,
                terminal,
                result,
                ExecutionEventType.EXECUTION_CANCELLED,
                {},
                idempotency,
            )
        )
        return CancelExecutionResult(execution_id, True)

    async def complete_task(
        self,
        execution_id: str,
        *,
        principal: Principal,
        output: JsonValue,
        attempt: ExecutionView | None = None,
    ) -> ExecutionResult:
        current = await self.load_authorized(
            execution_id,
            principal,
            AuthorizationAction.EXECUTION_RUN,
        )
        return await self._complete_task_output(
            current,
            principal=principal,
            output=output,
            attempt=attempt,
        )

    async def _complete_task_output(
        self,
        current: ExecutionRecord,
        *,
        principal: Principal,
        output: JsonValue,
        terminal_event_payload: "Mapping[str, JsonValue] | None" = None,
        attempt: ExecutionView | None = None,
    ) -> ExecutionResult:
        if not isinstance(current.binding, TaskBindingContract):
            raise AIError(ErrorCode.RUNTIME_SERVICE_MISMATCH)
        if attempt is not None and current.task_attempt != attempt.task_attempt:
            raise AIError(ErrorCode.TASK_FENCE_STALE)
        execution_id = current.execution_id
        if current.status is ExecutionStatus.SUCCEEDED:
            result = await self.result(execution_id, principal=principal)
            if canonical_sha256(result.output) != canonical_sha256(output):
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            return result
        if attempt is not None:
            _require_task_attempt(current, attempt)
            if current.status is not ExecutionStatus.STARTED:
                raise AIError(ErrorCode.TASK_FENCE_STALE)
        if current.status not in {
            ExecutionStatus.STARTED,
            ExecutionStatus.WAITING_DEFERRED,
            ExecutionStatus.RECOVERY_REQUIRED,
        }:
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        normalized = normalize_json_value(output)
        payload = await self._store_task_output(
            normalized,
            tenant_id=self._repositories.executions.tenant_id,
        )
        now = datetime.now(timezone.utc)
        result = ResultRecord(
            payload,
            StopReason.END_TURN,
            UsageMetrics(),
            now,
        )
        terminal = replace(
            current,
            status=ExecutionStatus.SUCCEEDED,
            error_code=None,
            safe_error_details={},
            error_diagnostics=None,
            task_next_attempt_at=None,
            updated_at=now,
        )
        idempotency = await self._task_terminal_idempotency(
            current,
            result,
            next_status=IdempotencyStatus.COMPLETED,
        )
        await self._repositories.executions.commit_terminal(
            ExecutionTerminalCommit(
                current.revision,
                current.event_seq,
                terminal,
                result,
                ExecutionEventType.EXECUTION_SUCCEEDED,
                {}
                if terminal_event_payload is None
                else dict(terminal_event_payload),
                idempotency,
            )
        )
        return await self.result(execution_id, principal=principal)

    async def fail_task(
        self,
        execution_id: str,
        *,
        principal: Principal,
        error: AIError,
        attempt: ExecutionView | None = None,
    ) -> ExecutionResult:
        current = await self.load_authorized(
            execution_id,
            principal,
            AuthorizationAction.EXECUTION_RUN,
        )
        if not isinstance(current.binding, TaskBindingContract):
            raise AIError(ErrorCode.RUNTIME_SERVICE_MISMATCH)
        if attempt is not None and current.task_attempt != attempt.task_attempt:
            raise AIError(ErrorCode.TASK_FENCE_STALE)
        if current.status is ExecutionStatus.FAILED:
            result = await self.result(execution_id, principal=principal)
            if (
                current.error_code != error.code.value
                or dict(current.safe_error_details) != dict(error.safe_details)
            ):
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            return result
        if attempt is not None:
            _require_task_attempt(current, attempt)
        if current.status not in {
            ExecutionStatus.STARTED,
            ExecutionStatus.WAITING_RETRY,
            ExecutionStatus.WAITING_DEFERRED,
            ExecutionStatus.RECOVERY_REQUIRED,
        }:
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        now = datetime.now(timezone.utc)
        details = dict(error.safe_details)
        result = ResultRecord(
            None,
            StopReason.ERROR,
            UsageMetrics(),
            now,
        )
        terminal = replace(
            current,
            status=ExecutionStatus.FAILED,
            error_code=error.code.value,
            safe_error_details=details,
            error_diagnostics=error.diagnostics,
            task_next_attempt_at=None,
            updated_at=now,
        )
        idempotency = await self._task_terminal_idempotency(
            current,
            result,
            next_status=IdempotencyStatus.FAILED,
            error_code=error.code.value,
        )
        payload = {
            "error_code": error.code.value,
            "safe_error_details": details,
        }
        if error.diagnostics is not None:
            payload["error_diagnostics"] = error.diagnostics.to_payload()
        await self._repositories.executions.commit_terminal(
            ExecutionTerminalCommit(
                current.revision,
                current.event_seq,
                terminal,
                result,
                ExecutionEventType.EXECUTION_FAILED,
                payload,
                idempotency,
            )
        )
        return await self.result(execution_id, principal=principal)

    async def _store_task_output(
        self,
        output: JsonValue,
        *,
        tenant_id: str,
    ) -> StoredPayload:
        inline = StoredPayload.inline_json(output)
        if self._output_policy is None or payload_fits_inline(inline, self._output_policy):
            return inline
        reference = await self._port.write_output(canonical_json_bytes(output), tenant_id=tenant_id)
        return StoredPayload.object(reference)

    async def _task_terminal_idempotency(
        self,
        execution: ExecutionRecord,
        result: ResultRecord,
        *,
        next_status: IdempotencyStatus,
        error_code: str | None = None,
    ) -> IdempotencyTerminalUpdate:
        records = await self._repositories.idempotency.list_by_resource(
            ResourceKind.EXECUTION,
            execution.execution_id,
            tenant_id=self._repositories.executions.tenant_id,
        )
        if len(records) != 1:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        record = records[0]
        if record.status is not IdempotencyStatus.STARTED:
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        return IdempotencyTerminalUpdate(
            record.scope,
            record.idempotency_key_digest,
            IdempotencyStatus.STARTED,
            next_status,
            record.request_digest,
            None if result.output is None else result.output.digest,
            error_code,
        )

    async def budget_usage(
        self, execution_id: str, *, principal: Principal,
    ) -> BudgetUsage | None:
        execution = await self.load_observed_authorized(
            execution_id, principal, AuthorizationAction.EXECUTION_READ,
        )
        if execution.budget_scope_id is None:
            return None
        return await self._repositories.budgets.read(execution.budget_scope_id)

    @consumed_terminal_query
    async def result(
        self, execution_id: str, *, principal: Principal
    ) -> ExecutionResult:
        execution = await self.load_authorized(
            execution_id, principal, AuthorizationAction.EXECUTION_READ
        )
        return await self.consume_result_from_execution(execution, tenant_id=principal.tenant_id)

    async def consume_result_from_execution(
        self, execution: ExecutionRecord, *, tenant_id: str
    ) -> ExecutionResult:
        result = await self._result_from_execution(execution, tenant_id=tenant_id)
        await self._require_retained_turn(execution)
        stored = execution.result
        if not self._port.embedded_results or (
            stored is not None and stored.output is not None and stored.output.kind != "inline"
        ):
            # Body and custom-owner reads may yield after the terminal snapshot.
            await self.request_handoff_if_terminal(execution.execution_id, tenant_id=tenant_id)
        else:
            await self.request_terminal_handoff(
                execution.execution_id, tenant_id=tenant_id, terminal_commit=execution.conversation,
            )
        return result

    async def _require_retained_turn(self, execution: ExecutionRecord) -> None:
        if (not execution.retention_closed and execution.conversation
                and execution.session_id is not None and self._session_repository is not None):
            await self._session_repository.state_store.read(
                lambda transaction: self._session_repository.pending_turn_commit_in_transaction(
                    transaction, session_id=execution.session_id,
                    execution_id=execution.execution_id,
                    allow_unadmitted=execution.started_at is None,
                )
            )

    async def _result_from_execution(self, execution: ExecutionRecord, *, tenant_id: str) -> ExecutionResult:
        return await self._manager.result_from_execution(execution, tenant_id=tenant_id)

    @consumed_terminal_query
    async def result_payload_size(
        self, execution_id: str, *, principal: Principal
    ) -> int:
        execution = await self.load_authorized(
            execution_id,
            principal,
            AuthorizationAction.EXECUTION_READ,
        )
        if execution.status is ExecutionStatus.RECOVERY_REQUIRED:
            raise _execution_recovery_error(execution)
        if execution.status is not ExecutionStatus.SUCCEEDED:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        repository = self._repositories.executions
        result = (execution.result if self._port.embedded_results else
                  await repository.get_result(execution_id, tenant_id=principal.tenant_id))
        if result is None or result.output is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        await self._require_retained_turn(execution)
        if self._port.embedded_results:
            await self.request_terminal_handoff(
                execution_id, tenant_id=principal.tenant_id, terminal_commit=execution.conversation,
            )
        else:
            await self.request_handoff_if_terminal(execution_id, tenant_id=principal.tenant_id)
        return result.output.size

    async def authorize_budget_scope(
        self,
        scope_id: str,
        principal: Principal,
        *,
        source_execution_id: str | None = None,
    ) -> None:
        if source_execution_id is not None or scope_id.startswith("execution:"):
            source = await self.load_authorized(
                source_execution_id or scope_id[len("execution:"):],
                principal,
                AuthorizationAction.EXECUTION_RUN,
            )
            await self._policy.authorize(
                principal,
                AuthorizationAction.EXECUTION_RUN,
                ResourceRef(
                    ResourceKind.EXECUTION,
                    source.execution_id,
                    principal.tenant_id,
                    source.principal_id,
                ),
            )
            if source.budget_scope_id != scope_id:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            await self._repositories.budgets.read(scope_id)
            return
        if scope_id.startswith("graph:"):
            if self._admissions is None:
                raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
            graph_id = scope_id[len("graph:"):]
            admission = await self._admissions.get(
                graph_id, tenant_id=principal.tenant_id,
            )
            if admission is None:
                raise AIError(ErrorCode.AUTHORIZATION_DENIED)
            await self._policy.authorize(
                principal,
                AuthorizationAction.TASK_RUN,
                ResourceRef(
                    ResourceKind.TASK_GRAPH,
                    admission.graph_id,
                    admission.principal.tenant_id,
                    admission.principal.principal_id,
                ),
            )
            if admission.graph_id != graph_id or admission.budget_scope_id != scope_id:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            usage = await self._repositories.budgets.read(scope_id)
            if usage.limits != admission.budget:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            return
        raise AIError(ErrorCode.AUTHORIZATION_DENIED)

    async def load_source_execution(
        self, execution_id: str, *, tenant_id: str,
    ) -> ExecutionRecord | None:
        return await self._port.read_source_execution(execution_id, tenant_id=tenant_id)

    async def load_authorized(self, execution_id: str, principal: Principal,
                               action: AuthorizationAction) -> ExecutionRecord:
        return await self._manager.load_authorized(execution_id, principal, action)

    async def authorize_execution(
        self, execution_id: str, principal: Principal, action: AuthorizationAction
    ) -> ResourceRef:
        return await self._manager.authorize_execution(execution_id, principal, action)

    async def authorize_loaded_execution(self, execution: ExecutionRecord, principal: Principal,
        action: AuthorizationAction, *, owner_principal_id: str | None) -> None:
        await self._manager.authorize_loaded_execution(execution, principal, action,
                                                         owner_principal_id=owner_principal_id)

    async def load_source_header(
        self, execution_id: str, *, tenant_id: str,
    ) -> ResourceRef | None:
        return await self._port.read_source_header(execution_id, tenant_id=tenant_id)



__all__ = ["TaskExecutionService", "ExecutionReleaseCallback", "LocalExecutionWaiter", "overlay_execution_correlation", "consumed_terminal_query", "observed_query"]
