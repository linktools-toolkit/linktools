#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluation query API and persistence-backed default service."""

import asyncio
import uuid
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Protocol

from linktools.core import environ

from ..core import (
    AuthorizationAction,
    AuthorizationPolicy,
    EvaluationStatus,
    IdempotencyStatus,
    Principal,
    ResourceKind,
    ResourceRef,
    canonical_sha256,
    idempotency_key_digest as compute_idempotency_key_digest,
)
from ..errors import AIError, ErrorCode
from ..observe import RunSnapshot
from .service_api import (
    CompareEvaluationRequest,
    EvaluationComparison,
    EvaluationHandle,
    EvaluationView,
    ExecutionHandle,
    ExecutionRequest,
    ExecutionService,
    ReplayEvaluationRequest,
    RunEvaluationRequest,
)
from .state._contracts import (
    EvaluationRecord,
    EvaluationState,
    ExecutionRepository,
    IdempotencyRecord,
)
from .state._plan import RuntimeDomain

_logger = environ.get_logger("ai.runtime.evaluation")


class _EvaluationReleaseCallback(Protocol):
    async def __call__(self, evaluation_id: str, *, tenant_id: str) -> None: ...


class _ExecutionHoldCallback(Protocol):
    async def __call__(self, execution_id: str, *, tenant_id: str, hold_id: str) -> None: ...


class _ExecutionHandoffCallback(Protocol):
    async def __call__(self, execution_id: str, *, tenant_id: str) -> None: ...


async def _no_release_terminal(evaluation_id: str, *, tenant_id: str) -> None:
    del evaluation_id, tenant_id


async def _no_execution_hold(execution_id: str, *, tenant_id: str, hold_id: str) -> None:
    del execution_id, tenant_id, hold_id


async def _no_execution_handoff(execution_id: str, *, tenant_id: str) -> None:
    del execution_id, tenant_id


@dataclass
class _EvaluationHandoffState:
    active_consumers: int = 0
    release_requested: bool = False
    release_in_progress: bool = False


def validate_compare_request(request: CompareEvaluationRequest) -> None:
    values = (
        request.baseline_id,
        request.candidate_id,
        request.dataset_id,
        request.evaluator_contract_id,
        request.target_kind,
        request.snapshot_digest,
        request.artifact_digest,
        request.output_schema_fingerprint,
    )
    revisions = (
        request.dataset_revision,
        request.evaluator_contract_revision,
        request.metric_contract_revision,
    )
    if any(value is None or not value.strip() for value in values) or any(value is None or value < 1 for value in revisions):
        raise AIError(ErrorCode.EVALUATION_INCOMPATIBLE)


class EvaluationQueryApi(Protocol):
    async def inspect(self, evaluation_id: str, *, principal: Principal) -> EvaluationView: ...
    async def compare(self, request: CompareEvaluationRequest) -> EvaluationComparison: ...
    async def snapshot(self, evaluation_id: str, *, principal: Principal) -> RunSnapshot: ...


class EvaluationApi(EvaluationQueryApi, Protocol):
    async def run(self, request: RunEvaluationRequest) -> EvaluationHandle: ...
    async def replay(self, snapshot_id: str, request: ReplayEvaluationRequest) -> ExecutionHandle: ...


class DefaultEvaluationService:
    """Persist evaluation identity and enforce compatibility before replay."""

    def __init__(self, state: EvaluationState, executions: ExecutionRepository, authorization: AuthorizationPolicy, execution: ExecutionService, *, release_terminal: _EvaluationReleaseCallback | None = None, acquire_execution_hold: _ExecutionHoldCallback | None = None, release_execution_hold: _ExecutionHoldCallback | None = None, request_execution_handoff: _ExecutionHandoffCallback | None = None) -> None:
        hold_callbacks = (acquire_execution_hold, release_execution_hold, request_execution_handoff)
        if any(callback is None for callback in hold_callbacks) and any(callback is not None for callback in hold_callbacks):
            raise ValueError("evaluation execution hold callbacks must be complete")
        self._state = state
        self._executions = executions
        self._authorization = authorization
        self._execution = execution
        self._release_terminal = release_terminal or _no_release_terminal
        self._acquire_execution_hold = acquire_execution_hold or _no_execution_hold
        self._release_execution_hold = release_execution_hold or _no_execution_hold
        self._request_execution_handoff = request_execution_handoff or _no_execution_handoff
        self._handoff_states: dict[tuple[str, str], _EvaluationHandoffState] = {}
        self._handoff_condition = asyncio.Condition()

    async def run(self, binding_digest: str, output_schema_fingerprint: str, request: RunEvaluationRequest) -> EvaluationHandle:
        evaluation_id = uuid.uuid4().hex
        idempotency_key_digest = compute_idempotency_key_digest(request.idempotency_key)
        await self._authorization.authorize(request.principal, AuthorizationAction.EVALUATION_RUN, ResourceRef(ResourceKind.EVALUATION, evaluation_id, request.principal.tenant_id))
        request_digest = canonical_sha256({"action": "evaluation.run", "tenant_id": request.principal.tenant_id, "principal_id": request.principal.principal_id, "dataset_digest": request.dataset_digest, "binding": binding_digest, "output_schema_fingerprint": output_schema_fingerprint})
        existing = await self._state.idempotency.get(
            "evaluation.run",
            idempotency_key_digest,
            tenant_id=request.principal.tenant_id,
        )
        if existing is not None:
            if existing.request_digest != request_digest:
                raise AIError(ErrorCode.IDEMPOTENCY_CONFLICT)
            if existing.status is IdempotencyStatus.FAILED:
                raise _stable_error(existing.error_code, ErrorCode.STORAGE_UNAVAILABLE)
            if existing.runtime_domain is not RuntimeDomain.EVALUATION or existing.resource_kind is not ResourceKind.EVALUATION:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            evaluation_id = existing.resource_id
            now = existing.created_at
        else:
            now = datetime.now(timezone.utc)
            await self._state.idempotency.reserve(IdempotencyRecord(
                tenant_id=request.principal.tenant_id,
                runtime_domain=RuntimeDomain.EVALUATION,
                scope="evaluation.run",
                idempotency_key_digest=idempotency_key_digest,
                request_digest=request_digest,
                resource_kind=ResourceKind.EVALUATION,
                resource_id=evaluation_id,
                status=IdempotencyStatus.RESERVED,
                result_digest=None,
                error_code=None,
                created_at=now,
                updated_at=now,
            ))
        try:
            record = await self._state.records.get(evaluation_id, tenant_id=request.principal.tenant_id)
            if record is None:
                if existing is not None and existing.status is IdempotencyStatus.COMPLETED:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                execution = await self._execution.run(
                    binding_digest,
                    ExecutionRequest(
                        user_prompt=f"evaluation:{request.dataset_digest}",
                        principal=request.principal,
                        idempotency_key=f"evaluation:{request.idempotency_key}",
                        memory_scope=request.memory_scope,
                    ),
                )
                await self._acquire_execution_hold(execution.execution_id, tenant_id=request.principal.tenant_id, hold_id=f"evaluation:{evaluation_id}")
                record = EvaluationRecord(evaluation_id, request.principal.tenant_id, execution.execution_id, request.dataset_digest, 1, "default", 1, binding_digest, output_schema_fingerprint, None, EvaluationStatus.PENDING, 0, {}, now, now)
                await self._state.records.create(record)
            elif record.status not in {EvaluationStatus.SUCCEEDED, EvaluationStatus.FAILED, EvaluationStatus.CANCELLED}:
                execution = await self._execution_record(record)
                if execution is not None:
                    await self._acquire_execution_hold(record.execution_id, tenant_id=request.principal.tenant_id, hold_id=f"evaluation:{evaluation_id}")
            elif existing is not None and existing.status is not IdempotencyStatus.COMPLETED:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        except asyncio.CancelledError:
            raise
        except Exception:
            _logger.warning("evaluation reservation remains recoverable: evaluation=%s tenant=%s", evaluation_id, request.principal.tenant_id, exc_info=environ.debug)
            raise
        try:
            await self._state.idempotency.compare_and_swap(
                "evaluation.run",
                idempotency_key_digest,
                tenant_id=request.principal.tenant_id,
                expected_status=IdempotencyStatus.RESERVED,
                next_record=IdempotencyRecord(
                tenant_id=request.principal.tenant_id,
                runtime_domain=RuntimeDomain.EVALUATION,
                scope="evaluation.run",
                idempotency_key_digest=idempotency_key_digest,
                request_digest=request_digest,
                resource_kind=ResourceKind.EVALUATION,
                resource_id=evaluation_id,
                status=IdempotencyStatus.COMPLETED,
                result_digest=None,
                error_code=None,
                created_at=now,
                updated_at=datetime.now(timezone.utc),
                ),
            )
        except AIError as error:
            if error.code is not ErrorCode.STORAGE_CONFLICT:
                raise
            current = await self._state.idempotency.get(
                "evaluation.run",
                idempotency_key_digest,
                tenant_id=request.principal.tenant_id,
            )
            if current is None or current.status is not IdempotencyStatus.COMPLETED:
                raise
        _logger.info("evaluation submitted: evaluation=%s tenant=%s", evaluation_id, request.principal.tenant_id)
        return EvaluationHandle(evaluation_id)

    async def inspect(self, evaluation_id: str, *, principal: Principal) -> EvaluationView:
        async with self._evaluation_consumer(evaluation_id, principal.tenant_id):
            record = await self._authorized(evaluation_id, principal, AuthorizationAction.EVALUATION_READ)
            record = await self._synchronize(record, principal=principal)
            view = EvaluationView(record.evaluation_id, record.status)
            if record.status in {EvaluationStatus.SUCCEEDED, EvaluationStatus.FAILED, EvaluationStatus.CANCELLED}:
                await self._request_evaluation_release(evaluation_id, principal.tenant_id)
            return view

    async def compare(self, request: CompareEvaluationRequest) -> EvaluationComparison:
        validate_compare_request(request)
        evaluation_ids = tuple(sorted({request.baseline_id, request.candidate_id}))
        async with AsyncExitStack() as stack:
            for evaluation_id in evaluation_ids:
                await stack.enter_async_context(self._evaluation_consumer(evaluation_id, request.principal.tenant_id))
            await self._authorization.authorize(request.principal, AuthorizationAction.EVALUATION_READ, ResourceRef(ResourceKind.EVALUATION, request.baseline_id, request.principal.tenant_id))
            await self._authorization.authorize(request.principal, AuthorizationAction.EVALUATION_READ, ResourceRef(ResourceKind.EVALUATION, request.candidate_id, request.principal.tenant_id))
            await self._authorization.authorize(request.principal, AuthorizationAction.EVALUATION_COMPARE, ResourceRef(ResourceKind.EVALUATION, request.candidate_id, request.principal.tenant_id))
            baseline = await self._synchronize(await self._authorized(request.baseline_id, request.principal, AuthorizationAction.EVALUATION_READ), principal=request.principal)
            candidate = baseline if request.candidate_id == request.baseline_id else await self._synchronize(await self._authorized(request.candidate_id, request.principal, AuthorizationAction.EVALUATION_READ), principal=request.principal)
            if (
                baseline.dataset_id != candidate.dataset_id
                or baseline.dataset_revision != candidate.dataset_revision
                or baseline.evaluator_id != candidate.evaluator_id
                or baseline.evaluator_revision != candidate.evaluator_revision
                or baseline.binding_digest != candidate.binding_digest
                or baseline.output_schema_fingerprint != candidate.output_schema_fingerprint
            ):
                raise AIError(ErrorCode.EVALUATION_INCOMPATIBLE)
            comparison = EvaluationComparison(request.baseline_id, request.candidate_id, True)
            for record in (baseline, candidate):
                if record.status in {EvaluationStatus.SUCCEEDED, EvaluationStatus.FAILED, EvaluationStatus.CANCELLED}:
                    await self._request_evaluation_release(record.evaluation_id, request.principal.tenant_id)
            return comparison

    async def snapshot(self, evaluation_id: str, *, principal: Principal) -> RunSnapshot:
        async with self._evaluation_consumer(evaluation_id, principal.tenant_id):
            record = await self._synchronize(await self._authorized(evaluation_id, principal, AuthorizationAction.EVALUATION_READ), principal=principal)
            result_value = record.metrics.get("result_digest")
            result_digest = result_value if isinstance(result_value, str) else None
            digest = canonical_sha256({"snapshot_id": evaluation_id, "execution_id": record.execution_id, "binding_digest": record.binding_digest, "trace_digest": record.artifact_digest or "", "result_digest": result_digest})
            snapshot = RunSnapshot(evaluation_id, record.execution_id, record.binding_digest, record.artifact_digest or "", result_digest, digest)
            if record.status in {EvaluationStatus.SUCCEEDED, EvaluationStatus.FAILED, EvaluationStatus.CANCELLED}:
                await self._request_evaluation_release(evaluation_id, principal.tenant_id)
            return snapshot

    async def replay(self, binding_digest: str, snapshot_id: str, request: ReplayEvaluationRequest) -> ExecutionHandle:
        async with self._evaluation_consumer(snapshot_id, request.principal.tenant_id):
            record = await self._synchronize(await self._authorized(snapshot_id, request.principal, AuthorizationAction.EVALUATION_READ), principal=request.principal)
            if record.binding_digest != binding_digest:
                raise AIError(ErrorCode.EVALUATION_INCOMPATIBLE)
            handle = await self._execution.run(
                binding_digest,
                ExecutionRequest(
                    user_prompt=f"replay:{record.evaluation_id}",
                    principal=request.principal,
                    idempotency_key=request.idempotency_key,
                    memory_scope=request.memory_scope,
                ),
            )
            if record.status in {EvaluationStatus.SUCCEEDED, EvaluationStatus.FAILED, EvaluationStatus.CANCELLED}:
                await self._request_evaluation_release(snapshot_id, request.principal.tenant_id)
            return handle

    async def _synchronize(self, record: EvaluationRecord, *, principal: Principal) -> EvaluationRecord:
        del principal
        hold_id = f"evaluation:{record.evaluation_id}"
        current = record
        terminal_statuses = {EvaluationStatus.SUCCEEDED, EvaluationStatus.FAILED, EvaluationStatus.CANCELLED}
        status_rank = {EvaluationStatus.PENDING: 0, EvaluationStatus.RUNNING: 1, EvaluationStatus.SUCCEEDED: 2, EvaluationStatus.FAILED: 2, EvaluationStatus.CANCELLED: 2}
        execution_status_map = {
            "PENDING_START": EvaluationStatus.PENDING,
            "START_UNKNOWN": EvaluationStatus.RUNNING,
            "STARTED": EvaluationStatus.RUNNING,
            "WAITING_APPROVAL": EvaluationStatus.RUNNING,
            "WAITING_EXTERNAL": EvaluationStatus.RUNNING,
            "CANCELLING": EvaluationStatus.RUNNING,
            "SUCCEEDED": EvaluationStatus.SUCCEEDED,
            "FAILED": EvaluationStatus.FAILED,
            "CANCELLED": EvaluationStatus.CANCELLED,
        }
        while True:
            if current.status in terminal_statuses:
                await self._request_execution_handoff(current.execution_id, tenant_id=current.tenant_id)
                await self._release_execution_hold(current.execution_id, tenant_id=current.tenant_id, hold_id=hold_id)
                return current
            execution = await self._execution_record(current)
            if execution is None:
                await self._release_execution_hold(current.execution_id, tenant_id=current.tenant_id, hold_id=hold_id)
                _logger.warning("evaluation dependency missing: evaluation=%s execution=%s dependency_missing=True", current.evaluation_id, current.execution_id)
                return current
            await self._acquire_execution_hold(current.execution_id, tenant_id=current.tenant_id, hold_id=hold_id)
            target_status = execution_status_map.get(execution.status.value)
            if target_status is None or status_rank[target_status] <= status_rank[current.status]:
                return current
            updated = replace(current, status=target_status, revision=current.revision + 1, updated_at=datetime.now(timezone.utc))
            try:
                result = await self._state.records.compare_and_swap(current.evaluation_id, tenant_id=current.tenant_id, expected_revision=current.revision, next_record=updated)
            except AIError as error:
                if error.code is not ErrorCode.STORAGE_CONFLICT:
                    raise
                result = await self._state.records.get(current.evaluation_id, tenant_id=current.tenant_id)
                if result is None or result.revision <= current.revision:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
                current = result
                continue
            if result.status in terminal_statuses:
                await self._request_execution_handoff(result.execution_id, tenant_id=result.tenant_id)
                await self._release_execution_hold(result.execution_id, tenant_id=result.tenant_id, hold_id=hold_id)
            return result

    @asynccontextmanager
    async def _evaluation_consumer(self, evaluation_id: str, tenant_id: str):
        key = (tenant_id, evaluation_id)
        async with self._handoff_condition:
            while True:
                state = self._handoff_states.get(key)
                if state is None:
                    state = _EvaluationHandoffState()
                    self._handoff_states[key] = state
                if not state.release_in_progress:
                    state.active_consumers += 1
                    break
                await self._handoff_condition.wait()
        cleanup_owner = False
        try:
            yield state
        finally:
            async with self._handoff_condition:
                state.active_consumers -= 1
                if state.active_consumers < 0:
                    raise RuntimeError("evaluation consumer count became negative")
                if state.active_consumers == 0:
                    if state.release_requested and not state.release_in_progress:
                        state.release_in_progress = True
                        cleanup_owner = True
                    elif not state.release_requested and self._handoff_states.get(key) is state:
                        self._handoff_states.pop(key, None)
                self._handoff_condition.notify_all()
            if cleanup_owner:
                cleanup_succeeded = False
                try:
                    await self._release_terminal(evaluation_id, tenant_id=tenant_id)
                    cleanup_succeeded = True
                except BaseException:
                    _logger.error("evaluation transient handoff cleanup failed: evaluation=%s", evaluation_id, exc_info=environ.debug)
                async with self._handoff_condition:
                    if self._handoff_states.get(key) is state:
                        if cleanup_succeeded and state.active_consumers == 0:
                            self._handoff_states.pop(key, None)
                        else:
                            state.release_in_progress = False
                            state.release_requested = True
                    self._handoff_condition.notify_all()

    async def _request_evaluation_release(self, evaluation_id: str, tenant_id: str) -> None:
        key = (tenant_id, evaluation_id)
        async with self._handoff_condition:
            state = self._handoff_states.get(key)
            if state is None:
                raise RuntimeError("evaluation release requested without consumer")
            state.release_requested = True
            self._handoff_condition.notify_all()

    async def _execution_record(self, record: EvaluationRecord):
        return await self._executions.get(record.execution_id, tenant_id=record.tenant_id)

    async def _authorized(self, evaluation_id: str, principal: Principal, action: AuthorizationAction) -> EvaluationRecord:
        header = await self._state.records.get_header(evaluation_id, tenant_id=principal.tenant_id)
        if header is None:
            raise AIError(ErrorCode.AUTHORIZATION_DENIED)
        await self._authorization.authorize(principal, action, header)
        record = await self._state.records.get(evaluation_id, tenant_id=principal.tenant_id)
        if record is None:
            raise AIError(ErrorCode.AUTHORIZATION_DENIED)
        return record


def _stable_error(error_code: str | None, fallback: ErrorCode) -> AIError:
    try:
        return AIError(fallback if error_code is None else ErrorCode(error_code))
    except ValueError:
        return AIError(fallback)


__all__ = ["DefaultEvaluationService", "EvaluationApi", "EvaluationQueryApi", "validate_compare_request"]
