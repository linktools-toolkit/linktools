#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluation query API and persistence-backed default service."""

import asyncio
import uuid
from dataclasses import replace
from datetime import datetime, timezone

from linktools.core import environ

from ..agent import AgentBindingContract
from ..core import (
    AuthorizationAction,
    AuthorizationPolicy,
    EvaluationStatus,
    ExecutionStatus,
    IdempotencyStatus,
    Principal,
    ResourceKind,
    ResourceRef,
    canonical_sha256,
    idempotency_key_digest as compute_idempotency_key_digest,
    principal_identity_payload,
)
from ..errors import AIError, ErrorCode
from .service_api import (
    CompareEvaluationRequest,
    EvaluationComparison,
    EvaluationHandle,
    EvaluationView,
    ExecutionHandle,
    ExecutionRequest,
    ExecutionService,
    ReplayEvaluationRequest,
    StartEvaluationRequest,
)
from .state._contracts import (
    EvaluationRecord,
    EvaluationRepositories,
    ExecutionRecord,
    ExecutionRepository,
    IdempotencyRecord,
)

_logger = environ.get_logger("ai.runtime.evaluation")
_TERMINAL_EVALUATION_STATUSES = frozenset(
    {
        EvaluationStatus.SUCCEEDED,
        EvaluationStatus.FAILED,
        EvaluationStatus.CANCELLED,
    }
)
_EXECUTION_EVALUATION_STATUS = {
    ExecutionStatus.PENDING_START: EvaluationStatus.PENDING,
    ExecutionStatus.START_UNKNOWN: EvaluationStatus.RUNNING,
    ExecutionStatus.STARTED: EvaluationStatus.RUNNING,
    ExecutionStatus.FINALIZING: EvaluationStatus.RUNNING,
    ExecutionStatus.RECOVERY_REQUIRED: EvaluationStatus.RUNNING,
    ExecutionStatus.WAITING_DEFERRED: EvaluationStatus.RUNNING,
    ExecutionStatus.WAITING_RETRY: EvaluationStatus.RUNNING,
    ExecutionStatus.CANCELLING: EvaluationStatus.RUNNING,
    ExecutionStatus.SUCCEEDED: EvaluationStatus.SUCCEEDED,
    ExecutionStatus.FAILED: EvaluationStatus.FAILED,
    ExecutionStatus.CANCELLED: EvaluationStatus.CANCELLED,
}
_EVALUATION_STATUS_RANK = {
    EvaluationStatus.PENDING: 0,
    EvaluationStatus.RUNNING: 1,
    EvaluationStatus.SUCCEEDED: 2,
    EvaluationStatus.FAILED: 2,
    EvaluationStatus.CANCELLED: 2,
}


class DefaultEvaluationService:
    """Track evaluation executions and replay their exact historical binding."""

    def __init__(
        self,
        state: EvaluationRepositories,
        executions: ExecutionRepository,
        authorization: AuthorizationPolicy,
        execution: ExecutionService,
    ) -> None:
        self._state = state
        self._executions = executions
        self._authorization = authorization
        self._execution = execution

    async def start(
        self,
        binding_digest: str,
        request: StartEvaluationRequest,
        *,
        binding_contract: "AgentBindingContract | None" = None,
    ) -> EvaluationHandle:
        evaluation_id = uuid.uuid4().hex
        idempotency_key_digest = compute_idempotency_key_digest(request.idempotency_key)
        await self._authorization.authorize(
            request.principal,
            AuthorizationAction.EVALUATION_RUN,
            ResourceRef(
                ResourceKind.EVALUATION,
                evaluation_id,
                request.principal.tenant_id,
            ),
        )
        request_digest = canonical_sha256(
            {
                "action": "evaluation.run",
                "principal": principal_identity_payload(request.principal),
                "dataset_id": request.dataset_id,
                "memory_scope": request.memory_scope,
                "binding_digest": binding_digest,
            }
        )
        existing = await self._state.idempotency.get(
            "evaluation.run",
            idempotency_key_digest,
            tenant_id=request.principal.tenant_id,
        )
        if existing is not None:
            if existing.request_digest != request_digest:
                raise AIError(ErrorCode.IDEMPOTENCY_CONFLICT)
            if existing.status is IdempotencyStatus.FAILED:
                raise _stable_error(existing.error_code)
            if existing.resource_kind is not ResourceKind.EVALUATION:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            evaluation_id = existing.resource_id
            now = existing.created_at
        else:
            now = datetime.now(timezone.utc)
            await self._state.idempotency.reserve(
                IdempotencyRecord(
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
                )
            )

        try:
            record = await self._state.records.get(
                evaluation_id,
                tenant_id=request.principal.tenant_id,
            )
            if record is None:
                if (
                    existing is not None
                    and existing.status is IdempotencyStatus.COMPLETED
                ):
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                execution = await self._execution.start(
                    binding_digest,
                    ExecutionRequest(
                        user_prompt=f"evaluation:{request.dataset_id}",
                        principal=request.principal,
                        idempotency_key=f"evaluation:{request.idempotency_key}",
                        memory_scope=request.memory_scope,
                        mode="run",
                        planning=False,
                        thinking=False,
                    ),
                    binding_contract=binding_contract,
                )
                await self._state.records.create(
                    EvaluationRecord(
                        evaluation_id,
                        execution.execution_id,
                        request.dataset_id,
                        EvaluationStatus.PENDING,
                        0,
                        now,
                        now,
                    )
                )
            else:
                source = await self._require_execution(record)
                if source.binding_digest != binding_digest:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                if (
                    existing is not None
                    and existing.status is not IdempotencyStatus.COMPLETED
                    and record.status in _TERMINAL_EVALUATION_STATUSES
                ):
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        except asyncio.CancelledError:
            raise
        except Exception:
            _logger.warning(
                "evaluation reservation remains recoverable: evaluation=%s tenant=%s",
                evaluation_id,
                request.principal.tenant_id,
                exc_info=environ.debug,
            )
            raise

        try:
            await self._state.idempotency.compare_and_swap(
                "evaluation.run",
                idempotency_key_digest,
                tenant_id=request.principal.tenant_id,
                expected_status=IdempotencyStatus.RESERVED,
                next_record=IdempotencyRecord(
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
        _logger.info(
            "evaluation submitted: evaluation=%s tenant=%s",
            evaluation_id,
            request.principal.tenant_id,
        )
        return EvaluationHandle(evaluation_id)

    async def inspect(
        self,
        evaluation_id: str,
        *,
        principal: Principal,
    ) -> EvaluationView:
        record, _source = await self._synchronize(
            await self._authorized(
                evaluation_id,
                principal,
                AuthorizationAction.EVALUATION_READ,
            )
        )
        return EvaluationView(record.evaluation_id, record.status)

    async def compare(
        self,
        request: CompareEvaluationRequest,
    ) -> EvaluationComparison:
        baseline, baseline_execution = await self._synchronize(
            await self._authorized(
                request.baseline_id,
                request.principal,
                AuthorizationAction.EVALUATION_READ,
            )
        )
        if request.candidate_id == request.baseline_id:
            candidate = baseline
            candidate_execution = baseline_execution
        else:
            candidate, candidate_execution = await self._synchronize(
                await self._authorized(
                    request.candidate_id,
                    request.principal,
                    AuthorizationAction.EVALUATION_READ,
                )
            )
        await self._authorization.authorize(
            request.principal,
            AuthorizationAction.EVALUATION_COMPARE,
            ResourceRef(
                ResourceKind.EVALUATION,
                request.candidate_id,
                request.principal.tenant_id,
            ),
        )
        if (
            baseline.dataset_id != candidate.dataset_id
            or baseline_execution.binding_digest
            != candidate_execution.binding_digest
        ):
            raise AIError(ErrorCode.EVALUATION_INCOMPATIBLE)
        return EvaluationComparison(
            request.baseline_id,
            request.candidate_id,
        )

    async def replay(
        self,
        agent_id: str,
        evaluation_id: str,
        request: ReplayEvaluationRequest,
    ) -> ExecutionHandle:
        record, source = await self._synchronize(
            await self._authorized(
                evaluation_id,
                request.principal,
                AuthorizationAction.EVALUATION_READ,
            )
        )
        if source.binding.agent_spec.id != agent_id:
            raise AIError(ErrorCode.EVALUATION_INCOMPATIBLE)
        return await self._execution.start(
            source.binding_digest,
            ExecutionRequest(
                user_prompt=f"evaluation:{record.dataset_id}",
                principal=request.principal,
                idempotency_key=request.idempotency_key,
                memory_scope=request.memory_scope,
                mode="run",
                planning=False,
                thinking=False,
            ),
            binding_contract=source.binding,
        )

    async def _synchronize(
        self,
        record: EvaluationRecord,
    ) -> tuple[EvaluationRecord, ExecutionRecord]:
        current = record
        while True:
            execution = await self._require_execution(current)
            target_status = _EXECUTION_EVALUATION_STATUS.get(execution.status)
            if target_status is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if current.status in _TERMINAL_EVALUATION_STATUSES:
                if target_status is not current.status:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                return current, execution
            target_rank = _EVALUATION_STATUS_RANK[target_status]
            current_rank = _EVALUATION_STATUS_RANK[current.status]
            if target_rank < current_rank:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if target_rank == current_rank:
                return current, execution
            updated = replace(
                current,
                status=target_status,
                revision=current.revision + 1,
                updated_at=datetime.now(timezone.utc),
            )
            try:
                result = await self._state.records.compare_and_swap(
                    current.evaluation_id,
                    tenant_id=self._state.records.tenant_id,
                    expected_revision=current.revision,
                    next_record=updated,
                )
                return result, execution
            except AIError as error:
                if error.code is not ErrorCode.STORAGE_CONFLICT:
                    raise
                latest = await self._state.records.get(
                    current.evaluation_id,
                    tenant_id=self._state.records.tenant_id,
                )
                if latest is None or latest.revision <= current.revision:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
                current = latest

    async def _require_execution(
        self,
        record: EvaluationRecord,
    ) -> ExecutionRecord:
        execution = await self._executions.get(
            record.execution_id,
            tenant_id=self._state.records.tenant_id,
        )
        if execution is None or not isinstance(execution.binding, AgentBindingContract):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return execution

    async def _authorized(
        self,
        evaluation_id: str,
        principal: Principal,
        action: AuthorizationAction,
    ) -> EvaluationRecord:
        header = await self._state.records.get_header(
            evaluation_id,
            tenant_id=principal.tenant_id,
        )
        if header is None:
            raise AIError(ErrorCode.AUTHORIZATION_DENIED)
        await self._authorization.authorize(principal, action, header)
        record = await self._state.records.get(
            evaluation_id,
            tenant_id=principal.tenant_id,
        )
        if record is None:
            raise AIError(ErrorCode.AUTHORIZATION_DENIED)
        return record


def _stable_error(error_code: str | None) -> AIError:
    if error_code is None:
        return AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    try:
        code = ErrorCode(error_code)
    except ValueError:
        return AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return AIError(code)


__all__ = ["DefaultEvaluationService"]
