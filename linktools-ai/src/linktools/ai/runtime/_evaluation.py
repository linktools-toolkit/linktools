#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluation query API and persistence-backed default service."""

import asyncio
import uuid
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
    idempotency_key_hash,
)
from ..errors import AIError, ErrorCode
from ..observe import RunSnapshot
from ._persistence import EvaluationRecord, IdempotencyRecord, RuntimeStores
from ._services import (
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

_logger = environ.get_logger("ai.runtime.evaluation")


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

    def __init__(self, persistence: RuntimeStores, authorization: AuthorizationPolicy, execution: ExecutionService) -> None:
        self._persistence = persistence
        self._authorization = authorization
        self._execution = execution

    async def run(self, binding_digest: str, output_schema_fingerprint: str, request: RunEvaluationRequest) -> EvaluationHandle:
        evaluation_id = uuid.uuid4().hex
        key_hash = idempotency_key_hash(request.idempotency_key)
        await self._authorization.authorize(request.principal, AuthorizationAction.EVALUATION_RUN, ResourceRef(ResourceKind.EVALUATION, evaluation_id, request.principal.tenant_id))
        request_digest = canonical_sha256({"action": "evaluation.run", "tenant_id": request.principal.tenant_id, "principal_id": request.principal.principal_id, "dataset_digest": request.dataset_digest, "binding": binding_digest, "output_schema_fingerprint": output_schema_fingerprint})
        existing = await self._persistence.evaluation.idempotency.get("evaluation.run", key_hash, tenant_id=request.principal.tenant_id)
        if existing is not None:
            if existing.request_digest != request_digest:
                raise AIError(ErrorCode.IDEMPOTENCY_CONFLICT)
            if existing.status is IdempotencyStatus.FAILED:
                raise _stable_error(existing.error_code, ErrorCode.STORAGE_UNAVAILABLE)
            evaluation_id = existing.execution_id
            if existing.status is IdempotencyStatus.COMPLETED:
                return EvaluationHandle(evaluation_id)
            now = existing.created_at
        else:
            now = datetime.now(timezone.utc)
            await self._persistence.evaluation.idempotency.reserve(IdempotencyRecord(request.principal.tenant_id, "evaluation.run", key_hash, request_digest, evaluation_id, IdempotencyStatus.RESERVED, None, None, now, now))
        try:
            record = await self._persistence.evaluation.records.get(evaluation_id, tenant_id=request.principal.tenant_id)
            if record is None:
                execution = await self._execution.run(
                    binding_digest,
                    ExecutionRequest(
                        prompt=f"evaluation:{request.dataset_digest}",
                        principal=request.principal,
                        idempotency_key=f"evaluation:{request.idempotency_key}",
                        memory_scope=request.memory_scope,
                    ),
                )
                record = EvaluationRecord(evaluation_id, request.principal.tenant_id, execution.execution_id, request.dataset_digest, 1, "default", 1, binding_digest, output_schema_fingerprint, None, EvaluationStatus.PENDING, 0, {}, now, now)
                await self._persistence.evaluation.records.create(record)
        except asyncio.CancelledError:
            raise
        except Exception:
            _logger.warning("evaluation reservation remains recoverable: evaluation=%s tenant=%s", evaluation_id, request.principal.tenant_id, exc_info=environ.debug)
            raise
        try:
            await self._persistence.evaluation.idempotency.compare_and_swap("evaluation.run", key_hash, tenant_id=request.principal.tenant_id, expected_status=IdempotencyStatus.RESERVED, next_record=IdempotencyRecord(request.principal.tenant_id, "evaluation.run", key_hash, request_digest, evaluation_id, IdempotencyStatus.COMPLETED, None, None, now, datetime.now(timezone.utc)))
        except AIError as error:
            if error.code is not ErrorCode.STORAGE_CONFLICT:
                raise
            current = await self._persistence.evaluation.idempotency.get("evaluation.run", key_hash, tenant_id=request.principal.tenant_id)
            if current is None or current.status is not IdempotencyStatus.COMPLETED:
                raise
        _logger.info("evaluation submitted: evaluation=%s tenant=%s", evaluation_id, request.principal.tenant_id)
        return EvaluationHandle(evaluation_id)

    async def inspect(self, evaluation_id: str, *, principal: Principal) -> EvaluationView:
        record = await self._authorized(evaluation_id, principal, AuthorizationAction.EVALUATION_READ)
        record = await self._synchronize(record)
        return EvaluationView(record.evaluation_id, record.status)

    async def compare(self, request: CompareEvaluationRequest) -> EvaluationComparison:
        validate_compare_request(request)
        await self._authorization.authorize(request.principal, AuthorizationAction.EVALUATION_READ, ResourceRef(ResourceKind.EVALUATION, request.baseline_id, request.principal.tenant_id))
        await self._authorization.authorize(request.principal, AuthorizationAction.EVALUATION_READ, ResourceRef(ResourceKind.EVALUATION, request.candidate_id, request.principal.tenant_id))
        await self._authorization.authorize(request.principal, AuthorizationAction.EVALUATION_COMPARE, ResourceRef(ResourceKind.EVALUATION, request.candidate_id, request.principal.tenant_id))
        baseline = await self._synchronize(await self._authorized(request.baseline_id, request.principal, AuthorizationAction.EVALUATION_READ))
        candidate = await self._synchronize(await self._authorized(request.candidate_id, request.principal, AuthorizationAction.EVALUATION_READ))
        if (
            baseline.dataset_id != candidate.dataset_id
            or baseline.dataset_revision != candidate.dataset_revision
            or baseline.evaluator_id != candidate.evaluator_id
            or baseline.evaluator_revision != candidate.evaluator_revision
            or baseline.binding_digest != candidate.binding_digest
            or baseline.output_schema_fingerprint != candidate.output_schema_fingerprint
        ):
            raise AIError(ErrorCode.EVALUATION_INCOMPATIBLE)
        return EvaluationComparison(request.baseline_id, request.candidate_id, True)

    async def snapshot(self, evaluation_id: str, *, principal: Principal) -> RunSnapshot:
        record = await self._synchronize(await self._authorized(evaluation_id, principal, AuthorizationAction.EVALUATION_READ))
        result_value = record.metrics.get("result_digest")
        result_digest = result_value if isinstance(result_value, str) else None
        digest = canonical_sha256({"snapshot_id": evaluation_id, "execution_id": record.execution_id, "binding_digest": record.binding_digest, "trace_digest": record.artifact_digest or "", "result_digest": result_digest})
        return RunSnapshot(evaluation_id, record.execution_id, record.binding_digest, record.artifact_digest or "", result_digest, digest)

    async def replay(self, binding_digest: str, snapshot_id: str, request: ReplayEvaluationRequest) -> ExecutionHandle:
        record = await self._authorized(snapshot_id, request.principal, AuthorizationAction.EVALUATION_READ)
        if record.binding_digest != binding_digest:
            raise AIError(ErrorCode.EVALUATION_INCOMPATIBLE)
        return await self._execution.run(
            binding_digest,
            ExecutionRequest(
                prompt=f"replay:{record.evaluation_id}",
                principal=request.principal,
                idempotency_key=request.idempotency_key,
                memory_scope=request.memory_scope,
            ),
        )

    async def _synchronize(self, record: EvaluationRecord) -> EvaluationRecord:
        return record

    async def _authorized(self, evaluation_id: str, principal: Principal, action: AuthorizationAction) -> EvaluationRecord:
        header = await self._persistence.evaluation.records.get_header(evaluation_id, tenant_id=principal.tenant_id)
        if header is None:
            raise AIError(ErrorCode.AUTHORIZATION_DENIED)
        await self._authorization.authorize(principal, action, header)
        record = await self._persistence.evaluation.records.get(evaluation_id, tenant_id=principal.tenant_id)
        if record is None:
            raise AIError(ErrorCode.AUTHORIZATION_DENIED)
        return record


def _stable_error(error_code: str | None, fallback: ErrorCode) -> AIError:
    try:
        return AIError(fallback if error_code is None else ErrorCode(error_code))
    except ValueError:
        return AIError(fallback)


__all__ = ["DefaultEvaluationService", "EvaluationApi", "EvaluationQueryApi", "validate_compare_request"]
