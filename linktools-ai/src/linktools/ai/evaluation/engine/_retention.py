#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluation expiry and explicit, restartable retention maintenance."""

from contextlib import AbstractAsyncContextManager, nullcontext
from collections.abc import Awaitable, Callable
from datetime import datetime
from typing import TYPE_CHECKING

from ...core import (
    AuthorizationAction,
    AuthorizationPolicy,
    Principal,
    ResourceKind,
    ResourceRef,
    TaskStatus,
    validate_page_limit,
)
from ...errors import AIError, ErrorCode
from ...evaluation import EvaluationPurgeResult
from ...task import TaskSubmissionCancellation
from ...persistence.captures import task_graph_binding_capture_key
from ...persistence.snapshots import RuntimeSnapshotOperations, SnapshotExclusiveGuard
from ...persistence.captures import input_capture_key
from ...persistence.contracts import EvaluationLaunchIntent, EvaluationRecord
from ...persistence.contracts import EvaluationRepository

if TYPE_CHECKING:
    from ...inputs import InputCaptures

_TERMINAL = frozenset({TaskStatus.SUCCEEDED, TaskStatus.FAILED, TaskStatus.CANCELLED, TaskStatus.BLOCKED})


def require_evaluation_content(record: EvaluationRecord, *, now: datetime) -> None:
    if record.content_deleted_at is not None or any(
        deadline is not None and deadline <= now
        for deadline in (record.content_expires_at, record.metadata_expires_at)
    ):
        raise AIError(ErrorCode.EVALUATION_EVIDENCE_UNAVAILABLE, safe_details={"reason": "evidence_expired"})


class _HeldExclusive:
    """Cleanup shares the offline window already held by retention."""

    def offline_exclusivity(self) -> AbstractAsyncContextManager[None]:
        return nullcontext()


class EvaluationRetention:
    def __init__(
        self, authorization: AuthorizationPolicy,
        cancel_submission: Callable[[EvaluationRecord, EvaluationLaunchIntent, Principal, str], Awaitable[TaskSubmissionCancellation]],
        captures: "InputCaptures",
        *, tenant_id: str, repository: EvaluationRepository,
        maintenance: RuntimeSnapshotOperations, release_scopes: Callable[[str], Awaitable[None]] | None = None,
    ) -> None:
        self._tenant_id = tenant_id
        self._repository = repository
        self._maintenance = maintenance
        self._authorization = authorization
        self._cancel_submission = cancel_submission
        self._captures = captures
        self._release_scopes = release_scopes

    async def purge_expired(
        self, *, principal: Principal, now: datetime, exclusive: SnapshotExclusiveGuard,
        limit: int = 100,
    ) -> EvaluationPurgeResult:
        validate_page_limit(limit)
        if now.tzinfo is None:
            raise ValueError("retention time must be timezone-aware")
        if principal.tenant_id != self._tenant_id:
            raise AIError(ErrorCode.AUTHORIZATION_DENIED)
        await self._authorization.authorize(principal, AuthorizationAction.EVALUATION_PURGE,
            ResourceRef(ResourceKind.EVALUATION, "retention", principal.tenant_id, principal.principal_id))
        repository = self._repository
        records = await repository.list_expired(now=now, limit=limit, owner_principal_id=principal.principal_id)
        purged, blocked, ready = [], [], []
        for record in records:
            await self._authorization.authorize(principal, AuthorizationAction.EVALUATION_PURGE,
                ResourceRef(ResourceKind.EVALUATION, record.experiment_id, principal.tenant_id,
                            record.manifest.principal.principal_id))
            record = await repository.close_reservation_gate(record.experiment_id)
            for intent in record.intents:
                if intent.released:
                    continue
                outcome = await self._cancel_submission(record, intent, principal,
                    f"evaluation-expire:{record.experiment_id}:{intent.slot_id}")
                record = await repository.settle_intent(record.experiment_id, intent.slot_id,
                    confirmed=outcome.admitted, released=outcome.status in _TERMINAL)
            if any(not intent.released for intent in record.intents):
                blocked.append(record.experiment_id)
                continue
            if self._release_scopes is not None:
                await self._release_scopes(record.experiment_id)
            ready.append(record)

        async with exclusive.offline_exclusivity():
            for record in ready:
                capture_keys = []
                for intent in record.intents:
                    if intent.confirmed:
                        continue
                    submission = intent.submission.ref
                    keys = (
                        input_capture_key(submission.namespace, submission.tenant_id, "declaration", submission.graph_id),
                        task_graph_binding_capture_key(submission.namespace, submission.tenant_id,
                                                       submission.graph_id, submission.request_digest),
                    )
                    capture_keys.extend(keys)
                if record.owned_input_captures:
                    capture_keys.extend(await self._captures.task_input_records(record.owned_input_captures, principal=principal))
                objects = await self._maintenance.capture_objects(tuple(capture_keys))
                await repository.purge(record.experiment_id, now=now, objects=objects, capture_keys=tuple(capture_keys))
                purged.append(record.experiment_id)

            datasets = await repository.purge_expired_datasets(now=now, owner_principal_id=principal.principal_id, limit=limit)
            expired_inputs = await self._captures.expire_inputs(principal=principal, now=now, limit=limit)
            pending = await repository.pending_cleanup(owner_principal_id=principal.principal_id, limit=None)
            for receipt in pending:
                await self._authorization.authorize(principal, AuthorizationAction.EVALUATION_PURGE,
                    ResourceRef(ResourceKind.EVALUATION, receipt.evaluation_id, principal.tenant_id, receipt.owner_principal_id))
            expired_objects = await self._maintenance.capture_objects(expired_inputs)
            captures = await self._maintenance.purge_unreferenced_captures(
                (*tuple(key for receipt in pending for key in receipt.capture_keys), *expired_inputs),
                exclusive=_HeldExclusive(), limit=limit)
            completed_captures = (*captures.deleted, *captures.missing)
            deleted_inputs = tuple(key for key in completed_captures if key.startswith("v1/input-capture/task/"))
            if deleted_inputs:
                await self._captures.expire_task_input_records(deleted_inputs, principal=principal, now=now)
            for receipt in pending:
                done = tuple(key for key in receipt.capture_keys if key in completed_captures)
                if done:
                    await repository.acknowledge_cleanup(receipt.evaluation_id, capture_keys=done)
            result = await self._maintenance.purge_unreferenced_objects(
                (*tuple(candidate for receipt in pending for candidate in receipt.objects), *expired_objects),
                exclusive=_HeldExclusive(), limit=limit)
            completed = (*result.deleted, *result.missing)
            if expired_inputs and completed:
                await self._captures.acknowledge_expired_input_objects(expired_inputs, completed, principal=principal)
            for receipt in pending:
                done = tuple(candidate for candidate in receipt.objects if candidate in completed)
                if done:
                    await repository.acknowledge_cleanup(receipt.evaluation_id, done)
            return EvaluationPurgeResult(tuple(purged), tuple(blocked), len(result.deleted),
                len(result.missing), len(result.retained), len(result.blocked), datasets)


__all__ = ["EvaluationRetention", "require_evaluation_content"]
