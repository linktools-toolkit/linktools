#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluation expiry and explicit, restartable retention maintenance."""

from contextlib import AbstractAsyncContextManager, nullcontext
from datetime import datetime
from typing import TYPE_CHECKING

from ..core import AuthorizationAction, AuthorizationPolicy, Principal, ResourceKind, ResourceRef, TaskStatus, validate_page_limit
from ..errors import AIError, ErrorCode
from ..evaluation import EvaluationPurgeResult
from ..task import TaskGraphService
from ..storage import ObjectRef
from ._runtime_identity import task_graph_binding_capture_key
from .state import RuntimeDomain, RuntimeStorage, SnapshotExclusiveGuard, input_capture_key
from .state._evaluation_records import EvaluationRecord

if TYPE_CHECKING:
    from ._input_capture import RuntimeInputCaptures

_TERMINAL = frozenset({TaskStatus.SUCCEEDED, TaskStatus.FAILED, TaskStatus.CANCELLED, TaskStatus.BLOCKED})


def require_evaluation_content(record: EvaluationRecord, *, now: datetime) -> None:
    if record.content_deleted_at is not None or any(
        deadline is not None and deadline <= now
        for deadline in (record.content_expires_at, record.metadata_expires_at)
    ):
        raise AIError(ErrorCode.EVALUATION_EVIDENCE_UNAVAILABLE, safe_details={"reason": "evidence_expired"})


class _HeldExclusive:
    """Object cleanup shares the offline window already held by retention."""

    def offline_exclusivity(self) -> AbstractAsyncContextManager[None]:
        return nullcontext()


class EvaluationRetention:
    def __init__(
        self, storage: RuntimeStorage, authorization: AuthorizationPolicy,
        graph: TaskGraphService, captures: "RuntimeInputCaptures",
    ) -> None:
        self._storage = storage
        self._authorization = authorization
        self._graph = graph
        self._captures = captures

    async def purge_expired(
        self, *, principal: Principal, now: datetime, exclusive: SnapshotExclusiveGuard,
        limit: int = 100,
    ) -> EvaluationPurgeResult:
        validate_page_limit(limit)
        if now.tzinfo is None:
            raise ValueError("retention time must be timezone-aware")
        if principal.tenant_id != self._storage.tenant_id:
            raise AIError(ErrorCode.AUTHORIZATION_DENIED)
        await self._authorization.authorize(principal, AuthorizationAction.EVALUATION_PURGE,
            ResourceRef(ResourceKind.EVALUATION, "retention", principal.tenant_id, principal.principal_id))
        repository = self._storage.evaluation.records
        records = await repository.list_expired(now=now, limit=limit, owner_principal_id=principal.principal_id)
        purged, blocked, ready = [], [], []
        for record in records:
            await self._authorization.authorize(principal, AuthorizationAction.EVALUATION_PURGE,
                ResourceRef(ResourceKind.EVALUATION, record.evaluation_id, principal.tenant_id,
                            record.manifest.principal.principal_id))
            record = await repository.close_reservation_gate(record.evaluation_id)
            for intent in record.intents:
                if intent.released:
                    continue
                outcome = await self._graph.cancel_submission(intent.submission.ref, principal=principal,
                    idempotency_key=f"evaluation-expire:{record.evaluation_id}:{intent.slot_id}")
                record = await repository.settle_intent(record.evaluation_id, intent.slot_id,
                    confirmed=outcome.admitted, released=outcome.status in _TERMINAL)
            if any(not intent.released for intent in record.intents):
                blocked.append(record.evaluation_id)
                continue
            ready.append(record)

        async with exclusive.offline_exclusivity():
            for record in ready:
                objects = []
                for intent in record.intents:
                    if intent.confirmed:
                        continue
                    submission = intent.submission.ref
                    keys = (
                        input_capture_key(submission.namespace, submission.tenant_id, "declaration", submission.graph_id),
                        task_graph_binding_capture_key(submission.namespace, submission.tenant_id,
                                                       submission.graph_id, submission.request_digest),
                    )
                    source = self._storage.object_store(RuntimeDomain.TASK)
                    for key in keys:
                        stat = await source.stat(key)
                        if stat is not None:
                            objects.append((RuntimeDomain.TASK, ObjectRef(source.store_id, key, stat.digest, stat.size)))
                templates = tuple(candidate.graph_template.template_ref for candidate in record.manifest.candidates
                                  if candidate.graph_template is not None and record.manifest.kind == "experiment")
                if templates:
                    references = await self._captures.expire_templates(templates, principal=principal, now=now)
                    objects.extend((RuntimeDomain.TASK, reference) for reference in references)
                await repository.purge(record.evaluation_id, now=now, objects=tuple(objects))
                purged.append(record.evaluation_id)

            datasets = await repository.purge_expired_datasets(now=now, owner_principal_id=principal.principal_id, limit=limit)
            expired_inputs = await self._captures.expire_inputs(principal=principal, now=now, limit=limit)
            pending = await repository.pending_cleanup(owner_principal_id=principal.principal_id, limit=None)
            for receipt in pending:
                await self._authorization.authorize(principal, AuthorizationAction.EVALUATION_PURGE,
                    ResourceRef(ResourceKind.EVALUATION, receipt.evaluation_id, principal.tenant_id, receipt.owner_principal_id))
            result = await self._storage.purge_unreferenced_objects(
                (*tuple(candidate for receipt in pending for candidate in receipt.objects),
                 *((RuntimeDomain.TASK, reference) for reference in expired_inputs)),
                exclusive=_HeldExclusive(), limit=limit)
            completed = (*result.deleted, *result.missing)
            for receipt in pending:
                done = tuple(candidate for candidate in receipt.objects if candidate in completed)
                if done:
                    await repository.acknowledge_cleanup(receipt.evaluation_id, done)
            return EvaluationPurgeResult(tuple(purged), tuple(blocked), len(result.deleted),
                len(result.missing), len(result.retained), len(result.blocked), datasets)


__all__ = ["EvaluationRetention", "require_evaluation_content"]
