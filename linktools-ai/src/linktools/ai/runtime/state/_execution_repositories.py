#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Execution-domain repository implementations."""

from collections.abc import Mapping, Sequence
from dataclasses import replace
from datetime import datetime
from linktools.core import environ
from ...core import ExecutionEventType, ExecutionStatus, IdempotencyStatus, JsonValue, Page, ResourceKind
from ...errors import AIError, ErrorCode
from ...task import TaskBindingSnapshot
from ._contracts import ExecutionCandidate, ExecutionCandidatePage, ExecutionCancelRequestCommit, ExecutionEventAppend, ExecutionEventRecord, ExecutionHistoryHeadRecord, ExecutionHistorySealRecord, ExecutionHistoryState, ExecutionRecord, ExecutionStartClaim, ExecutionStartReservation, ExecutionStartReservationResult, ExecutionStartUnknownCommit, ExecutionTerminalCommit, ExecutionTerminalCommitResult, IdempotencyRecord, IdempotencyTerminalUpdate, ResultRecord
from ._plan import RuntimeDomain
from ._store import FactQuery, RecordQuery, RecordReplacement, StateStore, StateTransaction, StoredFact, StoredRecord, operation_key, stream_digest
from ._repository_common import (
    RepositoryBase as _RepositoryBase,
    ResourceRepository as _ResourceRepository,
    decode_operation as _decode_operation,
    projected_record as _projected_record,
    record_cursor as _record_cursor,
    replace_checked as _replace_checked,
    require_repository_tenant as _require_repository_tenant,
    require_tenant as _require_tenant,
    stored_from_operation as _stored_from_operation,
    validate_page_limit as _validate_page_limit,
)

_logger = environ.get_logger("ai.runtime.state.repositories")


class IdempotencyRepositoryImpl(_ResourceRepository[IdempotencyRecord]):
    def __init__(
        self,
        store: StateStore,
        *,
        namespace: str,
        tenant_id: str,
        domain: RuntimeDomain,
    ) -> None:
        super().__init__(
            store,
            namespace=namespace,
            tenant_id=tenant_id,
            domain=domain,
            kind="idempotency",
            resource_kind=ResourceKind.EXECUTION
            if domain is RuntimeDomain.EXECUTION
            else ResourceKind.EVALUATION,
            value_type=IdempotencyRecord,
        )

    def _require_resource_kind(self, record: IdempotencyRecord) -> None:
        if record.resource_kind is not self._resource_kind:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

    def _identity_key(self, scope: str, key: str) -> list[str]:
        return [scope, key]

    async def reserve(self, record: IdempotencyRecord) -> IdempotencyRecord:
        _require_tenant(record, self._tenant_id)
        self._require_resource_kind(record)
        identity = self._identity_key(record.scope, record.idempotency_key_digest)
        try:
            await self._insert(
                self._stored("idempotency", identity, record, state=record.status.value)
            )
            return record
        except AIError as error:
            if error.code is not ErrorCode.STORAGE_CONFLICT:
                raise
            existing = await super().get(identity, tenant_id=record.tenant_id)
            if existing is not None and _same_idempotency(existing, record):
                return existing
            raise AIError(ErrorCode.STORAGE_CONFLICT) from error

    async def get(
        self,
        scope: str,
        idempotency_key_digest: str,
        *,
        tenant_id: str,
    ) -> IdempotencyRecord | None:  # type: ignore[override]
        return await super().get(
            self._identity_key(scope, idempotency_key_digest), tenant_id=tenant_id
        )

    async def list_by_resource(
        self, resource_kind: ResourceKind, resource_id: str, *, tenant_id: str
    ) -> tuple[IdempotencyRecord, ...]:
        if tenant_id != self._tenant_id:
            return ()
        records = await self._records(
            self._kind,
            scope=self._scope(
                "idempotency", "resource", [resource_kind.value, resource_id]
            ),
        )
        values = [await self._decode(record, self._value_type) for record in records]
        return tuple(values)

    async def compare_and_swap(
        self,
        scope: str,
        idempotency_key_digest: str,
        *,
        tenant_id: str,
        expected_status: IdempotencyStatus,
        next_record: IdempotencyRecord,
    ) -> IdempotencyRecord:  # type: ignore[override]
        if tenant_id != self._tenant_id:
            raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)
        _require_tenant(next_record, self._tenant_id)
        self._require_resource_kind(next_record)
        identity = self._identity_key(scope, idempotency_key_digest)

        async def mutate(transaction: StateTransaction) -> IdempotencyRecord:
            current_record = await transaction.get_record(
                self._key(self._kind, identity)
            )
            if current_record is None:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            current = await self._decode(current_record, IdempotencyRecord)
            if current.status is not expected_status:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            await _replace_checked(
                transaction,
                _projected_record(self, current_record, next_record),
                current_record.storage_version,
            )
            return next_record

        return await self._store.mutate(mutate)


class ExecutionRepositoryImpl(_ResourceRepository[ExecutionRecord]):
    def __init__(self, store: StateStore, *, namespace: str, tenant_id: str) -> None:
        super().__init__(
            store,
            namespace=namespace,
            tenant_id=tenant_id,
            domain=RuntimeDomain.EXECUTION,
            kind="execution",
            resource_kind=ResourceKind.EXECUTION,
            value_type=ExecutionRecord,
        )
        self._idempotency = IdempotencyRepositoryImpl(
            store,
            namespace=namespace,
            tenant_id=tenant_id,
            domain=RuntimeDomain.EXECUTION,
        )

    async def get_many(
        self,
        execution_ids: Sequence[str],
        *,
        tenant_id: str,
    ) -> Mapping[str, ExecutionRecord]:
        if tenant_id != self._tenant_id:
            return {}
        if isinstance(execution_ids, (str, bytes)):
            raise TypeError("execution_ids must be a sequence of strings")
        ordered = tuple(dict.fromkeys(execution_ids))
        if any(not isinstance(execution_id, str) or not execution_id for execution_id in ordered):
            raise ValueError("execution_ids must contain non-empty strings")
        if not ordered:
            return {}
        keys = {
            execution_id: self._key("execution", execution_id)
            for execution_id in ordered
        }

        async def read(
            transaction: StateTransaction,
        ) -> Mapping[str, ExecutionRecord]:
            records = await transaction.get_records(tuple(keys.values()))
            values: dict[str, ExecutionRecord] = {}
            for execution_id, key in keys.items():
                record = records.get(key)
                if record is None:
                    continue
                value = await self._decode(record, ExecutionRecord)
                if (
                    value.execution_id != execution_id
                    or value.tenant_id != self._tenant_id
                ):
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                values[execution_id] = value
            return values

        return await self._store.read(read)

    async def create_with_history_head(
        self, execution: ExecutionRecord
    ) -> ExecutionRecord:
        """Create a recovery execution and its OPEN history head atomically."""
        return await self._store.mutate(
            lambda transaction: self.create_with_history_head_in_transaction(
                transaction,
                execution,
            )
        )

    async def create_with_history_head_in_transaction(
        self,
        transaction: StateTransaction,
        execution: ExecutionRecord,
    ) -> ExecutionRecord:
        _require_tenant(execution, self._tenant_id)
        execution_key = self._key("execution", execution.execution_id)
        head_key = self._key("execution_history_head", execution.execution_id)
        records = await transaction.get_records((execution_key, head_key))
        current = records.get(execution_key)
        head_record = records.get(head_key)
        if current is not None:
            existing = await self._decode(current, ExecutionRecord)
            if existing != execution:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            if head_record is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            await self._decode_history_head_record(
                head_record,
                execution.execution_id,
            )
            return existing
        if head_record is not None:
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        head = ExecutionHistoryHeadRecord(
            execution.execution_id,
            execution.tenant_id,
            ExecutionHistoryState.OPEN,
            0,
            None,
        )
        await transaction.insert_records(
            (
                self._stored(
                    "execution",
                    execution.execution_id,
                    execution,
                    state=execution.status.value,
                ),
                self._stored_history_head(head),
            )
        )
        _logger.debug(
            "execution admitted with history head: execution=%s",
            execution.execution_id,
        )
        return execution

    async def list_by_session(
        self,
        session_id: str,
        *,
        tenant_id: str,
        statuses: frozenset[ExecutionStatus] | None = None,
    ) -> tuple[ExecutionRecord, ...]:
        if tenant_id != self._tenant_id:
            return ()
        records = await self._records(
            "execution",
            scope=self._scope("execution", "session", session_id),
            states=None
            if statuses is None
            else frozenset(status.value for status in statuses),
        )
        return tuple(
            [await self._decode(record, ExecutionRecord) for record in records]
        )

    async def list_children(
        self, execution_id: str, *, tenant_id: str
    ) -> tuple[ExecutionRecord, ...]:
        if tenant_id != self._tenant_id:
            return ()
        records = await self._records(
            "execution",
            parent=self._parent("execution", "execution", execution_id),
        )
        return tuple(
            [await self._decode(record, ExecutionRecord) for record in records]
        )

    async def list_candidates(
        self,
        *,
        tenant_id: str,
        session_id: str | None,
        parent_execution_id: str | None,
        cursor: str | None,
        limit: int,
    ) -> ExecutionCandidatePage:
        if tenant_id != self._tenant_id:
            return ExecutionCandidatePage((), False)
        if (
            isinstance(limit, bool)
            or not isinstance(limit, int)
            or not 1 <= limit <= 1000
        ):
            raise AIError(ErrorCode.PAGE_LIMIT_INVALID)
        scope = (
            None
            if session_id is None or parent_execution_id is not None
            else self._scope("execution", "session", session_id)
        )
        parent = (
            None
            if parent_execution_id is None
            else self._parent("execution", "execution", parent_execution_id)
        )
        records = await self._records(
            "execution",
            scope=scope,
            parent=parent,
            cursor=cursor,
            limit=limit,
        )
        selected = records[:limit]
        candidates: list[ExecutionCandidate] = []
        for record in selected:
            candidates.append(
                ExecutionCandidate(
                    await self._decode(record, ExecutionRecord),
                    _record_cursor(record),
                )
            )
        has_more = False
        if len(records) == limit:
            has_more = await self._has_records(
                "execution",
                scope=scope,
                parent=parent,
                cursor=_record_cursor(records[-1]),
            )
        return ExecutionCandidatePage(
            tuple(candidates),
            has_more,
        )

    async def get_in_transaction(
        self,
        transaction: StateTransaction,
        execution_id: str,
        *,
        tenant_id: str,
    ) -> ExecutionRecord | None:
        _require_repository_tenant(tenant_id, self._tenant_id)
        record = await transaction.get_record(self._key("execution", execution_id))
        return None if record is None else await self._decode(record, ExecutionRecord)

    async def get_start_idempotency(
        self,
        claim: ExecutionStartClaim,
    ) -> IdempotencyRecord | None:
        return await self._idempotency.get(
            claim.scope,
            claim.idempotency_key_digest,
            tenant_id=claim.tenant_id,
        )

    async def get_terminal_idempotency(
        self,
        execution_id: str,
        *,
        tenant_id: str,
    ) -> IdempotencyRecord | None:
        _require_repository_tenant(tenant_id, self._tenant_id)
        values = await self._idempotency.list_by_resource(
            ResourceKind.EXECUTION,
            execution_id,
            tenant_id=tenant_id,
        )
        if len(values) > 1:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return values[0] if values else None

    async def reserve_start(
        self, reservation: ExecutionStartReservation
    ) -> ExecutionStartReservationResult:
        _require_tenant(reservation.execution, self._tenant_id)
        _require_tenant(reservation.idempotency, self._tenant_id)
        self._idempotency._require_resource_kind(reservation.idempotency)

        async def mutate(
            transaction: StateTransaction,
        ) -> ExecutionStartReservationResult:
            identity = self._idempotency._identity_key(
                reservation.idempotency.scope,
                reservation.idempotency.idempotency_key_digest,
            )
            id_key = self._idempotency._key("idempotency", identity)
            execution_key = self._key("execution", reservation.execution.execution_id)
            head_key = self._key(
                "execution_history_head",
                reservation.execution.execution_id,
            )
            records = await transaction.get_records((id_key, execution_key, head_key))
            existing_id = records.get(id_key)
            if existing_id is not None:
                existing_idempotency = await self._idempotency._decode(
                    existing_id, IdempotencyRecord
                )
                if not _same_idempotency_identity(
                    existing_idempotency, reservation.idempotency
                ):
                    raise AIError(ErrorCode.IDEMPOTENCY_CONFLICT)
                winner_key = self._key("execution", existing_idempotency.resource_id)
                if winner_key == execution_key:
                    winner_records = records
                    winner_head_key = head_key
                else:
                    winner_head_key = self._key(
                        "execution_history_head",
                        existing_idempotency.resource_id,
                    )
                    winner_records = await transaction.get_records(
                        (winner_key, winner_head_key)
                    )
                winner_record = winner_records.get(winner_key)
                if winner_record is None:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                existing_value = await self._decode(winner_record, ExecutionRecord)
                winner_head = winner_records.get(winner_head_key)
                if winner_head is None:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                await self._decode_history_head_record(
                    winner_head,
                    existing_value.execution_id,
                )
                _logger.debug(
                    "execution start reservation replayed: execution=%s",
                    existing_value.execution_id,
                )
                return ExecutionStartReservationResult(
                    existing_value, existing_idempotency, False
                )
            if execution_key in records or head_key in records:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            head = ExecutionHistoryHeadRecord(
                reservation.execution.execution_id,
                reservation.execution.tenant_id,
                ExecutionHistoryState.OPEN,
                0,
                None,
            )
            await transaction.insert_records(
                (
                    self._stored(
                        "execution",
                        reservation.execution.execution_id,
                        reservation.execution,
                        state=reservation.execution.status.value,
                    ),
                    self._idempotency._stored(
                        "idempotency",
                        identity,
                        reservation.idempotency,
                        state=reservation.idempotency.status.value,
                    ),
                    self._stored_history_head(head),
                )
            )
            _logger.debug(
                "execution start reserved: execution=%s",
                reservation.execution.execution_id,
            )
            return ExecutionStartReservationResult(
                reservation.execution, reservation.idempotency, True
            )

        return await self._store.mutate(mutate)

    async def claim_start(self, claim: ExecutionStartClaim) -> ExecutionRecord:
        return await self._store.mutate(
            lambda transaction: self.claim_start_in_transaction(transaction, claim)
        )

    async def claim_start_in_transaction(
        self,
        transaction: StateTransaction,
        claim: ExecutionStartClaim,
    ) -> ExecutionRecord:
        _require_repository_tenant(claim.tenant_id, self._tenant_id)
        execution_key = self._key("execution", claim.execution_id)
        identity = self._idempotency._identity_key(
            claim.scope,
            claim.idempotency_key_digest,
        )
        idempotency_key = self._idempotency._key("idempotency", identity)
        stored = await transaction.get_records((execution_key, idempotency_key))
        execution_record = stored.get(execution_key)
        if execution_record is None:
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        current = await self._decode(execution_record, ExecutionRecord)
        if (
            current.revision != claim.expected_revision
            or current.event_sequence != claim.expected_event_sequence
            or current.status is not ExecutionStatus.PENDING_START
        ):
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        idempotency_record = stored.get(idempotency_key)
        idempotency_created = idempotency_record is None
        if idempotency_record is None:
            idempotency = IdempotencyRecord(
                tenant_id=claim.tenant_id,
                scope=claim.scope,
                idempotency_key_digest=claim.idempotency_key_digest,
                request_digest=claim.request_digest,
                resource_kind=ResourceKind.EXECUTION,
                resource_id=claim.execution_id,
                status=IdempotencyStatus.STARTED,
                result_digest=None,
                error_code=None,
                created_at=claim.started_at,
                updated_at=claim.started_at,
            )
            idempotency_record = self._idempotency._stored(
                "idempotency",
                identity,
                idempotency,
                state=idempotency.status.value,
            )
        if idempotency_record is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if not idempotency_created:
            idempotency = await self._idempotency._decode(
                idempotency_record,
                IdempotencyRecord,
            )
            if (
                not _same_idempotency(
                    idempotency,
                    IdempotencyRecord(
                        tenant_id=claim.tenant_id,
                        scope=claim.scope,
                        idempotency_key_digest=claim.idempotency_key_digest,
                        request_digest=claim.request_digest,
                        resource_kind=ResourceKind.EXECUTION,
                        resource_id=claim.execution_id,
                        status=idempotency.status,
                        result_digest=idempotency.result_digest,
                        error_code=idempotency.error_code,
                        created_at=idempotency.created_at,
                        updated_at=idempotency.updated_at,
                    ),
                )
                or idempotency.status is not IdempotencyStatus.RESERVED
            ):
                raise AIError(ErrorCode.STORAGE_CONFLICT)
        now = claim.started_at
        next_execution = replace(
            current,
            status=ExecutionStatus.STARTED,
            revision=current.revision + 1,
            event_sequence=current.event_sequence + 1,
            updated_at=now,
        )
        execution_replacement = RecordReplacement(
            _projected_record(self, execution_record, next_execution),
            execution_record.storage_version,
        )
        if idempotency_created:
            await transaction.insert_records((idempotency_record,))
            await _replace_checked(
                transaction,
                execution_replacement.record,
                execution_replacement.expected_storage_version,
            )
        else:
            replacements = [execution_replacement]
            next_idempotency = replace(
                idempotency,
                status=IdempotencyStatus.STARTED,
                updated_at=now,
            )
            replacements.append(
                RecordReplacement(
                    _projected_record(
                        self._idempotency,
                        idempotency_record,
                        next_idempotency,
                    ),
                    idempotency_record.storage_version,
                )
            )
            await transaction.replace_records(tuple(replacements))
        stream = stream_digest(
            self._namespace,
            self._tenant_id,
            self._domain.value,
            "execution",
            claim.execution_id,
        )
        await transaction.insert_facts(
            (
                StoredFact(
                    stream,
                    next_execution.event_sequence,
                    execution_key,
                    ExecutionEventType.EXECUTION_STARTED.value,
                    None,
                    None,
                    {},
                ),
            )
        )
        _logger.debug(
            "execution start claimed: execution=%s idempotency_created=%s",
            claim.execution_id,
            idempotency_created,
        )
        return next_execution

    async def claim_next_agent_run(
        self,
        execution_id: str,
        *,
        tenant_id: str,
        expected_revision: int,
        expected_agent_run_sequence: int,
    ) -> ExecutionRecord:
        _require_repository_tenant(tenant_id, self._tenant_id)

        async def mutate(transaction: StateTransaction) -> ExecutionRecord:
            return await self.claim_next_agent_run_in_transaction(
                transaction,
                execution_id,
                tenant_id=tenant_id,
                expected_revision=expected_revision,
                expected_agent_run_sequence=expected_agent_run_sequence,
            )

        return await self._store.mutate(mutate)

    async def claim_next_agent_run_in_transaction(
        self,
        transaction: StateTransaction,
        execution_id: str,
        *,
        tenant_id: str,
        expected_revision: int,
        expected_agent_run_sequence: int,
    ) -> ExecutionRecord:
        _require_repository_tenant(tenant_id, self._tenant_id)
        record = await transaction.get_record(self._key("execution", execution_id))
        if record is None:
            raise AIError(ErrorCode.STORAGE_NOT_FOUND)
        current = await self._decode(record, ExecutionRecord)
        if (
            current.revision != expected_revision
            or current.agent_run_sequence != expected_agent_run_sequence
        ):
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        next_value = replace(
            current,
            agent_run_sequence=current.agent_run_sequence + 1,
            revision=current.revision + 1,
            updated_at=await transaction.now(),
        )
        await _replace_checked(
            transaction,
            _projected_record(self, record, next_value),
            record.storage_version,
        )
        return next_value

    async def enter_deferred_wait_in_transaction(
        self,
        transaction: StateTransaction,
        execution_id: str,
        *,
        tenant_id: str,
        expected_revision: int,
        expected_event_sequence: int,
        expected_agent_run_sequence: int,
        audit_events: Sequence[ExecutionEventAppend] = (),
        deferred_events: Sequence[ExecutionEventAppend],
        occurred_at: datetime,
    ) -> ExecutionRecord:
        if not deferred_events or any(
            event.event_type
            not in {
                ExecutionEventType.APPROVAL_REQUESTED,
                ExecutionEventType.EXTERNAL_REQUESTED,
            }
            for event in deferred_events
        ):
            raise ValueError("deferred wait requires deferred events")
        if occurred_at.tzinfo is None:
            raise ValueError("deferred wait timestamp must be timezone-aware")
        current = await self.get_in_transaction(
            transaction,
            execution_id,
            tenant_id=tenant_id,
        )
        if (
            current is None
            or current.status is not ExecutionStatus.STARTED
            or current.revision != expected_revision
            or current.event_sequence != expected_event_sequence
            or current.agent_run_sequence != expected_agent_run_sequence
        ):
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        last = deferred_events[-1]
        updated = await self._transition_execution(
            execution_id,
            tenant_id=tenant_id,
            expected_revision=expected_revision,
            expected_event_sequence=expected_event_sequence,
            expected_status=ExecutionStatus.STARTED,
            next_status=ExecutionStatus.WAITING_DEFERRED,
            pending_events=(*audit_events, *deferred_events[:-1]),
            event_type=last.event_type,
            payload=last.payload,
            updated_at=occurred_at,
            transaction=transaction,
        )
        if updated.agent_run_sequence != expected_agent_run_sequence:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return updated

    async def claim_deferred_resume_in_transaction(
        self,
        transaction: StateTransaction,
        execution_id: str,
        *,
        tenant_id: str,
        expected_revision: int,
        expected_event_sequence: int,
        expected_agent_run_sequence: int,
    ) -> ExecutionRecord:
        _require_repository_tenant(tenant_id, self._tenant_id)
        record = await transaction.get_record(self._key("execution", execution_id))
        if record is None:
            raise AIError(ErrorCode.STORAGE_NOT_FOUND)
        current = await self._decode(record, ExecutionRecord)
        if (
            current.status is not ExecutionStatus.WAITING_DEFERRED
            or current.revision != expected_revision
            or current.event_sequence != expected_event_sequence
            or current.agent_run_sequence != expected_agent_run_sequence
        ):
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        next_value = replace(
            current,
            status=ExecutionStatus.STARTED,
            agent_run_sequence=current.agent_run_sequence + 1,
            revision=current.revision + 1,
            updated_at=await transaction.now(),
        )
        await _replace_checked(
            transaction,
            _projected_record(self, record, next_value),
            record.storage_version,
        )
        return next_value

    async def transition_task_execution(
        self,
        execution_id: str,
        *,
        tenant_id: str,
        expected_revision: int,
        expected_event_sequence: int,
        expected_status: ExecutionStatus,
        next_status: ExecutionStatus,
        task_attempt: int,
        task_deadline_at: datetime | None,
        task_next_attempt_at: datetime | None,
        error_code: str | None,
        safe_error_details: Mapping[str, JsonValue],
        event_type: str,
        payload: Mapping[str, JsonValue],
        occurred_at: datetime,
    ) -> ExecutionRecord:
        if (
            isinstance(task_attempt, bool)
            or not isinstance(task_attempt, int)
            or task_attempt < 0
            or occurred_at.tzinfo is None
        ):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        return await self._transition_execution(
            execution_id,
            tenant_id=tenant_id,
            expected_revision=expected_revision,
            expected_event_sequence=expected_event_sequence,
            expected_status=expected_status,
            next_status=next_status,
            event_type=event_type,
            payload=payload,
            updated_at=occurred_at,
            task_state=(
                task_attempt,
                task_deadline_at,
                task_next_attempt_at,
                error_code,
                safe_error_details,
            ),
        )

    async def mark_start_unknown(
        self, commit: ExecutionStartUnknownCommit
    ) -> ExecutionRecord:
        return await self._transition_execution(
            commit.execution_id,
            tenant_id=commit.tenant_id,
            expected_revision=commit.expected_revision,
            expected_event_sequence=commit.expected_event_sequence,
            next_status=ExecutionStatus.START_UNKNOWN,
            event_type=ExecutionEventType.EXECUTION_START_UNKNOWN,
            payload={},
            updated_at=commit.occurred_at,
        )

    async def request_cancel(
        self,
        commit: ExecutionCancelRequestCommit,
        *,
        pending_events: Sequence[ExecutionEventAppend] = (),
    ) -> ExecutionRecord:
        return await self._store.mutate(
            lambda transaction: self.request_cancel_in_transaction(
                transaction,
                commit,
                expected_status=None,
                pending_events=pending_events,
            )
        )

    async def request_cancel_in_transaction(
        self,
        transaction: StateTransaction,
        commit: ExecutionCancelRequestCommit,
        *,
        expected_status: ExecutionStatus | None = None,
        pending_events: Sequence[ExecutionEventAppend] = (),
    ) -> ExecutionRecord:
        return await self._transition_execution(
            commit.execution_id,
            tenant_id=commit.tenant_id,
            expected_revision=commit.expected_revision,
            expected_event_sequence=commit.expected_event_sequence,
            expected_status=expected_status,
            next_status=ExecutionStatus.CANCELLING,
            event_type=ExecutionEventType.CANCEL_REQUESTED,
            payload={"operation_id": commit.operation_id},
            updated_at=commit.requested_at,
            pending_events=pending_events,
            transaction=transaction,
        )

    async def advance_event_sequence(
        self, execution_id: str, *, tenant_id: str, expected_sequence: int
    ) -> ExecutionRecord:
        _require_repository_tenant(tenant_id, self._tenant_id)

        async def mutate(transaction: StateTransaction) -> ExecutionRecord:
            record = await transaction.get_record(self._key("execution", execution_id))
            if record is None:
                raise AIError(ErrorCode.STORAGE_NOT_FOUND)
            current = await self._decode(record, ExecutionRecord)
            if current.event_sequence != expected_sequence:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            next_value = replace(
                current,
                event_sequence=current.event_sequence + 1,
                revision=current.revision + 1,
                updated_at=await transaction.now(),
            )
            await _replace_checked(
                transaction,
                _projected_record(self, record, next_value),
                record.storage_version,
            )
            return next_value

        return await self._store.mutate(mutate)

    async def _transition_execution(
        self,
        execution_id: str,
        *,
        tenant_id: str,
        expected_revision: int,
        expected_event_sequence: int,
        next_status: ExecutionStatus,
        expected_status: ExecutionStatus | None = None,
        event_type: str,
        payload: Mapping[str, object],
        updated_at: datetime,
        pending_events: Sequence[ExecutionEventAppend] = (),
        task_state: "tuple[int, datetime | None, datetime | None, str | None, Mapping[str, object]] | None" = None,
        transaction: StateTransaction | None = None,
    ) -> ExecutionRecord:
        _require_repository_tenant(tenant_id, self._tenant_id)
        if any(
            not isinstance(event.event_type, str)
            for event in pending_events
        ):
            raise TypeError("pending execution events require a string event type")
        key = self._key("execution", execution_id)
        stream = stream_digest(
            self._namespace,
            self._tenant_id,
            self._domain.value,
            "execution",
            execution_id,
        )

        async def mutate(transaction: StateTransaction) -> ExecutionRecord:
            stored = await transaction.get_record(key)
            if stored is None:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            stored_value = await self._decode(stored, ExecutionRecord)
            if (
                stored_value.revision != expected_revision
                or stored_value.event_sequence != expected_event_sequence
                or expected_status is not None
                and stored_value.status is not expected_status
            ):
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            event_count = len(pending_events) + 1
            task_updates: dict[str, object] = {}
            if task_state is not None:
                if not isinstance(stored_value.binding, TaskBindingSnapshot):
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                (
                    task_attempt,
                    task_deadline_at,
                    task_next_attempt_at,
                    error_code,
                    safe_error_details,
                ) = task_state
                task_updates = {
                    "task_attempt": task_attempt,
                    "task_deadline_at": task_deadline_at,
                    "task_next_attempt_at": task_next_attempt_at,
                    "error_code": error_code,
                    "safe_error_details": dict(safe_error_details),
                }
            next_value = replace(
                stored_value,
                status=next_status,
                revision=stored_value.revision + event_count,
                event_sequence=stored_value.event_sequence + event_count,
                updated_at=updated_at,
                **task_updates,
            )
            candidate = _projected_record(self, stored, next_value)
            await _replace_checked(transaction, candidate, stored.storage_version)
            first_sequence = stored_value.event_sequence + 1
            facts = [
                StoredFact(
                    stream,
                    first_sequence + index,
                    key,
                    event.event_type,
                    None,
                    None,
                    event.payload,
                )
                for index, event in enumerate(pending_events)
            ]
            facts.append(
                StoredFact(
                    stream,
                    next_value.event_sequence,
                    key,
                    str(event_type),
                    None,
                    None,
                    payload,
                )
            )
            await transaction.insert_facts(tuple(facts))
            return next_value

        if transaction is not None:
            return await mutate(transaction)
        return await self._store.mutate(mutate)

    async def terminal_idempotency_in_transaction(
        self,
        transaction: StateTransaction,
        commit: ExecutionTerminalCommit,
    ) -> IdempotencyTerminalUpdate | None:
        """Build the terminal idempotency update from the active transaction."""
        scope = self._idempotency._scope(
            "idempotency",
            "resource",
            [ResourceKind.EXECUTION.value, commit.execution.execution_id],
        )
        records = await transaction.list_records(
            RecordQuery(scope_digest=scope, kind="idempotency")
        )
        if len(records) > 1:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if not records:
            return None
        identity = await self._idempotency._decode(records[0], IdempotencyRecord)
        next_status = (
            IdempotencyStatus.COMPLETED
            if commit.execution.status is ExecutionStatus.SUCCEEDED
            else IdempotencyStatus.CANCELLED
            if commit.execution.status is ExecutionStatus.CANCELLED
            else IdempotencyStatus.FAILED
        )
        return IdempotencyTerminalUpdate(
            identity.scope,
            identity.idempotency_key_digest,
            identity.status,
            next_status,
            identity.request_digest,
            None if commit.result.output is None else commit.result.output.digest,
            commit.execution.error_code,
        )

    async def commit_terminal(
        self,
        commit: ExecutionTerminalCommit,
        *,
        pending_events: Sequence[ExecutionEventAppend] = (),
    ) -> ExecutionTerminalCommitResult:
        return await self._commit_terminal(commit, pending_events=pending_events)

    async def _commit_terminal(
        self,
        commit: ExecutionTerminalCommit,
        *,
        pending_events: Sequence[ExecutionEventAppend] = (),
        transaction: StateTransaction | None = None,
    ) -> ExecutionTerminalCommitResult:
        _require_repository_tenant(commit.execution.tenant_id, self._tenant_id)
        key = self._key("execution", commit.execution.execution_id)
        stream = stream_digest(
            self._namespace,
            self._tenant_id,
            self._domain.value,
            "execution",
            commit.execution.execution_id,
        )

        async def mutate(
            transaction: StateTransaction,
        ) -> ExecutionTerminalCommitResult:
            if any(
                not isinstance(event.event_type, str)
                for event in pending_events
            ):
                raise TypeError("pending execution events require a string event type")
            record_keys = [key]
            id_key = None
            if commit.idempotency is not None:
                identity = self._idempotency._identity_key(
                    commit.idempotency.scope,
                    commit.idempotency.idempotency_key_digest,
                )
                id_key = self._idempotency._key("idempotency", identity)
                record_keys.append(id_key)
            stored_records = await transaction.get_records(record_keys)
            stored = stored_records.get(key)
            if stored is None:
                raise AIError(ErrorCode.EXECUTION_RESULT_CONFLICT)
            stored_value = await self._decode(stored, ExecutionRecord)
            if (
                stored_value.revision != commit.expected_revision
                or stored_value.event_sequence != commit.expected_event_sequence
            ):
                raise AIError(ErrorCode.EXECUTION_RESULT_CONFLICT)
            id_record = None
            id_value = None
            if commit.idempotency is not None:
                assert id_key is not None
                id_record = stored_records.get(id_key)
                if id_record is None:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                id_value = await self._idempotency._decode(
                    id_record,
                    IdempotencyRecord,
                )
                if id_value.status is not commit.idempotency.expected_status:
                    raise AIError(ErrorCode.EXECUTION_RESULT_CONFLICT)
            operation_record = None
            current_operation = None
            if commit.operation is not None:
                operation_key_value = operation_key(
                    self._namespace,
                    self._tenant_id,
                    self._domain.value,
                    commit.operation.operation_id,
                )
                operation_record = await transaction.get_operation(operation_key_value)
                if operation_record is None:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                current_operation = _decode_operation(operation_record)
                if current_operation.status is not commit.operation.expected_status:
                    raise AIError(ErrorCode.EXECUTION_RESULT_CONFLICT)
            now = (
                await transaction.now()
                if commit.idempotency is not None or commit.operation is not None
                else None
            )
            next_execution = replace(
                commit.execution,
                revision=stored_value.revision + len(pending_events) + 1,
                event_sequence=stored_value.event_sequence + len(pending_events) + 1,
                result=commit.result,
            )
            replacements = [
                RecordReplacement(
                    _projected_record(self, stored, next_execution),
                    stored.storage_version,
                )
            ]
            if commit.idempotency is not None:
                assert id_record is not None
                assert id_value is not None
                assert now is not None
                next_id = replace(
                    id_value,
                    status=commit.idempotency.next_status,
                    request_digest=commit.idempotency.request_digest,
                    result_digest=commit.idempotency.result_digest,
                    error_code=commit.idempotency.error_code,
                    updated_at=now,
                )
                replacements.append(
                    RecordReplacement(
                        _projected_record(
                            self._idempotency,
                            id_record,
                            next_id,
                        ),
                        id_record.storage_version,
                    )
                )
            first_sequence = stored_value.event_sequence + 1
            facts = [
                StoredFact(
                    stream,
                    first_sequence + index,
                    key,
                    event.event_type,
                    None,
                    None,
                    event.payload,
                )
                for index, event in enumerate(pending_events)
            ]
            facts.append(
                StoredFact(
                    stream,
                    next_execution.event_sequence,
                    key,
                    str(commit.terminal_event_type),
                    None,
                    None,
                    commit.terminal_event_payload,
                )
            )
            next_operation = None
            if commit.operation is not None:
                assert current_operation is not None
                assert operation_record is not None
                assert now is not None
                next_operation = replace(
                    current_operation,
                    status=commit.operation.next_status,
                    result_ref=commit.operation.result_ref,
                    result_digest=commit.operation.result_digest,
                    error_code=commit.operation.error_code,
                    updated_at=now,
                )
            await transaction.replace_records(tuple(replacements))
            await transaction.insert_facts(tuple(facts))
            if commit.operation is not None:
                assert next_operation is not None
                assert operation_record is not None
                if not await transaction.replace_operation(
                    _stored_from_operation(next_operation, operation_record),
                    expected_state=commit.operation.expected_status.value,
                ):
                    raise AIError(ErrorCode.EXECUTION_RESULT_CONFLICT)
            _logger.debug(
                "execution terminal boundary committed: execution=%s pending_events=%s "
                "idempotency=%s operation=%s",
                commit.execution.execution_id,
                len(pending_events),
                commit.idempotency is not None,
                commit.operation is not None,
            )
            return ExecutionTerminalCommitResult(next_execution, commit.result)

        if transaction is not None:
            return await mutate(transaction)
        return await self._store.mutate(mutate)

    async def commit_terminal_in_transaction(
        self,
        transaction: StateTransaction,
        commit: ExecutionTerminalCommit,
        *,
        pending_events: Sequence[ExecutionEventAppend] = (),
    ) -> ExecutionTerminalCommitResult:
        return await self._commit_terminal(
            commit,
            pending_events=pending_events,
            transaction=transaction,
        )

    async def get_result(
        self, execution_id: str, *, tenant_id: str
    ) -> ResultRecord | None:
        execution = await self.get(execution_id, tenant_id=tenant_id)
        return None if execution is None else execution.result

    async def acquire_dependency_hold(
        self,
        execution_id: str,
        *,
        tenant_id: str,
        hold_id: str,
    ) -> ExecutionRecord:
        _require_repository_tenant(tenant_id, self._tenant_id)
        if not isinstance(hold_id, str) or not hold_id.strip():
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        key = self._key("execution", execution_id)

        async def mutate(transaction: StateTransaction) -> ExecutionRecord:
            stored = await transaction.get_record(key)
            if stored is None:
                raise AIError(ErrorCode.STORAGE_NOT_FOUND)
            current = await self._decode(stored, ExecutionRecord)
            if current.retention_closed:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            if hold_id in current.dependency_hold_ids:
                return current
            next_value = replace(
                current,
                dependency_hold_ids=tuple(
                    sorted((*current.dependency_hold_ids, hold_id))
                ),
                revision=current.revision + 1,
                updated_at=await transaction.now(),
            )
            await _replace_checked(
                transaction,
                _projected_record(self, stored, next_value),
                stored.storage_version,
            )
            return next_value

        return await self._store.mutate(mutate)

    async def release_dependency_hold(
        self,
        execution_id: str,
        *,
        tenant_id: str,
        hold_id: str,
    ) -> ExecutionRecord:
        _require_repository_tenant(tenant_id, self._tenant_id)
        if not isinstance(hold_id, str) or not hold_id.strip():
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        key = self._key("execution", execution_id)

        async def mutate(transaction: StateTransaction) -> ExecutionRecord:
            stored = await transaction.get_record(key)
            if stored is None:
                raise AIError(ErrorCode.STORAGE_NOT_FOUND)
            current = await self._decode(stored, ExecutionRecord)
            if hold_id not in current.dependency_hold_ids:
                return current
            next_value = replace(
                current,
                dependency_hold_ids=tuple(
                    value
                    for value in current.dependency_hold_ids
                    if value != hold_id
                ),
                revision=current.revision + 1,
                updated_at=await transaction.now(),
            )
            await _replace_checked(
                transaction,
                _projected_record(self, stored, next_value),
                stored.storage_version,
            )
            return next_value

        return await self._store.mutate(mutate)

    async def close_retention(
        self,
        execution_id: str,
        *,
        tenant_id: str,
    ) -> bool:
        _require_repository_tenant(tenant_id, self._tenant_id)
        key = self._key("execution", execution_id)

        async def mutate(transaction: StateTransaction) -> bool:
            stored = await transaction.get_record(key)
            if stored is None:
                return True
            current = await self._decode(stored, ExecutionRecord)
            if current.status not in {
                ExecutionStatus.SUCCEEDED,
                ExecutionStatus.FAILED,
                ExecutionStatus.CANCELLED,
            }:
                return False
            if current.dependency_hold_ids:
                return False
            if current.retention_closed:
                return True
            next_value = replace(
                current,
                retention_closed=True,
                revision=current.revision + 1,
                updated_at=await transaction.now(),
            )
            await _replace_checked(
                transaction,
                _projected_record(self, stored, next_value),
                stored.storage_version,
            )
            return True

        return await self._store.mutate(mutate)

    async def get_history_seal(
        self,
        execution_id: str,
        *,
        tenant_id: str,
    ) -> ExecutionHistorySealRecord | None:
        _require_repository_tenant(tenant_id, self._tenant_id)

        async def read(
            transaction: StateTransaction,
        ) -> ExecutionHistorySealRecord | None:
            return await self._get_history_seal_in_transaction(
                transaction, execution_id
            )

        return await self._store.read(read)

    async def _get_history_seal_in_transaction(
        self,
        transaction: StateTransaction,
        execution_id: str,
    ) -> ExecutionHistorySealRecord | None:
        record = await transaction.get_record(
            self._key("execution_history_seal", execution_id)
        )
        if record is None:
            return None
        if record.kind != "execution_history_seal":
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        value = await self._decode(record, ExecutionHistorySealRecord)
        if value.execution_id != execution_id or value.tenant_id != self._tenant_id:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return value

    async def get_history_head(
        self,
        execution_id: str,
        *,
        tenant_id: str,
    ) -> ExecutionHistoryHeadRecord | None:
        _require_repository_tenant(tenant_id, self._tenant_id)
        return await self._store.read(
            lambda transaction: self._get_history_head_in_transaction(
                transaction,
                execution_id,
            )
        )

    async def _get_history_head_in_transaction(
        self,
        transaction: StateTransaction,
        execution_id: str,
    ) -> ExecutionHistoryHeadRecord | None:
        record = await transaction.get_record(
            self._key("execution_history_head", execution_id)
        )
        if record is None:
            return None
        return await self._decode_history_head_record(record, execution_id)

    async def _decode_history_head_record(
        self,
        record: StoredRecord,
        execution_id: str,
    ) -> ExecutionHistoryHeadRecord:
        if record.kind != "execution_history_head":
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        value = await self._decode(record, ExecutionHistoryHeadRecord)
        if value.execution_id != execution_id or value.tenant_id != self._tenant_id:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return value

    async def require_open_history_head_in_transaction(
        self,
        transaction: StateTransaction,
        execution_id: str,
        *,
        expected_revision: "int | None" = None,
    ) -> tuple[ExecutionHistoryHeadRecord, StoredRecord]:
        """Read and guard the OPEN history head for one execution-domain mutation."""
        key = self._key("execution_history_head", execution_id)
        record = await transaction.get_record(key)
        if record is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if record.kind != "execution_history_head":
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        head = await self._decode(record, ExecutionHistoryHeadRecord)
        if head.execution_id != execution_id or head.tenant_id != self._tenant_id:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if head.state is not ExecutionHistoryState.OPEN:
            _logger.info(
                "execution history head is sealed: execution=%s revision=%s",
                execution_id,
                head.revision,
            )
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        if expected_revision is not None and head.revision != expected_revision:
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        guarded = await transaction.guard_record(
            key,
            expected_storage_version=record.storage_version,
        )
        if guarded is None:
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        return head, guarded

    async def replace_history_head_in_transaction(
        self,
        transaction: StateTransaction,
        current_record: StoredRecord,
        next_head: ExecutionHistoryHeadRecord,
    ) -> ExecutionHistoryHeadRecord:
        key = self._key("execution_history_head", next_head.execution_id)
        if key != current_record.key_digest:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        upgraded = replace(
            self._stored(
                "execution_history_head",
                next_head.execution_id,
                next_head,
                state=next_head.state.value,
            ),
            storage_version=current_record.storage_version + 1,
        )
        if not await transaction.replace_record(
            upgraded,
            expected_storage_version=current_record.storage_version,
        ):
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        return next_head

    async def insert_history_head_in_transaction(
        self,
        transaction: StateTransaction,
        head: ExecutionHistoryHeadRecord,
    ) -> ExecutionHistoryHeadRecord:
        _require_repository_tenant(head.tenant_id, self._tenant_id)
        key = self._key("execution_history_head", head.execution_id)
        if await transaction.get_record(key) is not None:
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        await transaction.insert_record(self._stored_history_head(head))
        return head

    def _stored_history_head(self, head: ExecutionHistoryHeadRecord) -> StoredRecord:
        return self._stored(
            "execution_history_head",
            head.execution_id,
            head,
            state=head.state.value,
        )

    async def put_history_seal_in_transaction(
        self,
        transaction: StateTransaction,
        seal: ExecutionHistorySealRecord,
    ) -> ExecutionHistorySealRecord:
        _require_repository_tenant(seal.tenant_id, self._tenant_id)
        key = self._key("execution_history_seal", seal.execution_id)
        current = await transaction.get_record(key)
        if current is not None:
            existing = await self._decode(current, ExecutionHistorySealRecord)
            if existing != seal:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            return existing
        await transaction.insert_record(
            self._stored(
                "execution_history_seal",
                seal.execution_id,
                seal,
            )
        )
        _logger.info(
            "execution history seal persisted: execution=%s runs=%s",
            seal.execution_id,
            len(seal.run_heads),
        )
        return seal


class EventRepositoryImpl(_RepositoryBase):
    def __init__(self, store: StateStore, *, namespace: str, tenant_id: str) -> None:
        super().__init__(
            store,
            namespace=namespace,
            tenant_id=tenant_id,
            domain=RuntimeDomain.EXECUTION,
        )

    async def append_many(
        self,
        execution_id: str,
        *,
        tenant_id: str,
        events: Sequence[ExecutionEventAppend],
        expected_sequence: int | None = None,
    ) -> tuple[ExecutionEventRecord, ...]:
        _require_repository_tenant(tenant_id, self._tenant_id)
        if not events:
            return ()
        result = await self._store.mutate(
            lambda transaction: self.append_many_in_transaction(
                transaction,
                execution_id,
                tenant_id=tenant_id,
                events=events,
                expected_sequence=expected_sequence,
            )
        )
        _logger.debug(
            "execution events appended: execution=%s count=%s last_sequence=%s",
            execution_id,
            len(events),
            result[-1].sequence,
        )
        return result

    async def append_many_in_transaction(
        self,
        transaction: StateTransaction,
        execution_id: str,
        *,
        tenant_id: str,
        events: Sequence[ExecutionEventAppend],
        expected_sequence: int | None = None,
    ) -> tuple[ExecutionEventRecord, ...]:
        _require_repository_tenant(tenant_id, self._tenant_id)
        if not events:
            return ()
        if any(
            not isinstance(event.event_type, str) for event in events
        ):
            raise TypeError("event repository requires a string event type")
        if any(not isinstance(event.payload, Mapping) for event in events):
            raise TypeError("event payload must be a mapping")
        key = self._key("execution", execution_id)
        stream = stream_digest(
            self._namespace,
            self._tenant_id,
            self._domain.value,
            "execution",
            execution_id,
        )
        current = await transaction.get_record(key)
        if current is None:
            raise AIError(ErrorCode.STORAGE_NOT_FOUND)
        execution = await self._decode(current, ExecutionRecord)
        if (
            expected_sequence is not None
            and execution.event_sequence != expected_sequence
        ):
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        now = await transaction.now()
        first_sequence = execution.event_sequence + 1
        last_sequence = execution.event_sequence + len(events)
        next_execution = replace(
            execution,
            event_sequence=last_sequence,
            revision=execution.revision + len(events),
            updated_at=now,
        )
        await _replace_checked(
            transaction,
            _projected_record(self, current, next_execution),
            current.storage_version,
        )
        facts = tuple(
            StoredFact(
                stream,
                first_sequence + index,
                key,
                event.event_type,
                None,
                None,
                event.payload,
            )
            for index, event in enumerate(events)
        )
        await transaction.insert_facts(facts)
        return tuple(
            ExecutionEventRecord(
                execution_id,
                tenant_id,
                first_sequence + index,
                event.event_type,
                event.payload,
            )
            for index, event in enumerate(events)
        )

    async def append_next(
        self,
        execution_id: str,
        *,
        tenant_id: str,
        event_type: str,
        payload: object,
    ) -> ExecutionEventRecord:
        values = await self.append_many(
            execution_id,
            tenant_id=tenant_id,
            events=(ExecutionEventAppend(event_type, _event_payload(payload)),),
            expected_sequence=None,
        )
        return values[0]

    async def append_expected(
        self,
        execution_id: str,
        *,
        tenant_id: str,
        expected_sequence: int,
        event_type: str,
        payload: object,
    ) -> ExecutionEventRecord:
        values = await self.append_many(
            execution_id,
            tenant_id=tenant_id,
            events=(ExecutionEventAppend(event_type, _event_payload(payload)),),
            expected_sequence=expected_sequence,
        )
        return values[0]

    async def append(
        self,
        execution_id: str,
        *,
        tenant_id: str,
        expected_sequence: int,
        event_type: str,
        payload: object,
    ) -> ExecutionEventRecord:
        return await self.append_expected(
            execution_id,
            tenant_id=tenant_id,
            expected_sequence=expected_sequence,
            event_type=event_type,
            payload=payload,
        )

    async def list(
        self, execution_id: str, *, tenant_id: str, after_sequence: int, limit: int
    ) -> Page[ExecutionEventRecord]:
        if tenant_id != self._tenant_id:
            return Page(())
        _validate_page_limit(limit)
        stream = stream_digest(
            self._namespace,
            self._tenant_id,
            self._domain.value,
            "execution",
            execution_id,
        )
        values = await self._store.read(
            lambda transaction: transaction.list_facts(
                FactQuery(
                    stream,
                    after_sequence=after_sequence,
                    limit=min(limit + 1, 1000),
                )
            )
        )
        if limit == 1000 and len(values) == 1000:
            extra = await self._store.read(
                lambda transaction: transaction.list_facts(
                    FactQuery(
                        stream,
                        after_sequence=values[-1].sequence,
                        limit=1,
                    )
                )
            )
            if extra:
                values = (*values, extra[0])
        items = tuple(
            ExecutionEventRecord(
                execution_id,
                tenant_id,
                value.sequence,
                value.kind,
                value.data,
            )
            for value in values[:limit]
        )
        return Page(
            items, str(items[-1].sequence) if len(values) > limit and items else None
        )


def _event_payload(value: object) -> Mapping[str, JsonValue]:
    if isinstance(value, Mapping):
        return value  # type: ignore[return-value]
    return {"value": value}  # type: ignore[dict-item]


def _same_idempotency(left: IdempotencyRecord, right: IdempotencyRecord) -> bool:
    return (
        _same_idempotency_identity(left, right)
        and left.resource_id == right.resource_id
    )


def _same_idempotency_identity(
    left: IdempotencyRecord, right: IdempotencyRecord
) -> bool:
    """Compare only the immutable request identity, never a candidate resource id."""
    return (
        left.tenant_id == right.tenant_id
        and left.scope == right.scope
        and left.idempotency_key_digest == right.idempotency_key_digest
        and left.request_digest == right.request_digest
        and left.resource_kind is right.resource_kind
    )


def _execution_replay_matches(left: ExecutionRecord, right: ExecutionRecord) -> bool:
    return (
        left.execution_id == right.execution_id
        and left.tenant_id == right.tenant_id
        and left.session_id == right.session_id
        and left.binding_digest == right.binding_digest
        and left.parent_execution_id == right.parent_execution_id
        and left.root_execution_id == right.root_execution_id
        and left.parent_invocation_id == right.parent_invocation_id
        and left.source_execution_id == right.source_execution_id
        and left.base_execution_id == right.base_execution_id
        and left.lineage_kind is right.lineage_kind
        and left.repository_instructions == right.repository_instructions
    )


__all__ = [
    "EventRepositoryImpl",
    "ExecutionRepositoryImpl",
    "IdempotencyRepositoryImpl",
]
