#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Task repositories with durable recovery indexing and legacy admission repair."""

import asyncio
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, replace
from datetime import datetime
from typing import TypeVar

from ...core import OperationStatus, Page, TaskStatus
from ...errors import AIError, ErrorCode
from ...storage import StoredPayload
from ...task import (
    TaskEvent,
    TaskEventType,
    TaskGraph,
    TaskGraphAdmission,
    TaskGraphLaunch,
    TaskGraphSnapshot,
    TaskGraphView,
    TaskLease,
    TaskNodeView,
    TaskResultRecord,
    TaskTerminalRecord,
)
from ._repositories import (
    TaskAdmissionRepositoryImpl,
    TaskRepositoryImpl,
    _decode_operation,
    _decode_record_cursor,
    _record_cursor,
    _replace_checked,
    _require_live_task_lease,
    _stored_operation_error,
    _task_graph_record,
    _validate_task_lease_scope,
)
from ._store import (
    FactQuery,
    RecordQuery,
    StateTransaction,
    StoredFact,
    StoredOperation,
    StoredRecord,
    operation_key,
    sequence_key,
    stream_digest,
)

_ValueT = TypeVar("_ValueT")
_CURRENT_CURSOR_PREFIX = "a:"
_LEGACY_CURSOR_PREFIX = "g:"
_COMMIT_READBACK_CODES = frozenset(
    {ErrorCode.STORAGE_CONFLICT, ErrorCode.STORAGE_COMMIT_UNKNOWN}
)
_TERMINAL_TASK_STATUSES = frozenset(
    {
        TaskStatus.SUCCEEDED,
        TaskStatus.FAILED,
        TaskStatus.BLOCKED,
        TaskStatus.CANCELLED,
    }
)


class _TaskEventAppendConflict(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class _TaskEventDraft:
    event_type: TaskEventType
    status: TaskStatus
    previous_status: "TaskStatus | None" = None
    node_id: "str | None" = None
    owner: "str | None" = None
    fence: int = 0
    execution_id: "str | None" = None
    result_digest: "str | None" = None
    error_code: "str | None" = None
    error_digest: "str | None" = None


def _task_event_stream(repository: object, graph_id: str) -> bytes:
    return stream_digest(
        repository._namespace,  # type: ignore[attr-defined]
        repository._tenant_id,  # type: ignore[attr-defined]
        repository._domain.value,  # type: ignore[attr-defined]
        "task_event",
        graph_id,
    )


def _task_event_sequence(repository: object, graph_id: str) -> bytes:
    return sequence_key(
        repository._namespace,  # type: ignore[attr-defined]
        repository._tenant_id,  # type: ignore[attr-defined]
        repository._domain.value,  # type: ignore[attr-defined]
        "task_event",
        graph_id,
    )


def _task_node_changed(left: TaskNodeView, right: TaskNodeView) -> bool:
    return (
        left.status is not right.status
        or left.owner != right.owner
        or left.fence != right.fence
        or left.execution_id != right.execution_id
        or left.result_digest != right.result_digest
        or left.error_code != right.error_code
        or left.error_digest != right.error_digest
    )


def _task_event_drafts(
    before: "TaskGraphSnapshot | None",
    after: TaskGraphSnapshot,
) -> tuple[_TaskEventDraft, ...]:
    values: list[_TaskEventDraft] = []
    if before is None:
        values.append(_TaskEventDraft(TaskEventType.GRAPH_ADMITTED, after.status))
        for node in after.node_states:
            values.append(
                _TaskEventDraft(
                    TaskEventType.NODE_CHANGED,
                    node.status,
                    node_id=node.node_id,
                    owner=node.owner,
                    fence=node.fence,
                    execution_id=node.execution_id,
                    result_digest=node.result_digest,
                    error_code=node.error_code,
                    error_digest=node.error_digest,
                )
            )
        return tuple(values)
    before_nodes = {node.node_id: node for node in before.node_states}
    if len(before_nodes) != len(before.node_states):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    for node in after.node_states:
        previous = before_nodes.get(node.node_id)
        if previous is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if not _task_node_changed(previous, node):
            continue
        values.append(
            _TaskEventDraft(
                TaskEventType.NODE_CHANGED,
                node.status,
                previous_status=previous.status,
                node_id=node.node_id,
                owner=node.owner,
                fence=node.fence,
                execution_id=node.execution_id,
                result_digest=node.result_digest,
                error_code=node.error_code,
                error_digest=node.error_digest,
            )
        )
    if len(before_nodes) != len(after.node_states):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    if before.status is not after.status:
        values.append(
            _TaskEventDraft(
                TaskEventType.GRAPH_CHANGED,
                after.status,
                previous_status=before.status,
            )
        )
    return tuple(values)


def _task_event_payload(draft: _TaskEventDraft, occurred_at: datetime) -> dict[str, object]:
    return {
        "version": 1,
        "occurred_at": occurred_at.isoformat(),
        "previous_status": (
            None if draft.previous_status is None else draft.previous_status.value
        ),
        "status": draft.status.value,
        "node_id": draft.node_id,
        "owner": draft.owner,
        "fence": draft.fence,
        "execution_id": draft.execution_id,
        "result_digest": draft.result_digest,
        "error_code": draft.error_code,
        "error_digest": draft.error_digest,
    }


async def _append_task_events(
    repository: object,
    transaction: StateTransaction,
    graph_id: str,
    drafts: tuple[_TaskEventDraft, ...],
) -> None:
    if not drafts:
        return
    graph_key = repository._graph_key(graph_id)  # type: ignore[attr-defined]
    graph_record = await transaction.get_record(graph_key)
    if graph_record is None:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    guarded = await transaction.guard_record(
        graph_key,
        expected_storage_version=graph_record.storage_version,
    )
    if guarded is None:
        raise _TaskEventAppendConflict
    final_sequence = await transaction.reserve_sequence(
        _task_event_sequence(repository, graph_id),
        len(drafts),
    )
    first_sequence = final_sequence - len(drafts) + 1
    occurred_at = await transaction.now()
    stream = _task_event_stream(repository, graph_id)
    facts = tuple(
        StoredFact(
            stream,
            first_sequence + index,
            graph_key,
            draft.event_type.value,
            None,
            draft.status.value,
            _task_event_payload(draft, occurred_at),
        )
        for index, draft in enumerate(drafts)
    )
    await transaction.insert_facts(facts)


async def _append_task_snapshot_events(
    repository: object,
    transaction: StateTransaction,
    before: "TaskGraphSnapshot | None",
    after: TaskGraphSnapshot,
) -> None:
    await _append_task_events(
        repository,
        transaction,
        after.graph_id,
        _task_event_drafts(before, after),
    )


def _decode_task_event(graph_id: str, fact: StoredFact) -> TaskEvent:
    try:
        data = fact.data
        version = data["version"]
        occurred_at = data["occurred_at"]
        status = data["status"]
        previous_status = data.get("previous_status")
        node_id = data.get("node_id")
        owner = data.get("owner")
        fence = data.get("fence", 0)
        execution_id = data.get("execution_id")
        result_digest = data.get("result_digest")
        error_code = data.get("error_code")
        error_digest = data.get("error_digest")
        if not isinstance(version, int) or isinstance(version, bool):
            raise TypeError("task event version is invalid")
        if not isinstance(occurred_at, str):
            raise TypeError("task event time is invalid")
        if not isinstance(status, str):
            raise TypeError("task event status is invalid")
        if previous_status is not None and not isinstance(previous_status, str):
            raise TypeError("task event previous status is invalid")
        if node_id is not None and not isinstance(node_id, str):
            raise TypeError("task event node id is invalid")
        if owner is not None and not isinstance(owner, str):
            raise TypeError("task event owner is invalid")
        if not isinstance(fence, int) or isinstance(fence, bool):
            raise TypeError("task event fence is invalid")
        for value in (execution_id, result_digest, error_code, error_digest):
            if value is not None and not isinstance(value, str):
                raise TypeError("task event string field is invalid")
        return TaskEvent(
            version,
            graph_id,
            fact.sequence,
            TaskEventType(fact.kind),
            datetime.fromisoformat(occurred_at),
            TaskStatus(status),
            None if previous_status is None else TaskStatus(previous_status),
            node_id,
            owner,
            fence,
            execution_id,
            result_digest,
            error_code,
            error_digest,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error


class DurableTaskRepositoryImpl(TaskRepositoryImpl):
    """Keep Task authority and recovery projections convergent under CAS races."""

    def _admission_key(self, graph_id: str) -> bytes:
        return self._key("task_admission", graph_id)

    def _recovery_scope(self) -> bytes:
        return self._scope("task_admission", "recoverable", "graphs")

    def _legacy_recovery_scope(self) -> bytes:
        return self._scope("task_graph", "recoverable", "graphs")

    async def _mutate_with_event_retry(
        self,
        operation: Callable[[StateTransaction], Awaitable[_ValueT]],
    ) -> _ValueT:
        while True:
            try:
                return await self.state_store.mutate(operation)
            except _TaskEventAppendConflict:
                await asyncio.sleep(0)

    async def create_graph(self, graph: TaskGraph, *, tenant_id: str) -> TaskGraphView:
        async def mutate(transaction: StateTransaction) -> TaskGraphView:
            view = await super(DurableTaskRepositoryImpl, self).create_graph(
                graph,
                tenant_id=tenant_id,
            )
            snapshot = await self.snapshot_graph(graph.graph_id, tenant_id=tenant_id)
            if snapshot is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            await _append_task_snapshot_events(self, transaction, None, snapshot)
            return view

        return await self._mutate_with_event_retry(mutate)

    async def get_graph(self, graph_id: str, *, tenant_id: str) -> TaskGraphView | None:
        if tenant_id != self._tenant_id:
            return None
        record = await self._record(self._graph_key(graph_id))
        if record is None:
            return None
        graph = await self._decode(record, TaskGraphView)
        nodes = await self.list_nodes(graph_id, tenant_id=tenant_id)
        return TaskGraphView(
            graph.graph_id,
            _effective_graph_status(graph, nodes),
            graph.nodes,
        )

    async def snapshot_graph(
        self,
        graph_id: str,
        *,
        tenant_id: str,
    ) -> TaskGraphSnapshot | None:
        if tenant_id != self._tenant_id:
            return None

        async def read(transaction: StateTransaction) -> TaskGraphSnapshot | None:
            graph_record = await transaction.get_record(self._graph_key(graph_id))
            if graph_record is None:
                return None
            graph = await self._decode(graph_record, TaskGraphView)
            records = await transaction.list_records(
                RecordQuery(
                    parent_digest=self._parent("task_node", "graph", graph_id),
                    kind="task_node",
                )
            )
            states = await self._decode_many(records)
            by_id = {state.node_id: state for state in states}
            if len(by_id) != len(states):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            try:
                ordered = tuple(by_id[node.node_id] for node in graph.nodes)
            except KeyError as error:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
            if len(ordered) != len(states):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            return TaskGraphSnapshot(
                graph.graph_id,
                _effective_graph_status(graph, ordered),
                graph.nodes,
                ordered,
            )

        return await self.state_store.read(read)

    async def list_events(
        self,
        graph_id: str,
        *,
        tenant_id: str,
        after_sequence: int,
        limit: int,
    ) -> Page[TaskEvent]:
        if tenant_id != self._tenant_id:
            return Page(())
        if isinstance(after_sequence, bool) or not isinstance(after_sequence, int) or after_sequence < 0:
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 1000:
            raise AIError(ErrorCode.PAGE_LIMIT_INVALID)
        values = await self.state_store.read(
            lambda transaction: transaction.list_facts(
                FactQuery(
                    _task_event_stream(self, graph_id),
                    after_sequence=after_sequence,
                    limit=limit + 1,
                )
            )
        )
        items = tuple(_decode_task_event(graph_id, value) for value in values[:limit])
        return Page(
            items,
            str(items[-1].sequence) if len(values) > limit and items else None,
        )

    async def bind_execution(
        self,
        lease: TaskLease,
        *,
        tenant_id: str,
        execution_id: str,
    ) -> TaskNodeView:
        try:
            async def mutate(transaction: StateTransaction) -> TaskNodeView:
                before = await self.snapshot_graph(lease.graph_id, tenant_id=tenant_id)
                if before is None:
                    raise AIError(ErrorCode.TASK_FENCE_STALE)
                result = await super(DurableTaskRepositoryImpl, self).bind_execution(
                    lease,
                    tenant_id=tenant_id,
                    execution_id=execution_id,
                )
                after = await self.snapshot_graph(lease.graph_id, tenant_id=tenant_id)
                if after is None:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                await _append_task_snapshot_events(self, transaction, before, after)
                return result

            return await self._mutate_with_event_retry(mutate)
        except AIError as error:
            if error.code not in _COMMIT_READBACK_CODES:
                raise
            current = await self._node(lease.graph_id, lease.node_id, tenant_id)
            if current.fence != lease.fence:
                raise AIError(ErrorCode.TASK_FENCE_STALE) from error
            if current.execution_id == execution_id:
                return current
            if current.execution_id is not None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
            if current.owner != lease.owner:
                raise AIError(ErrorCode.TASK_OWNER_CONFLICT) from error
            if error.code is ErrorCode.STORAGE_COMMIT_UNKNOWN:
                raise AIError(
                    ErrorCode.STORAGE_RECOVERY_REQUIRED,
                    safe_details={
                        "phase": "task_execution_bind",
                        "graph_id": lease.graph_id,
                        "node_id": lease.node_id,
                    },
                ) from error
            raise AIError(ErrorCode.STORAGE_CONFLICT) from error

    async def reconcile_graph(self, graph_id: str, *, tenant_id: str) -> TaskGraphView:
        async def mutate(transaction: StateTransaction) -> TaskGraphView:
            before = await self.snapshot_graph(graph_id, tenant_id=tenant_id)
            if before is None:
                raise AIError(ErrorCode.STORAGE_NOT_FOUND)
            graph_record = await transaction.get_record(self._graph_key(graph_id))
            if graph_record is None:
                raise AIError(ErrorCode.STORAGE_NOT_FOUND)
            current_graph = await self._decode(graph_record, TaskGraphView)
            preserve_cancelled = current_graph.status is TaskStatus.CANCELLED
            await super(DurableTaskRepositoryImpl, self).reconcile_graph(
                graph_id,
                tenant_id=tenant_id,
            )
            view = await self._stabilize_graph_projection(
                transaction,
                graph_id,
                preserve_cancelled=preserve_cancelled,
            )
            await self._sync_recovery_projection(transaction, view)
            after = await self.snapshot_graph(graph_id, tenant_id=tenant_id)
            if after is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            await _append_task_snapshot_events(self, transaction, before, after)
            return view

        try:
            return await self._mutate_with_event_retry(mutate)
        except AIError as error:
            if error.code not in _COMMIT_READBACK_CODES:
                raise
            view, converged = await self._projection_readback(
                graph_id,
                tenant_id=tenant_id,
            )
            if converged:
                return view
            if error.code is ErrorCode.STORAGE_COMMIT_UNKNOWN:
                raise AIError(
                    ErrorCode.STORAGE_RECOVERY_REQUIRED,
                    safe_details={
                        "phase": "task_reconcile",
                        "graph_id": graph_id,
                    },
                ) from error
            raise AIError(ErrorCode.STORAGE_CONFLICT) from error

    async def cancel_graph(self, graph_id: str, *, tenant_id: str) -> TaskGraphView:
        async def mutate(transaction: StateTransaction) -> TaskGraphView:
            before = await self.snapshot_graph(graph_id, tenant_id=tenant_id)
            if before is None:
                raise AIError(ErrorCode.STORAGE_NOT_FOUND)
            graph_record = await transaction.get_record(self._graph_key(graph_id))
            if graph_record is None:
                raise AIError(ErrorCode.STORAGE_NOT_FOUND)
            graph = await self._decode(graph_record, TaskGraphView)
            node_records = await transaction.list_records(
                RecordQuery(
                    parent_digest=self._parent("task_node", "graph", graph_id),
                    kind="task_node",
                )
            )
            nodes = await self._decode_many(node_records)
            had_nonterminal = any(
                node.status not in _TERMINAL_TASK_STATUSES
                for node in nodes
            )
            await super(DurableTaskRepositoryImpl, self).cancel_graph(
                graph_id,
                tenant_id=tenant_id,
            )
            if not had_nonterminal:
                view = await self._stabilize_graph_projection(
                    transaction,
                    graph_id,
                    preserve_cancelled=graph.status is TaskStatus.CANCELLED,
                )
            else:
                view = await self._force_cancelled_graph_projection(
                    transaction,
                    graph_id,
                )
            await self._sync_recovery_projection(transaction, view)
            after = await self.snapshot_graph(graph_id, tenant_id=tenant_id)
            if after is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            await _append_task_snapshot_events(self, transaction, before, after)
            return view

        try:
            return await self._mutate_with_event_retry(mutate)
        except AIError as error:
            if error.code not in _COMMIT_READBACK_CODES:
                raise
            view, converged = await self._projection_readback(
                graph_id,
                tenant_id=tenant_id,
            )
            if converged and view.status is TaskStatus.CANCELLED:
                return view
            if view.status in {
                TaskStatus.SUCCEEDED,
                TaskStatus.FAILED,
                TaskStatus.BLOCKED,
            }:
                return view
            if error.code is ErrorCode.STORAGE_COMMIT_UNKNOWN:
                raise AIError(
                    ErrorCode.STORAGE_RECOVERY_REQUIRED,
                    safe_details={
                        "phase": "task_cancel",
                        "graph_id": graph_id,
                    },
                ) from error
            raise AIError(ErrorCode.STORAGE_CONFLICT) from error

    async def claim(
        self,
        graph_id: str,
        node_id: str,
        *,
        tenant_id: str,
        owner: str,
        lease_seconds: int,
    ) -> TaskLease:
        before_node = await self._node(graph_id, node_id, tenant_id)
        expected_fence = before_node.fence + 1
        try:
            async def mutate(transaction: StateTransaction) -> TaskLease:
                before = await self.snapshot_graph(graph_id, tenant_id=tenant_id)
                if before is None:
                    raise AIError(ErrorCode.TASK_NOT_READY)
                result = await super(DurableTaskRepositoryImpl, self).claim(
                    graph_id,
                    node_id,
                    tenant_id=tenant_id,
                    owner=owner,
                    lease_seconds=lease_seconds,
                )
                after = await self.snapshot_graph(graph_id, tenant_id=tenant_id)
                if after is None:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                await _append_task_snapshot_events(self, transaction, before, after)
                return result

            return await self._mutate_with_event_retry(mutate)
        except AIError as error:
            if error.code not in _COMMIT_READBACK_CODES:
                raise
            current = await self._node(graph_id, node_id, tenant_id)
            if current.status is TaskStatus.RUNNING:
                if current.owner != owner:
                    raise AIError(ErrorCode.TASK_OWNER_CONFLICT) from error
                if current.fence != expected_fence:
                    raise AIError(ErrorCode.TASK_FENCE_STALE) from error
                if current.lease_expires_at is None:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
                return TaskLease(
                    graph_id, node_id, tenant_id, owner, current.fence, current.lease_expires_at
                )
            if current.status in {
                TaskStatus.SUCCEEDED,
                TaskStatus.FAILED,
                TaskStatus.CANCELLED,
                TaskStatus.BLOCKED,
            }:
                raise AIError(ErrorCode.TASK_NOT_READY) from error
            if error.code is ErrorCode.STORAGE_COMMIT_UNKNOWN:
                raise AIError(
                    ErrorCode.STORAGE_RECOVERY_REQUIRED,
                    safe_details={
                        "phase": "task_claim",
                        "graph_id": graph_id,
                        "node_id": node_id,
                    },
                ) from error
            raise AIError(ErrorCode.STORAGE_CONFLICT) from error

    async def complete(
        self,
        lease: TaskLease,
        *,
        tenant_id: str,
        execution_id: str | None,
        result_digest: str,
        result_payload: StoredPayload | None = None,
    ) -> TaskTerminalRecord:
        try:
            async def mutate(transaction: StateTransaction) -> TaskTerminalRecord:
                before = await self.snapshot_graph(lease.graph_id, tenant_id=tenant_id)
                if before is None:
                    raise AIError(ErrorCode.TASK_FENCE_STALE)
                result = await super(DurableTaskRepositoryImpl, self).complete(
                    lease,
                    tenant_id=tenant_id,
                    execution_id=execution_id,
                    result_digest=result_digest,
                    result_payload=result_payload,
                )
                after = await self.snapshot_graph(lease.graph_id, tenant_id=tenant_id)
                if after is None:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                await _append_task_snapshot_events(self, transaction, before, after)
                return result

            return await self._mutate_with_event_retry(mutate)
        except AIError as error:
            if error.code not in _COMMIT_READBACK_CODES:
                raise
            return await self._classify_terminal_readback(
                lease,
                tenant_id=tenant_id,
                status=TaskStatus.SUCCEEDED,
                execution_id=execution_id,
                result_digest=result_digest,
                result_payload=result_payload,
                error_code=None,
                error_digest=None,
                conflict=error,
            )

    async def fail(
        self,
        lease: TaskLease,
        *,
        tenant_id: str,
        error_code: str,
        error_digest: str,
        execution_id: str | None = None,
    ) -> TaskTerminalRecord:
        try:
            async def mutate(transaction: StateTransaction) -> TaskTerminalRecord:
                before = await self.snapshot_graph(lease.graph_id, tenant_id=tenant_id)
                if before is None:
                    raise AIError(ErrorCode.TASK_FENCE_STALE)
                result = await super(DurableTaskRepositoryImpl, self).fail(
                    lease,
                    tenant_id=tenant_id,
                    error_code=error_code,
                    error_digest=error_digest,
                    execution_id=execution_id,
                )
                after = await self.snapshot_graph(lease.graph_id, tenant_id=tenant_id)
                if after is None:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                await _append_task_snapshot_events(self, transaction, before, after)
                return result

            return await self._mutate_with_event_retry(mutate)
        except AIError as error:
            if error.code not in _COMMIT_READBACK_CODES:
                raise
            return await self._classify_terminal_readback(
                lease,
                tenant_id=tenant_id,
                status=TaskStatus.FAILED,
                execution_id=execution_id,
                result_digest=None,
                result_payload=None,
                error_code=error_code,
                error_digest=error_digest,
                conflict=error,
            )

    async def _classify_terminal_readback(
        self,
        lease: TaskLease,
        *,
        tenant_id: str,
        status: TaskStatus,
        execution_id: str | None,
        result_digest: str | None,
        result_payload: StoredPayload | None,
        error_code: str | None,
        error_digest: str | None,
        conflict: AIError,
    ) -> TaskTerminalRecord:
        current = await self._node(lease.graph_id, lease.node_id, tenant_id)
        if current.fence != lease.fence:
            raise AIError(ErrorCode.TASK_FENCE_STALE) from conflict
        if current.status is status:
            if (
                current.result_digest != result_digest
                or current.error_code != error_code
                or current.error_digest != error_digest
                or (
                    execution_id is not None
                    and current.execution_id != execution_id
                )
            ):
                raise AIError(ErrorCode.TASK_RESULT_CONFLICT) from conflict
            if status is TaskStatus.SUCCEEDED and result_payload is not None:
                results = await self.get_results(
                    lease.graph_id,
                    (lease.node_id,),
                    tenant_id=tenant_id,
                )
                result = results.get(lease.node_id)
                if result is None or result.result_digest != result_digest:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from conflict
            return TaskTerminalRecord(
                lease.node_id,
                lease.owner,
                lease.fence,
                status,
                result_digest,
                error_code,
                error_digest,
                execution_id=current.execution_id,
            )
        if current.status in {
            TaskStatus.SUCCEEDED,
            TaskStatus.FAILED,
            TaskStatus.CANCELLED,
            TaskStatus.BLOCKED,
        }:
            raise AIError(ErrorCode.TASK_TERMINAL_CONFLICT) from conflict
        if current.owner != lease.owner:
            raise AIError(ErrorCode.TASK_FENCE_STALE) from conflict
        if conflict.code is ErrorCode.STORAGE_COMMIT_UNKNOWN:
            raise AIError(
                ErrorCode.STORAGE_RECOVERY_REQUIRED,
                safe_details={
                    "phase": "task_terminal_commit",
                    "graph_id": lease.graph_id,
                    "node_id": lease.node_id,
                },
            ) from conflict
        raise AIError(ErrorCode.STORAGE_CONFLICT) from conflict

    async def _stabilize_graph_projection(
        self,
        transaction: StateTransaction,
        graph_id: str,
        *,
        preserve_cancelled: bool,
    ) -> TaskGraphView:
        graph_record = await transaction.get_record(self._graph_key(graph_id))
        if graph_record is None:
            raise AIError(ErrorCode.STORAGE_NOT_FOUND)
        graph = await self._decode(graph_record, TaskGraphView)
        node_records = await transaction.list_records(
            RecordQuery(
                parent_digest=self._parent("task_node", "graph", graph_id),
                kind="task_node",
            )
        )
        nodes = await self._decode_many(node_records)
        status = (
            TaskStatus.CANCELLED
            if preserve_cancelled
            else _isolated_graph_status(nodes)
        )
        view = TaskGraphView(graph.graph_id, status, graph.nodes)
        if graph.status is status and graph_record.state == status.value:
            return view
        await _replace_checked(
            transaction,
            _task_graph_record(self, graph_record, replace(graph, status=status)),
            graph_record.storage_version,
        )
        return view

    async def _force_cancelled_graph_projection(
        self,
        transaction: StateTransaction,
        graph_id: str,
    ) -> TaskGraphView:
        graph_record = await transaction.get_record(self._graph_key(graph_id))
        if graph_record is None:
            raise AIError(ErrorCode.STORAGE_NOT_FOUND)
        graph = await self._decode(graph_record, TaskGraphView)
        view = TaskGraphView(graph.graph_id, TaskStatus.CANCELLED, graph.nodes)
        if graph.status is TaskStatus.CANCELLED and graph_record.state == TaskStatus.CANCELLED.value:
            return view
        await _replace_checked(
            transaction,
            _task_graph_record(
                self,
                graph_record,
                replace(graph, status=TaskStatus.CANCELLED),
            ),
            graph_record.storage_version,
        )
        return view

    async def _projection_readback(
        self,
        graph_id: str,
        *,
        tenant_id: str,
    ) -> tuple[TaskGraphView, bool]:
        if tenant_id != self._tenant_id:
            raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)

        async def read(transaction: StateTransaction) -> tuple[TaskGraphView, bool]:
            graph_key = self._graph_key(graph_id)
            admission_key = self._admission_key(graph_id)
            records = await transaction.get_records((graph_key, admission_key))
            graph_record = records.get(graph_key)
            if graph_record is None:
                raise AIError(ErrorCode.STORAGE_NOT_FOUND)
            graph = await self._decode(graph_record, TaskGraphView)
            node_records = await transaction.list_records(
                RecordQuery(
                    parent_digest=self._parent("task_node", "graph", graph_id),
                    kind="task_node",
                )
            )
            states = await self._decode_many(node_records)
            status = _effective_graph_status(graph, states)
            snapshot = TaskGraphSnapshot(
                graph.graph_id,
                status,
                graph.nodes,
                tuple(
                    {state.node_id: state for state in states}[node.node_id]
                    for node in graph.nodes
                ),
            )
            view = TaskGraphView(graph.graph_id, snapshot.status, graph.nodes)
            graph_converged = (
                graph.status is snapshot.status
                and graph_record.state == snapshot.status.value
            )
            admission_record = records.get(admission_key)
            if admission_record is None:
                return view, graph_converged
            if admission_record.scope_digest == self._recovery_scope():
                if (
                    admission_record.kind != "task_admission"
                    or admission_record.key_digest != admission_key
                ):
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                return view, graph_converged and admission_record.state == snapshot.status.value
            if (
                admission_record.scope_digest is None
                and graph_record.scope_digest == self._legacy_recovery_scope()
            ):
                return view, graph_converged
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

        try:
            return await self.state_store.read(read)
        except (KeyError, ValueError) as error:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error

    async def _sync_recovery_projection(
        self,
        transaction: StateTransaction,
        view: TaskGraphView,
    ) -> None:
        admission_key = self._admission_key(view.graph_id)
        graph_key = self._graph_key(view.graph_id)
        records = await transaction.get_records((admission_key, graph_key))
        admission_record = records.get(admission_key)
        if admission_record is None:
            return
        graph_record = records.get(graph_key)
        if graph_record is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if admission_record.key_digest != admission_key or admission_record.kind != "task_admission":
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if admission_record.scope_digest == self._recovery_scope():
            if admission_record.state == view.status.value:
                return
            candidate = replace(
                admission_record,
                state=view.status.value,
                storage_version=admission_record.storage_version + 1,
            )
            if not await transaction.replace_record(
                candidate,
                expected_storage_version=admission_record.storage_version,
            ):
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            return
        if (
            admission_record.scope_digest is None
            and graph_record.scope_digest == self._legacy_recovery_scope()
        ):
            return
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)


class DurableTaskAdmissionRepositoryImpl(TaskAdmissionRepositoryImpl):
    """Index recoverable graphs by admission without rewriting graph identity."""

    def _recovery_scope(self) -> bytes:
        return self._scope("task_admission", "recoverable", "graphs")

    def _legacy_recovery_scope(self) -> bytes:
        return self._scope("task_graph", "recoverable", "graphs")

    async def list_recoverable_page(
        self,
        *,
        cursor: str | None,
        limit: int,
    ) -> Page[TaskGraphLaunch]:
        if limit != 128:
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        phase, inner_cursor = _decode_recovery_cursor(cursor)
        if phase == "admission":
            current = await self._list_current_recoverable_page(
                cursor=inner_cursor,
                limit=limit,
            )
            if current.items or current.next_cursor is not None:
                if current.next_cursor is not None:
                    next_cursor = _CURRENT_CURSOR_PREFIX + current.next_cursor
                else:
                    legacy_probe = await self._list_legacy_recoverable_page(
                        cursor=None,
                        limit=1,
                    )
                    next_cursor = (
                        _LEGACY_CURSOR_PREFIX
                        if legacy_probe.items or legacy_probe.next_cursor is not None
                        else None
                    )
                return Page(current.items, next_cursor)
            inner_cursor = None
        legacy = await self._list_legacy_recoverable_page(
            cursor=inner_cursor,
            limit=limit,
        )
        return Page(
            legacy.items,
            (
                None
                if legacy.next_cursor is None
                else _LEGACY_CURSOR_PREFIX + legacy.next_cursor
            ),
        )

    async def _list_current_recoverable_page(
        self,
        *,
        cursor: str | None,
        limit: int,
    ) -> Page[TaskGraphLaunch]:
        after_sort_key, after_key_digest = _decode_record_cursor(cursor)

        async def read(transaction: StateTransaction) -> Page[TaskGraphLaunch]:
            records = await transaction.list_records(
                RecordQuery(
                    scope_digest=self._recovery_scope(),
                    kind="task_admission",
                    states=_RECOVERABLE_STATES,
                    after_sort_key=after_sort_key,
                    after_key_digest=after_key_digest,
                    limit=limit + 1,
                )
            )
            selected = records[:limit]
            admissions = tuple(
                [await self._decode(record, TaskGraphAdmission) for record in selected]
            )
            graph_records = await transaction.get_records(
                tuple(self._graph_key(admission.graph_id) for admission in admissions)
            )
            launches: list[TaskGraphLaunch] = []
            for record, admission in zip(selected, admissions, strict=True):
                graph_record = graph_records.get(self._graph_key(admission.graph_id))
                if graph_record is None:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                graph_view = await self._decode(graph_record, TaskGraphView)
                if (
                    record.key_digest != self._admission_key(admission.graph_id)
                    or record.scope_digest != self._recovery_scope()
                    or record.state != graph_view.status.value
                    or graph_record.key_digest != self._graph_key(admission.graph_id)
                    or graph_record.state != graph_view.status.value
                    or graph_view.graph_id != admission.graph_id
                ):
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                launches.append(
                    admission.bind(TaskGraph(graph_view.graph_id, graph_view.nodes))
                )
            next_cursor = (
                _record_cursor(selected[-1])
                if len(records) > limit and selected
                else None
            )
            return Page(tuple(launches), next_cursor)

        return await self.state_store.read(read)

    async def _list_legacy_recoverable_page(
        self,
        *,
        cursor: str | None,
        limit: int,
    ) -> Page[TaskGraphLaunch]:
        after_sort_key, after_key_digest = _decode_record_cursor(cursor)

        async def read(transaction: StateTransaction) -> Page[TaskGraphLaunch]:
            records = await transaction.list_records(
                RecordQuery(
                    scope_digest=self._legacy_recovery_scope(),
                    kind="task_graph",
                    states=_RECOVERABLE_STATES,
                    after_sort_key=after_sort_key,
                    after_key_digest=after_key_digest,
                    limit=limit + 1,
                )
            )
            selected = records[:limit]
            graph_views = tuple(
                [await self._decode(record, TaskGraphView) for record in selected]
            )
            admission_records = await transaction.get_records(
                tuple(
                    self._admission_key(graph_view.graph_id)
                    for graph_view in graph_views
                )
            )
            launches: list[TaskGraphLaunch] = []
            for graph_record, graph_view in zip(selected, graph_views, strict=True):
                admission_record = admission_records.get(
                    self._admission_key(graph_view.graph_id)
                )
                if admission_record is None:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                if admission_record.scope_digest == self._recovery_scope():
                    continue
                if admission_record.scope_digest is not None:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                admission = await self._decode(admission_record, TaskGraphAdmission)
                if (
                    graph_record.key_digest != self._graph_key(graph_view.graph_id)
                    or graph_record.scope_digest != self._legacy_recovery_scope()
                    or graph_record.state != graph_view.status.value
                    or admission.graph_id != graph_view.graph_id
                    or admission_record.key_digest != self._admission_key(graph_view.graph_id)
                ):
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                launches.append(
                    admission.bind(TaskGraph(graph_view.graph_id, graph_view.nodes))
                )
            next_cursor = (
                _record_cursor(selected[-1])
                if len(records) > limit and selected
                else None
            )
            return Page(
                tuple(launches),
                None if next_cursor is None else next_cursor,
            )

        return await self.state_store.read(read)

    async def _admit_in_transaction(
        self,
        transaction: StateTransaction,
        admission: TaskGraphAdmission,
        graph: TaskGraph,
    ) -> TaskGraphView:
        graph_key = self._graph_key(graph.graph_id)
        admission_key = self._admission_key(graph.graph_id)
        records = await transaction.get_records((graph_key, admission_key))
        graph_record = records.get(graph_key)
        admission_record = records.get(admission_key)
        stored_operation = await transaction.get_operation(
            operation_key(
                self._namespace,
                self._tenant_id,
                self._domain.value,
                admission.operation_id,
            )
        )
        if graph_record is None or stored_operation is None:
            return await super()._admit_in_transaction(transaction, admission, graph)

        operation = _decode_operation(stored_operation)
        if operation.request_digest != admission.request_digest:
            raise AIError(ErrorCode.IDEMPOTENCY_CONFLICT)
        self._validate_operation_identity(operation, admission)
        if operation.status in {OperationStatus.FAILED, OperationStatus.CANCELLED}:
            raise _stored_operation_error(operation)
        if operation.status not in {
            OperationStatus.PENDING,
            OperationStatus.RUNNING,
            OperationStatus.EFFECT_UNKNOWN,
            OperationStatus.SUCCEEDED,
        }:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

        node_records = await transaction.list_records(
            RecordQuery(
                parent_digest=self._parent("task_node", "graph", graph.graph_id),
                kind="task_node",
            )
        )
        if admission_record is not None:
            existing, view = await self._require_committed_admission(
                transaction,
                graph_record,
                admission_record,
                node_records,
                stored_operation=stored_operation,
            )
            if existing != admission:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            await self._repair_aggregate_projection(
                transaction,
                graph_record,
                admission_record,
                view,
            )
            return view

        graph_view = await self._decode(graph_record, TaskGraphView)
        nodes = await self._decode_task_nodes(node_records)
        self._validate_graph(graph_view, graph, nodes)
        if operation.status is OperationStatus.SUCCEEDED:
            self._validate_succeeded_operation(operation, graph)

        status = _effective_graph_status(graph_view, nodes)
        next_graph = TaskGraphView(graph.graph_id, status, graph.nodes)
        if graph_view.status is not status or graph_record.state != status.value:
            await _replace_checked(
                transaction,
                _task_graph_record(self, graph_record, next_graph),
                graph_record.storage_version,
            )
        await transaction.insert_record(
            self._stored(
                "task_admission",
                graph.graph_id,
                admission,
                scope=self._recovery_scope(),
                state=status.value,
            )
        )
        if operation.status is not OperationStatus.SUCCEEDED:
            await self._settle_operation(transaction, stored_operation, operation, graph)
        return next_graph

    async def _insert_admission_records(
        self,
        transaction: StateTransaction,
        admission: TaskGraphAdmission,
        graph: TaskGraph,
    ) -> TaskGraphView:
        status = TaskStatus.SUCCEEDED if not graph.nodes else TaskStatus.PENDING
        view = TaskGraphView(graph.graph_id, status, graph.nodes)
        records = [
            self._stored(
                "task_graph",
                graph.graph_id,
                view,
                state=status.value,
            ),
            self._stored(
                "task_admission",
                graph.graph_id,
                admission,
                scope=self._recovery_scope(),
                state=status.value,
            ),
        ]
        node_views: list[TaskNodeView] = []
        for node in graph.nodes:
            node_status = TaskStatus.READY if not node.dependencies else TaskStatus.PENDING
            node_view = TaskNodeView(
                graph.graph_id,
                node.node_id,
                node.dependencies,
                node_status,
                None,
                0,
                None,
                None,
                None,
                None,
            )
            node_views.append(node_view)
            records.append(
                self._stored(
                    "task_node",
                    [graph.graph_id, node.node_id],
                    node_view,
                    parent=self._parent("task_node", "graph", graph.graph_id),
                    state=node_status.value,
                )
            )
        await transaction.insert_records(records)
        snapshot = TaskGraphSnapshot(
            graph.graph_id,
            status,
            graph.nodes,
            tuple(node_views),
        )
        await _append_task_snapshot_events(self, transaction, None, snapshot)
        return view

    async def _require_committed_admission(
        self,
        transaction: StateTransaction,
        graph_record: StoredRecord,
        admission_record: StoredRecord,
        node_records: tuple[StoredRecord, ...],
        *,
        stored_operation: StoredOperation | None = None,
    ) -> tuple[TaskGraphAdmission, TaskGraphView]:
        existing = await self._decode(admission_record, TaskGraphAdmission)
        graph_view = await self._decode(graph_record, TaskGraphView)
        nodes = await self._decode_task_nodes(node_records)
        persisted_graph = TaskGraph(graph_view.graph_id, graph_view.nodes)
        existing.bind(persisted_graph)
        self._validate_graph(graph_view, persisted_graph, nodes)
        status = _effective_graph_status(graph_view, nodes)
        if graph_record.key_digest != self._graph_key(graph_view.graph_id):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if admission_record.key_digest != self._admission_key(graph_view.graph_id):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if not self._recognized_layout(graph_record, admission_record):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if stored_operation is None:
            stored_operation = await transaction.get_operation(
                operation_key(
                    self._namespace,
                    self._tenant_id,
                    self._domain.value,
                    existing.operation_id,
                )
            )
        if stored_operation is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        operation = _decode_operation(stored_operation)
        self._validate_operation_identity(operation, existing)
        if operation.request_digest != existing.request_digest:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        self._validate_succeeded_operation(operation, persisted_graph)
        return existing, TaskGraphView(graph_view.graph_id, status, graph_view.nodes)

    async def _repair_aggregate_projection(
        self,
        transaction: StateTransaction,
        graph_record: StoredRecord,
        admission_record: StoredRecord,
        view: TaskGraphView,
    ) -> None:
        graph_view = await self._decode(graph_record, TaskGraphView)
        if graph_view.status is not view.status or graph_record.state != view.status.value:
            await _replace_checked(
                transaction,
                _task_graph_record(self, graph_record, replace(graph_view, status=view.status)),
                graph_record.storage_version,
            )
        if (
            admission_record.scope_digest == self._recovery_scope()
            and admission_record.state != view.status.value
        ):
            candidate = replace(
                admission_record,
                state=view.status.value,
                storage_version=admission_record.storage_version + 1,
            )
            if not await transaction.replace_record(
                candidate,
                expected_storage_version=admission_record.storage_version,
            ):
                raise AIError(ErrorCode.STORAGE_CONFLICT)

    def _recognized_layout(
        self,
        graph_record: StoredRecord,
        admission_record: StoredRecord,
    ) -> bool:
        return admission_record.scope_digest == self._recovery_scope() or (
            admission_record.scope_digest is None
            and graph_record.scope_digest == self._legacy_recovery_scope()
        )


_RECOVERABLE_STATES = frozenset(
    {
        TaskStatus.PENDING.value,
        TaskStatus.READY.value,
        TaskStatus.RUNNING.value,
    }
)


def _isolated_graph_status(nodes: tuple[TaskNodeView, ...]) -> TaskStatus:
    statuses = {node.status for node in nodes}
    if not statuses or statuses <= {TaskStatus.SUCCEEDED}:
        return TaskStatus.SUCCEEDED
    if TaskStatus.FAILED in statuses:
        return TaskStatus.FAILED
    if TaskStatus.BLOCKED in statuses:
        return TaskStatus.BLOCKED
    if statuses <= {TaskStatus.CANCELLED, TaskStatus.SUCCEEDED}:
        return TaskStatus.CANCELLED
    if TaskStatus.RUNNING in statuses:
        return TaskStatus.RUNNING
    return TaskStatus.PENDING


def _effective_graph_status(
    graph: TaskGraphView,
    nodes: tuple[TaskNodeView, ...],
) -> TaskStatus:
    if graph.status is TaskStatus.CANCELLED:
        return TaskStatus.CANCELLED
    return _isolated_graph_status(nodes)


def _decode_recovery_cursor(cursor: str | None) -> tuple[str, str | None]:
    if cursor is None:
        return "admission", None
    if cursor.startswith(_CURRENT_CURSOR_PREFIX):
        inner = cursor[len(_CURRENT_CURSOR_PREFIX) :]
        return "admission", inner or None
    if cursor.startswith(_LEGACY_CURSOR_PREFIX):
        inner = cursor[len(_LEGACY_CURSOR_PREFIX) :]
        return "graph", inner or None
    return "graph", cursor


__all__ = ["DurableTaskAdmissionRepositoryImpl", "DurableTaskRepositoryImpl"]
