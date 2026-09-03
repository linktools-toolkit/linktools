#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Canonical Task persistence repositories and durable event history."""

import asyncio
import heapq
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from typing import TypeVar

from linktools.core import environ

from ...core import (
    OperationKind,
    OperationLedgerInput,
    OperationLedgerRecord,
    OperationStatus,
    Page,
    ResourceKind,
    ResourceRef,
    TaskStatus,
    canonical_sha256,
    validate_lease_owner,
    validate_lease_seconds,
)
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
from ._durability import CommitObservation, DurableCommitState, run_durable_commit
from ._plan import RuntimeDomain
from ._repositories import (
    RepositoryBase,
    append_operation,
    decode_operation,
    decode_record_cursor,
    projected_record,
    record_cursor,
    replace_checked,
    require_repository_tenant,
)
from ._store import (
    FactQuery,
    RecordQuery,
    RecordReplacement,
    StateStore,
    StateTransaction,
    StoredFact,
    StoredOperation,
    StoredRecord,
    operation_key,
    sequence_key,
    sortable_identity,
    stream_digest,
)

_logger = environ.get_logger("ai.runtime.state.task_repository")
_ValueT = TypeVar("_ValueT")
_TASK_EVENT_RETRY_LIMIT = 16
_TASK_EVENT_RETRY_BASE_SECONDS = 0.001
_TASK_EVENT_RETRY_MAX_SECONDS = 0.05
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


class _TaskEventAppendConflict(AIError):
    def __init__(self) -> None:
        super().__init__(ErrorCode.STORAGE_CONFLICT)


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


@dataclass(frozen=True, slots=True)
class _TaskEventState:
    graph: TaskGraphView
    node_states: tuple[TaskNodeView, ...]


def _resolve_task_execution_id(
    current: str | None,
    supplied: str | None,
) -> str | None:
    if current is None:
        return supplied
    if supplied is None or supplied == current:
        return current
    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)


def _require_live_task_lease(
    node: TaskNodeView,
    lease: TaskLease,
    now: datetime,
) -> None:
    if (
        node.status is not TaskStatus.RUNNING
        or node.owner != lease.owner
        or node.fence != lease.fence
        or node.lease_expires_at is None
        or node.lease_expires_at <= now
    ):
        raise AIError(ErrorCode.TASK_FENCE_STALE)


def _validate_task_lease_scope(lease: TaskLease, tenant_id: str) -> None:
    if lease.tenant_id != tenant_id:
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID)


def _task_submit_result_digest(graph: TaskGraph) -> str:
    status = TaskStatus.SUCCEEDED if not graph.nodes else TaskStatus.PENDING
    return canonical_sha256({"graph_id": graph.graph_id, "status": status.value})


def _require_canonical_graph_status(status: TaskStatus) -> None:
    if status is TaskStatus.READY:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)


def _task_event_stream(
    namespace: str,
    tenant_id: str,
    domain: str,
    graph_id: str,
) -> bytes:
    return stream_digest(namespace, tenant_id, domain, "task_event", graph_id)


def _task_event_sequence(
    namespace: str,
    tenant_id: str,
    domain: str,
    graph_id: str,
) -> bytes:
    return sequence_key(namespace, tenant_id, domain, "task_event", graph_id)


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


def _task_node_event_drafts(
    before: TaskNodeView,
    after: TaskNodeView,
) -> tuple[_TaskEventDraft, ...]:
    if before.graph_id != after.graph_id or before.node_id != after.node_id:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    if not _task_node_changed(before, after):
        return ()
    return (
        _TaskEventDraft(
            TaskEventType.NODE_CHANGED,
            after.status,
            previous_status=before.status,
            node_id=after.node_id,
            owner=after.owner,
            fence=after.fence,
            execution_id=after.execution_id,
            result_digest=after.result_digest,
            error_code=after.error_code,
            error_digest=after.error_digest,
        ),
    )


def _task_graph_event_drafts(
    before: TaskGraphView,
    after: TaskGraphView,
) -> tuple[_TaskEventDraft, ...]:
    if before.graph_id != after.graph_id or before.nodes != after.nodes:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    if before.status is after.status:
        return ()
    return (
        _TaskEventDraft(
            TaskEventType.GRAPH_CHANGED,
            after.status,
            previous_status=before.status,
        ),
    )


def _task_event_drafts(
    before: "_TaskEventState | None",
    after: _TaskEventState,
) -> tuple[_TaskEventDraft, ...]:
    if before is None:
        return (_TaskEventDraft(TaskEventType.GRAPH_ADMITTED, after.graph.status),)
    if before.graph.graph_id != after.graph.graph_id:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    before_nodes = {node.node_id: node for node in before.node_states}
    if len(before_nodes) != len(before.node_states):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    values: list[_TaskEventDraft] = []
    for node in after.node_states:
        previous = before_nodes.get(node.node_id)
        if previous is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        values.extend(_task_node_event_drafts(previous, node))
    if len(before_nodes) != len(after.node_states):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    values.extend(_task_graph_event_drafts(before.graph, after.graph))
    return tuple(values)


def _reconciled_task_nodes(
    nodes: tuple[TaskNodeView, ...],
) -> tuple[TaskNodeView, ...]:
    values = {node.node_id: node for node in nodes}
    if len(values) != len(nodes):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    indegree = {node.node_id: len(node.dependencies) for node in nodes}
    dependents: dict[str, list[str]] = {node.node_id: [] for node in nodes}
    for node in nodes:
        for dependency in node.dependencies:
            if dependency not in values:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            dependents[dependency].append(node.node_id)
    ready = [node_id for node_id, degree in indegree.items() if degree == 0]
    heapq.heapify(ready)
    order: list[str] = []
    while ready:
        node_id = heapq.heappop(ready)
        order.append(node_id)
        for dependent in dependents[node_id]:
            indegree[dependent] -= 1
            if indegree[dependent] == 0:
                heapq.heappush(ready, dependent)
    if len(order) != len(nodes):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    for node_id in order:
        node = values[node_id]
        if node.status in _TERMINAL_TASK_STATUSES:
            continue
        dependencies = tuple(values[dependency] for dependency in node.dependencies)
        if any(
            dependency.status
            in {TaskStatus.FAILED, TaskStatus.BLOCKED, TaskStatus.CANCELLED}
            for dependency in dependencies
        ):
            values[node_id] = replace(
                node,
                status=TaskStatus.BLOCKED,
                error_code=ErrorCode.TASK_DEPENDENCY_FAILED.value,
                error_digest=None,
            )
        elif node.status is TaskStatus.PENDING and all(
            dependency.status is TaskStatus.SUCCEEDED for dependency in dependencies
        ):
            values[node_id] = replace(node, status=TaskStatus.READY)
    return tuple(values[node.node_id] for node in nodes)


def _task_event_payload(
    draft: _TaskEventDraft, occurred_at: datetime
) -> dict[str, object]:
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


async def _guard_task_event_owner(
    transaction: StateTransaction,
    graph_key: bytes,
    *,
    missing_code: ErrorCode = ErrorCode.STORAGE_INTEGRITY_ERROR,
) -> StoredRecord:
    graph_record = await transaction.get_record(graph_key)
    if graph_record is None:
        raise AIError(missing_code)
    guarded = await transaction.guard_record(
        graph_key,
        expected_storage_version=graph_record.storage_version,
    )
    if guarded is None:
        raise _TaskEventAppendConflict
    return guarded


async def _append_task_events(
    transaction: StateTransaction,
    *,
    namespace: str,
    tenant_id: str,
    domain: str,
    graph_id: str,
    graph_key: bytes,
    drafts: tuple[_TaskEventDraft, ...],
    owner_guarded: bool = False,
) -> None:
    if not drafts:
        return
    if not owner_guarded:
        await _guard_task_event_owner(transaction, graph_key)
    final_sequence = await transaction.reserve_sequence(
        _task_event_sequence(namespace, tenant_id, domain, graph_id),
        len(drafts),
    )
    first_sequence = final_sequence - len(drafts) + 1
    occurred_at = await transaction.now()
    stream = _task_event_stream(namespace, tenant_id, domain, graph_id)
    events = tuple(
        TaskEvent(
            1,
            graph_id,
            first_sequence + index,
            draft.event_type,
            occurred_at,
            draft.status,
            draft.previous_status,
            draft.node_id,
            draft.owner,
            draft.fence,
            draft.execution_id,
            draft.result_digest,
            draft.error_code,
            draft.error_digest,
        )
        for index, draft in enumerate(drafts)
    )
    facts = tuple(
        StoredFact(
            stream,
            event.sequence,
            graph_key,
            event.event_type.value,
            None,
            None,
            _task_event_payload(draft, occurred_at),
        )
        for event, draft in zip(events, drafts, strict=True)
    )
    await transaction.insert_facts(facts)


async def _append_task_state_events(
    transaction: StateTransaction,
    *,
    namespace: str,
    tenant_id: str,
    domain: str,
    graph_key: bytes,
    before: "_TaskEventState | None",
    after: _TaskEventState,
) -> None:
    await _append_task_events(
        transaction,
        namespace=namespace,
        tenant_id=tenant_id,
        domain=domain,
        graph_id=after.graph.graph_id,
        graph_key=graph_key,
        drafts=_task_event_drafts(before, after),
    )


def _decode_task_event(graph_id: str, fact: StoredFact) -> TaskEvent:
    try:
        data = fact.data
        version = data["version"]
        if not isinstance(version, int) or isinstance(version, bool):
            raise TypeError("task event version is invalid")
        if version != 1:
            raise AIError(ErrorCode.STORAGE_VERSION_UNSUPPORTED)
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
        if not isinstance(occurred_at, str) or not isinstance(status, str):
            raise TypeError("task event payload is invalid")
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
    except AIError:
        raise
    except (KeyError, TypeError, ValueError) as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error


class TaskRepositoryImpl(RepositoryBase):
    def __init__(self, store: StateStore, *, namespace: str, tenant_id: str) -> None:
        super().__init__(
            store, namespace=namespace, tenant_id=tenant_id, domain=RuntimeDomain.TASK
        )

    def _graph_key(self, graph_id: str) -> bytes:
        return self._key("task_graph", graph_id)

    def _node_key(self, graph_id: str, node_id: str) -> bytes:
        return self._key("task_node", [graph_id, node_id])

    def _result_key(self, graph_id: str, node_id: str) -> bytes:
        return self._key("task_result", [graph_id, node_id])

    def _result_scope(self, graph_id: str) -> bytes:
        return self._scope("task_result", "graph", graph_id)

    def _result_parent(self, graph_id: str) -> bytes:
        return self._parent("task_result", "graph", graph_id)

    def _validate_graph_record(self, record: StoredRecord, graph_id: str) -> None:
        if (
            record.kind != "task_graph"
            or record.key_digest != self._graph_key(graph_id)
            or record.partition_digest != self._partition("task_graph")
            or record.scope_digest is not None
            or record.parent_digest is not None
            or record.sort_key != sortable_identity(graph_id)
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

    def _validate_node_record(
        self,
        record: StoredRecord,
        graph_id: str,
        node_id: str,
    ) -> None:
        if (
            record.kind != "task_node"
            or record.key_digest != self._node_key(graph_id, node_id)
            or record.partition_digest != self._partition("task_node")
            or record.scope_digest is not None
            or record.parent_digest != self._parent("task_node", "graph", graph_id)
            or record.sort_key != sortable_identity([graph_id, node_id])
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

    def _validate_result_record(
        self,
        record: StoredRecord,
        graph_id: str,
        node_id: str,
    ) -> None:
        if (
            record.kind != "task_result"
            or record.key_digest != self._result_key(graph_id, node_id)
            or record.partition_digest != self._partition("task_result")
            or record.scope_digest != self._result_scope(graph_id)
            or record.parent_digest != self._result_parent(graph_id)
            or record.sort_key != sortable_identity([graph_id, node_id])
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

    async def get_results(
        self,
        graph_id: str,
        node_ids: tuple[str, ...],
        *,
        tenant_id: str,
    ) -> Mapping[str, TaskResultRecord]:
        if tenant_id != self._tenant_id:
            return {}
        if not isinstance(node_ids, tuple):
            raise TypeError("node_ids must be a tuple")
        if any(not isinstance(node_id, str) or not node_id for node_id in node_ids):
            raise ValueError("node_ids must contain non-empty strings")
        if len(set(node_ids)) != len(node_ids):
            raise ValueError("node_ids must not contain duplicates")
        if not node_ids:
            return {}
        keys = tuple(self._result_key(graph_id, node_id) for node_id in node_ids)

        async def read(transaction: StateTransaction) -> Mapping[str, TaskResultRecord]:
            records = await transaction.get_records(keys)
            result: dict[str, TaskResultRecord] = {}
            for node_id, key in zip(node_ids, keys, strict=True):
                record = records.get(key)
                if record is None:
                    continue
                self._validate_result_record(record, graph_id, node_id)
                value = await self._decode(record, TaskResultRecord)
                if value.graph_id != graph_id or value.node_id != node_id:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                result[node_id] = value
            return result

        return await self._store.read(read)

    async def renew(
        self, lease: TaskLease, *, tenant_id: str, lease_seconds: int
    ) -> TaskLease:
        _validate_task_lease_scope(lease, tenant_id)
        if tenant_id != self._tenant_id:
            raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)
        validate_lease_seconds(lease_seconds)

        async def mutate(transaction: StateTransaction) -> TaskLease:
            records = await transaction.get_records(
                (
                    self._graph_key(lease.graph_id),
                    self._node_key(lease.graph_id, lease.node_id),
                )
            )
            graph_record = records.get(self._graph_key(lease.graph_id))
            node_record = records.get(self._node_key(lease.graph_id, lease.node_id))
            if graph_record is None or node_record is None:
                raise AIError(ErrorCode.TASK_FENCE_STALE)
            self._validate_graph_record(graph_record, lease.graph_id)
            self._validate_node_record(node_record, lease.graph_id, lease.node_id)
            node = await self._decode(node_record, TaskNodeView)
            now = await transaction.now()
            _require_live_task_lease(node, lease, now)
            expires = now + timedelta(seconds=lease_seconds)
            if not await transaction.update_record_lease(
                node_record.key_digest,
                expected_storage_version=node_record.storage_version,
                lease_owner=node.owner,
                lease_fence=node.fence,
                lease_expires_at=expires,
            ):
                raise AIError(ErrorCode.TASK_FENCE_STALE)
            return replace(lease, lease_expires_at=expires)

        return await self._store.mutate(mutate)

    async def list_nodes(
        self, graph_id: str, *, tenant_id: str
    ) -> tuple[TaskNodeView, ...]:
        if tenant_id != self._tenant_id:
            return ()
        records = await self._records(
            "task_node", parent=self._parent("task_node", "graph", graph_id)
        )
        return await self._decode_many(records)

    async def _decode_many(
        self,
        records: tuple[StoredRecord, ...],
    ) -> tuple[TaskNodeView, ...]:
        values: list[TaskNodeView] = []
        for record in records:
            value = await self._decode(record, TaskNodeView)
            self._validate_node_record(record, value.graph_id, value.node_id)
            values.append(value)
        return tuple(values)

    async def _node(self, graph_id: str, node_id: str, tenant_id: str) -> TaskNodeView:
        require_repository_tenant(tenant_id, self._tenant_id)
        record = await self._record(self._node_key(graph_id, node_id))
        if record is None:
            raise AIError(ErrorCode.STORAGE_NOT_FOUND)
        self._validate_node_record(record, graph_id, node_id)
        value = await self._decode(record, TaskNodeView)
        if value.graph_id != graph_id or value.node_id != node_id:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return value

    async def _update_node_in_transaction(
        self,
        transaction: StateTransaction,
        current: TaskNodeView,
        value: TaskNodeView,
        node_record: StoredRecord,
    ) -> None:
        self._validate_node_record(node_record, current.graph_id, current.node_id)
        stored_node = await self._decode(node_record, TaskNodeView)
        if stored_node != current:
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        node_candidate = projected_record(self, node_record, value)
        await replace_checked(transaction, node_candidate, node_record.storage_version)

    def _admission_key(self, graph_id: str) -> bytes:
        return self._key("task_admission", graph_id)

    def _validate_admission_record(self, record: StoredRecord, graph_id: str) -> None:
        if (
            record.kind != "task_admission"
            or record.key_digest != self._admission_key(graph_id)
            or record.partition_digest != self._partition("task_admission")
            or record.scope_digest != self._recovery_scope()
            or record.parent_digest is not None
            or record.sort_key != sortable_identity(graph_id)
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

    def _recovery_scope(self) -> bytes:
        return self._scope("task_admission", "recoverable", "graphs")

    async def _mutate_with_event_retry(
        self,
        operation: Callable[[StateTransaction], Awaitable[_ValueT]],
    ) -> _ValueT:
        for attempt in range(_TASK_EVENT_RETRY_LIMIT):
            try:
                return await self.state_store.mutate(operation)
            except _TaskEventAppendConflict as error:
                if attempt + 1 == _TASK_EVENT_RETRY_LIMIT:
                    raise AIError(ErrorCode.STORAGE_CONFLICT) from error
                delay = min(
                    _TASK_EVENT_RETRY_BASE_SECONDS * (2**attempt),
                    _TASK_EVENT_RETRY_MAX_SECONDS,
                )
                await asyncio.sleep(delay)
        raise AIError(ErrorCode.STORAGE_CONFLICT)

    async def _event_state_in_transaction(
        self,
        transaction: StateTransaction,
        graph_id: str,
    ) -> _TaskEventState | None:
        graph_record = await transaction.get_record(self._graph_key(graph_id))
        if graph_record is None:
            return None
        self._validate_graph_record(graph_record, graph_id)
        graph = await self._decode(graph_record, TaskGraphView)
        _require_canonical_graph_status(graph.status)
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
        if any(
            state.graph_id != graph.graph_id
            or state.node_id != node.node_id
            or state.dependencies != node.dependencies
            for node, state in zip(graph.nodes, ordered, strict=True)
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return _TaskEventState(graph, ordered)

    async def _snapshot_graph_in_transaction(
        self,
        transaction: StateTransaction,
        graph_id: str,
    ) -> TaskGraphSnapshot | None:
        state = await self._event_state_in_transaction(transaction, graph_id)
        if state is None:
            return None
        return TaskGraphSnapshot(
            state.graph.graph_id,
            _effective_graph_status(state.graph, state.node_states),
            state.graph.nodes,
            state.node_states,
        )

    async def _node_in_transaction(
        self,
        transaction: StateTransaction,
        graph_id: str,
        node_id: str,
        *,
        missing_code: ErrorCode,
    ) -> TaskNodeView:
        record = await transaction.get_record(self._node_key(graph_id, node_id))
        if record is None:
            raise AIError(missing_code)
        self._validate_node_record(record, graph_id, node_id)
        value = await self._decode(record, TaskNodeView)
        if value.graph_id != graph_id or value.node_id != node_id:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return value

    def _task_node_record(
        self,
        current: StoredRecord,
        value: TaskNodeView,
    ) -> StoredRecord:
        self._validate_node_record(current, value.graph_id, value.node_id)
        candidate = self._stored(
            "task_node",
            [value.graph_id, value.node_id],
            value,
            state=value.status.value,
        )
        return replace(candidate, storage_version=current.storage_version + 1)

    async def _apply_graph_transition(
        self,
        transaction: StateTransaction,
        before: _TaskEventState,
        graph_record: StoredRecord,
        next_nodes: tuple[TaskNodeView, ...],
        next_status: TaskStatus,
    ) -> TaskGraphView:
        if len(next_nodes) != len(before.node_states):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        changes = tuple(
            (current, value)
            for current, value in zip(before.node_states, next_nodes, strict=True)
            if current != value
        )
        view = TaskGraphView(before.graph.graph_id, next_status, before.graph.nodes)
        graph_changed = (
            before.graph.status is not next_status
            or graph_record.state != next_status.value
        )
        drafts: list[_TaskEventDraft] = []
        for current, value in changes:
            drafts.extend(_task_node_event_drafts(current, value))
        drafts.extend(_task_graph_event_drafts(before.graph, view))
        event_drafts = tuple(drafts)
        if event_drafts:
            graph_record = await _guard_task_event_owner(
                transaction,
                self._graph_key(before.graph.graph_id),
                missing_code=ErrorCode.STORAGE_NOT_FOUND,
            )

        replacements: list[RecordReplacement] = []
        if changes:
            node_keys = tuple(
                self._node_key(before.graph.graph_id, current.node_id)
                for current, _ in changes
            )
            node_records = await transaction.get_records(node_keys)
            if len(node_records) != len(node_keys):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            for current, value in changes:
                node_record = node_records.get(
                    self._node_key(before.graph.graph_id, current.node_id)
                )
                if node_record is None:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                self._validate_node_record(
                    node_record,
                    before.graph.graph_id,
                    current.node_id,
                )
                stored_node = await self._decode(node_record, TaskNodeView)
                if stored_node != current:
                    raise AIError(ErrorCode.STORAGE_CONFLICT)
                replacements.append(
                    RecordReplacement(
                        self._task_node_record(node_record, value),
                        node_record.storage_version,
                    )
                )
        if graph_changed:
            replacements.append(
                RecordReplacement(
                    projected_record(
                        self,
                        graph_record,
                        replace(before.graph, status=next_status),
                    ),
                    graph_record.storage_version,
                )
            )
        if replacements:
            await transaction.replace_records(tuple(replacements))
        await self._sync_recovery_projection(transaction, view)
        await _append_task_events(
            transaction,
            namespace=self._namespace,
            tenant_id=self._tenant_id,
            domain=self._domain.value,
            graph_id=before.graph.graph_id,
            graph_key=self._graph_key(before.graph.graph_id),
            drafts=event_drafts,
            owner_guarded=bool(event_drafts),
        )
        return view

    async def get_header(
        self,
        graph_id: str,
        *,
        tenant_id: str,
    ) -> ResourceRef | None:
        if tenant_id != self._tenant_id:
            return None
        record = await self._record(self._graph_key(graph_id))
        if record is None:
            return None
        self._validate_graph_record(record, graph_id)
        graph = await self._decode(record, TaskGraphView)
        if graph.graph_id != graph_id:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return ResourceRef(ResourceKind.TASK_GRAPH, graph_id, tenant_id)

    async def get_graph(self, graph_id: str, *, tenant_id: str) -> TaskGraphView | None:
        if tenant_id != self._tenant_id:
            return None
        snapshot = await self.snapshot_graph(graph_id, tenant_id=tenant_id)
        if snapshot is None:
            return None
        return TaskGraphView(snapshot.graph_id, snapshot.status, snapshot.nodes)

    async def snapshot_graph(
        self,
        graph_id: str,
        *,
        tenant_id: str,
    ) -> TaskGraphSnapshot | None:
        if tenant_id != self._tenant_id:
            return None
        return await self.state_store.read(
            lambda transaction: self._snapshot_graph_in_transaction(
                transaction,
                graph_id,
            )
        )

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
        if (
            isinstance(after_sequence, bool)
            or not isinstance(after_sequence, int)
            or after_sequence < 0
        ):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        if (
            isinstance(limit, bool)
            or not isinstance(limit, int)
            or not 1 <= limit <= 1000
        ):
            raise AIError(ErrorCode.PAGE_LIMIT_INVALID)
        stream = _task_event_stream(
            self._namespace,
            self._tenant_id,
            self._domain.value,
            graph_id,
        )
        owner = self._graph_key(graph_id)

        async def read(
            transaction: StateTransaction,
        ) -> tuple[tuple[StoredFact, ...], bool]:
            query_limit = min(limit + 1, 1000)
            values = await transaction.list_facts(
                FactQuery(
                    stream,
                    after_sequence=after_sequence,
                    limit=query_limit,
                )
            )
            has_more = len(values) > limit
            if not has_more and limit == 1000 and len(values) == limit:
                probe = await transaction.list_facts(
                    FactQuery(
                        stream,
                        after_sequence=values[-1].sequence,
                        limit=1,
                    )
                )
                has_more = bool(probe)
            return values, has_more

        values, has_more = await self.state_store.read(read)
        items: list[TaskEvent] = []
        expected_sequence = after_sequence
        for value in values[:limit]:
            expected_sequence += 1
            if (
                value.stream_digest != stream
                or value.owner_key_digest != owner
                or value.subject_digest is not None
                or value.sequence != expected_sequence
            ):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            event = _decode_task_event(graph_id, value)
            if value.state is not None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            items.append(event)
        page_items = tuple(items)
        return Page(
            page_items,
            str(page_items[-1].sequence) if has_more and page_items else None,
        )

    async def latest_event(
        self,
        graph_id: str,
        *,
        tenant_id: str,
    ) -> TaskEvent | None:
        if tenant_id != self._tenant_id:
            return None
        stream = _task_event_stream(
            self._namespace,
            self._tenant_id,
            self._domain.value,
            graph_id,
        )
        owner = self._graph_key(graph_id)

        async def read(transaction: StateTransaction) -> TaskEvent | None:
            values = await transaction.list_facts(
                FactQuery(stream, latest=True, limit=1)
            )
            if not values:
                return None
            fact = values[0]
            if (
                fact.stream_digest != stream
                or fact.owner_key_digest != owner
                or fact.subject_digest is not None
                or fact.state is not None
            ):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            return _decode_task_event(graph_id, fact)

        return await self.state_store.read(read)

    async def bind_execution(
        self,
        lease: TaskLease,
        *,
        tenant_id: str,
        execution_id: str,
    ) -> TaskNodeView:
        _validate_task_lease_scope(lease, tenant_id)
        if tenant_id != self._tenant_id:
            raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)
        if not isinstance(execution_id, str) or not execution_id.strip():
            raise ValueError("execution_id must be a non-empty string")

        async def mutate(transaction: StateTransaction) -> TaskNodeView:
            graph_key = self._graph_key(lease.graph_id)
            node_key = self._node_key(lease.graph_id, lease.node_id)
            records = await transaction.get_records((graph_key, node_key))
            graph_record = records.get(graph_key)
            node_record = records.get(node_key)
            if graph_record is None or node_record is None:
                raise AIError(ErrorCode.TASK_FENCE_STALE)
            self._validate_graph_record(graph_record, lease.graph_id)
            self._validate_node_record(node_record, lease.graph_id, lease.node_id)
            node = await self._decode(node_record, TaskNodeView)
            now = await transaction.now()
            _require_live_task_lease(node, lease, now)
            if node.execution_id == execution_id:
                return node
            if node.execution_id is not None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            value = replace(node, execution_id=execution_id)
            if (
                await transaction.guard_record(
                    graph_key,
                    expected_storage_version=graph_record.storage_version,
                )
                is None
            ):
                raise _TaskEventAppendConflict()
            await self._update_node_in_transaction(
                transaction, node, value, node_record
            )
            await _append_task_events(
                transaction,
                namespace=self._namespace,
                tenant_id=self._tenant_id,
                domain=self._domain.value,
                graph_id=lease.graph_id,
                graph_key=graph_key,
                drafts=_task_node_event_drafts(node, value),
                owner_guarded=True,
            )
            return value

        try:
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
            before = await self._event_state_in_transaction(transaction, graph_id)
            if before is None:
                raise AIError(ErrorCode.STORAGE_NOT_FOUND)
            graph_record = await transaction.get_record(self._graph_key(graph_id))
            if graph_record is None:
                raise AIError(ErrorCode.STORAGE_NOT_FOUND)
            next_nodes = _reconciled_task_nodes(before.node_states)
            next_status = (
                TaskStatus.CANCELLED
                if before.graph.status is TaskStatus.CANCELLED
                else _isolated_graph_status(next_nodes)
            )
            return await self._apply_graph_transition(
                transaction,
                before,
                graph_record,
                next_nodes,
                next_status,
            )

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
            before = await self._event_state_in_transaction(transaction, graph_id)
            if before is None:
                raise AIError(ErrorCode.STORAGE_NOT_FOUND)
            graph_record = await transaction.get_record(self._graph_key(graph_id))
            if graph_record is None:
                raise AIError(ErrorCode.STORAGE_NOT_FOUND)
            next_nodes = tuple(
                (
                    replace(
                        node,
                        status=TaskStatus.CANCELLED,
                        owner=None,
                        lease_expires_at=None,
                    )
                    if node.status not in _TERMINAL_TASK_STATUSES
                    else node
                )
                for node in before.node_states
            )
            next_status = (
                TaskStatus.CANCELLED
                if before.graph.status is TaskStatus.CANCELLED
                or next_nodes != before.node_states
                else _isolated_graph_status(next_nodes)
            )
            return await self._apply_graph_transition(
                transaction,
                before,
                graph_record,
                next_nodes,
                next_status,
            )

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
        if tenant_id != self._tenant_id:
            raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)
        validate_lease_owner(owner)
        validate_lease_seconds(lease_seconds)
        expected_fence: int | None = None

        async def mutate(transaction: StateTransaction) -> TaskLease:
            nonlocal expected_fence
            graph_key = self._graph_key(graph_id)
            node_key = self._node_key(graph_id, node_id)
            records = await transaction.get_records((graph_key, node_key))
            graph_record = records.get(graph_key)
            node_record = records.get(node_key)
            if graph_record is None or node_record is None:
                raise AIError(ErrorCode.TASK_NOT_READY)
            self._validate_graph_record(graph_record, graph_id)
            self._validate_node_record(node_record, graph_id, node_id)
            graph = await self._decode(graph_record, TaskGraphView)
            _require_canonical_graph_status(graph.status)
            node = await self._decode(node_record, TaskNodeView)
            if (
                graph.graph_id != graph_id
                or node.graph_id != graph_id
                or node.node_id != node_id
            ):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            graph_node = None
            for candidate in graph.nodes:
                if candidate.node_id != node_id:
                    continue
                if graph_node is not None:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                graph_node = candidate
            if graph_node is None or node.dependencies != graph_node.dependencies:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            dependency_keys = tuple(
                self._node_key(graph_id, dependency) for dependency in node.dependencies
            )
            dependency_records = await transaction.get_records(dependency_keys)
            dependencies: dict[str, TaskNodeView] = {}
            for dependency in node.dependencies:
                record = dependency_records.get(self._node_key(graph_id, dependency))
                if record is None:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                self._validate_node_record(record, graph_id, dependency)
                dependency_view = await self._decode(record, TaskNodeView)
                if (
                    dependency_view.graph_id != graph_id
                    or dependency_view.node_id != dependency
                ):
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                dependencies[dependency] = dependency_view
            now = await transaction.now()
            expired = node.lease_expires_at is not None and node.lease_expires_at <= now
            if (
                node.status is TaskStatus.RUNNING
                and node.owner not in {None, owner}
                and not expired
            ):
                raise AIError(ErrorCode.TASK_OWNER_CONFLICT)
            dependencies_succeeded = all(
                dependencies[dependency].status is TaskStatus.SUCCEEDED
                for dependency in node.dependencies
            )
            if node.status not in {TaskStatus.PENDING, TaskStatus.READY} and not (
                node.status is TaskStatus.RUNNING and expired
            ):
                raise AIError(ErrorCode.TASK_NOT_READY)
            if (
                node.status in {TaskStatus.PENDING, TaskStatus.READY}
                and not dependencies_succeeded
            ):
                raise AIError(ErrorCode.TASK_NOT_READY)
            expected_fence = node.fence + 1
            expires = now + timedelta(seconds=lease_seconds)
            value = replace(
                node,
                status=TaskStatus.RUNNING,
                owner=owner,
                fence=expected_fence,
                lease_expires_at=expires,
            )
            if (
                await transaction.guard_record(
                    graph_key,
                    expected_storage_version=graph_record.storage_version,
                )
                is None
            ):
                raise _TaskEventAppendConflict()
            await self._update_node_in_transaction(
                transaction, node, value, node_record
            )
            await _append_task_events(
                transaction,
                namespace=self._namespace,
                tenant_id=self._tenant_id,
                domain=self._domain.value,
                graph_id=graph_id,
                graph_key=graph_key,
                drafts=_task_node_event_drafts(node, value),
                owner_guarded=True,
            )
            return TaskLease(
                graph_id,
                node_id,
                tenant_id,
                owner,
                expected_fence,
                expires,
            )

        try:
            return await self._mutate_with_event_retry(mutate)
        except AIError as error:
            if error.code not in _COMMIT_READBACK_CODES:
                raise
            current = await self._node(graph_id, node_id, tenant_id)
            if current.status is TaskStatus.RUNNING:
                if current.owner != owner:
                    raise AIError(ErrorCode.TASK_OWNER_CONFLICT) from error
                if expected_fence is None or current.fence != expected_fence:
                    raise AIError(ErrorCode.TASK_FENCE_STALE) from error
                if current.lease_expires_at is None:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
                return TaskLease(
                    graph_id,
                    node_id,
                    tenant_id,
                    owner,
                    current.fence,
                    current.lease_expires_at,
                )
            if current.status in _TERMINAL_TASK_STATUSES:
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
        _validate_task_lease_scope(lease, tenant_id)
        if tenant_id != self._tenant_id:
            raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)
        if result_payload is not None and result_payload.digest != result_digest:
            raise AIError(ErrorCode.TASK_RESULT_CONFLICT)

        async def mutate(transaction: StateTransaction) -> TaskTerminalRecord:
            graph_key = self._graph_key(lease.graph_id)
            node_key = self._node_key(lease.graph_id, lease.node_id)
            result_key = self._result_key(lease.graph_id, lease.node_id)
            records = await transaction.get_records((graph_key, node_key, result_key))
            graph_record = records.get(graph_key)
            node_record = records.get(node_key)
            if graph_record is None or node_record is None:
                raise AIError(ErrorCode.TASK_FENCE_STALE)
            self._validate_graph_record(graph_record, lease.graph_id)
            self._validate_node_record(node_record, lease.graph_id, lease.node_id)
            node = await self._decode(node_record, TaskNodeView)
            current_result_record = records.get(result_key)
            current_result: TaskResultRecord | None = None
            if current_result_record is not None:
                self._validate_result_record(
                    current_result_record,
                    lease.graph_id,
                    lease.node_id,
                )
                current_result = await self._decode(
                    current_result_record,
                    TaskResultRecord,
                )
                if (
                    current_result.graph_id != lease.graph_id
                    or current_result.node_id != lease.node_id
                    or current_result.result_digest != node.result_digest
                ):
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if node.status in _TERMINAL_TASK_STATUSES:
                if node.fence != lease.fence:
                    raise AIError(ErrorCode.TASK_FENCE_STALE)
                if node.status is not TaskStatus.SUCCEEDED:
                    raise AIError(ErrorCode.TASK_TERMINAL_CONFLICT)
                if node.result_digest != result_digest or (
                    execution_id is not None and node.execution_id != execution_id
                ):
                    raise AIError(ErrorCode.TASK_RESULT_CONFLICT)
                if result_payload is not None:
                    if current_result is None:
                        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                    if current_result.payload != result_payload:
                        raise AIError(ErrorCode.TASK_RESULT_CONFLICT)
                return TaskTerminalRecord(
                    lease.node_id,
                    lease.owner,
                    lease.fence,
                    TaskStatus.SUCCEEDED,
                    result_digest,
                    None,
                    None,
                    execution_id=node.execution_id,
                )
            now = await transaction.now()
            _require_live_task_lease(node, lease, now)
            resolved_execution_id = _resolve_task_execution_id(
                node.execution_id,
                execution_id,
            )
            if current_result is not None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            result_to_insert = (
                None
                if result_payload is None
                else TaskResultRecord(
                    lease.graph_id,
                    lease.node_id,
                    result_digest,
                    result_payload,
                )
            )
            value = replace(
                node,
                status=TaskStatus.SUCCEEDED,
                owner=None,
                lease_expires_at=None,
                result_digest=result_digest,
                error_code=None,
                error_digest=None,
                execution_id=resolved_execution_id,
            )
            if (
                await transaction.guard_record(
                    graph_key,
                    expected_storage_version=graph_record.storage_version,
                )
                is None
            ):
                raise _TaskEventAppendConflict()
            if result_to_insert is not None:
                await transaction.insert_record(
                    self._stored(
                        "task_result",
                        [lease.graph_id, lease.node_id],
                        result_to_insert,
                        scope=self._result_scope(lease.graph_id),
                        parent=self._result_parent(lease.graph_id),
                    )
                )
            await self._update_node_in_transaction(
                transaction, node, value, node_record
            )
            await _append_task_events(
                transaction,
                namespace=self._namespace,
                tenant_id=self._tenant_id,
                domain=self._domain.value,
                graph_id=lease.graph_id,
                graph_key=graph_key,
                drafts=_task_node_event_drafts(node, value),
                owner_guarded=True,
            )
            return TaskTerminalRecord(
                lease.node_id,
                lease.owner,
                lease.fence,
                TaskStatus.SUCCEEDED,
                result_digest,
                None,
                None,
                execution_id=resolved_execution_id,
            )

        try:
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
        _validate_task_lease_scope(lease, tenant_id)
        if tenant_id != self._tenant_id:
            raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)

        async def mutate(transaction: StateTransaction) -> TaskTerminalRecord:
            graph_key = self._graph_key(lease.graph_id)
            node_key = self._node_key(lease.graph_id, lease.node_id)
            records = await transaction.get_records((graph_key, node_key))
            graph_record = records.get(graph_key)
            node_record = records.get(node_key)
            if graph_record is None or node_record is None:
                raise AIError(ErrorCode.TASK_FENCE_STALE)
            self._validate_graph_record(graph_record, lease.graph_id)
            self._validate_node_record(node_record, lease.graph_id, lease.node_id)
            node = await self._decode(node_record, TaskNodeView)
            if node.status in _TERMINAL_TASK_STATUSES:
                if node.fence != lease.fence:
                    raise AIError(ErrorCode.TASK_FENCE_STALE)
                if node.status is not TaskStatus.FAILED:
                    raise AIError(ErrorCode.TASK_TERMINAL_CONFLICT)
                if (
                    node.error_code != error_code
                    or node.error_digest != error_digest
                    or (execution_id is not None and node.execution_id != execution_id)
                ):
                    raise AIError(ErrorCode.TASK_RESULT_CONFLICT)
                return TaskTerminalRecord(
                    lease.node_id,
                    lease.owner,
                    lease.fence,
                    TaskStatus.FAILED,
                    None,
                    error_code,
                    error_digest,
                    execution_id=node.execution_id,
                )
            now = await transaction.now()
            _require_live_task_lease(node, lease, now)
            resolved_execution_id = _resolve_task_execution_id(
                node.execution_id,
                execution_id,
            )
            value = replace(
                node,
                status=TaskStatus.FAILED,
                owner=None,
                lease_expires_at=None,
                result_digest=None,
                error_code=error_code,
                error_digest=error_digest,
                execution_id=resolved_execution_id,
            )
            if (
                await transaction.guard_record(
                    graph_key,
                    expected_storage_version=graph_record.storage_version,
                )
                is None
            ):
                raise _TaskEventAppendConflict()
            await self._update_node_in_transaction(
                transaction, node, value, node_record
            )
            await _append_task_events(
                transaction,
                namespace=self._namespace,
                tenant_id=self._tenant_id,
                domain=self._domain.value,
                graph_id=lease.graph_id,
                graph_key=graph_key,
                drafts=_task_node_event_drafts(node, value),
                owner_guarded=True,
            )
            return TaskTerminalRecord(
                lease.node_id,
                lease.owner,
                lease.fence,
                TaskStatus.FAILED,
                None,
                error_code,
                error_digest,
                execution_id=resolved_execution_id,
            )

        try:
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
                or (execution_id is not None and current.execution_id != execution_id)
            ):
                raise AIError(ErrorCode.TASK_RESULT_CONFLICT) from conflict
            if status is TaskStatus.SUCCEEDED:
                results = await self.get_results(
                    lease.graph_id,
                    (lease.node_id,),
                    tenant_id=tenant_id,
                )
                result = results.get(lease.node_id)
                if result is not None and result.result_digest != current.result_digest:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from conflict
                if result_payload is not None:
                    if result is None:
                        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from conflict
                    if result.payload != result_payload:
                        raise AIError(ErrorCode.TASK_RESULT_CONFLICT) from conflict
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
            self._validate_graph_record(graph_record, graph_id)
            graph = await self._decode(graph_record, TaskGraphView)
            if graph.graph_id != graph_id:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            _require_canonical_graph_status(graph.status)
            node_records = await transaction.list_records(
                RecordQuery(
                    parent_digest=self._parent("task_node", "graph", graph_id),
                    kind="task_node",
                )
            )
            states = await self._decode_many(node_records)
            by_id = {state.node_id: state for state in states}
            if len(by_id) != len(states):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            try:
                ordered = tuple(by_id[node.node_id] for node in graph.nodes)
            except KeyError as error:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
            if len(ordered) != len(states):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            status = _effective_graph_status(graph, ordered)
            snapshot = TaskGraphSnapshot(
                graph.graph_id,
                status,
                graph.nodes,
                ordered,
            )
            view = TaskGraphView(graph.graph_id, snapshot.status, graph.nodes)
            graph_converged = (
                graph.status is snapshot.status
                and graph_record.state == snapshot.status.value
            )
            admission_record = records.get(admission_key)
            if admission_record is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            self._validate_admission_record(admission_record, graph_id)
            admission = await self._decode(admission_record, TaskGraphAdmission)
            if admission.graph_id != graph_id:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            return (
                view,
                graph_converged and admission_record.state == snapshot.status.value,
            )

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
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        graph_record = records.get(graph_key)
        if graph_record is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        self._validate_graph_record(graph_record, view.graph_id)
        self._validate_admission_record(admission_record, view.graph_id)
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


class TaskAdmissionRepositoryImpl(RepositoryBase):
    def __init__(self, store: StateStore, *, namespace: str, tenant_id: str) -> None:
        super().__init__(
            store,
            namespace=namespace,
            tenant_id=tenant_id,
            domain=RuntimeDomain.TASK,
        )
        self._background_tasks: set[asyncio.Task[object]] = set()

    def _graph_key(self, graph_id: str) -> bytes:
        return self._key("task_graph", graph_id)

    def _admission_key(self, graph_id: str) -> bytes:
        return self._key("task_admission", graph_id)

    def _recovery_scope(self) -> bytes:
        return self._scope("task_admission", "recoverable", "graphs")

    def _validate_graph_record(self, record: StoredRecord, graph_id: str) -> None:
        if (
            record.kind != "task_graph"
            or record.key_digest != self._graph_key(graph_id)
            or record.partition_digest != self._partition("task_graph")
            or record.scope_digest is not None
            or record.parent_digest is not None
            or record.sort_key != sortable_identity(graph_id)
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

    def _validate_admission_record(
        self,
        record: StoredRecord,
        graph_id: str,
    ) -> None:
        if (
            record.kind != "task_admission"
            or record.key_digest != self._admission_key(graph_id)
            or record.partition_digest != self._partition("task_admission")
            or record.scope_digest != self._recovery_scope()
            or record.parent_digest is not None
            or record.sort_key != sortable_identity(graph_id)
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

    def _validate_node_record(
        self,
        record: StoredRecord,
        graph_id: str,
        node_id: str,
    ) -> None:
        if (
            record.kind != "task_node"
            or record.key_digest != self._key("task_node", [graph_id, node_id])
            or record.partition_digest != self._partition("task_node")
            or record.scope_digest is not None
            or record.parent_digest != self._parent("task_node", "graph", graph_id)
            or record.sort_key != sortable_identity([graph_id, node_id])
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

    async def admit(
        self,
        admission: TaskGraphAdmission,
        graph: TaskGraph,
    ) -> TaskGraphView:
        launch = admission.bind(graph)
        if launch.principal.tenant_id != self._tenant_id:
            raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)

        async def operation() -> TaskGraphView:
            return await self._store.mutate(
                lambda transaction: self._admit_in_transaction(
                    transaction,
                    admission,
                    graph,
                )
            )

        async def readback() -> CommitObservation[TaskGraphView]:
            return await self._store.read(
                lambda transaction: self._read_admission(
                    transaction,
                    admission,
                    graph,
                )
            )

        result = await run_durable_commit(
            operation,
            readback,
            background_tasks=self._background_tasks,
        )
        if result.state is DurableCommitState.COMMITTED and result.value is not None:
            if result.cancelled:
                raise asyncio.CancelledError
            return result.value
        if result.state is DurableCommitState.PARTIAL_INTEGRITY_ERROR:
            if isinstance(result.error, AIError):
                raise result.error
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from result.error
        if result.state is DurableCommitState.NOT_COMMITTED:
            if result.cancelled:
                raise asyncio.CancelledError
            if result.error is not None:
                raise result.error
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        if result.cancelled:
            raise asyncio.CancelledError
        raise AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN) from result.error

    async def list_recoverable_page(
        self,
        *,
        cursor: str | None,
        limit: int,
    ) -> Page[TaskGraphLaunch]:
        if limit != 128:
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        after_sort_key, after_key_digest = decode_record_cursor(cursor)

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
                graph_key = self._graph_key(admission.graph_id)
                graph_record = graph_records.get(graph_key)
                if graph_record is None:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                self._validate_admission_record(record, admission.graph_id)
                self._validate_graph_record(graph_record, admission.graph_id)
                graph_view = await self._decode(graph_record, TaskGraphView)
                graph = TaskGraph(graph_view.graph_id, graph_view.nodes)
                if (
                    record.state not in _RECOVERABLE_STATES
                    or record.state != graph_view.status.value
                    or graph_record.state != graph_view.status.value
                    or graph_view.graph_id != admission.graph_id
                ):
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                launches.append(admission.bind(graph))
            next_cursor = (
                record_cursor(selected[-1])
                if len(records) > limit and selected
                else None
            )
            return Page(tuple(launches), next_cursor)

        return await self.state_store.read(read)

    async def _admit_in_transaction(
        self,
        transaction: StateTransaction,
        admission: TaskGraphAdmission,
        graph: TaskGraph,
    ) -> TaskGraphView:
        graph_key = self._graph_key(graph.graph_id)
        admission_key = self._admission_key(graph.graph_id)
        operation_key_value = operation_key(
            self._namespace,
            self._tenant_id,
            self._domain.value,
            admission.operation_id,
        )
        records = await transaction.get_records((graph_key, admission_key))
        graph_record = records.get(graph_key)
        admission_record = records.get(admission_key)
        stored_operation = await transaction.get_operation(operation_key_value)
        node_records = await transaction.list_records(
            RecordQuery(
                parent_digest=self._parent("task_node", "graph", graph.graph_id),
                kind="task_node",
            )
        )
        if (
            graph_record is None
            and admission_record is None
            and stored_operation is None
        ):
            if node_records:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            return await self._create_admission(transaction, admission, graph)

        if (
            graph_record is not None
            and admission_record is not None
            and stored_operation is None
        ):
            if await self._is_canonical_occupied_admission(
                transaction,
                graph_record,
                admission_record,
                node_records,
                admission,
            ):
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

        if graph_record is None or admission_record is None or stored_operation is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

        operation = decode_operation(stored_operation)
        if operation.request_digest != admission.request_digest:
            raise AIError(ErrorCode.IDEMPOTENCY_CONFLICT)
        self._validate_operation_identity(operation, admission)
        existing, view = await self._require_committed_admission(
            transaction,
            graph_record,
            admission_record,
            node_records,
            stored_operation=stored_operation,
        )
        if existing != admission:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return await self._repair_aggregate_projection(
            transaction,
            graph_record,
            admission_record,
            view,
        )

    async def _create_admission(
        self,
        transaction: StateTransaction,
        admission: TaskGraphAdmission,
        graph: TaskGraph,
    ) -> TaskGraphView:
        view = await self._insert_admission_records(transaction, admission, graph)
        now = await transaction.now()
        operation_input = OperationLedgerInput(
            admission.operation_id,
            self._tenant_id,
            ResourceKind.TASK_GRAPH,
            graph.graph_id,
            None,
            OperationKind.TASK_NODE,
            OperationStatus.SUCCEEDED,
            admission.request_digest,
            graph.graph_id,
            _task_submit_result_digest(graph),
            None,
            False,
            now,
            now,
        )
        await append_operation(transaction, self, operation_input)
        _logger.info(
            "task graph durably admitted: tenant=%s graph=%s",
            self._tenant_id,
            graph.graph_id,
        )
        return view

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
            node_status = (
                TaskStatus.READY if not node.dependencies else TaskStatus.PENDING
            )
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
        await transaction.insert_records(tuple(records))
        await _append_task_state_events(
            transaction,
            namespace=self._namespace,
            tenant_id=self._tenant_id,
            domain=self._domain.value,
            graph_key=self._graph_key(graph.graph_id),
            before=None,
            after=_TaskEventState(view, tuple(node_views)),
        )
        return view

    async def _read_admission(
        self,
        transaction: StateTransaction,
        admission: TaskGraphAdmission,
        graph: TaskGraph,
    ) -> CommitObservation[TaskGraphView]:
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
        node_records = await transaction.list_records(
            RecordQuery(
                parent_digest=self._parent("task_node", "graph", graph.graph_id),
                kind="task_node",
            )
        )
        if (
            graph_record is None
            and admission_record is None
            and stored_operation is None
            and not node_records
        ):
            return CommitObservation(DurableCommitState.NOT_COMMITTED)
        try:
            if (
                graph_record is None
                or admission_record is None
                or stored_operation is None
            ):
                if (
                    graph_record is not None
                    and admission_record is not None
                    and await self._is_canonical_occupied_admission(
                        transaction,
                        graph_record,
                        admission_record,
                        node_records,
                        admission,
                    )
                ):
                    return CommitObservation(
                        DurableCommitState.NOT_COMMITTED,
                        error=AIError(ErrorCode.STORAGE_CONFLICT),
                    )
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            operation = decode_operation(stored_operation)
            if operation.request_digest != admission.request_digest:
                return CommitObservation(
                    DurableCommitState.NOT_COMMITTED,
                    error=AIError(ErrorCode.IDEMPOTENCY_CONFLICT),
                )
            self._validate_operation_identity(operation, admission)
            existing, view = await self._require_committed_admission(
                transaction,
                graph_record,
                admission_record,
                node_records,
                stored_operation=stored_operation,
            )
            if existing != admission:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            return CommitObservation(DurableCommitState.COMMITTED, view)
        except (KeyError, TypeError, ValueError):
            return CommitObservation(
                DurableCommitState.PARTIAL_INTEGRITY_ERROR,
                error=AIError(ErrorCode.STORAGE_INTEGRITY_ERROR),
            )
        except AIError as error:
            return CommitObservation(
                DurableCommitState.PARTIAL_INTEGRITY_ERROR,
                error=error,
            )

    async def _is_canonical_occupied_admission(
        self,
        transaction: StateTransaction,
        graph_record: StoredRecord,
        admission_record: StoredRecord,
        node_records: tuple[StoredRecord, ...],
        candidate: TaskGraphAdmission,
    ) -> bool:
        existing = await self._decode(admission_record, TaskGraphAdmission)
        if existing.operation_id == candidate.operation_id:
            return False
        stored_operation = await transaction.get_operation(
            operation_key(
                self._namespace,
                self._tenant_id,
                self._domain.value,
                existing.operation_id,
            )
        )
        if stored_operation is None:
            return False
        await self._require_committed_admission(
            transaction,
            graph_record,
            admission_record,
            node_records,
            stored_operation=stored_operation,
        )
        return True

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
        self._validate_graph_record(graph_record, graph_view.graph_id)
        self._validate_admission_record(admission_record, graph_view.graph_id)
        _require_canonical_graph_status(graph_view.status)
        nodes = await self._decode_task_nodes(node_records, graph_view.graph_id)
        persisted_graph = TaskGraph(graph_view.graph_id, graph_view.nodes)
        existing.bind(persisted_graph)
        self._validate_graph(graph_view, persisted_graph, nodes)
        status = _effective_graph_status(graph_view, nodes)
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
        operation = decode_operation(stored_operation)
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
    ) -> TaskGraphView:
        graph_view = await self._decode(graph_record, TaskGraphView)
        if (
            graph_view.status is view.status
            and graph_record.state == view.status.value
            and admission_record.state == view.status.value
        ):
            return view
        graph_record = await _guard_task_event_owner(
            transaction,
            self._graph_key(view.graph_id),
        )
        if (
            graph_view.status is not view.status
            or graph_record.state != view.status.value
        ):
            await replace_checked(
                transaction,
                projected_record(
                    self,
                    graph_record,
                    replace(graph_view, status=view.status),
                ),
                graph_record.storage_version,
            )
        if admission_record.state != view.status.value:
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
        await _append_task_events(
            transaction,
            namespace=self._namespace,
            tenant_id=self._tenant_id,
            domain=self._domain.value,
            graph_id=view.graph_id,
            graph_key=self._graph_key(view.graph_id),
            drafts=_task_graph_event_drafts(graph_view, view),
            owner_guarded=True,
        )
        return view

    def _validate_succeeded_operation(
        self,
        operation: OperationLedgerRecord,
        graph: TaskGraph,
    ) -> None:
        if (
            operation.status is not OperationStatus.SUCCEEDED
            or operation.result_ref != graph.graph_id
            or operation.result_digest != _task_submit_result_digest(graph)
            or operation.error_code is not None
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

    def _validate_operation_identity(
        self,
        operation: OperationLedgerRecord,
        admission: TaskGraphAdmission,
    ) -> None:
        if (
            operation.operation_id != admission.operation_id
            or operation.tenant_id != self._tenant_id
            or operation.resource_kind is not ResourceKind.TASK_GRAPH
            or operation.resource_id != admission.graph_id
            or operation.execution_id is not None
            or operation.operation_kind is not OperationKind.TASK_NODE
            or operation.compactable
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

    def _validate_graph(
        self,
        stored: TaskGraphView,
        graph: TaskGraph,
        nodes: tuple[TaskNodeView, ...],
    ) -> None:
        if stored.graph_id != graph.graph_id or stored.nodes != graph.nodes:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        expected = {node.node_id: node.dependencies for node in graph.nodes}
        actual = {node.node_id: node.dependencies for node in nodes}
        if actual != expected or len(nodes) != len(expected):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

    async def _decode_task_nodes(
        self,
        records: tuple[StoredRecord, ...],
        graph_id: str,
    ) -> tuple[TaskNodeView, ...]:
        values: list[TaskNodeView] = []
        for record in records:
            value = await self._decode(record, TaskNodeView)
            if value.graph_id != graph_id:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            self._validate_node_record(record, graph_id, value.node_id)
            values.append(value)
        return tuple(values)


_RECOVERABLE_STATES = frozenset({TaskStatus.PENDING.value, TaskStatus.RUNNING.value})


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


__all__ = ["TaskAdmissionRepositoryImpl", "TaskRepositoryImpl"]
