#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Canonical Task persistence repositories and durable event history."""

import asyncio
import heapq
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import replace
from datetime import datetime, timedelta
from typing import TypeVar

from linktools.core import environ

from ...core import (
    Page,
    ResourceKind,
    ResourceRef,
    TaskStatus,
    validate_lease_owner,
    validate_lease_seconds,
)
from ...errors import AIError, ErrorCode
from ...storage import StoredPayload
from ...task import (
    TaskEvent,
    TaskGraph,
    TaskGraphAdmission,
    TaskGraphLimits,
    TaskGraphSnapshot,
    TaskGraphView,
    TaskLease,
    TaskNode,
    TaskNodeView,
    TaskResultRecord,
    TaskTerminalRecord,
)
from ._plan import RuntimeDomain
from ._repositories import (
    RepositoryBase,
    projected_record,
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
    StoredRecord,
    sortable_identity,
)

from ._task_events import (
    _TaskEventAppendConflict,
    _TaskEventDraft,
    _TaskEventState,
    _append_task_events,
    _decode_task_event,
    _guard_task_event_owner,
    _task_completion_event_drafts,
    _task_event_stream,
    _task_graph_event_drafts,
    _task_node_event_drafts,
)
from ._task_state import (
    _effective_graph_status,
    _is_sha256,
    _isolated_graph_status,
    _require_canonical_graph_status,
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
_RECOVERY_REQUIRED_CODES = frozenset(
    {
        ErrorCode.TASK_EFFECT_UNKNOWN.value,
        ErrorCode.TOOL_EFFECT_UNKNOWN.value,
        ErrorCode.STORAGE_COMMIT_UNKNOWN.value,
        ErrorCode.STORAGE_RECOVERY_REQUIRED.value,
        ErrorCode.EXECUTION_START_UNKNOWN.value,
    }
)




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
























def _reconciled_task_nodes(
    nodes: tuple[TaskNodeView, ...],
    *,
    now: datetime | None = None,
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
        if node.status is TaskStatus.RUNNING and (
            node.owner is None
            or node.fence < 1
            or node.lease_expires_at is None
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if (
            now is not None
            and node.status is TaskStatus.RUNNING
            and node.lease_expires_at is not None
            and node.lease_expires_at <= now
        ):
            node = replace(
                node,
                status=TaskStatus.READY,
                owner=None,
                lease_expires_at=None,
                next_attempt_at=None,
                occupies_concurrency=False,
            )
            values[node_id] = node
        if (
            node.status in _TERMINAL_TASK_STATUSES
            or node.status is TaskStatus.RECOVERY_REQUIRED
        ):
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


def _expansion_error(
    graph_id: str,
    source_node_id: str,
    reason: str,
    *,
    conflict: str | None = None,
) -> AIError:
    details: dict[str, object] = {
        "phase": "task_graph_expansion",
        "reason": reason,
        "graph_id": graph_id,
        "source_node_id": source_node_id,
    }
    if conflict is not None:
        details["conflict"] = conflict
    return AIError(ErrorCode.TASK_DAG_INVALID, safe_details=details)


def _validate_expansion(
    graph: TaskGraph,
    source_node_id: str,
    expanded_nodes: tuple[TaskNode, ...],
    limits: TaskGraphLimits,
) -> tuple[TaskNode, ...]:
    source = next(
        (node for node in graph.nodes if node.node_id == source_node_id),
        None,
    )
    if source is None:
        raise _expansion_error(graph.graph_id, source_node_id, "source_missing")
    if source.expander is None and expanded_nodes:
        raise _expansion_error(
            graph.graph_id,
            source_node_id,
            "source_has_no_expander",
        )
    existing_ids = {node.node_id for node in graph.nodes}
    expanded_ids = tuple(node.node_id for node in expanded_nodes)
    if len(set(expanded_ids)) != len(expanded_ids):
        raise _expansion_error(
            graph.graph_id,
            source_node_id,
            "duplicate_node_id",
        )
    conflict = next(
        (node_id for node_id in expanded_ids if node_id in existing_ids),
        None,
    )
    if conflict is not None:
        raise _expansion_error(
            graph.graph_id,
            source_node_id,
            "node_id_conflict",
            conflict=conflict,
        )
    all_ids = existing_ids.union(expanded_ids)
    for node in expanded_nodes:
        missing = next(
            (dependency for dependency in node.dependencies if dependency not in all_ids),
            None,
        )
        if missing is not None:
            raise _expansion_error(
                graph.graph_id,
                source_node_id,
                "dependency_unknown",
                conflict=missing,
            )
    ordered = tuple(sorted(expanded_nodes, key=lambda node: node.node_id))
    try:
        TaskGraph(graph.graph_id, (*graph.nodes, *ordered)).validate_limits(limits)
    except AIError as error:
        reason = str(error.safe_details.get("reason", "limits_exceeded"))
        raise _expansion_error(
            graph.graph_id,
            source_node_id,
            reason,
        ) from error
    return ordered












class TaskRepositoryImpl(RepositoryBase):
    def __init__(self, store: StateStore, *, namespace: str, tenant_id: str) -> None:
        super().__init__(
            store, namespace=namespace, tenant_id=tenant_id, domain=RuntimeDomain.TASK
        )

    def _graph_key(self, graph_id: str) -> bytes:
        return self._key("task_graph", graph_id)

    def _definition_key(self, graph_id: str, node_id: str) -> bytes:
        return self._key("task_node_definition", [graph_id, node_id])

    def _state_key(self, graph_id: str, node_id: str) -> bytes:
        return self._key("task_node_state", [graph_id, node_id])

    def _definition_parent(self, graph_id: str) -> bytes:
        return self._parent("task_node_definition", "graph", graph_id)

    def _state_parent(self, graph_id: str) -> bytes:
        return self._parent("task_node_state", "graph", graph_id)

    def _admission_key(self, graph_id: str) -> bytes:
        return self._key("task_admission", graph_id)

    def _recovery_scope(self) -> bytes:
        return self._scope("task_admission", "recoverable", "graphs")

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

    async def _graph_limits_in_transaction(
        self,
        transaction: StateTransaction,
        graph_id: str,
    ) -> TaskGraphLimits:
        record = await transaction.get_record(self._admission_key(graph_id))
        if record is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        self._validate_admission_record(record, graph_id)
        admission = await self._decode(record, TaskGraphAdmission)
        if admission.graph_id != graph_id:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return admission.limits

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

    def _validate_state_record(
        self,
        record: StoredRecord,
        graph_id: str,
        node_id: str,
    ) -> None:
        if (
            record.kind != "task_node_state"
            or record.key_digest != self._state_key(graph_id, node_id)
            or record.partition_digest != self._partition("task_node_state")
            or record.scope_digest is not None
            or record.parent_digest != self._state_parent(graph_id)
            or record.sort_key != sortable_identity([graph_id, node_id])
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

    def _validate_definition_record(
        self,
        record: StoredRecord,
        graph_id: str,
        node_id: str,
    ) -> None:
        if (
            record.kind != "task_node_definition"
            or record.key_digest != self._definition_key(graph_id, node_id)
            or record.partition_digest != self._partition("task_node_definition")
            or record.scope_digest is not None
            or record.parent_digest != self._definition_parent(graph_id)
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
                    self._state_key(lease.graph_id, lease.node_id),
                )
            )
            graph_record = records.get(self._graph_key(lease.graph_id))
            node_record = records.get(self._state_key(lease.graph_id, lease.node_id))
            if graph_record is None or node_record is None:
                raise AIError(ErrorCode.TASK_FENCE_STALE)
            self._validate_graph_record(graph_record, lease.graph_id)
            self._validate_state_record(node_record, lease.graph_id, lease.node_id)
            node = await self._node_in_transaction(
                transaction,
                lease.graph_id,
                lease.node_id,
                missing_code=ErrorCode.TASK_FENCE_STALE,
            )
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
        return await self.state_store.read(
            lambda transaction: self._nodes_in_transaction(transaction, graph_id)
        )

    async def _nodes_in_transaction(
        self,
        transaction: StateTransaction,
        graph_id: str,
    ) -> tuple[TaskNodeView, ...]:
        definition_records = await transaction.list_records(
            RecordQuery(
                parent_digest=self._definition_parent(graph_id),
                kind="task_node_definition",
            )
        )
        state_records = await transaction.list_records(
            RecordQuery(
                parent_digest=self._state_parent(graph_id),
                kind="task_node_state",
            )
        )
        definitions: dict[str, TaskNode] = {}
        for record in definition_records:
            value = await self._decode(record, TaskNode)
            self._validate_definition_record(record, graph_id, value.node_id)
            if value.node_id in definitions:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            definitions[value.node_id] = value
        states: dict[str, TaskNodeView] = {}
        for record in state_records:
            value = await self._decode(record, TaskNodeView)
            self._validate_state_record(record, graph_id, value.node_id)
            if value.node_id in states:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            states[value.node_id] = value
        if set(definitions) != set(states):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return tuple(
            replace(states[node_id], dependencies=definitions[node_id].dependencies)
            for node_id in sorted(definitions)
        )

    async def _node(self, graph_id: str, node_id: str, tenant_id: str) -> TaskNodeView:
        require_repository_tenant(tenant_id, self._tenant_id)
        return await self.state_store.read(
            lambda transaction: self._node_in_transaction(
                transaction,
                graph_id,
                node_id,
                missing_code=ErrorCode.STORAGE_NOT_FOUND,
            )
        )

    async def _update_node_in_transaction(
        self,
        transaction: StateTransaction,
        current: TaskNodeView,
        value: TaskNodeView,
        node_record: StoredRecord,
    ) -> None:
        self._validate_state_record(node_record, current.graph_id, current.node_id)
        stored_node = await self._decode(node_record, TaskNodeView)
        stored_node = replace(stored_node, dependencies=current.dependencies)
        if stored_node != current:
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        node_candidate = projected_record(self, node_record, value)
        await replace_checked(transaction, node_candidate, node_record.storage_version)

    async def _current_graph_in_transaction(
        self,
        transaction: StateTransaction,
        graph_id: str,
    ) -> tuple[TaskGraphView, tuple[TaskNodeView, ...]]:
        graph_record = await transaction.get_record(self._graph_key(graph_id))
        if graph_record is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        self._validate_graph_record(graph_record, graph_id)
        header = await self._decode(graph_record, TaskGraphView)
        if header.graph_id != graph_id:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        definition_records = await transaction.list_records(
            RecordQuery(
                parent_digest=self._definition_parent(graph_id),
                kind="task_node_definition",
            )
        )
        state_records = await transaction.list_records(
            RecordQuery(
                parent_digest=self._state_parent(graph_id),
                kind="task_node_state",
            )
        )
        definitions: dict[str, TaskNode] = {}
        for record in definition_records:
            value = await self._decode(record, TaskNode)
            self._validate_definition_record(record, graph_id, value.node_id)
            if value.node_id in definitions:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            definitions[value.node_id] = value
        states: dict[str, TaskNodeView] = {}
        for record in state_records:
            value = await self._decode(record, TaskNodeView)
            self._validate_state_record(record, graph_id, value.node_id)
            if value.node_id in states:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            states[value.node_id] = value
        if set(definitions) != set(states):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        nodes = tuple(definitions[node_id] for node_id in sorted(definitions))
        ordered_states = tuple(
            replace(states[node.node_id], dependencies=node.dependencies)
            for node in nodes
        )
        TaskGraph(graph_id, nodes)
        return (
            TaskGraphView(
                graph_id,
                _effective_graph_status(header, ordered_states),
                nodes,
            ),
            ordered_states,
        )

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
        header = await self._decode(graph_record, TaskGraphView)
        _require_canonical_graph_status(header.status)
        states = await self._nodes_in_transaction(transaction, graph_id)
        definition_records = await transaction.list_records(
            RecordQuery(
                parent_digest=self._definition_parent(graph_id),
                kind="task_node_definition",
            )
        )
        definitions: list[TaskNode] = []
        for record in definition_records:
            definition = await self._decode(record, TaskNode)
            self._validate_definition_record(record, graph_id, definition.node_id)
            definitions.append(definition)
        definitions.sort(key=lambda value: value.node_id)
        graph = TaskGraph(graph_id, tuple(definitions))
        if tuple(state.node_id for state in states) != tuple(
            node.node_id for node in graph.nodes
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return _TaskEventState(
            TaskGraphView(graph.graph_id, header.status, graph.nodes),
            states,
        )

    async def _snapshot_graph_in_transaction(
        self,
        transaction: StateTransaction,
        graph_id: str,
    ) -> TaskGraphSnapshot | None:
        state = await self._event_state_in_transaction(transaction, graph_id)
        if state is None:
            return None
        stream = _task_event_stream(
            self._namespace,
            self._tenant_id,
            self._domain.value,
            graph_id,
        )
        owner = self._graph_key(graph_id)
        facts = await transaction.list_facts(
            FactQuery(stream, latest=True, limit=1)
        )
        if not facts:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        latest = facts[0]
        if (
            latest.stream_digest != stream
            or latest.owner_key_digest != owner
            or latest.subject_digest is not None
            or latest.state is not None
            or latest.sequence < 1
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        _decode_task_event(graph_id, latest)
        return TaskGraphSnapshot(
            state.graph.graph_id,
            _effective_graph_status(state.graph, state.node_states),
            state.graph.nodes,
            state.node_states,
            latest.sequence,
        )

    async def _node_in_transaction(
        self,
        transaction: StateTransaction,
        graph_id: str,
        node_id: str,
        *,
        missing_code: ErrorCode,
    ) -> TaskNodeView:
        record = await transaction.get_record(self._state_key(graph_id, node_id))
        if record is None:
            raise AIError(missing_code)
        self._validate_state_record(record, graph_id, node_id)
        value = await self._decode(record, TaskNodeView)
        if value.graph_id != graph_id or value.node_id != node_id:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        definition_record = await transaction.get_record(
            self._definition_key(graph_id, node_id)
        )
        if definition_record is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        self._validate_definition_record(definition_record, graph_id, node_id)
        definition = await self._decode(definition_record, TaskNode)
        return replace(value, dependencies=definition.dependencies)

    def _task_node_record(
        self,
        current: StoredRecord,
        value: TaskNodeView,
    ) -> StoredRecord:
        self._validate_state_record(current, value.graph_id, value.node_id)
        candidate = self._stored(
            "task_node_state",
            [value.graph_id, value.node_id],
            value,
            parent=self._state_parent(value.graph_id),
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
                self._state_key(before.graph.graph_id, current.node_id)
                for current, _ in changes
            )
            node_records = await transaction.get_records(node_keys)
            if len(node_records) != len(node_keys):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            for current, value in changes:
                node_record = node_records.get(
                    self._state_key(before.graph.graph_id, current.node_id)
                )
                if node_record is None:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                self._validate_state_record(
                    node_record,
                    before.graph.graph_id,
                    current.node_id,
                )
                stored_node = await self._decode(node_record, TaskNodeView)
                stored_node = replace(
                    stored_node,
                    dependencies=current.dependencies,
                )
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
            node_key = self._state_key(lease.graph_id, lease.node_id)
            records = await transaction.get_records((graph_key, node_key))
            graph_record = records.get(graph_key)
            node_record = records.get(node_key)
            if graph_record is None or node_record is None:
                raise AIError(ErrorCode.TASK_FENCE_STALE)
            self._validate_graph_record(graph_record, lease.graph_id)
            self._validate_state_record(node_record, lease.graph_id, lease.node_id)
            node = await self._node_in_transaction(
                transaction,
                lease.graph_id,
                lease.node_id,
                missing_code=ErrorCode.TASK_FENCE_STALE,
            )
            now = await transaction.now()
            _require_live_task_lease(node, lease, now)
            if node.execution_id is not None:
                if node.execution_id == execution_id:
                    return node
                raise AIError(ErrorCode.TASK_RESULT_CONFLICT)
            value = replace(node, execution_id=execution_id)
            guarded_graph_record = await transaction.guard_record(
                graph_key,
                expected_storage_version=graph_record.storage_version,
            )
            if guarded_graph_record is None:
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

        return await self._mutate_with_event_retry(mutate)


    async def handoff_execution(
        self,
        lease: TaskLease,
        *,
        tenant_id: str,
        execution_id: str,
        occupies_concurrency: bool = True,
    ) -> TaskNodeView:
        _validate_task_lease_scope(lease, tenant_id)
        if tenant_id != self._tenant_id:
            raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)
        if not isinstance(execution_id, str) or not execution_id.strip():
            raise ValueError("execution_id must be a non-empty string")
        if not isinstance(occupies_concurrency, bool):
            raise TypeError("occupies_concurrency must be bool")

        async def mutate(transaction: StateTransaction) -> TaskNodeView:
            graph_key = self._graph_key(lease.graph_id)
            node_key = self._state_key(lease.graph_id, lease.node_id)
            records = await transaction.get_records((graph_key, node_key))
            graph_record = records.get(graph_key)
            node_record = records.get(node_key)
            if graph_record is None or node_record is None:
                raise AIError(ErrorCode.TASK_FENCE_STALE)
            self._validate_graph_record(graph_record, lease.graph_id)
            self._validate_state_record(node_record, lease.graph_id, lease.node_id)
            node = await self._node_in_transaction(
                transaction,
                lease.graph_id,
                lease.node_id,
                missing_code=ErrorCode.TASK_FENCE_STALE,
            )
            if node.status is TaskStatus.WAITING and node.execution_id == execution_id:
                return node
            now = await transaction.now()
            _require_live_task_lease(node, lease, now)
            if node.execution_id is not None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            value = replace(
                node,
                status=TaskStatus.WAITING,
                owner=None,
                lease_expires_at=None,
                execution_id=execution_id,
                result_digest=None,
                error_code=None,
                error_digest=None,
                next_attempt_at=None,
                occupies_concurrency=occupies_concurrency,
            )
            guarded_graph_record = await transaction.guard_record(
                graph_key,
                expected_storage_version=graph_record.storage_version,
            )
            if guarded_graph_record is None:
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
                        "phase": "task_execution_handoff",
                        "graph_id": lease.graph_id,
                        "node_id": lease.node_id,
                    },
                ) from error
            raise AIError(ErrorCode.STORAGE_CONFLICT) from error

    async def requeue_retry(
        self,
        lease: TaskLease,
        *,
        tenant_id: str,
        execution_id: str,
        next_attempt_at: datetime,
    ) -> TaskNodeView:
        _validate_task_lease_scope(lease, tenant_id)
        if tenant_id != self._tenant_id:
            raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)
        if (
            not isinstance(execution_id, str)
            or not execution_id.strip()
            or not isinstance(next_attempt_at, datetime)
            or next_attempt_at.tzinfo is None
        ):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)

        async def mutate(transaction: StateTransaction) -> TaskNodeView:
            before = await self._event_state_in_transaction(transaction, lease.graph_id)
            if before is None:
                raise AIError(ErrorCode.TASK_FENCE_STALE)
            graph_record = await transaction.get_record(self._graph_key(lease.graph_id))
            if graph_record is None:
                raise AIError(ErrorCode.TASK_FENCE_STALE)
            current = next(
                (node for node in before.node_states if node.node_id == lease.node_id),
                None,
            )
            if current is None:
                raise AIError(ErrorCode.TASK_FENCE_STALE)
            now = await transaction.now()
            _require_live_task_lease(current, lease, now)
            if current.execution_id != execution_id:
                raise AIError(ErrorCode.TASK_RESULT_CONFLICT)
            value = replace(
                current,
                status=TaskStatus.READY,
                owner=None,
                lease_expires_at=None,
                next_attempt_at=next_attempt_at,
                occupies_concurrency=False,
                result_digest=None,
                error_code=None,
                error_digest=None,
            )
            next_nodes = tuple(
                value if node.node_id == lease.node_id else node
                for node in before.node_states
            )
            await self._apply_graph_transition(
                transaction,
                before,
                graph_record,
                next_nodes,
                _isolated_graph_status(next_nodes),
            )
            return value

        return await self._mutate_with_event_retry(mutate)


    async def mark_recovery_required(
        self,
        lease: TaskLease | None,
        *,
        tenant_id: str,
        error_code: str,
        error_digest: str,
        execution_id: str | None = None,
        graph_id: str | None = None,
        node_id: str | None = None,
    ) -> TaskNodeView:
        if tenant_id != self._tenant_id:
            raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)
        if error_code not in _RECOVERY_REQUIRED_CODES or not _is_sha256(error_digest):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        if lease is None:
            if (
                not isinstance(graph_id, str)
                or not graph_id.strip()
                or not isinstance(node_id, str)
                or not node_id.strip()
                or not isinstance(execution_id, str)
                or not execution_id.strip()
            ):
                raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
            target_graph_id = graph_id
            target_node_id = node_id
        else:
            _validate_task_lease_scope(lease, tenant_id)
            if graph_id is not None and graph_id != lease.graph_id:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if node_id is not None and node_id != lease.node_id:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            target_graph_id = lease.graph_id
            target_node_id = lease.node_id

        async def mutate(transaction: StateTransaction) -> TaskNodeView:
            before = await self._event_state_in_transaction(transaction, target_graph_id)
            if before is None:
                raise AIError(ErrorCode.TASK_FENCE_STALE)
            graph_record = await transaction.get_record(self._graph_key(target_graph_id))
            if graph_record is None:
                raise AIError(ErrorCode.TASK_FENCE_STALE)
            current = next(
                (
                    node
                    for node in before.node_states
                    if node.node_id == target_node_id
                ),
                None,
            )
            if current is None:
                raise AIError(ErrorCode.TASK_FENCE_STALE)
            if current.status is TaskStatus.RECOVERY_REQUIRED:
                if lease is not None and current.fence != lease.fence:
                    raise AIError(ErrorCode.TASK_FENCE_STALE)
                resolved_execution_id = _resolve_task_execution_id(
                    current.execution_id,
                    execution_id,
                )
                if (
                    current.error_code == error_code
                    and current.error_digest == error_digest
                    and current.execution_id == resolved_execution_id
                ):
                    return current
                raise AIError(ErrorCode.TASK_RESULT_CONFLICT)
            now = await transaction.now()
            if lease is None:
                if (
                    current.status is not TaskStatus.WAITING
                    or current.execution_id != execution_id
                    or current.owner is not None
                    or current.lease_expires_at is not None
                ):
                    raise AIError(ErrorCode.TASK_FENCE_STALE)
            else:
                _require_live_task_lease(current, lease, now)
            resolved_execution_id = _resolve_task_execution_id(
                current.execution_id,
                execution_id,
            )
            value = replace(
                current,
                status=TaskStatus.RECOVERY_REQUIRED,
                owner=None,
                lease_expires_at=None,
                            next_attempt_at=None,
                            occupies_concurrency=False,
                            result_digest=None,
                error_code=error_code,
                error_digest=error_digest,
                execution_id=resolved_execution_id,
                next_attempt_at=None,
                occupies_concurrency=False,
            )
            next_nodes = tuple(
                value if node.node_id == target_node_id else node
                for node in before.node_states
            )
            await self._apply_graph_transition(
                transaction,
                before,
                graph_record,
                next_nodes,
                _isolated_graph_status(next_nodes),
            )
            return value

        try:
            return await self._mutate_with_event_retry(mutate)
        except AIError as error:
            if error.code not in _COMMIT_READBACK_CODES:
                raise
            current = await self._node(target_graph_id, target_node_id, tenant_id)
            view, converged = await self._projection_readback(
                target_graph_id,
                tenant_id=tenant_id,
            )
            if (
                converged
                and view.status is TaskStatus.RECOVERY_REQUIRED
                and current.status is TaskStatus.RECOVERY_REQUIRED
                and (lease is None or current.fence == lease.fence)
                and current.error_code == error_code
                and current.error_digest == error_digest
                and current.execution_id == execution_id
            ):
                return current
            if lease is not None and current.fence != lease.fence:
                raise AIError(ErrorCode.TASK_FENCE_STALE) from error
            if error.code is ErrorCode.STORAGE_COMMIT_UNKNOWN:
                raise AIError(
                    ErrorCode.STORAGE_RECOVERY_REQUIRED,
                    safe_details={
                        "phase": "task_recovery_required",
                        "graph_id": target_graph_id,
                        "node_id": target_node_id,
                    },
                ) from error
            raise AIError(ErrorCode.STORAGE_CONFLICT) from error

    async def recover_graph(
        self,
        graph_id: str,
        *,
        tenant_id: str,
        cancel_requested: bool = False,
    ) -> TaskGraphView:
        if tenant_id != self._tenant_id:
            raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)
        if not isinstance(cancel_requested, bool):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)

        async def mutate(transaction: StateTransaction) -> TaskGraphView:
            before = await self._event_state_in_transaction(transaction, graph_id)
            if before is None:
                raise AIError(ErrorCode.STORAGE_NOT_FOUND)
            if (
                _effective_graph_status(before.graph, before.node_states)
                is not TaskStatus.RECOVERY_REQUIRED
            ):
                raise AIError(ErrorCode.TASK_NOT_READY)
            graph_record = await transaction.get_record(self._graph_key(graph_id))
            if graph_record is None:
                raise AIError(ErrorCode.STORAGE_NOT_FOUND)
            if cancel_requested:
                next_nodes = tuple(
                    (
                        replace(
                            node,
                            status=TaskStatus.CANCELLED,
                            owner=None,
                            lease_expires_at=None,
                            next_attempt_at=None,
                            occupies_concurrency=False,
                            result_digest=None,
                            error_code=None,
                            error_digest=None,
                        )
                        if node.status not in _TERMINAL_TASK_STATUSES
                        else node
                    )
                    for node in before.node_states
                )
                next_status = TaskStatus.CANCELLED
            else:
                cleared = tuple(
                    (
                        replace(
                            node,
                            status=(
                                TaskStatus.WAITING
                                if node.execution_id is not None
                                else TaskStatus.PENDING
                            ),
                            owner=None,
                            lease_expires_at=None,
                            next_attempt_at=None,
                            occupies_concurrency=False,
                            result_digest=None,
                            error_code=None,
                            error_digest=None,
                        )
                        if node.status is TaskStatus.RECOVERY_REQUIRED
                        else node
                    )
                    for node in before.node_states
                )
                next_nodes = _reconciled_task_nodes(cleared)
                next_status = _isolated_graph_status(next_nodes)
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
            if converged and (
                view.status is TaskStatus.CANCELLED
                if cancel_requested
                else view.status is not TaskStatus.RECOVERY_REQUIRED
            ):
                return view
            if error.code is ErrorCode.STORAGE_COMMIT_UNKNOWN:
                raise AIError(
                    ErrorCode.STORAGE_RECOVERY_REQUIRED,
                    safe_details={
                        "phase": "task_recover",
                        "graph_id": graph_id,
                    },
                ) from error
            raise AIError(ErrorCode.STORAGE_CONFLICT) from error

    async def requeue_recovery(
        self,
        graph_id: str,
        node_id: str,
        *,
        tenant_id: str,
        expected_fence: int,
    ) -> TaskGraphView:
        if tenant_id != self._tenant_id:
            raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)
        if (
            isinstance(expected_fence, bool)
            or not isinstance(expected_fence, int)
            or expected_fence < 1
        ):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)

        async def mutate(transaction: StateTransaction) -> TaskGraphView:
            before = await self._event_state_in_transaction(transaction, graph_id)
            if before is None:
                raise AIError(ErrorCode.STORAGE_NOT_FOUND)
            graph_record = await transaction.get_record(self._graph_key(graph_id))
            if graph_record is None:
                raise AIError(ErrorCode.STORAGE_NOT_FOUND)
            selected = next(
                (value for value in before.node_states if value.node_id == node_id),
                None,
            )
            if selected is None:
                raise AIError(ErrorCode.STORAGE_NOT_FOUND)
            if (
                selected.status is not TaskStatus.RECOVERY_REQUIRED
                or selected.fence != expected_fence
                or selected.execution_id is None
            ):
                raise AIError(ErrorCode.TASK_FENCE_STALE)
            next_nodes = tuple(
                replace(
                    value,
                    status=TaskStatus.READY,
                    owner=None,
                    lease_expires_at=None,
                    execution_id=None,
                    result_digest=None,
                    error_code=None,
                    error_digest=None,
                )
                if value.node_id == node_id
                else value
                for value in before.node_states
            )
            next_nodes = _reconciled_task_nodes(next_nodes)
            return await self._apply_graph_transition(
                transaction,
                before,
                graph_record,
                next_nodes,
                _isolated_graph_status(next_nodes),
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
                current = next(
                    (
                        value
                        for value in await self.list_nodes(
                            graph_id,
                            tenant_id=tenant_id,
                        )
                        if value.node_id == node_id
                    ),
                    None,
                )
                if current is not None and current.status is TaskStatus.READY:
                    return view
            if error.code is ErrorCode.STORAGE_COMMIT_UNKNOWN:
                raise AIError(
                    ErrorCode.STORAGE_RECOVERY_REQUIRED,
                    safe_details={
                        "phase": "task_requeue_recovery",
                        "graph_id": graph_id,
                        "node_id": node_id,
                    },
                ) from error
            raise AIError(ErrorCode.STORAGE_CONFLICT) from error

    async def scheduler_snapshot(self, graph_id: str, *, tenant_id: str) -> TaskGraphSnapshot:
        if tenant_id != self._tenant_id:
            raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)

        async def mutate(transaction: StateTransaction) -> TaskGraphSnapshot:
            before = await self._event_state_in_transaction(transaction, graph_id)
            if before is None:
                raise AIError(ErrorCode.STORAGE_NOT_FOUND)
            graph_record = await transaction.get_record(self._graph_key(graph_id))
            if graph_record is None:
                raise AIError(ErrorCode.STORAGE_NOT_FOUND)
            now = await transaction.now()
            next_nodes = _reconciled_task_nodes(before.node_states, now=now)
            isolated = _isolated_graph_status(next_nodes)
            next_status = (
                TaskStatus.CANCELLED
                if before.graph.status is TaskStatus.CANCELLED
                and isolated is not TaskStatus.RECOVERY_REQUIRED
                else isolated
            )
            await self._apply_graph_transition(
                transaction,
                before,
                graph_record,
                next_nodes,
                next_status,
            )
            return TaskGraphSnapshot(
                graph_id,
                next_status,
                before.graph.nodes,
                next_nodes,
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
                return TaskGraphSnapshot(
                    view.graph_id,
                    view.status,
                    view.nodes,
                    await self.list_nodes(graph_id, tenant_id=tenant_id),
                )
            if error.code is ErrorCode.STORAGE_COMMIT_UNKNOWN:
                raise AIError(
                    ErrorCode.STORAGE_RECOVERY_REQUIRED,
                    safe_details={
                        "phase": "task_reconcile",
                        "graph_id": graph_id,
                    },
                ) from error
            raise AIError(ErrorCode.STORAGE_CONFLICT) from error

    async def cancel_node(
        self,
        graph_id: str,
        node_id: str,
        *,
        tenant_id: str,
        execution_id: str,
    ) -> TaskGraphView:
        if tenant_id != self._tenant_id:
            raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)
        if not isinstance(node_id, str) or not node_id.strip():
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        if not isinstance(execution_id, str) or not execution_id.strip():
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)

        async def mutate(transaction: StateTransaction) -> TaskGraphView:
            before = await self._event_state_in_transaction(transaction, graph_id)
            if before is None:
                raise AIError(ErrorCode.STORAGE_NOT_FOUND)
            graph_record = await transaction.get_record(self._graph_key(graph_id))
            if graph_record is None:
                raise AIError(ErrorCode.STORAGE_NOT_FOUND)
            definitions = {node.node_id: node for node in before.graph.nodes}
            definition = definitions.get(node_id)
            current = next(
                (node for node in before.node_states if node.node_id == node_id),
                None,
            )
            if definition is None or current is None:
                raise AIError(ErrorCode.STORAGE_NOT_FOUND)
            if current.execution_id != execution_id:
                raise AIError(ErrorCode.TASK_RESULT_CONFLICT)
            if current.status in _TERMINAL_TASK_STATUSES:
                return before.graph
            if current.status is TaskStatus.RECOVERY_REQUIRED:
                return before.graph

            if (
                current.status is TaskStatus.RUNNING
                and definition.effect == "non_replay_safe"
            ):
                value = replace(
                    current,
                    status=TaskStatus.RECOVERY_REQUIRED,
                    owner=None,
                    lease_expires_at=None,
                    next_attempt_at=None,
                    occupies_concurrency=False,
                    result_digest=None,
                    error_code=ErrorCode.TASK_EFFECT_UNKNOWN.value,
                    error_digest=canonical_sha256(
                        {
                            "graph_id": graph_id,
                            "node_id": node_id,
                            "code": ErrorCode.TASK_EFFECT_UNKNOWN.value,
                        }
                    ),
                )
                next_nodes = tuple(
                    value if node.node_id == node_id else node
                    for node in before.node_states
                )
            else:
                value = replace(
                    current,
                    status=TaskStatus.CANCELLED,
                    owner=None,
                    lease_expires_at=None,
                    next_attempt_at=None,
                    occupies_concurrency=False,
                    result_digest=None,
                    error_code=None,
                    error_digest=None,
                )
                next_nodes = _reconciled_task_nodes(
                    tuple(
                        value if node.node_id == node_id else node
                        for node in before.node_states
                    )
                )
            return await self._apply_graph_transition(
                transaction,
                before,
                graph_record,
                next_nodes,
                _isolated_graph_status(next_nodes),
            )

        try:
            return await self._mutate_with_event_retry(mutate)
        except AIError as error:
            if error.code not in _COMMIT_READBACK_CODES:
                raise
            snapshot = await self.snapshot_graph(graph_id, tenant_id=tenant_id)
            if snapshot is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
            current = next(
                (node for node in snapshot.node_states if node.node_id == node_id),
                None,
            )
            if current is None or current.execution_id != execution_id:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
            if current.status in {
                TaskStatus.CANCELLED,
                TaskStatus.RECOVERY_REQUIRED,
                TaskStatus.SUCCEEDED,
                TaskStatus.FAILED,
            }:
                return TaskGraphView(
                    snapshot.graph_id,
                    snapshot.status,
                    snapshot.nodes,
                )
            if error.code is ErrorCode.STORAGE_COMMIT_UNKNOWN:
                raise AIError(
                    ErrorCode.STORAGE_RECOVERY_REQUIRED,
                    safe_details={
                        "phase": "task_node_cancel",
                        "graph_id": graph_id,
                        "node_id": node_id,
                    },
                ) from error
            raise AIError(ErrorCode.STORAGE_CONFLICT) from error

    async def cancel_graph(self, graph_id: str, *, tenant_id: str) -> TaskGraphView:
        if tenant_id != self._tenant_id:
            raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)

        async def mutate(transaction: StateTransaction) -> TaskGraphView:
            before = await self._event_state_in_transaction(transaction, graph_id)
            if before is None:
                raise AIError(ErrorCode.STORAGE_NOT_FOUND)
            graph_record = await transaction.get_record(self._graph_key(graph_id))
            if graph_record is None:
                raise AIError(ErrorCode.STORAGE_NOT_FOUND)
            definitions = {
                node.node_id: node for node in before.graph.nodes
            }
            next_values: list[TaskNodeView] = []
            for node in before.node_states:
                definition = definitions.get(node.node_id)
                if definition is None:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                if node.status is TaskStatus.RECOVERY_REQUIRED:
                    next_values.append(node)
                    continue
                if node.status in _TERMINAL_TASK_STATUSES:
                    next_values.append(node)
                    continue
                if (
                    node.status is TaskStatus.RUNNING
                    and node.execution_id is not None
                    and definition.effect == "non_replay_safe"
                ):
                    next_values.append(
                        replace(
                            node,
                            status=TaskStatus.RECOVERY_REQUIRED,
                            owner=None,
                            lease_expires_at=None,
                            next_attempt_at=None,
                            occupies_concurrency=False,
                            result_digest=None,
                            error_code=ErrorCode.TASK_EFFECT_UNKNOWN.value,
                            error_digest=canonical_sha256(
                                {
                                    "graph_id": graph_id,
                                    "node_id": node.node_id,
                                    "code": ErrorCode.TASK_EFFECT_UNKNOWN.value,
                                }
                            ),
                        )
                    )
                    continue
                next_values.append(
                    replace(
                        node,
                        status=TaskStatus.CANCELLED,
                        owner=None,
                        lease_expires_at=None,
                        next_attempt_at=None,
                        occupies_concurrency=False,
                    )
                )
            next_nodes = tuple(next_values)
            isolated = _isolated_graph_status(next_nodes)
            next_status = (
                TaskStatus.RECOVERY_REQUIRED
                if isolated is TaskStatus.RECOVERY_REQUIRED
                else TaskStatus.CANCELLED
                if before.graph.status is TaskStatus.CANCELLED
                or next_nodes != before.node_states
                else isolated
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
            if converged and view.status in {
                TaskStatus.CANCELLED,
                TaskStatus.RECOVERY_REQUIRED,
            }:
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
            node_key = self._state_key(graph_id, node_id)
            records = await transaction.get_records((graph_key, node_key))
            graph_record = records.get(graph_key)
            node_record = records.get(node_key)
            if graph_record is None or node_record is None:
                raise AIError(ErrorCode.TASK_NOT_READY)
            self._validate_graph_record(graph_record, graph_id)
            self._validate_state_record(node_record, graph_id, node_id)
            event_state = await self._event_state_in_transaction(transaction, graph_id)
            if event_state is None:
                raise AIError(ErrorCode.TASK_NOT_READY)
            graph = event_state.graph
            _require_canonical_graph_status(graph.status)
            if graph.status is TaskStatus.RECOVERY_REQUIRED:
                raise AIError(ErrorCode.TASK_NOT_READY)
            node = next(
                (
                    value
                    for value in event_state.node_states
                    if value.node_id == node_id
                ),
                None,
            )
            if node is None:
                raise AIError(ErrorCode.TASK_NOT_READY)
            if (
                graph.graph_id != graph_id
                or node.graph_id != graph_id
                or node.node_id != node_id
            ):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            dependencies = {
                value.node_id: value for value in event_state.node_states
            }
            now = await transaction.now()
            limits = await self._graph_limits_in_transaction(transaction, graph_id)
            active = 0
            for current in event_state.node_states:
                if (
                    current.status is TaskStatus.WAITING
                    and current.occupies_concurrency
                ):
                    active += 1
                    continue
                if current.status is not TaskStatus.RUNNING:
                    continue
                if (
                    current.owner is None
                    or current.fence < 1
                    or current.lease_expires_at is None
                ):
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                if current.lease_expires_at > now:
                    active += 1
            if active >= limits.max_concurrency:
                raise AIError(ErrorCode.TASK_NOT_READY)
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
                node.status is TaskStatus.READY
                and node.next_attempt_at is not None
                and node.next_attempt_at > now
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
                next_attempt_at=None,
                occupies_concurrency=False,
            )
            guarded_graph_record = await transaction.guard_record(
                graph_key,
                expected_storage_version=graph_record.storage_version,
            )
            if guarded_graph_record is None:
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
                value.execution_id,
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
                    current.execution_id,
                )
            if current.status in _TERMINAL_TASK_STATUSES or current.status is TaskStatus.RECOVERY_REQUIRED:
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
        lease: TaskLease | None,
        *,
        tenant_id: str,
        execution_id: str | None,
        result_digest: str,
        result_payload: StoredPayload | None = None,
        graph_id: str | None = None,
        node_id: str | None = None,
        expanded_nodes: tuple[TaskNode, ...] = (),
        expected_fence: int | None = None,
    ) -> TaskTerminalRecord:
        if tenant_id != self._tenant_id:
            raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)
        if not _is_sha256(result_digest):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        if result_payload is None or result_payload.digest != result_digest:
            raise AIError(ErrorCode.TASK_RESULT_CONFLICT)
        if lease is None:
            if (
                not isinstance(graph_id, str)
                or not graph_id.strip()
                or not isinstance(node_id, str)
                or not node_id.strip()
            ):
                raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
            target_graph_id = graph_id
            target_node_id = node_id
        else:
            _validate_task_lease_scope(lease, tenant_id)
            if graph_id is not None and graph_id != lease.graph_id:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if node_id is not None and node_id != lease.node_id:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            target_graph_id = lease.graph_id
            target_node_id = lease.node_id
        try:
            expanded = tuple(expanded_nodes)
        except TypeError as error:
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID) from error
        if any(not isinstance(node, TaskNode) for node in expanded):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)

        async def mutate(transaction: StateTransaction) -> TaskTerminalRecord:
            graph_key = self._graph_key(target_graph_id)
            node_key = self._state_key(target_graph_id, target_node_id)
            result_key = self._result_key(target_graph_id, target_node_id)
            records = await transaction.get_records((graph_key, node_key, result_key))
            graph_record = records.get(graph_key)
            node_record = records.get(node_key)
            if graph_record is None or node_record is None:
                raise AIError(ErrorCode.TASK_FENCE_STALE)
            self._validate_graph_record(graph_record, target_graph_id)
            self._validate_state_record(node_record, target_graph_id, target_node_id)
            before = await self._event_state_in_transaction(
                transaction,
                target_graph_id,
            )
            if before is None:
                raise AIError(ErrorCode.TASK_FENCE_STALE)
            node = next(
                (
                    value
                    for value in before.node_states
                    if value.node_id == target_node_id
                ),
                None,
            )
            if node is None:
                raise AIError(ErrorCode.TASK_FENCE_STALE)
            current_result_record = records.get(result_key)
            current_result: TaskResultRecord | None = None
            if current_result_record is not None:
                self._validate_result_record(
                    current_result_record,
                    target_graph_id,
                    target_node_id,
                )
                current_result = await self._decode(
                    current_result_record,
                    TaskResultRecord,
                )
                if (
                    current_result.graph_id != target_graph_id
                    or current_result.node_id != target_node_id
                    or current_result.result_digest != node.result_digest
                    or (
                        current_result.execution_id is not None
                        and current_result.execution_id != node.execution_id
                    )
                    or (
                        current_result.payload is not None
                        and current_result.payload.digest != node.result_digest
                    )
                ):
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if node.status in _TERMINAL_TASK_STATUSES:
                if lease is not None and node.fence != lease.fence:
                    raise AIError(ErrorCode.TASK_FENCE_STALE)
                if node.status is not TaskStatus.SUCCEEDED:
                    raise AIError(ErrorCode.TASK_TERMINAL_CONFLICT)
                if node.result_digest != result_digest or (
                    execution_id is not None and node.execution_id != execution_id
                ):
                    raise AIError(ErrorCode.TASK_RESULT_CONFLICT)
                if current_result is None:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                if (
                    current_result.payload is not None
                    and current_result.payload != result_payload
                ):
                    raise AIError(ErrorCode.TASK_RESULT_CONFLICT)
                if (
                    current_result.execution_id is not None
                    and current_result.execution_id != node.execution_id
                ):
                    raise AIError(ErrorCode.TASK_RESULT_CONFLICT)
                if expanded:
                    source = next(
                        value
                        for value in before.graph.nodes
                        if value.node_id == target_node_id
                    )
                    if source.expander is None:
                        raise _expansion_error(
                            target_graph_id,
                            target_node_id,
                            "source_has_no_expander",
                        )
                return TaskTerminalRecord(
                    target_node_id,
                    None if lease is None else lease.owner,
                    node.fence,
                    TaskStatus.SUCCEEDED,
                    result_digest,
                    None,
                    None,
                    execution_id=node.execution_id,
                )
            now = await transaction.now()
            if lease is None:
                if node.status is TaskStatus.WAITING:
                    if (
                        node.execution_id is None
                        or execution_id != node.execution_id
                        or node.owner is not None
                        or node.lease_expires_at is not None
                    ):
                        raise AIError(ErrorCode.TASK_FENCE_STALE)
                elif node.status is TaskStatus.RECOVERY_REQUIRED:
                    if (
                        expected_fence != node.fence
                        or node.execution_id is None
                        or execution_id != node.execution_id
                    ):
                        raise AIError(ErrorCode.TASK_FENCE_STALE)
                else:
                    raise AIError(ErrorCode.TASK_FENCE_STALE)
                resolved_execution_id = node.execution_id
            else:
                _require_live_task_lease(node, lease, now)
                resolved_execution_id = _resolve_task_execution_id(
                    node.execution_id,
                    execution_id,
                )
            limits = await self._graph_limits_in_transaction(
                transaction,
                target_graph_id,
            )
            added = _validate_expansion(
                TaskGraph(target_graph_id, before.graph.nodes),
                target_node_id,
                expanded,
                limits,
            )
            next_nodes = tuple(
                sorted((*before.graph.nodes, *added), key=lambda value: value.node_id)
            )
            state_by_id = {
                value.node_id: value for value in before.node_states
            }
            source_value = replace(
                node,
                status=TaskStatus.SUCCEEDED,
                owner=None,
                lease_expires_at=None,
                result_digest=result_digest,
                error_code=None,
                error_digest=None,
                execution_id=resolved_execution_id,
                next_attempt_at=None,
                occupies_concurrency=False,
            )
            state_by_id[target_node_id] = source_value
            new_states: list[TaskNodeView] = []
            for value in added:
                new_states.append(
                    TaskNodeView(
                        target_graph_id,
                        value.node_id,
                        value.dependencies,
                        TaskStatus.READY
                        if not value.dependencies
                        else TaskStatus.PENDING,
                        None,
                        0,
                        None,
                        None,
                        None,
                        None,
                    )
                )
            new_state_by_id = {value.node_id: value for value in new_states}
            state_by_id.update(new_state_by_id)
            after_states = tuple(state_by_id[value.node_id] for value in next_nodes)
            after_status = _isolated_graph_status(after_states)
            after = _TaskEventState(
                TaskGraphView(target_graph_id, after_status, next_nodes),
                after_states,
            )
            event_drafts = _task_completion_event_drafts(
                before,
                after,
                source_node_id=target_node_id,
                added_node_ids=tuple(value.node_id for value in added),
            )
            guarded_graph_record = await transaction.guard_record(
                graph_key,
                expected_storage_version=graph_record.storage_version,
            )
            if guarded_graph_record is None:
                raise _TaskEventAppendConflict()
            replacements: list[RecordReplacement] = []
            if (
                after_status is not before.graph.status
                or guarded_graph_record.state != after_status.value
            ):
                replacements.append(
                    RecordReplacement(
                        projected_record(
                            self,
                            guarded_graph_record,
                            replace(before.graph, status=after_status),
                        ),
                        guarded_graph_record.storage_version,
                    )
                )
            if current_result is not None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if resolved_execution_id is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            result_to_insert = TaskResultRecord(
                target_graph_id,
                target_node_id,
                result_digest,
                execution_id=resolved_execution_id,
                payload=result_payload,
            )
            if added:
                await transaction.insert_records(
                    tuple(
                        record
                        for value in added
                        for record in (
                            self._stored(
                                "task_node_definition",
                                [target_graph_id, value.node_id],
                                value,
                                parent=self._definition_parent(target_graph_id),
                            ),
                            self._stored(
                                "task_node_state",
                                [target_graph_id, value.node_id],
                                new_state_by_id[value.node_id],
                                parent=self._state_parent(target_graph_id),
                                state=new_state_by_id[value.node_id].status.value,
                            ),
                        )
                    )
                )
            await transaction.insert_record(
                self._stored(
                    "task_result",
                    [target_graph_id, target_node_id],
                    result_to_insert,
                    scope=self._result_scope(target_graph_id),
                    parent=self._result_parent(target_graph_id),
                )
            )
            await self._update_node_in_transaction(
                transaction, node, source_value, node_record
            )
            if replacements:
                await transaction.replace_records(tuple(replacements))
            await _append_task_events(
                transaction,
                namespace=self._namespace,
                tenant_id=self._tenant_id,
                domain=self._domain.value,
                graph_id=target_graph_id,
                graph_key=graph_key,
                drafts=event_drafts,
                owner_guarded=True,
            )
            return TaskTerminalRecord(
                target_node_id,
                None if lease is None else lease.owner,
                source_value.fence,
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
            if lease is None:
                current = await self._node(target_graph_id, target_node_id, tenant_id)
                if (
                    current.status is TaskStatus.SUCCEEDED
                    and current.result_digest == result_digest
                    and current.execution_id == execution_id
                ):
                    return TaskTerminalRecord(
                        target_node_id,
                        None,
                        current.fence,
                        TaskStatus.SUCCEEDED,
                        result_digest,
                        None,
                        None,
                        execution_id=current.execution_id,
                    )
                if error.code is ErrorCode.STORAGE_COMMIT_UNKNOWN:
                    raise AIError(
                        ErrorCode.STORAGE_RECOVERY_REQUIRED,
                        safe_details={
                            "phase": "task_terminal_commit",
                            "graph_id": target_graph_id,
                            "node_id": target_node_id,
                        },
                    ) from error
                raise AIError(ErrorCode.STORAGE_CONFLICT) from error
            return await self._classify_terminal_readback(
                lease,
                tenant_id=tenant_id,
                status=TaskStatus.SUCCEEDED,
                execution_id=execution_id,
                result_digest=result_digest,
                error_code=None,
                error_digest=None,
                conflict=error,
            )

    async def fail(
        self,
        lease: TaskLease | None,
        *,
        tenant_id: str,
        error_code: str,
        error_digest: str,
        execution_id: str | None = None,
        graph_id: str | None = None,
        node_id: str | None = None,
    ) -> TaskTerminalRecord:
        if tenant_id != self._tenant_id:
            raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)
        if lease is None:
            if (
                not isinstance(graph_id, str)
                or not graph_id.strip()
                or not isinstance(node_id, str)
                or not node_id.strip()
            ):
                raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
            target_graph_id = graph_id
            target_node_id = node_id
        else:
            _validate_task_lease_scope(lease, tenant_id)
            if graph_id is not None and graph_id != lease.graph_id:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if node_id is not None and node_id != lease.node_id:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            target_graph_id = lease.graph_id
            target_node_id = lease.node_id

        async def mutate(transaction: StateTransaction) -> TaskTerminalRecord:
            graph_key = self._graph_key(target_graph_id)
            node_key = self._state_key(target_graph_id, target_node_id)
            records = await transaction.get_records((graph_key, node_key))
            graph_record = records.get(graph_key)
            node_record = records.get(node_key)
            if graph_record is None or node_record is None:
                raise AIError(ErrorCode.TASK_FENCE_STALE)
            self._validate_graph_record(graph_record, target_graph_id)
            self._validate_state_record(node_record, target_graph_id, target_node_id)
            before = await self._event_state_in_transaction(
                transaction,
                target_graph_id,
            )
            if before is None:
                raise AIError(ErrorCode.TASK_FENCE_STALE)
            node = next(
                (
                    value
                    for value in before.node_states
                    if value.node_id == target_node_id
                ),
                None,
            )
            if node is None:
                raise AIError(ErrorCode.TASK_FENCE_STALE)
            if node.status in _TERMINAL_TASK_STATUSES:
                if lease is not None and node.fence != lease.fence:
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
                    target_node_id,
                    None if lease is None else lease.owner,
                    node.fence,
                    TaskStatus.FAILED,
                    None,
                    error_code,
                    error_digest,
                    execution_id=node.execution_id,
                )
            now = await transaction.now()
            if lease is None:
                if (
                    node.status is not TaskStatus.WAITING
                    or node.execution_id is None
                    or execution_id != node.execution_id
                    or node.owner is not None
                    or node.lease_expires_at is not None
                ):
                    raise AIError(ErrorCode.TASK_FENCE_STALE)
                resolved_execution_id = node.execution_id
            else:
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
                            next_attempt_at=None,
                            occupies_concurrency=False,
                            result_digest=None,
                error_code=error_code,
                error_digest=error_digest,
                execution_id=resolved_execution_id,
                next_attempt_at=None,
                occupies_concurrency=False,
            )
            guarded_graph_record = await transaction.guard_record(
                graph_key,
                expected_storage_version=graph_record.storage_version,
            )
            if guarded_graph_record is None:
                raise _TaskEventAppendConflict()
            after_states = tuple(
                value if state.node_id == target_node_id else state
                for state in before.node_states
            )
            after_status = _isolated_graph_status(after_states)
            after = _TaskEventState(
                TaskGraphView(target_graph_id, after_status, before.graph.nodes),
                after_states,
            )
            graph_replacement: RecordReplacement | None = None
            if (
                after_status is not before.graph.status
                or guarded_graph_record.state != after_status.value
            ):
                graph_replacement = RecordReplacement(
                    projected_record(
                        self,
                        guarded_graph_record,
                        replace(before.graph, status=after_status),
                    ),
                    guarded_graph_record.storage_version,
                )
            await self._update_node_in_transaction(
                transaction, node, value, node_record
            )
            if graph_replacement is not None:
                await transaction.replace_records((graph_replacement,))
            await _append_task_events(
                transaction,
                namespace=self._namespace,
                tenant_id=self._tenant_id,
                domain=self._domain.value,
                graph_id=target_graph_id,
                graph_key=graph_key,
                drafts=_task_completion_event_drafts(
                    before,
                    after,
                    source_node_id=target_node_id,
                    added_node_ids=(),
                ),
                owner_guarded=True,
            )
            return TaskTerminalRecord(
                target_node_id,
                None if lease is None else lease.owner,
                value.fence,
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
            if lease is None:
                current = await self._node(target_graph_id, target_node_id, tenant_id)
                if (
                    current.status is TaskStatus.FAILED
                    and current.execution_id == execution_id
                    and current.error_code == error_code
                    and current.error_digest == error_digest
                ):
                    return TaskTerminalRecord(
                        target_node_id,
                        None,
                        current.fence,
                        TaskStatus.FAILED,
                        None,
                        error_code,
                        error_digest,
                        execution_id=current.execution_id,
                    )
                if error.code is ErrorCode.STORAGE_COMMIT_UNKNOWN:
                    raise AIError(
                        ErrorCode.STORAGE_RECOVERY_REQUIRED,
                        safe_details={
                            "phase": "task_terminal_commit",
                            "graph_id": target_graph_id,
                            "node_id": target_node_id,
                        },
                    ) from error
                raise AIError(ErrorCode.STORAGE_CONFLICT) from error
            return await self._classify_terminal_readback(
                lease,
                tenant_id=tenant_id,
                status=TaskStatus.FAILED,
                execution_id=execution_id,
                result_digest=None,
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
                if result is None or result.execution_id != current.execution_id:
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
            TaskStatus.RECOVERY_REQUIRED,
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
            records = await transaction.get_records((graph_key,))
            graph_record = records.get(graph_key)
            if graph_record is None:
                raise AIError(ErrorCode.STORAGE_NOT_FOUND)
            self._validate_graph_record(graph_record, graph_id)
            header = await self._decode(graph_record, TaskGraphView)
            if header.graph_id != graph_id:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            _require_canonical_graph_status(header.status)
            state = await self._event_state_in_transaction(transaction, graph_id)
            if state is None:
                raise AIError(ErrorCode.STORAGE_NOT_FOUND)
            status = _effective_graph_status(state.graph, state.node_states)
            snapshot = TaskGraphSnapshot(
                state.graph.graph_id,
                status,
                state.graph.nodes,
                state.node_states,
            )
            view = TaskGraphView(
                state.graph.graph_id,
                snapshot.status,
                state.graph.nodes,
            )
            graph_converged = (
                header.status is snapshot.status
                and graph_record.state == snapshot.status.value
            )
            return view, graph_converged

        try:
            return await self.state_store.read(read)
        except (KeyError, ValueError) as error:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error








__all__ = ["TaskRepositoryImpl"]
