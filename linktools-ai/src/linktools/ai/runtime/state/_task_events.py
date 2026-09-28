#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Durable Task event projection helpers."""

from dataclasses import dataclass
from datetime import datetime

from ...core import TaskStatus
from ...errors import AIError, ErrorCode
from ...task import TaskEvent, TaskEventType, TaskGraphView, TaskNodeView
from ._store import (
    StateTransaction,
    StoredFact,
    StoredRecord,
    sequence_key,
    stream_digest,
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
    source_node_id: "str | None" = None
    added_node_ids: "tuple[str, ...]" = ()


@dataclass(frozen=True, slots=True)
class _TaskEventState:
    graph: TaskGraphView
    node_states: tuple[TaskNodeView, ...]


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
    if before.graph_id != after.graph_id:
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


def _task_completion_event_drafts(
    before: _TaskEventState,
    after: _TaskEventState,
    *,
    source_node_id: str,
    added_node_ids: tuple[str, ...],
) -> tuple[_TaskEventDraft, ...]:
    before_nodes = {node.node_id: node for node in before.node_states}
    after_nodes = {node.node_id: node for node in after.node_states}
    if set(before_nodes) - set(after_nodes):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    source_before = before_nodes.get(source_node_id)
    source_after = after_nodes.get(source_node_id)
    if source_before is None or source_after is None:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    values = list(_task_node_event_drafts(source_before, source_after))
    if added_node_ids:
        values.append(
            _TaskEventDraft(
                TaskEventType.GRAPH_EXPANDED,
                after.graph.status,
                source_node_id=source_node_id,
                added_node_ids=added_node_ids,
            )
        )
    for node_id in sorted(set(after_nodes) - set(before_nodes)):
        if node_id not in added_node_ids:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    for node_id in sorted(before_nodes):
        if node_id == source_node_id:
            continue
        values.extend(_task_node_event_drafts(before_nodes[node_id], after_nodes[node_id]))
    values.extend(_task_graph_event_drafts(before.graph, after.graph))
    return tuple(values)


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
        "source_node_id": draft.source_node_id,
        "added_node_ids": list(draft.added_node_ids),
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
            draft.source_node_id,
            draft.added_node_ids,
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
        source_node_id = data.get("source_node_id")
        added_node_ids = data.get("added_node_ids", [])
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
        if source_node_id is not None and not isinstance(source_node_id, str):
            raise TypeError("task event source node id is invalid")
        if not isinstance(added_node_ids, list) or any(
            not isinstance(node_id, str) for node_id in added_node_ids
        ):
            raise TypeError("task event added node ids are invalid")
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
            source_node_id,
            tuple(added_node_ids),
        )
    except AIError:
        raise
    except (KeyError, TypeError, ValueError) as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
