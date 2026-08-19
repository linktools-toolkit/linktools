#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Project Harness step facts into Runtime trace and transcript views."""

import re
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Protocol, runtime_checkable

from linktools.core import environ
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    SystemPromptPart,
    ThinkingPart,
    TextContent,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai_harness.step_persistence import RunRecord, StepEvent, StepStore

from ..core import (
    CursorPayload,
    CursorSigner,
    ExecutionStatus,
    JsonValue,
    Page,
    canonical_sha256,
    step_conversation_id,
    step_run_id,
    validate_persistence_namespace,
)
from ..errors import AIError, ErrorCode
from ..runtime.service_api import (
    ExecutionHistoryItem,
    ExecutionTraceItem,
    SessionHistoryItem,
    TranscriptItem,
)
from ..runtime.state import ExecutionRecord, ExecutionRepository

_logger = environ.get_logger("ai.adapter.history")


@dataclass(frozen=True, slots=True)
class _ProjectedHistoryItem:
    item_kind: str
    content: JsonValue
    tool_name: "str | None" = None
    tool_call_id: "str | None" = None


@runtime_checkable
class _SessionHistoryStore(Protocol):
    async def session_message_count(
        self,
        history_id: str,
        *,
        tenant_id: str,
    ) -> int: ...

    def iter_session_message_range(
        self,
        history_id: str,
        *,
        tenant_id: str,
        start: int,
        end: int,
    ) -> AsyncIterator[object]: ...

    async def iter_session_messages(
        self,
        history_id: str,
        *,
        tenant_id: str,
    ) -> AsyncIterator[object]: ...


class StepExecutionHistoryReader:
    """Own the adapter projection between StepStore facts and Runtime views."""

    def __init__(self, *, namespace: str, executions: ExecutionRepository, store: StepStore, cursor_signer: CursorSigner) -> None:
        try:
            validate_persistence_namespace(namespace)
        except AIError as error:
            raise ValueError("execution history namespace is invalid") from error
        self._namespace = namespace
        self._executions = executions
        self._store = store
        self._cursor_signer = cursor_signer

    async def trace(self, execution_id: str, *, tenant_id: str, cursor: "str | None", limit: int) -> "Page[ExecutionTraceItem]":
        record = await self._executions.get(execution_id, tenant_id=tenant_id)
        if record is None:
            raise AIError(ErrorCode.STORAGE_NOT_FOUND)
        if not 1 <= limit <= 200:
            raise AIError(ErrorCode.PAGE_LIMIT_INVALID)
        entries = await self._history_tree(record, tenant_id)
        projected: "list[tuple[tuple[object, ...], ExecutionTraceItem]]" = []
        for item, depth in entries:
            for segment_sequence, events in await self._segment_events(item, tenant_id):
                for ordinal, event in enumerate(events):
                    mapped = _trace_item(item, segment_sequence, depth, ordinal, event)
                    if mapped is not None:
                        projected.append(((_event_timestamp(event), depth, item.execution_id, segment_sequence, ordinal, mapped.payload.get("kind", "")), mapped))
        projected.sort(key=lambda value: value[0])
        values = [item for _, item in projected]
        start = _cursor_offset(cursor, len(values))
        selected = tuple(ExecutionTraceItem(item.execution_id, start + index + 1, item.payload) for index, item in enumerate(values[start:start + limit]))
        next_offset = start + len(selected)
        return Page(selected, str(next_offset) if next_offset < len(values) else None)

    async def history(self, execution_id: str, *, tenant_id: str, cursor: str | None, limit: int) -> "Page[ExecutionHistoryItem]":
        if not 1 <= limit <= 200:
            raise AIError(ErrorCode.PAGE_LIMIT_INVALID)
        record = await self._executions.get(execution_id, tenant_id=tenant_id)
        if record is None:
            raise AIError(ErrorCode.STORAGE_NOT_FOUND)
        entries = await self._history_tree(record, tenant_id)
        values: list[ExecutionHistoryItem] = []
        for item, _depth in entries:
            for segment_sequence, _events in await self._segment_events(item, tenant_id):
                run_id = step_run_id(namespace=self._namespace, tenant_id=tenant_id, execution_id=item.execution_id, segment_sequence=segment_sequence)
                snapshot = await self._store.latest_snapshot(run_id=run_id)
                if snapshot is None:
                    if item.status is ExecutionStatus.SUCCEEDED and segment_sequence == item.agent_run_sequence:
                        raise AIError(ErrorCode.EXECUTION_HISTORY_UNAVAILABLE)
                    continue
                if (
                    snapshot.run_id != run_id
                    or snapshot.conversation_id
                    != step_conversation_id(
                        namespace=self._namespace,
                        tenant_id=tenant_id,
                        execution_id=item.execution_id,
                    )
                ):
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                snapshot_items = [
                    ExecutionHistoryItem(
                        item.execution_id,
                        0,
                        projected.item_kind,
                        projected.content,
                        projected.tool_name,
                        projected.tool_call_id,
                    )
                    for message in snapshot.messages
                    for projected in _project_message(message)
                ]
                values = _merge_history_occurrences(values, snapshot_items)
        source_revision = canonical_sha256(
            {
                "execution_id": execution_id,
                "items": [
                    {"item_kind": item.item_kind, "content": item.content, "tool_name": item.tool_name, "tool_call_id": item.tool_call_id}
                    for item in values
                ],
            }
        )
        start = _history_cursor_offset(cursor, tenant_id, execution_id, source_revision, len(values), self._cursor_signer)
        selected = tuple(ExecutionHistoryItem(item.execution_id, start + index + 1, item.item_kind, item.content, item.tool_name, item.tool_call_id) for index, item in enumerate(values[start:start + limit]))
        next_offset = start + len(selected)
        next_cursor = _history_cursor(tenant_id, execution_id, source_revision, next_offset, self._cursor_signer) if next_offset < len(values) else None
        return Page(selected, next_cursor)

    async def transcript(self, execution_id: str, *, tenant_id: str, cursor: str | None, limit: int) -> Page[TranscriptItem]:
        if not 1 <= limit <= 200:
            raise AIError(ErrorCode.PAGE_LIMIT_INVALID)
        record = await self._executions.get(execution_id, tenant_id=tenant_id)
        if record is None:
            raise AIError(ErrorCode.STORAGE_NOT_FOUND)
        if record.agent_run_sequence == 0:
            if record.status is ExecutionStatus.SUCCEEDED:
                raise AIError(ErrorCode.EXECUTION_HISTORY_UNAVAILABLE)
            return Page((), None)
        await self._history_tree(record, tenant_id)
        final_run_id = step_run_id(namespace=self._namespace, tenant_id=tenant_id, execution_id=execution_id, segment_sequence=record.agent_run_sequence)
        snapshot = await self._store.latest_snapshot(run_id=final_run_id)
        if snapshot is None:
            if record.status is ExecutionStatus.SUCCEEDED:
                raise AIError(ErrorCode.EXECUTION_HISTORY_UNAVAILABLE)
            return Page((), None)
        if (
            snapshot.run_id != final_run_id
            or snapshot.conversation_id
            != step_conversation_id(
                namespace=self._namespace,
                tenant_id=tenant_id,
                execution_id=execution_id,
            )
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        conversation_id = step_conversation_id(namespace=self._namespace, tenant_id=tenant_id, execution_id=execution_id)
        values: list[str] = []
        for message in snapshot.messages:
            if isinstance(message, (ModelRequest, ModelResponse)):
                if message.conversation_id is None:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                if message.conversation_id != conversation_id:
                    continue
            if isinstance(message, ModelRequest):
                for part in message.parts:
                    if isinstance(part, UserPromptPart):
                        values.extend(_user_text(part))
            elif isinstance(message, ModelResponse):
                values.extend(part.content for part in message.parts if isinstance(part, TextPart) and part.content)
        start = _cursor_offset(cursor, len(values))
        selected = tuple(TranscriptItem(execution_id, start + index + 1, value) for index, value in enumerate(values[start:start + limit]))
        next_offset = start + len(selected)
        return Page(selected, str(next_offset) if next_offset < len(values) else None)

    async def _history_tree(self, root: ExecutionRecord, tenant_id: str) -> list[tuple[ExecutionRecord, int]]:
        result: list[tuple[ExecutionRecord, int]] = []
        visited: set[str] = set()

        async def visit(record: ExecutionRecord, depth: int) -> None:
            if record.execution_id in visited or depth > 8 or record.tenant_id != tenant_id:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if depth == 0:
                if record.parent_execution_id is not None or record.lineage_kind.value not in {"RUN", "RETRY", "FORK", "SESSION_RESUME"}:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            elif record.lineage_kind.value != "SUBAGENT" or record.root_execution_id != root.root_execution_id:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            visited.add(record.execution_id)
            result.append((record, depth))
            for child in await self._executions.list_children(record.execution_id, tenant_id=tenant_id):
                await visit(child, depth + 1)

        await visit(root, 0)
        return result


    async def _segment_events(self, record: ExecutionRecord, tenant_id: str) -> list[tuple[int, list[StepEvent]]]:
        if record.agent_run_sequence < 0:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if record.status is ExecutionStatus.SUCCEEDED and record.agent_run_sequence == 0:
            raise AIError(ErrorCode.EXECUTION_HISTORY_UNAVAILABLE)
        conversation_id = step_conversation_id(
            namespace=self._namespace,
            tenant_id=tenant_id,
            execution_id=record.execution_id,
        )
        result: list[tuple[int, list[StepEvent]]] = []
        for sequence in range(1, record.agent_run_sequence + 1):
            deterministic_id = step_run_id(
                namespace=self._namespace,
                tenant_id=tenant_id,
                execution_id=record.execution_id,
                segment_sequence=sequence,
            )
            run = await self._store.get_run(run_id=deterministic_id)
            if run is None:
                if (
                    record.status is ExecutionStatus.SUCCEEDED
                    and sequence == record.agent_run_sequence
                ):
                    raise AIError(ErrorCode.EXECUTION_HISTORY_UNAVAILABLE)
                continue
            _validate_run(run, deterministic_id, conversation_id, sequence)
            events = await self._store.list_events(run_id=deterministic_id)
            if any(
                event.run_id != deterministic_id
                or event.conversation_id not in {None, conversation_id}
                for event in events
            ):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            result.append((sequence, events))
        return result


class StepSessionHistoryReader:
    """Project one committed Conversation snapshot into Session history."""

    def __init__(self, *, store: StepStore, cursor_signer: CursorSigner) -> None:
        self._store = store
        self._cursor_signer = cursor_signer

    async def history(
        self,
        session_id: str,
        *,
        tenant_id: str,
        continuation_step_run_id: "str | None",
        continuation_history_id: "str | None" = None,
        cursor: "str | None",
        limit: int,
    ) -> "Page[SessionHistoryItem]":
        if not 1 <= limit <= 200:
            raise AIError(ErrorCode.PAGE_LIMIT_INVALID)
        if continuation_step_run_id is None:
            if cursor is not None:
                raise AIError(ErrorCode.CURSOR_INVALID)
            return Page((), None)
        cursor_values = (
            None
            if cursor is None
            else _decode_session_history_cursor(
                cursor,
                tenant_id,
                session_id,
                self._cursor_signer,
            )
        )
        history_store = (
            self._store
            if continuation_history_id is not None
            and isinstance(self._store, _SessionHistoryStore)
            else None
        )
        requested_history_id = (
            continuation_history_id
            if history_store is not None and continuation_history_id is not None
            else continuation_step_run_id
        )
        if cursor_values is not None and cursor_values[0] != requested_history_id:
            raise AIError(ErrorCode.CURSOR_INVALID)
        if history_store is not None:
            history_id = continuation_history_id
            if history_id is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            snapshot_message_count = (
                cursor_values[1]
                if cursor_values is not None
                else await history_store.session_message_count(
                    history_id,
                    tenant_id=tenant_id,
                )
            )
        else:
            history_id = continuation_step_run_id
            run = await self._store.get_run(run_id=continuation_step_run_id)
            if run is None:
                raise AIError(ErrorCode.SESSION_HISTORY_UNAVAILABLE)
            if run.run_id != continuation_step_run_id:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            snapshot = await self._store.latest_snapshot(
                run_id=continuation_step_run_id,
                include_interrupted=True,
            )
            if snapshot is None:
                raise AIError(ErrorCode.SESSION_HISTORY_UNAVAILABLE)
            if (
                snapshot.run_id != continuation_step_run_id
                or snapshot.conversation_id != run.conversation_id
            ):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if snapshot.state != "complete":
                raise AIError(ErrorCode.SESSION_HISTORY_UNAVAILABLE)
            messages = tuple([
                message
                async for message in self._store.iter_messages(
                    run_id=continuation_step_run_id,
                )
            ])
            snapshot_message_count = (
                cursor_values[1]
                if cursor_values is not None
                else len(messages)
            )

        if cursor_values is None:
            next_message_index = 0
            next_projected_item_offset = 0
        else:
            (
                next_message_index,
                next_projected_item_offset,
            ) = cursor_values[2:]
        if next_message_index > snapshot_message_count:
            raise AIError(ErrorCode.CURSOR_INVALID)

        selected: list[SessionHistoryItem] = []
        message_index = next_message_index
        item_offset = next_projected_item_offset
        if history_store is not None:
            messages_iterator = history_store.iter_session_message_range(
                history_id,
                tenant_id=tenant_id,
                start=next_message_index,
                end=snapshot_message_count,
            )
        else:
            messages_iterator = _iter_message_values(
                messages,
                start=next_message_index,
                end=snapshot_message_count,
            )
        async for message in messages_iterator:
            projected = _project_message(message)
            offset = item_offset if message_index == next_message_index else 0
            if offset > len(projected):
                raise AIError(ErrorCode.CURSOR_INVALID)
            remaining = projected[offset:]
            available = limit - len(selected)
            take = remaining[:available]
            selected.extend(
                SessionHistoryItem(
                    len(selected) + 1,
                    item.item_kind,
                    item.content,
                    item.tool_name,
                    item.tool_call_id,
                )
                for item in take
            )
            if len(take) < len(remaining):
                next_message_index = message_index
                next_projected_item_offset = offset + len(take)
                break
            message_index += 1
            next_message_index = message_index
            next_projected_item_offset = 0
            if len(selected) == limit:
                break
        else:
            next_message_index = message_index
            next_projected_item_offset = 0
        next_cursor = (
            _session_history_cursor(
                tenant_id,
                session_id,
                history_id,
                snapshot_message_count,
                next_message_index,
                next_projected_item_offset,
                self._cursor_signer,
            )
            if next_message_index < snapshot_message_count
            else None
        )
        _logger.debug(
            "session history projected page: session=%s history=%s "
            "message_start=%s message_snapshot=%s items=%s",
            session_id,
            history_id,
            next_message_index,
            snapshot_message_count,
            len(selected),
        )
        return Page(tuple(selected), next_cursor)


def _trace_item(record: ExecutionRecord, segment_sequence: int, depth: int, ordinal: int, event: StepEvent) -> "ExecutionTraceItem | None":
    mapping = {
        "model_request_started": ("MODEL_REQUEST", "STARTED"), "model_request_completed": ("MODEL_RESPONSE", "SUCCEEDED"),
        "model_request_failed": ("MODEL_RESPONSE", "FAILED"), "tool_call_started": ("TOOL_CALL", "STARTED"),
        "tool_call_completed": ("TOOL_RESULT", "SUCCEEDED"), "tool_call_failed": ("TOOL_ERROR", "FAILED"),
    }
    value = mapping.get(event.kind)
    if value is None:
        return None
    kind, status = value
    payload = {"kind": kind, "status": status, "step_index": event.step_index, "segment_sequence": segment_sequence, "scope": "root" if depth == 0 else "subagent", "depth": depth}
    if event.agent_name is not None:
        payload["agent_name"] = event.agent_name
    if event.tool_call_id is not None:
        payload["tool_call_id"] = event.tool_call_id
    if event.tool_name is not None:
        payload["tool_name"] = event.tool_name
    if depth > 0:
        payload["child_execution_id"] = record.execution_id
    return ExecutionTraceItem(record.execution_id, ordinal, payload)


def _validate_run(run: RunRecord, expected_id: str, conversation_id: str, sequence: int) -> None:
    if run.run_id != expected_id or run.conversation_id != conversation_id or run.metadata.get("segment_sequence") != str(sequence):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    agent_name = run.metadata.get("agent_name")
    if agent_name is None or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", agent_name) is None:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)


def _event_timestamp(event: StepEvent) -> datetime:
    return event.timestamp.astimezone(timezone.utc)


def _cursor_offset(cursor: str | None, size: int) -> int:
    if cursor is None:
        return 0
    try:
        offset = int(cursor)
    except ValueError as error:
        raise AIError(ErrorCode.CURSOR_INVALID) from error
    if offset < 0 or offset > size:
        raise AIError(ErrorCode.CURSOR_INVALID)
    return offset


def _history_cursor_offset(cursor: str | None, tenant_id: str, execution_id: str, source_revision: str, size: int, signer: CursorSigner) -> int:
    if cursor is None:
        return 0
    try:
        payload = signer.decode(cursor)
    except AIError as error:
        raise AIError(ErrorCode.CURSOR_INVALID) from error
    if (
        payload.cursor_version != 1
        or payload.tenant_id != tenant_id
        or payload.resource_kind != "execution_history"
        or payload.filter_digest != canonical_sha256({"execution_id": execution_id})
        or payload.sort_key != source_revision
    ):
        raise AIError(ErrorCode.CURSOR_INVALID)
    offset = payload.snapshot_or_store_revision
    if offset < 0 or offset > size:
        raise AIError(ErrorCode.CURSOR_INVALID)
    return offset


def _history_cursor(tenant_id: str, execution_id: str, source_revision: str, offset: int, signer: CursorSigner) -> str:
    return signer.encode(
        CursorPayload(
            1,
            tenant_id,
            "execution_history",
            canonical_sha256({"execution_id": execution_id}),
            source_revision,
            offset,
            int(time.time()) + 3600,
        )
    )


def _decode_session_history_cursor(
    cursor: str,
    tenant_id: str,
    session_id: str,
    signer: CursorSigner,
) -> tuple[str, int, int, int]:
    try:
        payload = signer.decode(cursor)
    except AIError as error:
        raise AIError(ErrorCode.CURSOR_INVALID) from error
    if (
        payload.cursor_version != 1
        or payload.tenant_id != tenant_id
        or payload.resource_kind != "session_history"
        or payload.filter_digest != canonical_sha256({"session_id": session_id})
        or payload.history_id is None
        or payload.snapshot_message_count is None
        or payload.next_message_index is None
        or payload.next_projected_item_offset is None
    ):
        raise AIError(ErrorCode.CURSOR_INVALID)
    if (
        payload.snapshot_message_count < 0
        or payload.next_message_index < 0
        or payload.next_projected_item_offset < 0
        or payload.next_message_index > payload.snapshot_message_count
    ):
        raise AIError(ErrorCode.CURSOR_INVALID)
    return (
        payload.history_id,
        payload.snapshot_message_count,
        payload.next_message_index,
        payload.next_projected_item_offset,
    )


def _session_history_cursor(
    tenant_id: str,
    session_id: str,
    history_id: str,
    snapshot_message_count: int,
    next_message_index: int,
    next_projected_item_offset: int,
    signer: CursorSigner,
) -> str:
    return signer.encode(
        CursorPayload(
            1,
            tenant_id,
            "session_history",
            canonical_sha256({"session_id": session_id}),
            "session_history",
            snapshot_message_count,
            int(time.time()) + 3600,
            history_id=history_id,
            snapshot_message_count=snapshot_message_count,
            next_message_index=next_message_index,
            next_projected_item_offset=next_projected_item_offset,
        )
    )


async def _iter_message_values(
    values: tuple[object, ...],
    *,
    start: int,
    end: int,
) -> AsyncIterator[object]:
    for value in values[start:end]:
        yield value


def _merge_history_occurrences(accumulated: list[ExecutionHistoryItem], snapshot: list[ExecutionHistoryItem]) -> list[ExecutionHistoryItem]:
    maximum = min(len(accumulated), len(snapshot))
    for overlap in range(maximum, 0, -1):
        if accumulated[-overlap:] == snapshot[:overlap]:
            return [*accumulated, *snapshot[overlap:]]
    return [*accumulated, *snapshot]


def _project_message(message: object) -> tuple[_ProjectedHistoryItem, ...]:
    if isinstance(message, ModelRequest):
        values: list[_ProjectedHistoryItem] = []
        for part in message.parts:
            if isinstance(part, SystemPromptPart):
                values.append(_ProjectedHistoryItem("system", _json_content(part.content)))
            elif isinstance(part, UserPromptPart):
                content = _user_content(part)
                if content is not None:
                    values.append(_ProjectedHistoryItem("user", content))
            elif isinstance(part, ToolReturnPart):
                values.append(
                    _ProjectedHistoryItem(
                        "tool_result",
                        _json_content(part.content),
                        part.tool_name,
                        part.tool_call_id,
                    )
                )
            elif isinstance(part, RetryPromptPart):
                values.append(_ProjectedHistoryItem("retry", str(part.content)))
        return tuple(values)
    if isinstance(message, ModelResponse):
        values = []
        for part in message.parts:
            if isinstance(part, TextPart):
                values.append(_ProjectedHistoryItem("assistant", part.content))
            elif isinstance(part, ThinkingPart):
                values.append(_ProjectedHistoryItem("thinking", part.content))
            elif isinstance(part, ToolCallPart):
                values.append(
                    _ProjectedHistoryItem(
                        "tool_call",
                        part.args_as_dict(),
                        part.tool_name,
                        part.tool_call_id,
                    )
                )
        return tuple(values)
    return ()


def _user_content(part: UserPromptPart) -> "str | list[str] | None":
    if isinstance(part.content, str):
        return part.content
    values: list[str] = []
    for item in part.content:
        if isinstance(item, str):
            values.append(item)
        elif isinstance(item, TextContent):
            values.append(item.content)
    return values if values else None


def _json_content(value: object) -> JsonValue:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return [_json_content(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_content(item) for key, item in value.items()}
    return str(value)


def _user_text(part: UserPromptPart) -> list[str]:
    if isinstance(part.content, str):
        return [part.content] if part.content else []
    values: list[str] = []
    for item in part.content:
        if isinstance(item, str) and item:
            values.append(item)
        elif isinstance(item, TextContent) and item.content:
            values.append(item.content)
    return values


__all__ = ["StepExecutionHistoryReader", "StepSessionHistoryReader"]
