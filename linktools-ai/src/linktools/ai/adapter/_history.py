#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Project Harness step facts into Runtime trace and transcript views."""

import re
import time
from datetime import datetime, timezone

from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    SystemPromptPart,
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
    TranscriptItem,
)
from ..runtime.state import ExecutionRecord, ExecutionRepository


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
                    continue
                snapshot_items = [projected for message in snapshot.messages for projected in _message_items(item.execution_id, message)]
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
            return Page((), None)
        await self._history_tree(record, tenant_id)
        final_run_id = step_run_id(namespace=self._namespace, tenant_id=tenant_id, execution_id=execution_id, segment_sequence=record.agent_run_sequence)
        snapshot = await self._store.latest_snapshot(run_id=final_run_id)
        if snapshot is None:
            if record.status.value == "SUCCEEDED":
                raise AIError(ErrorCode.EXECUTION_HISTORY_UNAVAILABLE)
            return Page((), None)
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
        conversation_id = step_conversation_id(namespace=self._namespace, tenant_id=tenant_id, execution_id=record.execution_id)
        terminal = record.status.value in {"SUCCEEDED", "FAILED", "CANCELLED"}
        result: list[tuple[int, list[StepEvent]]] = []
        for sequence in range(1, record.agent_run_sequence + 1):
            deterministic_id = step_run_id(namespace=self._namespace, tenant_id=tenant_id, execution_id=record.execution_id, segment_sequence=sequence)
            run = await self._store.get_run(run_id=deterministic_id)
            if run is None:
                if not terminal and sequence == record.agent_run_sequence:
                    continue
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            _validate_run(run, deterministic_id, conversation_id, sequence)
            events = await self._store.list_events(run_id=deterministic_id)
            if any(event.run_id != deterministic_id or event.conversation_id not in {None, conversation_id} for event in events):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            result.append((sequence, events))
        return result


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
        payload.tenant_id != tenant_id
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


def _merge_history_occurrences(accumulated: list[ExecutionHistoryItem], snapshot: list[ExecutionHistoryItem]) -> list[ExecutionHistoryItem]:
    maximum = min(len(accumulated), len(snapshot))
    for overlap in range(maximum, 0, -1):
        if accumulated[-overlap:] == snapshot[:overlap]:
            return [*accumulated, *snapshot[overlap:]]
    return [*accumulated, *snapshot]


def _message_items(execution_id: str, message: object) -> tuple[ExecutionHistoryItem, ...]:
    if isinstance(message, ModelRequest):
        values: list[ExecutionHistoryItem] = []
        for part in message.parts:
            if isinstance(part, SystemPromptPart):
                values.append(ExecutionHistoryItem(execution_id, 0, "system", _json_content(part.content)))
            elif isinstance(part, UserPromptPart):
                content = _user_content(part)
                if content:
                    values.append(ExecutionHistoryItem(execution_id, 0, "user", content))
            elif isinstance(part, ToolReturnPart):
                values.append(ExecutionHistoryItem(execution_id, 0, "tool_result", _json_content(part.content), part.tool_name, part.tool_call_id))
            elif isinstance(part, RetryPromptPart):
                values.append(ExecutionHistoryItem(execution_id, 0, "retry", str(part.content)))
        return tuple(values)
    if isinstance(message, ModelResponse):
        values = []
        for part in message.parts:
            if isinstance(part, TextPart) and part.content:
                values.append(ExecutionHistoryItem(execution_id, 0, "assistant", part.content))
            elif isinstance(part, ToolCallPart):
                values.append(ExecutionHistoryItem(execution_id, 0, "tool_call", part.args_as_dict(), part.tool_name, part.tool_call_id))
        return tuple(values)
    return ()


def _user_content(part: UserPromptPart) -> "str | list[str] | None":
    if isinstance(part.content, str):
        return part.content or None
    values: list[str] = []
    for item in part.content:
        if isinstance(item, str) and item:
            values.append(item)
        elif isinstance(item, TextContent) and item.content:
            values.append(item.content)
    return values or None


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


__all__ = ["StepExecutionHistoryReader"]
