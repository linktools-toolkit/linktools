#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Project Harness step facts into Runtime trace and transcript views."""

import re
import time
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Protocol, runtime_checkable

from linktools.core import environ
from pydantic_ai.messages import ModelRequest, ModelResponse
from pydantic_ai_harness.step_persistence import (
    RunRecord,
    StepEvent,
    StepStore,
)

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
from .service_api import (
    ExecutionHistoryItem,
    ExecutionTraceItem,
    SessionHistoryItem,
    TranscriptItem,
)
from .state import (
    SESSION_HISTORY_VIEW_V1,
    ExecutionReadModelBuild,
    ExecutionReadModelRepository,
    ExecutionRecord,
    ExecutionRepository,
    LoadedContextMessage,
    RuntimeDomain,
    TranscriptMessageRef,
    project_execution_transcript_message,
    project_session_history_message,
)

_logger = environ.get_logger("ai.runtime.history")
_MODEL_USAGE_INPUT_METADATA_KEY = "linktools.ai.model_usage.input_tokens"
_MODEL_USAGE_OUTPUT_METADATA_KEY = "linktools.ai.model_usage.output_tokens"
_MODEL_USAGE_CACHE_READ_METADATA_KEY = "linktools.ai.model_usage.cache_read_tokens"
_MODEL_USAGE_CACHE_WRITE_METADATA_KEY = "linktools.ai.model_usage.cache_write_tokens"
_MODEL_USAGE_METADATA_KEYS = frozenset(
    {
        _MODEL_USAGE_INPUT_METADATA_KEY,
        _MODEL_USAGE_OUTPUT_METADATA_KEY,
        _MODEL_USAGE_CACHE_READ_METADATA_KEY,
        _MODEL_USAGE_CACHE_WRITE_METADATA_KEY,
    }
)


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

    async def session_history_item_count(
        self,
        history_id: str,
        *,
        tenant_id: str,
    ) -> int: ...

    def iter_session_history_item_range(
        self,
        history_id: str,
        *,
        tenant_id: str,
        start: int,
        end: int,
    ) -> AsyncIterator[object]: ...

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


@runtime_checkable
class _CanonicalTranscriptStore(Protocol):
    async def resolve_transcript_message_refs(
        self,
        refs: tuple[TranscriptMessageRef, ...],
    ) -> tuple[LoadedContextMessage, ...]: ...


@runtime_checkable
class _ExecutionTranscriptStore(Protocol):
    async def execution_transcript_item_count(self, run_id: str) -> int: ...

    def iter_execution_transcript_item_range(
        self,
        run_id: str,
        *,
        start: int,
        end: int,
    ) -> AsyncIterator[object]: ...


async def _canonical_transcript(
    store: StepStore,
    run_id: str,
) -> tuple[object, ...]:
    """Read one run's canonical raw transcript instead of the latest snapshot."""
    return tuple([message async for message in store.iter_messages(run_id=run_id)])


class StepExecutionHistoryReader:
    """Own the adapter projection between StepStore facts and Runtime views."""

    def __init__(
        self,
        *,
        namespace: str,
        executions: ExecutionRepository,
        store: StepStore,
        cursor_signer: CursorSigner,
        read_model: ExecutionReadModelRepository | None = None,
    ) -> None:
        try:
            validate_persistence_namespace(namespace)
        except AIError as error:
            raise ValueError("execution history namespace is invalid") from error
        self._namespace = namespace
        self._executions = executions
        self._store = store
        self._cursor_signer = cursor_signer
        self._read_model = read_model

    async def trace(self, execution_id: str, *, tenant_id: str, cursor: "str | None", limit: int) -> "Page[ExecutionTraceItem]":
        record = await self._executions.get(execution_id, tenant_id=tenant_id)
        if record is None:
            raise AIError(ErrorCode.STORAGE_NOT_FOUND)
        if not 1 <= limit <= 200:
            raise AIError(ErrorCode.PAGE_LIMIT_INVALID)
        if self._is_terminal_root(record) and self._read_model is not None:
            model = await self._read_model.ensure(
                execution_id,
                tenant_id=tenant_id,
                builder=lambda: self._build_read_model(record, tenant_id),
            )
            start = _cursor_offset(cursor, model.trace_count)
            _model, refs = await self._read_model.page(
                execution_id,
                tenant_id=tenant_id,
                stream_name="trace",
                offset=start,
                limit=limit,
            )
            values = tuple(
                ExecutionTraceItem(
                    str(ref["execution_id"]),
                    start + index + 1,
                    ref["payload"],
                )
                for index, ref in enumerate(refs)
            )
            next_offset = start + len(values)
            _logger.debug(
                "terminal execution trace page read from model: execution=%s offset=%s items=%s",
                execution_id,
                start,
                len(values),
            )
            return Page(
                values,
                str(next_offset) if next_offset < model.trace_count else None,
            )
        entries = await self._history_tree(record, tenant_id)
        projected: list[tuple[tuple[object, ...], ExecutionTraceItem]] = []
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
        if self._is_terminal_root(record) and self._read_model is not None:
            model = await self._read_model.ensure(
                execution_id,
                tenant_id=tenant_id,
                builder=lambda: self._build_read_model(record, tenant_id),
            )
            start = _history_cursor_offset(
                cursor,
                tenant_id,
                execution_id,
                model.source_digest,
                model.history_count,
                self._cursor_signer,
            )
            _model, refs = await self._read_model.page(
                execution_id,
                tenant_id=tenant_id,
                stream_name="history",
                offset=start,
                limit=limit,
            )
            values = await _resolve_history_refs(
                refs,
                namespace=self._namespace,
                tenant_id=tenant_id,
                store=self._store,
                sequence_start=start,
            )
            next_offset = start + len(values)
            next_cursor = (
                _history_cursor(
                    tenant_id,
                    execution_id,
                    model.source_digest,
                    next_offset,
                    self._cursor_signer,
                )
                if next_offset < model.history_count
                else None
            )
            _logger.debug(
                "terminal execution history page read from model: execution=%s offset=%s items=%s",
                execution_id,
                start,
                len(values),
            )
            return Page(tuple(values), next_cursor)
        entries = await self._history_tree(record, tenant_id)
        values: list[ExecutionHistoryItem] = []
        for item, _depth in entries:
            for segment_sequence, _events in await self._segment_events(item, tenant_id):
                run_id = step_run_id(namespace=self._namespace, tenant_id=tenant_id, execution_id=item.execution_id, segment_sequence=segment_sequence)
                messages = await _canonical_transcript(self._store, run_id)
                if not messages:
                    if item.status is ExecutionStatus.SUCCEEDED and segment_sequence == item.agent_run_sequence:
                        raise AIError(ErrorCode.EXECUTION_HISTORY_UNAVAILABLE)
                    continue
                snapshot_items = [
                    ExecutionHistoryItem(
                        item.execution_id,
                        0,
                        projected.item_kind,
                        projected.content,
                        projected.tool_name,
                        projected.tool_call_id,
                    )
                    for message in messages
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
        if self._is_terminal_root(record) and self._read_model is not None:
            model = await self._read_model.ensure(
                execution_id,
                tenant_id=tenant_id,
                builder=lambda: self._build_read_model(record, tenant_id),
            )
            start = _cursor_offset(cursor, model.transcript_count)
            _model, refs = await self._read_model.page(
                execution_id,
                tenant_id=tenant_id,
                stream_name="transcript",
                offset=start,
                limit=limit,
            )
            values = await _resolve_transcript_refs(
                refs,
                execution_id=execution_id,
                namespace=self._namespace,
                tenant_id=tenant_id,
                store=self._store,
                sequence_start=start,
            )
            next_offset = start + len(values)
            _logger.debug(
                "terminal execution transcript page read from model: execution=%s offset=%s items=%s",
                execution_id,
                start,
                len(values),
            )
            return Page(
                tuple(values),
                str(next_offset) if next_offset < model.transcript_count else None,
            )
        if record.agent_run_sequence == 0:
            if record.status is ExecutionStatus.SUCCEEDED:
                raise AIError(ErrorCode.EXECUTION_HISTORY_UNAVAILABLE)
            return Page((), None)
        await self._history_tree(record, tenant_id)
        final_run_id = step_run_id(namespace=self._namespace, tenant_id=tenant_id, execution_id=execution_id, segment_sequence=record.agent_run_sequence)
        if isinstance(self._store, _ExecutionTranscriptStore):
            total = await self._store.execution_transcript_item_count(final_run_id)
            if total == 0:
                if record.status is ExecutionStatus.SUCCEEDED:
                    raise AIError(ErrorCode.EXECUTION_HISTORY_UNAVAILABLE)
                return Page((), None)
            start = _cursor_offset(cursor, total)
            end = min(total, start + limit)
            values: list[TranscriptItem] = []
            async for text in self._store.iter_execution_transcript_item_range(
                final_run_id,
                start=start,
                end=end,
            ):
                if not isinstance(text, str):
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                values.append(TranscriptItem(execution_id, start + len(values) + 1, text))
            if len(values) != end - start:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            return Page(
                tuple(values),
                str(end) if end < total else None,
            )
        messages = await _canonical_transcript(self._store, final_run_id)
        if not messages:
            if record.status is ExecutionStatus.SUCCEEDED:
                raise AIError(ErrorCode.EXECUTION_HISTORY_UNAVAILABLE)
            return Page((), None)
        conversation_id = step_conversation_id(namespace=self._namespace, tenant_id=tenant_id, execution_id=execution_id)
        values = [
            value
            for message in messages
            for value in _transcript_message_values(message, conversation_id)
        ]
        start = _cursor_offset(cursor, len(values))
        selected = tuple(TranscriptItem(execution_id, start + index + 1, value) for index, value in enumerate(values[start:start + limit]))
        next_offset = start + len(selected)
        return Page(selected, str(next_offset) if next_offset < len(values) else None)

    def _is_terminal_root(self, record: ExecutionRecord) -> bool:
        return (
            record.execution_id == record.root_execution_id
            and record.parent_execution_id is None
            and record.status in {
                ExecutionStatus.SUCCEEDED,
                ExecutionStatus.FAILED,
                ExecutionStatus.CANCELLED,
            }
        )

    async def _build_read_model(
        self,
        root: ExecutionRecord,
        tenant_id: str,
    ) -> ExecutionReadModelBuild:
        entries = await self._history_tree(root, tenant_id)
        trace_values: list[tuple[tuple[object, ...], dict[str, JsonValue]]] = []
        history_values: list[tuple[ExecutionHistoryItem, dict[str, JsonValue]]] = []
        for item, depth in entries:
            for segment_sequence, events in await self._segment_events(item, tenant_id):
                for ordinal, event in enumerate(events):
                    mapped = _trace_item(item, segment_sequence, depth, ordinal, event)
                    if mapped is not None:
                        trace_values.append(
                            (
                                (
                                    _event_timestamp(event),
                                    depth,
                                    item.execution_id,
                                    segment_sequence,
                                    ordinal,
                                    str(mapped.payload.get("kind", "")),
                                ),
                                {
                                    "execution_id": mapped.execution_id,
                                    "payload": mapped.payload,
                                },
                            )
                        )
                run_id = step_run_id(
                    namespace=self._namespace,
                    tenant_id=tenant_id,
                    execution_id=item.execution_id,
                    segment_sequence=segment_sequence,
                )
                messages = await _canonical_transcript(self._store, run_id)
                if not messages:
                    if item.status is ExecutionStatus.SUCCEEDED and segment_sequence == item.agent_run_sequence:
                        raise AIError(ErrorCode.EXECUTION_HISTORY_UNAVAILABLE)
                    continue
                snapshot_items = [
                    (
                        ExecutionHistoryItem(
                            item.execution_id,
                            0,
                            projected.item_kind,
                            projected.content,
                            projected.tool_name,
                            projected.tool_call_id,
                        ),
                        {
                            "execution_id": item.execution_id,
                            "source_domain": "execution",
                            "owner_id": run_id,
                            "segment_sequence": segment_sequence,
                            "message_index": message_index,
                            "projected_item_offset": projected_offset,
                            "item_kind": projected.item_kind,
                            "tool_name": projected.tool_name,
                            "tool_call_id": projected.tool_call_id,
                        },
                    )
                    for message_index, message in enumerate(messages)
                    for projected_offset, projected in enumerate(_project_message(message))
                ]
                history_values = _merge_history_refs(history_values, snapshot_items)
        trace_values.sort(key=lambda value: value[0])
        transcript_values: list[dict[str, JsonValue]] = []
        if root.agent_run_sequence > 0:
            run_id = step_run_id(
                namespace=self._namespace,
                tenant_id=tenant_id,
                execution_id=root.execution_id,
                segment_sequence=root.agent_run_sequence,
            )
            messages = await _canonical_transcript(self._store, run_id)
            if not messages:
                if root.status is ExecutionStatus.SUCCEEDED:
                    raise AIError(ErrorCode.EXECUTION_HISTORY_UNAVAILABLE)
            else:
                conversation_id = step_conversation_id(
                    namespace=self._namespace,
                    tenant_id=tenant_id,
                    execution_id=root.execution_id,
                )
                for message_index, message in enumerate(messages):
                    for projected_offset, _value in enumerate(
                        _transcript_message_values(message, conversation_id)
                    ):
                        transcript_values.append(
                            {
                                "source_domain": "execution",
                                "owner_id": run_id,
                                "segment_sequence": root.agent_run_sequence,
                                "message_index": message_index,
                                "projected_item_offset": projected_offset,
                            }
                        )
        source = []
        for item, _depth in entries:
            source.append(
                (
                    item.execution_id,
                    item.parent_execution_id or "",
                    item.status,
                    await self._executions.get_history_seal(
                        item.execution_id,
                        tenant_id=tenant_id,
                    ),
                )
            )
        source.sort(key=lambda value: value[0])
        seals = []
        for execution_id, parent_execution_id, status, seal in source:
            if (
                seal is None
                or seal.execution_id != execution_id
                or seal.tenant_id != tenant_id
                or status
                not in {
                    ExecutionStatus.SUCCEEDED,
                    ExecutionStatus.FAILED,
                    ExecutionStatus.CANCELLED,
                }
            ):
                raise AIError(ErrorCode.EXECUTION_HISTORY_UNAVAILABLE)
            seals.append(
                {
                    "execution_id": execution_id,
                    "parent_execution_id": parent_execution_id,
                    "seal_digest": seal.seal_digest,
                }
            )
        source_digest = canonical_sha256(
            {
                "model_version": 3,
                "root_execution_id": root.execution_id,
                "seals": seals,
            }
        )
        return ExecutionReadModelBuild(
            root.execution_id,
            tenant_id,
            source_digest,
            tuple(value for _sort_key, value in trace_values),
            tuple(ref for _item, ref in history_values),
            tuple(transcript_values),
        )

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
            snapshot_history_item_count = (
                cursor_values[2]
                if cursor_values is not None
                else await history_store.session_history_item_count(
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
            snapshot_message_count = (
                cursor_values[1]
                if cursor_values is not None
                else len(snapshot.messages)
            )
            snapshot_history_item_count = (
                cursor_values[2]
                if cursor_values is not None
                else sum(
                    len(project_session_history_message(message))
                    for message in snapshot.messages
                )
            )
        if cursor_values is None:
            next_history_item_offset = 0
        else:
            next_history_item_offset = cursor_values[3]
        if next_history_item_offset > snapshot_history_item_count:
            raise AIError(ErrorCode.CURSOR_INVALID)

        selected: list[SessionHistoryItem] = []
        item_offset = next_history_item_offset
        remaining_items = snapshot_history_item_count - item_offset
        if remaining_items > 0:
            page_end = min(snapshot_history_item_count, item_offset + limit)
            if history_store is not None:
                async for projected in history_store.iter_session_history_item_range(
                    history_id,
                    tenant_id=tenant_id,
                    start=item_offset,
                    end=page_end,
                ):
                    if not isinstance(projected, SessionHistoryItem):
                        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                    selected.append(
                        SessionHistoryItem(
                            item_offset + len(selected) + 1,
                            projected.item_kind,
                            projected.content,
                            projected.tool_name,
                            projected.tool_call_id,
                        )
                    )
            else:
                item_cursor = 0
                for message in snapshot.messages:
                    projected_items = project_session_history_message(message)
                    for item in projected_items:
                        if item_cursor < item_offset:
                            item_cursor += 1
                            continue
                        if len(selected) == limit:
                            break
                        selected.append(
                            SessionHistoryItem(
                                item_offset + len(selected) + 1,
                                item.item_kind,
                                item.content,
                                item.tool_name,
                                item.tool_call_id,
                            )
                        )
                        item_cursor += 1
                    if len(selected) == limit:
                        break
        next_cursor = (
            _session_history_cursor(
                tenant_id,
                session_id,
                history_id,
                snapshot_message_count,
                snapshot_history_item_count,
                next_history_item_offset + len(selected),
                self._cursor_signer,
            )
            if next_history_item_offset + len(selected) < snapshot_history_item_count
            else None
        )
        _logger.debug(
            "session history projected page: session=%s history=%s "
            "item_start=%s item_snapshot=%s items=%s",
            session_id,
            history_id,
            next_history_item_offset,
            snapshot_history_item_count,
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
    if kind == "MODEL_RESPONSE":
        payload["token_usage"] = _model_token_usage(event) if status == "SUCCEEDED" else None
    if event.agent_name is not None:
        payload["agent_name"] = event.agent_name
    if event.tool_call_id is not None:
        payload["tool_call_id"] = event.tool_call_id
    if event.tool_name is not None:
        payload["tool_name"] = event.tool_name
    if depth > 0:
        payload["child_execution_id"] = record.execution_id
    return ExecutionTraceItem(record.execution_id, ordinal, payload)


def _model_token_usage(event: StepEvent) -> "dict[str, JsonValue] | None":
    metadata = event.metadata
    present = _MODEL_USAGE_METADATA_KEYS.intersection(metadata)
    if not present:
        return None
    if (
        _MODEL_USAGE_INPUT_METADATA_KEY not in metadata
        or _MODEL_USAGE_OUTPUT_METADATA_KEY not in metadata
    ):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return {
        "input_tokens": _metadata_token(metadata, _MODEL_USAGE_INPUT_METADATA_KEY),
        "output_tokens": _metadata_token(metadata, _MODEL_USAGE_OUTPUT_METADATA_KEY),
        "cache_read_tokens": _metadata_token(
            metadata,
            _MODEL_USAGE_CACHE_READ_METADATA_KEY,
            required=False,
        ),
        "cache_write_tokens": _metadata_token(
            metadata,
            _MODEL_USAGE_CACHE_WRITE_METADATA_KEY,
            required=False,
        ),
    }


def _metadata_token(
    metadata: Mapping[str, str],
    key: str,
    *,
    required: bool = True,
) -> "int | None":
    raw = metadata.get(key)
    if raw is None:
        if required:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return None
    if not raw.isdigit():
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return int(raw)


def _merge_history_refs(
    accumulated: list[tuple[ExecutionHistoryItem, dict[str, JsonValue]]],
    snapshot: list[tuple[ExecutionHistoryItem, dict[str, JsonValue]]],
) -> list[tuple[ExecutionHistoryItem, dict[str, JsonValue]]]:
    overlap = _suffix_prefix_overlap(
        [item for item, _ref in accumulated],
        [item for item, _ref in snapshot],
    )
    return [*accumulated, *snapshot[overlap:]]


async def _resolve_history_refs(
    refs: tuple[Mapping[str, JsonValue], ...],
    *,
    namespace: str,
    tenant_id: str,
    store: StepStore,
    sequence_start: int,
) -> tuple[ExecutionHistoryItem, ...]:
    if not isinstance(store, _CanonicalTranscriptStore):
        raise AIError(ErrorCode.STORAGE_DEPENDENCY_NOT_READY)
    raw_refs: list[TranscriptMessageRef] = []
    metadata: list[tuple[str, int, str, str | None, str | None]] = []
    for ref in refs:
        source_domain = _ref_string(ref, "source_domain")
        run_id = _ref_string(ref, "owner_id")
        execution_id = _ref_string(ref, "execution_id")
        segment_sequence = _ref_int(ref, "segment_sequence")
        message_index = _ref_int(ref, "message_index")
        projected_offset = _ref_int(ref, "projected_item_offset")
        expected_run_id = step_run_id(
            namespace=namespace,
            tenant_id=tenant_id,
            execution_id=execution_id,
            segment_sequence=segment_sequence,
        )
        if (
            source_domain != "execution"
            or run_id != expected_run_id
            or message_index < 0
            or projected_offset < 0
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        raw_refs.append(
            TranscriptMessageRef(RuntimeDomain.EXECUTION, run_id, message_index)
        )
        metadata.append(
            (
                execution_id,
                projected_offset,
                _ref_string(ref, "item_kind"),
                _ref_optional_string(ref, "tool_name"),
                _ref_optional_string(ref, "tool_call_id"),
            )
        )
    loaded = await store.resolve_transcript_message_refs(tuple(raw_refs))
    if len(loaded) != len(metadata):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    values: list[ExecutionHistoryItem] = []
    for loaded_message, (execution_id, projected_offset, item_kind, tool_name, tool_call_id) in zip(
        loaded,
        metadata,
        strict=True,
    ):
        projected = _project_message(loaded_message.message)
        if projected_offset >= len(projected):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        item = projected[projected_offset]
        if (
            item.item_kind != item_kind
            or item.tool_name != tool_name
            or item.tool_call_id != tool_call_id
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        values.append(
            ExecutionHistoryItem(
                execution_id,
                0,
                item.item_kind,
                item.content,
                item.tool_name,
                item.tool_call_id,
            )
        )
    return tuple(
        ExecutionHistoryItem(
            item.execution_id,
            sequence_start + index + 1,
            item.item_kind,
            item.content,
            item.tool_name,
            item.tool_call_id,
        )
        for index, item in enumerate(values)
    )


async def _resolve_transcript_refs(
    refs: tuple[Mapping[str, JsonValue], ...],
    *,
    execution_id: str,
    namespace: str,
    tenant_id: str,
    store: StepStore,
    sequence_start: int,
) -> tuple[TranscriptItem, ...]:
    if not isinstance(store, _CanonicalTranscriptStore):
        raise AIError(ErrorCode.STORAGE_DEPENDENCY_NOT_READY)
    raw_refs: list[TranscriptMessageRef] = []
    projected_offsets: list[int] = []
    for ref in refs:
        source_domain = _ref_string(ref, "source_domain")
        run_id = _ref_string(ref, "owner_id")
        segment_sequence = _ref_int(ref, "segment_sequence")
        expected_run_id = step_run_id(
            namespace=namespace,
            tenant_id=tenant_id,
            execution_id=execution_id,
            segment_sequence=segment_sequence,
        )
        if source_domain != "execution" or run_id != expected_run_id:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        message_index = _ref_int(ref, "message_index")
        projected_offset = _ref_int(ref, "projected_item_offset")
        if message_index < 0 or projected_offset < 0:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        raw_refs.append(
            TranscriptMessageRef(RuntimeDomain.EXECUTION, run_id, message_index)
        )
        projected_offsets.append(projected_offset)
    loaded = await store.resolve_transcript_message_refs(tuple(raw_refs))
    if len(loaded) != len(projected_offsets):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    values: list[str] = []
    for loaded_message, projected_offset in zip(loaded, projected_offsets, strict=True):
        projected = project_execution_transcript_message(loaded_message.message)
        if projected_offset >= len(projected):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        values.append(projected[projected_offset])
    return tuple(
        TranscriptItem(execution_id, sequence_start + index + 1, value)
        for index, value in enumerate(values)
    )


def _ref_string(ref: Mapping[str, JsonValue], name: str) -> str:
    value = ref.get(name)
    if not isinstance(value, str) or not value:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return value


def _ref_optional_string(ref: Mapping[str, JsonValue], name: str) -> str | None:
    value = ref.get(name)
    if value is not None and not isinstance(value, str):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return value


def _ref_int(ref: Mapping[str, JsonValue], name: str) -> int:
    value = ref.get(name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return value


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
        payload.cursor_version != 2
        or payload.tenant_id != tenant_id
        or payload.resource_kind != "session_history"
        or payload.filter_digest != canonical_sha256({"session_id": session_id})
        or payload.history_id is None
        or payload.snapshot_message_count is None
        or payload.snapshot_history_item_count is None
        or payload.next_history_item_offset is None
        or payload.history_view_version != SESSION_HISTORY_VIEW_V1
    ):
        raise AIError(ErrorCode.CURSOR_INVALID)
    if (
        payload.snapshot_message_count < 0
        or payload.next_history_item_offset < 0
    ):
        raise AIError(ErrorCode.CURSOR_INVALID)
    return (
        payload.history_id,
        payload.snapshot_message_count,
        payload.snapshot_history_item_count,
        payload.next_history_item_offset,
    )


def _session_history_cursor(
    tenant_id: str,
    session_id: str,
    history_id: str,
    snapshot_message_count: int,
    snapshot_history_item_count: int,
    next_history_item_offset: int,
    signer: CursorSigner,
) -> str:
    return signer.encode(
        CursorPayload(
            2,
            tenant_id,
            "session_history",
            canonical_sha256({"session_id": session_id}),
            "session_history",
            snapshot_message_count,
            int(time.time()) + 3600,
            history_id=history_id,
            snapshot_message_count=snapshot_message_count,
            snapshot_history_item_count=snapshot_history_item_count,
            next_history_item_offset=next_history_item_offset,
            history_view_version=SESSION_HISTORY_VIEW_V1,
        )
    )


def _merge_history_occurrences(
    accumulated: list[ExecutionHistoryItem],
    snapshot: list[ExecutionHistoryItem],
) -> list[ExecutionHistoryItem]:
    overlap = _suffix_prefix_overlap(accumulated, snapshot)
    return [*accumulated, *snapshot[overlap:]]


def _suffix_prefix_overlap(
    stored: list[object],
    incoming: list[object],
) -> int:
    if not stored or not incoming:
        return 0
    prefix = [0] * len(incoming)
    matched = 0
    for index in range(1, len(incoming)):
        while matched and incoming[index] != incoming[matched]:
            matched = prefix[matched - 1]
        if incoming[index] == incoming[matched]:
            matched += 1
        prefix[index] = matched
    matched = 0
    for index, value in enumerate(stored):
        while matched and value != incoming[matched]:
            matched = prefix[matched - 1]
        if value == incoming[matched]:
            matched += 1
        if matched == len(incoming):
            if index == len(stored) - 1:
                return matched
            matched = prefix[matched - 1]
    return matched


def _project_message(message: object) -> tuple[_ProjectedHistoryItem, ...]:
    if not isinstance(message, (ModelRequest, ModelResponse)):
        return ()
    return tuple(
        _ProjectedHistoryItem(
            item.item_kind,
            item.content,
            item.tool_name,
            item.tool_call_id,
        )
        for item in project_session_history_message(message)
    )


def _transcript_message_values(message: object, conversation_id: str) -> tuple[str, ...]:
    del conversation_id
    if not isinstance(message, (ModelRequest, ModelResponse)):
        return ()
    return project_execution_transcript_message(message)


__all__ = ["StepExecutionHistoryReader", "StepSessionHistoryReader"]
