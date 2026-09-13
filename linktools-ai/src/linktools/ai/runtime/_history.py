#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Project Runtime step facts into Runtime trace and transcript views."""

import heapq
import re
import time
from collections.abc import AsyncGenerator, AsyncIterator, Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Protocol, runtime_checkable

from linktools.core import environ
from pydantic_ai.messages import ModelRequest, ModelResponse

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
from ._journal import (
    DURATION_NS_METADATA_KEY,
    MODEL_USAGE_CACHE_READ_METADATA_KEY,
    MODEL_USAGE_CACHE_WRITE_METADATA_KEY,
    MODEL_USAGE_INPUT_METADATA_KEY,
    MODEL_USAGE_METADATA_KEYS,
    MODEL_USAGE_OUTPUT_METADATA_KEY,
    OBSERVATION_ID_METADATA_KEY,
    OUTPUT_RETRY_INDEX_METADATA_KEY,
    REQUEST_PURPOSE_METADATA_KEY,
    REQUEST_SEQUENCE_METADATA_KEY,
)
from .service_api import (
    ExecutionHistoryItem,
    ExecutionTraceItem,
    SessionHistoryItem,
    TranscriptItem,
)
from .state._contracts import ExecutionRecord, ExecutionRepository
from .state._step_contracts import RunRecord, StepEvent, StepStore
from .state._views import (
    SESSION_HISTORY_VIEW_V1,
    project_execution_transcript_message,
    project_session_history_message,
)

_logger = environ.get_logger("ai.runtime.history")


@dataclass(frozen=True, slots=True)
class _ProjectedHistoryItem:
    item_kind: str
    content: JsonValue
    tool_name: "str | None" = None
    tool_call_id: "str | None" = None


@dataclass(frozen=True, slots=True)
class _HistoryOccurrence:
    item: ExecutionHistoryItem
    source_execution_id: str
    segment_sequence: int
    message_index: int
    item_offset: int
    merge_key: tuple[object, ...]


@dataclass(frozen=True, slots=True)
class _TraceOccurrence:
    item: ExecutionTraceItem
    source_execution_id: str
    segment_sequence: int
    event_sequence: int
    merge_key: tuple[object, ...]


@dataclass(frozen=True, slots=True)
class _HistorySource:
    record: ExecutionRecord
    depth: int
    segment_sequence: int
    merge_prefix: tuple[object, ...]


async def _iter_sequence(
    values: Sequence[object],
    *,
    start: int,
) -> AsyncIterator[object]:
    for value in values[start:]:
        yield value


async def _read_projected_page(
    messages: AsyncIterator[object],
    *,
    start_message_index: int,
    start_item_offset: int,
    project: Callable[[object], Sequence[object]],
    limit: int,
) -> tuple[list[tuple[int, int, object]], tuple[int, int] | None]:
    selected: list[tuple[int, int, object]] = []
    message_index = start_message_index
    first = True
    async for message in messages:
        projected = tuple(project(message))
        item_offset = start_item_offset if first else 0
        if first and item_offset > len(projected):
            raise AIError(ErrorCode.CURSOR_INVALID)
        first = False
        for item_index in range(item_offset, len(projected)):
            if len(selected) == limit:
                return selected, (message_index, item_index)
            selected.append((message_index, item_index, projected[item_index]))
        message_index += 1
    if first and start_item_offset:
        raise AIError(ErrorCode.CURSOR_INVALID)
    return selected, None


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


@runtime_checkable
class _RangedTranscriptStore(Protocol):
    async def transcript_message_count(self, owner_id: str) -> int: ...

    def iter_message_range(
        self,
        *,
        run_id: str,
        start: int,
        end: int,
    ) -> AsyncIterator[object]: ...


class StepExecutionHistoryReader:
    """Own the adapter projection between StepStore facts and Runtime views."""

    def __init__(
        self,
        *,
        namespace: str,
        executions: ExecutionRepository,
        store: StepStore,
        cursor_signer: CursorSigner,
    ) -> None:
        try:
            validate_persistence_namespace(namespace)
        except AIError as error:
            raise ValueError("execution history namespace is invalid") from error
        self._namespace = namespace
        self._executions = executions
        self._store = store
        self._cursor_signer = cursor_signer

    async def trace(
        self, execution_id: str, *, tenant_id: str, cursor: "str | None", limit: int
    ) -> "Page[ExecutionTraceItem]":
        record = await self._executions.get(execution_id, tenant_id=tenant_id)
        if record is None:
            raise AIError(ErrorCode.STORAGE_NOT_FOUND)
        if not 1 <= limit <= 200:
            raise AIError(ErrorCode.PAGE_LIMIT_INVALID)
        entries = await self._history_tree(record, tenant_id)
        occurrences: list[_TraceOccurrence] = []
        for item, depth in entries:
            for segment_sequence, events in await self._segment_events(item, tenant_id):
                for ordinal, event in enumerate(events):
                    mapped = _trace_item(item, segment_sequence, depth, ordinal, event)
                    if mapped is not None:
                        occurrences.append(
                            _TraceOccurrence(
                                mapped,
                                item.execution_id,
                                segment_sequence,
                                ordinal + 1,
                                (
                                    _event_timestamp(event),
                                    depth,
                                    item.execution_id,
                                    segment_sequence,
                                    ordinal + 1,
                                    mapped.payload.get("kind", ""),
                                ),
                            )
                        )
        occurrences.sort(key=lambda occurrence: occurrence.merge_key)
        start = _trace_cursor_index(
            cursor,
            tenant_id=tenant_id,
            execution_id=execution_id,
            signer=self._cursor_signer,
            occurrences=occurrences,
        )
        page = occurrences[start : start + limit + 1]
        selected = tuple(occurrence.item for occurrence in page[:limit])
        next_cursor = None
        if len(page) > limit:
            next_cursor = _trace_cursor(
                tenant_id,
                execution_id,
                page[limit],
                self._cursor_signer,
            )
        _logger.debug(
            "execution trace projected page: execution=%s source_index=%s items=%s",
            execution_id,
            start,
            len(selected),
        )
        return Page(selected, next_cursor)

    async def history(
        self, execution_id: str, *, tenant_id: str, cursor: str | None, limit: int
    ) -> "Page[ExecutionHistoryItem]":
        if not 1 <= limit <= 200:
            raise AIError(ErrorCode.PAGE_LIMIT_INVALID)
        record = await self._executions.get(execution_id, tenant_id=tenant_id)
        if record is None:
            raise AIError(ErrorCode.STORAGE_NOT_FOUND)
        entries = await self._history_tree(record, tenant_id)
        sources: list[_HistorySource] = []
        for item, depth in entries:
            for segment_sequence in await self._segment_sequences(item, tenant_id):
                sources.append(
                    _HistorySource(
                        item,
                        depth,
                        segment_sequence,
                        (
                            item.created_at.astimezone(timezone.utc),
                            depth,
                            item.execution_id,
                            segment_sequence,
                        ),
                    )
                )
        page = await self._history_page(
            sources,
            cursor=cursor,
            tenant_id=tenant_id,
            execution_id=execution_id,
            limit=limit,
        )
        selected = tuple(occurrence.item for occurrence in page[:limit])
        next_cursor = None
        if len(page) > limit:
            next_cursor = _history_cursor(
                tenant_id,
                execution_id,
                page[limit],
                self._cursor_signer,
            )
        _logger.debug(
            "execution history projected page: execution=%s items=%s",
            execution_id,
            len(selected),
        )
        return Page(selected, next_cursor)

    async def _history_page(
        self,
        sources: Sequence[_HistorySource],
        *,
        cursor: str | None,
        tenant_id: str,
        execution_id: str,
        limit: int,
    ) -> list[_HistoryOccurrence]:
        cursor_coordinate = _decode_history_cursor(
            cursor,
            tenant_id=tenant_id,
            execution_id=execution_id,
            signer=self._cursor_signer,
        )
        source_by_identity = {
            (source.record.execution_id, source.segment_sequence): source
            for source in sources
        }
        cursor_source: _HistorySource | None = None
        if cursor_coordinate is not None:
            cursor_source = source_by_identity.get(cursor_coordinate[:2])
            if cursor_source is None:
                raise AIError(ErrorCode.CURSOR_INVALID)

        iterators: list[AsyncGenerator[_HistoryOccurrence, None]] = []
        pending: list[
            tuple[
                tuple[object, ...],
                int,
                _HistoryOccurrence,
                AsyncGenerator[_HistoryOccurrence, None],
            ]
        ] = []
        cursor_prefix = (
            None if cursor_source is None else cursor_source.merge_prefix
        )
        try:
            for index, source in enumerate(sources):
                start_message_index = 0
                start_item_offset = 0
                if cursor_coordinate is not None:
                    if cursor_prefix is not None and source.merge_prefix < cursor_prefix:
                        continue
                    if source is cursor_source:
                        start_message_index = cursor_coordinate[2]
                        start_item_offset = cursor_coordinate[3]
                iterator = self._iter_history_source(
                    source,
                    tenant_id=tenant_id,
                    start_message_index=start_message_index,
                    start_item_offset=start_item_offset,
                    from_cursor=source is cursor_source,
                )
                iterators.append(iterator)
                try:
                    occurrence = await iterator.__anext__()
                except StopAsyncIteration:
                    continue
                heapq.heappush(
                    pending,
                    (occurrence.merge_key, index, occurrence, iterator),
                )

            page: list[_HistoryOccurrence] = []
            while pending and len(page) < limit + 1:
                _merge_key, index, occurrence, iterator = heapq.heappop(pending)
                page.append(occurrence)
                try:
                    next_occurrence = await iterator.__anext__()
                except StopAsyncIteration:
                    continue
                heapq.heappush(
                    pending,
                    (next_occurrence.merge_key, index, next_occurrence, iterator),
                )
            return page
        finally:
            for iterator in iterators:
                await iterator.aclose()

    async def _iter_history_source(
        self,
        source: _HistorySource,
        *,
        tenant_id: str,
        start_message_index: int,
        start_item_offset: int,
        from_cursor: bool,
    ) -> AsyncGenerator[_HistoryOccurrence, None]:
        run_id = step_run_id(
            namespace=self._namespace,
            tenant_id=tenant_id,
            execution_id=source.record.execution_id,
            segment_sequence=source.segment_sequence,
        )
        messages = self._message_range(
            run_id,
            start=start_message_index,
            from_cursor=from_cursor,
        )
        first = True
        message_index = start_message_index
        saw_message = False
        async for message in messages:
            saw_message = True
            projected = tuple(_project_message(message))
            item_offset = start_item_offset if first else 0
            if first and item_offset > len(projected):
                raise AIError(ErrorCode.CURSOR_INVALID)
            first = False
            for projected_offset in range(item_offset, len(projected)):
                value = projected[projected_offset]
                yield _HistoryOccurrence(
                    ExecutionHistoryItem(
                        source.record.execution_id,
                        message_index + 1,
                        value.item_kind,
                        value.content,
                        value.tool_name,
                        value.tool_call_id,
                    ),
                    source.record.execution_id,
                    source.segment_sequence,
                    message_index,
                    projected_offset,
                    (*source.merge_prefix, message_index, projected_offset),
                )
            message_index += 1
        if from_cursor and not saw_message:
            raise AIError(ErrorCode.CURSOR_INVALID)
        if (
            not saw_message
            and source.record.status is ExecutionStatus.SUCCEEDED
            and source.segment_sequence == source.record.agent_run_sequence
        ):
            raise AIError(ErrorCode.EXECUTION_HISTORY_UNAVAILABLE)

    async def _message_range(
        self,
        run_id: str,
        *,
        start: int,
        from_cursor: bool,
    ) -> AsyncIterator[object]:
        if isinstance(self._store, _RangedTranscriptStore):
            total = await self._store.transcript_message_count(run_id)
            if start > total or from_cursor and start == total:
                raise AIError(ErrorCode.CURSOR_INVALID)
            async for message in self._store.iter_message_range(
                run_id=run_id,
                start=start,
                end=total,
            ):
                yield message
            return
        index = 0
        async for message in self._store.iter_messages(run_id=run_id):
            if index >= start:
                yield message
            index += 1
        if from_cursor and index <= start:
            raise AIError(ErrorCode.CURSOR_INVALID)

    async def transcript(
        self, execution_id: str, *, tenant_id: str, cursor: str | None, limit: int
    ) -> Page[TranscriptItem]:
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
        final_run_id = step_run_id(
            namespace=self._namespace,
            tenant_id=tenant_id,
            execution_id=execution_id,
            segment_sequence=record.agent_run_sequence,
        )
        message_index, item_offset = _decode_transcript_cursor(
            cursor,
            tenant_id=tenant_id,
            execution_id=execution_id,
            run_id=final_run_id,
            signer=self._cursor_signer,
        )
        messages = self._store.iter_messages(run_id=final_run_id)
        conversation_id = step_conversation_id(
            namespace=self._namespace,
            tenant_id=tenant_id,
            execution_id=execution_id,
        )
        projected, next_coordinate = await _read_projected_page(
            messages,
            start_message_index=message_index,
            start_item_offset=item_offset,
            project=lambda message: _transcript_message_values(
                message,
                conversation_id,
            ),
            limit=limit,
        )
        if not projected:
            if record.status is ExecutionStatus.SUCCEEDED:
                raise AIError(ErrorCode.EXECUTION_HISTORY_UNAVAILABLE)
            return Page((), None)
        selected = tuple(
            TranscriptItem(execution_id, source_message_index + 1, value)
            for source_message_index, _source_item_offset, value in projected
        )
        next_cursor = None
        if next_coordinate is not None:
            next_cursor = _transcript_cursor(
                tenant_id,
                execution_id,
                final_run_id,
                next_coordinate[0],
                next_coordinate[1],
                self._cursor_signer,
            )
        return Page(selected, next_cursor)

    async def _history_tree(
        self, selected: ExecutionRecord, tenant_id: str
    ) -> list[tuple[ExecutionRecord, int]]:
        if selected.tenant_id != tenant_id:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if selected.lineage_kind.value == "SUBAGENT":
            if (
                selected.parent_execution_id is None
                or selected.root_execution_id == selected.execution_id
            ):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            return [(selected, 1)]
        if (
            selected.parent_execution_id is not None
            or selected.lineage_kind.value
            not in {"RUN", "RETRY", "FORK", "SESSION_RESUME"}
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

        result = [(selected, 0)]
        visited = {selected.execution_id}
        for child in await self._executions.list_children(
            selected.execution_id, tenant_id=tenant_id
        ):
            if (
                child.execution_id in visited
                or child.tenant_id != tenant_id
                or child.lineage_kind.value != "SUBAGENT"
                or child.parent_execution_id != selected.execution_id
                or child.root_execution_id != selected.root_execution_id
            ):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            visited.add(child.execution_id)
            result.append((child, 1))
        return result

    async def _segment_events(
        self, record: ExecutionRecord, tenant_id: str
    ) -> list[tuple[int, list[StepEvent]]]:
        if record.agent_run_sequence < 0:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if (
            record.status is ExecutionStatus.SUCCEEDED
            and record.agent_run_sequence == 0
        ):
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

    async def _segment_sequences(
        self, record: ExecutionRecord, tenant_id: str
    ) -> tuple[int, ...]:
        if record.agent_run_sequence < 0:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if (
            record.status is ExecutionStatus.SUCCEEDED
            and record.agent_run_sequence == 0
        ):
            raise AIError(ErrorCode.EXECUTION_HISTORY_UNAVAILABLE)
        conversation_id = step_conversation_id(
            namespace=self._namespace,
            tenant_id=tenant_id,
            execution_id=record.execution_id,
        )
        result: list[int] = []
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
            result.append(sequence)
        return tuple(result)


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
        cursor_values = None if cursor is None else _decode_session_history_cursor(
            cursor,
            tenant_id,
            session_id,
            self._cursor_signer,
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
        message_index = 0 if cursor_values is None else cursor_values[1]
        item_offset = 0 if cursor_values is None else cursor_values[2]
        if history_store is not None:
            history_id = continuation_history_id
            if history_id is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            total_messages = await history_store.session_message_count(
                history_id,
                tenant_id=tenant_id,
            )
            if message_index > total_messages:
                raise AIError(ErrorCode.CURSOR_INVALID)
            messages = history_store.iter_session_message_range(
                history_id,
                tenant_id=tenant_id,
                start=message_index,
                end=total_messages,
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
            total_messages = len(snapshot.messages)
            if message_index > total_messages:
                raise AIError(ErrorCode.CURSOR_INVALID)
            messages = _iter_sequence(snapshot.messages, start=message_index)
        projected, next_coordinate = await _read_projected_page(
            messages,
            start_message_index=message_index,
            start_item_offset=item_offset,
            project=_project_message,
            limit=limit,
        )
        selected = tuple(
            SessionHistoryItem(
                item_message_index + 1,
                item.item_kind,
                item.content,
                item.tool_name,
                item.tool_call_id,
            )
            for item_message_index, _item_offset, item in projected
        )
        next_cursor = None
        if next_coordinate is not None:
            next_cursor = _session_history_cursor(
                tenant_id,
                session_id,
                history_id,
                next_coordinate[0],
                next_coordinate[1],
                self._cursor_signer,
            )
        _logger.debug(
            "session history projected page: session=%s history=%s "
            "message_start=%s item_offset=%s items=%s",
            session_id,
            history_id,
            message_index,
            item_offset,
            len(selected),
        )
        return Page(selected, next_cursor)


def _trace_item(
    record: ExecutionRecord,
    segment_sequence: int,
    depth: int,
    ordinal: int,
    event: StepEvent,
) -> "ExecutionTraceItem | None":
    mapping = {
        "model_request_started": ("MODEL_REQUEST", "STARTED"),
        "model_request_completed": ("MODEL_RESPONSE", "SUCCEEDED"),
        "model_request_failed": ("MODEL_RESPONSE", "FAILED"),
        "tool_call_started": ("TOOL_CALL", "STARTED"),
        "tool_call_completed": ("TOOL_RESULT", "SUCCEEDED"),
        "tool_call_failed": ("TOOL_ERROR", "FAILED"),
    }
    value = mapping.get(event.kind)
    if value is None:
        return None
    kind, status = value
    payload = {
        "kind": kind,
        "status": status,
        "step_index": event.step_index,
        "segment_sequence": segment_sequence,
        "scope": "root" if depth == 0 else "subagent",
        "depth": depth,
        "occurred_at": _event_timestamp(event).isoformat(),
    }
    observation_id = event.metadata.get(OBSERVATION_ID_METADATA_KEY)
    if observation_id is not None:
        if not observation_id:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        payload["observation_id"] = observation_id
    duration_ns = event.metadata.get(DURATION_NS_METADATA_KEY)
    if duration_ns is not None:
        if not duration_ns.isdigit():
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        payload["duration_ns"] = int(duration_ns)
    request_sequence = event.metadata.get(REQUEST_SEQUENCE_METADATA_KEY)
    request_purpose = event.metadata.get(REQUEST_PURPOSE_METADATA_KEY)
    if request_sequence is not None or request_purpose is not None:
        if (
            request_sequence is None
            or not request_sequence.isdigit()
            or int(request_sequence) < 1
            or request_purpose not in {"agent", "compaction"}
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        payload["request_sequence"] = int(request_sequence)
        payload["purpose"] = request_purpose
    retry_value = event.metadata.get(OUTPUT_RETRY_INDEX_METADATA_KEY)
    if retry_value is not None:
        if not retry_value.isdigit() or int(retry_value) < 1:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        payload["output_retry_index"] = int(retry_value)
    if kind == "MODEL_RESPONSE":
        payload["token_usage"] = (
            _model_token_usage(event) if status == "SUCCEEDED" else None
        )
    if event.agent_name is not None:
        payload["agent_name"] = event.agent_name
    if event.tool_call_id is not None:
        payload["tool_call_id"] = event.tool_call_id
    if event.tool_name is not None:
        payload["tool_name"] = event.tool_name
    if depth > 0:
        payload["child_execution_id"] = record.execution_id
    return ExecutionTraceItem(record.execution_id, ordinal + 1, payload)


def _model_token_usage(event: StepEvent) -> "dict[str, JsonValue] | None":
    metadata = event.metadata
    present = MODEL_USAGE_METADATA_KEYS.intersection(metadata)
    if not present:
        return None
    if (
        MODEL_USAGE_INPUT_METADATA_KEY not in metadata
        or MODEL_USAGE_OUTPUT_METADATA_KEY not in metadata
    ):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return {
        "input_tokens": _metadata_token(metadata, MODEL_USAGE_INPUT_METADATA_KEY),
        "output_tokens": _metadata_token(metadata, MODEL_USAGE_OUTPUT_METADATA_KEY),
        "cache_read_tokens": _metadata_token(
            metadata,
            MODEL_USAGE_CACHE_READ_METADATA_KEY,
            required=False,
        ),
        "cache_write_tokens": _metadata_token(
            metadata,
            MODEL_USAGE_CACHE_WRITE_METADATA_KEY,
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


def _validate_run(
    run: RunRecord, expected_id: str, conversation_id: str, sequence: int
) -> None:
    if (
        run.run_id != expected_id
        or run.conversation_id != conversation_id
        or run.metadata.get("segment_sequence") != str(sequence)
    ):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    agent_name = run.metadata.get("agent_name")
    if (
        agent_name is None
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", agent_name) is None
    ):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)


def _event_timestamp(event: StepEvent) -> datetime:
    return event.timestamp.astimezone(timezone.utc)


def _decode_history_cursor(
    cursor: str | None,
    *,
    tenant_id: str,
    execution_id: str,
    signer: CursorSigner,
) -> tuple[str, int, int, int] | None:
    if cursor is None:
        return None
    return _decode_source_cursor(
        cursor,
        tenant_id=tenant_id,
        resource_kind="execution_history",
        filter_digest=canonical_sha256({"execution_id": execution_id}),
        sort_key="canonical-source",
        projection_version=1,
        signer=signer,
    )


def _decode_source_cursor(
    cursor: str,
    *,
    tenant_id: str,
    resource_kind: str,
    filter_digest: str,
    sort_key: str,
    projection_version: int,
    signer: CursorSigner,
) -> tuple[str, int, int, int]:
    try:
        payload = signer.decode(cursor)
    except AIError as error:
        raise AIError(ErrorCode.CURSOR_INVALID) from error
    if (
        payload.cursor_version != 1
        or payload.tenant_id != tenant_id
        or payload.resource_kind != resource_kind
        or payload.filter_digest != filter_digest
        or payload.sort_key != sort_key
        or payload.source_execution_id is None
        or payload.segment_sequence is None
        or payload.next_message_index is None
        or payload.intra_message_item_offset is None
        or payload.projection_version != projection_version
    ):
        raise AIError(ErrorCode.CURSOR_INVALID)
    return (
        payload.source_execution_id,
        payload.segment_sequence,
        payload.next_message_index,
        payload.intra_message_item_offset,
    )


def _history_cursor(
    tenant_id: str,
    execution_id: str,
    occurrence: _HistoryOccurrence,
    signer: CursorSigner,
) -> str:
    return signer.encode(
        CursorPayload(
            1,
            tenant_id,
            "execution_history",
            canonical_sha256({"execution_id": execution_id}),
            "canonical-source",
            0,
            int(time.time()) + 3600,
            source_execution_id=occurrence.source_execution_id,
            segment_sequence=occurrence.segment_sequence,
            next_message_index=occurrence.message_index,
            intra_message_item_offset=occurrence.item_offset,
            projection_version=1,
        )
    )


def _trace_cursor_index(
    cursor: str | None,
    *,
    tenant_id: str,
    execution_id: str,
    signer: CursorSigner,
    occurrences: Sequence[_TraceOccurrence],
) -> int:
    if cursor is None:
        return 0
    try:
        payload = signer.decode(cursor)
    except AIError as error:
        raise AIError(ErrorCode.CURSOR_INVALID) from error
    if (
        payload.cursor_version != 1
        or payload.tenant_id != tenant_id
        or payload.resource_kind != "execution_trace"
        or payload.filter_digest != canonical_sha256({"execution_id": execution_id})
        or payload.sort_key != "durable-event"
        or payload.source_execution_id is None
        or payload.segment_sequence is None
        or payload.source_event_sequence is None
        or payload.projection_version != 1
    ):
        raise AIError(ErrorCode.CURSOR_INVALID)
    coordinate = (
        payload.source_execution_id,
        payload.segment_sequence,
        payload.source_event_sequence,
    )
    for index, occurrence in enumerate(occurrences):
        if (
            occurrence.source_execution_id,
            occurrence.segment_sequence,
            occurrence.event_sequence,
        ) == coordinate:
            return index
    raise AIError(ErrorCode.CURSOR_INVALID)


def _trace_cursor(
    tenant_id: str,
    execution_id: str,
    occurrence: _TraceOccurrence,
    signer: CursorSigner,
) -> str:
    return signer.encode(
        CursorPayload(
            1,
            tenant_id,
            "execution_trace",
            canonical_sha256({"execution_id": execution_id}),
            "durable-event",
            0,
            int(time.time()) + 3600,
            source_execution_id=occurrence.source_execution_id,
            segment_sequence=occurrence.segment_sequence,
            source_event_sequence=occurrence.event_sequence,
            projection_version=1,
        )
    )


def _decode_transcript_cursor(
    cursor: str | None,
    *,
    tenant_id: str,
    execution_id: str,
    run_id: str,
    signer: CursorSigner,
) -> tuple[int, int]:
    if cursor is None:
        return 0, 0
    try:
        payload = signer.decode(cursor)
    except AIError as error:
        raise AIError(ErrorCode.CURSOR_INVALID) from error
    if (
        payload.cursor_version != 1
        or payload.tenant_id != tenant_id
        or payload.resource_kind != "execution_transcript"
        or payload.filter_digest != canonical_sha256({"execution_id": execution_id})
        or payload.sort_key != run_id
        or payload.next_message_index is None
        or payload.intra_message_item_offset is None
        or payload.projection_version != 1
    ):
        raise AIError(ErrorCode.CURSOR_INVALID)
    return payload.next_message_index, payload.intra_message_item_offset


def _transcript_cursor(
    tenant_id: str,
    execution_id: str,
    run_id: str,
    message_index: int,
    item_offset: int,
    signer: CursorSigner,
) -> str:
    return signer.encode(
        CursorPayload(
            1,
            tenant_id,
            "execution_transcript",
            canonical_sha256({"execution_id": execution_id}),
            run_id,
            0,
            int(time.time()) + 3600,
            next_message_index=message_index,
            intra_message_item_offset=item_offset,
            projection_version=1,
        )
    )


def _decode_session_history_cursor(
    cursor: str,
    tenant_id: str,
    session_id: str,
    signer: CursorSigner,
) -> tuple[str, int, int]:
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
        or payload.next_message_index is None
        or payload.intra_message_item_offset is None
        or payload.projection_version != SESSION_HISTORY_VIEW_V1
    ):
        raise AIError(ErrorCode.CURSOR_INVALID)
    if payload.next_message_index < 0 or payload.intra_message_item_offset < 0:
        raise AIError(ErrorCode.CURSOR_INVALID)
    return (
        payload.history_id,
        payload.next_message_index,
        payload.intra_message_item_offset,
    )


def _session_history_cursor(
    tenant_id: str,
    session_id: str,
    history_id: str,
    next_message_index: int,
    intra_message_item_offset: int,
    signer: CursorSigner,
) -> str:
    return signer.encode(
        CursorPayload(
            1,
            tenant_id,
            "session_history",
            canonical_sha256({"session_id": session_id}),
            "session_history",
            0,
            int(time.time()) + 3600,
            history_id=history_id,
            next_message_index=next_message_index,
            intra_message_item_offset=intra_message_item_offset,
            projection_version=SESSION_HISTORY_VIEW_V1,
        )
    )


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


def _transcript_message_values(
    message: object, conversation_id: str
) -> tuple[str, ...]:
    del conversation_id
    if not isinstance(message, (ModelRequest, ModelResponse)):
        return ()
    return project_execution_transcript_message(message)


__all__ = ["StepExecutionHistoryReader", "StepSessionHistoryReader"]
