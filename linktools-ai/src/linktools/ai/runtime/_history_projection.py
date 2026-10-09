#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Project Runtime step facts into trace, transcript, and history views."""

import heapq
import json
from collections.abc import AsyncGenerator, AsyncIterator, Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Protocol, cast, runtime_checkable

from linktools.core import environ
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    ToolReturnPart,
    ToolCallPart,
    RetryPromptPart,
)

from ..core import (
    CursorSigner,
    ExecutionStatus,
    JsonValue,
    Page,
    ToolOperationStatus,
    canonical_sha256,
    agent_conversation_id as make_agent_conversation_id,
    agent_run_id as make_agent_run_id,
    validate_page_limit,
    validate_persistence_namespace,
)
from ..errors import AIError, ErrorCode
from ._cursor import decode_cursor as decode_runtime_cursor
from ._cursor import encode_cursor as encode_runtime_cursor
from ._journal import (
    DURATION_NS_METADATA_KEY,
    MODEL_USAGE_CACHE_READ_METADATA_KEY,
    MODEL_USAGE_CACHE_WRITE_METADATA_KEY,
    MODEL_USAGE_INPUT_METADATA_KEY,
    MODEL_USAGE_OUTPUT_METADATA_KEY,
    OBSERVATION_ID_METADATA_KEY,
    OUTPUT_RETRY_INDEX_METADATA_KEY,
    REQUEST_PURPOSE_METADATA_KEY,
    MODEL_REQUEST_SEQ_METADATA_KEY,
    MESSAGE_SEQ_METADATA_KEY,
    REPLAYED_TOOL_CALL_INDICES_METADATA_KEY,
)
from ._model_interaction import StagedModelInteraction, project_public_messages
from .service_api import (
    AttachmentFact,
    ExecutionHistoryItem,
    ExecutionTraceItem,
    ModelInteractionItem,
    ModelInteractionReadBoundary,
    ModelInteractionSubscription,
    SessionHistoryItem,
    UsageReadCutoff,
    UsageSummary,
    TranscriptItem,
)
from .state._contracts import (
    ExecutionRecord,
    ExecutionRepository,
    ModelInteractionRecord,
    SessionRepository,
    ToolOperationRecord,
)
from .state._step_contracts import AgentRunHistoryCapture, AgentRunRecord, StepEvent, AgentRunStore
from .state._views import (
    SESSION_HISTORY_VIEW_V1,
    project_execution_transcript_message,
    project_session_history_message,
)

_logger = environ.get_logger("ai.runtime.history_projection")
_EXECUTION_HISTORY_PROJECTION_VERSION = 1
_EXECUTION_TRACE_PROJECTION_VERSION = 1
_EXECUTION_TRANSCRIPT_PROJECTION_VERSION = 1
_MODEL_INTERACTION_PROJECTION_VERSION = 1
_ATTACHMENT_FACT_PROJECTION_VERSION = 1


@dataclass(frozen=True, slots=True)
class _ProjectedHistoryItem:
    item_kind: str
    content: JsonValue
    tool_name: "str | None" = None
    tool_call_id: "str | None" = None
    part_index: int | None = None


@dataclass(frozen=True, slots=True)
class _HistoryOccurrence:
    item: ExecutionHistoryItem
    source_execution_id: str
    agent_run_seq: int
    message_index: int
    item_offset: int
    merge_key: tuple[object, ...]


@dataclass(frozen=True, slots=True)
class _TraceOccurrence:
    item: ExecutionTraceItem
    source_execution_id: str
    agent_run_seq: int
    step_event_seq: int
    merge_key: tuple[object, ...]


@dataclass(frozen=True, slots=True)
class _InteractionOccurrence:
    key: tuple[object, ...]
    source_execution_id: str
    agent_run_seq: int
    depth: int
    interaction: ModelInteractionRecord | StagedModelInteraction


@dataclass(frozen=True, slots=True)
class _AttachmentOccurrence:
    key: tuple[int, int, str, str, int]
    item: AttachmentFact


@dataclass(frozen=True, slots=True)
class _HistorySource:
    record: ExecutionRecord
    depth: int
    agent_run_seq: int
    merge_prefix: tuple[object, ...]
    capture: AgentRunHistoryCapture


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


class _ExecutionHistoryStore(Protocol):
    async def get_agent_run(self, *, agent_run_id: str) -> AgentRunRecord | None: ...

    async def capture_history(
        self, agent_run_ids: Sequence[str], *, include_pending: bool = False,
    ) -> Mapping[str, AgentRunHistoryCapture]: ...

    async def read_pending_message(self, capture: AgentRunHistoryCapture) -> object | None: ...

    def iter_message_range(
        self, *, agent_run_id: str, start: int, end: int,
    ) -> AsyncIterator[object]: ...

    async def list_event_range(self, *, agent_run_id: str, start: int, end: int) -> list[StepEvent]: ...

    async def read_history_associations(
        self, *, agent_run_id: str, message_seqs: Sequence[int],
        tool_call_ids: Sequence[str], event_high_water: int,
    ) -> list[StepEvent]: ...

    async def list_trace_events(
        self, *, agent_run_id: str, event_high_water: int,
        after_timestamp: datetime | None, after_sequence: int, limit: int,
        model_request_seq: int | None = None, step_index: int | None = None,
        tool_call_id: str | None = None,
    ) -> list[tuple[int, StepEvent]]: ...

    async def list_model_interactions(
        self, *, agent_run_id: str, after_model_request_seq: int | None = None,
        limit: int | None = None,
    ) -> list[object]: ...

    async def resolve_model_interactions(self, interactions: Sequence[object]) -> Sequence[object]: ...


class _ToolOperationHistoryReader(Protocol):
    async def get_by_call_ids(
        self, agent_run_id: str, tool_call_ids: Sequence[str], *, tenant_id: str,
    ) -> tuple[ToolOperationRecord, ...]: ...


class _ModelInteractionNotifications(Protocol):
    def subscribe_model_interactions(
        self, agent_conversation_id: str,
    ) -> ModelInteractionSubscription: ...


class StepExecutionHistoryReader:
    """Own the adapter projection between AgentRunStore facts and Runtime views."""

    def __init__(
        self,
        *,
        namespace: str,
        executions: ExecutionRepository,
        store: _ExecutionHistoryStore,
        cursor_signer: CursorSigner,
        tool_operations: "_ToolOperationHistoryReader | None" = None,
        notifications: "_ModelInteractionNotifications | None" = None,
        durable_history_available: bool = True,
    ) -> None:
        try:
            validate_persistence_namespace(namespace)
        except AIError as error:
            raise ValueError("execution history namespace is invalid") from error
        self._namespace = namespace
        self._executions = executions
        self._store = store
        self._cursor_signer = cursor_signer
        self._tool_operations = tool_operations
        self._notifications = notifications
        self._durable_history_available = durable_history_available

    async def trace(
        self, execution_id: str, *, tenant_id: str, cursor: "str | None", limit: int,
        agent_run_seq: int | None = None,
        model_request_seq: int | None = None,
        step_index: int | None = None,
        tool_call_id: str | None = None,
    ) -> "Page[ExecutionTraceItem]":
        filters = {
            "agent_run_seq": agent_run_seq,
            "model_request_seq": model_request_seq,
            "step_index": step_index,
            "tool_call_id": tool_call_id,
        }
        record = await self._executions.get(execution_id, tenant_id=tenant_id)
        if record is None:
            raise AIError(ErrorCode.STORAGE_NOT_FOUND)
        limit = validate_page_limit(limit)
        entries = await self._history_tree(record, tenant_id)
        return await self._trace_page(
            execution_id, tenant_id=tenant_id, entries=entries,
            cursor=cursor, limit=limit, filters=filters,
        )

    async def _trace_page(
        self, execution_id: str, *, tenant_id: str,
        entries: Sequence[tuple[ExecutionRecord, int]], cursor: str | None,
        limit: int, filters: Mapping[str, JsonValue],
    ) -> Page[ExecutionTraceItem]:
        store = self._store
        sources = await self._capture_sources(entries, tenant_id, cast("int | None", filters["agent_run_seq"]))
        by_identity = {(source.record.execution_id, source.agent_run_seq): source for source in sources}
        state = _decode_trace_cursor(cursor, tenant_id=tenant_id, execution_id=execution_id,
                                     signer=self._cursor_signer, filters=filters)
        if state is None:
            coordinate = None
            cutoffs = tuple(sorted((identity[0], identity[1], source.capture.event_count)
                                   for identity, source in by_identity.items()))
        else:
            coordinate, cutoffs = state
        if any((execution, sequence) not in by_identity for execution, sequence, _ in cutoffs):
            raise AIError(ErrorCode.CURSOR_INVALID)
        captured_counts = {(execution, sequence): count for execution, sequence, count in cutoffs}
        cursor_key: tuple[object, ...] | None = None
        cursor_timestamp: datetime | None = None
        if coordinate is not None:
            source = by_identity.get(coordinate[:2])
            if source is None or coordinate[2] > captured_counts.get(coordinate[:2], 0):
                raise AIError(ErrorCode.CURSOR_INVALID)
            run_id = make_agent_run_id(namespace=self._namespace, tenant_id=tenant_id,
                                      execution_id=coordinate[0], agent_run_seq=coordinate[1])
            event, = await store.list_event_range(agent_run_id=run_id, start=coordinate[2] - 1, end=coordinate[2])
            mapped = _trace_item(source.record, source.agent_run_seq, source.depth, coordinate[2] - 1, event)
            if mapped is None or any(value is not None and mapped.payload.get(name) != value for name, value in filters.items()):
                raise AIError(ErrorCode.CURSOR_INVALID)
            cursor_timestamp = _event_timestamp(event)
            cursor_key = (cursor_timestamp, source.depth, coordinate[0], coordinate[1], coordinate[2], mapped.payload.get("kind", ""))

        async def occurrences(source: _HistorySource, high_water: int) -> AsyncGenerator[_TraceOccurrence, None]:
            if high_water > source.capture.event_count:
                raise AIError(ErrorCode.CURSOR_INVALID)
            if high_water == 0:
                return
            run_id = make_agent_run_id(namespace=self._namespace, tenant_id=tenant_id,
                                      execution_id=source.record.execution_id, agent_run_seq=source.agent_run_seq)
            after_timestamp = cursor_timestamp
            after_sequence = 0
            if cursor_key is not None and coordinate is not None:
                suffix = (source.depth, source.record.execution_id, source.agent_run_seq)
                if suffix < cursor_key[1:4]:
                    after_sequence = high_water
                elif suffix == cursor_key[1:4]:
                    after_sequence = coordinate[2] - 1
            while True:
                batch = await store.list_trace_events(
                    agent_run_id=run_id, event_high_water=high_water,
                    after_timestamp=after_timestamp, after_sequence=after_sequence,
                    limit=min(64, limit + 1),
                    model_request_seq=cast("int | None", filters["model_request_seq"]),
                    step_index=cast("int | None", filters["step_index"]),
                    tool_call_id=cast("str | None", filters["tool_call_id"]),
                )
                if not batch:
                    return
                for sequence, event in batch:
                    if source.capture.run is None or event.agent_conversation_id not in {None, source.capture.run.agent_conversation_id}:
                        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                    after_timestamp, after_sequence = _event_timestamp(event), sequence
                    mapped = _trace_item(source.record, source.agent_run_seq, source.depth, sequence - 1, event)
                    if mapped is None or any(value is not None and mapped.payload.get(name) != value for name, value in filters.items()):
                        continue
                    key = (after_timestamp, source.depth, source.record.execution_id, source.agent_run_seq,
                           sequence, mapped.payload.get("kind", ""))
                    if cursor_key is None or key >= cursor_key:
                        yield _TraceOccurrence(mapped, source.record.execution_id, source.agent_run_seq, sequence, key)

        iterators: list[AsyncGenerator[_TraceOccurrence, None]] = []
        pending: list[tuple[tuple[object, ...], int, _TraceOccurrence, AsyncGenerator[_TraceOccurrence, None]]] = []
        page: list[_TraceOccurrence] = []
        try:
            for index, (execution, sequence, high_water) in enumerate(cutoffs):
                iterator = occurrences(by_identity[(execution, sequence)], high_water)
                iterators.append(iterator)
                try:
                    value = await iterator.__anext__()
                except StopAsyncIteration:
                    continue
                heapq.heappush(pending, (value.merge_key, index, value, iterator))
            while pending and len(page) <= limit:
                _key, index, value, iterator = heapq.heappop(pending)
                page.append(value)
                if len(page) > limit:
                    break
                try:
                    following = await iterator.__anext__()
                except StopAsyncIteration:
                    continue
                heapq.heappush(pending, (following.merge_key, index, following, iterator))
        finally:
            for iterator in iterators:
                await iterator.aclose()
        return Page(tuple(value.item for value in page[:limit]),
                    None if len(page) <= limit else _trace_cursor(
                        tenant_id, execution_id, page[limit], cutoffs, self._cursor_signer, filters=filters))

    async def history(
        self,
        execution_id: str,
        *,
        tenant_id: str,
        cursor: str | None,
        limit: int,
        model_request_seq: int | None = None,
        step_index: int | None = None,
        agent_run_seq: int | None = None,
        tool_call_id: str | None = None,
        message_seq: int | None = None,
        part_index: int | None = None,
    ) -> "Page[ExecutionHistoryItem]":
        filters = {
            "agent_run_seq": agent_run_seq,
            "model_request_seq": model_request_seq,
            "step_index": step_index,
            "tool_call_id": tool_call_id,
            "message_seq": message_seq,
            "part_index": part_index,
        }
        limit = validate_page_limit(limit)
        record = await self._executions.get(execution_id, tenant_id=tenant_id)
        if record is None:
            raise AIError(ErrorCode.STORAGE_NOT_FOUND)
        entries = await self._history_tree(record, tenant_id)
        current_sources = await self._capture_sources(entries, tenant_id, agent_run_seq, include_pending=True)
        source_by_identity = {
            (source.record.execution_id, source.agent_run_seq): source
            for source in current_sources
        }
        cursor_state = _decode_history_cursor(
            cursor,
            tenant_id=tenant_id,
            execution_id=execution_id,
            signer=self._cursor_signer,
            filters=filters,
        )
        if cursor_state is None:
            cursor_coordinate = None
            fixed_cutoffs = tuple(
                (source.record.execution_id, source.agent_run_seq, source.capture.message_count,
                 source.capture.event_count, source.capture.pending_keys)
                for source in current_sources
            )
        else:
            cursor_coordinate, fixed_cutoffs = cursor_state
        cutoff_by_identity = {
            (source_execution_id, agent_run_seq): (
                message_count,
                event_count,
                tail_keys,
            )
            for (
                source_execution_id,
                agent_run_seq,
                message_count,
                event_count,
                tail_keys,
            ) in fixed_cutoffs
        }
        if len(cutoff_by_identity) != len(fixed_cutoffs):
            raise AIError(ErrorCode.CURSOR_INVALID)
        if any(identity not in source_by_identity for identity in cutoff_by_identity):
            raise AIError(ErrorCode.CURSOR_INVALID)
        sources = tuple(
            source
            for source in current_sources
            if (source.record.execution_id, source.agent_run_seq) in cutoff_by_identity
        )
        page = await self._history_page(
            sources,
            cursor_coordinate=cursor_coordinate,
            high_waters=cutoff_by_identity,
            tenant_id=tenant_id,
            limit=limit,
            tool_call_id=tool_call_id,
            model_request_seq=model_request_seq,
            step_index=step_index,
            message_seq=message_seq,
            part_index=part_index,
        )
        selected = tuple(occurrence.item for occurrence in page[:limit])
        next_cursor = None
        if len(page) > limit:
            next_cursor = _history_cursor(
                tenant_id,
                execution_id,
                page[limit],
                fixed_cutoffs,
                self._cursor_signer,
                filters=filters,
            )
        _logger.debug(
            "execution history projected page: execution=%s items=%s",
            execution_id,
            len(selected),
        )
        return Page(selected, next_cursor)

    async def capture_model_interaction_cutoffs(
        self, execution_id: str, *, tenant_id: str,
    ) -> ModelInteractionReadBoundary:
        """Capture committed identities and declared run coverage without tree discovery."""
        record = await self._executions.get(execution_id, tenant_id=tenant_id)
        if record is None:
            raise AIError(ErrorCode.STORAGE_NOT_FOUND)
        if record.agent_run_seq < 0:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        run_ids = tuple(make_agent_run_id(
            namespace=self._namespace, tenant_id=tenant_id,
            execution_id=execution_id, agent_run_seq=sequence,
        ) for sequence in range(1, record.agent_run_seq + 1))
        captured = await self._store.capture_history(run_ids)
        cutoffs: list[UsageReadCutoff] = []
        available = self._durable_history_available
        conversation_id = make_agent_conversation_id(
            namespace=self._namespace, tenant_id=tenant_id, execution_id=execution_id,
        )
        for sequence, run_id in enumerate(run_ids, 1):
            value = captured.get(run_id)
            if value is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if value.run is None:
                available = False
                continue
            _validate_agent_run(value.run, run_id, conversation_id, sequence)
            cutoffs.append(UsageReadCutoff(execution_id, sequence, value.model_interaction_count))
        if record.agent_run_seq == 0 and record.status is ExecutionStatus.SUCCEEDED:
            available = False
        local_available = (not self._durable_history_available and bool(run_ids)
                           and captured[run_ids[-1]].run is not None)
        durable_cutoffs = tuple(cutoffs) if self._durable_history_available else tuple(
            UsageReadCutoff(value.execution_id, value.agent_run_seq, 0) for value in cutoffs
        )
        return ModelInteractionReadBoundary(tuple(cutoffs), durable_cutoffs, local_available, available)

    async def read_model_interaction_metadata(
        self, execution_id: str, *, tenant_id: str, agent_run_seq: int,
        after_model_request_seq: int, through_model_request_seq: int, limit: int = 200,
    ) -> tuple[ModelInteractionItem, ...]:
        """Read a bounded suffix or one active identity from an explicit cutoff."""
        limit = validate_page_limit(limit)
        if (isinstance(agent_run_seq, bool) or not isinstance(agent_run_seq, int)
                or agent_run_seq < 1
                or any(isinstance(value, bool) or not isinstance(value, int) or value < 0
                       for value in (after_model_request_seq, through_model_request_seq))
                or after_model_request_seq > through_model_request_seq):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        record = await self._executions.get(execution_id, tenant_id=tenant_id)
        if record is None:
            raise AIError(ErrorCode.STORAGE_NOT_FOUND)
        if agent_run_seq > record.agent_run_seq:
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        count = min(limit, through_model_request_seq - after_model_request_seq)
        if count == 0:
            return ()
        run_id = make_agent_run_id(
            namespace=self._namespace, tenant_id=tenant_id,
            execution_id=execution_id, agent_run_seq=agent_run_seq,
        )
        run = await self._store.get_agent_run(agent_run_id=run_id)
        if run is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        _validate_agent_run(run, run_id, make_agent_conversation_id(
            namespace=self._namespace, tenant_id=tenant_id, execution_id=execution_id,
        ), agent_run_seq)
        interactions = await self._store.list_model_interactions(
            agent_run_id=run_id, after_model_request_seq=after_model_request_seq, limit=count,
        )
        selected: dict[int, ModelInteractionItem] = {}
        for interaction in interactions:
            if (not isinstance(interaction, (ModelInteractionRecord, StagedModelInteraction)) or interaction.agent_run_id != run_id
                    or interaction.model_request_seq in selected
                    or not after_model_request_seq < interaction.model_request_seq <= after_model_request_seq + count):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            selected[interaction.model_request_seq] = self._project_model_interaction(
                interaction, execution_id, agent_run_seq, 0, None, include_content=False,
            )
        expected = tuple(range(after_model_request_seq + 1, after_model_request_seq + count + 1))
        if tuple(sorted(selected)) != expected:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return tuple(selected[sequence] for sequence in expected)

    async def subscribe_model_interactions(
        self, execution_id: str, *, tenant_id: str,
    ) -> ModelInteractionSubscription | None:
        if await self._executions.get(execution_id, tenant_id=tenant_id) is None:
            raise AIError(ErrorCode.STORAGE_NOT_FOUND)
        if self._notifications is None:
            return None
        return self._notifications.subscribe_model_interactions(make_agent_conversation_id(
            namespace=self._namespace, tenant_id=tenant_id, execution_id=execution_id,
        ))

    async def model_interactions(
        self,
        execution_id: str,
        *,
        tenant_id: str,
        cursor: "str | None",
        limit: int,
        cutoffs: "tuple[UsageReadCutoff, ...] | None" = None,
        include_content: bool = True,
    ) -> Page[ModelInteractionItem]:
        limit = validate_page_limit(limit)
        record = await self._executions.get(execution_id, tenant_id=tenant_id)
        if record is None:
            raise AIError(ErrorCode.STORAGE_NOT_FOUND)
        entries = await self._recursive_history_tree(record, tenant_id)
        current_sources = await self._capture_sources(entries, tenant_id)
        source_by_identity = {
            (source.record.execution_id, source.agent_run_seq): source
            for source in current_sources
        }

        cursor_state = _decode_model_interaction_cursor(
            cursor,
            tenant_id=tenant_id,
            execution_id=execution_id,
            signer=self._cursor_signer,
        )
        explicit_cutoffs = _normalize_usage_cutoffs(cutoffs)
        if cursor_state is not None:
            cursor_coordinate, cursor_cutoffs = cursor_state
            if (
                explicit_cutoffs is not None
                and explicit_cutoffs != cursor_cutoffs
            ):
                raise AIError(ErrorCode.CURSOR_INVALID)
            fixed_cutoffs = cursor_cutoffs
        else:
            cursor_coordinate = None
            if explicit_cutoffs is None:
                fixed_cutoffs = tuple(UsageReadCutoff(
                    source.record.execution_id, source.agent_run_seq, source.capture.model_interaction_count,
                ) for source in current_sources)
            else:
                fixed_cutoffs = explicit_cutoffs

        cutoff_by_identity = {
            (value.execution_id, value.agent_run_seq): value.model_request_seq
            for value in fixed_cutoffs
        }
        if len(cutoff_by_identity) != len(fixed_cutoffs):
            raise AIError(ErrorCode.CURSOR_INVALID)
        if any(identity not in source_by_identity for identity in cutoff_by_identity):
            raise AIError(ErrorCode.CURSOR_INVALID)
        sources = tuple(
            source
            for source in current_sources
            if (source.record.execution_id, source.agent_run_seq) in cutoff_by_identity
        )
        cursor_source = None
        if cursor_coordinate is not None:
            cursor_source = source_by_identity.get(cursor_coordinate[:2])
            if (
                cursor_source is None
                or cursor_coordinate[:2] not in cutoff_by_identity
                or cursor_coordinate[2] > cutoff_by_identity[cursor_coordinate[:2]]
            ):
                raise AIError(ErrorCode.CURSOR_INVALID)
        cursor_prefix = None if cursor_source is None else cursor_source.merge_prefix

        page: list[_InteractionOccurrence] = []
        for source in sources:
            if cursor_prefix is not None and source.merge_prefix < cursor_prefix:
                continue
            identity = (source.record.execution_id, source.agent_run_seq)
            high_water = cutoff_by_identity[identity]
            after_model_request_seq = (
                cursor_coordinate[2]
                if cursor_coordinate is not None and source is cursor_source
                else 0
            )
            available = high_water - after_model_request_seq
            if available <= 0:
                continue
            remaining = limit + 1 - len(page)
            if remaining <= 0:
                break
            agent_run_id = make_agent_run_id(
                namespace=self._namespace,
                tenant_id=tenant_id,
                execution_id=source.record.execution_id,
                agent_run_seq=source.agent_run_seq,
            )
            fetch_limit = min(remaining, available)
            interactions = await self._store.list_model_interactions(
                agent_run_id=agent_run_id, after_model_request_seq=after_model_request_seq, limit=fetch_limit,
            )
            if len(interactions) != fetch_limit:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            for expected_sequence, interaction in enumerate(interactions, after_model_request_seq + 1):
                if (not isinstance(interaction, (ModelInteractionRecord, StagedModelInteraction))
                        or interaction.agent_run_id != agent_run_id
                        or interaction.model_request_seq != expected_sequence):
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                page.append(
                    _InteractionOccurrence(
                        (*source.merge_prefix, interaction.model_request_seq),
                        source.record.execution_id,
                        source.agent_run_seq,
                        source.depth,
                        interaction,
                    )
                )

        selected_occurrences = page[:limit]
        selected_items: dict[tuple[str, int], ModelInteractionItem] = {}
        for occurrence_group in _interaction_occurrence_groups(selected_occurrences):
            resolved_values: tuple[object, ...]
            if include_content:
                resolved_values = tuple(await self._store.resolve_model_interactions(
                    tuple(value.interaction for value in occurrence_group)
                ))
                if len(resolved_values) != len(occurrence_group):
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            else:
                resolved_values = (None,) * len(occurrence_group)
            for occurrence, resolved_context in zip(
                occurrence_group,
                resolved_values,
                strict=True,
            ):
                selected_items[
                    (occurrence.interaction.agent_run_id, occurrence.interaction.model_request_seq)
                ] = self._project_model_interaction(
                    occurrence.interaction,
                    occurrence.source_execution_id,
                    occurrence.agent_run_seq,
                    occurrence.depth,
                    resolved_context,
                    include_content=include_content,
                )
        selected = tuple(
            selected_items[
                (value.interaction.agent_run_id, value.interaction.model_request_seq)
            ]
            for value in selected_occurrences
        )
        next_cursor = None
        if len(page) > limit and selected_occurrences:
            next_cursor = _model_interaction_cursor(
                tenant_id,
                execution_id,
                selected_occurrences[-1],
                fixed_cutoffs,
                self._cursor_signer,
            )
        _logger.debug(
            "model interactions projected page: execution=%s items=%s",
            execution_id,
            len(selected),
        )
        return Page(selected, next_cursor)


    async def attachment_facts(
        self,
        execution_id: str,
        *,
        tenant_id: str,
        cursor: "str | None",
        limit: int,
    ) -> Page[AttachmentFact]:
        limit = validate_page_limit(limit)
        record = await self._executions.get(execution_id, tenant_id=tenant_id)
        if record is None:
            raise AIError(ErrorCode.STORAGE_NOT_FOUND)

        cursor_state = _decode_attachment_fact_cursor(
            cursor,
            tenant_id=tenant_id,
            execution_id=execution_id,
            signer=self._cursor_signer,
        )
        after_key = None if cursor_state is None else cursor_state[0]
        fixed_cutoffs = None if cursor_state is None else dict(cursor_state[1])
        target = limit + 1

        facts: list[_AttachmentOccurrence] = []
        accepted_ids: set[str] = set()
        view = record.stored_user_input.view
        if view is not None:
            raw_attachments = view.get("attachments", [])
            if not isinstance(raw_attachments, list):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            for occurrence in _attachment_occurrences(record.execution_id, raw_attachments):
                fact = occurrence.item
                if fact.fact != "accepted":
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                accepted_ids.add(fact.attachment_id)
                if after_key is None or occurrence.key > after_key:
                    facts.append(occurrence)

        cutoffs: list[tuple[int, int]] = []
        if record.binding_kind != "task":
            boundary = await self.capture_model_interaction_cutoffs(execution_id, tenant_id=tenant_id)
            current = {value.agent_run_seq: value.model_request_seq for value in boundary.cutoffs}
            sequences = tuple(current) if fixed_cutoffs is None else tuple(sorted(fixed_cutoffs))
            if any(sequence not in current for sequence in sequences):
                raise AIError(ErrorCode.CURSOR_INVALID)
            agent_run_high_waters: list[tuple[int, str, int]] = []
            for agent_run_seq in sequences:
                high_water = current[agent_run_seq] if fixed_cutoffs is None else fixed_cutoffs[agent_run_seq]
                if high_water > current[agent_run_seq]:
                    raise AIError(ErrorCode.CURSOR_INVALID)
                cutoffs.append((agent_run_seq, high_water))
                agent_run_high_waters.append((agent_run_seq, make_agent_run_id(
                    namespace=self._namespace, tenant_id=tenant_id, execution_id=execution_id,
                    agent_run_seq=agent_run_seq,
                ), high_water))

            for agent_run_seq, agent_run_id, high_water in agent_run_high_waters:
                after_sequence = 0
                while after_sequence < high_water and len(facts) < target:
                    batch_limit = min(256, high_water - after_sequence)
                    values = await self._store.list_model_interactions(
                        agent_run_id=agent_run_id, after_model_request_seq=after_sequence, limit=batch_limit,
                    )
                    if len(values) != batch_limit:
                        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                    for value in values:
                        if (
                            not isinstance(
                                value, (ModelInteractionRecord, StagedModelInteraction)
                            )
                            or value.agent_run_id != agent_run_id
                            or value.model_request_seq != after_sequence + 1
                        ):
                            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                        after_sequence = value.model_request_seq
                        for occurrence in _attachment_occurrences(
                            record.execution_id,
                            value.attachments,
                            agent_run_seq=agent_run_seq,
                            model_request_seq=value.model_request_seq,
                            step_index=value.step_index,
                        ):
                            fact = occurrence.item
                            if fact.fact == "accepted":
                                if fact.attachment_id in accepted_ids:
                                    continue
                                accepted_ids.add(fact.attachment_id)
                            if after_key is not None and occurrence.key <= after_key:
                                continue
                            facts.append(occurrence)
                            if len(facts) >= target:
                                break
                        if len(facts) >= target:
                            break

                if len(facts) >= target:
                    break

        selected = tuple(value.item for value in facts[:limit])
        next_cursor = None
        if len(facts) > limit:
            next_cursor = _attachment_fact_cursor(
                tenant_id,
                execution_id,
                facts[limit - 1].key,
                tuple(cutoffs),
                self._cursor_signer,
            )
        return Page(selected, next_cursor)

    async def usage(
        self,
        execution_id: str,
        *,
        tenant_id: str,
        cutoffs: "tuple[UsageReadCutoff, ...] | None" = None,
    ) -> UsageSummary:
        record = await self._executions.get(
            execution_id,
            tenant_id=tenant_id,
        )
        if record is None:
            raise AIError(ErrorCode.STORAGE_NOT_FOUND)
        if record.binding_kind == "task":
            return UsageSummary()

        boundary = await self.capture_model_interaction_cutoffs(execution_id, tenant_id=tenant_id)
        current_cutoffs = {value.agent_run_seq: value.model_request_seq for value in boundary.cutoffs}
        explicit_cutoffs = _normalize_usage_cutoffs(cutoffs)
        if explicit_cutoffs is None:
            fixed_cutoffs = list(boundary.cutoffs)
        else:
            if any(value.execution_id != record.execution_id for value in explicit_cutoffs):
                raise AIError(ErrorCode.CURSOR_INVALID)
            fixed_cutoffs = list(explicit_cutoffs)
        if any(value.agent_run_seq not in current_cutoffs for value in fixed_cutoffs):
            raise AIError(ErrorCode.CURSOR_INVALID)

        logical_requests = 0
        succeeded_requests = 0
        failed_requests = 0
        cancelled_requests = 0
        running_requests = 0
        interrupted_requests = 0
        output_correction_retries = 0
        input_tokens = 0
        output_tokens = 0
        cache_read_tokens = 0
        cache_write_tokens = 0
        model_duration_ns = 0
        unknown_usage_requests = 0
        unknown_duration_requests = 0

        for cutoff in fixed_cutoffs:
            agent_run_id = make_agent_run_id(
                namespace=self._namespace,
                tenant_id=tenant_id,
                execution_id=record.execution_id,
                agent_run_seq=cutoff.agent_run_seq,
            )
            if cutoff.model_request_seq > current_cutoffs[cutoff.agent_run_seq]:
                raise AIError(ErrorCode.CURSOR_INVALID)
            after_sequence = 0
            while after_sequence < cutoff.model_request_seq:
                batch_limit = min(500, cutoff.model_request_seq - after_sequence)
                values = await self._store.list_model_interactions(
                    agent_run_id=agent_run_id,
                    after_model_request_seq=after_sequence,
                    limit=batch_limit,
                )
                if len(values) != batch_limit:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                for raw in values:
                    if not isinstance(raw, (ModelInteractionRecord, StagedModelInteraction)):
                        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                    expected = after_sequence + 1
                    if (
                        raw.agent_run_id != agent_run_id
                        or raw.model_request_seq != expected
                    ):
                        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                    after_sequence = raw.model_request_seq
                    logical_requests += 1
                    if raw.duration_ns is None:
                        unknown_duration_requests += 1
                    else:
                        model_duration_ns += raw.duration_ns
                    if raw.status == "SUCCEEDED":
                        succeeded_requests += 1
                    elif raw.status == "FAILED":
                        failed_requests += 1
                    elif raw.status == "CANCELLED":
                        cancelled_requests += 1
                    elif raw.status == "RUNNING":
                        running_requests += 1
                    elif raw.status == "INTERRUPTED":
                        interrupted_requests += 1
                    else:
                        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                    if raw.output_retry_index is not None:
                        output_correction_retries += 1
                    request_usage = raw.usage
                    if request_usage is None:
                        unknown_usage_requests += 1
                    else:
                        input_tokens += request_usage.input_tokens
                        output_tokens += request_usage.output_tokens
                        cache_read_tokens += request_usage.cache_read_tokens
                        cache_write_tokens += request_usage.cache_write_tokens

        return UsageSummary(
            logical_requests=logical_requests,
            succeeded_requests=succeeded_requests,
            failed_requests=failed_requests,
            cancelled_requests=cancelled_requests,
            running_requests=running_requests,
            interrupted_requests=interrupted_requests,
            output_correction_retries=output_correction_retries,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_tokens=cache_read_tokens,
            cache_write_tokens=cache_write_tokens,
            model_duration_ns=model_duration_ns,
            unknown_usage_requests=unknown_usage_requests,
            unknown_duration_requests=unknown_duration_requests,
            transport_retries=None,
            cutoffs=tuple(fixed_cutoffs),
        )


    def _project_model_interaction(
        self,
        interaction: ModelInteractionRecord | StagedModelInteraction,
        execution_id: str,
        agent_run_seq: int,
        depth: int,
        resolved: object | None,
        *,
        include_content: bool,
    ) -> ModelInteractionItem:
        request: dict[str, JsonValue] = {}
        response: JsonValue | None = None
        if include_content and resolved == (None, None, None) and interaction.request_context is not None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if include_content and resolved != (None, None, None):
            if not isinstance(resolved, tuple) or len(resolved) != 3:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            request_messages, response_messages, envelope_raw = resolved
            if not isinstance(request_messages, tuple) or not isinstance(
                envelope_raw, bytes
            ):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            try:
                envelope = json.loads(envelope_raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
            if not isinstance(envelope, Mapping):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            request = {
                "messages": project_public_messages(request_messages),
                **dict(envelope),
            }
            instructions = [
                message.instructions
                for message in request_messages
                if isinstance(message, ModelRequest)
                and message.instructions is not None
            ]
            if instructions:
                request["instructions"] = instructions
            if response_messages is not None:
                projected = project_public_messages(response_messages)
                response = projected[0] if len(projected) == 1 else projected
        return ModelInteractionItem(
            execution_id=execution_id,
            agent_run_seq=agent_run_seq,
            depth=depth,
            model_request_seq=interaction.model_request_seq,
            purpose=interaction.purpose,
            step_index=interaction.step_index,
            output_retry_index=interaction.output_retry_index,
            model=interaction.model,
            request=request,
            response=response,
            status=interaction.status,
            error_code=interaction.error_code,
            duration_ns=interaction.duration_ns,
            usage=interaction.usage,
            content_included=include_content,
            started_at=interaction.started_at,
            finished_at=interaction.finished_at,
        )

    async def _tool_call_metadata(
        self,
        agent_run_id: str,
        *,
        execution_id: str,
        tenant_id: str,
        events: Sequence[StepEvent],
    ) -> dict[
        str,
        tuple[
            datetime | None,
            datetime | None,
            int | None,
            str,
            int | None,
            str | None,
        ],
    ]:
        values: dict[str, list[object | None]] = {}
        for event in events:
            if event.event_type not in {
                "TOOL_CALL_STARTED",
                "TOOL_CALL_SUCCEEDED",
                "TOOL_CALL_FAILED",
            }:
                continue
            call_id = event.tool_call_id
            if call_id is None or not call_id:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            value = values.setdefault(call_id, [None, None, None, "STARTED", None, None])
            model_request_seq = _event_model_request_seq(event)
            if model_request_seq is not None:
                if value[4] is not None and value[4] != model_request_seq:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                value[4] = model_request_seq
            timestamp = _event_timestamp(event)
            if event.event_type == "TOOL_CALL_STARTED":
                if value[0] is not None:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                value[0] = timestamp
                continue
            if value[1] is not None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            value[1] = timestamp
            raw_duration = event.metadata.get(DURATION_NS_METADATA_KEY)
            if raw_duration is not None:
                if not raw_duration.isdigit():
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                value[2] = int(raw_duration)
            value[3] = "SUCCEEDED" if event.event_type == "TOOL_CALL_SUCCEEDED" else "FAILED"

        if self._tool_operations is not None and values:
            operations = await self._tool_operations.get_by_call_ids(
                agent_run_id, tuple(values), tenant_id=tenant_id,
            )
            by_call: dict[str, ToolOperationRecord] = {}
            for operation in operations:
                if operation.agent_run_id != agent_run_id or operation.execution_id != execution_id:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                if operation.tool_call_id in by_call:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                by_call[operation.tool_call_id] = operation
            for call_id, value in values.items():
                operation = by_call.get(call_id)
                if operation is not None:
                    value[5] = operation.tool_operation_id
                    if operation.status is ToolOperationStatus.EFFECT_UNKNOWN:
                        value[3] = "EFFECT_UNKNOWN"
                    elif operation.status is ToolOperationStatus.CANCELLED:
                        value[3] = "CANCELLED"

        return {
            call_id: (
                cast("datetime | None", value[0]),
                cast("datetime | None", value[1]),
                cast("int | None", value[2]),
                cast(str, value[3]),
                cast("int | None", value[4]),
                cast("str | None", value[5]),
            )
            for call_id, value in values.items()
        }

    async def _history_page(
        self,
        sources: Sequence[_HistorySource],
        *,
        cursor_coordinate: "tuple[str, int, int, int] | None",
        high_waters: Mapping[tuple[str, int], tuple[int, int, tuple[str, ...]]],
        tenant_id: str,
        limit: int,
        model_request_seq: int | None,
        step_index: int | None,
        tool_call_id: str | None,
        message_seq: int | None,
        part_index: int | None,
    ) -> list[_HistoryOccurrence]:
        source_by_identity = {
            (source.record.execution_id, source.agent_run_seq): source
            for source in sources
        }
        cursor_source: _HistorySource | None = None
        if cursor_coordinate is not None:
            cursor_source = source_by_identity.get(cursor_coordinate[:2])
            if cursor_source is None:
                raise AIError(ErrorCode.CURSOR_INVALID)

        cursor_prefix = None if cursor_source is None else cursor_source.merge_prefix
        page: list[_HistoryOccurrence] = []
        # The source prefix precedes every message coordinate in history order.
        # Later sources need no body reads until this source is exhausted.
        for source in sources:
            if cursor_prefix is not None and source.merge_prefix < cursor_prefix:
                continue
            start_message_index = cursor_coordinate[2] if source is cursor_source and cursor_coordinate is not None else 0
            start_item_offset = cursor_coordinate[3] if source is cursor_source and cursor_coordinate is not None else 0
            identity = (source.record.execution_id, source.agent_run_seq)
            high_waters_for_source = high_waters.get(identity)
            if high_waters_for_source is None:
                raise AIError(ErrorCode.CURSOR_INVALID)
            message_high_water, event_high_water, tail_keys = high_waters_for_source
            if start_message_index > message_high_water:
                raise AIError(ErrorCode.CURSOR_INVALID)
            iterator = self._iter_history_source(
                source, tenant_id=tenant_id, start_message_index=start_message_index,
                end_message_index=message_high_water, event_high_water=event_high_water,
                start_item_offset=start_item_offset, from_cursor=source is cursor_source,
                tail_keys=tail_keys, tool_call_id=tool_call_id,
                model_request_seq=model_request_seq, step_index=step_index,
                message_seq=message_seq, part_index=part_index,
            )
            try:
                async for occurrence in iterator:
                    page.append(occurrence)
                    if len(page) > limit:
                        return page
            finally:
                await iterator.aclose()
        return page

    async def _iter_history_source(
        self,
        source: _HistorySource,
        *,
        tenant_id: str,
        start_message_index: int,
        end_message_index: int,
        event_high_water: int,
        start_item_offset: int,
        from_cursor: bool,
        tail_keys: tuple[str, ...],
        model_request_seq: int | None,
        step_index: int | None,
        tool_call_id: str | None,
        message_seq: int | None,
        part_index: int | None,
    ) -> AsyncGenerator[_HistoryOccurrence, None]:
        agent_run_id = make_agent_run_id(
            namespace=self._namespace,
            tenant_id=tenant_id,
            execution_id=source.record.execution_id,
            agent_run_seq=source.agent_run_seq,
        )
        if end_message_index == 0 and source.capture.message_count > 0:
            return
        range_end = end_message_index
        if message_seq is not None:
            selected_index = message_seq - 1
            if selected_index < start_message_index or selected_index >= end_message_index:
                return
            if selected_index != start_message_index:
                start_message_index = selected_index
                start_item_offset = 0
            range_end = selected_index + 1
        messages = self._message_range(
            agent_run_id,
            start=start_message_index,
            end=range_end,
            from_cursor=from_cursor,
            capture=source.capture,
        )
        if event_high_water > source.capture.event_count:
            raise AIError(ErrorCode.CURSOR_INVALID)
        inline_events = source.capture.inline_events
        if inline_events is not None:
            inline_events = inline_events[:event_high_water]
            responses, request_steps, replayed_parts = _request_associations(inline_events)
            tool_metadata = await self._tool_call_metadata(
                agent_run_id, execution_id=source.record.execution_id,
                tenant_id=tenant_id, events=inline_events,
            )
        first = True
        message_index = start_message_index
        saw_message = False
        async for message in messages:
            saw_message = True
            if inline_events is None:
                message_events = await self._store.read_history_associations(
                    agent_run_id=agent_run_id, message_seqs=(message_index + 1,),
                    tool_call_ids=(), event_high_water=event_high_water,
                )
                responses, request_steps, replayed_parts = _request_associations(message_events)
                tool_metadata = await self._tool_call_metadata(
                    agent_run_id, execution_id=source.record.execution_id,
                    tenant_id=tenant_id, events=message_events,
                )
            replayed_indices = replayed_parts.get(message_index + 1, ())
            if replayed_indices and (
                not isinstance(message, ModelResponse)
                or any(index >= len(message.parts) or not isinstance(message.parts[index], ToolCallPart)
                       for index in replayed_indices)
            ):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            projected = _history_parts(
                message,
                staged_keys=(
                    source.capture.pending_keys
                    if message_index == source.capture.transcript_message_count else ()
                ),
                selected_keys=tail_keys
                if message_index == end_message_index - 1
                else (),
            )
            item_offset = start_item_offset if first else 0
            if first and item_offset > len(projected):
                raise AIError(ErrorCode.CURSOR_INVALID)
            first = False
            for projected_offset in range(item_offset, len(projected)):
                value = projected[projected_offset]
                if value.part_index in replayed_indices:
                    continue
                if tool_call_id is not None and value.tool_call_id != tool_call_id:
                    continue
                if (
                    message_seq is not None
                    and message_index + 1 != message_seq
                ):
                    continue
                if part_index is not None and value.part_index != part_index:
                    continue
                if inline_events is None and value.tool_call_id is not None and value.tool_call_id not in tool_metadata:
                    call_events = await self._store.read_history_associations(
                        agent_run_id=agent_run_id, message_seqs=(),
                        tool_call_ids=(value.tool_call_id,), event_high_water=event_high_water,
                    )
                    call_responses, call_steps, _call_replays = _request_associations(call_events)
                    for response_sequence, response_request in call_responses.items():
                        if response_sequence in responses and responses[response_sequence] != response_request:
                            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                        responses[response_sequence] = response_request
                    for request_sequence, request_step in call_steps.items():
                        if request_sequence in request_steps and request_steps[request_sequence] != request_step:
                            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                        request_steps[request_sequence] = request_step
                    tool_metadata.update(await self._tool_call_metadata(
                        agent_run_id, execution_id=source.record.execution_id,
                        tenant_id=tenant_id, events=call_events,
                    ))
                metadata = (
                    None
                    if value.tool_call_id is None
                    else tool_metadata.get(value.tool_call_id)
                )
                origin_request = (
                    responses.get(message_index + 1)
                    if isinstance(message, ModelResponse)
                    else None
                )
                tool_request = None if metadata is None else metadata[4]
                if value.item_kind == "tool_call" and origin_request is None and tool_request is not None:
                    original_sequences = tuple(sequence for sequence, request in responses.items()
                                               if request == tool_request and sequence < message_index + 1)
                    if len(original_sequences) > 1:
                        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                    if original_sequences:
                        original_messages = [original async for original in self._store.iter_message_range(
                            agent_run_id=agent_run_id, start=original_sequences[0] - 1, end=original_sequences[0],
                        )]
                        if len(original_messages) != 1 or not isinstance(original_messages[0], ModelResponse):
                            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                        original_calls = [part for part in original_messages[0].parts
                                          if isinstance(part, ToolCallPart) and part.tool_call_id == value.tool_call_id]
                        if (len(original_calls) != 1 or original_calls[0].tool_name != value.tool_name
                                or original_calls[0].args_as_dict() != value.content):
                            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                        continue
                if origin_request is not None and tool_request is not None and origin_request != tool_request:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                if origin_request is None:
                    origin_request = tool_request
                origin_step = None if origin_request is None else request_steps.get(origin_request)
                if model_request_seq is not None and origin_request != model_request_seq:
                    continue
                if step_index is not None and origin_step != step_index:
                    continue
                status = None if metadata is None else metadata[3]
                if status == "STARTED" and source.record.status in {
                    ExecutionStatus.FAILED,
                    ExecutionStatus.CANCELLED,
                }:
                    status = source.record.status.value
                yield _HistoryOccurrence(
                    ExecutionHistoryItem(
                        execution_id=source.record.execution_id,
                        message_seq=message_index + 1,
                        item_kind=value.item_kind,
                        content=value.content,
                        tool_name=value.tool_name,
                        tool_call_id=value.tool_call_id,
                        content_included=True,
                        part_index=value.part_index,
                        agent_run_seq=source.agent_run_seq,
                        model_request_seq=origin_request,
                        step_index=origin_step,
                        tool_operation_id=None if metadata is None else metadata[5],
                        started_at=None if metadata is None else metadata[0],
                        finished_at=None if metadata is None else metadata[1],
                        duration_ns=None if metadata is None else metadata[2],
                        status=status,
                    ),
                    source.record.execution_id,
                    source.agent_run_seq,
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
            and source.agent_run_seq == source.record.agent_run_seq
        ):
            raise AIError(ErrorCode.EXECUTION_HISTORY_UNAVAILABLE)

    async def _message_range(
        self, agent_run_id: str, *, start: int, end: int,
        from_cursor: bool, capture: AgentRunHistoryCapture,
    ) -> AsyncIterator[object]:
        if start < 0 or end < start or end > capture.message_count or from_cursor and start == end:
            raise AIError(ErrorCode.CURSOR_INVALID)
        complete_end = min(end, capture.transcript_message_count)
        if start < complete_end:
            async for message in self._store.iter_message_range(
                agent_run_id=agent_run_id, start=start, end=complete_end,
            ):
                yield message
        if end > capture.transcript_message_count:
            pending = await self._store.read_pending_message(capture)
            if pending is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            yield pending

    async def transcript(
        self, execution_id: str, *, tenant_id: str, cursor: str | None, limit: int
    ) -> Page[TranscriptItem]:
        limit = validate_page_limit(limit)
        record = await self._executions.get(execution_id, tenant_id=tenant_id)
        if record is None:
            raise AIError(ErrorCode.STORAGE_NOT_FOUND)
        if record.agent_run_seq == 0:
            if record.status is ExecutionStatus.SUCCEEDED:
                raise AIError(ErrorCode.EXECUTION_HISTORY_UNAVAILABLE)
            return Page((), None)
        await self._history_tree(record, tenant_id)
        cursor_state = _decode_transcript_cursor(
            cursor,
            tenant_id=tenant_id,
            execution_id=execution_id,
            signer=self._cursor_signer,
        )
        if cursor_state is None:
            agent_run_seq = record.agent_run_seq
            agent_run_id = make_agent_run_id(
                namespace=self._namespace,
                tenant_id=tenant_id,
                execution_id=execution_id,
                agent_run_seq=agent_run_seq,
            )
            message_index = 0
            item_offset = 0
        else:
            (
                agent_run_id,
                agent_run_seq,
                high_water,
                message_index,
                item_offset,
                tail_keys,
            ) = cursor_state
            if (
                agent_run_seq > record.agent_run_seq
                or agent_run_id
                != make_agent_run_id(
                    namespace=self._namespace,
                    tenant_id=tenant_id,
                    execution_id=execution_id,
                    agent_run_seq=agent_run_seq,
                )
            ):
                raise AIError(ErrorCode.CURSOR_INVALID)

        captured = await self._store.capture_history((agent_run_id,), include_pending=True)
        capture = captured.get(agent_run_id)
        if capture is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        run = capture.run
        if run is None:
            if cursor is not None:
                raise AIError(ErrorCode.CURSOR_INVALID)
            if record.status is ExecutionStatus.SUCCEEDED:
                raise AIError(ErrorCode.EXECUTION_HISTORY_UNAVAILABLE)
            return Page((), None)
        if cursor_state is None:
            high_water = capture.message_count
            tail_keys = capture.pending_keys
        elif (capture.message_count) < high_water:
            raise AIError(ErrorCode.CURSOR_INVALID)
        _validate_agent_run(
            run,
            agent_run_id,
            make_agent_conversation_id(
                namespace=self._namespace,
                tenant_id=tenant_id,
                execution_id=execution_id,
            ),
            agent_run_seq,
        )
        messages = self._message_range(
            agent_run_id,
            start=message_index,
            end=high_water,
            from_cursor=cursor is not None,
            capture=capture,
        )
        agent_conversation_id = make_agent_conversation_id(
            namespace=self._namespace,
            tenant_id=tenant_id,
            execution_id=execution_id,
        )
        projection_index = message_index

        def project_message(message: object) -> tuple[str, ...]:
            nonlocal projection_index
            index = projection_index
            projection_index += 1
            if index == high_water - 1 and tail_keys:
                staged_keys = (
                    capture.pending_keys
                    if index == capture.transcript_message_count else ()
                )
                message = _select_message_parts(message, staged_keys, tail_keys)
            return _transcript_message_values(message, agent_conversation_id)

        projected, next_coordinate = await _read_projected_page(
            messages,
            start_message_index=message_index,
            start_item_offset=item_offset,
            project=project_message,
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
                agent_run_id,
                agent_run_seq,
                high_water,
                next_coordinate[0],
                next_coordinate[1],
                self._cursor_signer,
                tail_keys,
            )
        return Page(selected, next_cursor)

    async def _capture_sources(
        self, entries: Sequence[tuple[ExecutionRecord, int]], tenant_id: str,
        selected_run_sequence: int | None = None, *, include_pending: bool = False,
    ) -> list[_HistorySource]:
        candidates = tuple(
            (record, depth, sequence, make_agent_run_id(
                namespace=self._namespace, tenant_id=tenant_id,
                execution_id=record.execution_id, agent_run_seq=sequence,
            ))
            for record, depth in entries
            if selected_run_sequence is None or record is entries[0][0]
            for sequence in range(1, record.agent_run_seq + 1)
            if selected_run_sequence is None or sequence == selected_run_sequence
        )
        captured = await self._store.capture_history(
            tuple(value[3] for value in candidates), include_pending=include_pending,
        )
        sources: list[_HistorySource] = []
        for record, depth, sequence, run_id in candidates:
            capture = captured.get(run_id)
            if capture is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            run = capture.run
            if run is None:
                if record.status is ExecutionStatus.SUCCEEDED and sequence == record.agent_run_seq:
                    raise AIError(ErrorCode.EXECUTION_HISTORY_UNAVAILABLE)
                continue
            _validate_agent_run(run, run_id, make_agent_conversation_id(
                namespace=self._namespace, tenant_id=tenant_id, execution_id=record.execution_id,
            ), sequence)
            sources.append(_HistorySource(
                record, depth, sequence,
                (record.created_at.astimezone(timezone.utc), depth, record.execution_id, sequence),
                capture,
            ))
        for record, _depth in entries:
            if record.agent_run_seq < 0:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if record.agent_run_seq == 0 and record.status is ExecutionStatus.SUCCEEDED:
                raise AIError(ErrorCode.EXECUTION_HISTORY_UNAVAILABLE)
        sources.sort(key=lambda value: value.merge_prefix)
        return sources

    async def _history_tree(
        self, selected: ExecutionRecord, tenant_id: str
    ) -> list[tuple[ExecutionRecord, int]]:
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
                or child.lineage_kind.value != "SUBAGENT"
                or child.parent_execution_id != selected.execution_id
                or child.root_execution_id != selected.root_execution_id
            ):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            visited.add(child.execution_id)
            result.append((child, 1))
        return result


    async def _recursive_history_tree(
        self, selected: ExecutionRecord, tenant_id: str
    ) -> list[tuple[ExecutionRecord, int]]:
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

        result: list[tuple[ExecutionRecord, int]] = [(selected, 0)]
        visited = {selected.execution_id}
        pending: list[tuple[ExecutionRecord, int]] = [(selected, 0)]
        while pending:
            parent, depth = pending.pop(0)
            for child in await self._executions.list_children(
                parent.execution_id,
                tenant_id=tenant_id,
            ):
                if (
                    child.execution_id in visited
                    or child.lineage_kind.value != "SUBAGENT"
                    or child.parent_execution_id != parent.execution_id
                    or child.root_execution_id != selected.root_execution_id
                ):
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                visited.add(child.execution_id)
                result.append((child, depth + 1))
                pending.append((child, depth + 1))
        return result




class StepSessionHistoryReader:
    """Project one committed Conversation checkpoint into Session history."""

    def __init__(
        self,
        *,
        store: AgentRunStore,
        cursor_signer: CursorSigner,
        sessions: SessionRepository | None = None,
    ) -> None:
        self._store = store
        self._cursor_signer = cursor_signer
        self._sessions = sessions

    async def history(
        self,
        session_id: str,
        *,
        tenant_id: str,
        continuation_agent_run_id: "str | None",
        continuation_history_id: "str | None" = None,
        cursor: "str | None",
        limit: int,
    ) -> "Page[SessionHistoryItem]":
        limit = validate_page_limit(limit)
        if continuation_agent_run_id is None:
            if cursor is not None:
                raise AIError(ErrorCode.CURSOR_INVALID)
            return Page((), None)
        continuation_message_count: int | None = None
        if self._sessions is not None:
            session = await self._sessions.get(session_id, tenant_id=tenant_id)
            if session is None or session.continuation is None:
                raise AIError(ErrorCode.SESSION_HISTORY_UNAVAILABLE)
            current = session.continuation
            current_history_id = current.history_id or session.history_id
            if (
                current.agent_run_id != continuation_agent_run_id
                or continuation_history_id is not None
                and current_history_id != continuation_history_id
            ):
                raise AIError(ErrorCode.CURSOR_INVALID)
            continuation_message_count = current.message_count
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
            else continuation_agent_run_id
        )
        if cursor_values is not None and cursor_values[0] != requested_history_id:
            raise AIError(ErrorCode.CURSOR_INVALID)
        cursor_high_water = None if cursor_values is None else cursor_values[1]
        message_index = 0 if cursor_values is None else cursor_values[2]
        item_offset = 0 if cursor_values is None else cursor_values[3]
        if history_store is not None:
            history_id = continuation_history_id
            if history_id is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            physical_total = await history_store.session_message_count(
                history_id,
                tenant_id=tenant_id,
            )
            if continuation_message_count is None:
                total_messages = physical_total
            else:
                run = await self._store.get_agent_run(
                    agent_run_id=continuation_agent_run_id
                )
                checkpoint = await self._store.latest_checkpoint(
                    agent_run_id=continuation_agent_run_id,
                    include_interrupted=True,
                )
                if (
                    run is None
                    or checkpoint is None
                    or checkpoint.agent_run_id != continuation_agent_run_id
                    or checkpoint.state != "complete"
                ):
                    raise AIError(ErrorCode.SESSION_HISTORY_UNAVAILABLE)
                if (
                    isinstance(continuation_message_count, bool)
                    or not isinstance(continuation_message_count, int)
                    or continuation_message_count < 0
                ):
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                if continuation_message_count > physical_total:
                    raise AIError(ErrorCode.SESSION_HISTORY_UNAVAILABLE)
                total_messages = continuation_message_count
            if cursor_high_water is None:
                if cursor_values is not None and continuation_message_count is not None:
                    raise AIError(ErrorCode.CURSOR_INVALID)
            elif cursor_high_water != total_messages:
                raise AIError(ErrorCode.CURSOR_INVALID)
            if message_index > total_messages:
                raise AIError(ErrorCode.CURSOR_INVALID)
            messages = history_store.iter_session_message_range(
                history_id,
                tenant_id=tenant_id,
                start=message_index,
                end=total_messages,
            )
        else:
            history_id = continuation_agent_run_id
            run = await self._store.get_agent_run(agent_run_id=continuation_agent_run_id)
            if run is None:
                raise AIError(ErrorCode.SESSION_HISTORY_UNAVAILABLE)
            if run.agent_run_id != continuation_agent_run_id:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            checkpoint = await self._store.latest_checkpoint(
                agent_run_id=continuation_agent_run_id,
                include_interrupted=True,
            )
            if checkpoint is None:
                raise AIError(ErrorCode.SESSION_HISTORY_UNAVAILABLE)
            if (
                checkpoint.agent_run_id != continuation_agent_run_id
                or checkpoint.agent_conversation_id != run.agent_conversation_id
            ):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if checkpoint.state != "complete":
                raise AIError(ErrorCode.SESSION_HISTORY_UNAVAILABLE)
            total_messages = len(checkpoint.messages)
            if cursor_high_water is not None and cursor_high_water != total_messages:
                raise AIError(ErrorCode.CURSOR_INVALID)
            if message_index > total_messages:
                raise AIError(ErrorCode.CURSOR_INVALID)
            messages = _iter_sequence(checkpoint.messages, start=message_index)
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
                total_messages,
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
    agent_run_seq: int,
    depth: int,
    ordinal: int,
    event: StepEvent,
) -> "ExecutionTraceItem | None":
    mapping = {
        "MODEL_REQUEST_STARTED": ("MODEL_REQUEST", "STARTED"),
        "MODEL_REQUEST_SUCCEEDED": ("MODEL_RESPONSE", "SUCCEEDED"),
        "MODEL_REQUEST_FAILED": ("MODEL_RESPONSE", "FAILED"),
        "MODEL_REQUEST_CANCELLED": ("MODEL_RESPONSE", "CANCELLED"),
        "TOOL_CALL_STARTED": ("TOOL_CALL", "STARTED"),
        "TOOL_CALL_SUCCEEDED": ("TOOL_RESULT", "SUCCEEDED"),
        "TOOL_CALL_FAILED": ("TOOL_ERROR", "FAILED"),
    }
    value = mapping.get(event.event_type)
    if value is None:
        return None
    kind, status = value
    payload = {
        "kind": kind,
        "status": status,
        "step_index": event.step_index,
        "agent_run_seq": agent_run_seq,
        "scope": "root" if depth == 0 else "subagent",
        "depth": depth,
        "occurred_at": _event_timestamp(event).isoformat(),
    }
    observation_id = event.metadata.get(OBSERVATION_ID_METADATA_KEY)
    if observation_id is not None:
        payload["observation_id"] = observation_id
    duration_ns = event.metadata.get(DURATION_NS_METADATA_KEY)
    if duration_ns is not None:
        if not duration_ns.isdigit():
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        payload["duration_ns"] = int(duration_ns)
    message_seq = event.metadata.get(MESSAGE_SEQ_METADATA_KEY)
    if message_seq is not None:
        if event.event_type != "MODEL_REQUEST_SUCCEEDED" or not message_seq.isdigit() or int(message_seq) < 1:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        payload["message_seq"] = int(message_seq)
    model_request_seq = _event_model_request_seq(event)
    request_purpose = event.metadata.get(REQUEST_PURPOSE_METADATA_KEY)
    if request_purpose is not None:
        payload["purpose"] = request_purpose
    if model_request_seq is not None:
        payload["model_request_seq"] = model_request_seq
    retry_value = event.metadata.get(OUTPUT_RETRY_INDEX_METADATA_KEY)
    if retry_value is not None:
        if not retry_value.isdigit() or int(retry_value) < 1:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        payload["output_retry_index"] = int(retry_value)
    if kind == "MODEL_RESPONSE":
        payload["token_usage"] = (
            _model_token_usage(event) if status == "SUCCEEDED" else None
        )
    if event.agent_id is not None:
        payload["agent_id"] = event.agent_id
    if event.tool_call_id is not None:
        payload["tool_call_id"] = event.tool_call_id
    if event.tool_name is not None:
        payload["tool_name"] = event.tool_name
    if depth > 0:
        payload["child_execution_id"] = record.execution_id
    return ExecutionTraceItem(record.execution_id, ordinal + 1, payload)


def _request_associations(
    events: Sequence[StepEvent],
) -> tuple[dict[int, int], dict[int, int], dict[int, tuple[int, ...]]]:
    responses: dict[int, int] = {}
    steps: dict[int, int] = {}
    replayed_parts: dict[int, tuple[int, ...]] = {}
    for event in events:
        if not event.event_type.startswith("MODEL_REQUEST_"):
            continue
        request = _event_model_request_seq(event)
        raw_replays = event.metadata.get(REPLAYED_TOOL_CALL_INDICES_METADATA_KEY)
        if raw_replays is not None and (
            event.event_type != "MODEL_REQUEST_SUCCEEDED" or request is None
            or MESSAGE_SEQ_METADATA_KEY not in event.metadata
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if request is None:
            continue
        if request in steps and steps[request] != event.step_index:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        steps[request] = event.step_index
        raw = event.metadata.get(MESSAGE_SEQ_METADATA_KEY)
        if raw is None:
            continue
        if event.event_type != "MODEL_REQUEST_SUCCEEDED" or not raw.isdigit() or int(raw) < 1:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        message = int(raw)
        if message in responses and responses[message] != request:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        responses[message] = request
        if raw_replays is not None:
            try:
                indices = json.loads(raw_replays)
            except (TypeError, ValueError) as error:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
            if (not isinstance(indices, list)
                    or any(isinstance(index, bool) or not isinstance(index, int) or index < 0 for index in indices)
                    or indices != sorted(set(indices))):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            frozen = tuple(indices)
            if message in replayed_parts and replayed_parts[message] != frozen:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            replayed_parts[message] = frozen
    return responses, steps, replayed_parts


def _event_model_request_seq(event: StepEvent) -> "int | None":
    raw = event.metadata.get(MODEL_REQUEST_SEQ_METADATA_KEY)
    if raw is None:
        return None
    if not raw.isdigit() or int(raw) < 1:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return int(raw)


def _model_token_usage(event: StepEvent) -> "dict[str, JsonValue] | None":
    metadata = event.metadata
    values = {
        "input_tokens": MODEL_USAGE_INPUT_METADATA_KEY,
        "output_tokens": MODEL_USAGE_OUTPUT_METADATA_KEY,
        "cache_read_tokens": MODEL_USAGE_CACHE_READ_METADATA_KEY,
        "cache_write_tokens": MODEL_USAGE_CACHE_WRITE_METADATA_KEY,
    }
    usage = {
        name: value
        for name, key in values.items()
        if (value := _metadata_token(metadata, key)) is not None
    }
    return usage or None


def _metadata_token(
    metadata: Mapping[str, str],
    key: str,
) -> "int | None":
    raw = metadata.get(key)
    if raw is None:
        return None
    if not raw.isdigit():
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return int(raw)


def _validate_agent_run(
    run: AgentRunRecord, expected_id: str, agent_conversation_id: str, sequence: int
) -> None:
    if (
        run.agent_run_id != expected_id
        or run.agent_conversation_id != agent_conversation_id
        or run.metadata.get("agent_run_seq") != str(sequence)
    ):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)


def _event_timestamp(event: StepEvent) -> datetime:
    return event.timestamp.astimezone(timezone.utc)


def _execution_filter_digest(
    execution_id: str, projection_version: int, filters: Mapping[str, JsonValue] | None = None
) -> str:
    payload: dict[str, JsonValue] = {
        "execution_id": execution_id,
        "projection_version": projection_version,
    }
    if filters is not None:
        payload["filters"] = dict(filters)
    return canonical_sha256(payload)


def _attachment_fact_filter_digest(execution_id: str) -> str:
    return _execution_filter_digest(execution_id, _ATTACHMENT_FACT_PROJECTION_VERSION)


def _decode_attachment_fact_cursor(
    cursor: str | None,
    *,
    tenant_id: str,
    execution_id: str,
    signer: CursorSigner,
) -> tuple[tuple[int, int, str, str, int], tuple[tuple[int, int], ...]] | None:
    if cursor is None:
        return None
    payload = decode_runtime_cursor(
        cursor,
        signer,
        tenant_id=tenant_id,
        resource_kind="attachment_facts",
        filter_digest=_attachment_fact_filter_digest(execution_id),
    )
    try:
        value = json.loads(payload.position)
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise AIError(ErrorCode.CURSOR_INVALID) from error
    if (
        payload.revision != 0
        or not isinstance(value, dict)
        or set(value) != {"after", "cutoffs"}
    ):
        raise AIError(ErrorCode.CURSOR_INVALID)
    after = value.get("after")
    raw_cutoffs = value.get("cutoffs")
    if (
        not isinstance(after, list)
        or len(after) != 5
        or any(isinstance(after[index], bool) or not isinstance(after[index], int)
               or after[index] < 0 for index in (0, 1, 4))
        or not isinstance(after[2], str)
        or after[2] not in {"accepted", "included_in_request"}
        or not isinstance(after[3], str)
        or len(after[3]) != 64
        or any(character not in "0123456789abcdef" for character in after[3])
        or (after[0] == 0) != (after[1] == 0)
        or after[0] == 0 and after[2] != "accepted"
        or not isinstance(raw_cutoffs, list)
    ):
        raise AIError(ErrorCode.CURSOR_INVALID)
    cutoffs: list[tuple[int, int]] = []
    previous = 0
    for raw in raw_cutoffs:
        if (
            not isinstance(raw, list)
            or len(raw) != 2
            or isinstance(raw[0], bool)
            or not isinstance(raw[0], int)
            or raw[0] < 1
            or raw[0] <= previous
            or isinstance(raw[1], bool)
            or not isinstance(raw[1], int)
            or raw[1] < 0
        ):
            raise AIError(ErrorCode.CURSOR_INVALID)
        cutoffs.append((raw[0], raw[1]))
        previous = raw[0]
    if after[0] and after[1] > dict(cutoffs).get(after[0], 0):
        raise AIError(ErrorCode.CURSOR_INVALID)
    return (after[0], after[1], after[2], after[3], after[4]), tuple(cutoffs)


def _attachment_fact_cursor(
    tenant_id: str,
    execution_id: str,
    after: tuple[int, int, str, str, int],
    cutoffs: tuple[tuple[int, int], ...],
    signer: CursorSigner,
) -> str:
    return encode_runtime_cursor(
        signer,
        tenant_id=tenant_id,
        resource_kind="attachment_facts",
        filter_digest=_attachment_fact_filter_digest(execution_id),
        position=json.dumps(
            {
                "after": list(after),
                "cutoffs": [list(value) for value in cutoffs],
            },
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ),
    )


def _attachment_occurrences(
    execution_id: str,
    attachments: Sequence[Mapping[str, JsonValue]],
    *,
    agent_run_seq: int | None = None,
    model_request_seq: int | None = None,
    step_index: int | None = None,
) -> tuple[_AttachmentOccurrence, ...]:
    occurrences: list[_AttachmentOccurrence] = []
    counts: dict[tuple[str, str], int] = {}
    for raw in attachments:
        if not isinstance(raw, Mapping):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        fact = _project_attachment_fact(
            execution_id, raw, agent_run_seq=agent_run_seq,
            model_request_seq=model_request_seq, step_index=step_index,
        )
        identity = (fact.fact, fact.attachment_id)
        ordinal = counts.get(identity, 0)
        counts[identity] = ordinal + 1
        occurrences.append(_AttachmentOccurrence(
            (agent_run_seq or 0, model_request_seq or 0, *identity, ordinal), fact,
        ))
    return tuple(sorted(occurrences, key=lambda value: value.key))


def _project_attachment_fact(
    execution_id: str,
    value: Mapping[str, JsonValue],
    *,
    agent_run_seq: int | None = None,
    model_request_seq: int | None = None,
    step_index: int | None = None,
) -> AttachmentFact:
    try:
        fact = cast(str, value.get("fact"))
        return AttachmentFact(
            execution_id=execution_id,
            attachment_id=cast(str, value.get("attachment_id")),
            fact=fact,
            source=cast(str, value.get("source")),
            media_type=cast("str | None", value.get("media_type")),
            size=cast("int | None", value.get("size")),
            digest=cast("str | None", value.get("digest")),
            position=cast(int, value.get("position")),
            processing_status="unknown",
            agent_run_seq=agent_run_seq,
            model_request_seq=(
                model_request_seq if fact == "included_in_request" else None
            ),
            step_index=step_index if fact == "included_in_request" else None,
            call_id=cast("str | None", value.get("call_id")),
            input_identifier=cast(
                "str | None",
                value.get("input_identifier"),
            ),
        )
    except (TypeError, ValueError) as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error


def _model_interaction_filter_digest(execution_id: str) -> str:
    return _execution_filter_digest(execution_id, _MODEL_INTERACTION_PROJECTION_VERSION)


def _interaction_occurrence_groups(
    values: Sequence[_InteractionOccurrence],
) -> tuple[tuple[_InteractionOccurrence, ...], ...]:
    groups: dict[str, list[_InteractionOccurrence]] = {}
    for value in values:
        groups.setdefault(value.interaction.agent_run_id, []).append(value)
    return tuple(tuple(group) for group in groups.values())


def _normalize_usage_cutoffs(
    value: "tuple[UsageReadCutoff, ...] | None",
) -> "tuple[UsageReadCutoff, ...] | None":
    if value is None:
        return None
    try:
        normalized = tuple(
            sorted(
                value,
                key=lambda item: (item.execution_id, item.agent_run_seq),
            )
        )
    except (AttributeError, TypeError) as error:
        raise AIError(ErrorCode.CURSOR_INVALID) from error
    if (
        any(not isinstance(item, UsageReadCutoff) for item in normalized)
        or len({(item.execution_id, item.agent_run_seq) for item in normalized})
        != len(normalized)
    ):
        raise AIError(ErrorCode.CURSOR_INVALID)
    return normalized


def _decode_model_interaction_cursor(
    cursor: str | None,
    *,
    tenant_id: str,
    execution_id: str,
    signer: CursorSigner,
) -> "tuple[tuple[str, int, int], tuple[UsageReadCutoff, ...]] | None":
    if cursor is None:
        return None
    payload = decode_runtime_cursor(
        cursor,
        signer,
        tenant_id=tenant_id,
        resource_kind="model_interactions",
        filter_digest=_model_interaction_filter_digest(execution_id),
    )
    try:
        value = json.loads(payload.position)
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise AIError(ErrorCode.CURSOR_INVALID) from error
    if (
        payload.revision != 0
        or not isinstance(value, Mapping)
        or set(value) != {"position", "cutoffs"}
    ):
        raise AIError(ErrorCode.CURSOR_INVALID)
    coordinate = value["position"]
    raw_cutoffs = value["cutoffs"]
    if (
        not isinstance(coordinate, list)
        or len(coordinate) != 3
        or not isinstance(coordinate[0], str)
        or not coordinate[0]
        or isinstance(coordinate[1], bool)
        or not isinstance(coordinate[1], int)
        or coordinate[1] < 1
        or isinstance(coordinate[2], bool)
        or not isinstance(coordinate[2], int)
        or coordinate[2] < 1
        or not isinstance(raw_cutoffs, list)
    ):
        raise AIError(ErrorCode.CURSOR_INVALID)
    cutoffs: list[UsageReadCutoff] = []
    for raw in raw_cutoffs:
        if (
            not isinstance(raw, list)
            or len(raw) != 3
            or not isinstance(raw[0], str)
            or not raw[0]
            or isinstance(raw[1], bool)
            or not isinstance(raw[1], int)
            or raw[1] < 1
            or isinstance(raw[2], bool)
            or not isinstance(raw[2], int)
            or raw[2] < 0
        ):
            raise AIError(ErrorCode.CURSOR_INVALID)
        cutoffs.append(UsageReadCutoff(raw[0], raw[1], raw[2]))
    normalized = _normalize_usage_cutoffs(tuple(cutoffs))
    if normalized is None:
        raise AIError(ErrorCode.CURSOR_INVALID)
    return (coordinate[0], coordinate[1], coordinate[2]), normalized


def _model_interaction_cursor(
    tenant_id: str,
    execution_id: str,
    occurrence: _InteractionOccurrence,
    cutoffs: tuple[UsageReadCutoff, ...],
    signer: CursorSigner,
) -> str:
    normalized = _normalize_usage_cutoffs(cutoffs)
    if normalized is None:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return encode_runtime_cursor(
        signer,
        tenant_id=tenant_id,
        resource_kind="model_interactions",
        filter_digest=_model_interaction_filter_digest(execution_id),
        position=json.dumps(
            {
                "position": [
                    occurrence.source_execution_id,
                    occurrence.agent_run_seq,
                    occurrence.interaction.model_request_seq,
                ],
                "cutoffs": [
                    [
                        value.execution_id,
                        value.agent_run_seq,
                        value.model_request_seq,
                    ]
                    for value in normalized
                ],
            },
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ),
    )


def _decode_position(position: str, size: int) -> list[object]:
    try:
        value = json.loads(position)
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise AIError(ErrorCode.CURSOR_INVALID) from error
    if not isinstance(value, list) or len(value) != size:
        raise AIError(ErrorCode.CURSOR_INVALID)
    return value


def _decode_history_cursor(
    cursor: str | None,
    *,
    tenant_id: str,
    execution_id: str,
    signer: CursorSigner,
    filters: Mapping[str, JsonValue] | None = None,
) -> "tuple[tuple[str, int, int, int], tuple[tuple[str, int, int, int, tuple[str, ...]], ...]] | None":
    if cursor is None:
        return None
    payload = decode_runtime_cursor(
        cursor,
        signer,
        tenant_id=tenant_id,
        resource_kind="execution_history",
        filter_digest=_execution_filter_digest(
            execution_id, _EXECUTION_HISTORY_PROJECTION_VERSION, filters
        ),
    )
    try:
        value = json.loads(payload.position)
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise AIError(ErrorCode.CURSOR_INVALID) from error
    if (
        payload.revision != 0
        or not isinstance(value, Mapping)
        or set(value) != {"position", "cutoffs"}
    ):
        raise AIError(ErrorCode.CURSOR_INVALID)
    coordinate = value["position"]
    raw_cutoffs = value["cutoffs"]
    if (
        not isinstance(coordinate, list)
        or len(coordinate) != 4
        or not isinstance(coordinate[0], str)
        or not coordinate[0]
        or isinstance(coordinate[1], bool)
        or not isinstance(coordinate[1], int)
        or coordinate[1] < 1
        or isinstance(coordinate[2], bool)
        or not isinstance(coordinate[2], int)
        or coordinate[2] < 0
        or isinstance(coordinate[3], bool)
        or not isinstance(coordinate[3], int)
        or coordinate[3] < 0
        or not isinstance(raw_cutoffs, list)
    ):
        raise AIError(ErrorCode.CURSOR_INVALID)
    cutoffs: list[tuple[str, int, int, int, tuple[str, ...]]] = []
    for raw in raw_cutoffs:
        if (
            not isinstance(raw, list)
            or len(raw) != 5
            or not isinstance(raw[0], str)
            or not raw[0]
            or isinstance(raw[1], bool)
            or not isinstance(raw[1], int)
            or raw[1] < 1
            or isinstance(raw[2], bool)
            or not isinstance(raw[2], int)
            or raw[2] < 0
            or isinstance(raw[3], bool)
            or not isinstance(raw[3], int)
            or raw[3] < 0
            or not isinstance(raw[4], list)
            or any(not isinstance(key, str) or not key for key in raw[4])
            or len(set(raw[4])) != len(raw[4])
        ):
            raise AIError(ErrorCode.CURSOR_INVALID)
        cutoffs.append((raw[0], raw[1], raw[2], raw[3], tuple(raw[4])))
    normalized = tuple(sorted(cutoffs, key=lambda item: (item[0], item[1])))
    if len({(item[0], item[1]) for item in normalized}) != len(normalized):
        raise AIError(ErrorCode.CURSOR_INVALID)
    return (
        (coordinate[0], coordinate[1], coordinate[2], coordinate[3]),
        normalized,
    )


def _history_cursor(
    tenant_id: str,
    execution_id: str,
    occurrence: _HistoryOccurrence,
    cutoffs: tuple[tuple[str, int, int, int, tuple[str, ...]], ...],
    signer: CursorSigner,
    filters: Mapping[str, JsonValue] | None = None,
) -> str:
    normalized = tuple(sorted(cutoffs, key=lambda item: (item[0], item[1])))
    if len({(item[0], item[1]) for item in normalized}) != len(normalized):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return encode_runtime_cursor(
        signer,
        tenant_id=tenant_id,
        resource_kind="execution_history",
        filter_digest=_execution_filter_digest(
            execution_id, _EXECUTION_HISTORY_PROJECTION_VERSION, filters
        ),
        position=json.dumps(
            {
                "position": [
                    occurrence.source_execution_id,
                    occurrence.agent_run_seq,
                    occurrence.message_index,
                    occurrence.item_offset,
                ],
                "cutoffs": [list(item) for item in normalized],
            },
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ),
    )


def _decode_trace_cursor(
    cursor: str | None,
    *,
    tenant_id: str,
    execution_id: str,
    signer: CursorSigner,
    filters: Mapping[str, JsonValue] | None = None,
) -> "tuple[tuple[str, int, int], tuple[tuple[str, int, int], ...]] | None":
    if cursor is None:
        return None
    payload = decode_runtime_cursor(
        cursor,
        signer,
        tenant_id=tenant_id,
        resource_kind="execution_trace",
        filter_digest=_execution_filter_digest(
            execution_id, _EXECUTION_TRACE_PROJECTION_VERSION, filters
        ),
    )
    try:
        value = json.loads(payload.position)
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise AIError(ErrorCode.CURSOR_INVALID) from error
    if (
        payload.revision != 0
        or not isinstance(value, Mapping)
        or set(value) != {"position", "cutoffs"}
    ):
        raise AIError(ErrorCode.CURSOR_INVALID)
    coordinate = value["position"]
    raw_cutoffs = value["cutoffs"]
    if (
        not isinstance(coordinate, list)
        or len(coordinate) != 3
        or not isinstance(coordinate[0], str)
        or not coordinate[0]
        or isinstance(coordinate[1], bool)
        or not isinstance(coordinate[1], int)
        or coordinate[1] < 1
        or isinstance(coordinate[2], bool)
        or not isinstance(coordinate[2], int)
        or coordinate[2] < 1
        or not isinstance(raw_cutoffs, list)
    ):
        raise AIError(ErrorCode.CURSOR_INVALID)
    cutoffs: list[tuple[str, int, int]] = []
    for raw in raw_cutoffs:
        if (
            not isinstance(raw, list)
            or len(raw) != 3
            or not isinstance(raw[0], str)
            or not raw[0]
            or isinstance(raw[1], bool)
            or not isinstance(raw[1], int)
            or raw[1] < 1
            or isinstance(raw[2], bool)
            or not isinstance(raw[2], int)
            or raw[2] < 0
        ):
            raise AIError(ErrorCode.CURSOR_INVALID)
        cutoffs.append((raw[0], raw[1], raw[2]))
    normalized = tuple(sorted(cutoffs, key=lambda item: (item[0], item[1])))
    if len({(item[0], item[1]) for item in normalized}) != len(normalized):
        raise AIError(ErrorCode.CURSOR_INVALID)
    return (coordinate[0], coordinate[1], coordinate[2]), normalized


def _trace_cursor(
    tenant_id: str,
    execution_id: str,
    occurrence: _TraceOccurrence,
    cutoffs: tuple[tuple[str, int, int], ...],
    signer: CursorSigner,
    filters: Mapping[str, JsonValue] | None = None,
) -> str:
    normalized = tuple(sorted(cutoffs, key=lambda item: (item[0], item[1])))
    if len({(item[0], item[1]) for item in normalized}) != len(normalized):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return encode_runtime_cursor(
        signer,
        tenant_id=tenant_id,
        resource_kind="execution_trace",
        filter_digest=_execution_filter_digest(
            execution_id, _EXECUTION_TRACE_PROJECTION_VERSION, filters
        ),
        position=json.dumps(
            {
                "position": [
                    occurrence.source_execution_id,
                    occurrence.agent_run_seq,
                    occurrence.step_event_seq,
                ],
                "cutoffs": [list(item) for item in normalized],
            },
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ),
    )


def _decode_transcript_cursor(
    cursor: str | None,
    *,
    tenant_id: str,
    execution_id: str,
    signer: CursorSigner,
) -> "tuple[str, int, int, int, int, tuple[str, ...]] | None":
    if cursor is None:
        return None
    payload = decode_runtime_cursor(
        cursor,
        signer,
        tenant_id=tenant_id,
        resource_kind="execution_transcript",
        filter_digest=_execution_filter_digest(
            execution_id, _EXECUTION_TRANSCRIPT_PROJECTION_VERSION
        ),
    )
    coordinate = _decode_position(payload.position, 6)
    if (
        payload.revision != 0
        or not isinstance(coordinate[0], str)
        or not coordinate[0]
        or isinstance(coordinate[1], bool)
        or not isinstance(coordinate[1], int)
        or coordinate[1] < 1
        or isinstance(coordinate[2], bool)
        or not isinstance(coordinate[2], int)
        or coordinate[2] < 0
        or isinstance(coordinate[3], bool)
        or not isinstance(coordinate[3], int)
        or coordinate[3] < 0
        or coordinate[3] > coordinate[2]
        or isinstance(coordinate[4], bool)
        or not isinstance(coordinate[4], int)
        or coordinate[4] < 0
        or not isinstance(coordinate[5], list)
        or any(not isinstance(key, str) or not key for key in coordinate[5])
        or len(set(coordinate[5])) != len(coordinate[5])
    ):
        raise AIError(ErrorCode.CURSOR_INVALID)
    return (
        coordinate[0],
        coordinate[1],
        coordinate[2],
        coordinate[3],
        coordinate[4],
        tuple(coordinate[5]),
    )


def _transcript_cursor(
    tenant_id: str,
    execution_id: str,
    agent_run_id: str,
    agent_run_seq: int,
    high_water: int,
    message_index: int,
    item_offset: int,
    signer: CursorSigner,
    tail_keys: tuple[str, ...],
) -> str:
    if (
        not agent_run_id
        or agent_run_seq < 1
        or high_water < 0
        or message_index < 0
        or message_index > high_water
        or item_offset < 0
    ):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return encode_runtime_cursor(
        signer,
        tenant_id=tenant_id,
        resource_kind="execution_transcript",
        filter_digest=_execution_filter_digest(
            execution_id, _EXECUTION_TRANSCRIPT_PROJECTION_VERSION
        ),
        position=json.dumps(
            [
                agent_run_id,
                agent_run_seq,
                high_water,
                message_index,
                item_offset,
                list(tail_keys),
            ],
            ensure_ascii=False,
            separators=(",", ":"),
        ),
    )


def _session_history_filter_digest(session_id: str) -> str:
    return canonical_sha256(
        {
            "session_id": session_id,
            "projection_version": SESSION_HISTORY_VIEW_V1,
        }
    )


def _decode_session_history_cursor(
    cursor: str,
    tenant_id: str,
    session_id: str,
    signer: CursorSigner,
) -> tuple[str, int | None, int, int]:
    payload = decode_runtime_cursor(
        cursor,
        signer,
        tenant_id=tenant_id,
        resource_kind="session_history",
        filter_digest=_session_history_filter_digest(session_id),
    )
    try:
        coordinate = json.loads(payload.position)
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise AIError(ErrorCode.CURSOR_INVALID) from error
    if not isinstance(coordinate, list) or len(coordinate) not in {3, 4}:
        raise AIError(ErrorCode.CURSOR_INVALID)
    if len(coordinate) == 3:
        history_id, message_index, item_offset = coordinate
        high_water = None
    else:
        history_id, high_water, message_index, item_offset = coordinate
    if (
        payload.revision != 0
        or not isinstance(history_id, str)
        or not history_id
        or high_water is not None
        and (
            isinstance(high_water, bool)
            or not isinstance(high_water, int)
            or high_water < 0
        )
        or isinstance(message_index, bool)
        or not isinstance(message_index, int)
        or message_index < 0
        or isinstance(item_offset, bool)
        or not isinstance(item_offset, int)
        or item_offset < 0
    ):
        raise AIError(ErrorCode.CURSOR_INVALID)
    return history_id, high_water, message_index, item_offset


def _session_history_cursor(
    tenant_id: str,
    session_id: str,
    history_id: str,
    high_water: int,
    next_message_index: int,
    intra_message_item_offset: int,
    signer: CursorSigner,
) -> str:
    return encode_runtime_cursor(
        signer,
        tenant_id=tenant_id,
        resource_kind="session_history",
        filter_digest=_session_history_filter_digest(session_id),
        position=json.dumps(
            [
                history_id,
                high_water,
                next_message_index,
                intra_message_item_offset,
            ],
            ensure_ascii=False,
            separators=(",", ":"),
        ),
    )


def _select_message_parts(
    message: object, staged_keys: tuple[str, ...], selected_keys: tuple[str, ...]
) -> object:
    if not isinstance(message, (ModelRequest, ModelResponse)):
        raise AIError(ErrorCode.CURSOR_INVALID)
    keys = staged_keys or tuple(
        f"tool_result:{part.tool_call_id}"
        if isinstance(part, ToolReturnPart)
        else f"retry:{part.tool_call_id}"
        if isinstance(part, RetryPromptPart)
        else f"part:{index}"
        for index, part in enumerate(message.parts)
    )
    by_key = dict(zip(keys, message.parts, strict=True))
    try:
        return replace(message, parts=[by_key[key] for key in selected_keys])
    except KeyError as error:
        raise AIError(ErrorCode.CURSOR_INVALID) from error


def _history_parts(
    message: object,
    *,
    staged_keys: tuple[str, ...] = (),
    selected_keys: tuple[str, ...] = (),
) -> tuple[_ProjectedHistoryItem, ...]:
    values = _project_message(message)
    has_instructions = isinstance(message, ModelRequest) and message.instructions is not None
    raw_parts = values[1:] if has_instructions else values
    keys = staged_keys or tuple(
        f"{item.item_kind}:{item.tool_call_id}"
        if item.tool_call_id is not None and item.item_kind in {"tool_result", "retry"}
        else f"part:{index}"
        for index, item in enumerate(raw_parts)
    )
    if has_instructions:
        keys = ("instructions", *keys)
    values = tuple(
        replace(item, part_index=int(key[5:]) if key.startswith("part:") else None)
        for key, item in zip(keys, values, strict=True)
    )
    if not selected_keys:
        return values
    by_key = dict(zip(keys, values, strict=True))
    try:
        return tuple(by_key[key] for key in selected_keys)
    except KeyError as error:
        raise AIError(ErrorCode.CURSOR_INVALID) from error


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
    message: object, agent_conversation_id: str
) -> tuple[str, ...]:
    del agent_conversation_id
    if not isinstance(message, (ModelRequest, ModelResponse)):
        return ()
    return project_execution_transcript_message(message)


__all__ = ["StepExecutionHistoryReader", "StepSessionHistoryReader"]
