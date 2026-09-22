#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Project Runtime step facts into Runtime trace and transcript views."""

import heapq
import json
import re
from collections.abc import AsyncGenerator, AsyncIterator, Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Protocol, cast, runtime_checkable

from linktools.core import environ
from pydantic_ai.messages import ModelRequest, ModelResponse

from ..core import (
    CursorSigner,
    ExecutionStatus,
    JsonValue,
    Page,
    ToolOperationStatus,
    canonical_sha256,
    step_conversation_id,
    step_run_id,
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
    MODEL_USAGE_METADATA_KEYS,
    MODEL_USAGE_OUTPUT_METADATA_KEY,
    OBSERVATION_ID_METADATA_KEY,
    OUTPUT_RETRY_INDEX_METADATA_KEY,
    REQUEST_PURPOSE_METADATA_KEY,
    REQUEST_SEQUENCE_METADATA_KEY,
)
from ._model_interaction import project_public_messages
from .service_api import (
    AttachmentFact,
    ExecutionHistoryItem,
    ExecutionTraceItem,
    ModelInteractionItem,
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
from .state._step_contracts import RunRecord, StepEvent, StepStore
from .state._views import (
    SESSION_HISTORY_VIEW_V1,
    project_execution_transcript_message,
    project_session_history_message,
)

_logger = environ.get_logger("ai.runtime.history")
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
class _InteractionOccurrence:
    key: tuple[object, ...]
    source_execution_id: str
    segment_sequence: int
    depth: int
    interaction: ModelInteractionRecord


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


class _ToolOperationHistoryReader(Protocol):
    async def list_by_execution(
        self,
        execution_id: str,
        *,
        tenant_id: str,
    ) -> tuple[ToolOperationRecord, ...]: ...


class StepExecutionHistoryReader:
    """Own the adapter projection between StepStore facts and Runtime views."""

    def __init__(
        self,
        *,
        namespace: str,
        executions: ExecutionRepository,
        store: StepStore,
        cursor_signer: CursorSigner,
        tool_operations: "_ToolOperationHistoryReader | None" = None,
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

    async def trace(
        self, execution_id: str, *, tenant_id: str, cursor: "str | None", limit: int
    ) -> "Page[ExecutionTraceItem]":
        record = await self._executions.get(execution_id, tenant_id=tenant_id)
        if record is None:
            raise AIError(ErrorCode.STORAGE_NOT_FOUND)
        limit = validate_page_limit(limit)
        entries = await self._history_tree(record, tenant_id)
        current: dict[tuple[str, int], tuple[ExecutionRecord, int, list[StepEvent]]] = {}
        for item, depth in entries:
            for segment_sequence, events in await self._segment_events(item, tenant_id):
                identity = (item.execution_id, segment_sequence)
                if identity in current:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                current[identity] = (item, depth, events)

        cursor_state = _decode_trace_cursor(
            cursor,
            tenant_id=tenant_id,
            execution_id=execution_id,
            signer=self._cursor_signer,
        )
        if cursor_state is None:
            cursor_coordinate = None
            fixed_cutoffs = tuple(
                sorted(
                    (
                        source_execution_id,
                        segment_sequence,
                        len(events),
                    )
                    for (source_execution_id, segment_sequence), (
                        _item,
                        _depth,
                        events,
                    ) in current.items()
                )
            )
        else:
            cursor_coordinate, fixed_cutoffs = cursor_state

        occurrences: list[_TraceOccurrence] = []
        for source_execution_id, segment_sequence, event_count in fixed_cutoffs:
            value = current.get((source_execution_id, segment_sequence))
            if value is None:
                raise AIError(ErrorCode.CURSOR_INVALID)
            item, depth, events = value
            if event_count > len(events):
                raise AIError(ErrorCode.CURSOR_INVALID)
            for ordinal, event in enumerate(events[:event_count]):
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
        start_index = _trace_cursor_index(
            cursor_coordinate,
            occurrences=occurrences,
        )
        page = occurrences[start_index : start_index + limit + 1]
        selected = tuple(occurrence.item for occurrence in page[:limit])
        next_cursor = None
        if len(page) > limit:
            next_cursor = _trace_cursor(
                tenant_id,
                execution_id,
                page[limit],
                fixed_cutoffs,
                self._cursor_signer,
            )
        _logger.debug(
            "execution trace projected page: execution=%s source_index=%s items=%s",
            execution_id,
            start_index,
            len(selected),
        )
        return Page(selected, next_cursor)

    async def history(
        self, execution_id: str, *, tenant_id: str, cursor: str | None, limit: int
    ) -> "Page[ExecutionHistoryItem]":
        limit = validate_page_limit(limit)
        record = await self._executions.get(execution_id, tenant_id=tenant_id)
        if record is None:
            raise AIError(ErrorCode.STORAGE_NOT_FOUND)
        entries = await self._history_tree(record, tenant_id)
        current_sources: list[_HistorySource] = []
        for item, depth in entries:
            for segment_sequence in await self._segment_sequences(item, tenant_id):
                current_sources.append(
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
        current_sources.sort(key=lambda source: source.merge_prefix)
        source_by_identity = {
            (source.record.execution_id, source.segment_sequence): source
            for source in current_sources
        }
        cursor_state = _decode_history_cursor(
            cursor,
            tenant_id=tenant_id,
            execution_id=execution_id,
            signer=self._cursor_signer,
        )
        if cursor_state is None:
            cursor_coordinate = None
            captured: list[tuple[str, int, int, int]] = []
            for source in current_sources:
                run_id = step_run_id(
                    namespace=self._namespace,
                    tenant_id=tenant_id,
                    execution_id=source.record.execution_id,
                    segment_sequence=source.segment_sequence,
                )
                captured.append(
                    (
                        source.record.execution_id,
                        source.segment_sequence,
                        await self._transcript_high_water(run_id),
                        len(await self._store.list_events(run_id=run_id)),
                    )
                )
            fixed_cutoffs = tuple(captured)
        else:
            cursor_coordinate, fixed_cutoffs = cursor_state
        cutoff_by_identity = {
            (source_execution_id, segment_sequence): (
                message_count,
                event_count,
            )
            for (
                source_execution_id,
                segment_sequence,
                message_count,
                event_count,
            ) in fixed_cutoffs
        }
        if len(cutoff_by_identity) != len(fixed_cutoffs):
            raise AIError(ErrorCode.CURSOR_INVALID)
        if any(identity not in source_by_identity for identity in cutoff_by_identity):
            raise AIError(ErrorCode.CURSOR_INVALID)
        sources = tuple(
            source
            for source in current_sources
            if (source.record.execution_id, source.segment_sequence) in cutoff_by_identity
        )
        for source in sources:
            identity = (source.record.execution_id, source.segment_sequence)
            run_id = step_run_id(
                namespace=self._namespace,
                tenant_id=tenant_id,
                execution_id=source.record.execution_id,
                segment_sequence=source.segment_sequence,
            )
            message_high_water, event_high_water = cutoff_by_identity[identity]
            if (
                await self._transcript_high_water(run_id) < message_high_water
                or len(await self._store.list_events(run_id=run_id)) < event_high_water
            ):
                raise AIError(ErrorCode.CURSOR_INVALID)
        page = await self._history_page(
            sources,
            cursor_coordinate=cursor_coordinate,
            high_waters=cutoff_by_identity,
            tenant_id=tenant_id,
            limit=limit,
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
            )
        _logger.debug(
            "execution history projected page: execution=%s items=%s",
            execution_id,
            len(selected),
        )
        return Page(selected, next_cursor)

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
        entries = await self._history_tree(record, tenant_id)
        current_sources: list[_HistorySource] = []
        for item, depth in entries:
            for segment_sequence in await self._segment_sequences(item, tenant_id):
                current_sources.append(
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
        current_sources.sort(key=lambda source: source.merge_prefix)
        source_by_identity = {
            (source.record.execution_id, source.segment_sequence): source
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
            if explicit_cutoffs is not None and explicit_cutoffs != cursor_cutoffs:
                raise AIError(ErrorCode.CURSOR_INVALID)
            fixed_cutoffs = cursor_cutoffs
        else:
            cursor_coordinate = None
            if explicit_cutoffs is None:
                captured: list[UsageReadCutoff] = []
                for source in current_sources:
                    run_id = step_run_id(
                        namespace=self._namespace,
                        tenant_id=tenant_id,
                        execution_id=source.record.execution_id,
                        segment_sequence=source.segment_sequence,
                    )
                    captured.append(
                        UsageReadCutoff(
                            source.record.execution_id,
                            source.segment_sequence,
                            await self._store.model_interaction_count(run_id=run_id),
                        )
                    )
                fixed_cutoffs = tuple(captured)
            else:
                fixed_cutoffs = explicit_cutoffs

        cutoff_by_identity = {
            (value.execution_id, value.segment_sequence): value.request_sequence
            for value in fixed_cutoffs
        }
        if len(cutoff_by_identity) != len(fixed_cutoffs):
            raise AIError(ErrorCode.CURSOR_INVALID)
        if any(identity not in source_by_identity for identity in cutoff_by_identity):
            raise AIError(ErrorCode.CURSOR_INVALID)
        sources = tuple(
            source
            for source in current_sources
            if (source.record.execution_id, source.segment_sequence) in cutoff_by_identity
        )
        for source in sources:
            identity = (source.record.execution_id, source.segment_sequence)
            run_id = step_run_id(
                namespace=self._namespace,
                tenant_id=tenant_id,
                execution_id=source.record.execution_id,
                segment_sequence=source.segment_sequence,
            )
            if await self._store.model_interaction_count(run_id=run_id) < cutoff_by_identity[identity]:
                raise AIError(ErrorCode.CURSOR_INVALID)

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
            identity = (source.record.execution_id, source.segment_sequence)
            high_water = cutoff_by_identity[identity]
            after_request_sequence = (
                cursor_coordinate[2]
                if cursor_coordinate is not None and source is cursor_source
                else 0
            )
            available = high_water - after_request_sequence
            if available <= 0:
                continue
            remaining = limit + 1 - len(page)
            if remaining <= 0:
                break
            run_id = step_run_id(
                namespace=self._namespace,
                tenant_id=tenant_id,
                execution_id=source.record.execution_id,
                segment_sequence=source.segment_sequence,
            )
            fetch_limit = min(remaining, available)
            interactions = await self._store.list_model_interactions(
                run_id=run_id,
                after_request_sequence=after_request_sequence,
                limit=fetch_limit,
            )
            if len(interactions) != fetch_limit:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            expected_sequence = after_request_sequence
            for interaction in interactions:
                expected_sequence += 1
                if (
                    not isinstance(interaction, ModelInteractionRecord)
                    or interaction.run_id != run_id
                    or interaction.request_sequence != expected_sequence
                ):
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                page.append(
                    _InteractionOccurrence(
                        (*source.merge_prefix, interaction.request_sequence),
                        source.record.execution_id,
                        source.segment_sequence,
                        source.depth,
                        interaction,
                    )
                )

        selected_occurrences = page[:limit]
        selected_items: dict[tuple[str, int], ModelInteractionItem] = {}
        for occurrence_group in _interaction_occurrence_groups(selected_occurrences):
            times = await self._model_request_times(occurrence_group[0].interaction.run_id)
            resolved_values: tuple[object, ...]
            if include_content:
                resolved_values = await self._store.resolve_model_interactions(
                    tuple(value.interaction for value in occurrence_group)
                )
                if len(resolved_values) != len(occurrence_group):
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            else:
                resolved_values = (None,) * len(occurrence_group)
            for occurrence, resolved_context in zip(
                occurrence_group,
                resolved_values,
                strict=True,
            ):
                started_at, finished_at = times.get(
                    occurrence.interaction.request_sequence,
                    (None, None),
                )
                selected_items[
                    (occurrence.interaction.run_id, occurrence.interaction.request_sequence)
                ] = self._project_model_interaction(
                    occurrence.interaction,
                    occurrence.source_execution_id,
                    occurrence.segment_sequence,
                    occurrence.depth,
                    resolved_context,
                    include_content=include_content,
                    started_at=started_at,
                    finished_at=finished_at,
                )
        selected = tuple(
            selected_items[
                (value.interaction.run_id, value.interaction.request_sequence)
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
        offset = 0 if cursor_state is None else cursor_state[0]
        fixed_cutoffs = None if cursor_state is None else dict(cursor_state[1])
        target = offset + limit + 1

        facts: list[AttachmentFact] = []
        accepted_ids: set[str] = set()
        view = record.stored_user_input.view
        if view is not None:
            raw_attachments = view.get("attachments", [])
            if not isinstance(raw_attachments, list):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            for raw in raw_attachments:
                if not isinstance(raw, Mapping):
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                fact = _project_attachment_fact(record.execution_id, raw)
                if fact.fact != "accepted":
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                facts.append(fact)
                accepted_ids.add(fact.attachment_id)

        cutoffs: list[tuple[int, int]] = []
        if record.binding_kind != "task":
            current_sequences = await self._segment_sequences(record, tenant_id)
            if fixed_cutoffs is None:
                sequences = current_sequences
            else:
                if any(
                    sequence not in current_sequences
                    for sequence in fixed_cutoffs
                ):
                    raise AIError(ErrorCode.CURSOR_INVALID)
                sequences = tuple(sorted(fixed_cutoffs))

            run_high_waters: list[tuple[int, str, int]] = []
            for segment_sequence in sequences:
                run_id = step_run_id(
                    namespace=self._namespace,
                    tenant_id=tenant_id,
                    execution_id=record.execution_id,
                    segment_sequence=segment_sequence,
                )
                current_high_water = await self._store.model_interaction_count(
                    run_id=run_id,
                )
                high_water = (
                    current_high_water
                    if fixed_cutoffs is None
                    else fixed_cutoffs[segment_sequence]
                )
                if high_water > current_high_water:
                    raise AIError(ErrorCode.CURSOR_INVALID)
                cutoffs.append((segment_sequence, high_water))
                run_high_waters.append((segment_sequence, run_id, high_water))

            for segment_sequence, run_id, high_water in run_high_waters:
                after_sequence = 0
                while after_sequence < high_water and len(facts) < target:
                    batch_limit = min(256, high_water - after_sequence)
                    values = await self._store.list_model_interactions(
                        run_id=run_id,
                        after_request_sequence=after_sequence,
                        limit=batch_limit,
                    )
                    if len(values) != batch_limit:
                        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                    for value in values:
                        if (
                            not isinstance(value, ModelInteractionRecord)
                            or value.run_id != run_id
                            or value.request_sequence != after_sequence + 1
                        ):
                            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                        after_sequence = value.request_sequence
                        for raw in value.attachments:
                            fact = _project_attachment_fact(
                                record.execution_id,
                                raw,
                                segment_sequence=segment_sequence,
                                request_sequence=value.request_sequence,
                                step_index=value.step_index,
                            )
                            if fact.fact == "accepted":
                                if fact.attachment_id in accepted_ids:
                                    continue
                                accepted_ids.add(fact.attachment_id)
                            facts.append(fact)
                            if len(facts) >= target:
                                break
                        if len(facts) >= target:
                            break

                if len(facts) >= target:
                    break

        if offset > len(facts):
            raise AIError(ErrorCode.CURSOR_INVALID)
        page = facts[offset : offset + limit + 1]
        selected = tuple(page[:limit])
        next_cursor = None
        if len(page) > limit:
            next_cursor = _attachment_fact_cursor(
                tenant_id,
                execution_id,
                offset + limit,
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

        current_sequences = await self._segment_sequences(record, tenant_id)
        explicit_cutoffs = _normalize_usage_cutoffs(cutoffs)
        if explicit_cutoffs is None:
            fixed_cutoffs: list[UsageReadCutoff] = []
            for segment_sequence in current_sequences:
                run_id = step_run_id(
                    namespace=self._namespace,
                    tenant_id=tenant_id,
                    execution_id=record.execution_id,
                    segment_sequence=segment_sequence,
                )
                fixed_cutoffs.append(
                    UsageReadCutoff(
                        record.execution_id,
                        segment_sequence,
                        await self._store.model_interaction_count(run_id=run_id),
                    )
                )
        else:
            if any(value.execution_id != record.execution_id for value in explicit_cutoffs):
                raise AIError(ErrorCode.CURSOR_INVALID)
            fixed_cutoffs = list(explicit_cutoffs)
        current_set = set(current_sequences)
        if any(value.segment_sequence not in current_set for value in fixed_cutoffs):
            raise AIError(ErrorCode.CURSOR_INVALID)

        logical_requests = 0
        succeeded_requests = 0
        failed_requests = 0
        cancelled_requests = 0
        output_correction_retries = 0
        input_tokens = 0
        output_tokens = 0
        cache_read_tokens = 0
        cache_write_tokens = 0
        model_duration_ns = 0
        unknown_usage_requests = 0
        unknown_duration_requests = 0

        for cutoff in fixed_cutoffs:
            run_id = step_run_id(
                namespace=self._namespace,
                tenant_id=tenant_id,
                execution_id=record.execution_id,
                segment_sequence=cutoff.segment_sequence,
            )
            current_high_water = await self._store.model_interaction_count(run_id=run_id)
            if cutoff.request_sequence > current_high_water:
                raise AIError(ErrorCode.CURSOR_INVALID)
            after_sequence = 0
            while after_sequence < cutoff.request_sequence:
                batch_limit = min(500, cutoff.request_sequence - after_sequence)
                values = await self._store.list_model_interactions(
                    run_id=run_id,
                    after_request_sequence=after_sequence,
                    limit=batch_limit,
                )
                if len(values) != batch_limit:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                for raw in values:
                    if not isinstance(raw, ModelInteractionRecord):
                        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                    expected = after_sequence + 1
                    if (
                        raw.run_id != run_id
                        or raw.request_sequence != expected
                        or raw.status not in {"SUCCEEDED", "FAILED", "CANCELLED"}
                    ):
                        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                    after_sequence = raw.request_sequence
                    logical_requests += 1
                    if raw.duration_ns is None:
                        unknown_duration_requests += 1
                    else:
                        model_duration_ns += raw.duration_ns
                    if raw.status == "SUCCEEDED":
                        succeeded_requests += 1
                    elif raw.status == "FAILED":
                        failed_requests += 1
                    else:
                        cancelled_requests += 1
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
        interaction: ModelInteractionRecord,
        execution_id: str,
        segment_sequence: int,
        depth: int,
        resolved: object | None,
        *,
        include_content: bool,
        started_at: datetime | None,
        finished_at: datetime | None,
    ) -> ModelInteractionItem:
        request: dict[str, JsonValue] = {}
        response: JsonValue | None = None
        if include_content:
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
            segment_sequence=segment_sequence,
            depth=depth,
            request_sequence=interaction.request_sequence,
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
            started_at=started_at,
            finished_at=finished_at,
        )

    async def _model_request_times(
        self,
        run_id: str,
    ) -> dict[int, tuple[datetime | None, datetime | None]]:
        values: dict[int, list[datetime | None]] = {}
        for event in await self._store.list_events(run_id=run_id):
            if event.kind not in {
                "model_request_started",
                "model_request_completed",
                "model_request_failed",
            }:
                continue
            raw_sequence = event.metadata.get(REQUEST_SEQUENCE_METADATA_KEY)
            if raw_sequence is None:
                continue
            if not raw_sequence.isdigit() or int(raw_sequence) < 1:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            sequence = int(raw_sequence)
            pair = values.setdefault(sequence, [None, None])
            index = 0 if event.kind == "model_request_started" else 1
            timestamp = _event_timestamp(event)
            if pair[index] is not None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            pair[index] = timestamp
        return {
            sequence: (pair[0], pair[1])
            for sequence, pair in values.items()
        }


    async def _tool_call_metadata(
        self,
        run_id: str,
        *,
        execution_id: str,
        tenant_id: str,
        event_high_water: int,
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
        events = await self._store.list_events(run_id=run_id)
        if event_high_water < 0 or event_high_water > len(events):
            raise AIError(ErrorCode.CURSOR_INVALID)
        values: dict[str, list[object | None]] = {}
        for event in events[:event_high_water]:
            if event.kind not in {
                "tool_call_started",
                "tool_call_completed",
                "tool_call_failed",
            }:
                continue
            call_id = event.tool_call_id
            if call_id is None or not call_id:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            value = values.setdefault(call_id, [None, None, None, "STARTED", None, None])
            raw_request = event.metadata.get(REQUEST_SEQUENCE_METADATA_KEY)
            if raw_request is not None:
                if not raw_request.isdigit() or int(raw_request) < 1:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                request_sequence = int(raw_request)
                if value[4] is not None and value[4] != request_sequence:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                value[4] = request_sequence
            timestamp = _event_timestamp(event)
            if event.kind == "tool_call_started":
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
            value[3] = "SUCCEEDED" if event.kind == "tool_call_completed" else "FAILED"

        if self._tool_operations is not None and values:
            operations = await self._tool_operations.list_by_execution(
                execution_id,
                tenant_id=tenant_id,
            )
            by_call: dict[str, ToolOperationRecord] = {}
            for operation in operations:
                if operation.step_run_id != run_id:
                    continue
                if operation.tool_call_id in by_call:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                by_call[operation.tool_call_id] = operation
            for call_id, value in values.items():
                operation = by_call.get(call_id)
                if operation is not None:
                    value[5] = operation.tool_operation_id

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
        high_waters: Mapping[tuple[str, int], tuple[int, int]],
        tenant_id: str,
        limit: int,
    ) -> list[_HistoryOccurrence]:
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
        cursor_prefix = None if cursor_source is None else cursor_source.merge_prefix
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
                identity = (source.record.execution_id, source.segment_sequence)
                high_waters_for_source = high_waters.get(identity)
                if high_waters_for_source is None:
                    raise AIError(ErrorCode.CURSOR_INVALID)
                message_high_water, event_high_water = high_waters_for_source
                if start_message_index > message_high_water:
                    raise AIError(ErrorCode.CURSOR_INVALID)
                iterator = self._iter_history_source(
                    source,
                    tenant_id=tenant_id,
                    start_message_index=start_message_index,
                    end_message_index=message_high_water,
                    event_high_water=event_high_water,
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
        end_message_index: int,
        event_high_water: int,
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
            end=end_message_index,
            from_cursor=from_cursor,
        )
        tool_metadata = await self._tool_call_metadata(
            run_id,
            execution_id=source.record.execution_id,
            tenant_id=tenant_id,
            event_high_water=event_high_water,
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
                metadata = (
                    None
                    if value.tool_call_id is None
                    else tool_metadata.get(value.tool_call_id)
                )
                yield _HistoryOccurrence(
                    ExecutionHistoryItem(
                        execution_id=source.record.execution_id,
                        sequence=message_index + 1,
                        item_kind=value.item_kind,
                        content=value.content,
                        tool_name=value.tool_name,
                        tool_call_id=value.tool_call_id,
                        content_included=True,
                        segment_sequence=source.segment_sequence,
                        request_sequence=None if metadata is None else metadata[4],
                        tool_operation_id=None if metadata is None else metadata[5],
                        started_at=None if metadata is None else metadata[0],
                        finished_at=None if metadata is None else metadata[1],
                        duration_ns=None if metadata is None else metadata[2],
                        status=None if metadata is None else metadata[3],
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

    async def _transcript_high_water(self, run_id: str) -> int:
        if isinstance(self._store, _RangedTranscriptStore):
            return await self._store.transcript_message_count(run_id)
        count = 0
        async for _message in self._store.iter_messages(run_id=run_id):
            count += 1
        return count

    async def _message_range(
        self,
        run_id: str,
        *,
        start: int,
        end: int,
        from_cursor: bool,
    ) -> AsyncIterator[object]:
        if start < 0 or end < start:
            raise AIError(ErrorCode.CURSOR_INVALID)
        if isinstance(self._store, _RangedTranscriptStore):
            total = await self._store.transcript_message_count(run_id)
            if end > total or from_cursor and start == end:
                raise AIError(ErrorCode.CURSOR_INVALID)
            async for message in self._store.iter_message_range(
                run_id=run_id,
                start=start,
                end=end,
            ):
                yield message
            return
        index = 0
        async for message in self._store.iter_messages(run_id=run_id):
            if index >= end:
                break
            if index >= start:
                yield message
            index += 1
        if index < end or from_cursor and start == end:
            raise AIError(ErrorCode.CURSOR_INVALID)

    async def transcript(
        self, execution_id: str, *, tenant_id: str, cursor: str | None, limit: int
    ) -> Page[TranscriptItem]:
        limit = validate_page_limit(limit)
        record = await self._executions.get(execution_id, tenant_id=tenant_id)
        if record is None:
            raise AIError(ErrorCode.STORAGE_NOT_FOUND)
        if record.agent_run_sequence == 0:
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
            segment_sequence = record.agent_run_sequence
            run_id = step_run_id(
                namespace=self._namespace,
                tenant_id=tenant_id,
                execution_id=execution_id,
                segment_sequence=segment_sequence,
            )
            message_index = 0
            item_offset = 0
        else:
            (
                run_id,
                segment_sequence,
                high_water,
                message_index,
                item_offset,
            ) = cursor_state
            if (
                segment_sequence > record.agent_run_sequence
                or run_id
                != step_run_id(
                    namespace=self._namespace,
                    tenant_id=tenant_id,
                    execution_id=execution_id,
                    segment_sequence=segment_sequence,
                )
            ):
                raise AIError(ErrorCode.CURSOR_INVALID)

        run = await self._store.get_run(run_id=run_id)
        if run is None:
            if cursor is not None:
                raise AIError(ErrorCode.CURSOR_INVALID)
            if record.status is ExecutionStatus.SUCCEEDED:
                raise AIError(ErrorCode.EXECUTION_HISTORY_UNAVAILABLE)
            return Page((), None)
        if cursor_state is None:
            high_water = await self._transcript_high_water(run_id)
        elif await self._transcript_high_water(run_id) < high_water:
            raise AIError(ErrorCode.CURSOR_INVALID)
        _validate_run(
            run,
            run_id,
            step_conversation_id(
                namespace=self._namespace,
                tenant_id=tenant_id,
                execution_id=execution_id,
            ),
            segment_sequence,
        )
        messages = self._message_range(
            run_id,
            start=message_index,
            end=high_water,
            from_cursor=cursor is not None,
        )
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
                run_id,
                segment_sequence,
                high_water,
                next_coordinate[0],
                next_coordinate[1],
                self._cursor_signer,
            )
        return Page(selected, next_cursor)

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

    def __init__(
        self,
        *,
        store: StepStore,
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
        continuation_step_run_id: "str | None",
        continuation_history_id: "str | None" = None,
        cursor: "str | None",
        limit: int,
    ) -> "Page[SessionHistoryItem]":
        limit = validate_page_limit(limit)
        if continuation_step_run_id is None:
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
                current.step_run_id != continuation_step_run_id
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
            else continuation_step_run_id
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
                run = await self._store.get_run(
                    run_id=continuation_step_run_id
                )
                snapshot = await self._store.latest_snapshot(
                    run_id=continuation_step_run_id,
                    include_interrupted=True,
                )
                if (
                    run is None
                    or snapshot is None
                    or snapshot.run_id != continuation_step_run_id
                    or snapshot.state != "complete"
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
            if cursor_high_water is not None and cursor_high_water != total_messages:
                raise AIError(ErrorCode.CURSOR_INVALID)
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


def _execution_filter_digest(execution_id: str, projection_version: int) -> str:
    return canonical_sha256(
        {
            "execution_id": execution_id,
            "projection_version": projection_version,
        }
    )


def _attachment_fact_filter_digest(execution_id: str) -> str:
    return _execution_filter_digest(execution_id, _ATTACHMENT_FACT_PROJECTION_VERSION)


def _decode_attachment_fact_cursor(
    cursor: str | None,
    *,
    tenant_id: str,
    execution_id: str,
    signer: CursorSigner,
) -> tuple[int, tuple[tuple[int, int], ...]] | None:
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
        or set(value) != {"offset", "cutoffs"}
    ):
        raise AIError(ErrorCode.CURSOR_INVALID)
    offset = value.get("offset")
    raw_cutoffs = value.get("cutoffs")
    if (
        isinstance(offset, bool)
        or not isinstance(offset, int)
        or offset < 0
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
    return offset, tuple(cutoffs)


def _attachment_fact_cursor(
    tenant_id: str,
    execution_id: str,
    offset: int,
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
                "offset": offset,
                "cutoffs": [list(value) for value in cutoffs],
            },
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ),
    )


def _project_attachment_fact(
    execution_id: str,
    value: Mapping[str, JsonValue],
    *,
    segment_sequence: int | None = None,
    request_sequence: int | None = None,
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
            segment_sequence=segment_sequence,
            request_sequence=(
                request_sequence if fact == "included_in_request" else None
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
        groups.setdefault(value.interaction.run_id, []).append(value)
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
                key=lambda item: (item.execution_id, item.segment_sequence),
            )
        )
    except (AttributeError, TypeError) as error:
        raise AIError(ErrorCode.CURSOR_INVALID) from error
    if (
        any(not isinstance(item, UsageReadCutoff) for item in normalized)
        or len({(item.execution_id, item.segment_sequence) for item in normalized})
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
                    occurrence.segment_sequence,
                    occurrence.interaction.request_sequence,
                ],
                "cutoffs": [
                    [
                        value.execution_id,
                        value.segment_sequence,
                        value.request_sequence,
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
) -> "tuple[tuple[str, int, int, int], tuple[tuple[str, int, int, int], ...]] | None":
    if cursor is None:
        return None
    payload = decode_runtime_cursor(
        cursor,
        signer,
        tenant_id=tenant_id,
        resource_kind="execution_history",
        filter_digest=_execution_filter_digest(
            execution_id, _EXECUTION_HISTORY_PROJECTION_VERSION
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
    cutoffs: list[tuple[str, int, int, int]] = []
    for raw in raw_cutoffs:
        if (
            not isinstance(raw, list)
            or len(raw) != 4
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
        ):
            raise AIError(ErrorCode.CURSOR_INVALID)
        cutoffs.append((raw[0], raw[1], raw[2], raw[3]))
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
    cutoffs: tuple[tuple[str, int, int, int], ...],
    signer: CursorSigner,
) -> str:
    normalized = tuple(sorted(cutoffs, key=lambda item: (item[0], item[1])))
    if len({(item[0], item[1]) for item in normalized}) != len(normalized):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return encode_runtime_cursor(
        signer,
        tenant_id=tenant_id,
        resource_kind="execution_history",
        filter_digest=_execution_filter_digest(
            execution_id, _EXECUTION_HISTORY_PROJECTION_VERSION
        ),
        position=json.dumps(
            {
                "position": [
                    occurrence.source_execution_id,
                    occurrence.segment_sequence,
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
) -> "tuple[tuple[str, int, int], tuple[tuple[str, int, int], ...]] | None":
    if cursor is None:
        return None
    payload = decode_runtime_cursor(
        cursor,
        signer,
        tenant_id=tenant_id,
        resource_kind="execution_trace",
        filter_digest=_execution_filter_digest(
            execution_id, _EXECUTION_TRACE_PROJECTION_VERSION
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


def _trace_cursor_index(
    coordinate: "tuple[str, int, int] | None",
    *,
    occurrences: Sequence[_TraceOccurrence],
) -> int:
    if coordinate is None:
        return 0
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
    cutoffs: tuple[tuple[str, int, int], ...],
    signer: CursorSigner,
) -> str:
    normalized = tuple(sorted(cutoffs, key=lambda item: (item[0], item[1])))
    if len({(item[0], item[1]) for item in normalized}) != len(normalized):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return encode_runtime_cursor(
        signer,
        tenant_id=tenant_id,
        resource_kind="execution_trace",
        filter_digest=_execution_filter_digest(
            execution_id, _EXECUTION_TRACE_PROJECTION_VERSION
        ),
        position=json.dumps(
            {
                "position": [
                    occurrence.source_execution_id,
                    occurrence.segment_sequence,
                    occurrence.event_sequence,
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
) -> "tuple[str, int, int, int, int] | None":
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
    coordinate = _decode_position(payload.position, 5)
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
    ):
        raise AIError(ErrorCode.CURSOR_INVALID)
    return (
        coordinate[0],
        coordinate[1],
        coordinate[2],
        coordinate[3],
        coordinate[4],
    )


def _transcript_cursor(
    tenant_id: str,
    execution_id: str,
    run_id: str,
    segment_sequence: int,
    high_water: int,
    message_index: int,
    item_offset: int,
    signer: CursorSigner,
) -> str:
    if (
        not run_id
        or segment_sequence < 1
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
                run_id,
                segment_sequence,
                high_water,
                message_index,
                item_offset,
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
