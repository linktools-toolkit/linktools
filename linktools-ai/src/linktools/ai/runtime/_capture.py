#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime-owned capture for step persistence and model interactions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol, cast

from pydantic_ai.messages import ModelMessage, ModelResponse
from pydantic_ai.models import Model, ModelRequestParameters
from pydantic_ai.settings import ModelSettings

from ..core import JsonValue, UsageMetrics
from ..errors import AIError, ErrorCode
from ._attachment import request_attachment_facts
from ._journal import (
    DURATION_NS_METADATA_KEY,
    MODEL_USAGE_CACHE_READ_METADATA_KEY,
    MODEL_USAGE_CACHE_WRITE_METADATA_KEY,
    MODEL_USAGE_INPUT_METADATA_KEY,
    MODEL_USAGE_OUTPUT_METADATA_KEY,
    ModelRequestFact,
)
from ._message import freeze_model_messages, project_transient_binary_content
from ._model_interaction import (
    StagedContextProjection,
    StagedModelInteraction,
    build_context_projection,
    build_inline_context_projection,
    model_identity,
    request_envelope,
)
from .state._contracts import LoadedModelContext, TranscriptMessageRef
from .state._plan import RuntimeDomain
from .state._step_contracts import (
    ContinuableSnapshot,
    EventKind,
    RunRecord,
    StepEvent,
    StepStore,
)


class _InteractionStagingPort(Protocol):
    def intern_payload(self, run_id: str, payload: bytes) -> tuple[str, int]: ...

    def stage_model_interaction(self, interaction: object) -> None: ...


class RuntimeCaptureStore:
    """Own one agent attempt's stable capture inputs and staged persistence facts."""

    def __init__(
        self,
        store: StepStore,
        *,
        execution_id: str | None,
        step_run_id: str,
        initial_messages: Sequence[ModelMessage] = (),
        initial_context: LoadedModelContext | None = None,
        initial_attachments: Sequence[Mapping[str, JsonValue]] = (),
    ) -> None:
        if not isinstance(step_run_id, str) or not step_run_id:
            raise ValueError("step_run_id is required")
        self._store = store
        self._interaction_store = cast(_InteractionStagingPort, store)
        self._execution_id = execution_id
        self._step_run_id = step_run_id
        self._run: RunRecord | None = None
        self._event_sequence = 0
        self._initial_attachments = tuple(dict(value) for value in initial_attachments)
        self._accepted_attachment_ids = {
            attachment_id
            for value in self._initial_attachments
            if isinstance((attachment_id := value.get("attachment_id")), str)
            and attachment_id
        }

        frozen_initial = freeze_model_messages(initial_messages)
        baseline = initial_context or LoadedModelContext(())
        baseline_messages = baseline.model_messages()
        baseline_refs: tuple[TranscriptMessageRef | None, ...]
        if baseline.messages:
            if (
                len(baseline_messages) != len(frozen_initial)
                or freeze_model_messages(baseline_messages) != frozen_initial
            ):
                raise AIError(
                    ErrorCode.STORAGE_INTEGRITY_ERROR,
                    "initial model context does not match initial messages",
                )
            baseline_refs = tuple(
                value.source
                if value.source is not None
                and value.source.source_domain is RuntimeDomain.CONVERSATION
                else None
                for value in baseline.messages
            )
        else:
            baseline_refs = (None,) * len(frozen_initial)
        self._source_messages: list[ModelMessage] = list(frozen_initial)
        self._source_refs: list[TranscriptMessageRef | int | None] = list(
            baseline_refs
        )
        self._transcript_messages: list[ModelMessage] = []

        self._projection_source_count: int | None = None
        self._projection_messages: tuple[ModelMessage, ...] | None = None

        self._interaction_projections: dict[int, StagedContextProjection] = {}
        self._interaction_payloads: dict[int, str] = {}
        self._interaction_models: dict[int, dict[str, str]] = {}
        self._interaction_attachments: dict[
            int,
            tuple[Mapping[str, JsonValue], ...],
        ] = {}

    @property
    def step_run_id(self) -> str:
        return self._step_run_id

    async def register_run(self, record: RunRecord) -> None:
        if record.run_id != self._step_run_id:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if self._run is not None and self._run != record:
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        self._run = record
        await self._store.register_run(record, execution_id=self._execution_id)

    async def append_event(self, event: StepEvent) -> None:
        if event.run_id != self._step_run_id:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        await self._store.append_event(event, execution_id=self._execution_id)

    async def record_event(
        self,
        kind: EventKind,
        step_index: int,
        *,
        tool_call_id: str | None = None,
        tool_name: str | None = None,
        error: str | None = None,
        metadata: Mapping[str, str] | None = None,
    ) -> None:
        run = self._run
        if run is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        event_index = self._event_sequence
        self._event_sequence += 1
        await self.append_event(
            StepEvent(
                run_id=run.run_id,
                kind=kind,
                step_index=step_index,
                conversation_id=run.conversation_id,
                parent_run_id=run.parent_run_id,
                agent_name=run.agent_name,
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                error=error,
                metadata={} if metadata is None else dict(metadata),
                idempotency_key=(
                    f"{event_index}:{step_index}:{kind}:{tool_call_id or ''}"
                ),
                event_index=event_index,
            )
        )

    async def save_snapshot(self, snapshot: ContinuableSnapshot) -> None:
        if snapshot.run_id != self._step_run_id:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        await self._store.save_snapshot(snapshot, execution_id=self._execution_id)

    async def latest_snapshot(
        self,
        *,
        include_interrupted: bool = False,
    ) -> ContinuableSnapshot | None:
        return await self._store.latest_snapshot(
            run_id=self._step_run_id,
            include_interrupted=include_interrupted,
        )

    def append_transcript_message(self, message: ModelMessage) -> ModelMessage:
        frozen = freeze_model_messages((message,))[0]
        local_index = len(self._transcript_messages)
        self._transcript_messages.append(frozen)
        self._source_messages.append(frozen)
        self._source_refs.append(local_index)
        return frozen

    def transcript_messages(self) -> tuple[ModelMessage, ...]:
        return tuple(self._transcript_messages)

    def remember_context_projection(
        self,
        source: Sequence[ModelMessage],
        projected: Sequence[ModelMessage] | None,
    ) -> None:
        if projected is None:
            self._projection_source_count = None
            self._projection_messages = None
            return
        self._projection_source_count = len(tuple(source))
        self._projection_messages = freeze_model_messages(projected)

    def snapshot_context(
        self,
        messages: Sequence[ModelMessage],
        *,
        pending: ModelMessage | None = None,
    ) -> tuple[list[ModelMessage], int | None]:
        current = freeze_model_messages(messages)
        source_count = self._projection_source_count
        projected = self._projection_messages
        if source_count is not None and projected is not None:
            if source_count > len(current):
                raise AIError(
                    ErrorCode.STORAGE_INTEGRITY_ERROR,
                    "captured context projection exceeds live history",
                )
            current = (*projected, *current[source_count:])
        pending_index = None
        if pending is not None:
            frozen_pending = freeze_model_messages((pending,))[0]
            pending_index = len(current)
            current = (*current, frozen_pending)
        current = project_transient_binary_content(current)
        return list(freeze_model_messages(current)), pending_index

    def begin_model_interaction(
        self,
        fact: ModelRequestFact,
        model: Model,
        messages: Sequence[ModelMessage],
        model_settings: ModelSettings | None,
        parameters: ModelRequestParameters,
        streaming: bool,
        model_id: str | None = None,
        source_messages: Sequence[ModelMessage] | None = None,
    ) -> None:
        del source_messages
        frozen = freeze_model_messages(messages)
        projection = build_context_projection(
            self._source_messages,
            frozen,
            lambda payload: self._interaction_store.intern_payload(
                self._step_run_id,
                payload,
            ),
            source_refs=self._source_refs,
        )
        _envelope, envelope_bytes = request_envelope(
            model_settings=model_settings,
            parameters=parameters,
            streaming=streaming,
        )
        digest, _size = self._interaction_store.intern_payload(
            self._step_run_id,
            envelope_bytes,
        )
        self._interaction_projections[fact.request_sequence] = projection
        self._interaction_payloads[fact.request_sequence] = digest
        self._interaction_models[fact.request_sequence] = model_identity(
            model,
            route_id=model_id,
        )
        self._interaction_attachments[fact.request_sequence] = request_attachment_facts(
            frozen,
            self._initial_attachments,
            accepted_attachment_ids=self._accepted_attachment_ids,
        )

    def finish_model_interaction(
        self,
        fact: ModelRequestFact,
        *,
        model: Model,
        response: ModelResponse | None,
        status: str,
        error_code: str | None,
        duration_ns: int,
        usage: object | None,
    ) -> None:
        del model
        request_sequence = fact.request_sequence
        try:
            projection = self._interaction_projections.pop(request_sequence)
            envelope_digest = self._interaction_payloads.pop(request_sequence)
            model_value = self._interaction_models.pop(request_sequence)
            attachments = self._interaction_attachments.pop(request_sequence)
        except KeyError as error:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error

        response_projection = None
        if response is not None:
            frozen_response = freeze_model_messages((response,))
            response_projection = build_inline_context_projection(
                frozen_response,
                lambda payload: self._interaction_store.intern_payload(
                    self._step_run_id,
                    payload,
                ),
            )
        self._interaction_store.stage_model_interaction(
            StagedModelInteraction(
                self._step_run_id,
                fact.step_index,
                request_sequence,
                fact.purpose,
                fact.output_retry_index,
                model_value,
                projection,
                envelope_digest,
                response_projection,
                status,
                error_code,
                duration_ns,
                _usage_metrics(usage),
                attachments,
            )
        )

    async def record_model_event(
        self,
        fact: ModelRequestFact,
        *,
        phase: str,
        response: ModelResponse | None = None,
        error_code: str | None = None,
        include_observation: bool,
    ) -> None:
        run = self._run
        if run is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        kinds = {
            "started": "model_request_started",
            "completed": "model_request_completed",
            "failed": "model_request_failed",
            "cancelled": "model_request_failed",
        }
        kind = kinds.get(phase)
        if kind is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        metadata = fact.metadata(
            include_observation=include_observation and fact.duration_ns is not None
        )
        if not include_observation:
            metadata.pop(DURATION_NS_METADATA_KEY, None)
        if response is not None:
            usage = response.usage
            metadata.update(
                {
                    MODEL_USAGE_INPUT_METADATA_KEY: str(usage.input_tokens),
                    MODEL_USAGE_OUTPUT_METADATA_KEY: str(usage.output_tokens),
                    MODEL_USAGE_CACHE_READ_METADATA_KEY: str(usage.cache_read_tokens),
                    MODEL_USAGE_CACHE_WRITE_METADATA_KEY: str(usage.cache_write_tokens),
                }
            )
        await self.record_event(
            cast(EventKind, kind),
            fact.step_index,
            error=error_code,
            metadata=metadata,
        )


def _usage_metrics(value: object | None) -> UsageMetrics | None:
    if value is None:
        return None
    return UsageMetrics(
        model_requests=1,
        input_tokens=int(getattr(value, "input_tokens", 0)),
        output_tokens=int(getattr(value, "output_tokens", 0)),
        cache_read_tokens=int(getattr(value, "cache_read_tokens", 0)),
        cache_write_tokens=int(getattr(value, "cache_write_tokens", 0)),
    )


__all__ = ["RuntimeCaptureStore"]
