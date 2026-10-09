#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Agent-run recording for step persistence and model interactions."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import replace
from datetime import datetime, timezone
from typing import Protocol, cast

from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ToolCallPart,
    ToolReturnPart,
    RetryPromptPart,
    ModelResponsePart,
    ModelResponsePartDelta,
)
from pydantic_ai.models import Model, ModelRequestParameters
from pydantic_ai.settings import ModelSettings

from ..core import JsonValue, UsageMetrics
from ..errors import AIError, ErrorCode
from ._attachment import request_attachment_facts
from ._transcript_staging import StagedTranscript
from ._journal import (
    DURATION_NS_METADATA_KEY,
    MODEL_REQUEST_SEQ_METADATA_KEY,
    REPLAYED_TOOL_CALL_INDICES_METADATA_KEY,
    MESSAGE_SEQ_METADATA_KEY,
    MODEL_USAGE_CACHE_READ_METADATA_KEY,
    MODEL_USAGE_CACHE_WRITE_METADATA_KEY,
    MODEL_USAGE_INPUT_METADATA_KEY,
    MODEL_USAGE_OUTPUT_METADATA_KEY,
    ModelRequestFact,
    _await_request_handoff,
)
from ._message import encode_model_messages, freeze_model_messages, project_transient_binary_content
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
from .state._steps import RuntimeAgentRunStore, StagingAgentRunStore
from .state._step_contracts import (
    AgentRunCheckpoint,
    StepEventType,
    AgentRunRecord,
    StepEvent,
    AgentRunStore,
)


class _RunStagingPort(Protocol):
    def stage_transcript(
        self, agent_run_id: str, transcript: StagedTranscript
    ) -> None: ...

    def intern_payload(self, agent_run_id: str, payload: bytes) -> tuple[str, int]: ...

    def stage_model_interaction(self, interaction: object) -> None: ...

    def prepare_model_interaction(self, interaction: object) -> None: ...


class AgentRunRecorder:
    """Own one agent attempt's stable inputs and staged persistence facts."""

    def __init__(
        self,
        store: AgentRunStore,
        *,
        execution_id: str | None,
        agent_run_id: str,
        initial_messages: Sequence[ModelMessage] = (),
        initial_context: LoadedModelContext | None = None,
        initial_attachments: Sequence[Mapping[str, JsonValue]] = (),
        history_boundary: Callable[[], Awaitable[None]] | None = None,
        checkpoint_sink: Callable[[AgentRunCheckpoint], Awaitable[None]] | None = None,
    ) -> None:
        if not isinstance(agent_run_id, str) or not agent_run_id:
            raise ValueError("agent_run_id is required")
        self._store = store
        self._staging_store = cast(_RunStagingPort, store)
        self._execution_id = execution_id
        self._history_boundary = history_boundary
        self._checkpoint_sink = checkpoint_sink
        self._agent_run_id = agent_run_id
        self._run: AgentRunRecord | None = None
        self._next_event_index = 0
        self._tool_events: dict[str, set[str]] = {}
        self._tool_execution_metadata: dict[str, tuple[dict[str, str], str | None]] = {}
        self._tool_results: dict[str, ToolReturnPart | RetryPromptPart] = {}
        self._restored_tool_results: set[str] = set()
        self._restored_tool_calls: dict[str, ToolCallPart] = {}
        self._restored_pending = False
        self._model_request_seq_by_tool_call: dict[str, int] = {}
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
        baseline_refs: tuple[TranscriptMessageRef | int | None, ...]
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
                value.source.message_index
                if value.source is not None
                and value.source.source_domain is RuntimeDomain.RECOVERY
                and value.source.owner_id == agent_run_id
                else value.source
                if value.source is not None
                and value.source.source_domain is RuntimeDomain.CONVERSATION
                else None
                for value in baseline.messages
            )
        else:
            baseline_refs = (None,) * len(frozen_initial)
        self._source_keys: list[bytes] = [
            encode_model_messages((message,))
            for message in frozen_initial
        ]
        self._source_refs: list[TranscriptMessageRef | int | None] = list(
            baseline_refs
        )
        self._source_indices: dict[bytes, int | None] = {}
        for index, key in enumerate(self._source_keys):
            self._source_indices[key] = (
                None if key in self._source_indices else index
            )
        self._transcript_messages: list[ModelMessage] = []
        self._pending_parts: dict[str, ModelMessage] = {}

        self._projection_source_count: int | None = None
        self._projection_messages: tuple[ModelMessage, ...] | None = None

        self._interaction_projections: dict[int, StagedContextProjection | None] = {}
        self._interaction_payloads: dict[int, str | None] = {}
        self._interaction_models: dict[int, dict[str, str]] = {}
        self._interaction_attachments: dict[
            int,
            tuple[Mapping[str, JsonValue], ...],
        ] = {}

    @property
    def agent_run_id(self) -> str:
        return self._agent_run_id

    async def register_agent_run(self, record: AgentRunRecord) -> None:
        if record.agent_run_id != self._agent_run_id:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if self._run is not None and self._run != record:
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        self._run = record
        await self._store.register_agent_run(record, execution_id=self._execution_id)
        previous = await self.latest_checkpoint(include_interrupted=True)
        if previous is not None:
            self._transcript_messages = list(freeze_model_messages(previous.messages))
        observed = (
            self._store.staged_transcript(record.agent_run_id)
            if isinstance(self._store, (RuntimeAgentRunStore, StagingAgentRunStore))
            else None
        )
        if observed is not None:
            self._transcript_messages = list(freeze_model_messages(observed.messages))
            if observed.pending is not None:
                self._restored_pending = True
                if len(observed.pending.parts) != len(observed.pending_keys):
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                for key, part in zip(observed.pending_keys, observed.pending.parts):
                    self._pending_parts[key] = freeze_model_messages((
                        ModelResponse(parts=[part])
                        if isinstance(observed.pending, ModelResponse)
                        else ModelRequest(parts=[part]),
                    ))[0]
        self._restored_tool_results = {
            part.tool_call_id
            for message in self._transcript_messages
            for part in message.parts
            if isinstance(part, (ToolReturnPart, RetryPromptPart)) and part.tool_call_id
        }
        for message in (*self._transcript_messages, *self._pending_parts.values()):
            for part in message.parts:
                if isinstance(part, ToolCallPart) and part.tool_call_id:
                    previous_call = self._restored_tool_calls.get(part.tool_call_id)
                    if previous_call is not None and not self._same_tool_call(previous_call, part):
                        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                    self._restored_tool_calls[part.tool_call_id] = part
                if isinstance(part, (ToolReturnPart, RetryPromptPart)) and part.tool_call_id:
                    previous_result = self._tool_results.get(part.tool_call_id)
                    if previous_result is not None and not self._same_tool_result(previous_result, part):
                        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                    self._tool_results[part.tool_call_id] = part
        events = await self._store.list_events(agent_run_id=record.agent_run_id)
        for event in events:
            message_seq = event.metadata.get(MESSAGE_SEQ_METADATA_KEY)
            if event.event_type == "MODEL_REQUEST_SUCCEEDED" and message_seq is not None:
                model_request_seq = event.metadata.get(MODEL_REQUEST_SEQ_METADATA_KEY)
                if (
                    not message_seq.isdigit()
                    or int(message_seq) < 1
                    or model_request_seq is None
                    or not model_request_seq.isdigit()
                    or int(model_request_seq) < 1
                ):
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                message_index = int(message_seq) - 1
                if message_index >= len(self._transcript_messages):
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                response = self._transcript_messages[message_index]
                if not isinstance(response, ModelResponse):
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                replayed_indices: list[int] = []
                seen_call_ids: set[str] = set()
                for part_index, part in enumerate(response.parts):
                    if isinstance(part, ToolCallPart):
                        if part.tool_call_id in seen_call_ids:
                            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                        seen_call_ids.add(part.tool_call_id)
                        previous_request = self._model_request_seq_by_tool_call.get(part.tool_call_id)
                        if previous_request is not None:
                            replayed_indices.append(part_index)
                        else:
                            self._model_request_seq_by_tool_call[part.tool_call_id] = int(model_request_seq)
                replayed = event.metadata.get(REPLAYED_TOOL_CALL_INDICES_METADATA_KEY)
                if replayed != (json.dumps(replayed_indices) if replayed_indices else None):
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if event.tool_call_id is not None and event.event_type.startswith(
                "TOOL_CALL_"
            ):
                self._tool_events.setdefault(event.tool_call_id, set()).add(
                    event.event_type
                )
        self._next_event_index = max((event.event_index for event in events), default=-1) + 1

    async def append_event(self, event: StepEvent) -> None:
        if event.agent_run_id != self._agent_run_id:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        await self._store.append_event(event, execution_id=self._execution_id)

    async def record_event(
        self,
        event_type: StepEventType,
        step_index: int,
        *,
        tool_call_id: str | None = None,
        tool_name: str | None = None,
        error: str | None = None,
        metadata: Mapping[str, str] | None = None,
        timestamp: datetime | None = None,
    ) -> None:
        run = self._run
        if run is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if tool_call_id is not None and event_type.startswith("TOOL_CALL_"):
            self._tool_events.setdefault(tool_call_id, set()).add(event_type)
        event_index = self._next_event_index
        self._next_event_index += 1
        await self.append_event(
            StepEvent(
                agent_run_id=run.agent_run_id,
                event_type=event_type,
                step_index=step_index,
                timestamp=(datetime.now(timezone.utc) if timestamp is None else timestamp),
                agent_conversation_id=run.agent_conversation_id,
                parent_agent_run_id=run.parent_agent_run_id,
                agent_id=run.agent_id,
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                error=error,
                metadata={} if metadata is None else dict(metadata),
                idempotency_key=(
                    f"{event_index}:{step_index}:{event_type}:{tool_call_id or ''}"
                ),
                event_index=event_index,
            )
        )

    async def save_checkpoint(self, checkpoint: AgentRunCheckpoint) -> None:
        if checkpoint.agent_run_id != self._agent_run_id:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if self._checkpoint_sink is not None:
            await self._checkpoint_sink(checkpoint)
        else:
            await self._store.save_checkpoint(checkpoint, execution_id=self._execution_id)

    async def latest_checkpoint(
        self,
        *,
        include_interrupted: bool = False,
    ) -> AgentRunCheckpoint | None:
        return await self._store.latest_checkpoint(
            agent_run_id=self._agent_run_id,
            include_interrupted=include_interrupted,
        )

    def request_already_captured(self, messages: Sequence[ModelMessage]) -> bool:
        if self._pending_parts or not messages or not self._transcript_messages:
            return False
        latest = self._transcript_messages[-1]
        return isinstance(latest, ModelRequest) and latest == freeze_model_messages((messages[-1],))[0]

    def _pending_matches(self, message: ModelMessage) -> bool:
        if isinstance(message, ModelResponse):
            parts = {f"part:{index}": part for index, part in enumerate(message.parts)}
        else:
            parts = {
                f"{'tool_result' if isinstance(part, ToolReturnPart) else 'retry'}:{part.tool_call_id}": part
                for part in message.parts
                if isinstance(part, (ToolReturnPart, RetryPromptPart))
            }
        return all(
            type(pending) is type(message)
            and key in parts and pending.parts[0] == parts[key]
            for key, pending in self._pending_parts.items()
        )

    def append_transcript_message(self, message: ModelMessage) -> ModelMessage:
        frozen = freeze_model_messages((message,))[0]
        if isinstance(frozen, ModelRequest):
            parts = []
            for part in frozen.parts:
                if isinstance(part, (ToolReturnPart, RetryPromptPart)) and part.tool_call_id:
                    previous = self._tool_results.get(part.tool_call_id)
                    if previous is not None:
                        if not self._same_tool_result(previous, part):
                            raise AIError(ErrorCode.STORAGE_CONFLICT)
                        if part.tool_call_id in self._restored_tool_results:
                            continue
                        part = previous
                parts.append(part)
            if not parts and frozen.parts:
                return frozen
            frozen = replace(frozen, parts=parts)
        if self._restored_pending and self._pending_parts and not self._pending_matches(frozen):
            self.finish_transcript(interrupted=True)
        for part in frozen.parts:
            if isinstance(part, (ToolReturnPart, RetryPromptPart)) and part.tool_call_id:
                previous = self._tool_results.get(part.tool_call_id)
                if previous is not None and not self._same_tool_result(previous, part):
                    raise AIError(ErrorCode.STORAGE_CONFLICT)
                self._tool_results[part.tool_call_id] = part
        local_index = len(self._transcript_messages)
        self._transcript_messages.append(frozen)
        key = encode_model_messages((frozen,))
        self._source_indices[key] = (
            None if key in self._source_indices else len(self._source_keys)
        )
        self._source_keys.append(key)
        self._source_refs.append(local_index)
        self._pending_parts.clear()
        self._restored_pending = False
        self._stage_transcript()
        return frozen

    def stage_response_part(
        self, part: ModelResponsePart, part_index: int, *, complete: bool = True,
    ) -> int:
        if self._restored_pending and self._pending_parts and not isinstance(
            next(iter(self._pending_parts.values())), ModelResponse,
        ):
            self.finish_transcript(interrupted=True)
        if complete and isinstance(part, ToolCallPart) and part.tool_call_id in self._model_request_seq_by_tool_call:
            previous = self._restored_tool_calls.get(part.tool_call_id)
            if previous is None or not self._same_tool_call(previous, part):
                raise AIError(ErrorCode.STORAGE_CONFLICT)
        key = f"part:{part_index}"
        self._pending_parts[key] = freeze_model_messages(
            (ModelResponse(parts=[part]),)
        )[0]
        # A completed response part is still inside an unfinished request.
        # Keep it local until request completion or an interrupted terminal drain.
        return len(self._transcript_messages) + 1

    def stage_response_delta(self, delta: ModelResponsePartDelta, part_index: int) -> None:
        key = f"part:{part_index}"
        pending = self._pending_parts.get(key)
        if not isinstance(pending, ModelResponse) or len(pending.parts) != 1:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        self._pending_parts[key] = replace(pending, parts=[delta.apply(pending.parts[0])])

    def stage_tool_result(self, part: ToolReturnPart | RetryPromptPart) -> None:
        # Each result is complete even while sibling calls are still running.
        kind = "tool_result" if isinstance(part, ToolReturnPart) else "retry"
        self._pending_parts[f"{kind}:{part.tool_call_id}"] = freeze_model_messages(
            (ModelRequest(parts=[part]),)
        )[0]
        self._stage_transcript()

    async def commit_history_boundary(self) -> None:
        if self._history_boundary is not None:
            await self._history_boundary()

    def remember_tool_execution(
        self,
        tool_call_id: str,
        *,
        metadata: Mapping[str, str],
        error: str | None = None,
    ) -> None:
        self._tool_execution_metadata[tool_call_id] = (dict(metadata), error)

    async def record_tool_start(
        self, part: ToolCallPart | ToolReturnPart | RetryPromptPart, step_index: int
    ) -> None:
        if "TOOL_CALL_STARTED" in self._tool_events.get(part.tool_call_id, ()):
            return
        model_request_seq = self.model_request_seq_for_tool_call(part.tool_call_id)
        metadata = (
            {}
            if model_request_seq is None
            else {MODEL_REQUEST_SEQ_METADATA_KEY: str(model_request_seq)}
        )
        await self.record_event(
            "TOOL_CALL_STARTED",
            step_index,
            tool_call_id=part.tool_call_id,
            tool_name=part.tool_name,
            metadata=metadata,
        )
        await self.commit_history_boundary()

    async def record_tool_result_boundary(
        self, part: ToolReturnPart | RetryPromptPart, step_index: int
    ) -> None:
        interrupted = await _await_request_handoff(
            self._record_tool_result_boundary(part, step_index)
        )
        if interrupted:
            raise asyncio.CancelledError

    def _same_tool_call(self, previous: ToolCallPart, current: ToolCallPart) -> bool:
        return previous.tool_name == current.tool_name and previous.args_as_dict() == current.args_as_dict()

    def _same_tool_result(
        self,
        previous: ToolReturnPart | RetryPromptPart,
        current: ToolReturnPart | RetryPromptPart,
    ) -> bool:
        return type(previous) is type(current) and previous == replace(
            current, timestamp=previous.timestamp,
        )

    async def _record_tool_result_boundary(
        self, part: ToolReturnPart | RetryPromptPart, step_index: int
    ) -> None:
        previous = self._tool_results.get(part.tool_call_id)
        recorded = self._tool_events.get(part.tool_call_id, set())
        if previous is not None:
            if not self._same_tool_result(previous, part):
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            if recorded.intersection({"TOOL_CALL_SUCCEEDED", "TOOL_CALL_FAILED"}):
                return
        else:
            self.stage_tool_result(part)
        model_request_seq = self.model_request_seq_for_tool_call(part.tool_call_id)
        metadata = (
            {}
            if model_request_seq is None
            else {MODEL_REQUEST_SEQ_METADATA_KEY: str(model_request_seq)}
        )
        execution_metadata, execution_error = self._tool_execution_metadata.pop(
            part.tool_call_id, ({}, None)
        )
        metadata.update(execution_metadata)
        await self.record_tool_start(part, step_index)
        if not recorded.intersection({"TOOL_CALL_SUCCEEDED", "TOOL_CALL_FAILED"}):
            succeeded = isinstance(part, ToolReturnPart) and part.outcome == "success"
            await self.record_event(
                "TOOL_CALL_SUCCEEDED" if succeeded else "TOOL_CALL_FAILED",
                step_index,
                tool_call_id=part.tool_call_id,
                tool_name=part.tool_name,
                metadata=metadata,
                error=execution_error if not succeeded else None,
            )
        await self.commit_history_boundary()
        if previous is None:
            self._tool_results[part.tool_call_id] = freeze_model_messages((
                ModelRequest(parts=[part]),
            ))[0].parts[0]

    def _stage_transcript(self) -> None:
        pending = None
        keys = tuple(self._pending_parts)
        if keys:
            first = self._pending_parts[keys[0]]
            parts = [part for key in keys for part in self._pending_parts[key].parts]
            pending = (
                ModelResponse(parts=parts)
                if isinstance(first, ModelResponse)
                else ModelRequest(parts=parts)
            )
        self._staging_store.stage_transcript(
            self._agent_run_id,
            StagedTranscript(tuple(self._transcript_messages), pending, keys),
        )

    def finish_transcript(self, *, interrupted: bool = False) -> None:
        """Retain confirmed parts when no next request can capture their message."""
        if self._pending_parts:
            first = next(iter(self._pending_parts.values()))
            parts = [
                part
                for message in self._pending_parts.values()
                for part in message.parts
            ]
            message = (
                ModelResponse(parts=parts)
                if isinstance(first, ModelResponse)
                else ModelRequest(parts=parts)
            )
            if interrupted:
                message.state = "interrupted"
            restored = self._restored_pending
            self.append_transcript_message(message)
            if restored:
                self._restored_tool_results.update(
                    part.tool_call_id for part in parts
                    if isinstance(part, (ToolReturnPart, RetryPromptPart)) and part.tool_call_id
                )

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

    def checkpoint_context(
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

    def _source_refs_for(
        self,
        messages: Sequence[ModelMessage],
    ) -> tuple[
        tuple[TranscriptMessageRef | int | None, ...],
        tuple[bytes, ...],
    ]:
        values = freeze_model_messages(messages)
        requested_keys = tuple(
            encode_model_messages((message,))
            for message in values
        )
        refs: list[TranscriptMessageRef | int | None] = []
        for key in requested_keys:
            index = self._source_indices.get(key)
            refs.append(None if index is None else self._source_refs[index])
        return tuple(refs), requested_keys

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
        self._stage_model_interaction(
            fact, model, messages, model_settings, parameters, streaming,
            model_id, source_messages, prepared=False,
        )

    def prepare_model_interaction(
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
        self._stage_model_interaction(
            fact, model, messages, model_settings, parameters, streaming,
            model_id, source_messages, prepared=True,
        )

    def _stage_model_interaction(
        self,
        fact: ModelRequestFact,
        model: Model,
        messages: Sequence[ModelMessage],
        model_settings: ModelSettings | None,
        parameters: ModelRequestParameters,
        streaming: bool,
        model_id: str | None,
        source_messages: Sequence[ModelMessage] | None,
        *,
        prepared: bool,
    ) -> None:
        frozen = freeze_model_messages(messages)
        projection = None
        digest = None
        if prepared:
            source = frozen if source_messages is None else freeze_model_messages(source_messages)
            source_refs, source_keys = self._source_refs_for(source)
            projection = build_context_projection(
                source,
                frozen,
                lambda payload: self._staging_store.intern_payload(
                    self._agent_run_id,
                    payload,
                ),
                source_refs=source_refs,
                source_keys=source_keys,
            )
            _envelope, envelope_bytes = request_envelope(
                model_settings=model_settings,
                parameters=parameters,
                streaming=streaming,
            )
            digest, _size = self._staging_store.intern_payload(
                self._agent_run_id,
                envelope_bytes,
            )
        model_value = model_identity(
            model,
            route_id=model_id,
        )
        attachments = request_attachment_facts(
            frozen,
            self._initial_attachments,
            accepted_attachment_ids=self._accepted_attachment_ids,
        )
        # Acceptance survives provider preparation even when the final input
        # drops an attachment. Inclusion follows the current prepared request.
        accepted = tuple(
            value
            for value in self._interaction_attachments.get(fact.model_request_seq, ())
            if value["fact"] == "accepted"
        )
        attachments = (*accepted, *attachments)
        if not prepared:
            attachments = tuple(value for value in attachments if value["fact"] == "accepted")
        stage = (
            self._staging_store.prepare_model_interaction
            if prepared
            else self._staging_store.stage_model_interaction
        )
        stage(
            StagedModelInteraction(
                agent_run_id=self._agent_run_id,
                step_index=fact.step_index,
                model_request_seq=fact.model_request_seq,
                purpose=fact.purpose,
                output_retry_index=fact.output_retry_index,
                model=model_value,
                request_context=projection,
                request_envelope_digest=digest,
                response_context=None,
                status="RUNNING",
                error_code=None,
                duration_ns=None,
                usage=None,
                attachments=attachments,
                started_at=fact.started_at,
                finished_at=None,
            )
        )
        self._interaction_projections[fact.model_request_seq] = projection
        self._interaction_payloads[fact.model_request_seq] = digest
        self._interaction_models[fact.model_request_seq] = model_value
        self._interaction_attachments[fact.model_request_seq] = attachments

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
        model_request_seq = fact.model_request_seq
        try:
            projection = self._interaction_projections.pop(model_request_seq)
            envelope_digest = self._interaction_payloads.pop(model_request_seq)
            model_value = self._interaction_models.pop(model_request_seq)
            attachments = self._interaction_attachments.pop(model_request_seq)
        except KeyError as error:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error

        response_projection = None
        if status in {"FAILED", "CANCELLED"} and fact.purpose == "agent":
            if response is None and self._pending_parts:
                first = next(iter(self._pending_parts.values()))
                if isinstance(first, ModelResponse):
                    response = replace(first, parts=[
                        part for message in self._pending_parts.values() for part in message.parts
                    ])
            if response is not None:
                response = replace(response, state="interrupted")
                self.append_transcript_message(response)
        if response is not None:
            frozen_response = freeze_model_messages((response,))
            response_projection = build_inline_context_projection(
                frozen_response,
                lambda payload: self._staging_store.intern_payload(
                    self._agent_run_id,
                    payload,
                ),
            )
        self._staging_store.stage_model_interaction(
            StagedModelInteraction(
                agent_run_id=self._agent_run_id,
                step_index=fact.step_index,
                model_request_seq=model_request_seq,
                purpose=fact.purpose,
                output_retry_index=fact.output_retry_index,
                model=model_value,
                request_context=projection,
                request_envelope_digest=envelope_digest,
                response_context=response_projection,
                status=status,
                error_code=error_code,
                duration_ns=duration_ns,
                usage=_usage_metrics(usage),
                attachments=attachments,
                started_at=fact.started_at,
                finished_at=fact.finished_at,
            )
        )

    def model_request_seq_for_tool_call(self, tool_call_id: str) -> int | None:
        return self._model_request_seq_by_tool_call.get(tool_call_id)

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
        event_types = {
            "started": "MODEL_REQUEST_STARTED",
            "completed": "MODEL_REQUEST_SUCCEEDED",
            "failed": "MODEL_REQUEST_FAILED",
            "cancelled": "MODEL_REQUEST_CANCELLED",
        }
        event_type = event_types.get(phase)
        if event_type is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        replayed_indices: list[int] = []
        if (
            fact.purpose == "agent"
            and phase == "completed"
            and response is not None
        ):
            seen_call_ids: set[str] = set()
            for part_index, part in enumerate(response.parts):
                if not isinstance(part, ToolCallPart):
                    continue
                call_id = part.tool_call_id
                if not isinstance(call_id, str) or not call_id or call_id in seen_call_ids:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                seen_call_ids.add(call_id)
                if call_id in self._model_request_seq_by_tool_call:
                    previous_call = self._restored_tool_calls.get(call_id)
                    if previous_call is None or not self._same_tool_call(previous_call, part):
                        raise AIError(ErrorCode.STORAGE_CONFLICT)
                    replayed_indices.append(part_index)
                else:
                    self._model_request_seq_by_tool_call[call_id] = fact.model_request_seq
        metadata = fact.metadata(
            include_observation=include_observation and fact.duration_ns is not None
        )
        if replayed_indices:
            metadata[REPLAYED_TOOL_CALL_INDICES_METADATA_KEY] = json.dumps(replayed_indices)
        if fact.purpose == "agent" and phase == "completed" and response is not None:
            self.append_transcript_message(response)
            metadata[MESSAGE_SEQ_METADATA_KEY] = str(len(self._transcript_messages))
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
        if phase == "started":
            timestamp = fact.started_at
        elif fact.finished_at is not None:
            timestamp = fact.finished_at
        else:
            timestamp = None
        await self.record_event(
            cast(StepEventType, event_type),
            fact.step_index,
            error=error_code,
            metadata=metadata,
            timestamp=timestamp,
        )
        await self.commit_history_boundary()


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


__all__ = ["AgentRunRecorder"]
