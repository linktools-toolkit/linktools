#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Per-run context candidates and provenance shared by persistence owners."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from pydantic_ai.messages import ModelMessage
from pydantic_ai.tools import RunContext as PydanticRunContext

from ..errors import AIError, ErrorCode
from ._message import encode_model_messages, freeze_model_messages, project_transient_binary_content
from .state._contracts import LoadedModelContext, TranscriptMessageRef
from .state._plan import RuntimeDomain
from .state._step_contracts import AgentRunCheckpoint, CheckpointState

if TYPE_CHECKING:
    from ._agent_run_recorder import AgentRunRecorder


@dataclass
class RunContextCapture:
    """Build one immutable context candidate per distinct run frontier."""

    recorder: AgentRunRecorder
    agent_run_id: str
    agent_id: str
    parent_agent_run_id: str | None
    _last_checkpoint_transcript_count: int = field(default=0, init=False)
    _last_checkpoint_context: tuple[ModelMessage, ...] | None = field(default=None, init=False)
    _last_checkpoint_state: CheckpointState | None = field(default=None, init=False)
    _last_checkpoint_pending_index: int | None = field(default=None, init=False)

    def start(self) -> None:
        self._last_checkpoint_transcript_count = len(self.recorder.transcript_messages())

    async def save_checkpoint(
        self,
        ctx: PydanticRunContext[None],
        *,
        messages: Sequence[ModelMessage],
        pending: ModelMessage | None = None,
        state: CheckpointState,
    ) -> None:
        context_messages, pending_index = self.recorder.checkpoint_context(
            messages,
            pending=pending,
        )
        frozen_context = tuple(context_messages)
        raw = self.recorder.transcript_messages()
        if (
            self._last_checkpoint_context == frozen_context
            and self._last_checkpoint_transcript_count == len(raw)
            and self._last_checkpoint_state == state
            and self._last_checkpoint_pending_index == pending_index
        ):
            return
        await self.recorder.save_checkpoint(
            AgentRunCheckpoint(
                agent_run_id=self.agent_run_id,
                step_index=ctx.run_step,
                messages=list(raw),
                agent_conversation_id=ctx.conversation_id,
                parent_agent_run_id=self.parent_agent_run_id,
                agent_id=self.agent_id,
                state=state,
                context_messages=context_messages,
                transcript_message_count_before=self._last_checkpoint_transcript_count,
                pending_request_index=pending_index,
            )
        )
        self._last_checkpoint_transcript_count = len(raw)
        self._last_checkpoint_context = frozen_context
        self._last_checkpoint_state = state
        self._last_checkpoint_pending_index = pending_index


class RunContextProjection:
    """Preserve source provenance while projecting the model's live context."""

    def __init__(
        self, agent_run_id: str, initial_messages: Sequence[ModelMessage],
        initial_context: LoadedModelContext | None,
    ) -> None:
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
                and value.source.source_domain is RuntimeDomain.CHECKPOINT
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
        self._projection_source_count: int | None = None
        self._projection_messages: tuple[ModelMessage, ...] | None = None

    def append_source(self, message: ModelMessage, message_index: int) -> None:
        key = encode_model_messages((message,))
        self._source_indices[key] = None if key in self._source_indices else len(self._source_keys)
        self._source_keys.append(key)
        self._source_refs.append(message_index)

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

    def source_refs_for(
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


__all__ = ["RunContextCapture", "RunContextProjection"]
