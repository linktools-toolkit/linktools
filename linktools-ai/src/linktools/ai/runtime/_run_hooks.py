#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared per-run capture for independently bound native persistence features."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from time import monotonic_ns
from typing import Any

from pydantic_ai import CallToolsNode, ModelRequestNode
from pydantic_ai.capabilities import AbstractCapability, AgentNode, CapabilityOrdering, NodeResult
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, ToolCallPart
from pydantic_ai.models import ModelRequestContext
from pydantic_ai.run import AgentRunResult
from pydantic_ai.tools import DeferredToolRequests, ToolDefinition
from pydantic_ai.tools import RunContext as PydanticRunContext

from ..errors import AIError, ErrorCode
from ._agent_run_recorder import AgentRunRecorder
from ._journal import DURATION_NS_METADATA_KEY, MODEL_REQUEST_SEQ_METADATA_KEY, await_request_handoff
from .state._step_contracts import TOOL_ERROR_CODE_METADATA_KEY, AgentRunCheckpoint, AgentRunRecord, CheckpointState


@dataclass(kw_only=True, eq=False)
class AgentRunCapture(AbstractCapability[None]):
    """Share raw run capture and safety facts between independent native features."""

    recorder: AgentRunRecorder = field(repr=False, compare=False)
    agent_id: str
    agent_run_id: str
    parent_agent_run_id: str | None = None
    metadata: dict[str, str] = field(default_factory=dict)
    id: str | None = field(
        default="linktools.ai.run-capture",
        init=False,
        repr=False,
        compare=False,
    )
    deferred_pause_sink: Callable[[int], None] | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    _registered: bool = field(default=False, init=False)
    _finished: bool = field(default=False, init=False)
    _captured_request: ModelMessage | None = field(default=None, init=False)
    _last_observed_step_index: int | None = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )
    _last_checkpoint_transcript_count: int = field(
        default=0,
        init=False,
        repr=False,
        compare=False,
    )
    _last_checkpoint_context: tuple[ModelMessage, ...] | None = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )
    _last_checkpoint_state: CheckpointState | None = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )
    _last_checkpoint_pending_index: int | None = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )
    _replay_request_captured: bool = field(default=False, init=False, repr=False, compare=False)
    _live_messages: Sequence[ModelMessage] | None = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )
    _tool_started_ns: dict[str, int] = field(
        default_factory=dict,
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if not isinstance(self.recorder, AgentRunRecorder):
            raise TypeError("recorder must be AgentRunRecorder")
        if not self.agent_run_id or self.recorder.agent_run_id != self.agent_run_id:
            raise ValueError("Runtime AgentRun identity is invalid")

    def get_ordering(self) -> CapabilityOrdering:
        return CapabilityOrdering(position="innermost")

    async def before_run(self, ctx: PydanticRunContext[None]) -> None:
        if self._registered:
            return
        self._live_messages = ctx.messages
        await self.recorder.register_agent_run(
            AgentRunRecord(
                agent_run_id=self.agent_run_id,
                agent_conversation_id=ctx.conversation_id,
                parent_agent_run_id=self.parent_agent_run_id,
                agent_id=self.agent_id,
                metadata=dict(self.metadata),
                started_at=datetime.now(timezone.utc),
            )
        )
        transcript = self.recorder.transcript_messages()
        self._last_checkpoint_transcript_count = len(transcript)
        self._replay_request_captured = self.recorder.request_already_captured(ctx.messages)
        self._registered = True

    async def before_model_request(
        self,
        ctx: PydanticRunContext[None],
        request_context: ModelRequestContext,
    ) -> ModelRequestContext:
        await self.before_run(ctx)
        self._live_messages = ctx.messages
        if (ctx.messages and isinstance(ctx.messages[-1], ModelRequest)
                and self._captured_request is not ctx.messages[-1]):
            self._captured_request = ctx.messages[-1]
            if self._replay_request_captured:
                # A resumed outstanding request already has its raw occurrence.
                self._replay_request_captured = False
            else:
                self.recorder.append_transcript_message(ctx.messages[-1])
        return request_context

    async def after_node_run(
        self,
        ctx: PydanticRunContext[None],
        *,
        node: AgentNode[None],
        result: NodeResult[None],
    ) -> NodeResult[None]:
        self._live_messages = ctx.messages
        self._last_observed_step_index = ctx.run_step
        return result

    async def after_run(
        self,
        ctx: PydanticRunContext[None],
        *,
        result: AgentRunResult[Any],
    ) -> AgentRunResult[Any]:
        if self._finished:
            return result
        self._live_messages = result.all_messages()
        self.recorder.finish_transcript()
        interrupted = isinstance(result.output, DeferredToolRequests)
        if interrupted:
            if self._last_observed_step_index is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if self.deferred_pause_sink is None:
                raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
            self.deferred_pause_sink(self._last_observed_step_index)
            await self.save_checkpoint(ctx, messages=result.all_messages(), state="interrupted")
        self._finished = True
        return result

    async def on_run_error(
        self,
        ctx: PydanticRunContext[None],
        *,
        error: BaseException,
    ) -> AgentRunResult[Any]:
        self.recorder.finish_transcript(interrupted=True)
        raise error

    async def capture_final_context(
        self, ctx: PydanticRunContext[None], *, result: AgentRunResult[Any],
    ) -> None:
        await self.after_run(ctx, result=result)
        await self.save_checkpoint(
            ctx, messages=result.all_messages(),
            state="interrupted" if isinstance(result.output, DeferredToolRequests) else "complete",
        )

    async def capture_interrupted_context(self, ctx: PydanticRunContext[None]) -> None:
        self.recorder.finish_transcript(interrupted=True)
        await self.save_checkpoint(
            ctx, messages=self._live_messages or ctx.messages, state="interrupted",
        )

    async def checkpoint_after_node(
        self, ctx: PydanticRunContext[None], *, node: AgentNode[None], result: NodeResult[None],
    ) -> NodeResult[None]:
        if isinstance(node, ModelRequestNode):
            response = None
            if isinstance(result, CallToolsNode):
                response = result.model_response
            elif ctx.messages and isinstance(ctx.messages[-1], ModelResponse):
                response = ctx.messages[-1]
            if response is not None:
                # The recorder captured this response with its request identity.
                # It must be recoverable before any tool effect.
                await self.save_checkpoint(ctx, messages=ctx.messages, state="complete")
        if isinstance(node, CallToolsNode):
            pending = result.request if isinstance(result, ModelRequestNode) else None
            await self.save_checkpoint(
                ctx,
                messages=ctx.messages,
                pending=pending,
                state="complete",
            )
        return result

    def _tool_request_metadata(self, tool_call_id: str) -> dict[str, str]:
        sequence = self.recorder.model_request_seq_for_tool_call(tool_call_id)
        return (
            {}
            if sequence is None
            else {MODEL_REQUEST_SEQ_METADATA_KEY: str(sequence)}
        )

    async def record_tool_start(
        self,
        ctx: PydanticRunContext[None],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: dict[str, Any],
    ) -> dict[str, Any]:
        if call.tool_call_id in self._tool_started_ns:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        self._tool_started_ns[call.tool_call_id] = monotonic_ns()
        await self.recorder.record_tool_start(call, ctx.run_step)
        return args

    async def record_tool_result(
        self,
        ctx: PydanticRunContext[None],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: dict[str, Any],
        result: Any,
    ) -> Any:
        del args
        started_ns = self._tool_started_ns.pop(call.tool_call_id, None)
        if started_ns is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        metadata = self._tool_request_metadata(call.tool_call_id)
        metadata[DURATION_NS_METADATA_KEY] = str(max(0, monotonic_ns() - started_ns))
        self.recorder.remember_tool_execution(
            call.tool_call_id,
            metadata=metadata,
        )
        return result

    async def record_tool_error(
        self,
        ctx: PydanticRunContext[None],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: dict[str, Any],
        error: Exception,
    ) -> Any:
        del args
        started_ns = self._tool_started_ns.pop(call.tool_call_id, None)
        if started_ns is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
        metadata = self._tool_request_metadata(call.tool_call_id)
        metadata[DURATION_NS_METADATA_KEY] = str(max(0, monotonic_ns() - started_ns))
        if isinstance(error, AIError):
            metadata[TOOL_ERROR_CODE_METADATA_KEY] = error.code.value
        async def record_failure() -> None:
            await self.recorder.record_event(
                "TOOL_CALL_FAILED",
                ctx.run_step,
                tool_call_id=call.tool_call_id,
                tool_name=tool_def.name,
                error=repr(error),
                metadata=metadata,
            )
            await self.recorder.commit_history_boundary()

        interrupted = await await_request_handoff(record_failure())
        if interrupted:
            raise asyncio.CancelledError from error
        raise error

    def remember_context_projection(
        self,
        source: Sequence[ModelMessage],
        projected: Sequence[ModelMessage] | None,
    ) -> None:
        self.recorder.remember_context_projection(source, projected)

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


__all__ = ["AgentRunCapture"]
