#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared per-run capture for independently bound native persistence features."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from pydantic_ai import ModelRequestNode
from pydantic_ai.capabilities import AbstractCapability, AgentNode, CapabilityOrdering, NodeResult
from pydantic_ai.messages import AgentStreamEvent, ModelMessage, ModelRequest, OutputToolResultEvent, RetryFeedbackPart
from pydantic_ai.models import ModelRequestContext
from pydantic_ai.run import AgentRunResult
from pydantic_ai.tools import DeferredToolRequests
from pydantic_ai.tools import RunContext as PydanticRunContext

from ..errors import AIError, ErrorCode
from ..spec import CapabilityRef
from ._agent_run_recorder import AgentRunRecorder
from ._run_context import RunContextCapture
from ..persistence.runs import AgentRunRecord


@dataclass(kw_only=True, eq=False)
class AgentRunCapture(AbstractCapability[None]):
    """Share raw run capture and safety facts between independent native features."""

    recorder: AgentRunRecorder = field(repr=False, compare=False)
    agent_id: str
    agent_run_id: str
    parent_agent_run_id: str | None = None
    initial_output_retries_used: int = 0
    metadata: dict[str, str] = field(default_factory=dict)
    id: str | None = field(
        default="linktools.ai.run-capture",
        init=False,
        repr=False,
        compare=False,
    )
    producer_guard: Callable[[], Awaitable[None]] | None = field(
        default=None, repr=False, compare=False,
    )
    deferred_pause_sink: Callable[[int], None] | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    deferred_capability_refs: dict[str, CapabilityRef] = field(default_factory=dict, repr=False, compare=False)
    _registered: bool = field(default=False, init=False)
    _finished: bool = field(default=False, init=False)
    _captured_request: ModelMessage | None = field(default=None, init=False)
    _last_observed_step_index: int | None = field(
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

    context: RunContextCapture = field(init=False, repr=False)

    @property
    def live_messages(self) -> Sequence[ModelMessage] | None:
        return self._live_messages

    def __post_init__(self) -> None:
        self.context = RunContextCapture(
            self.recorder, self.agent_run_id, self.agent_id, self.parent_agent_run_id,
            output_retries_used=self.initial_output_retries_used,
        )
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
        self.context.start()
        self._replay_request_captured = self.recorder.request_already_captured(ctx.messages)
        self._registered = True

    async def before_model_request(
        self,
        ctx: PydanticRunContext[None],
        request_context: ModelRequestContext,
    ) -> ModelRequestContext:
        self.context.output_retries_used = self.initial_output_retries_used + ctx.retry
        if self.producer_guard is not None:
            await self.producer_guard()
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

    async def on_event(self, ctx: PydanticRunContext[None], *, event: AgentStreamEvent) -> None:
        if isinstance(event, OutputToolResultEvent) and event.part.outcome == "retried":
            # The SDK emits settled output retries before a co-emitted approval
            # can end the run. Run/node lifecycle contexts do not refresh retry.
            self.context.output_retries_used += 1

    async def after_node_run(
        self,
        ctx: PydanticRunContext[None],
        *,
        node: AgentNode[None],
        result: NodeResult[None],
    ) -> NodeResult[None]:
        self._live_messages = ctx.messages
        self._last_observed_step_index = ctx.run_step
        if isinstance(result, ModelRequestNode):
            for part in result.request.parts:
                if isinstance(part, RetryFeedbackPart) and part.retry.attempt is not None:
                    self.context.output_retries_used = self.initial_output_retries_used + part.retry.attempt
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
            pending_ids = {call.tool_call_id for call in (*result.output.approvals, *result.output.calls)}
            for call_id in tuple(self.deferred_capability_refs):
                if call_id not in pending_ids:
                    del self.deferred_capability_refs[call_id]
            self.deferred_pause_sink(self._last_observed_step_index)
            await self.context.save_checkpoint(ctx, messages=result.all_messages(), state="interrupted")
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


    def remember_context_projection(
        self,
        source: Sequence[ModelMessage],
        projected: Sequence[ModelMessage] | None,
    ) -> None:
        self.recorder.remember_context_projection(source, projected)


__all__ = ["AgentRunCapture"]
