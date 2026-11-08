#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Immutable invocation identity projected from admitted and native contexts."""

from dataclasses import dataclass

from pydantic_ai.tools import RunContext

from ..core import Principal
from ..errors import AIError, ErrorCode
from ._context import AgentContext
from ._persistence import TraceCapability


@dataclass(frozen=True, slots=True)
class ToolCallIdentity:
    namespace: str
    principal: Principal
    execution_id: str
    root_execution_id: str | None
    parent_execution_id: str | None
    session_id: str | None
    agent_id: str | None
    agent_run_id: str
    agent_run_seq: int | None
    model_request_seq: int | None
    tool_call_id: str
    tool_name: str
    retry: int

    @classmethod
    def from_context(cls, ctx: RunContext[AgentContext[object]]) -> "ToolCallIdentity":
        """Read current call identity without history I/O or shared call state.

        Request sequence is absent when no Trace capture proves its provenance.
        Native run_step and retry never substitute for that sequence.
        """
        if ctx.run_id is None or ctx.tool_call_id is None or ctx.tool_name is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        trace = ctx.capabilities.get(TraceCapability.id)
        sequence = (
            trace.model_request_seq_for_tool_call(ctx.tool_call_id)
            if isinstance(trace, TraceCapability) else None
        )
        return cls(
            namespace=ctx.deps.namespace,
            principal=ctx.deps.principal,
            execution_id=ctx.deps.execution_id,
            root_execution_id=ctx.deps.root_execution_id,
            parent_execution_id=ctx.deps.parent_execution_id,
            session_id=ctx.deps.session_id,
            agent_id=ctx.deps.agent_id,
            agent_run_id=ctx.run_id,
            agent_run_seq=ctx.deps.agent_run_seq,
            model_request_seq=sequence,
            tool_call_id=ctx.tool_call_id,
            tool_name=ctx.tool_name,
            retry=ctx.retry,
        )


__all__ = ["ToolCallIdentity"]
