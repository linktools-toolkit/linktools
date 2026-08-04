#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Map protocol-neutral execution events to ACP session updates."""

from typing import Any

from ..execution.live_events import (
    AssistantTextDelta,
    AssistantThoughtDelta,
    ExecutionEvent,
    PlanUpdated,
    ToolCallCompleted,
    ToolCallFailed,
    ToolCallProgress,
    ToolCallStarted,
    UsageUpdated,
)


class AcpEventMapper:
    def map(self, event: ExecutionEvent) -> "Any | None":
        import acp.schema as schema

        if isinstance(event, AssistantTextDelta):
            return schema.AgentMessageChunk(
                content=schema.TextContentBlock(type="text", text=event.text),
                messageId=event.execution_id,
                sessionUpdate="agent_message_chunk",
            )
        if isinstance(event, AssistantThoughtDelta):
            return schema.AgentThoughtChunk(
                content=schema.TextContentBlock(type="text", text=event.text),
                messageId=event.execution_id,
                sessionUpdate="agent_thought_chunk",
            )
        if isinstance(event, ToolCallStarted):
            return schema.ToolCallStart(
                toolCallId=event.tool_call_id,
                title=event.tool_name,
                status="pending",
                rawInput=event.arguments,
                sessionUpdate="tool_call",
            )
        if isinstance(event, (ToolCallProgress, ToolCallCompleted, ToolCallFailed)):
            status = event.status
            raw_output = (
                event.error
                if isinstance(event, ToolCallFailed)
                else getattr(event, "result", None)
            )
            return schema.ToolCallProgress(
                toolCallId=event.tool_call_id,
                title=event.tool_name,
                status=status,
                rawInput=event.arguments,
                rawOutput=raw_output,
                sessionUpdate="tool_call_update",
            )
        if isinstance(event, PlanUpdated):
            return schema.AgentPlanUpdate(entries=list(event.entries), sessionUpdate="plan")
        if isinstance(event, UsageUpdated):
            used = getattr(event, "context_tokens_used", None)
            size = getattr(event, "context_window_size", None)
            if (
                not isinstance(used, int)
                or isinstance(used, bool)
                or not isinstance(size, int)
                or isinstance(size, bool)
                or used < 0
                or size < used
            ):
                return None
            return schema.UsageUpdate(
                used=used,
                size=size,
                sessionUpdate="usage_update",
            )
        return None


__all__ = ["AcpEventMapper"]
