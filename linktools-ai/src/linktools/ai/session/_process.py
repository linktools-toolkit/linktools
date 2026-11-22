#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""User-facing tool progress without tool inputs, results, or diagnostics."""

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Literal, cast

from pydantic_ai.messages import (
    SYNTHESIZED_TOOL_RETURN_METADATA_KEY,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ToolCallPart,
    ToolReturnPart,
)

from ..core import ExecutionEventType, JsonValue

ToolProcessStatus = Literal["RUNNING", "SUCCEEDED", "FAILED", "INTERRUPTED", "CANCELLED", "UNCONFIRMED"]


@dataclass(frozen=True, slots=True)
class ToolProcessItem:
    key: str
    tool_name: str
    status: ToolProcessStatus
    position: tuple[int, int] | None = None
    kind: Literal["tool"] = field(default="tool", init=False)


def project_tool_process_event(event_type: str, payload: JsonValue) -> ToolProcessItem | None:
    """Project a tool lifecycle event after the caller authorizes its observation."""
    if not isinstance(payload, Mapping):
        return None
    call_id, name = payload.get("call_id"), payload.get("tool_name")
    if not isinstance(call_id, str) or not call_id or not isinstance(name, str) or not name:
        return None
    status: ToolProcessStatus
    if event_type == ExecutionEventType.TOOL_CALL_STARTED.value:
        status = "RUNNING"
    elif event_type == ExecutionEventType.TOOL_CALL_FINISHED.value:
        value = payload.get("status")
        if value not in ("SUCCEEDED", "FAILED", "INTERRUPTED", "CANCELLED"):
            return None
        status = cast(ToolProcessStatus, value)
    else:
        return None
    position = None
    raw_position = payload.get("tool_position")
    if isinstance(raw_position, (list, tuple)) and len(raw_position) == 2:
        message_seq, part_index = raw_position
        if type(message_seq) is int and message_seq > 0 and type(part_index) is int and part_index >= 0:
            position = (message_seq, part_index)
    return ToolProcessItem(f"tool:{call_id}", name, status, position)


def project_turn_tool_process(messages: tuple[ModelMessage, ...]) -> tuple[ToolProcessItem, ...]:
    """Summarize root transcript tool calls in occurrence order, without inferring outcomes."""
    calls: dict[str, ToolProcessItem] = {}
    for message_seq, message in enumerate(messages, start=1):
        if isinstance(message, ModelResponse):
            for part_index, part in enumerate(message.parts):
                if isinstance(part, ToolCallPart):
                    calls.setdefault(
                        part.tool_call_id,
                        ToolProcessItem(
                            f"tool:{part.tool_call_id}", part.tool_name, "UNCONFIRMED", (message_seq, part_index),
                        ),
                    )
        elif isinstance(message, ModelRequest):
            for part in message.parts:
                if not isinstance(part, ToolReturnPart):
                    continue
                call = calls.get(part.tool_call_id)
                if call is None or part.tool_name != call.tool_name:
                    continue
                status: ToolProcessStatus
                if part.outcome == "success":
                    status = "SUCCEEDED"
                elif part.outcome in ("failed", "denied", "retried"):
                    status = "FAILED"
                elif part.outcome == "interrupted" and not (
                    isinstance(part.metadata, Mapping)
                    and part.metadata.get(SYNTHESIZED_TOOL_RETURN_METADATA_KEY) is True
                ):
                    status = "INTERRUPTED"
                else:
                    continue
                calls[part.tool_call_id] = replace(call, status=status)
    return tuple(calls.values())


__all__ = ["ToolProcessItem", "ToolProcessStatus", "project_tool_process_event", "project_turn_tool_process"]
