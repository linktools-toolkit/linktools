#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""User-facing tool progress without tool inputs, results, or diagnostics."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal, cast

from ..core import ExecutionEventType, JsonValue

ToolProcessStatus = Literal["RUNNING", "SUCCEEDED", "FAILED", "INTERRUPTED", "CANCELLED", "UNCONFIRMED"]


@dataclass(frozen=True, slots=True)
class ToolProcessItem:
    key: str
    tool_name: str
    status: ToolProcessStatus
    position: tuple[int, int] | None = None
    elapsed_ms: int | None = None
    kind: Literal["tool"] = field(default="tool", init=False)


@dataclass(frozen=True, slots=True)
class SessionTurnNotice:
    """Best-effort turn producer notice; never an authorization or state source."""

    kind: Literal["admitted", "tool", "terminal"]
    session_id: str
    execution_id: str
    process: ToolProcessItem | None = None


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


__all__ = ["SessionTurnNotice", "ToolProcessItem", "ToolProcessStatus", "project_tool_process_event"]
