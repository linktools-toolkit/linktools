#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""User-facing tool progress without raw inputs, results, or diagnostics."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal, cast

from ..core import ExecutionEventType, JsonValue
from ..errors import ErrorCode

ToolProcessStatus = Literal["RUNNING", "SUCCEEDED", "FAILED", "INTERRUPTED", "CANCELLED", "UNCONFIRMED"]


@dataclass(frozen=True, slots=True)
class ToolProcessItem:
    key: str
    tool_name: str
    status: ToolProcessStatus
    position: tuple[int, int] | None = None
    elapsed_ms: int | None = None
    error_code: ErrorCode | None = None
    argument_preview: str | None = None
    kind: Literal["tool"] = field(default="tool", init=False)


    @classmethod
    def from_notice(cls, row: Mapping[str, object]) -> "ToolProcessItem | None":
        key, name, status = row.get("key"), row.get("tool_name"), row.get("status")
        if (row.get("kind") != "tool" or not isinstance(key, str) or not key.startswith("tool:")
                or key == "tool:" or not isinstance(name, str) or not name
                or status not in ("RUNNING", "SUCCEEDED", "FAILED", "INTERRUPTED", "CANCELLED")):
            return None
        position = None
        raw_position = row.get("position")
        if raw_position is not None:
            if not isinstance(raw_position, (list, tuple)) or len(raw_position) != 2:
                return None
            message_seq, part_index = raw_position
            if type(message_seq) is not int or message_seq < 1 or type(part_index) is not int or part_index < 0:
                return None
            position = (message_seq, part_index)
        elapsed = row.get("elapsed_ms")
        preview = row.get("argument_preview")
        code = row.get("error_code") if status == "FAILED" else None
        try:
            reason = ErrorCode(code) if isinstance(code, str) else None
        except ValueError:
            reason = None
        return cls(
            key, name, cast(ToolProcessStatus, status), position,
            elapsed if status != "RUNNING" and type(elapsed) is int and elapsed >= 0 else None,
            reason, preview if isinstance(preview, str) else None,
        )


@dataclass(frozen=True, slots=True)
class SessionTurnNotice:
    """Best-effort turn producer notice; never an authorization or state source."""

    kind: Literal["admitted", "tool", "terminal"]
    session_id: str
    execution_id: str
    process: ToolProcessItem | None = None


def tool_argument_preview(path: object) -> str | None:
    """Format one optional, registered, public scalar without exposing raw arguments."""
    if not isinstance(path, str) or not 0 < len(path) <= 80 or path != path.strip():
        return None
    if path == ".":
        return path
    if any(fragment in {"", ".", ".."} for fragment in path.split("/")):
        return None
    if any(not (character.isalnum() or character in "/._- ") for character in path):
        return None
    return path


def project_tool_process_event(event_type: str, payload: JsonValue) -> ToolProcessItem | None:
    """Project a tool lifecycle event after the caller authorizes its observation."""
    if not isinstance(payload, Mapping):
        return None
    call_id = payload.get("call_id")
    if not isinstance(call_id, str) or not call_id:
        return None
    if event_type == ExecutionEventType.TOOL_CALL_STARTED.value:
        status = "RUNNING"
    elif event_type == ExecutionEventType.TOOL_CALL_FINISHED.value:
        status = payload.get("status")
    else:
        return None
    return ToolProcessItem.from_notice({
        "kind": "tool", "key": f"tool:{call_id}", "tool_name": payload.get("tool_name"),
        "status": status, "position": payload.get("tool_position"), "elapsed_ms": payload.get("tool_elapsed_ms"),
        "error_code": payload.get("error_code"), "argument_preview": payload.get("argument_preview"),
    })



__all__ = ["SessionTurnNotice", "ToolProcessItem", "ToolProcessStatus", "project_tool_process_event", "tool_argument_preview"]
