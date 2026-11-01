#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Terminal rendering and interruption cleanup for AI executions."""

import asyncio
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from rich import get_console
from rich.console import Console
from rich.text import Text

from linktools.cli import CommandError
from linktools.core import environ
from linktools.ai.core import ExecutionDeltaType, ExecutionEventType, ExecutionStatus

if TYPE_CHECKING:
    from linktools.ai.execution import Execution, ExecutionResult, ExecutionTreeEvent
    from linktools.ai.runtime import Runtime


async def emit_result(
    runtime: "Runtime",
    prompt: str,
    session_id: str,
    as_json: bool,
    planning: bool,
    thinking: bool | None,
) -> int:
    """Wait or stream an execution, cancelling owned work on interruption."""
    execution = await runtime.agents.get().start(
        prompt,
        session_id=session_id,
        planning=planning,
        thinking=thinking,
    )
    try:
        if as_json:
            result = (await execution.wait()).result
            print(json.dumps(_result_payload(result), ensure_ascii=False, sort_keys=True))
            _raise_for_failure(result)
            return 0
        return await _stream_result(execution)
    except asyncio.CancelledError:
        await _cancel_interrupted_execution(execution)
        raise


async def _stream_result(execution: "Execution[object]") -> int:
    console = get_console()
    event_console = Console(stderr=True)
    status = "UNKNOWN"
    error_code: object = None
    safe_details: object = {}
    succeeded = False
    output = _AttemptOutput(console, event_console)

    async def items():
        try:
            async for item in execution.watch(include_content=True):
                yield item
        except BaseException:
            output.finish("INTERRUPTED")
            output.close_line()
            raise

    async for item in items():
        if item.depth != 0:
            continue
        event = item.event
        if output.consume(item):
            continue
        if event.event_type == ExecutionDeltaType.ASSISTANT_TEXT_DELTA.value:
            text = event.payload.get("text") if isinstance(event.payload, Mapping) else None
            if isinstance(text, str):
                output.write_text(text)
        elif event.event_type == ExecutionDeltaType.ASSISTANT_THINKING_DELTA.value:
            _emit_event(
                event_console,
                "[thinking]",
                _payload_text(event.payload),
                "cyan",
            )
        elif event.event_type == ExecutionEventType.TOOL_CALL_STARTED.value:
            _emit_event(
                event_console,
                "[tool]",
                _tool_event_text(event.payload),
                "yellow",
            )
        elif event.event_type == ExecutionEventType.TOOL_CALL_FINISHED.value:
            _emit_event(
                event_console,
                "[tool] finished",
                _tool_event_text(event.payload),
                "green",
            )
        elif event.event_type == ExecutionEventType.EXECUTION_SUCCEEDED.value:
            succeeded = True
            status = ExecutionStatus.SUCCEEDED.value
        elif event.event_type in {
            ExecutionEventType.EXECUTION_FAILED.value,
            ExecutionEventType.EXECUTION_CANCELLED.value,
        }:
            status = event.event_type.removeprefix("EXECUTION_")
            output.finish(status)
            if isinstance(event.payload, Mapping):
                error_code = event.payload.get("error_code")
                safe_details = event.payload.get("safe_error_details", {})
        elif event.event_type == ExecutionEventType.EXECUTION_RECOVERY_REQUIRED.value:
            status = ExecutionStatus.RECOVERY_REQUIRED.value
            if isinstance(event.payload, Mapping):
                error_code = event.payload.get("error_code")
                safe_details = event.payload.get("safe_error_details", {})

    if not succeeded:
        output.finish(status)
    else:
        await output.read_completed(execution)
    output.close_line()
    if not succeeded:
        raise CommandError(
            _failure_message(execution.execution_id, status, error_code, safe_details)
        )

    summary = Text()
    summary.append("✓ ", style="green")
    summary.append("SUCCEEDED", style="bold green")
    summary.append("  ")
    summary.append(execution.execution_id, style="dim")
    event_console.print(summary)
    return 0


@dataclass
class _PartDisplay:
    kind: str
    chars: int = 0
    start: bool = False


@dataclass
class _AttemptDisplay:
    status: str = "RUNNING"
    text_chars: int = 0
    thinking: bool = False
    announced: bool = False
    truncated: bool = False
    replaces: int | None = None
    parts: dict[int, _PartDisplay] = field(default_factory=dict)


@dataclass
class _RequestDisplay:
    current: int = 0
    finished: bool = False
    framed: bool = False
    truncated: bool = False
    gap_announced: bool = False
    succeeded: bool = False
    attempts: dict[int, _AttemptDisplay] = field(default_factory=dict)


class _AttemptOutput:
    """Keep streamed bytes, but visibly invalidate partial answers in the same stream."""

    def __init__(self, console: Console, event_console: Console) -> None:
        self.console = console
        self.event_console = event_console
        self.requests: dict[tuple[str, int, int], _RequestDisplay] = {}
        self.line_open = False

    def write_text(self, value: str) -> None:
        self.console.print(Text(value), end="", soft_wrap=True)
        self.line_open = True

    def close_line(self) -> None:
        if self.line_open:
            self.console.print()
            self.line_open = False

    def boundary(self, label: str) -> None:
        self.close_line()
        # Plain text on pipes, styled on terminals; neither claims to erase emitted bytes.
        self.console.print(Text(f"[{label}]", style="yellow"), soft_wrap=True)

    def identity(self, key: tuple[str, int, int], seq: int) -> str:
        return f"run {key[1]} request {key[2]} attempt {seq}"

    def preview_gap(self, key: tuple[str, int, int], request: _RequestDisplay) -> None:
        self.boundary(f"run {key[1]} request {key[2]} preview GAP: following output is not contiguous; no retry implied")
        request.gap_announced = True

    async def read_completed(self, execution: "Execution[object]") -> None:
        for key, request in self.requests.items():
            if not request.succeeded or not (request.framed or request.truncated):
                continue
            label = f"run {key[1]} request {key[2]}"
            try:
                page = await execution.model_interactions(include_content=True, agent_run_seq=key[1], model_request_seq=key[2], limit=1)
                item = next((item for item in page.items if (item.execution_id, item.agent_run_seq, item.model_request_seq) == key), None)
                response = item.response if item is not None and item.status == "SUCCEEDED" and item.content_included else None
                parts = response.get("parts") if isinstance(response, Mapping) else None
                if not isinstance(parts, (list, tuple)):
                    self.boundary(f"{label} FINAL UNAVAILABLE: earlier output is preview only, not a completed answer")
                    continue
                self.boundary(f"{label} CANONICAL TEXT RESPONSE: all earlier preview text for this request is superseded")
                for part in parts:
                    if isinstance(part, Mapping) and part.get("part_kind") == "text" and isinstance(part.get("content"), str):
                        self.write_text(part["content"])
                self.close_line()
                self.boundary(f"{label} COMPLETED: committed SDK text above")
            except Exception:
                self.boundary(f"{label} FINAL UNAVAILABLE: committed response read failed; earlier output is preview only, not a completed answer")

    def invalidate(self, key: tuple[str, int, int], request: _RequestDisplay, seq: int, status: str) -> None:
        attempt = request.attempts[seq]
        if attempt.status != "RUNNING":
            return
        attempt.status = status
        if attempt.text_chars:
            request.framed = True
            self.boundary(
                f"{self.identity(key, seq)} {status}: this attempt's "
                f"{attempt.text_chars} streamed text characters INVALIDATED; "
                "not a completed answer"
            )
        if attempt.thinking:
            _emit_event(self.event_console, "[thinking invalidated]", self.identity(key, seq), "yellow")

    def finish(self, status: str) -> None:
        for key, request in self.requests.items():
            for seq in request.attempts:
                self.invalidate(key, request, seq, status)
            if not request.finished and any(value.status == "SUCCEEDED" and value.text_chars for value in request.attempts.values()):
                self.boundary(f"run {key[1]} request {key[2]} {status}: logical response incomplete; completed segments are not a final answer")
            request.finished = True

    def visible_part(self, request: _RequestDisplay, index: int) -> _PartDisplay | None:
        result = None
        for seq, attempt in sorted(request.attempts.items()):
            if attempt.status != "SUCCEEDED" and not (seq == request.current and attempt.status == "RUNNING"):
                continue
            part = attempt.parts.get(index)
            if part is not None:
                result = _PartDisplay(part.kind, part.chars + (result.chars if result and not part.start else 0), part.start)
        return result

    def update_part(self, key: tuple[str, int, int], request: _RequestDisplay,
                    attempt: _AttemptDisplay, payload: Mapping[str, object], kind: str, text: str) -> None:
        index, start = payload.get("part_index"), payload.get("part_start")
        if not isinstance(index, int) or isinstance(index, bool) or index < 0 or not isinstance(start, bool):
            return
        if start:
            indices = {position for value in request.attempts.values() for position in value.parts}
            visible = {position: part for position in indices if (part := self.visible_part(request, position)) is not None}
            if any(position >= index for position in visible):
                request.framed = True
                label = f"run {key[1]} request {key[2]} part {index} UPDATED: earlier preview may be superseded; committed response readback required"
                if any(part.kind == "text" and part.chars for part in visible.values()):
                    self.boundary(label)
                elif any(part.chars for part in visible.values()):
                    _emit_event(self.event_console, "[thinking updated]", label, "cyan")
            attempt.parts[index] = _PartDisplay(kind, start=True)
        part = attempt.parts.setdefault(index, _PartDisplay(kind))
        part.chars += len(text)

    def consume(self, item: "ExecutionTreeEvent") -> bool:
        event = item.event
        if event.event_type == "STREAM_RESET":
            for key, request in self.requests.items():
                if key[0] != item.execution_id or request.finished:
                    continue
                request.truncated = True
                request.gap_announced = False
                visible = [value for value in request.attempts.values() if value.status in {"RUNNING", "SUCCEEDED"}]
                if any(value.text_chars for value in visible):
                    self.preview_gap(key, request)
                for attempt in visible:
                    attempt.truncated = True
                    for part in attempt.parts.values():
                        part.chars = 0
            return True
        if event.event_type not in {
            "MODEL_ATTEMPT_STARTED", "MODEL_ATTEMPT_FINISHED", "MODEL_REQUEST_FINISHED",
            "ASSISTANT_TEXT_DELTA", "ASSISTANT_THINKING_DELTA",
        } or not isinstance(event.payload, Mapping):
            return False
        payload = event.payload
        values = tuple(payload.get(name) for name in ("agent_run_seq", "model_request_seq", "stream_attempt_seq"))
        if not all(isinstance(value, int) and not isinstance(value, bool) and value > 0 for value in values):
            return False
        run, model_request, seq = values
        key = (item.execution_id, run, model_request)
        request = self.requests.setdefault(key, _RequestDisplay())
        if request.finished and seq not in request.attempts:
            return True
        attempt = request.attempts.setdefault(seq, _AttemptDisplay())
        if seq > request.current:
            request.current = seq
        if event.event_type == "MODEL_ATTEMPT_STARTED":
            replaces = payload.get("replaces_stream_attempt_seq")
            if isinstance(replaces, int) and not isinstance(replaces, bool) and 0 < replaces < seq:
                attempt.replaces = replaces
        elif event.event_type == "MODEL_ATTEMPT_FINISHED":
            status = payload.get("status")
            if status == "SUCCEEDED" and attempt.status == "RUNNING":
                attempt.status = "SUCCEEDED"
            elif status in {"FAILED", "CANCELLED"}:
                self.invalidate(key, request, seq, status)
        elif event.event_type == "MODEL_REQUEST_FINISHED":
            if seq != request.current:
                return True
            status = payload.get("status")
            if status == "SUCCEEDED":
                request.succeeded = True
                truncated = request.truncated or any(value.truncated for value in request.attempts.values() if value.status in {"RUNNING", "SUCCEEDED"})
                if not request.finished and (request.framed or truncated):
                    self.boundary(f"run {run} request {model_request} model succeeded; preview is incomplete until committed response readback")
                if attempt.status == "RUNNING":
                    attempt.status = "SUCCEEDED"
            elif status in {"FAILED", "CANCELLED"}:
                self.invalidate(key, request, seq, status)
                if not request.finished and any(value.status == "SUCCEEDED" and value.text_chars for value in request.attempts.values()):
                    self.boundary(f"run {run} request {model_request} {status}: logical response incomplete; completed segments are not a final answer")
            if seq == request.current:
                request.finished = True
        elif seq == request.current and attempt.status == "RUNNING":
            value = payload.get("text")
            kind = "text" if event.event_type == "ASSISTANT_TEXT_DELTA" else "thinking"
            if kind == "text" and isinstance(value, str) and value and request.truncated and not request.gap_announced:
                self.preview_gap(key, request)
            if isinstance(value, str):
                self.update_part(key, request, attempt, payload, kind, value)
            if request.framed and attempt.replaces is not None and not attempt.announced:
                self.boundary(f"{self.identity(key, seq)} begins; replacement output follows")
                attempt.announced = True
            if event.event_type == "ASSISTANT_TEXT_DELTA" and isinstance(value, str):
                if value or payload.get("part_start") is not True:
                    self.write_text(value)
                attempt.text_chars += len(value)
                attempt.truncated |= payload.get("stream_truncated") is True
            elif event.event_type == "ASSISTANT_THINKING_DELTA" and isinstance(value, str):
                if value or payload.get("part_start") is not True:
                    _emit_event(self.event_console, "[thinking]", value, "cyan")
                attempt.thinking |= bool(value)
        return True


def _emit_event(console: Console, label: str, value: str, style: str) -> None:
    line = Text()
    line.append(label, style=f"bold {style}")
    line.append(" ")
    line.append(value or "-", style="dim")
    console.print(line)


def _tool_event_text(payload: object) -> str:
    if not isinstance(payload, Mapping):
        return ""
    name = payload.get("tool_name")
    call_id = payload.get("call_id")
    parts = [str(name)] if isinstance(name, str) else []
    if isinstance(call_id, str) and call_id:
        parts.append(f"#{call_id}")
    return " · ".join(parts)


def _raise_for_failure(result: "ExecutionResult") -> None:
    if result.status is not ExecutionStatus.SUCCEEDED:
        raise CommandError(
            _failure_message(
                result.execution_id,
                result.status.value,
                result.error_code,
                dict(result.safe_error_details),
            )
        )


def _failure_message(
    execution_id: str,
    status: str,
    error_code: object,
    safe_details: object,
) -> str:
    return (
        "execution failed: "
        f"execution_id={execution_id} status={status} "
        f"error_code={error_code} safe_error_details={safe_details}"
    )


async def _cancel_interrupted_execution(execution: "Execution[object]") -> None:
    result = await execution.cancel(
        idempotency_key=f"ai-run-interrupt:{execution.execution_id}",
    )
    if not result.cancelled:
        environ.logger.warning(
            "ai run cancellation effect not yet confirmed: execution=%s",
            execution.execution_id,
        )
    await execution.wait()


def _result_payload(result: "ExecutionResult") -> dict[str, object]:
    diagnostics = result.error_diagnostics
    return {
        "execution_id": result.execution_id,
        "status": result.status.value,
        "output": result.output,
        "error_code": result.error_code,
        "safe_error_details": dict(result.safe_error_details),
        "error_diagnostics": (
            None
            if diagnostics is None
            else {
                "exception_type": diagnostics.exception_type,
                "exception_message": diagnostics.exception_message,
                "cause_digest": diagnostics.cause_digest,
            }
        ),
    }


def _payload_text(payload: object) -> str:
    if not isinstance(payload, Mapping):
        return ""
    if isinstance(payload.get("text"), str):
        return payload["text"]
    if isinstance(payload.get("tool_name"), str):
        return payload["tool_name"]
    return ""


__all__ = ["emit_result"]
