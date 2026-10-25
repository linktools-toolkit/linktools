#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""`ai run`: execute one workspace Agent through Runtime."""

import asyncio
import json
from argparse import Namespace
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from rich import get_console
from rich.console import Console
from rich.text import Text

from linktools.cli import BaseCommand, CommandError
from linktools.core import environ

from linktools.ai.core import ExecutionDeltaType, ExecutionEventType, ExecutionStatus

from ._common import (
    _add_local_runtime_arguments,
    _load_workspace,
    _local_runtime_model,
    _open_local_runtime,
    _run_async,
)

if TYPE_CHECKING:
    from linktools.ai.runtime import Execution, ExecutionResult, ExecutionTreeEvent, Runtime
    from linktools.cli import CommandParser

_logger = environ.get_logger("commands.ai.run")


class Command(BaseCommand):
    """Run a prompt against Agent specs in the current workspace."""

    def init_arguments(self, parser: "CommandParser") -> None:
        parser.add_argument("prompt", help="the prompt")
        _add_local_runtime_arguments(parser)
        parser.add_argument(
            "--planning",
            action="store_true",
            help="enable the planning capability for this execution",
        )
        parser.add_argument(
            "--thinking",
            action="store_true",
            help="enable model thinking for this execution",
        )
        parser.add_argument(
            "--json",
            action="store_true",
            help="emit one final JSON result",
        )

    def run(self, args: Namespace) -> int:
        workspace = _load_workspace(args.project)
        memory_scope = args.memory if args.memory is not None else "default"
        _logger.info(
            "ai run session selected: namespace=%s session=%s memory_scope=%s",
            "default",
            "default",
            memory_scope,
        )

        async def execute() -> int:
            async with _open_local_runtime(
                workspace,
                model=_local_runtime_model(workspace, args),
            ) as runtime:
                return await _emit_result(
                    runtime,
                    args.prompt,
                    "default",
                    memory_scope,
                    args.json,
                    args.planning,
                    args.thinking,
                )

        return _run_async(execute())


async def _emit_result(
    runtime: "Runtime",
    prompt: str,
    session_id: str,
    memory_scope: str,
    as_json: bool,
    planning: bool,
    thinking: bool,
) -> int:
    execution = await runtime.agents.get().start(
        prompt,
        session_id=session_id,
        memory_scope=memory_scope,
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
            text = event.payload.get("text") if isinstance(event.payload, dict) else None
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
            if isinstance(event.payload, dict):
                error_code = event.payload.get("error_code")
                safe_details = event.payload.get("safe_error_details", {})
        elif event.event_type == ExecutionEventType.EXECUTION_RECOVERY_REQUIRED.value:
            status = ExecutionStatus.RECOVERY_REQUIRED.value
            if isinstance(event.payload, dict):
                error_code = event.payload.get("error_code")
                safe_details = event.payload.get("safe_error_details", {})

    if not succeeded:
        output.finish(status)
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
class _AttemptDisplay:
    status: str = "RUNNING"
    text_chars: int = 0
    thinking: bool = False
    announced: bool = False
    truncated: bool = False


@dataclass
class _RequestDisplay:
    current: int = 0
    finished: bool = False
    framed: bool = False
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

    def invalidate(self, key: tuple[str, int, int], request: _RequestDisplay, seq: int, status: str) -> None:
        attempt = request.attempts[seq]
        if attempt.status != "RUNNING":
            return
        attempt.status = status
        if attempt.text_chars:
            request.framed = True
            self.boundary(
                f"{self.identity(key, seq)} {status}: partial output INVALIDATED; "
                "not a completed answer"
            )
        if attempt.thinking:
            _emit_event(self.event_console, "[thinking invalidated]", self.identity(key, seq), "yellow")

    def finish(self, status: str) -> None:
        for key, request in self.requests.items():
            for seq in request.attempts:
                self.invalidate(key, request, seq, status)

    def consume(self, item: "ExecutionTreeEvent") -> bool:
        event = item.event
        if event.event_type == "STREAM_RESET":
            for key, request in self.requests.items():
                if key[0] != item.execution_id:
                    continue
                attempt = request.attempts.get(request.current)
                if attempt is not None and attempt.status == "RUNNING":
                    attempt.truncated = True
                    if attempt.text_chars:
                        self.boundary(f"{self.identity(key, request.current)} preview GAP: following output is not contiguous; no retry implied")
            return True
        if event.event_type not in {
            "MODEL_ATTEMPT_STARTED", "MODEL_ATTEMPT_FAILED", "MODEL_REQUEST_FINISHED",
            "ASSISTANT_TEXT_DELTA", "ASSISTANT_THINKING_DELTA",
        } or not isinstance(event.payload, dict):
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
        attempt = request.attempts.setdefault(seq, _AttemptDisplay(status="FAILED" if seq < request.current else "RUNNING"))
        if seq > request.current:
            if request.current:
                self.invalidate(key, request, request.current, "FAILED")
            request.current = seq
        if event.event_type == "MODEL_ATTEMPT_FAILED":
            self.invalidate(key, request, seq, "CANCELLED" if payload.get("status") == "CANCELLED" else "FAILED")
        elif event.event_type == "MODEL_REQUEST_FINISHED":
            if seq != request.current:
                return True
            status = payload.get("status")
            if status == "SUCCEEDED":
                if attempt.status != "SUCCEEDED" and request.framed:
                    self.boundary(f"{self.identity(key, seq)} COMPLETED" + ("; preview is incomplete" if attempt.truncated else ""))
                elif attempt.status != "SUCCEEDED" and attempt.truncated and attempt.text_chars:
                    self.boundary(f"{self.identity(key, seq)} model completed; preview is incomplete")
                attempt.status = "SUCCEEDED"
            elif status in {"FAILED", "CANCELLED"}:
                self.invalidate(key, request, seq, status)
            if seq == request.current:
                request.finished = True
        elif seq == request.current and attempt.status == "RUNNING":
            if request.framed and not attempt.announced:
                self.boundary(f"{self.identity(key, seq)} begins; replacement output follows")
                attempt.announced = True
            value = payload.get("text")
            if event.event_type == "ASSISTANT_TEXT_DELTA" and isinstance(value, str):
                self.write_text(value)
                attempt.text_chars += len(value)
                attempt.truncated |= payload.get("stream_truncated") is True
            elif event.event_type == "ASSISTANT_THINKING_DELTA" and isinstance(value, str):
                _emit_event(self.event_console, "[thinking]", value, "cyan")
                attempt.thinking = True
        return True


def _emit_event(console: Console, label: str, value: str, style: str) -> None:
    line = Text()
    line.append(label, style=f"bold {style}")
    line.append(" ")
    line.append(value or "-", style="dim")
    console.print(line)


def _tool_event_text(payload: object) -> str:
    if not isinstance(payload, dict):
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
        _logger.warning(
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
    if not isinstance(payload, dict):
        return ""
    if isinstance(payload.get("text"), str):
        return payload["text"]
    if isinstance(payload.get("tool_name"), str):
        return payload["tool_name"]
    return ""


command = Command()
