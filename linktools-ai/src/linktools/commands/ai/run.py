#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""`ai run`: execute one workspace Agent through Runtime."""

import asyncio
import json
import sys
from argparse import Namespace
from pathlib import Path
from typing import TYPE_CHECKING

from linktools.cli import BaseCommand, CommandError
from linktools.cli.argparse import ConfigAction
from linktools.core import ConfigField, environ

from linktools.ai.core import ExecutionDeltaType, ExecutionEventType, ExecutionStatus
from linktools.ai.model import ModelRegistry
from linktools.ai.runtime import Execution, ExecutionResult, Runtime

from ._common import _load_workspace, _open_local_runtime, _run_async

if TYPE_CHECKING:
    from linktools.cli import CommandParser

OPENAI_BASE_URL = ConfigField(name="OPENAI_BASE_URL", cast=str, default=None)
OPENAI_MODEL = ConfigField(name="OPENAI_MODEL", cast=str, default=None)
OPENAI_API_KEY = ConfigField(name="OPENAI_API_KEY", cast=str, default=None, secret=True)
OPENAI_VISION = ConfigField(name="OPENAI_VISION", cast=bool, default=False)
_logger = environ.get_logger("commands.ai.run")


class Command(BaseCommand):
    """Run a prompt against the Agent definitions in the current workspace."""

    def init_arguments(self, parser: "CommandParser") -> None:
        parser.add_argument("prompt", help="the prompt")
        parser.add_argument(
            "--project", type=Path, default=None, help="working directory"
        )
        parser.add_argument("--base-url", action=ConfigAction, config=OPENAI_BASE_URL)
        parser.add_argument("--model", action=ConfigAction, config=OPENAI_MODEL)
        parser.add_argument("--api-key", action=ConfigAction, config=OPENAI_API_KEY)
        parser.add_argument("--vision", action=ConfigAction, config=OPENAI_VISION)
        parser.add_argument(
            "--planning",
            action="store_true",
            help="enable planning for this execution",
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
        if not isinstance(args.model, str) or not args.model.strip():
            raise CommandError("--model is required")
        workspace_id = workspace.workspace_id
        _logger.info(
            "ai run session selected: workspace=%s session=%s memory_scope=%s",
            workspace_id,
            workspace_id,
            workspace_id,
        )

        async def execute() -> int:
            async with _open_local_runtime(
                workspace,
                models=ModelRegistry.openai(
                    model=args.model,
                    vision=args.vision,
                    base_url=args.base_url,
                    api_key=args.api_key,
                ),
            ) as runtime:
                return await _emit_result(
                    runtime,
                    args.prompt,
                    workspace_id,
                    workspace_id,
                    args.json,
                    args.planning,
                    args.thinking,
                )

        return _run_async(execute())


async def _emit_result(
    runtime: Runtime,
    prompt: str,
    session_id: str,
    memory_scope: str,
    as_json: bool,
    planning: bool,
    thinking: bool,
) -> int:
    execution = await runtime.agent().start(
        prompt,
        session_id=session_id,
        memory_scope=memory_scope,
        planning=planning,
        thinking=thinking,
    )
    try:
        if as_json:
            result = await execution.wait()
            print(json.dumps(_result_payload(result), ensure_ascii=False, sort_keys=True))
            _raise_for_failure(result)
            return 0
        return await _stream_result(execution)
    except asyncio.CancelledError:
        await _cancel_interrupted_execution(execution)
        raise


async def _stream_result(execution: "Execution[object]") -> int:
    status = "UNKNOWN"
    error_code: object = None
    safe_details: object = {}
    succeeded = False
    async for item in execution.watch():
        if item.depth != 0:
            continue
        event = item.event
        if event.event_type == ExecutionDeltaType.ASSISTANT_TEXT_DELTA.value:
            text = event.payload.get("text") if isinstance(event.payload, dict) else None
            if isinstance(text, str):
                sys.stdout.write(text)
                sys.stdout.flush()
        elif event.event_type == ExecutionDeltaType.ASSISTANT_THINKING_DELTA.value:
            _write_stderr("[thinking] " + _payload_text(event.payload))
        elif event.event_type == ExecutionEventType.TOOL_CALL_STARTED.value:
            _write_stderr("[tool] " + _payload_text(event.payload))
        elif event.event_type == ExecutionEventType.TOOL_CALL_FINISHED.value:
            _write_stderr("[tool] finished " + _payload_text(event.payload))
        elif event.event_type == ExecutionEventType.EXECUTION_SUCCEEDED.value:
            succeeded = True
            status = ExecutionStatus.SUCCEEDED.value
        elif event.event_type in {
            ExecutionEventType.EXECUTION_FAILED.value,
            ExecutionEventType.EXECUTION_CANCELLED.value,
        }:
            status = event.event_type.removeprefix("EXECUTION_")
            if isinstance(event.payload, dict):
                error_code = event.payload.get("error_code")
                safe_details = event.payload.get("safe_error_details", {})
    sys.stdout.write("\n")
    sys.stdout.flush()
    if not succeeded:
        raise CommandError(
            _failure_message(execution.execution_id, status, error_code, safe_details)
        )
    return 0


def _raise_for_failure(result: ExecutionResult) -> None:
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


def _result_payload(result: ExecutionResult) -> dict[str, object]:
    diagnostics = result.error_diagnostics
    return {
        "execution_id": result.execution_id,
        "status": result.status.value,
        "output": result.output,
        "output_fingerprint": result.output_fingerprint,
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


def _write_stderr(value: str) -> None:
    sys.stderr.write(value + "\n")
    sys.stderr.flush()


command = Command()
