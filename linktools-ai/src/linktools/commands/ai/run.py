#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""`ai run`: execute one workspace Agent through Runtime."""

import asyncio
import json
import sys
from collections.abc import AsyncIterator
from argparse import Namespace
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING

from linktools.cli import BaseCommand, CommandError
from linktools.cli.argparse import ConfigAction
from linktools.core import ConfigField, environ

from linktools.ai.core import ExecutionDeltaType, ExecutionEventType, ExecutionStatus
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.migrate import provision_runtime_database
from linktools.ai.model import ModelRegistry
from linktools.ai.runtime import Execution, ExecutionResult, Runtime, RuntimeState
from linktools.ai.workspace import Workspace

if TYPE_CHECKING:
    from linktools.cli import CommandParser

OPENAI_BASE_URL = ConfigField(name="OPENAI_BASE_URL", cast=str, default=None)
OPENAI_MODEL = ConfigField(name="OPENAI_MODEL", cast=str, default=None)
OPENAI_API_KEY = ConfigField(name="OPENAI_API_KEY", cast=str, default=None, secret=True)
_logger = environ.get_logger("commands.ai.run")


class Command(BaseCommand):
    """Run a prompt against the Agent definitions in the current workspace."""

    def init_arguments(self, parser: "CommandParser") -> None:
        parser.add_argument("prompt", help="the prompt")
        parser.add_argument("--project", type=Path, default=None, help="working directory")
        parser.add_argument(
            "--storage",
            choices=("filesystem", "sqlite"),
            default="sqlite",
            help="Runtime state storage backend (default: sqlite)",
        )
        parser.add_argument("--base-url", action=ConfigAction, config=OPENAI_BASE_URL)
        parser.add_argument("--model", action=ConfigAction, config=OPENAI_MODEL)
        parser.add_argument("--api-key", action=ConfigAction, config=OPENAI_API_KEY)
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
        workspace_root = Path.cwd() if args.project is None else args.project
        try:
            workspace = Workspace.discover(Path.cwd(), root=workspace_root)
        except AIError as error:
            if error.code is not ErrorCode.WORKSPACE_CONFIG_INVALID:
                raise
            workspace = Workspace.initialize(workspace_root)
        if not isinstance(args.model, str) or not args.model.strip():
            raise CommandError("--model is required")
        session_id = workspace.workspace_id
        memory_scope = workspace.workspace_id
        _logger.info(
            "ai run session selected: workspace=%s session=%s memory_scope=%s",
            workspace.workspace_id,
            session_id,
            memory_scope,
        )

        async def execute() -> int:
            async with _open_runtime_state(workspace, args.storage) as state:
                async with Runtime.open(
                    workspace,
                    state=state,
                    models=ModelRegistry.openai(
                        model=args.model,
                        base_url=args.base_url,
                        api_key=args.api_key,
                    ),
                ) as runtime:
                    return await _emit_result(
                        runtime,
                        args.prompt,
                        session_id,
                        memory_scope,
                        args.json,
                        args.planning,
                        args.thinking,
                    )

        try:
            return asyncio.run(execute())
        except (TypeError, ValueError, AIError) as error:
            raise CommandError(str(error)) from error


@asynccontextmanager
async def _open_runtime_state(
    workspace: Workspace,
    storage: str,
) -> AsyncIterator[RuntimeState]:
    if storage == "filesystem":
        path = workspace.storage_root / "runtime"
        _logger.info("ai run storage selected: backend=filesystem path=%s", path)
        yield RuntimeState.filesystem(path)
        return
    if storage != "sqlite":
        raise ValueError(f"unsupported Runtime storage backend: {storage}")

    path = workspace.storage_root / "runtime.db"
    await asyncio.to_thread(path.parent.mkdir, parents=True, exist_ok=True)
    from sqlalchemy.ext.asyncio import create_async_engine

    bootstrap_engine = create_async_engine(f"sqlite+aiosqlite:///{path}")
    try:
        await provision_runtime_database(bootstrap_engine)
    finally:
        _logger.debug("ai run SQL bootstrap engine disposing: path=%s", path)
        await bootstrap_engine.dispose()
    _logger.info("ai run storage selected: backend=sqlite path=%s", path)
    yield RuntimeState.sqlite(path)


async def _emit_result(
    runtime: Runtime,
    prompt: str,
    session_id: str,
    memory_scope: str,
    as_json: bool,
    planning: bool,
    thinking: bool,
) -> int:
    agent = runtime.agent()
    if as_json:
        execution = await agent.start(
            prompt,
            session_id=session_id,
            memory_scope=memory_scope,
            planning=planning,
            thinking=thinking,
        )
        try:
            result = await execution.wait()
        except asyncio.CancelledError:
            await _cancel_interrupted_execution(execution)
            raise
        payload = _result_payload(result)
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        _require_success(result)
        return 0

    execution = await agent.start(
        prompt,
        session_id=session_id,
        memory_scope=memory_scope,
        planning=planning,
        thinking=thinking,
    )
    try:
        async for item in execution.watch():
            if item.depth != 0:
                continue
            event = item.event
            event_type = event.event_type
            if event_type == ExecutionDeltaType.ASSISTANT_TEXT_DELTA.value:
                text = event.payload.get("text") if isinstance(event.payload, dict) else None
                if isinstance(text, str):
                    sys.stdout.write(text)
                    sys.stdout.flush()
            elif event_type == ExecutionDeltaType.ASSISTANT_THINKING_DELTA.value:
                _write_stderr("[thinking] " + _payload_text(event.payload))
            elif event_type == ExecutionEventType.TOOL_CALL_STARTED.value:
                _write_stderr("[tool] " + _payload_text(event.payload))
            elif event_type == ExecutionEventType.TOOL_CALL_FINISHED.value:
                _write_stderr("[tool] finished " + _payload_text(event.payload))
        result = await execution.wait()
    except asyncio.CancelledError:
        await _cancel_interrupted_execution(execution)
        raise
    sys.stdout.write("\n")
    sys.stdout.flush()
    _require_success(result)
    return 0


def _require_success(result: ExecutionResult) -> None:
    if result.status is ExecutionStatus.SUCCEEDED:
        return
    raise CommandError(
        "execution failed: "
        f"execution_id={result.execution_id} status={result.status.value} "
        f"error_code={result.error_code} "
        f"safe_error_details={dict(result.safe_error_details)}"
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
