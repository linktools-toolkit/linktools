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
from pydantic_ai.exceptions import ModelAPIError, UserError

from ...ai.core import ExecutionDeltaType, ExecutionEventType, ExecutionStatus
from ...ai.errors import AIError, ErrorCode
from ...ai.migrate import provision_runtime_database
from ...ai.model import ModelRegistry
from ...ai.runtime import ExecutionResult, Runtime, RuntimeState
from ...ai.workspace import Workspace, open_workspace_runtime

if TYPE_CHECKING:
    from linktools.cli import CommandParser

OPENAI_BASE_URL = ConfigField(name="OPENAI_BASE_URL", cast=str, default=None)
OPENAI_MODEL = ConfigField(name="OPENAI_MODEL", cast=str, default=None)
OPENAI_API_KEY = ConfigField(name="OPENAI_API_KEY", cast=str, default=None, secret=True)
_logger = environ.get_logger("commands.ai.run")


class Command(BaseCommand):
    """Run a prompt against the Agent definitions in the current workspace."""

    @property
    def known_errors(self) -> "list[type[BaseException]]":
        return super().known_errors + [ModelAPIError, UserError]

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
        parser.add_argument("--planning", action="store_true", help="enable planning for this execution")
        parser.add_argument("--thinking", action="store_true", help="enable model thinking for this execution")
        parser.add_argument(
            "--json",
            action="store_true",
            help="emit one final JSON result",
        )

    def run(self, args: Namespace) -> int:
        workspace = Workspace.discover(Path.cwd(), root=args.project)
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
            state = await _build_runtime_state(workspace, args.storage)
            async with open_workspace_runtime(
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


async def _build_runtime_state(workspace: Workspace, storage: str) -> RuntimeState:
    if storage == "filesystem":
        path = workspace.storage_root / "runtime"
        _logger.info("ai run storage selected: backend=filesystem path=%s", path)
        return RuntimeState.filesystem(path)
    if storage != "sqlite":
        raise ValueError(f"unsupported Runtime storage backend: {storage}")

    path = workspace.storage_root / "runtime.db"
    await asyncio.to_thread(path.parent.mkdir, parents=True, exist_ok=True)
    from sqlalchemy.ext.asyncio import create_async_engine

    engine = create_async_engine(f"sqlite+aiosqlite:///{path}")
    try:
        await provision_runtime_database(engine)
    finally:
        await engine.dispose()
    _logger.info("ai run storage selected: backend=sqlite path=%s", path)
    return RuntimeState.sqlite(path)


async def _emit_result(
    runtime: Runtime,
    prompt: str,
    session_id: str,
    memory_scope: str,
    as_json: bool,
    planning: bool,
    thinking: bool,
) -> int:
    agent = runtime.agent(planning=planning, thinking=thinking)
    if as_json:
        result = await agent.run(prompt, session_id=session_id, memory_scope=memory_scope)
        payload = _result_payload(result)
        if result.status is not ExecutionStatus.SUCCEEDED:
            error_code, safe_details = await _terminal_failure_details(runtime, result.execution_id)
            payload["error_code"] = error_code
            payload["safe_error_details"] = safe_details
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        if result.status is not ExecutionStatus.SUCCEEDED:
            raise CommandError(
                "execution failed: "
                f"execution_id={result.execution_id} status={result.status.value} "
                f"error_code={payload['error_code']} "
                f"safe_error_details={payload['safe_error_details']}"
            )
        return 0

    succeeded = False
    execution_id = "unknown"
    terminal_status = "UNKNOWN"
    terminal_error_code: object = None
    terminal_safe_details: object = {}
    async for event in agent.stream(prompt, session_id=session_id, memory_scope=memory_scope):
        execution_id = event.execution_id
        if event.event_type is ExecutionDeltaType.ASSISTANT_TEXT_DELTA:
            text = event.payload.get("text") if isinstance(event.payload, dict) else None
            if isinstance(text, str):
                sys.stdout.write(text)
                sys.stdout.flush()
        elif event.event_type is ExecutionDeltaType.ASSISTANT_THINKING_DELTA:
            _write_stderr("[thinking] " + _payload_text(event.payload))
        elif event.event_type is ExecutionEventType.TOOL_CALL_STARTED:
            _write_stderr("[tool] " + _payload_text(event.payload))
        elif event.event_type is ExecutionEventType.TOOL_CALL_FINISHED:
            _write_stderr("[tool] finished " + _payload_text(event.payload))
        elif event.event_type is ExecutionEventType.EXECUTION_SUCCEEDED:
            succeeded = True
            terminal_status = ExecutionStatus.SUCCEEDED.value
        elif event.event_type in {
            ExecutionEventType.EXECUTION_FAILED,
            ExecutionEventType.EXECUTION_CANCELLED,
        }:
            terminal_status = event.event_type.value.removeprefix("EXECUTION_")
            if isinstance(event.payload, dict):
                terminal_error_code = event.payload.get("error_code")
                terminal_safe_details = event.payload.get("safe_error_details", {})
    sys.stdout.write("\n")
    sys.stdout.flush()
    if not succeeded:
        raise CommandError(
            "execution failed: "
            f"execution_id={execution_id} status={terminal_status} "
            f"error_code={terminal_error_code} safe_error_details={terminal_safe_details}"
        )
    return 0


async def _terminal_failure_details(
    runtime: Runtime,
    execution_id: str,
) -> tuple[object, object]:
    after_sequence = 0
    while True:
        page = await runtime.event.list(
            execution_id,
            principal=runtime.default_principal,
            after_sequence=after_sequence,
            limit=100,
        )
        for event in reversed(page.items):
            if event.event_type not in {
                ExecutionEventType.EXECUTION_FAILED,
                ExecutionEventType.EXECUTION_CANCELLED,
            }:
                continue
            if not isinstance(event.payload, dict):
                return None, {}
            return event.payload.get("error_code"), event.payload.get("safe_error_details", {})
        if page.next_cursor is None:
            return None, {}
        try:
            next_sequence = int(page.next_cursor)
        except (TypeError, ValueError) as error:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
        if next_sequence <= after_sequence:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        after_sequence = next_sequence


def _result_payload(result: ExecutionResult) -> dict[str, object]:
    return {
        "execution_id": result.execution_id,
        "status": result.status.value,
        "output": result.output,
        "output_schema_id": result.output_schema_id,
        "output_schema_revision": result.output_schema_revision,
        "output_schema_fingerprint": result.output_schema_fingerprint,
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
