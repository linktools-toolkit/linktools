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
from linktools.core import ConfigField
from pydantic_ai.exceptions import ModelAPIError, UserError

from ...ai.core import ExecutionEventType, ExecutionStatus
from ...ai.errors import AIError
from ...ai.runtime import Runtime
from ...ai.workspace import Workspace, open_workspace_runtime

if TYPE_CHECKING:
    from linktools.cli import CommandParser

OPENAI_BASE_URL = ConfigField(name="OPENAI_BASE_URL", cast=str, default=None)
OPENAI_MODEL = ConfigField(name="OPENAI_MODEL", cast=str, default=None)
OPENAI_API_KEY = ConfigField(name="OPENAI_API_KEY", cast=str, default=None, secret=True)


class Command(BaseCommand):
    """Run a prompt against the Agent definitions in the current workspace."""

    @property
    def known_errors(self) -> "list[type[BaseException]]":
        return super().known_errors + [ModelAPIError, UserError]

    def init_arguments(self, parser: "CommandParser") -> None:
        parser.add_argument("prompt", help="the prompt")
        parser.add_argument("--project", type=Path, default=None, help="working directory")
        parser.add_argument("--base-url", action=ConfigAction, config=OPENAI_BASE_URL)
        parser.add_argument("--model", action=ConfigAction, config=OPENAI_MODEL)
        parser.add_argument("--api-key", action=ConfigAction, config=OPENAI_API_KEY)
        parser.add_argument("--json", action="store_true", help="emit one final JSON result")

    def run(self, args: Namespace) -> int:
        workspace = Workspace.discover(Path.cwd(), root=args.project)
        if not isinstance(args.model, str) or not args.model.strip():
            raise CommandError("--model is required")
        memory_namespace = workspace.workspace_id

        async def execute() -> int:
            async with open_workspace_runtime(
                workspace,
                model=args.model,
                base_url=args.base_url,
                api_key=args.api_key,
            ) as runtime:
                return await _emit_result(runtime, workspace, args.prompt, memory_namespace, args.json)

        try:
            return asyncio.run(execute())
        except (TypeError, ValueError, AIError) as error:
            raise CommandError(str(error)) from error


async def _emit_result(runtime: Runtime, workspace: Workspace, prompt: str, memory_namespace: str, as_json: bool) -> int:
    principal = _trusted_principal(workspace.workspace_id)
    if as_json:
        result = await runtime.run(prompt, principal=principal, memory_namespace=memory_namespace)
        payload = _result_payload(result)
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        if result.status is not ExecutionStatus.SUCCEEDED:
            raise CommandError(f"execution {result.execution_id} finished with status {result.status.value}")
        return 0

    succeeded = False
    async for event in runtime.stream(prompt, principal=principal, memory_namespace=memory_namespace):
        if event.event_type is ExecutionEventType.ASSISTANT_TEXT_DELTA:
            text = event.payload.get("text") if isinstance(event.payload, dict) else None
            if isinstance(text, str):
                sys.stdout.write(text)
                sys.stdout.flush()
        elif event.event_type is ExecutionEventType.ASSISTANT_THINKING_DELTA:
            _write_stderr("[thinking] " + _payload_text(event.payload))
        elif event.event_type is ExecutionEventType.TOOL_CALL_STARTED:
            _write_stderr("[tool] " + _payload_text(event.payload))
        elif event.event_type is ExecutionEventType.TOOL_CALL_FINISHED:
            _write_stderr("[tool] finished " + _payload_text(event.payload))
        elif event.event_type is ExecutionEventType.EXECUTION_SUCCEEDED:
            succeeded = True
    sys.stdout.write("\n")
    sys.stdout.flush()
    if not succeeded:
        raise CommandError("execution did not succeed")
    return 0


def _result_payload(result: object) -> dict[str, object]:
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


def _trusted_principal(workspace_id: str):
    from ...ai.workspace import trusted_workspace_principal

    return trusted_workspace_principal(workspace_id)


command = Command()
