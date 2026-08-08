#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""`lt ai run`: run one Workspace Agent task."""

import asyncio
import json
import os
import sys
from argparse import Namespace
from pathlib import Path
from typing import TYPE_CHECKING

from linktools.cli import BaseCommand, CommandError
from pydantic_ai.exceptions import ModelAPIError, UserError

from ...ai.agent.runner import WorkspaceAgentRunner
from ...ai.app.workspace import open_workspace_runtime
from ...ai.core.json import JsonValue
from ...ai.workspace import Workspace, build_workspace_capabilities

if TYPE_CHECKING:
    from linktools.cli import CommandParser


class Command(BaseCommand):
    """run one Agent task locally"""

    @property
    def known_errors(self) -> "list[type[BaseException]]":
        return super().known_errors + [ModelAPIError, UserError]

    def init_arguments(self, parser: "CommandParser") -> None:
        parser.add_argument("prompt", nargs="?", help="the prompt")
        parser.add_argument("--project", type=Path, default=None, help="working directory")
        parser.add_argument("--storage", type=Path, default=None, help="runtime storage directory")
        parser.add_argument("--agent", default=None, help="agent id (default: project default)")
        parser.add_argument("--session", default="main", help="session id (default main)")
        parser.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL"))
        parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
        parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY"))
        parser.add_argument("--json", action="store_true", help="emit one JSON event per line")

    def run(self, args: Namespace) -> int:
        if args.prompt is None:
            raise CommandError("a prompt is required")
        workspace = Workspace.discover(Path.cwd(), root=args.project, storage_root=args.storage)
        capabilities = () if args.model == "test" else build_workspace_capabilities(workspace.root)
        runner = WorkspaceAgentRunner(workspace.root, workspace.config, model=args.model, base_url=args.base_url, api_key=args.api_key, capabilities=capabilities)
        output_terminated = False

        async def on_event(event: "dict[str, JsonValue]") -> None:
            nonlocal output_terminated
            event_type = event.get("type")
            if args.json:
                print(json.dumps(event, ensure_ascii=False, default=str), flush=True)
                return
            if event_type == "text":
                sys.stdout.write(str(event.get("text", "")))
                sys.stdout.flush()
            elif event_type == "text_end" and not output_terminated:
                sys.stdout.write("\n")
                sys.stdout.flush()
                output_terminated = True
            elif event_type == "thinking":
                print(f"[thinking] {event.get('text', '')}", file=sys.stderr, flush=True)
            elif event_type == "tool":
                status = " ok" if event.get("ok") else ""
                print(f"[tool: {event.get('name')} {event.get('phase')}{status}]", file=sys.stderr, flush=True)

        async def execute() -> None:
            nonlocal output_terminated
            async with open_workspace_runtime(workspace, runner=runner) as runtime:
                result = await runtime.run(args.session, args.prompt, agent_id=args.agent, on_event=on_event)
                if args.json:
                    print(json.dumps({"type": "completed", "run_id": result.run_id}, ensure_ascii=False), flush=True)
                elif not output_terminated:
                    print(result.output)

        try:
            asyncio.run(execute())
        except (TypeError, ValueError) as error:
            raise CommandError(str(error)) from error
        return 0


command = Command()
