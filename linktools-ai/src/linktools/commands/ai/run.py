#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""`lt ai run`: run one local Agent task."""

import asyncio
import json
import sys
from argparse import Namespace
from pathlib import Path
from typing import TYPE_CHECKING

from linktools.cli import BaseCommand, CommandError
from linktools.cli.argparse import ConfigAction
from pydantic_ai.exceptions import ModelAPIError, UserError

from ...ai.agent.runner import LocalAgentRunner
from ...ai.core.json import JsonValue
from ...ai.local.project import OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL
from ...ai.local import LocalAgentRuntime, LocalProject, LocalRunResult, build_local_capabilities

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
        parser.add_argument("--base-url", action=ConfigAction, config=OPENAI_BASE_URL)
        parser.add_argument("--model", action=ConfigAction, config=OPENAI_MODEL)
        parser.add_argument("--api-key", action=ConfigAction, config=OPENAI_API_KEY)
        parser.add_argument("--json", action="store_true", help="emit one JSON event per line")

    def run(self, args: Namespace) -> int:
        if args.prompt is None:
            raise CommandError("a prompt is required")
        project = LocalProject.discover(Path.cwd(), root=args.project, storage_root=args.storage)
        if args.model == "test":
            runner = LocalAgentRunner(
                project.root,
                project.config,
                model=args.model,
                base_url=args.base_url,
                api_key=args.api_key,
            )
        else:
            runner = LocalAgentRunner(
                project.root,
                project.config,
                model=args.model,
                base_url=args.base_url,
                api_key=args.api_key,
                capabilities=build_local_capabilities(project.root),
            )
        runtime = LocalAgentRuntime(
            project,
            runner=runner,
        )
        output_terminated = False

        async def on_event(event: "dict[str, JsonValue]") -> None:
            nonlocal output_terminated
            event_type = event.get("type")
            if args.json:
                if event_type != "text_end":
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
                print(
                    f"[tool: {event.get('name')} {event.get('phase')}{status}]",
                    file=sys.stderr,
                    flush=True,
                )

        async def execute() -> LocalRunResult:
            return await runtime.run(args.session, args.prompt, agent_id=args.agent, on_event=on_event)

        try:
            result = asyncio.run(execute())
        except (TypeError, ValueError) as error:
            raise CommandError(str(error)) from error
        if args.json:
            print(json.dumps({"type": "completed", "run_id": result.run_id}, ensure_ascii=False), flush=True)
        elif not output_terminated:
            print()
        return 0


command = Command()
