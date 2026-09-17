#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""`lt ai acp`: start the local ACP stdio Agent."""

from argparse import Namespace
from pathlib import Path
from typing import TYPE_CHECKING

from linktools.cli import BaseCommand, CommandError

from linktools.ai.acp import ACPAgent, serve_stdio

from .._ai_common import _load_workspace, _open_local_runtime, _run_async

if TYPE_CHECKING:
    from linktools.cli import CommandParser


class Command(BaseCommand):
    """start ACP for the local Agent runtime"""

    def init_arguments(self, parser: "CommandParser") -> None:
        parser.add_argument("--project", type=Path, default=None, help="working directory")
        parser.add_argument(
            "--memory",
            default=None,
            help="caller-owned memory scope (default: workspace id)",
        )

    def run(self, args: Namespace) -> int:
        workspace = _load_workspace(args.project)
        memory_scope = args.memory if args.memory is not None else workspace.workspace_id

        async def execute() -> int:
            async with _open_local_runtime(workspace) as runtime:
                await serve_stdio(
                    ACPAgent(
                        runtime,
                        principal=runtime.default_principal,
                        memory_scope=memory_scope,
                    )
                )
            return 0

        try:
            return _run_async(execute())
        except ModuleNotFoundError as error:
            raise CommandError(
                "ai acp requires the agent-client-protocol dependency"
            ) from error


command = Command()
