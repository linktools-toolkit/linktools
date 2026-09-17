#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""`lt ai acp`: start the local ACP stdio Agent."""

import asyncio
from argparse import Namespace
from pathlib import Path
from typing import TYPE_CHECKING

from linktools.cli import BaseCommand, CommandError

from linktools.ai.acp import ACPAgent, serve_stdio
from linktools.ai.errors import AIError
from linktools.ai.runtime import Runtime

from .._ai_common import _load_workspace, _local_metrics, _local_runtime_state

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

        async def execute() -> None:
            state = _local_runtime_state(workspace)
            metrics = await _local_metrics(workspace)
            async with Runtime.open(
                workspace,
                state=state,
                metrics=metrics,
            ) as runtime:
                await serve_stdio(
                    ACPAgent(
                        runtime,
                        principal=runtime.default_principal,
                        memory_scope=memory_scope,
                    )
                )

        try:
            asyncio.run(execute())
        except ModuleNotFoundError as error:
            raise CommandError(
                "ai acp requires the agent-client-protocol dependency"
            ) from error
        except (AIError, ValueError) as error:
            raise CommandError(str(error)) from error
        return 0


command = Command()
