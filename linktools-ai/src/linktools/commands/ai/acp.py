#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""`lt ai acp`: start the local ACP stdio Agent."""

import asyncio
from argparse import Namespace
from pathlib import Path
from typing import TYPE_CHECKING

from linktools.cli import BaseCommand, CommandError

from ...ai.inbound.acp import LocalACPAgent, serve_stdio
from ...ai.local import LocalAgentRuntime, LocalProject

if TYPE_CHECKING:
    from linktools.cli import CommandParser


class Command(BaseCommand):
    """start ACP for the local Agent runtime"""

    def init_arguments(self, parser: "CommandParser") -> None:
        parser.add_argument("--project", type=Path, default=None)

    def run(self, args: Namespace) -> int:
        try:
            project = LocalProject.discover(Path.cwd(), root=args.project)
            runtime = LocalAgentRuntime(project)
            asyncio.run(serve_stdio(LocalACPAgent(runtime)))
        except ModuleNotFoundError as error:
            raise CommandError("ai acp requires the agent-client-protocol dependency") from error
        except ValueError as error:
            raise CommandError(str(error)) from error
        return 0


command = Command()
