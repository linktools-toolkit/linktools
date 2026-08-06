#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""`lt ai acp`: start the local ACP stdio Agent."""

import asyncio
from argparse import Namespace
from pathlib import Path
from typing import TYPE_CHECKING

from linktools.cli import BaseCommand, CommandError

from ...ai.entry.acp import ACPApplication
from ...ai.local import LocalProject

if TYPE_CHECKING:
    from linktools.cli import CommandParser


class Command(BaseCommand):
    """start ACP for the local Agent runtime"""

    def init_arguments(self, parser: "CommandParser") -> None:
        parser.add_argument("--project", type=Path, default=None, help="working directory")
        parser.add_argument("--storage", type=Path, default=None, help="runtime storage directory")

    def run(self, args: Namespace) -> int:
        try:
            project = LocalProject.discover(Path.cwd(), root=args.project, storage_root=args.storage)
            asyncio.run(ACPApplication.for_project(project).serve())
        except ModuleNotFoundError as error:
            raise CommandError("ai acp requires pydantic-ai-harness and agent-client-protocol dependencies") from error
        except ValueError as error:
            raise CommandError(str(error)) from error
        return 0


command = Command()
