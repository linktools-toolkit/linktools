#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""`lt ai init`: initialize a `.linktools` project.

Thin shell -- delegates to
:func:`linktools.ai_cli.console.init_project.initialize_project`."""

from pathlib import Path
from typing import TYPE_CHECKING

from linktools.cli import BaseCommand

if TYPE_CHECKING:
    from argparse import Namespace

    from linktools.cli import CommandParser


class Command(BaseCommand):
    """initialize a .linktools project"""

    def init_arguments(self, parser: "CommandParser") -> None:
        parser.add_argument(
            "path",
            nargs="?",
            type=Path,
            default=None,
            help="project root (default: current directory)",
        )

    def run(self, args: "Namespace") -> "int | None":
        from linktools.ai_cli.console.init_project import initialize_project

        return initialize_project(args.path)


command = Command()
