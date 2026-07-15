#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""`lt ai tui`: start the interactive Textual interface.

Thin shell -- delegates to :func:`linktools.ai_cli.tui.run_tui`, which builds a
local RuntimeClient and runs the Textual chat app. Textual is an optional
dependency (``linktools-ai[tui]``); if it is absent ``run_tui`` raises a clear
``CommandError``."""

from pathlib import Path
from typing import TYPE_CHECKING

from linktools.cli import BaseCommand

if TYPE_CHECKING:
    from argparse import Namespace

    from linktools.cli import CommandParser


class Command(BaseCommand):
    """start the interactive Textual interface"""

    def init_arguments(self, parser: "CommandParser") -> None:
        parser.add_argument(
            "--project", type=Path, default=None, help="project root (default: cwd)"
        )
        parser.add_argument("--remote", default=None, help="remote Runtime url")

    def run(self, args: "Namespace") -> "int | None":
        from linktools.ai_cli.tui import run_tui

        return run_tui(project=args.project, remote=args.remote)


command = Command()
