#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""`lt ai tui`: start the interactive Textual interface."""

from pathlib import Path
from typing import TYPE_CHECKING

from linktools.ai_cli.fields import OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL
from linktools.cli import BaseCommand
from linktools.cli.argparse import ConfigAction

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
        parser.add_argument(
            "--base-url", action=ConfigAction, config=OPENAI_BASE_URL
        )
        parser.add_argument(
            "--model", action=ConfigAction, config=OPENAI_MODEL
        )
        parser.add_argument(
            "--api-key", action=ConfigAction, config=OPENAI_API_KEY
        )

    def run(self, args: "Namespace") -> "int | None":
        from linktools.ai_cli.tui import run_tui

        return run_tui(
            project=args.project,
            remote=args.remote,
            base_url=args.base_url,
            model=args.model,
            api_key=args.api_key,
        )


command = Command()
