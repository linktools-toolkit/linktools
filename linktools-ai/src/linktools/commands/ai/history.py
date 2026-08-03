#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""`lt ai history`: show persisted conversation history.

Thin shell -- delegates to
:func:`linktools.ai.cli.console.history.run_history`, which reads sessions /
turns / run messages through :class:`LocalRuntimeClient`."""

from pathlib import Path
from typing import TYPE_CHECKING

from linktools.cli import BaseCommand

if TYPE_CHECKING:
    from argparse import Namespace

    from linktools.cli import CommandParser


class Command(BaseCommand):
    """show persisted conversation history"""

    def init_arguments(self, parser: "CommandParser") -> None:
        parser.add_argument(
            "session",
            nargs="?",
            default=None,
            help="session id (omit to list sessions)",
        )
        parser.add_argument(
            "--run",
            default=None,
            help="show a run's full message trace (within the session)",
        )
        parser.add_argument(
            "-r",
            "--turn",
            type=int,
            default=None,
            help="show a single turn's recorded messages (by sequence)",
        )
        parser.add_argument(
            "--project", type=Path, default=None, help="project root (default: cwd)"
        )
        parser.add_argument("--remote", default=None, help="remote Runtime url")
        parser.add_argument(
            "--json", action="store_true", help="emit one JSON item per line"
        )

    def run(self, args: "Namespace") -> "int | None":
        from linktools.ai.cli.console.history import run_history

        return run_history(
            session=args.session,
            run_id=args.run,
            turn=args.turn,
            project=args.project,
            remote=args.remote,
            json_output=args.json,
        )


command = Command()
