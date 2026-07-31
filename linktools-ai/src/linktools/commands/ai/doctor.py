#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""`lt ai doctor`: validate project + Runtime configuration.

Thin shell -- delegates to
:func:`linktools.ai.cli.console.doctor.run_doctor`, which renders the
``DoctorReport`` produced by :meth:`LocalRuntimeClient.doctor`."""

from pathlib import Path
from typing import TYPE_CHECKING

from linktools.cli import BaseCommand

if TYPE_CHECKING:
    from argparse import Namespace

    from linktools.cli import CommandParser


class Command(BaseCommand):
    """validate project and Runtime configuration"""

    def init_arguments(self, parser: "CommandParser") -> None:
        parser.add_argument(
            "--project", type=Path, default=None, help="project root (default: cwd)"
        )
        parser.add_argument("--remote", default=None, help="remote Runtime url")
        parser.add_argument(
            "--json", action="store_true", help="emit the report as JSON"
        )

    def run(self, args: "Namespace") -> "int | None":
        from linktools.ai.cli.console.doctor import run_doctor

        return run_doctor(
            project=args.project,
            remote=args.remote,
            json_output=args.json,
        )


command = Command()
