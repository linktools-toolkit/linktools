#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""`lt ai continue`: approve, reject, or resume an existing Run.

Thin shell -- delegates to
:func:`linktools.ai_cli.console.continue_run.continue_run`, which dispatches on
the run's status. The command name is ``continue`` (a Python keyword, so the
module is ``continue_`` and the name is set explicitly)."""

from typing import TYPE_CHECKING

from linktools.ai_cli.errors import (
    InvalidRunTransitionError,
    RunConflictError,
    RunNotFoundError,
)
from linktools.cli import BaseCommand

if TYPE_CHECKING:
    from argparse import Namespace

    from linktools.cli import CommandParser


class Command(BaseCommand):
    """approve, reject, or resume an existing Run"""

    name = "continue"

    @property
    def known_errors(self) -> "list[type[BaseException]]":
        return super().known_errors + [
            RunNotFoundError,
            InvalidRunTransitionError,
            RunConflictError,
        ]

    def init_arguments(self, parser: "CommandParser") -> None:
        parser.add_argument("run_id", help="the run id to continue")
        action = parser.add_mutually_exclusive_group()
        action.add_argument("--approve", action="store_true", help="approve + resume")
        action.add_argument("--reject", action="store_true", help="reject + cancel")
        action.add_argument(
            "--resume", action="store_true", help="resume an already-approved run"
        )

    def run(self, args: "Namespace") -> "int | None":
        from linktools.ai_cli.console.continue_run import continue_run

        return continue_run(
            run_id=args.run_id,
            approve=args.approve,
            reject=args.reject,
            resume=args.resume,
        )


command = Command()
