#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""`lt ai continue`: approve, reject, or resume an existing Run."""

from typing import TYPE_CHECKING

from linktools.ai_cli.errors import (
    InvalidRunTransitionError,
    RunConflictError,
    RunNotFoundError,
)
from linktools.ai_cli.fields import OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL
from linktools.cli import BaseCommand
from linktools.cli.argparse import ConfigAction

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
        from linktools.ai_cli.console.continue_run import continue_run

        return continue_run(
            run_id=args.run_id,
            approve=args.approve,
            reject=args.reject,
            resume=args.resume,
            base_url=args.base_url,
            model=args.model,
            api_key=args.api_key,
        )


command = Command()
