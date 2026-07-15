#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""`lt ai run`: run one Agent task without the TUI."""

from typing import TYPE_CHECKING

from linktools.ai_cli.errors import (
    InvalidRunTransitionError,
    ModelClientUnavailable,
    ModelOutputError,
    ModelTurnLimitExceeded,
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
    """run one Agent task without the TUI"""

    @property
    def known_errors(self) -> "list[type[BaseException]]":
        return super().known_errors + [
            ModelClientUnavailable,
            ModelOutputError,
            ModelTurnLimitExceeded,
            RunNotFoundError,
            InvalidRunTransitionError,
            RunConflictError,
        ]

    def init_arguments(self, parser: "CommandParser") -> None:
        parser.add_argument("prompt", nargs="?", help="the prompt")
        parser.add_argument(
            "--agent", default=None, help="agent id (default: project default)"
        )
        parser.add_argument(
            "--session", default="main", help="session id (default main)"
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
        parser.add_argument(
            "--json", action="store_true", help="emit one JSON event per line"
        )

    def run(self, args: "Namespace") -> "int | None":
        from linktools.ai_cli.console.run_once import run_once

        return run_once(
            prompt=args.prompt,
            agent=args.agent,
            session=args.session,
            base_url=args.base_url,
            model=args.model,
            api_key=args.api_key,
            json_output=args.json,
        )


command = Command()
