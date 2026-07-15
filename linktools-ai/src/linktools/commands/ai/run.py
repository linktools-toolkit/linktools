#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""`lt ai run`: run one Agent task without the TUI.

Thin shell -- declares the arguments and delegates to
:func:`linktools.ai_cli.console.run_once.run_once`, which owns the streaming,
exit-code (0/4/130) and Ctrl+C-cancel semantics."""

from typing import TYPE_CHECKING

from linktools.ai_cli.errors import (
    InvalidRunTransitionError,
    ModelClientUnavailable,
    ModelOutputError,
    ModelTurnLimitExceeded,
    RunConflictError,
    RunNotFoundError,
)
from linktools.cli import BaseCommand

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
            "--base-url", default=None, help="OpenAI-compatible base url"
        )
        parser.add_argument("--model", default=None, help="model name")
        parser.add_argument("--api-key", default=None, help="api key")
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
