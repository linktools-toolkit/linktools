#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""`lt ai run`: run the agent against a single prompt and print the result."""

import asyncio
from typing import TYPE_CHECKING

from linktools.ai.model.registry import (
    ModelClientUnavailable,
    ModelOutputError,
    ModelTurnLimitExceeded,
)
from linktools.cli import BaseCommand

from .support import (
    build_agent_spec,
    build_runtime,
    build_storage,
    ensure_session,
    validate_session_id,
)

if TYPE_CHECKING:
    from argparse import Namespace

    from linktools.cli import CommandParser


class Command(BaseCommand):
    """run agent with a single prompt"""

    @property
    def known_errors(self) -> "list[type[BaseException]]":
        return super().known_errors + [
            ModelClientUnavailable,
            ModelOutputError,
            ModelTurnLimitExceeded,
        ]

    def init_arguments(self, parser: "CommandParser") -> None:
        parser.add_argument("prompt", help="the prompt")
        parser.add_argument(
            "--model", default=None, help="model name (default: $OPENAI_MODEL)"
        )
        parser.add_argument(
            "--base-url",
            default=None,
            help="OpenAI-compatible base url (default: $OPENAI_BASE_URL)",
        )
        parser.add_argument(
            "--api-key", default=None, help="api key (default: $OPENAI_API_KEY)"
        )
        parser.add_argument(
            "--session", default="main", help="session id (default main)"
        )
        parser.add_argument(
            "--workdir",
            default=None,
            help="agent working directory (default: current directory)",
        )

    def run(self, args: "Namespace") -> "int | None":
        return asyncio.run(self._run_async(args, args.prompt))

    async def _run_async(self, args: "Namespace", prompt: str) -> "int | None":
        runtime = build_runtime(args)
        spec = build_agent_spec(args)
        session_id = validate_session_id(args.session)
        storage = build_storage()
        await ensure_session(storage, session_id)
        result = await runtime.run(spec, prompt, session_id=session_id)
        print(result.output)
        return 0


command = Command()
