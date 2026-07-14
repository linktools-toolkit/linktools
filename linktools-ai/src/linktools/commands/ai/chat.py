#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""`lt ai chat`: interactive agent chat session."""

import asyncio
from pathlib import Path
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

_EXIT_WORDS = {"exit", "quit"}


class Command(BaseCommand):
    """interactive agent chat session"""

    @property
    def known_errors(self) -> "list[type[BaseException]]":
        return super().known_errors + [
            ModelClientUnavailable,
            ModelOutputError,
            ModelTurnLimitExceeded,
        ]

    def init_arguments(self, parser: "CommandParser") -> None:
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
        return asyncio.run(self._chat_async(args))

    async def _chat_async(self, args: "Namespace") -> "int | None":
        runtime = build_runtime(args)
        spec = build_agent_spec(args)
        session_id = validate_session_id(args.session)
        storage = build_storage()
        await ensure_session(storage, session_id)

        workdir = Path(args.workdir) if args.workdir else Path.cwd()
        self.logger.info(f"session: {session_id} (workdir: {workdir})")
        while True:
            try:
                line = await asyncio.to_thread(input, "> ")
            except EOFError:
                break
            line = line.strip()
            if not line:
                continue
            if line in _EXIT_WORDS:
                break
            try:
                await self._run_turn(runtime, spec, session_id, line)
            except asyncio.CancelledError:
                self.logger.warning("turn cancelled")
            except KeyboardInterrupt:
                self.logger.warning("turn cancelled")
        return 0

    async def _run_turn(self, runtime, spec, session_id: str, line: str) -> None:
        async for event in runtime.run_stream(spec, line, session_id=session_id):
            if event["type"] == "text":
                print(event["text"], end="", flush=True)
            elif event["type"] == "tool":
                print(
                    f"\n[tool: {event['name']} {event['phase']}"
                    f"{' ok' if event.get('ok') else ''}]"
                )
        print()


command = Command()
