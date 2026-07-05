#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""`lt ai chat`: interactive local agent REPL."""

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from linktools.cli import BaseCommand
from linktools.core import environ
from linktools.ai.core.model_runtime import (
    ModelClientUnavailable,
    ModelOutputError,
    ModelTurnLimitExceeded,
    model_registry,
)
from linktools.ai.runtime import Runtime
from linktools.ai.storage.facade import FileStorage
from linktools.ai.agent.spec import AgentSpec, PromptSpec
from linktools.ai.model.policy import ModelPolicy
from linktools.ai.model.router import ModelRouter
from linktools.ai.session.models import SessionRecord, SessionStatus

from .support import resolve_model_config, validate_session_id

if TYPE_CHECKING:
    from argparse import Namespace
    from linktools.cli import CommandParser

_EXIT_WORDS = {"exit", "quit"}

_SYSTEM_PROMPT = (
    "You are a general-purpose local assistant running in a terminal. "
    "You can read/write files and run shell commands in the current working "
    "directory via your file and terminal tools. Be direct and concise."
)


class Command(BaseCommand):
    """
    Run an interactive local agent chat session
    """

    @property
    def known_errors(self) -> "list[type[BaseException]]":
        return super().known_errors + [
            ModelClientUnavailable, ModelOutputError, ModelTurnLimitExceeded,
        ]

    def init_arguments(self, parser: "CommandParser") -> None:
        parser.add_argument("--model", default=None, help="model name (default: $OPENAI_MODEL)")
        parser.add_argument("--base-url", default=None, help="OpenAI-compatible base url (default: $OPENAI_BASE_URL)")
        parser.add_argument("--api-key", default=None, help="api key (default: $OPENAI_API_KEY)")
        parser.add_argument("--session", default="main", help="session id (default: main)")
        parser.add_argument("--workdir", default=None, help="agent working directory (default: current directory)")

    def run(self, args: "Namespace") -> "int | None":
        return asyncio.run(self._run_async(args))

    async def _run_async(self, args: "Namespace") -> "int | None":
        config = resolve_model_config(args.model, args.base_url, args.api_key)
        model_registry.register(config.model_type, config=config)
        session_id = validate_session_id(args.session)
        workdir = Path(args.workdir) if args.workdir else Path.cwd()
        storage = FileStorage(root=environ.get_data_path("ai"))
        runtime = Runtime.build(
            storage=storage,
            model_router=ModelRouter(registry=model_registry),
            workdir=workdir,
        )
        spec = AgentSpec(
            id="ai", name="ai",
            model=ModelPolicy(primary=config.model_type),
            instructions=PromptSpec(instructions=_SYSTEM_PROMPT),
            output_schema=str,
        )
        # Runtime.run_stream requires a pre-existing Session when session_id is
        # provided (it does not auto-create). The legacy chat path created the
        # session directory implicitly; mirror that by creating the SessionRecord
        # up-front when validate_session_id's id is unseen. This get-or-create
        # mirrors run_stream's own session_id=None branch exactly.
        if await storage.sessions.get(session_id) is None:
            now = datetime.now(timezone.utc)
            await storage.sessions.create(SessionRecord(
                id=session_id, parent_id=None, status=SessionStatus.ACTIVE,
                version=1, created_at=now, updated_at=now,
            ))

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

    async def _run_turn(self, runtime: Runtime, spec: AgentSpec,
                        session_id: str, line: str) -> None:
        async for event in runtime.run_stream(spec, line, session_id=session_id):
            if event["type"] == "text":
                print(event["text"], end="", flush=True)
            elif event["type"] == "tool":
                print(f"\n[tool: {event['name']} {event['phase']}"
                      f"{' ok' if event.get('ok') else ''}]")
        print()


command = Command()
if __name__ == "__main__":
    command.main()
