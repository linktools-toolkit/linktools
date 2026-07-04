#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""`lt ai chat`: interactive local agent REPL."""

import asyncio
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
from linktools.ai.core.registry import AgentSpec
from linktools.ai.core.runtime import AgentKernel
from linktools.ai.skill.registry import SkillRegistry
from linktools.ai.subagent.registry import SubagentRegistry
from linktools.ai.mcp.registry import MCPRegistry
from linktools.ai.session.types import FileSessionSpec, Session
from linktools.ai.agent import RuntimeAgent

from .support import resolve_model_config

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
        model_registry.register(config.model_type, config)

        skill_registry = SkillRegistry()
        subagent_registry = SubagentRegistry()
        mcp_registry = MCPRegistry()
        await asyncio.gather(
            skill_registry.preload(),
            subagent_registry.preload(),
            mcp_registry.preload(),
        )

        spec_dir = environ.get_data_path("ai", "spec", "chat", create_parent=True) / "agent.md"
        spec = AgentSpec(
            name="ai",
            path=spec_dir,
            base_dir=None,
            enabled=True,
            model=config.model_type,
            allowed_tools=["file", "terminal"],
            allowed_skills=[],
            allowed_subagents=[],
            system_prompt=_SYSTEM_PROMPT,
        )

        kernel = AgentKernel(
            skill_registry=skill_registry,
            subagent_registry=subagent_registry,
            mcp_registry=mcp_registry,
        )
        session_root = environ.get_data_path("ai", "sessions", args.session, create_parent=True)
        session = Session.create(session_root, FileSessionSpec(session_id=args.session))
        context = kernel.build_context(
            spec, session, builtin_tool_names=frozenset({"file", "terminal"}),
        )
        agent = RuntimeAgent(
            spec, session, execution_context=context,
            workdir=Path(args.workdir) if args.workdir else Path.cwd(),
        )

        self.logger.info(f"session: {args.session} (workdir: {agent.workdir})")
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
                await self._run_turn(agent, line)
            except asyncio.CancelledError:
                self.logger.warning("turn cancelled")
            except KeyboardInterrupt:
                self.logger.warning("turn cancelled")
        return 0

    async def _run_turn(self, agent: RuntimeAgent, line: str) -> None:
        async for event in agent.stream({"question": line}):
            if event["type"] == "text":
                print(event["text"], end="", flush=True)
            elif event["type"] == "tool":
                print(f"\n[tool: {event['name']} {event['phase']}"
                      f"{' ok' if event.get('ok') else ''}]")
        print()


command = Command()
if __name__ == "__main__":
    command.main()
