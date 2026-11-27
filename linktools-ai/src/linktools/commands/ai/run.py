#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""`ai run`: execute one workspace Agent through Runtime."""

from argparse import BooleanOptionalAction, Namespace
from typing import TYPE_CHECKING

from linktools.cli import BaseCommand
from ._common import add_local_runtime_arguments, local_turn_notices, run_local
from ._output import emit_result

if TYPE_CHECKING:
    from linktools.cli import CommandParser
    from ._common import LocalRuntime


class Command(BaseCommand):
    """Run a prompt against Agent specs in the current workspace."""

    def init_arguments(self, parser: "CommandParser") -> None:
        parser.add_argument("prompt", help="the prompt")
        add_local_runtime_arguments(parser)
        parser.add_argument("--planning", action="store_true", help="enable planning for this execution")
        parser.add_argument(
            "--thinking", action=BooleanOptionalAction, default=None,
            help="override inherited thinking (--no-thinking requests off or the lowest effort)",
        )
        parser.add_argument("--json", action="store_true", help="emit one final JSON result")

    def run(self, args: Namespace) -> int:
        async def execute(local: "LocalRuntime") -> int:
            from linktools.ai.runtime.turn_clients import uses_turn_sessions

            async def run_prompt() -> int:
                return await emit_result(local.runtime, args.prompt, "default", args.json, args.planning, args.thinking)

            if not args.planning and uses_turn_sessions(local.runtime):
                async with local_turn_notices(local):
                    return await run_prompt()
            return await run_prompt()

        return run_local(args, execute)


command = Command()
