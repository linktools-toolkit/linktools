#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""`lt ai acp`: start the local ACP stdio Agent."""

from argparse import Namespace
from typing import TYPE_CHECKING

from linktools.cli import BaseCommand, CommandError

from ._common import add_local_runtime_arguments, run_local

if TYPE_CHECKING:
    from linktools.cli import CommandParser
    from ._common import LocalRuntime


class Command(BaseCommand):
    """start ACP for the local Agent runtime"""

    def init_arguments(self, parser: "CommandParser") -> None:
        add_local_runtime_arguments(parser)

    def run(self, args: Namespace) -> int:
        try:
            from linktools.ai.acp import ACPAgent, serve_stdio
        except ModuleNotFoundError as error:
            raise CommandError("ai acp requires the agent-client-protocol dependency") from error

        async def execute(local: "LocalRuntime") -> int:
            await serve_stdio(ACPAgent(local.runtime, principal=local.runtime.default_principal))
            return 0

        return run_local(args, execute)


command = Command()
