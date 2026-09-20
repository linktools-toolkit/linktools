#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""`lt ai acp`: start the local ACP stdio Agent."""

from argparse import Namespace
from typing import TYPE_CHECKING

from linktools.cli import BaseCommand, CommandError

from linktools.ai.acp import ACPAgent, serve_stdio

from ._common import (
    _add_local_runtime_arguments,
    _load_workspace,
    _local_runtime_models,
    _open_local_runtime,
    _run_async,
)

if TYPE_CHECKING:
    from linktools.cli import CommandParser


class Command(BaseCommand):
    """start ACP for the local Agent runtime"""

    def init_arguments(self, parser: "CommandParser") -> None:
        _add_local_runtime_arguments(parser)

    def run(self, args: Namespace) -> int:
        workspace = _load_workspace(args.project)
        models = _local_runtime_models(args)
        memory_scope = args.memory if args.memory is not None else "default"

        async def execute() -> int:
            async with _open_local_runtime(workspace, models=models) as runtime:
                await serve_stdio(
                    ACPAgent(
                        runtime,
                        principal=runtime.default_principal,
                        memory_scope=memory_scope,
                    )
                )
            return 0

        try:
            return _run_async(execute())
        except ModuleNotFoundError as error:
            raise CommandError(
                "ai acp requires the agent-client-protocol dependency"
            ) from error


command = Command()
