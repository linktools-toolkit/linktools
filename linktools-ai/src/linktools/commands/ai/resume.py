#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""`lt ai resume`: resume a run paused waiting for approval.

``Runtime.resume`` restores the ORIGINAL agent spec + identity from the
persisted RunDefinitionSnapshot, so this command passes only the run id --
never an AgentSpec. A second pause during resume again exits 4."""

import asyncio
from typing import TYPE_CHECKING

from linktools.ai.errors import (
    InvalidRunTransitionError,
    RunConflictError,
    RunNotFoundError,
)
from linktools.cli import BaseCommand

from .assembly import build_project_bundle
from .support import announce_paused

if TYPE_CHECKING:
    from argparse import Namespace

    from linktools.cli import CommandParser


class Command(BaseCommand):
    """resume a paused run"""

    @property
    def known_errors(self) -> "list[type[BaseException]]":
        return super().known_errors + [
            RunNotFoundError,
            InvalidRunTransitionError,
            RunConflictError,
        ]

    def init_arguments(self, parser: "CommandParser") -> None:
        parser.add_argument("run_id", help="the run id to resume")
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
            "--workdir",
            default=None,
            help="agent working directory (default: current directory)",
        )

    def run(self, args: "Namespace") -> "int | None":
        bundle = build_project_bundle(args)
        return asyncio.run(self._resume_async(bundle, args.run_id))

    async def _resume_async(self, bundle, run_id: str) -> "int | None":
        try:
            async for event in bundle.runtime.resume(run_id):
                kind = event["type"]
                if kind == "resumed":
                    self.logger.info(f"resumed run: {event.get('run_id')}")
                elif kind == "text":
                    print(event["text"], end="", flush=True)
                elif kind == "tool":
                    self.logger.info(
                        f"[tool: {event['name']} {event['phase']}"
                        f"{' ok' if event.get('ok') else ''}]"
                    )
                elif kind == "paused":
                    # Paused again after resume: surface and exit 4.
                    await announce_paused(bundle.storage, event, self.logger)
                    return 4
            print()
            return 0
        except asyncio.CancelledError:
            await bundle.runtime.cancel(run_id)
            self.logger.warning("resume cancelled")
            return 130


command = Command()
