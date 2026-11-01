#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""`lt ai web`: serve a local Runtime console."""

import asyncio
from argparse import Namespace
from typing import TYPE_CHECKING

from linktools.cli import BaseCommand, CommandError
from ._common import add_local_runtime_arguments, run_local

if TYPE_CHECKING:
    from linktools.cli import CommandParser
    from ._common import LocalRuntime


class Command(BaseCommand):
    """Open one local Web console for conversations, executions, history and metrics."""

    def init_arguments(self, parser: "CommandParser") -> None:
        add_local_runtime_arguments(parser)
        parser.add_argument("--port", type=int, default=8765, help="loopback HTTP port (default: 8765)")
        parser.add_argument("--open", action="store_true", help="open the console in your browser")

    def run(self, args: Namespace) -> int:
        try:
            import uvicorn
            from linktools.ai.web import create_app
        except ModuleNotFoundError as error:
            raise CommandError("ai web requires linktools-ai[web]") from error
        if not 1 <= args.port <= 65535:
            raise CommandError("port must be between 1 and 65535")

        async def serve(local: "LocalRuntime") -> int:
            app = create_app(
                runtime=local.runtime, metrics=local.metrics,
                status=local.status, capabilities=local.capabilities,
            )
            url = f"http://127.0.0.1:{args.port}"
            self.logger.info("AI Web console: %s", url)
            server = uvicorn.Server(uvicorn.Config(
                app, host="127.0.0.1", port=args.port, log_config=None,
                access_log=False, proxy_headers=False, timeout_graceful_shutdown=5,
            ))

            async def open_browser() -> None:
                import webbrowser
                while not server.started:
                    await asyncio.sleep(0.05)
                await asyncio.to_thread(webbrowser.open, url)

            browser_task = asyncio.create_task(open_browser()) if args.open else None
            try:
                await server.serve()
            finally:
                if browser_task is not None:
                    browser_task.cancel()
                    await asyncio.gather(browser_task, return_exceptions=True)
            return 0

        return run_local(args, serve)


command = Command()
