#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""`lt ai web`: serve a local Runtime console."""

import asyncio
import signal
from argparse import Namespace
from typing import TYPE_CHECKING

from linktools.cli import BaseCommand, CommandError
from ._common import add_local_runtime_arguments, local_turn_notices, run_local

if TYPE_CHECKING:
    from linktools.cli import CommandParser
    from ._common import LocalRuntime
    from socket import socket
    from types import FrameType


class Command(BaseCommand):
    """Open one local Web console for conversations, executions, history and metrics."""

    def init_arguments(self, parser: "CommandParser") -> None:
        add_local_runtime_arguments(parser)
        parser.add_argument("--port", type=int, default=8765, help="loopback HTTP port (default: 8765)")
        parser.add_argument("--open", action="store_true", help="open the console in your browser")
        parser.add_argument(
            "--ordinary", action="store_true",
            help="use low-I/O conversations without tools, memory, restart checkpoints or trace",
        )

    def run(self, args: Namespace) -> int:
        try:
            import uvicorn
            from linktools.ai.web import create_app
        except ModuleNotFoundError as error:
            raise CommandError("ai web requires linktools-ai[web]") from error
        if not 1 <= args.port <= 65535:
            raise CommandError("port must be between 1 and 65535")

        async def serve(local: "LocalRuntime") -> int:
            shutdown_event = asyncio.Event()

            class Server(uvicorn.Server):
                interrupted = False

                def handle_exit(self, sig: int, frame: "FrameType | None") -> None:
                    if sig == signal.SIGINT:
                        self.interrupted = True
                    super().handle_exit(sig, frame)

                async def shutdown(self, sockets: "list[socket] | None" = None) -> None:
                    # Uvicorn waits for HTTP streams before dispatching ASGI lifespan shutdown.
                    shutdown_event.set()
                    await super().shutdown(sockets=sockets)

            async with local_turn_notices(local, receive=True) as notices:
                app = create_app(
                    runtime=local.runtime, metrics=local.metrics,
                    status=local.status, capabilities=local.capabilities, shutdown_event=shutdown_event,
                    session_notices=notices, local_workspace=True, ordinary_turns=args.ordinary,
                )
                url = f"http://127.0.0.1:{args.port}"
                self.logger.info("AI Web console: %s", url)
                server = Server(uvicorn.Config(
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
            # Uvicorn may consume Ctrl-C after graceful shutdown instead of
            # letting the base CLI translate a replayed KeyboardInterrupt.
            return 128 + signal.SIGINT if server.interrupted else 0

        return run_local(args, serve, ordinary=args.ordinary)


command = Command()
