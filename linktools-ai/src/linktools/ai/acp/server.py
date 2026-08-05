#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""ACP stdio transport."""

import asyncio
import signal

from .agent import LinktoolsAcpAgent
from .protocol import require_sdk


async def run_acp_server(agent: LinktoolsAcpAgent) -> None:
    acp = require_sdk()
    loop = asyncio.get_running_loop()
    task = asyncio.current_task()
    installed: list[int] = []

    if task is not None:
        def request_shutdown(reason: str) -> None:
            if task is not None:
                task.cancel()

        for name in ("SIGINT", "SIGTERM"):
            signum = getattr(signal, name, None)
            if signum is None:
                continue
            try:
                loop.add_signal_handler(signum, request_shutdown, name.lower())
                installed.append(signum)
            except (NotImplementedError, RuntimeError):
                pass
    try:
        await acp.run_agent(agent, use_unstable_protocol=True)
    finally:
        for signum in installed:
            loop.remove_signal_handler(signum)


async def serve_stdio(agent: LinktoolsAcpAgent) -> None:
    await run_acp_server(agent)


def run_stdio(agent: LinktoolsAcpAgent) -> None:
    asyncio.run(run_acp_server(agent))


__all__ = ["run_acp_server", "run_stdio", "serve_stdio"]
