#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""ACP stdio transport entry point."""

import asyncio
import signal
from typing import Any

from .agent import LinktoolsAcpAgent
from .errors import require_sdk


async def serve_stdio(agent: LinktoolsAcpAgent) -> None:
    acp = require_sdk()
    loop = asyncio.get_running_loop()
    task = asyncio.current_task()
    installed = []
    if task is not None:
        for name in ("SIGINT", "SIGTERM"):
            signum = getattr(signal, name, None)
            if signum is None:
                continue
            try:
                loop.add_signal_handler(signum, task.cancel)
                installed.append(signum)
            except (NotImplementedError, RuntimeError):
                pass
    try:
        await acp.run_agent(agent, use_unstable_protocol=True)
    finally:
        for signum in installed:
            loop.remove_signal_handler(signum)
        for session_id in tuple(agent.sessions.active_sessions):
            try:
                await agent.sessions.close(session_id)
            except Exception:
                pass


def run_stdio(agent: LinktoolsAcpAgent) -> None:
    asyncio.run(serve_stdio(agent))


__all__ = ["run_stdio", "serve_stdio"]
