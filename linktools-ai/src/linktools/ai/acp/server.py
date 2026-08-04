#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""ACP stdio transport entry point."""

import asyncio
import logging
import signal

from .agent import LinktoolsAcpAgent
from .errors import require_sdk
from .session_models import CloseReason

logger = logging.getLogger("linktools.ai.acp.server")


async def serve_stdio(agent: LinktoolsAcpAgent) -> None:
    acp = require_sdk()
    loop = asyncio.get_running_loop()
    task = asyncio.current_task()
    shutdown_reason: CloseReason = "eof"
    installed = []
    if task is not None:
        def request_shutdown(reason: str) -> None:
            nonlocal shutdown_reason
            shutdown_reason = reason
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
        failures = []
        for session_id in tuple(agent.sessions.active_sessions):
            try:
                result = await agent.sessions.close_session_resources(
                    session_id,
                    reason=shutdown_reason,
                )
                if not result.closed:
                    failures.extend(result.failures)
            except Exception as exc:
                logger.error(
                    "ACP shutdown cleanup failed session=%s error_type=%s",
                    session_id,
                    type(exc).__name__,
                )
                failures.append(exc)
        if failures:
            logger.error("ACP shutdown left %s cleanup failures", len(failures))
            raise RuntimeError("ACP session cleanup failed")


def run_stdio(agent: LinktoolsAcpAgent) -> None:
    asyncio.run(serve_stdio(agent))


__all__ = ["run_stdio", "serve_stdio"]
