#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""ACP protocol boundary and stdio transport for local-coding sessions."""


class ACPServer:
    """Forward ACP messages to the explicitly supplied local application."""

    def __init__(self, application: object) -> None:
        self._application = application

    async def handle(self, request: object) -> object:
        return await self._application.handle(request)


async def serve_stdio(agent: object) -> None:
    import acp

    await acp.run_agent(agent, use_unstable_protocol=True)


def run_stdio(agent: object) -> None:
    import asyncio

    asyncio.run(serve_stdio(agent))


__all__ = ["ACPServer", "run_stdio", "serve_stdio"]
