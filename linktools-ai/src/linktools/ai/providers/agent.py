#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AgentSpecProvider: the configuration-source-agnostic surface a Runtime
consumes to obtain AgentSpec objects. Any backend -- file registry, DB, config
center, HTTP API -- can implement it; the Runtime never imports a concrete
registry."""

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from ..agent.spec import AgentSpec


@runtime_checkable
class AgentSpecProvider(Protocol):
    """Provides AgentSpec objects from any configuration source."""

    async def list_ids(self) -> "tuple[str, ...]":
        ...

    async def get(self, agent_id: str) -> "AgentSpec":
        ...
