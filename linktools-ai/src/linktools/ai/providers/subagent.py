#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SubagentSpecProvider: source-agnostic surface for delegated child agents.

A subagent reuses AgentSpec as its declaration but keeps a dedicated Protocol so
the resolution path can diverge from top-level agents later (scoping, depth
accounting, package-scoped entrypoints). The default adapter,
AgentBackedSubagentSpecProvider, forwards to any AgentSpecProvider so a single
agent registry backs both top-level and delegated agents out of the box."""

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from ..agent.spec import AgentSpec
    from .agent import AgentSpecProvider


@runtime_checkable
class SubagentSpecProvider(Protocol):
    """Provides AgentSpec declarations usable as delegated subagents."""

    async def list_ids(self) -> "tuple[str, ...]":
        ...

    async def get(self, agent_id: str) -> "AgentSpec":
        ...


class AgentBackedSubagentSpecProvider:
    """Adapts an AgentSpecProvider into a SubagentSpecProvider. Subagents share
    the same declaration store as top-level agents unless a caller wires a
    separate, narrower provider."""

    def __init__(self, agents: "AgentSpecProvider") -> None:
        self._agents = agents

    async def list_ids(self) -> "tuple[str, ...]":
        return await self._agents.list_ids()

    async def get(self, agent_id: str) -> "AgentSpec":
        return await self._agents.get(agent_id)
