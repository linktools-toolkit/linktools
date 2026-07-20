#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SubagentResult: the structured return value of a delegated
subagent run. Carries the child session/run ids, terminal status, output or
structured error, and accounting fields."""

from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from ..agent.spec import AgentSpec, AgentSpecProvider

SubagentStatus = Literal["succeeded", "failed", "cancelled"]


class SubagentResult(BaseModel):
    agent_id: str
    scope: "dict[str, Any] | None" = None
    session_id: str
    run_id: str
    status: SubagentStatus
    output: Any = None
    error: "dict[str, Any] | None" = None
    token_usage: "dict[str, Any] | None" = None
    duration_ms: "int | None" = None
    metadata: "dict[str, Any]" = Field(default_factory=dict)


@runtime_checkable
class SubagentSpecProvider(Protocol):
    """Provides AgentSpec declarations usable as delegated subagents. A
    subagent reuses AgentSpec as its declaration but keeps a dedicated Protocol
    so the resolution path can diverge from top-level agents later (scoping,
    depth accounting, extension-scoped entrypoints)."""

    async def list_ids(self) -> "tuple[str, ...]": ...

    async def get(self, agent_id: str) -> "AgentSpec": ...


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
