#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Subagent launch protocol."""

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class SubagentRunRequest:
    parent_execution_id: str
    agent_id: str
    prompt: str


@dataclass(frozen=True, slots=True)
class SubagentRunResult:
    execution_id: str
    output: str


class RunLauncher(Protocol):
    async def launch(self, request: SubagentRunRequest) -> SubagentRunResult: ...


class SubagentProvider(Protocol):
    def manifest(self) -> str: ...
    def resolve_ref(self, agent_id: str, revision: 'int | None' = None) -> str: ...
    async def launcher(self) -> RunLauncher: ...


class AgentBackedSubagentProvider(SubagentProvider, Protocol):
    pass


__all__ = ["AgentBackedSubagentProvider", "RunLauncher", "SubagentProvider", "SubagentRunRequest", "SubagentRunResult"]
