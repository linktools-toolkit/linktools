#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Subagent launch protocol."""

from dataclasses import dataclass
from typing import Protocol

from ..core import ErrorCode, AIError
from ..core import ResourceRef
from ..core import Principal


@dataclass(frozen=True, slots=True)
class SubagentRunRequest:
    parent_execution_id: str
    agent_id: str
    prompt: str
    principal: Principal
    parent: ResourceRef
    operation_id: str
    depth: int = 0
    max_depth: int = 8
    timeout_seconds: int = 300

    def __post_init__(self) -> None:
        if self.parent.id != self.parent_execution_id or self.parent.kind.value != "EXECUTION" or self.parent.tenant_id != self.principal.tenant_id or not self.operation_id.strip() or not 0 <= self.depth <= self.max_depth or not 1 <= self.timeout_seconds <= 900:
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)


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


__all__ = ["RunLauncher", "SubagentProvider", "SubagentRunRequest", "SubagentRunResult"]
