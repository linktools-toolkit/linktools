#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime-scoped memory declaration."""

from pydantic_ai.capabilities import AbstractCapability

from ..core import validate_memory_scope
from ..errors import AIError, ErrorCode
from ._context import AgentContext


class MemoryCapability(AbstractCapability[AgentContext[object]]):
    """Select memory for new executions; admitted executions retain their scope.

    Runtime owns storage and exposes only memory tools allowed by the Agent.
    Imported input contexts use their own isolated memory instead of this scope.
    Omitting this declaration leaves ordinary execution memory disabled.
    """

    id = "runtime.memory"

    def __init__(self, *, scope: str = "default") -> None:
        self._scope = validate_memory_scope(scope)

    @property
    def scope(self) -> str:
        return self._scope

    def for_agent(self, agent: object) -> "MemoryCapability":
        raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID,
                      safe_details={"reason": "memory_declaration_requires_host"})


__all__ = ["MemoryCapability"]
