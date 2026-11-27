#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime-scoped session management declaration."""

from pydantic_ai.capabilities import AbstractCapability

from ._context import AgentContext
from ..errors import AIError, ErrorCode


class SessionCapability(AbstractCapability[AgentContext[object]]):
    """Configure the default session owner using Runtime's storage resources.

    One explicit declaration replaces the default. Disabling session management
    leaves independent executions available without exposing session operations.
    """

    id = "runtime.session"

    def __init__(self, *, enabled: bool = True, checkpoints: bool = False) -> None:
        if not isinstance(enabled, bool):
            raise TypeError("enabled must be bool")
        if not isinstance(checkpoints, bool):
            raise TypeError("checkpoints must be bool")
        if checkpoints and not enabled:
            raise ValueError("session checkpoints require enabled sessions")
        self._enabled = enabled
        self._checkpoints = checkpoints

    def for_agent(self, agent: object) -> "SessionCapability":
        raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID,
                      safe_details={"reason": "session_declaration_requires_host"})

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def checkpoints(self) -> bool:
        """Enable native Step for ordinary SQL session turns only."""
        return self._checkpoints


__all__ = ["SessionCapability"]
