#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime-scoped session management declaration."""

from pydantic_ai.capabilities import AbstractCapability

from ._context import AgentContext


class SessionCapability(AbstractCapability[AgentContext[object]]):
    """Configure the default session owner using Runtime's storage resources.

    One explicit declaration replaces the default. Disabling session management
    leaves independent executions available without exposing session operations.
    """

    id = "runtime.session"

    def __init__(self, *, enabled: bool = True) -> None:
        if not isinstance(enabled, bool):
            raise TypeError("enabled must be bool")
        self._enabled = enabled

    @property
    def enabled(self) -> bool:
        return self._enabled


__all__ = ["SessionCapability"]
