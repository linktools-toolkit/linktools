#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime-wide deterministic tool response configuration."""

from typing import TYPE_CHECKING, Generic, TypeVar

from pydantic_ai.capabilities import AbstractCapability

from ._context import AgentContext

if TYPE_CHECKING:
    from ..runtime import ToolResponseFixture

AppT = TypeVar("AppT")


class ToolResponsesCapability(AbstractCapability[AgentContext[AppT]], Generic[AppT]):
    """Use one borrowed response fixture for Runtime execution admission.

    Runtime consumes this declaration before constructing native Agents. Saved
    execution references, rather than this default, select versions on recovery.
    """

    id = "runtime.tool_responses"

    def __init__(self, fixture: "ToolResponseFixture") -> None:
        self._fixture = fixture

    @property
    def fixture(self) -> "ToolResponseFixture":
        return self._fixture


__all__ = ["ToolResponsesCapability"]
