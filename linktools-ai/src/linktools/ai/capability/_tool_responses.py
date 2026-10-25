#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime-wide deterministic tool response configuration."""

from dataclasses import dataclass, field
from typing import Generic, TypeVar

from pydantic_ai.capabilities import AbstractCapability

from ..asset import AssetStoreReader, AssetVersionRef
from ._context import AgentContext
from ..errors import AIError, ErrorCode

AppT = TypeVar("AppT")


@dataclass(frozen=True, slots=True)
class ToolResponseFixture:
    """Bind a default immutable response Asset to a borrowed version reader.

    The Runtime pins ``ref`` at admission. Recovery reads its saved reference
    through the same reader, even when the Runtime's default has changed.
    This object never initializes or closes the reader.
    """

    ref: AssetVersionRef
    reader: AssetStoreReader = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.ref, AssetVersionRef):
            raise TypeError("tool response fixture ref must be AssetVersionRef")
        if not isinstance(self.reader, AssetStoreReader):
            raise TypeError("tool response fixture reader must be AssetStoreReader")


class ToolResponsesCapability(AbstractCapability[AgentContext[AppT]], Generic[AppT]):
    """Use one borrowed response fixture for Runtime execution admission.

    Runtime consumes this declaration before constructing native Agents. Saved
    execution references, rather than this default, select versions on recovery.
    """

    id = "runtime.tool_responses"

    def __init__(self, fixture: ToolResponseFixture) -> None:
        self._fixture = fixture

    def for_agent(self, agent: object) -> "ToolResponsesCapability[AppT]":
        raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID,
                      safe_details={"reason": "response_fixture_requires_host"})

    @property
    def fixture(self) -> ToolResponseFixture:
        return self._fixture


__all__ = ["ToolResponseFixture", "ToolResponsesCapability"]
