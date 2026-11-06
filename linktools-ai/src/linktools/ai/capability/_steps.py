#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime-scoped native step persistence declaration."""

from collections.abc import Mapping
from types import MappingProxyType
from typing import Generic, TypeVar

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai_harness.step_persistence import StepStore

from ..errors import AIError, ErrorCode
from ._context import AgentContext
from .steps import AtomicStepPersistence

AppT = TypeVar("AppT")


class StepPersistenceCapability(AbstractCapability[AgentContext[AppT]], Generic[AppT]):
    """Persist native steps using Runtime's Execution SQL storage by default.

    Runtime owns the default adapter; an explicit store remains caller-owned.
    Memory-only deployments must explicitly supply an ephemeral StepStore.
    Native capture and recovery hooks belong to the bound native capability.
    """

    id = "runtime.steps"

    def __init__(
        self, *, store: StepStore | None = None, capture_frontier: bool = True,
        metadata: Mapping[str, str] | None = None,
    ) -> None:
        if not isinstance(capture_frontier, bool):
            raise TypeError("capture_frontier must be bool")
        self.store = store
        self.capture_frontier = capture_frontier
        self.metadata = MappingProxyType({} if metadata is None else dict(metadata))

    def for_agent(self, agent: object) -> "StepPersistenceCapability[AppT]":
        raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID,
                      safe_details={"reason": "step_declaration_requires_runtime"})

    def bind(self, store: StepStore) -> AtomicStepPersistence[AgentContext[AppT]]:
        """Bind Runtime's selected store without copying native hook semantics."""
        return AtomicStepPersistence(
            id=self.id, store=store, capture_frontier=self.capture_frontier,
            metadata=dict(self.metadata),
        )


__all__ = ["StepPersistenceCapability"]
