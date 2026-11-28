#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime-scoped native step persistence declaration."""

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai_harness.step_persistence import StepStore

from ..capability import AgentContext
from ..capability.steps import AtomicStepPersistence
from ..errors import AIError, ErrorCode


class _StepPersistenceCapability(AbstractCapability[AgentContext[object]]):
    """Unbound Runtime-owned execution checkpoint hooks."""

    id = "runtime.steps"

    def for_agent(self, agent: object) -> "_StepPersistenceCapability":
        raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID,
                      safe_details={"reason": "step_declaration_requires_runtime"})

    def bind(self, store: StepStore) -> AtomicStepPersistence[AgentContext[object]]:
        return AtomicStepPersistence(id=self.id, store=store, capture_frontier=True)
