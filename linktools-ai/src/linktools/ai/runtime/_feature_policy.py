#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Resolve current host capability declarations without owning their resources."""

from collections.abc import Sequence
from copy import copy
from dataclasses import dataclass

from pydantic_ai.capabilities import AbstractCapability, WrapperCapability
from pydantic_ai_harness.step_persistence import StepPersistence

from ..capability import (
    AgentContext, AssetCapability, CapabilityContribution, ExecutionCapability,
    MemoryCapability, SessionCapability, StepPersistenceCapability,
    ToolResponseFixture, ToolResponsesCapability, TraceCapability,
)
from ..errors import AIError, ErrorCode
from ..persistence.capabilities import CapabilityPersistence
from ._capabilities import runtime_feature_leaves
from ._persistence_configuration import validate_persistence_capabilities, capability_persistence_ids


@dataclass(frozen=True, slots=True)
class _RuntimeFeaturePolicy:
    capabilities: tuple[AbstractCapability[AgentContext[object]], ...]
    execution: ExecutionCapability
    session: SessionCapability
    memory_scope: str | None
    tool_responses: ToolResponseFixture | None
    capability_revisions: dict[str, int]
    capability_ids: tuple[str, ...]

    @property
    def restart_checkpoints(self) -> bool:
        return any(isinstance(leaf, (StepPersistence, StepPersistenceCapability))
                   for item in self.capabilities for leaf in runtime_feature_leaves(item))

    @property
    def durable_trace(self) -> bool:
        return any(isinstance(leaf, TraceCapability)
                   for item in self.capabilities for leaf in runtime_feature_leaves(item))


def _runtime_feature_policy(
    candidates: Sequence[CapabilityContribution[object]],
) -> _RuntimeFeaturePolicy:
    execution_capability = ExecutionCapability()
    execution_declared = False
    session_capability = SessionCapability()
    session_declared = False
    memory_scope: str | None = None
    tool_responses: ToolResponseFixture | None = None
    def require_runtime_feature_scope(capability: AbstractCapability[object]) -> None:
        if isinstance(capability, CapabilityPersistence):
            raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID, "Persistent capabilities require runtime scope")
        if isinstance(capability, (
            StepPersistence,
            StepPersistenceCapability,
            TraceCapability,
            ToolResponsesCapability,
            SessionCapability,
            MemoryCapability,
            ExecutionCapability,
        )):
            raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID, "Runtime feature capabilities require Runtime scope")

    for candidate in candidates:
        if candidate.kind == "capability" and candidate.scope != "runtime":
            for leaf in runtime_feature_leaves(candidate.value):
                require_runtime_feature_scope(leaf)
    declared_ids: set[str] = set()
    for candidate in candidates:
        if candidate.scope != "runtime":
            continue
        for leaf in runtime_feature_leaves(candidate.value):
            if leaf.id is None:
                continue
            if leaf.id in declared_ids:
                raise AIError(ErrorCode.CAPABILITY_CONFLICT)
            declared_ids.add(leaf.id)

    def consume_runtime_declaration(capability: AbstractCapability[AgentContext[object]]) -> AbstractCapability[AgentContext[object]] | None:
        nonlocal tool_responses, session_capability, session_declared, execution_capability, execution_declared, memory_scope
        if isinstance(capability, ExecutionCapability):
            if execution_declared:
                raise AIError(ErrorCode.CAPABILITY_CONFLICT)
            execution_capability, execution_declared = capability, True
            return capability
        if isinstance(capability, MemoryCapability):
            if memory_scope is not None:
                raise AIError(ErrorCode.CAPABILITY_CONFLICT)
            memory_scope = capability.scope
            return None
        if isinstance(capability, SessionCapability):
            if session_declared:
                raise AIError(ErrorCode.CAPABILITY_CONFLICT)
            session_capability, session_declared = capability, True
            return None
        if isinstance(capability, ToolResponsesCapability):
            if tool_responses is not None:
                raise AIError(ErrorCode.CAPABILITY_CONFLICT)
            if not isinstance(capability.fixture, ToolResponseFixture):
                raise TypeError("fixture must be ToolResponseFixture")
            tool_responses = capability.fixture
            return None
        if isinstance(capability, AssetCapability):
            return None
        if isinstance(capability, WrapperCapability):
            wrapped = capability.wrapped.visit_and_replace(consume_runtime_declaration)
            if wrapped is None:
                return None
            rebound = copy(capability)
            rebound.wrapped = wrapped
            return rebound
        return capability

    captured_persistence_ids: set[str] = set()
    capability_revisions: dict[str, int] = {}
    runtime_capability_values = []
    for candidate in candidates:
        if candidate.scope != "runtime":
            continue

        def capture_persistence(capability: AbstractCapability[AgentContext[object]]) -> AbstractCapability[AgentContext[object]] | None:
            if isinstance(capability, WrapperCapability):
                wrapped = capability.wrapped.visit_and_replace(capture_persistence)
                if wrapped is None:
                    return None
                rebound = copy(capability)
                rebound.wrapped = wrapped
                return rebound
            selected = consume_runtime_declaration(capability)
            if selected is None or not isinstance(selected, CapabilityPersistence):
                return selected
            identity = selected.id
            if identity is None:
                if capability is not candidate.value:
                    raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID, "Persistent capability leaves require a registered identity")
                identity = candidate.id
            if identity in captured_persistence_ids:
                raise AIError(ErrorCode.CAPABILITY_CONFLICT)
            captured_persistence_ids.add(identity)
            capability_revisions[identity] = candidate.revision
            bound = copy(selected)
            bound.id = identity
            return bound

        selected = candidate.value.visit_and_replace(capture_persistence)
        if selected is not None:
            runtime_capability_values.append(selected)
    if not execution_declared:
        runtime_capability_values.append(execution_capability)
    runtime_capabilities = tuple(runtime_capability_values)
    runtime_ids: set[str] = set()

    def register_runtime_id(capability: AbstractCapability[AgentContext[object]]) -> None:
        if capability.id is not None:
            if capability.id in runtime_ids:
                raise AIError(ErrorCode.CAPABILITY_CONFLICT)
            runtime_ids.add(capability.id)

    for capability in runtime_capabilities:
        capability.apply(register_runtime_id)
    validate_persistence_capabilities(runtime_capabilities)
    capability_ids = capability_persistence_ids(
        tuple((capability.id or "", capability) for capability in runtime_capabilities),
    )
    return _RuntimeFeaturePolicy(
        runtime_capabilities, execution_capability, session_capability,
        memory_scope, tool_responses, capability_revisions, tuple(capability_ids),
    )
