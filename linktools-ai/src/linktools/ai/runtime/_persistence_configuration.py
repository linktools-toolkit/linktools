#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared persistence declarations for active and read-only Runtime assembly."""

from collections.abc import Sequence

from pydantic_ai.capabilities import AbstractCapability, CombinedCapability, WrapperCapability

from ..errors import AIError, ErrorCode
from ..core import CommitPolicy
from ..capability import CheckpointCapability, ExecutionCapability, TraceCapability
from linktools.ai.persistence import PersistenceDomain
from linktools.ai.persistence.capabilities import CapabilityPersistence


def validate_persistence_capabilities(
    capabilities: Sequence[AbstractCapability[object]],
) -> None:
    selected: set[PersistenceDomain] = set()

    def collect(feature: AbstractCapability[object]) -> None:
        if isinstance(feature, WrapperCapability):
            collect(feature.wrapped)
            return
        if isinstance(feature, CombinedCapability):
            for child in feature.capabilities:
                collect(child)
            return
        if isinstance(feature, ExecutionCapability) and feature.retain_history:
            domain = PersistenceDomain.CONVERSATION
        elif isinstance(feature, CheckpointCapability):
            domain = PersistenceDomain.CHECKPOINT
        elif isinstance(feature, TraceCapability):
            domain = PersistenceDomain.TRACE
        else:
            return
        if domain in selected:
            raise AIError(ErrorCode.CAPABILITY_CONFLICT, "one persistence feature is allowed per module")
        selected.add(domain)

    for capability in capabilities:
        collect(capability)


__all__ = ["validate_persistence_capabilities", "capability_persistence_ids", "execution_configuration"]


def capability_persistence_ids(
    capabilities: Sequence[tuple[str, AbstractCapability[object]]],
) -> tuple[str, ...]:
    """Resolve optional participants without binding their run callbacks."""
    identities: dict[str, None] = {}

    def collect(capability: AbstractCapability[object], registered_id: str | None) -> None:
        if isinstance(capability, WrapperCapability):
            collect(capability.wrapped, None)
        elif isinstance(capability, CombinedCapability):
            for child in capability.capabilities:
                collect(child, None)
        elif isinstance(capability, CapabilityPersistence):
            policy = capability.commit
            if not isinstance(policy, CommitPolicy):
                raise TypeError("capability commit must be a CommitPolicy")
            if policy.mode == "batch" and (
                policy.batch_size is None or policy.flush_interval is None
            ):
                raise ValueError("custom capability must resolve batch timing before registration")
            identity = capability.id or registered_id
            if not identity:
                raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
            if identity in identities:
                raise AIError(ErrorCode.CAPABILITY_CONFLICT)
            identities[identity] = None

    for identity, capability in capabilities:
        collect(capability, identity)
    return tuple(identities)


def execution_configuration(
    capabilities: Sequence[AbstractCapability[object]],
) -> ExecutionCapability:
    """Select the declared execution policy without binding or mutating its tree."""
    selected: ExecutionCapability | None = None

    def select(capability: AbstractCapability[object]) -> None:
        nonlocal selected
        if isinstance(capability, ExecutionCapability):
            if selected is not None:
                raise AIError(ErrorCode.CAPABILITY_CONFLICT)
            selected = capability

    def collect(capability: AbstractCapability[object]) -> None:
        if isinstance(capability, WrapperCapability):
            collect(capability.wrapped)
        elif isinstance(capability, CombinedCapability):
            for child in capability.capabilities:
                collect(child)
        else:
            capability.apply(select)

    for capability in capabilities:
        collect(capability)
    return ExecutionCapability() if selected is None else selected
