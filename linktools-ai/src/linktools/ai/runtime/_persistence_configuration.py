#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared persistence declarations for active and read-only Runtime assembly."""

from collections.abc import Sequence

from pydantic_ai_harness.step_persistence import StepPersistence
from pydantic_ai.capabilities import AbstractCapability, CombinedCapability, WrapperCapability

from ..errors import AIError, ErrorCode
from ..core import CommitPolicy, validate_capability_id
from ..capability import TraceCapability
from ..persistence import PersistenceDomain
from ..persistence.capabilities import CapabilityPersistence

from ._step_capability import _StepPersistenceCapability


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
        if isinstance(feature, (StepPersistence, _StepPersistenceCapability)):
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


__all__ = ["validate_persistence_capabilities", "capability_persistence_ids"]


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
            try:
                validate_capability_id(identity)
            except AIError as error:
                raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID) from error
            if identity in identities:
                raise AIError(ErrorCode.CAPABILITY_CONFLICT)
            identities[identity] = None

    for identity, capability in capabilities:
        collect(capability, identity)
    return tuple(identities)
