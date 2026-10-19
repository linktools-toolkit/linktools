#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared persistence declarations for active and read-only Runtime assembly."""

from collections.abc import Sequence

from pydantic_ai.capabilities import AbstractCapability, CombinedCapability, WrapperCapability

from ..errors import AIError, ErrorCode
from ..core import CommitPolicy
from ..capability import CheckpointCapability, ConversationCapability, TraceCapability
from ..storage import StorageRoute
from .state import RuntimeDomain
from .persistence import CapabilityPersistence


def persistence_storage_configuration(
    capabilities: Sequence[AbstractCapability[object]],
) -> dict[RuntimeDomain, StorageRoute]:
    routes: dict[RuntimeDomain, StorageRoute] = {}
    selected: set[RuntimeDomain] = set()

    def collect(feature: AbstractCapability[object]) -> None:
        if isinstance(feature, WrapperCapability):
            collect(feature.wrapped)
            return
        if isinstance(feature, CombinedCapability):
            for child in feature.capabilities:
                collect(child)
            return
        if isinstance(feature, ConversationCapability):
            domain = RuntimeDomain.CONVERSATION
        elif isinstance(feature, CheckpointCapability):
            domain = RuntimeDomain.CHECKPOINT
        elif isinstance(feature, TraceCapability):
            domain = RuntimeDomain.TRACE
        else:
            return
        if domain in selected:
            raise AIError(ErrorCode.CAPABILITY_CONFLICT, "one persistence feature is allowed per module")
        selected.add(domain)
        if isinstance(feature, (CheckpointCapability, TraceCapability)) and feature.storage is not None:
            if not isinstance(feature.storage, StorageRoute):
                raise TypeError("storage must be StorageRoute")
            routes[domain] = feature.storage

    for capability in capabilities:
        collect(capability)
    return routes


__all__ = ["persistence_storage_configuration", "capability_storage_configuration"]


def capability_storage_configuration(
    capabilities: Sequence[tuple[str, AbstractCapability[object]]], *, default_route: StorageRoute,
) -> dict[str, StorageRoute]:
    """Resolve optional participants without binding their run callbacks."""
    routes: dict[str, StorageRoute] = {}

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
            if identity in routes:
                raise AIError(ErrorCode.CAPABILITY_CONFLICT)
            route = capability.storage or default_route
            if not isinstance(route, StorageRoute):
                raise TypeError("Persistent capability storage must use StorageRoute")
            routes[identity] = route

    for identity, capability in capabilities:
        collect(capability, identity)
    return routes
