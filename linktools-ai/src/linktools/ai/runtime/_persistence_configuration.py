#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared persistence declarations for active and read-only Runtime assembly."""

from collections.abc import Sequence

from pydantic_ai.capabilities import AbstractCapability, CombinedCapability, WrapperCapability

from ..errors import AIError, ErrorCode
from ..capability import CheckpointCapability, ConversationCapability, TraceCapability
from ..storage import StorageRoute
from .state import RuntimeDomain


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


__all__ = ["persistence_storage_configuration"]
