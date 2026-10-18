#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared persistence declarations for active and read-only Runtime assembly."""

from collections.abc import Sequence

from pydantic_ai.capabilities import AbstractCapability, CombinedCapability, WrapperCapability

from ..errors import AIError, ErrorCode
from linktools.ai.capability import ConversationCapability
from linktools.ai.capability import CheckpointCapability, TraceCapability
from .state import RuntimeDomain, RuntimeStorageRoute


def persistence_storage_configuration(
    capabilities: Sequence[AbstractCapability[object]],
) -> tuple[dict[RuntimeDomain, RuntimeStorageRoute], dict[RuntimeDomain, object | None]]:
    routes: dict[RuntimeDomain, RuntimeStorageRoute] = {}
    caches: dict[RuntimeDomain, object | None] = {}

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
        if domain in caches:
            raise AIError(ErrorCode.CAPABILITY_CONFLICT, "one persistence feature is allowed per module")
        if feature.storage is not None:
            if not isinstance(feature.storage, RuntimeStorageRoute):
                raise TypeError("storage must be RuntimeStorageRoute")
            routes[domain] = feature.storage
        caches[domain] = feature.cache if isinstance(feature, (ConversationCapability, TraceCapability)) else None

    for capability in capabilities:
        collect(capability)
    return routes, caches


__all__ = ["persistence_storage_configuration"]
