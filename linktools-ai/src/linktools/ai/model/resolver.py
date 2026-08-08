#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Read-only model route resolution."""

from typing import Protocol

from ..core.errors import ErrorCode, AIError
from .registry import ModelRegistrySnapshot, ModelRoute


class ModelResolver(Protocol):
    def resolve(self, route_id: str) -> ModelRoute: ...
    def snapshot(self) -> ModelRegistrySnapshot: ...


class SnapshotModelResolver:
    def __init__(self, snapshot: ModelRegistrySnapshot) -> None:
        self._snapshot = snapshot

    def resolve(self, route_id: str) -> ModelRoute:
        try:
            return self._snapshot.routes[route_id]
        except KeyError as exc:
            raise AIError(ErrorCode.STORAGE_NOT_FOUND) from exc

    def snapshot(self) -> ModelRegistrySnapshot:
        return self._snapshot


__all__ = ["ModelResolver", "SnapshotModelResolver"]
