#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Instance-owned immutable model route registry."""

from collections.abc import Mapping
from dataclasses import dataclass
from threading import Lock
from types import MappingProxyType

from linktools.core import environ

from ..core import canonical_sha256
from ..errors import AIError, ErrorCode

_logger = environ.get_logger("ai.model.registry")


@dataclass(frozen=True, slots=True)
class ModelRoute:
    route_id: str
    provider: str
    model: str
    connection_id: "str | None" = None

    def __post_init__(self) -> None:
        if not self.route_id.strip() or not self.provider.strip() or not self.model.strip() or (self.connection_id is not None and not self.connection_id.strip()):
            raise ValueError("model route is incomplete")


@dataclass(frozen=True, slots=True)
class ModelRegistrySnapshot:
    revision: int
    routes: "Mapping[str, ModelRoute]"
    digest: str


class ModelRegistry:
    def __init__(self) -> None:
        self._snapshot = ModelRegistrySnapshot(0, MappingProxyType({}), canonical_sha256({"revision": 0, "routes": {}}))
        self._lock = Lock()

    def prime(self, routes: 'Mapping[str, ModelRoute]') -> ModelRegistrySnapshot:
        with self._lock:
            if self._snapshot.revision != 0:
                if dict(routes) == dict(self._snapshot.routes):
                    return self._snapshot
                raise AIError(ErrorCode.MODEL_REGISTRY_CONFLICT)
            return self._commit(routes)

    def register(self, route: ModelRoute, *, expected_revision: int) -> ModelRegistrySnapshot:
        with self._lock:
            if expected_revision != self._snapshot.revision:
                raise AIError(ErrorCode.MODEL_REGISTRY_CONFLICT)
            routes = dict(self._snapshot.routes)
            routes[route.route_id] = route
            return self._commit(routes)

    def remove(self, route_id: str, *, expected_revision: int) -> ModelRegistrySnapshot:
        with self._lock:
            if expected_revision != self._snapshot.revision:
                raise AIError(ErrorCode.MODEL_REGISTRY_CONFLICT)
            routes = dict(self._snapshot.routes)
            routes.pop(route_id, None)
            return self._commit(routes)

    def apply(self, routes: 'Mapping[str, ModelRoute]', *, expected_revision: int) -> ModelRegistrySnapshot:
        with self._lock:
            if expected_revision != self._snapshot.revision:
                raise AIError(ErrorCode.MODEL_REGISTRY_CONFLICT)
            return self._commit(routes)

    def snapshot(self) -> ModelRegistrySnapshot:
        return self._snapshot

    def _commit(self, routes: 'Mapping[str, ModelRoute]') -> ModelRegistrySnapshot:
        if any(not key.strip() or value.route_id != key for key, value in routes.items()):
            raise ValueError("model registry route keys must match route ids")
        normalized = dict(sorted(routes.items()))
        digest = canonical_sha256({"routes": {key: {"route_id": value.route_id, "provider": value.provider, "model": value.model, "connection_id": value.connection_id} for key, value in normalized.items()}})
        if normalized == dict(self._snapshot.routes):
            return self._snapshot
        self._snapshot = ModelRegistrySnapshot(self._snapshot.revision + 1, MappingProxyType(normalized), digest)
        _logger.info("model registry committed: revision=%s routes=%s", self._snapshot.revision, len(normalized))
        return self._snapshot


__all__ = ["ModelRegistry", "ModelRegistrySnapshot", "ModelRoute"]
