#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Instance-owned immutable model route registry."""

from dataclasses import dataclass
from types import MappingProxyType
from collections.abc import Mapping

from linktools.core import environ

from ..core.errors import ErrorCode, LinktoolsAIError
from ..core.ids import canonical_sha256

_logger = environ.get_logger("ai.model.registry")


@dataclass(frozen=True, slots=True)
class ModelRoute:
    route_id: str
    provider: str
    model: str

    def __post_init__(self) -> None:
        if not self.route_id.strip() or not self.provider.strip() or not self.model.strip():
            raise ValueError("model route is incomplete")


@dataclass(frozen=True, slots=True)
class ModelRegistrySnapshot:
    revision: int
    routes: "Mapping[str, ModelRoute]"
    digest: str


class ModelRegistry:
    def __init__(self) -> None:
        self._snapshot = ModelRegistrySnapshot(0, MappingProxyType({}), canonical_sha256({"revision": 0, "routes": {}}))

    def prime(self, routes: 'Mapping[str, ModelRoute]') -> ModelRegistrySnapshot:
        if self._snapshot.revision != 0:
            if dict(routes) == dict(self._snapshot.routes):
                return self._snapshot
            raise LinktoolsAIError(ErrorCode.MODEL_REGISTRY_CONFLICT)
        return self._commit(routes)

    def register(self, route: ModelRoute) -> ModelRegistrySnapshot:
        routes = dict(self._snapshot.routes)
        routes[route.route_id] = route
        return self._commit(routes)

    def remove(self, route_id: str) -> ModelRegistrySnapshot:
        routes = dict(self._snapshot.routes)
        routes.pop(route_id, None)
        return self._commit(routes)

    def apply(self, routes: 'Mapping[str, ModelRoute]', *, expected_revision: 'int | None' = None) -> ModelRegistrySnapshot:
        if expected_revision is not None and expected_revision != self._snapshot.revision:
            raise LinktoolsAIError(ErrorCode.MODEL_REGISTRY_CONFLICT)
        return self._commit(routes)

    def snapshot(self) -> ModelRegistrySnapshot:
        return self._snapshot

    def _commit(self, routes: 'Mapping[str, ModelRoute]') -> ModelRegistrySnapshot:
        if any(not key.strip() or value.route_id != key for key, value in routes.items()):
            raise ValueError("model registry route keys must match route ids")
        normalized = dict(sorted(routes.items()))
        digest = canonical_sha256({"routes": {key: {"route_id": value.route_id, "provider": value.provider, "model": value.model} for key, value in normalized.items()}})
        if normalized == dict(self._snapshot.routes):
            return self._snapshot
        self._snapshot = ModelRegistrySnapshot(self._snapshot.revision + 1, MappingProxyType(normalized), digest)
        _logger.info("model registry committed: revision=%s routes=%s", self._snapshot.revision, len(normalized))
        return self._snapshot


__all__ = ["ModelRegistry", "ModelRegistrySnapshot", "ModelRoute"]
