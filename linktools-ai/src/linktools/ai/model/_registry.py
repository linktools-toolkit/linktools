#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Thread-safe model registry and immutable snapshots."""

from collections.abc import Mapping
from threading import RLock
from types import MappingProxyType

from linktools.core import environ

from ..errors import AIError, ErrorCode
from ._contract import ModelBinding, ModelResolver
from ._openai import _OpenAIModelBinding

_logger = environ.get_logger("ai.model.registry")


class ModelRegistry:
    def __init__(self) -> None:
        self._bindings: "dict[str, ModelBinding]" = {}
        self._revision = 0
        self._lock = RLock()

    @classmethod
    def openai(
        cls,
        *,
        model: str,
        base_url: "str | None" = None,
        api_key: "str | None" = None,
    ) -> "ModelRegistry":
        registry = cls()
        registry.register_openai("default", model=model, base_url=base_url, api_key=api_key)
        return registry

    def register(self, binding: ModelBinding) -> None:
        if not binding.route_id.strip():
            raise ValueError("model route_id is required")
        with self._lock:
            current = self._bindings.get(binding.route_id)
            if current is not None and current.fingerprint == binding.fingerprint:
                return
            self._bindings[binding.route_id] = binding
            self._revision += 1
            _logger.info("model binding registered: route=%s revision=%s", binding.route_id, self._revision)

    def register_openai(
        self,
        route_id: str,
        *,
        model: str,
        base_url: "str | None" = None,
        api_key: "str | None" = None,
    ) -> None:
        self.register(_OpenAIModelBinding(route_id, model, base_url, api_key))

    def remove(self, route_id: str) -> None:
        with self._lock:
            if route_id in self._bindings:
                del self._bindings[route_id]
                self._revision += 1
                _logger.info("model binding removed: route=%s revision=%s", route_id, self._revision)

    def snapshot(self) -> ModelResolver:
        with self._lock:
            return _ModelRegistrySnapshot(self._revision, MappingProxyType(dict(self._bindings)))


class _ModelRegistrySnapshot:
    def __init__(self, revision: int, bindings: "Mapping[str, ModelBinding]") -> None:
        self._revision = revision
        self._bindings = bindings

    def resolve(self, route_id: str) -> ModelBinding:
        try:
            return self._bindings[route_id]
        except KeyError as error:
            raise AIError(ErrorCode.MODEL_CONNECTION_NOT_FOUND) from error


__all__ = ["ModelRegistry"]
