#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Thread-safe model registry and immutable snapshots."""

from collections.abc import Mapping
from threading import RLock
from types import MappingProxyType

from linktools.core import environ

from ..core import JsonValue
from ..errors import AIError, ErrorCode
from ._contract import ModelBinding, ModelResolver
from ._openai import _OpenAIModelBinding

_logger = environ.get_logger("ai.model.registry")


class ModelRegistry:
    def __init__(self) -> None:
        self._bindings: dict[str, ModelBinding] = {}
        self._lock = RLock()

    @classmethod
    def openai(
        cls,
        *,
        model: str,
        provider_instance: "str | None" = None,
        base_url: "str | None" = None,
        api_key: "str | None" = None,
        timeout: "int | float | None" = None,
        max_retries: int = 2,
        retry_delay: "int | float" = 1.0,
        max_tokens: "int | None" = None,
    ) -> "ModelRegistry":
        registry = cls()
        registry.register_openai(
            "default",
            model=model,
            provider_instance=provider_instance,
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
            max_retries=max_retries,
            retry_delay=retry_delay,
            max_tokens=max_tokens,
        )
        return registry

    def register(self, binding: ModelBinding) -> None:
        if not binding.route_id.strip():
            raise ValueError("model route_id is required")
        with self._lock:
            self._bindings[binding.route_id] = binding
            _logger.info("model binding registered: route=%s", binding.route_id)

    def register_openai(
        self,
        route_id: str,
        *,
        model: str,
        provider_instance: "str | None" = None,
        base_url: "str | None" = None,
        api_key: "str | None" = None,
        timeout: "int | float | None" = None,
        max_retries: int = 2,
        retry_delay: "int | float" = 1.0,
        max_tokens: "int | None" = None,
    ) -> None:
        del provider_instance
        self.register(
            _OpenAIModelBinding(
                route_id=route_id,
                model=model,
                base_url=base_url,
                api_key=api_key,
                timeout=timeout,
                max_retries=max_retries,
                retry_delay=retry_delay,
                max_tokens=max_tokens,
            )
        )

    def remove(self, route_id: str) -> None:
        with self._lock:
            if route_id in self._bindings:
                del self._bindings[route_id]
                _logger.info("model binding removed: route=%s", route_id)

    def snapshot(self) -> ModelResolver:
        with self._lock:
            return _ModelRegistrySnapshot(MappingProxyType(dict(self._bindings)))


class _ModelRegistrySnapshot:
    def __init__(self, bindings: "Mapping[str, ModelBinding]") -> None:
        self._bindings = bindings

    def resolve(self, route_id: str) -> ModelBinding:
        try:
            return self._bindings[route_id]
        except KeyError as error:
            raise AIError(ErrorCode.MODEL_CONNECTION_NOT_FOUND) from error

    def restore(
        self,
        payload: "Mapping[str, JsonValue]",
        *,
        route_id: "str | None" = None,
    ) -> ModelBinding:
        if route_id is None:
            raise AIError(ErrorCode.MODEL_CONNECTION_NOT_FOUND)
        binding = self._bindings.get(route_id)
        if binding is None:
            raise AIError(ErrorCode.MODEL_CONNECTION_NOT_FOUND)
        if dict(binding.semantic_payload) != dict(payload):
            raise AIError(ErrorCode.AGENT_DEFINITION_UNAVAILABLE)
        return binding


__all__ = ["ModelRegistry"]
