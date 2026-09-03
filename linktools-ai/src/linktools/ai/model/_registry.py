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
        self._revision = 0
        self._lock = RLock()

    @classmethod
    def openai(
        cls,
        *,
        model: str,
        base_url: "str | None" = None,
        api_key: "str | None" = None,
        timeout_seconds: "float | None" = None,
        max_retries: "int | None" = None,
        retry_delay_seconds: "float | None" = None,
        max_output_tokens: "int | None" = None,
        context_window_tokens: "int | None" = None,
    ) -> "ModelRegistry":
        registry = cls()
        registry.register_openai(
            "default",
            model=model,
            base_url=base_url,
            api_key=api_key,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            retry_delay_seconds=retry_delay_seconds,
            max_output_tokens=max_output_tokens,
            context_window_tokens=context_window_tokens,
        )
        return registry

    def register(self, binding: ModelBinding) -> None:
        if not binding.route_id.strip():
            raise ValueError("model route_id is required")
        with self._lock:
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
        timeout_seconds: "float | None" = None,
        max_retries: "int | None" = None,
        retry_delay_seconds: "float | None" = None,
        max_output_tokens: "int | None" = None,
        context_window_tokens: "int | None" = None,
    ) -> None:
        self.register(
            _OpenAIModelBinding(
                route_id=route_id,
                model=model,
                base_url=base_url,
                api_key=api_key,
                timeout_seconds=timeout_seconds,
                max_retries=max_retries,
                retry_delay_seconds=retry_delay_seconds,
                max_output_tokens=max_output_tokens,
                context_window_tokens=context_window_tokens,
            )
        )

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

    def restore(
        self,
        payload: "Mapping[str, JsonValue]",
        *,
        route_id: "str | None" = None,
    ) -> ModelBinding:
        semantic = dict(payload)
        if route_id is not None:
            preferred = self._bindings.get(route_id)
            if preferred is not None and dict(preferred.semantic_payload) == semantic:
                return preferred
        matches = tuple(
            binding
            for _name, binding in sorted(self._bindings.items())
            if dict(binding.semantic_payload) == semantic
        )
        if not matches:
            raise AIError(ErrorCode.MODEL_CONNECTION_NOT_FOUND)
        return matches[0]


__all__ = ["ModelRegistry"]
