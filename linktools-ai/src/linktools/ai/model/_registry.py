#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Thread-safe model registry and immutable snapshots."""

from collections.abc import Mapping
from threading import RLock
from types import MappingProxyType

from linktools.core import environ
from pydantic_ai.models import Model

from ..core import JsonValue, canonical_sha256
from ..errors import AIError, ErrorCode
from ._contract import ModelBinding, ModelResolver
from ._openai import _OpenAIModelBinding

_logger = environ.get_logger("ai.model.registry")
_MODEL_PAYLOAD_FIELDS = frozenset({"version", "provider", "model_identity", "settings"})
_MODEL_SETTINGS_FIELDS = frozenset({"max_tokens", "context_window"})


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
        timeout: "float | None" = None,
        max_retries: "int | None" = None,
        retry_delay: "float | None" = None,
        max_tokens: "int | None" = None,
        context_window: "int | None" = None,
    ) -> "ModelRegistry":
        registry = cls()
        registry.register_openai(
            "default",
            model=model,
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
            max_retries=max_retries,
            retry_delay=retry_delay,
            max_tokens=max_tokens,
            context_window=context_window,
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
        timeout: "float | None" = None,
        max_retries: "int | None" = None,
        retry_delay: "float | None" = None,
        max_tokens: "int | None" = None,
        context_window: "int | None" = None,
    ) -> None:
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
                context_window=context_window,
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
            if preferred is not None:
                restored = _restore_binding(preferred, semantic)
                if restored is not None:
                    return restored
        matches = tuple(
            restored
            for _name, binding in sorted(self._bindings.items())
            if (restored := _restore_binding(binding, semantic)) is not None
        )
        if not matches:
            raise AIError(ErrorCode.MODEL_CONNECTION_NOT_FOUND)
        return matches[0]


class _HistoricalModelBinding:
    def __init__(self, current: ModelBinding, semantic_payload: "Mapping[str, JsonValue]") -> None:
        self._current = current
        self._semantic_payload = MappingProxyType(dict(semantic_payload))

    @property
    def route_id(self) -> str:
        return self._current.route_id

    @property
    def provider(self) -> str:
        return self._current.provider

    @property
    def model_identity(self) -> str:
        return self._current.model_identity

    @property
    def semantic_payload(self) -> "Mapping[str, JsonValue]":
        return self._semantic_payload

    @property
    def fingerprint(self) -> str:
        return canonical_sha256({"contract": "model-v1", **self._semantic_payload})

    def materialize(self) -> Model:
        return self._current.materialize()


def _restore_binding(binding: ModelBinding, semantic: "Mapping[str, JsonValue]") -> "ModelBinding | None":
    current = dict(binding.semantic_payload)
    if current == dict(semantic):
        return binding
    if _legacy_model_payload_matches(current, semantic):
        return _HistoricalModelBinding(binding, semantic)
    return None


def _legacy_model_payload_matches(
    current: "Mapping[str, JsonValue]",
    historical: "Mapping[str, JsonValue]",
) -> bool:
    if set(current) != _MODEL_PAYLOAD_FIELDS or set(historical) != _MODEL_PAYLOAD_FIELDS:
        return False
    if current.get("version") != 1 or historical.get("version") != 1:
        return False
    if current.get("provider") != historical.get("provider") or current.get("model_identity") != historical.get("model_identity"):
        return False
    historical_settings = historical.get("settings")
    current_settings = current.get("settings")
    return (
        isinstance(historical_settings, Mapping)
        and not historical_settings
        and isinstance(current_settings, Mapping)
        and set(current_settings).issubset(_MODEL_SETTINGS_FIELDS)
    )


__all__ = ["ModelRegistry"]
