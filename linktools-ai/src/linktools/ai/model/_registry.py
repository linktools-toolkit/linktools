#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Thread-safe model registry and immutable snapshots."""

from collections.abc import Callable, Mapping
from threading import RLock
from types import MappingProxyType

from linktools.core import environ
from pydantic_ai.models import Model

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
        vision: bool = False,
        base_url: "str | None" = None,
        api_key: "str | None" = None,
        timeout: "int | float | None" = None,
        max_retries: int = 2,
        retry_delay: "int | float" = 1.0,
        max_tokens: "int | None" = None,
        connection_resolver: "Callable[[], Mapping[str, object]] | None" = None,
    ) -> "ModelRegistry":
        registry = cls()
        registry.register_openai(
            "default",
            model=model,
            vision=vision,
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
            max_retries=max_retries,
            retry_delay=retry_delay,
            max_tokens=max_tokens,
            connection_resolver=connection_resolver,
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
        vision: bool = False,
        base_url: "str | None" = None,
        api_key: "str | None" = None,
        timeout: "int | float | None" = None,
        max_retries: int = 2,
        retry_delay: "int | float" = 1.0,
        max_tokens: "int | None" = None,
        connection_resolver: "Callable[[], Mapping[str, object]] | None" = None,
    ) -> None:
        self.register(
            _OpenAIModelBinding(
                route_id=route_id,
                model=model,
                vision=vision,
                base_url=base_url,
                api_key=api_key,
                timeout=timeout,
                max_retries=max_retries,
                retry_delay=retry_delay,
                max_tokens=max_tokens,
                connection_resolver=connection_resolver,
            )
        )

    def register_alias(self, alias_id: str, target_route_id: str) -> None:
        """Register an immutable route alias to an existing binding."""
        if not isinstance(alias_id, str) or not alias_id.strip():
            raise ValueError("model alias is required")
        if not isinstance(target_route_id, str) or not target_route_id.strip():
            raise ValueError("model alias target is required")
        with self._lock:
            target = self._bindings.get(target_route_id)
            if target is None:
                raise AIError(ErrorCode.MODEL_CONNECTION_NOT_FOUND)
            while isinstance(target, _ModelAliasBinding):
                target = target._target
            self._bindings[alias_id] = _ModelAliasBinding(alias_id, target)
            _logger.info(
                "model binding alias registered: alias=%s target=%s",
                alias_id,
                target.route_id,
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


class _ModelAliasBinding:
    __slots__ = ("_route_id", "_target")

    def __init__(self, route_id: str, target: ModelBinding) -> None:
        self._route_id = route_id
        self._target = target

    def __setattr__(self, name: str, value: object) -> None:
        if name in self.__slots__ and hasattr(self, name):
            raise AttributeError("model alias binding is immutable")
        object.__setattr__(self, name, value)

    @property
    def route_id(self) -> str:
        return self._route_id

    @property
    def provider(self) -> str:
        return self._target.provider

    @property
    def model_identity(self) -> str:
        return self._target.model_identity

    @property
    def vision(self) -> bool:
        return self._target.vision

    @property
    def semantic_payload(self) -> Mapping[str, JsonValue]:
        return self._target.semantic_payload

    @property
    def fingerprint(self) -> str:
        return self._target.fingerprint

    def materialize(self) -> Model:
        return self._target.materialize()
