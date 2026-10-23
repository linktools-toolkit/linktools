#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Immutable model routes with deferred, per-admission binding resolution."""

import inspect
from collections.abc import Callable, Mapping
from threading import RLock, local
from types import MappingProxyType
from typing import Protocol, cast, runtime_checkable

from pydantic_ai.models import Model

from ..core import JsonValue
from ..errors import AIError, ErrorCode
from ..model import ModelBinding, model_thinking_control


@runtime_checkable
class _MaterializableModel(Protocol):
    def materialize(self) -> Model: ...


class _LoadingRoutes(local):
    def __init__(self) -> None:
        self.routes: set[str] = set()


class _ModelResolver:
    """Capture route sources without loading models or creating provider clients."""

    def __init__(self, sources: Mapping[str, ModelBinding | Callable[[], ModelBinding]]) -> None:
        bindings: dict[str, ModelBinding | Callable[[], ModelBinding]] = {}
        for route, source in sources.items():
            if not isinstance(route, str) or not route.strip():
                raise ValueError("model route must be non-empty")
            if inspect.iscoroutinefunction(source):
                raise TypeError("model loaders must be synchronous")
            if isinstance(source, _MaterializableModel):
                bindings[route] = cast(ModelBinding, source)
            elif callable(source):
                bindings[route] = source
            else:
                raise TypeError("model source must be a ModelBinding or synchronous loader")
        self._bindings = MappingProxyType(bindings)
        self._loading = _LoadingRoutes()

    def resolve(self, route_id: str) -> ModelBinding:
        try:
            source = self._bindings[route_id]
            return source if isinstance(source, _MaterializableModel) else _LazyModelBinding(
                lambda: self._load(route_id, source)
            )
        except KeyError as error:
            raise AIError(ErrorCode.MODEL_CONNECTION_NOT_FOUND) from error

    def _load(self, route: str, source: Callable[[], ModelBinding]) -> ModelBinding:
        if route in self._loading.routes:
            raise AIError(ErrorCode.MODEL_CONFIG_INVALID, "Recursive model loader")
        self._loading.routes.add(route)
        try:
            return source()
        finally:
            self._loading.routes.remove(route)



class _LazyModelBinding:
    def __init__(self, load: Callable[[], ModelBinding]) -> None:
        self._load = load
        self._binding: ModelBinding | None = None
        self._lock = RLock()
        self._loading = False

    def _resolve(self) -> ModelBinding:
        with self._lock:
            if self._binding is None:
                if self._loading:
                    raise AIError(ErrorCode.MODEL_CONFIG_INVALID, "Recursive model loader")
                self._loading = True
                try:
                    binding = self._load()
                    if inspect.isawaitable(binding):
                        if inspect.iscoroutine(binding):
                            binding.close()
                        raise TypeError("model loaders must return a synchronous ModelBinding")
                    if not isinstance(binding, ModelBinding):
                        raise TypeError("model loaders must return a ModelBinding")
                    self._binding = binding
                finally:
                    self._loading = False
            return self._binding

    @property
    def provider(self) -> str:
        return self._resolve().provider

    @property
    def model_identity(self) -> str:
        return self._resolve().model_identity

    @property
    def contract(self) -> Mapping[str, JsonValue]:
        return self._resolve().contract

    def thinking_control(self) -> Mapping[str, JsonValue]:
        return model_thinking_control(self._resolve())

    def materialize(self) -> Model:
        return self._resolve().materialize()


__all__ = []
