#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime default storage resources and typed route configuration."""

from pathlib import Path
from typing import TYPE_CHECKING

from ...storage import CacheCapabilities, StorageRoute, FilesystemObjectStore, ObjectStore
from ._plan import RuntimeStoragePlan

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncEngine

def _validate_cache(cache: CacheCapabilities | None) -> None:
    if cache is None:
        return
    for methods in (
        ("get", "put", "delete"),
        ("append", "read_batch", "ack"),
        ("open_stream", "read_stream"),
        ("claim_turn", "read_turn", "renew_turn", "release_turn", "cancel_turn"),
    ):
        available = tuple(getattr(cache, method, None) for method in methods)
        if any(method is not None for method in available) and not all(map(callable, available)):
            raise TypeError(f"cache must implement all methods in {', '.join(methods)}")


class RuntimeStorage:
    """Reusable resource declaration; each Runtime binds its own operational state."""

    def __init__(self, plan: RuntimeStoragePlan, *, object_store: ObjectStore | None = None) -> None:
        _validate_storage_configuration(plan, object_store)
        for route in plan.resolved_routes.values():
            _validate_cache(route.cache)
        self._plan = plan
        self._object_store = object_store

    @property
    def plan(self) -> RuntimeStoragePlan:
        return self._plan

    @property
    def object_store(self) -> ObjectStore | None:
        """Borrowed object resource; ownership stays with its provider."""
        return self._object_store

    @classmethod
    def in_memory(
        cls, *, cache: CacheCapabilities | None = None,
    ) -> "RuntimeStorage":
        route = StorageRoute.memory(cache=cache)
        return cls(RuntimeStoragePlan.from_default(route))

    @classmethod
    def filesystem(
        cls,
        path: "str | Path",
        *,
        object_store: "ObjectStore | None" = None,
        cache: CacheCapabilities | None = None,
    ) -> "RuntimeStorage":
        base = _normalize_path(path)
        return cls(
            RuntimeStoragePlan.filesystem(base, cache=cache),
            object_store=object_store,
        )

    @classmethod
    def sqlite(
        cls,
        path: "str | Path",
        *,
        object_store: "ObjectStore | None" = None,
        cache: CacheCapabilities | None = None,
    ) -> "RuntimeStorage":
        route = StorageRoute.sqlite(path, cache=cache)
        return cls(
            RuntimeStoragePlan.from_default(route),
            object_store=object_store,
        )

    @classmethod
    def from_root(
        cls, root: "str | Path", *, cache: CacheCapabilities | None = None,
    ) -> "RuntimeStorage":
        base = _normalize_path(root)
        return cls.sqlite(
            base / "runtime.db",
            object_store=FilesystemObjectStore(base / "objects"),
            cache=cache,
        )

    @classmethod
    def sql(
        cls,
        engine: "AsyncEngine",
        *,
        object_store: "ObjectStore | None" = None,
        cache: CacheCapabilities | None = None,
    ) -> "RuntimeStorage":
        route = StorageRoute.sql(engine, cache=cache)
        return cls(
            RuntimeStoragePlan.from_default(route),
            object_store=object_store,
        )

    @classmethod
    def from_plan(
        cls,
        plan: RuntimeStoragePlan,
        *,
        object_store: "ObjectStore | None" = None,
    ) -> "RuntimeStorage":
        return cls(
            plan, object_store=object_store,
        )


def _validate_storage_configuration(plan: RuntimeStoragePlan, object_store: ObjectStore | None) -> None:
    if not isinstance(plan, RuntimeStoragePlan):
        raise TypeError("plan must be a RuntimeStoragePlan")
    plan.validate_object_store(object_store)


def _normalize_path(value: "str | Path") -> Path:
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise ValueError("RuntimeStorage path is required")
    return Path(value).expanduser().resolve(strict=False)



__all__ = ["RuntimeStorage"]
