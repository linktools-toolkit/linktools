#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime default storage resources and typed route configuration."""

from pathlib import Path
from typing import TYPE_CHECKING

from ...storage import CacheBackend, StorageRoute, FilesystemObjectStore, ObjectStore
from ._plan import RuntimeStoragePlan

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncEngine

class RuntimeStorage:
    """Reusable resource declaration; each Runtime binds its own operational state."""

    def __init__(self, plan: RuntimeStoragePlan, *, object_store: ObjectStore | None = None) -> None:
        _validate_storage_configuration(plan, object_store)
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
        cls, *, cache: CacheBackend | None = None,
    ) -> "RuntimeStorage":
        route = StorageRoute.memory(cache=cache)
        return cls(RuntimeStoragePlan.from_default(route))

    @classmethod
    def filesystem(
        cls,
        path: "str | Path",
        *,
        object_store: "ObjectStore | None" = None,
        cache: CacheBackend | None = None,
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
        cache: CacheBackend | None = None,
    ) -> "RuntimeStorage":
        route = StorageRoute.sqlite(path, cache=cache)
        return cls(
            RuntimeStoragePlan.from_default(route),
            object_store=object_store,
        )

    @classmethod
    def from_root(
        cls, root: "str | Path", *, cache: CacheBackend | None = None,
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
        cache: CacheBackend | None = None,
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
