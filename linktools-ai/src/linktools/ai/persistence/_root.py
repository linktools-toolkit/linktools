#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Physical persistence resources and typed route configuration."""

import json
from itertools import chain
from pathlib import Path
from typing import TYPE_CHECKING

from ..storage import CacheBackend, TurnCoordinationBackend, StreamBackend, StorageRoute, FilesystemObjectStore, ObjectStore
from ..errors import AIError, ErrorCode
from ._plan import PersistenceDomain, PersistencePlan

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncEngine

class PersistenceStorage:
    """Reusable resource declaration; each Persistence owns its operational handles."""

    def __init__(
        self, plan: PersistencePlan, *, object_store: ObjectStore | None = None,
        coordination: CacheBackend | TurnCoordinationBackend | StreamBackend | None = None,
    ) -> None:
        _validate_storage_configuration(plan, object_store)
        if coordination is not None and not isinstance(
            coordination, (CacheBackend, TurnCoordinationBackend, StreamBackend),
        ):
            raise TypeError("coordination must implement cache, turn coordination or stream operations")
        for names in (
            ("get", "put", "delete"),
            ("claim_turn", "read_turn", "renew_turn", "release_turn", "cancel_turn"),
            ("open_stream", "read_stream"),
        ):
            methods = tuple(getattr(coordination, name, None) for name in names)
            if any(method is not None for method in methods) and not all(callable(method) for method in methods):
                raise TypeError(f"coordination must implement {', '.join(names)} together")
        self._coordination = coordination
        self._plan = plan
        self._object_store = object_store

    @property
    def coordination(self) -> CacheBackend | TurnCoordinationBackend | StreamBackend | None:
        """Borrowed coordination resource shared by producers and read-only views."""
        return self._coordination

    @property
    def plan(self) -> PersistencePlan:
        return self._plan

    @property
    def object_store(self) -> ObjectStore | None:
        """Borrowed object resource; ownership stays with its provider."""
        return self._object_store

    @classmethod
    def in_memory(
        cls, *, cache: CacheBackend | None = None,
        coordination: CacheBackend | TurnCoordinationBackend | StreamBackend | None = None,
    ) -> "PersistenceStorage":
        route = StorageRoute.memory(cache=cache)
        return cls(PersistencePlan.from_default(route), coordination=coordination)

    @classmethod
    def filesystem(
        cls,
        path: "str | Path",
        *,
        object_store: "ObjectStore | None" = None,
        cache: CacheBackend | None = None,
        coordination: CacheBackend | TurnCoordinationBackend | StreamBackend | None = None,
    ) -> "PersistenceStorage":
        base = _normalize_path(path)
        return cls(
            PersistencePlan.filesystem(base, cache=cache),
            object_store=object_store, coordination=coordination,
        )

    @classmethod
    def sqlite(
        cls,
        path: "str | Path",
        *,
        object_store: "ObjectStore | None" = None,
        cache: CacheBackend | None = None,
        coordination: CacheBackend | TurnCoordinationBackend | StreamBackend | None = None,
    ) -> "PersistenceStorage":
        route = StorageRoute.sqlite(path, cache=cache)
        return cls(
            PersistencePlan.from_default(route),
            object_store=object_store, coordination=coordination,
        )

    @classmethod
    def from_root(
        cls, root: "str | Path", *, cache: CacheBackend | None = None,
        coordination: CacheBackend | TurnCoordinationBackend | StreamBackend | None = None,
    ) -> "PersistenceStorage":
        base = _normalize_path(root)
        if not (base / "runtime.db").exists():
            _reject_filesystem_state_root(base)
        return cls.sqlite(
            base / "runtime.db",
            object_store=FilesystemObjectStore(base / "objects"),
            cache=cache, coordination=coordination,
        )

    @classmethod
    def sql(
        cls,
        engine: "AsyncEngine",
        *,
        object_store: "ObjectStore | None" = None,
        cache: CacheBackend | None = None,
        coordination: CacheBackend | TurnCoordinationBackend | StreamBackend | None = None,
    ) -> "PersistenceStorage":
        route = StorageRoute.sql(engine, cache=cache)
        return cls(
            PersistencePlan.from_default(route),
            object_store=object_store, coordination=coordination,
        )

    @classmethod
    def from_plan(
        cls,
        plan: PersistencePlan,
        *,
        object_store: "ObjectStore | None" = None,
        coordination: CacheBackend | TurnCoordinationBackend | StreamBackend | None = None,
    ) -> "PersistenceStorage":
        return cls(
            plan, object_store=object_store, coordination=coordination,
        )


def _reject_filesystem_state_root(root: Path) -> None:
    manifests = chain(
        (root / "manifest.json",),
        (root / ".state-groups").glob("*/manifest.json"),
        *((root / domain.value).glob("*/*/manifest.json") for domain in PersistenceDomain),
    )
    for path in manifests:
        if not path.is_file():
            continue
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        if isinstance(manifest, dict) and manifest.get("format") in (
            "linktools-ai-state", "linktools-ai-state-group",
        ):
            raise AIError(
                ErrorCode.STORAGE_RECOVERY_REQUIRED,
                "Filesystem Runtime state requires explicit migration into a new SQLite root",
            )


def _validate_storage_configuration(plan: PersistencePlan, object_store: ObjectStore | None) -> None:
    if not isinstance(plan, PersistencePlan):
        raise TypeError("plan must be a PersistencePlan")
    plan.validate_object_store(object_store)


def _normalize_path(value: "str | Path") -> Path:
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise ValueError("PersistenceStorage path is required")
    return Path(value).expanduser().resolve(strict=False)



__all__ = ["PersistenceStorage"]
