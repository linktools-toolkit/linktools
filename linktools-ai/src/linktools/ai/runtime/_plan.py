#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime-owned storage selection and retention policy values."""

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING

from ..errors import AIError, ErrorCode

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncEngine

    from ..storage import ObjectStore


class RuntimeDomain(StrEnum):
    CONVERSATION = "conversation"
    EXECUTION = "execution"
    MEMORY = "memory"
    ARTIFACT = "artifact"
    TASK = "task"
    EVALUATION = "evaluation"
    RECOVERY = "recovery"


class RuntimeRetention(StrEnum):
    DURABLE = "durable"
    VOLATILE = "volatile"
    TRANSIENT = "transient"


_OBJECT_DOMAINS = frozenset(
    {
        RuntimeDomain.CONVERSATION,
        RuntimeDomain.EXECUTION,
        RuntimeDomain.MEMORY,
        RuntimeDomain.ARTIFACT,
        RuntimeDomain.RECOVERY,
    }
)


@dataclass(frozen=True, slots=True)
class RuntimeStorageRoute:
    retention: RuntimeRetention
    object_store: "ObjectStore | None" = None

    @classmethod
    def durable(cls, *, object_store: "ObjectStore | None" = None) -> "RuntimeStorageRoute":
        return cls(RuntimeRetention.DURABLE, object_store)

    @classmethod
    def volatile(cls) -> "RuntimeStorageRoute":
        return cls(RuntimeRetention.VOLATILE)

    @classmethod
    def transient(cls) -> "RuntimeStorageRoute":
        return cls(RuntimeRetention.TRANSIENT)


@dataclass(frozen=True, slots=True)
class RuntimeStoragePlan:
    routes: Mapping[RuntimeDomain, RuntimeStorageRoute]

    def __post_init__(self) -> None:
        normalized: dict[RuntimeDomain, RuntimeStorageRoute] = {}
        stores: dict[str, object] = {}
        for domain, route in self.routes.items():
            if not isinstance(domain, RuntimeDomain) or not isinstance(route, RuntimeStorageRoute):
                raise ValueError("runtime storage plan contains an invalid route")
            if route.object_store is not None and route.retention is not RuntimeRetention.DURABLE:
                raise ValueError("only durable routes may own an object store")
            if domain not in _OBJECT_DOMAINS and route.object_store is not None:
                raise ValueError(f"{domain.value} cannot own an object store")
            if route.object_store is not None:
                try:
                    store_id = route.object_store.store_id
                except AttributeError as error:
                    raise ValueError("object store must expose a stable store_id") from error
                if not isinstance(store_id, str) or not store_id or store_id in {"builtin", "memory", "transient"}:
                    raise ValueError("custom object store id is reserved or invalid")
                previous = stores.get(store_id)
                if previous is not None and previous is not route.object_store:
                    raise ValueError("distinct object stores cannot share a store_id")
                stores[store_id] = route.object_store
            normalized[domain] = route
        object.__setattr__(self, "routes", MappingProxyType(normalized))

    def route(self, domain: RuntimeDomain) -> RuntimeStorageRoute:
        return self.routes.get(domain, RuntimeStorageRoute.volatile())

    @classmethod
    def all(cls, *, object_store: "ObjectStore | None" = None) -> "RuntimeStoragePlan":
        return cls(
            {
                domain: RuntimeStorageRoute.durable(
                    object_store=object_store if domain in _OBJECT_DOMAINS else None,
                )
                for domain in RuntimeDomain
            }
        )

    @classmethod
    def volatile(cls) -> "RuntimeStoragePlan":
        return cls({domain: RuntimeStorageRoute.volatile() for domain in RuntimeDomain})


@dataclass(frozen=True, slots=True)
class _MemoryRuntimeTarget:
    pass


@dataclass(frozen=True, slots=True)
class _FilesystemRuntimeTarget:
    path: Path


@dataclass(frozen=True, slots=True)
class _SqliteRuntimeTarget:
    path: Path


@dataclass(frozen=True, slots=True)
class _SqlRuntimeTarget:
    engine: "AsyncEngine"


class RuntimeStorage:
    """Declarative Runtime target and retention policy."""

    __slots__ = ("_target", "plan")

    def __init__(
        self,
        target: "_MemoryRuntimeTarget | _FilesystemRuntimeTarget | _SqliteRuntimeTarget | _SqlRuntimeTarget",
        plan: RuntimeStoragePlan,
    ) -> None:
        if not isinstance(plan, RuntimeStoragePlan):
            raise TypeError("plan must be a RuntimeStoragePlan")
        if not isinstance(target, (_MemoryRuntimeTarget, _FilesystemRuntimeTarget, _SqliteRuntimeTarget, _SqlRuntimeTarget)):
            raise TypeError("RuntimeStorage target is private and must be created by a classmethod")
        if isinstance(target, _MemoryRuntimeTarget) and any(
            route.retention is RuntimeRetention.DURABLE for route in plan.routes.values()
        ):
            raise ValueError("memory RuntimeStorage cannot contain durable routes")
        self._target = target
        self.plan = plan

    @property
    def target_kind(self) -> str:
        if isinstance(self._target, _MemoryRuntimeTarget):
            return "memory"
        if isinstance(self._target, _FilesystemRuntimeTarget):
            return "filesystem"
        if isinstance(self._target, _SqliteRuntimeTarget):
            return "sqlite"
        return "sql"

    @property
    def target_path(self) -> Path | None:
        if isinstance(self._target, (_FilesystemRuntimeTarget, _SqliteRuntimeTarget)):
            return self._target.path
        return None

    @property
    def target_engine(self) -> "AsyncEngine | None":
        if isinstance(self._target, _SqlRuntimeTarget):
            return self._target.engine
        return None

    @classmethod
    def memory(cls, *, plan: RuntimeStoragePlan | None = None) -> "RuntimeStorage":
        return cls(_MemoryRuntimeTarget(), plan or RuntimeStoragePlan.volatile())

    @classmethod
    def filesystem(
        cls,
        path: str | Path,
        *,
        plan: RuntimeStoragePlan | None = None,
    ) -> "RuntimeStorage":
        return cls(_FilesystemRuntimeTarget(_resolve_path(path)), plan or _default_plan())

    @classmethod
    def sqlite(
        cls,
        path: str | Path,
        *,
        plan: RuntimeStoragePlan | None = None,
    ) -> "RuntimeStorage":
        return cls(_SqliteRuntimeTarget(_resolve_path(path)), plan or _default_plan())

    @classmethod
    def sql(
        cls,
        engine: "AsyncEngine",
        *,
        plan: RuntimeStoragePlan | None = None,
    ) -> "RuntimeStorage":
        from sqlalchemy.ext.asyncio import AsyncEngine

        if not isinstance(engine, AsyncEngine):
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        return cls(_SqlRuntimeTarget(engine), plan or _default_plan())


def _default_plan() -> RuntimeStoragePlan:
    return RuntimeStoragePlan({RuntimeDomain.CONVERSATION: RuntimeStorageRoute.durable()})


def _resolve_path(value: str | Path) -> Path:
    if not isinstance(value, (str, Path)) or not str(value).strip() or str(value) == ":memory:":
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
    return Path(value).expanduser().resolve()


__all__ = [
    "RuntimeDomain",
    "RuntimeRetention",
    "RuntimeStorage",
    "RuntimeStoragePlan",
    "RuntimeStorageRoute",
]
