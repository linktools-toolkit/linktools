#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Declarative Runtime domain routing."""

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING

from ...core import canonical_sha256
from ...errors import AIError, ErrorCode
from ._contracts import RuntimeDomain, RuntimeRetentionMode

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncEngine


class _RuntimeRouteKind(StrEnum):
    MEMORY = "memory"
    FILESYSTEM = "filesystem"
    SQLITE = "sqlite"
    SQL = "sql"


@dataclass(frozen=True, slots=True)
class RuntimeStateRoute:
    kind: str
    retention: RuntimeRetentionMode
    path: "Path | None" = None
    engine: "AsyncEngine | None" = None

    @classmethod
    def memory(cls) -> "RuntimeStateRoute":
        return cls(_RuntimeRouteKind.MEMORY.value, RuntimeRetentionMode.VOLATILE)

    @classmethod
    def transient(cls) -> "RuntimeStateRoute":
        return cls(_RuntimeRouteKind.MEMORY.value, RuntimeRetentionMode.TRANSIENT)

    @classmethod
    def filesystem(cls, path: "str | Path") -> "RuntimeStateRoute":
        return cls(_RuntimeRouteKind.FILESYSTEM.value, RuntimeRetentionMode.DURABLE, _normalize_path(path))

    @classmethod
    def sqlite(cls, path: "str | Path") -> "RuntimeStateRoute":
        normalized = _normalize_path(path)
        if str(path).strip() == ":memory:" or not str(normalized):
            raise ValueError("RuntimeStateRoute.sqlite requires a filesystem path")
        return cls(_RuntimeRouteKind.SQLITE.value, RuntimeRetentionMode.DURABLE, normalized)

    @classmethod
    def sql(cls, engine: "AsyncEngine") -> "RuntimeStateRoute":
        from sqlalchemy.ext.asyncio import AsyncEngine

        if not isinstance(engine, AsyncEngine):
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY, "SQL route requires an AsyncEngine")
        if engine.dialect.name == "sqlite" and engine.url.database in {None, "", ":memory:"}:
            raise ValueError("external SQL route requires durable SQLite or external SQL")
        return cls(_RuntimeRouteKind.SQL.value, RuntimeRetentionMode.DURABLE, engine=engine)

    @property
    def route_identity(self) -> str:
        if self.kind in {_RuntimeRouteKind.FILESYSTEM.value, _RuntimeRouteKind.SQLITE.value}:
            payload: object = {"kind": self.kind, "path": self.path.as_posix() if self.path is not None else None}
        elif self.kind == _RuntimeRouteKind.SQL.value:
            if self.engine is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            payload = {
                "kind": self.kind,
                "dialect": self.engine.dialect.name,
                "driver": self.engine.dialect.driver,
                "host": self.engine.url.host,
                "port": self.engine.url.port,
                "database": self.engine.url.database,
            }
        else:
            payload = {"kind": self.kind, "retention": self.retention.value}
        return canonical_sha256(payload)


@dataclass(frozen=True, slots=True)
class RuntimeStatePlan:
    conversation: RuntimeStateRoute = field(default_factory=RuntimeStateRoute.memory)
    execution: RuntimeStateRoute = field(default_factory=RuntimeStateRoute.memory)
    memory: RuntimeStateRoute = field(default_factory=RuntimeStateRoute.memory)
    artifact: RuntimeStateRoute = field(default_factory=RuntimeStateRoute.memory)
    task: RuntimeStateRoute = field(default_factory=RuntimeStateRoute.memory)
    evaluation: RuntimeStateRoute = field(default_factory=RuntimeStateRoute.memory)
    recovery: RuntimeStateRoute = field(default_factory=RuntimeStateRoute.memory)

    def __post_init__(self) -> None:
        routes = tuple(self.route(domain) for domain in RuntimeDomain)
        if any(not isinstance(route, RuntimeStateRoute) for route in routes):
            raise ValueError("RuntimeStatePlan contains an invalid route")
        filesystem_roots = [route.path for route in routes if route.kind == _RuntimeRouteKind.FILESYSTEM.value]
        if len(filesystem_roots) != len({path for path in filesystem_roots}):
            raise ValueError("filesystem RuntimeStateRoute path must be unique across RuntimeDomain values")

    def route(self, domain: RuntimeDomain) -> RuntimeStateRoute:
        if not isinstance(domain, RuntimeDomain):
            raise ValueError("RuntimeDomain is required")
        return {
            RuntimeDomain.CONVERSATION: self.conversation,
            RuntimeDomain.EXECUTION: self.execution,
            RuntimeDomain.MEMORY: self.memory,
            RuntimeDomain.ARTIFACT: self.artifact,
            RuntimeDomain.TASK: self.task,
            RuntimeDomain.EVALUATION: self.evaluation,
            RuntimeDomain.RECOVERY: self.recovery,
        }[domain]

    @property
    def durable_domains(self) -> frozenset[RuntimeDomain]:
        return frozenset(domain for domain in RuntimeDomain if self.route(domain).retention is RuntimeRetentionMode.DURABLE)


def _normalize_path(value: "str | Path") -> Path:
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise ValueError("RuntimeStateRoute path is required")
    return Path(value).expanduser().resolve(strict=False)


__all__ = ["RuntimeStatePlan", "RuntimeStateRoute"]
