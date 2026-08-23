"""Declarative Runtime domain routing."""

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING

from ...core import canonical_sha256
from ...errors import AIError, ErrorCode

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncEngine


class RuntimeDomain(StrEnum):
    CONVERSATION = "conversation"
    EXECUTION = "execution"
    MEMORY = "memory"
    ARTIFACT = "artifact"
    TASK = "task"
    EVALUATION = "evaluation"
    RECOVERY = "recovery"


class RuntimeRetentionMode(StrEnum):
    DURABLE = "durable"
    VOLATILE = "volatile"
    TRANSIENT = "transient"


class _RuntimeStateBackendKind(StrEnum):
    MEMORY = "memory"
    FILESYSTEM = "filesystem"
    SQLITE = "sqlite"
    SQL = "sql"


@dataclass(frozen=True, slots=True, init=False)
class RuntimeStateRoute:
    _kind: _RuntimeStateBackendKind
    _retention: RuntimeRetentionMode
    _path: "Path | None"
    _transaction_root: "Path | None"
    _engine: "AsyncEngine | None"

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise TypeError("use RuntimeStateRoute factory methods")

    @classmethod
    def _create(
        cls,
        *,
        kind: _RuntimeStateBackendKind,
        retention: RuntimeRetentionMode,
        path: "Path | None" = None,
        transaction_root: "Path | None" = None,
        engine: "AsyncEngine | None" = None,
    ) -> "RuntimeStateRoute":
        value = object.__new__(cls)
        object.__setattr__(value, "_kind", kind)
        object.__setattr__(value, "_retention", retention)
        object.__setattr__(value, "_path", path)
        object.__setattr__(value, "_transaction_root", transaction_root)
        object.__setattr__(value, "_engine", engine)
        value._validate()
        return value

    def _validate(self) -> None:
        valid = (
            (
                self._kind is _RuntimeStateBackendKind.MEMORY
                and self._retention in {RuntimeRetentionMode.VOLATILE, RuntimeRetentionMode.TRANSIENT}
                and self._path is None
                and self._transaction_root is None
                and self._engine is None
            )
            or (
                self._kind is _RuntimeStateBackendKind.FILESYSTEM
                and self._retention is RuntimeRetentionMode.DURABLE
                and self._path is not None
                and self._engine is None
                and (
                    self._transaction_root is None
                    or self._transaction_root != self._path
                )
            )
            or (
                self._kind is _RuntimeStateBackendKind.SQLITE
                and self._retention is RuntimeRetentionMode.DURABLE
                and self._path is not None
                and self._transaction_root is None
                and self._engine is None
            )
            or (
                self._kind is _RuntimeStateBackendKind.SQL
                and self._retention is RuntimeRetentionMode.DURABLE
                and self._path is None
                and self._transaction_root is None
                and self._engine is not None
            )
        )
        if not valid:
            raise ValueError("RuntimeStateRoute has an invalid backend and retention combination")

    @property
    def kind(self) -> str:
        return self._kind.value

    @property
    def retention(self) -> RuntimeRetentionMode:
        return self._retention

    @property
    def path(self) -> "Path | None":
        return self._path

    @property
    def transaction_root(self) -> "Path | None":
        return self._transaction_root

    @property
    def engine(self) -> "AsyncEngine | None":
        return self._engine

    @classmethod
    def memory(cls) -> "RuntimeStateRoute":
        return cls._create(kind=_RuntimeStateBackendKind.MEMORY, retention=RuntimeRetentionMode.VOLATILE)

    @classmethod
    def transient(cls) -> "RuntimeStateRoute":
        return cls._create(kind=_RuntimeStateBackendKind.MEMORY, retention=RuntimeRetentionMode.TRANSIENT)

    @classmethod
    def filesystem(
        cls,
        path: "str | Path",
        *,
        transaction_root: "str | Path | None" = None,
    ) -> "RuntimeStateRoute":
        return cls._create(
            kind=_RuntimeStateBackendKind.FILESYSTEM,
            retention=RuntimeRetentionMode.DURABLE,
            path=_normalize_path(path),
            transaction_root=None if transaction_root is None else _normalize_path(transaction_root),
        )

    @classmethod
    def sqlite(cls, path: "str | Path") -> "RuntimeStateRoute":
        if not isinstance(path, (str, Path)) or not str(path).strip() or str(path).strip() == ":memory:":
            raise ValueError("RuntimeStateRoute.sqlite requires a filesystem path")
        normalized = _normalize_path(path)
        return cls._create(kind=_RuntimeStateBackendKind.SQLITE, retention=RuntimeRetentionMode.DURABLE, path=normalized)

    @classmethod
    def sql(cls, engine: "AsyncEngine") -> "RuntimeStateRoute":
        from sqlalchemy.ext.asyncio import AsyncEngine

        if not isinstance(engine, AsyncEngine):
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY, "SQL route requires an AsyncEngine")
        if engine.dialect.name == "sqlite" and engine.url.database in {None, "", ":memory:"}:
            raise ValueError("external SQL route requires durable SQLite or external SQL")
        return cls._create(kind=_RuntimeStateBackendKind.SQL, retention=RuntimeRetentionMode.DURABLE, engine=engine)

    @property
    def route_identity(self) -> str:
        if self._kind in {_RuntimeStateBackendKind.FILESYSTEM, _RuntimeStateBackendKind.SQLITE}:
            payload = {
                "kind": self.kind,
                "path": self.path.as_posix() if self.path is not None else None,
            }
        elif self._kind is _RuntimeStateBackendKind.SQL:
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
        filesystem_roots = [route.path for route in routes if route.kind == _RuntimeStateBackendKind.FILESYSTEM.value]
        if len(filesystem_roots) != len({path for path in filesystem_roots}):
            raise ValueError("filesystem RuntimeStateRoute path must be unique across RuntimeDomain values")
        grouped: dict[Path, list[Path]] = {}
        for route in routes:
            if route.kind != _RuntimeStateBackendKind.FILESYSTEM.value or route.transaction_root is None:
                continue
            if route.path is None:
                raise ValueError("filesystem route path is required")
            try:
                relative = route.path.relative_to(route.transaction_root)
            except ValueError as error:
                raise ValueError("filesystem route must be under transaction_root") from error
            if not relative.parts or relative.parts[0] in {".state-groups"} or relative.parts[0].startswith(".txn-"):
                raise ValueError("filesystem route points at a reserved transaction path")
            grouped.setdefault(route.transaction_root, []).append(route.path)
        for paths in grouped.values():
            for index, left in enumerate(sorted(paths)):
                for right in sorted(paths)[index + 1 :]:
                    if left in right.parents or right in left.parents:
                        raise ValueError("filesystem group member paths must be disjoint")

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


__all__ = ["RuntimeDomain", "RuntimeRetentionMode", "RuntimeStatePlan", "RuntimeStateRoute"]
