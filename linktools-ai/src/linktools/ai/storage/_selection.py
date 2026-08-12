#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime storage selection and target identity.

The runtime owns the physical target.  Callers only select the durable
domains; all unselected domains are backed by the process-local target.
"""

from collections.abc import Collection
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING

from ..errors import AIError, ErrorCode

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncEngine


class StorageDomain(StrEnum):
    CONVERSATION = "conversation"
    EXECUTION = "execution"
    MEMORY = "memory"
    ARTIFACT = "artifact"
    TASK = "task"
    EVALUATION = "evaluation"
    RECOVERY = "recovery"
    ASSET = "asset"
    ALL = "all"

    @classmethod
    def durable(cls) -> "frozenset[StorageDomain]":
        return frozenset(item for item in cls if item is not cls.ALL)


@dataclass(frozen=True, slots=True)
class RuntimeStorage:
    """Describe one durable target and its explicitly selected domains."""

    target_kind: str
    location: "Path | None"
    engine: "AsyncEngine | None"
    persist: frozenset[StorageDomain]

    @classmethod
    def memory(cls) -> "RuntimeStorage":
        return cls("memory", None, None, frozenset())

    @classmethod
    def filesystem(
        cls,
        path: "str | Path",
        *,
        persist: "Collection[StorageDomain] | StorageDomain | None" = None,
    ) -> "RuntimeStorage":
        location = _absolute_path(path)
        return cls("filesystem", location, None, _normalize_persist(persist))

    @classmethod
    def sqlite(
        cls,
        path: "str | Path",
        *,
        persist: "Collection[StorageDomain] | StorageDomain | None" = None,
    ) -> "RuntimeStorage":
        location = _absolute_path(path)
        if str(location) == ":memory:":
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        return cls("sqlite", location, None, _normalize_persist(persist))

    @classmethod
    def sql(
        cls,
        engine: "AsyncEngine",
        *,
        persist: "Collection[StorageDomain] | StorageDomain | None" = None,
    ) -> "RuntimeStorage":
        from sqlalchemy.ext.asyncio import AsyncEngine

        if not isinstance(engine, AsyncEngine):
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        return cls("sql", None, engine, _normalize_persist(persist))

    @property
    def owns_engine(self) -> bool:
        return self.target_kind == "sqlite"

    @property
    def durable(self) -> bool:
        return self.target_kind != "memory"


def _absolute_path(value: "str | Path") -> Path:
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
    if str(value) == ":memory:":
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
    path = Path(value).expanduser().resolve()
    return path


def _normalize_persist(
    value: "Collection[StorageDomain] | StorageDomain | None",
) -> frozenset[StorageDomain]:
    if value is None:
        return frozenset({StorageDomain.CONVERSATION})
    values = frozenset({value}) if isinstance(value, StorageDomain) else frozenset(value)
    if not all(isinstance(item, StorageDomain) for item in values):
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
    if StorageDomain.ALL in values:
        if len(values) != 1:
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        return StorageDomain.durable()
    return values


__all__ = ["RuntimeStorage", "StorageDomain"]
