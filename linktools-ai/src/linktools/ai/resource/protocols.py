#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ResourceFile / Operation / ResourceBackend: the generic resource-store contract.

Every backend operates on one unified `path: str` ("/{namespace}/{rest}") rather than
a separate namespace + relative-path pair -- ResourceStore normalizes paths before
handing them to any backend, so backends never parse or construct path segments
themselves.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime


@dataclass(slots=True, frozen=True)
class ResourceFile:
    path: str
    content: str
    version: int


@dataclass(slots=True, frozen=True)
class PutOp:
    path: str
    content: str


@dataclass(slots=True, frozen=True)
class DeleteOp:
    path: str


@dataclass(slots=True, frozen=True)
class MoveOp:
    src_path: str
    dst_path: str


Operation = PutOp | DeleteOp | MoveOp


class ResourceBackend(ABC):
    @abstractmethod
    async def get(self, path: str, version: "int | None" = None) -> "ResourceFile | None": ...

    @abstractmethod
    async def list(self, *, pattern: "str | None" = None, since: "datetime | None" = None) -> "list[ResourceFile]": ...

    @abstractmethod
    async def put(self, path: str, content: str, *, updated_by: str = "") -> ResourceFile: ...

    @abstractmethod
    async def delete(self, path: str, *, updated_by: str = "") -> bool: ...

    @abstractmethod
    async def move(self, src_path: str, dst_path: str, *, updated_by: str = "") -> "ResourceFile | None": ...

    @abstractmethod
    async def apply_batch(self, ops: "list[Operation]", *, updated_by: str = "") -> "list[ResourceFile]": ...

    @abstractmethod
    async def revision(self) -> int: ...
