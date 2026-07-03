#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ResourceFile / Operation / ResourceBackend: the generic resource-store contract.

Every backend operates on one unified `path: str` ("/{namespace}/{rest}") rather than
a separate namespace + relative-path pair -- ResourceStore normalizes paths before
handing them to any backend, so backends never parse or construct path segments
themselves.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol, runtime_checkable


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


@runtime_checkable
class ResourceBackend(Protocol):
    async def propfind(self, path: str) -> "list[ResourceFile]": ...

    async def get(self, path: str) -> "ResourceFile | None": ...

    async def get_at_version(self, path: str, version: int) -> "ResourceFile | None": ...

    async def get_by_name(self, namespace: str, name: str) -> "list[ResourceFile]": ...

    async def put(self, path: str, content: str, *, updated_by: str = "engine") -> ResourceFile: ...

    async def delete(self, path: str, *, updated_by: str = "engine") -> bool: ...

    async def move(self, src_path: str, dst_path: str, *, updated_by: str = "engine") -> "ResourceFile | None": ...

    async def list_since(self, since: "datetime | None") -> "list[ResourceFile]": ...

    async def apply_batch(self, ops: "list[Operation]", *, updated_by: str = "engine") -> "list[ResourceFile]": ...

    async def get_revision(self) -> int: ...
