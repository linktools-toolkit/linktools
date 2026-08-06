#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Source, index and cache protocols."""

from typing import Protocol


class DocumentSource(Protocol):
    async def head(self) -> object: ...
    async def load(self, document_id: str) -> "object | None": ...
    async def changes(self, revision: "int | None" = None) -> "tuple[object, ...]": ...


class DocumentIndex(Protocol):
    async def refresh(self) -> object: ...
    async def resolve(self, document_id: str) -> "object | None": ...
    async def list(self) -> "tuple[object, ...]": ...


__all__ = ["DocumentIndex", "DocumentSource"]
