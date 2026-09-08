#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Memory repository projection ordered by the durable logical path."""

from dataclasses import replace

from ...errors import AIError, ErrorCode
from ._contracts import MemoryRecord
from ._repositories import MemoryRepositoryImpl
from ._store import StoredRecord

_MAX_SORT_KEY_CHARS = 128


class MemoryPathRepository(MemoryRepositoryImpl):
    """Reuse Memory durable semantics with path-ordered keyset pagination."""

    def _stored(
        self,
        kind: str,
        identity: object,
        value: object,
        *,
        scope: bytes | None = None,
        parent: bytes | None = None,
        state: str | None = None,
    ) -> StoredRecord:
        record = super()._stored(
            kind,
            identity,
            value,
            scope=scope,
            parent=parent,
            state=state,
        )
        if not isinstance(value, MemoryRecord):
            return record
        path = value.metadata.get("path")
        if (
            not isinstance(path, str)
            or not path
            or not path.isascii()
            or len(path) > _MAX_SORT_KEY_CHARS
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return replace(record, sort_key=path)


__all__ = ["MemoryPathRepository"]
