#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Bounded ordinary record reads over source-owned transaction boundaries."""

from collections.abc import Mapping, Sequence
import hashlib
import json

from ...core import canonical_json_bytes
from ...errors import AIError
from ...storage import CacheBackend, StorageReadCache
from ._codec import decode_record, encode_record
from ._store import StateStore, StoredRecord, active_state_transaction


class StateRecordCache:
    """Share committed record bytes without making cached values a CAS input."""

    def __init__(self, store: StateStore, cache: object | None, *, scope: str) -> None:
        self._store = store
        self._scope = scope
        self._cache = StorageReadCache(cache) if isinstance(cache, CacheBackend) else None

    async def read_record(self, key: bytes) -> StoredRecord | None:
        records = await self.read_records((key,))
        return records.get(key)

    async def read_records(self, keys: Sequence[bytes]) -> Mapping[bytes, StoredRecord]:
        ordered = tuple(sorted(set(keys)))
        active = active_state_transaction(self._store)
        if active is not None:
            return await active.get_records(ordered)
        if not ordered:
            return {}
        if self._cache is None:
            return await self._store.read(lambda transaction: transaction.get_records(ordered))

        async def load() -> bytes:
            records = await self._store.read(lambda transaction: transaction.get_records(ordered))
            return canonical_json_bytes([encode_record(records[key]) for key in ordered if key in records])

        key_digest = hashlib.sha256(canonical_json_bytes([key.hex() for key in ordered])).hexdigest()
        cache_key = f"runtime-record:{self._scope}:{key_digest}"
        content = await self._cache.get(cache_key, load)
        try:
            if content is None:
                raise ValueError("cached record batch is absent")
            values = json.loads(content)
            if not isinstance(values, list):
                raise ValueError("cached record batch must be a list")
            result: dict[bytes, StoredRecord] = {}
            for value in values:
                record = decode_record(value)
                if record.key_digest not in ordered or record.key_digest in result:
                    raise ValueError("cached record identity does not match")
                result[record.key_digest] = record
            return result
        except (AIError, TypeError, ValueError):
            await self._cache.invalidate(cache_key)
            return await self._store.read(lambda transaction: transaction.get_records(ordered))

    async def invalidate(self) -> None:
        if self._cache is not None:
            await self._cache.clear()

    async def close(self) -> None:
        if self._cache is not None:
            await self._cache.close()


__all__ = ["StateRecordCache"]
