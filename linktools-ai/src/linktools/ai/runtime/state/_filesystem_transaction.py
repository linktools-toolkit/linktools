#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Filesystem StateStore transaction implementation."""

import asyncio
from collections.abc import Mapping, Sequence
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from linktools.core import environ

from ...errors import AIError, ErrorCode
from ._codec import encode_alias, encode_fact, encode_operation, encode_record
from ._filesystem_layout import (
    _CowMap, _FactStreamInfo, _FilesystemIndex, _RECORD_INDEX_MARKER, _RecordIndexNode,
    _alias_path, _decode_record_index_node_bytes, _fact_item_path, _fact_meta_path,
    _fact_subject_path, _json_bytes, _matches_record, _operation_path, _operation_ref_path,
    _read_fact_batch, _read_record_index_node, _read_sequence_metadata, _record_index_child, _record_index_common_prefix,
    _record_index_identity, _record_index_node_path, _record_index_node_payload,
    _record_index_replace_child, _record_path, _relative_path, _require_scan_limit, _sequence_path,
)
from ._store import (
    FactQuery, FactScanCursor, OperationQuery, OperationScanCursor, RecordQuery,
    RecordReplacement, RecordScanCursor, StoredAlias, StoredFact, StoredOperation, StoredRecord,
    validate_operation_replacement, validate_record_identity, validate_record_replacement,
)

_logger = environ.get_logger("ai.runtime.state.filesystem")


class _FilesystemTransaction:
    def __init__(
        self,
        root: Path,
        index: _FilesystemIndex,
        *,
        now: datetime | None = None,
    ) -> None:
        self._root = root
        self._cache = index.cache
        self.records = _CowMap(index.records)
        self.aliases = _CowMap(index.aliases)
        self.sequences = _CowMap(index.sequences)
        self.operations = _CowMap(index.operations)
        self.fact_streams = _CowMap(index.fact_streams)
        self._owned_fact_streams: set[bytes] = set()
        self._facts: dict[tuple[bytes, int], StoredFact] = {}
        self._deleted_facts: set[tuple[bytes, int]] = set()
        self.guarded_record_keys: set[bytes] = set()
        self._now = now
        self.writes: dict[str, bytes] = {}
        self.deletes: set[str] = set()

    @property
    def has_changes(self) -> bool:
        return bool(
            self.records.changes()
            or self.records.deleted()
            or self.aliases.changes()
            or self.aliases.deleted()
            or self.sequences.changes()
            or self.sequences.deleted()
            or self.fact_streams.changes()
            or self.fact_streams.deleted()
            or self.operations.changes()
            or self.operations.deleted()
            or self.writes
            or self.deletes
        )

    async def now(self) -> datetime:
        if self._now is None:
            self._now = datetime.now(timezone.utc)
        return self._now

    async def validate_integrity(self) -> None:
        aliases = dict(await asyncio.to_thread(self._cache.list_aliases))
        aliases.update(self.aliases.changes())
        for alias in self.aliases.deleted():
            aliases.pop(alias, None)
        for key in aliases.values():
            if await self.get_record(key) is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        streams = {
            info.stream_digest: info
            for info in await asyncio.to_thread(self._cache.list_fact_streams)
        }
        streams.update(self.fact_streams.changes())
        for stream in self.fact_streams.deleted():
            streams.pop(stream, None)
        for info in streams.values():
            if await self.get_record(info.owner_key_digest) is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

    async def get_record(self, key: bytes) -> StoredRecord | None:
        if key in self.records.deleted():
            return None
        if key in self.records:
            return self.records[key]
        return await asyncio.to_thread(self._cache.get_record, key)

    async def get_records(self, keys: Sequence[bytes]) -> Mapping[bytes, StoredRecord]:
        unique_keys = tuple(dict.fromkeys(keys))
        if not unique_keys:
            return {}
        cached = await asyncio.to_thread(
            lambda: {key: self._cache.get_record(key) for key in unique_keys}
        )
        values = {key: value for key, value in cached.items() if value is not None}
        values.update(
            key_value
            for key_value in self.records.changes().items()
            if key_value[0] in unique_keys
        )
        for key in self.records.deleted():
            values.pop(key, None)
        return values

    async def insert_record(self, record: StoredRecord) -> None:
        await self.insert_records((record,))

    async def insert_records(self, records: Sequence[StoredRecord]) -> None:
        values = tuple(records)
        keys = [record.key_digest for record in values]
        if len(keys) != len(set(keys)):
            raise ValueError("insert_records contains duplicate keys")
        if not values:
            return
        if await self.get_records(keys):
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        for record in sorted(values, key=lambda value: value.key_digest):
            validate_record_identity(record)
            self.records[record.key_digest] = record
            self.guarded_record_keys.add(record.key_digest)
            self._write(_record_path(record), encode_record(record))
            self._sync_record_index(None, record)

    async def guard_record(
        self, key: bytes, *, expected_storage_version: int
    ) -> StoredRecord | None:
        if (
            isinstance(expected_storage_version, bool)
            or not isinstance(expected_storage_version, int)
            or expected_storage_version < 0
        ):
            raise ValueError("expected_storage_version must be a non-negative integer")
        current = await self.get_record(key)
        if key in self.guarded_record_keys:
            return current
        if current is None or current.storage_version != expected_storage_version:
            return None
        guarded = StoredRecord(
            current.key_digest,
            current.partition_digest,
            current.scope_digest,
            current.parent_digest,
            current.kind,
            current.sort_key,
            current.state,
            expected_storage_version + 1,
            current.lease_owner,
            current.lease_fence,
            current.lease_expires_at,
            current.data,
        )
        self.records[key] = guarded
        self.guarded_record_keys.add(key)
        self._write(_record_path(guarded), encode_record(guarded))
        return guarded

    async def replace_record(
        self, record: StoredRecord, *, expected_storage_version: int
    ) -> bool:
        try:
            await self.replace_records(
                (RecordReplacement(record, expected_storage_version),)
            )
        except AIError as error:
            if error.code is ErrorCode.STORAGE_CONFLICT:
                return False
            raise
        return True

    async def replace_records(self, replacements: Sequence[RecordReplacement]) -> None:
        values = tuple(replacements)
        keys = [replacement.record.key_digest for replacement in values]
        if len(keys) != len(set(keys)):
            raise ValueError("replace_records contains duplicate keys")
        if not values:
            return
        current_values = await self.get_records(keys)
        candidates: list[tuple[StoredRecord, StoredRecord]] = []
        for replacement in sorted(values, key=lambda value: value.record.key_digest):
            current = current_values.get(replacement.record.key_digest)
            if (
                current is None
                or current.storage_version != replacement.expected_storage_version
            ):
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            validate_record_replacement(current, replacement.record)
            validate_record_identity(replacement.record)
            if (
                replacement.record.storage_version
                != replacement.expected_storage_version + 1
            ):
                raise ValueError(
                    "replacement must increment storage_version exactly once"
                )
            candidates.append((current, replacement.record))
        for current, record in candidates:
            self.records[record.key_digest] = record
            self.guarded_record_keys.add(record.key_digest)
            self._write(_record_path(record), encode_record(record))
            self._sync_record_index(current, record)

    async def update_record_lease(
        self,
        key: bytes,
        *,
        expected_storage_version: int,
        lease_owner: str | None,
        lease_fence: int,
        lease_expires_at: datetime | None,
    ) -> bool:
        if (
            isinstance(expected_storage_version, bool)
            or not isinstance(expected_storage_version, int)
            or expected_storage_version < 0
            or isinstance(lease_fence, bool)
            or not isinstance(lease_fence, int)
            or lease_fence < 0
        ):
            raise ValueError("record lease integer fields are invalid")
        current = await self.get_record(key)
        if current is None or current.storage_version != expected_storage_version:
            return False
        if lease_expires_at is not None and lease_expires_at.tzinfo is None:
            raise ValueError("record lease is invalid")
        updated = StoredRecord(
            current.key_digest,
            current.partition_digest,
            current.scope_digest,
            current.parent_digest,
            current.kind,
            current.sort_key,
            current.state,
            expected_storage_version + 1,
            lease_owner,
            lease_fence,
            lease_expires_at,
            current.data,
        )
        self.records[key] = updated
        self.guarded_record_keys.add(key)
        self._write(_record_path(updated), encode_record(updated))
        return True

    async def delete_record(
        self, key: bytes, *, expected_storage_version: int | None = None
    ) -> bool:
        if expected_storage_version is not None and (
            isinstance(expected_storage_version, bool)
            or not isinstance(expected_storage_version, int)
            or expected_storage_version < 0
        ):
            raise ValueError(
                "expected_storage_version must be a non-negative integer or None"
            )
        current = await self.get_record(key)
        if current is None:
            return False
        expected = (
            current.storage_version
            if expected_storage_version is None
            else expected_storage_version
        )
        if await self.guard_record(key, expected_storage_version=expected) is None:
            return False
        await self.delete_fact_streams(key)
        aliases = dict(await asyncio.to_thread(self._cache.list_aliases))
        aliases.update(self.aliases.changes())
        for alias, record_key in tuple(aliases.items()):
            if record_key == key:
                self.aliases[alias] = record_key
                del self.aliases[alias]
                self._delete(_alias_path(self._root, alias))
        del self.records[key]
        self.guarded_record_keys.discard(key)
        self._delete(_record_path(current))
        self._sync_record_index(current, None)
        return True

    async def list_records(self, query: RecordQuery) -> tuple[StoredRecord, ...]:
        cache_query = query
        if query.limit is not None and (
            self.records.changes() or self.records.deleted()
        ):
            pending = len(self.records.changes()) + len(self.records.deleted())
            cache_query = replace(
                query,
                limit=(
                    query.limit + pending if query.limit + pending <= 1000 else None
                ),
            )
        values = {
            record.key_digest: record
            for record in await asyncio.to_thread(
                self._cache.list_records,
                cache_query,
            )
        }
        values.update(self.records.changes())
        for key in self.records.deleted():
            values.pop(key, None)
        values = [
            record for record in values.values() if _matches_record(record, query)
        ]
        values.sort(key=lambda record: (record.sort_key, record.key_digest))
        if query.after_sort_key is not None and query.after_key_digest is not None:
            values = [
                record
                for record in values
                if (record.sort_key, record.key_digest)
                > (query.after_sort_key, query.after_key_digest)
            ]
        if query.limit is not None:
            values = values[: query.limit]
        return tuple(values)

    async def scan_records(self) -> tuple[StoredRecord, ...]:
        values = {
            record.key_digest: record
            for record in await asyncio.to_thread(self._cache.scan_records)
        }
        values.update(self.records.changes())
        for key in self.records.deleted():
            values.pop(key, None)
        return tuple(values.values())

    async def scan_records_page(
        self,
        *,
        after: RecordScanCursor | None,
        limit: int,
    ) -> tuple[StoredRecord, ...]:
        _require_scan_limit(limit)
        local = sorted(
            (
                value
                for key, value in self.records.changes().items()
                if key not in self.records.deleted()
                and (
                    after is None or (value.kind, key) > (after.kind, after.key_digest)
                )
            ),
            key=lambda value: (value.kind, value.key_digest),
        )
        local_index = 0
        base_page: tuple[StoredRecord, ...] = ()
        base_index = 0
        base_after = after
        base_done = False
        result: list[StoredRecord] = []
        while len(result) < limit:
            if base_index >= len(base_page) and not base_done:
                base_page = await asyncio.to_thread(
                    self._cache.scan_records_page,
                    after=base_after,
                    limit=limit,
                )
                base_index = 0
                if not base_page:
                    base_done = True
                else:
                    last = base_page[-1]
                    base_after = RecordScanCursor(last.kind, last.key_digest)
                    base_done = len(base_page) < limit
            base_value = None if base_index >= len(base_page) else base_page[base_index]
            local_value = None if local_index >= len(local) else local[local_index]
            if base_value is None and local_value is None:
                break
            if local_value is None or (
                base_value is not None
                and (base_value.kind, base_value.key_digest)
                < (local_value.kind, local_value.key_digest)
            ):
                value = base_value
                base_index += 1
            else:
                value = local_value
                local_index += 1
                if base_value is not None and (
                    base_value.kind,
                    base_value.key_digest,
                ) == (value.kind, value.key_digest):
                    base_index += 1
            if value is not None and value.key_digest not in self.records.deleted():
                result.append(value)
        return tuple(result)

    async def resolve_alias(self, alias: bytes) -> bytes | None:
        return (await self.resolve_aliases((alias,))).get(alias)

    async def resolve_aliases(self, aliases: Sequence[bytes]) -> Mapping[bytes, bytes]:
        unique_aliases = tuple(dict.fromkeys(aliases))
        if not unique_aliases:
            return {}
        cached = await asyncio.to_thread(
            lambda: {alias: self._cache.get_alias(alias) for alias in unique_aliases}
        )
        values = {alias: value for alias, value in cached.items() if value is not None}
        values.update(
            alias_value
            for alias_value in self.aliases.changes().items()
            if alias_value[0] in unique_aliases
        )
        for alias in self.aliases.deleted():
            values.pop(alias, None)
        records = await self.get_records(tuple(values.values()))
        if len(records) != len(set(values.values())):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return values

    async def scan_aliases(self) -> tuple[StoredAlias, ...]:
        values = dict(await asyncio.to_thread(self._cache.list_aliases))
        values.update(self.aliases.changes())
        for alias in self.aliases.deleted():
            values.pop(alias, None)
        return tuple(
            StoredAlias(alias, record_key)
            for alias, record_key in sorted(values.items())
        )

    async def insert_alias(self, alias: StoredAlias) -> None:
        await self.insert_aliases((alias,))

    async def insert_aliases(self, aliases: Sequence[StoredAlias]) -> None:
        values = tuple(sorted(aliases, key=lambda value: value.alias_digest))
        if not values:
            return
        existing = await self.resolve_aliases(
            tuple(value.alias_digest for value in values)
        )
        for alias in values:
            current = existing.get(alias.alias_digest)
            if current is not None and current != alias.record_key_digest:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            if alias.record_key_digest not in self.guarded_record_keys:
                raise RuntimeError(
                    "alias owner must be guarded in the current transaction"
                )
            self.aliases[alias.alias_digest] = alias.record_key_digest
            self._write(
                _alias_path(self._root, alias.alias_digest), encode_alias(alias)
            )

    async def get_sequence(self, key: bytes) -> int:
        if key in self.sequences.deleted():
            return 0
        if key in self.sequences:
            return self.sequences[key]
        return await asyncio.to_thread(self._cache.get_sequence, key)

    async def get_sequences(self, keys: Sequence[bytes]) -> Mapping[bytes, int]:
        unique_keys = tuple(dict.fromkeys(keys))
        if not unique_keys:
            return {}
        cached = await asyncio.to_thread(
            lambda: {key: self._cache.get_sequence(key) for key in unique_keys}
        )
        cached.update(self.sequences.changes())
        for key in self.sequences.deleted():
            cached[key] = 0
        return cached

    async def scan_sequences(self) -> Mapping[bytes, int]:
        values: dict[bytes, int] = {}
        root = self._root / "sequences"
        for path in root.glob("*/*.json"):
            key, value = await asyncio.to_thread(_read_sequence_metadata, path)
            values[key] = value
        values.update(self.sequences.changes())
        for key in self.sequences.deleted():
            values.pop(key, None)
        return dict(sorted(values.items()))

    async def next_sequence(self, key: bytes) -> int:
        value = await self.get_sequence(key) + 1
        self.sequences[key] = value
        self._write(_sequence_path(self._root, key), {"key": key.hex(), "value": value})
        return value

    async def reserve_sequence(self, key: bytes, count: int) -> int:
        return (await self.reserve_sequences({key: count}))[key]

    async def reserve_sequences(
        self, requests: Mapping[bytes, int]
    ) -> Mapping[bytes, int]:
        if any(
            isinstance(count, bool) or not isinstance(count, int) or count < 1
            for count in requests.values()
        ):
            raise ValueError("sequence reservation count must be a positive integer")
        current = await self.get_sequences(tuple(requests))
        values = {key: current[key] + requests[key] for key in sorted(requests)}
        for key, value in values.items():
            self.sequences[key] = value
            self._write(
                _sequence_path(self._root, key), {"key": key.hex(), "value": value}
            )
        return values

    async def advance_sequence(self, key: bytes, expected: int) -> int:
        if isinstance(expected, bool) or not isinstance(expected, int) or expected < 0:
            raise ValueError("expected sequence must be a non-negative integer")
        if await self.get_sequence(key) != expected:
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        return await self.next_sequence(key)

    async def delete_sequence(self, key: bytes) -> None:
        await self.delete_sequences((key,))

    async def delete_sequences(self, keys: Sequence[bytes]) -> None:
        for key in sorted(set(keys)):
            if key in self.sequences:
                del self.sequences[key]
            else:
                self.sequences[key] = 0
                del self.sequences[key]
            self._delete(_sequence_path(self._root, key))

    async def insert_fact(self, fact: StoredFact) -> None:
        await self.insert_facts((fact,))

    async def insert_facts(self, facts: Sequence[StoredFact]) -> None:
        if not facts:
            return
        streams: dict[bytes, list[StoredFact]] = {}
        seen: set[tuple[bytes, int]] = set()
        for fact in facts:
            if fact.owner_key_digest not in self.guarded_record_keys:
                raise RuntimeError(
                    "fact owner must be guarded in the current transaction"
                )
            key = (fact.stream_digest, fact.sequence)
            if key in seen:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            seen.add(key)
            streams.setdefault(fact.stream_digest, []).append(fact)

        owned: dict[bytes, _FactStreamInfo] = {}
        for stream, values in streams.items():
            info = await self._own_fact_stream(stream)
            if info is None:
                info = _FactStreamInfo(stream, values[0].owner_key_digest, 0, {})
                self.fact_streams[stream] = info
                self._owned_fact_streams.add(stream)
            for fact in values:
                if info.owner_key_digest != fact.owner_key_digest:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                if fact.sequence <= info.last_sequence:
                    raise AIError(ErrorCode.STORAGE_CONFLICT)
                if fact.sequence != info.last_sequence + 1:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                info.last_sequence = fact.sequence
            owned[stream] = info

        subject_count = 0
        for stream, values in streams.items():
            info = owned[stream]
            subjects: dict[bytes, int] = {}
            for fact in values:
                key = (fact.stream_digest, fact.sequence)
                self._deleted_facts.discard(key)
                self._facts[key] = fact
                self._write(
                    _fact_item_path(self._root, fact.stream_digest, fact.sequence),
                    encode_fact(fact),
                )
                if (
                    fact.subject_digest is not None
                    and subjects.get(fact.subject_digest, 0) < fact.sequence
                ):
                    subjects[fact.subject_digest] = fact.sequence
            for subject, sequence in subjects.items():
                if info.subjects.get(subject, 0) < sequence:
                    info.subjects[subject] = sequence
            subject_count += len(subjects)
            self._sync_fact_stream(info, subjects=subjects)
        _logger.debug(
            "filesystem fact batch staged: facts=%s streams=%s subjects=%s",
            len(facts),
            len(streams),
            subject_count,
        )

    async def list_facts(self, query: FactQuery) -> tuple[StoredFact, ...]:
        if query.stream_digest in self.fact_streams.deleted():
            return ()
        info = self.fact_streams.get(query.stream_digest)
        if info is None and query.stream_digest not in self.fact_streams:
            info = await asyncio.to_thread(
                self._cache.get_fact_stream, query.stream_digest
            )
        if info is None:
            return ()
        if query.latest:
            if query.subject_digest is None:
                latest = info.last_sequence
            else:
                await self._load_fact_subjects(info)
                latest = info.subjects.get(query.subject_digest)
            if (
                latest is None
                or query.after_sequence is not None
                and latest <= query.after_sequence
            ):
                return ()
            await self._load_facts(info, (latest,))
            value = self._facts.get((info.stream_digest, latest))
            return () if value is None else (value,)
        if query.latest_per_subject:
            await self._load_fact_subjects(info)
            sequences = tuple(
                sequence
                for sequence in info.subjects.values()
                if query.after_sequence is None or sequence > query.after_sequence
            )
            await self._load_facts(info, sequences)
            values = [
                self._facts.get((info.stream_digest, sequence))
                for sequence in sequences
            ]
            values = [value for value in values if value is not None]
            values.sort(key=lambda value: value.sequence)
            if query.limit is not None:
                values = values[: query.limit]
            return tuple(values)
        start = 1 if query.after_sequence is None else query.after_sequence + 1
        end = (
            info.last_sequence + 1
            if query.limit is None
            else min(
                info.last_sequence + 1,
                start + query.limit,
            )
        )
        sequences = range(start, end)
        await self._load_facts(info, tuple(sequences))
        values = [
            self._facts.get((info.stream_digest, sequence)) for sequence in sequences
        ]
        values = [value for value in values if value is not None]
        if query.subject_digest is not None:
            values = [
                value
                for value in values
                if value.subject_digest == query.subject_digest
            ]
        if query.limit is not None:
            values = values[: query.limit]
        return tuple(values)

    async def scan_facts(self) -> tuple[StoredFact, ...]:
        values = {
            (fact.stream_digest, fact.sequence): fact
            for fact in await asyncio.to_thread(self._cache.scan_facts)
        }
        values.update(self._facts)
        for key in self._deleted_facts:
            values.pop(key, None)
        return tuple(values.values())

    async def scan_facts_page(
        self,
        *,
        after: FactScanCursor | None,
        limit: int,
    ) -> tuple[StoredFact, ...]:
        _require_scan_limit(limit)
        local = sorted(
            (
                value
                for key, value in self._facts.items()
                if key not in self._deleted_facts
                and (
                    after is None
                    or (value.stream_digest, value.sequence)
                    > (after.stream_digest, after.sequence)
                )
            ),
            key=lambda value: (value.stream_digest, value.sequence),
        )
        local_index = 0
        base_page: tuple[StoredFact, ...] = ()
        base_index = 0
        base_after = after
        base_done = False
        result: list[StoredFact] = []
        while len(result) < limit:
            if base_index >= len(base_page) and not base_done:
                base_page = await asyncio.to_thread(
                    self._cache.scan_facts_page,
                    after=base_after,
                    limit=limit,
                )
                base_index = 0
                if not base_page:
                    base_done = True
                else:
                    last = base_page[-1]
                    base_after = FactScanCursor(
                        last.stream_digest,
                        last.sequence,
                    )
                    base_done = len(base_page) < limit
            base_value = None if base_index >= len(base_page) else base_page[base_index]
            local_value = None if local_index >= len(local) else local[local_index]
            if base_value is None and local_value is None:
                break
            if local_value is None or (
                base_value is not None
                and (base_value.stream_digest, base_value.sequence)
                < (local_value.stream_digest, local_value.sequence)
            ):
                value = base_value
                base_index += 1
            else:
                value = local_value
                local_index += 1
                if base_value is not None and (
                    base_value.stream_digest,
                    base_value.sequence,
                ) == (value.stream_digest, value.sequence):
                    base_index += 1
            if (
                value is not None
                and (
                    value.stream_digest,
                    value.sequence,
                )
                not in self._deleted_facts
            ):
                result.append(value)
        return tuple(result)

    async def delete_fact_streams(self, owner_key: bytes) -> None:
        sources = {
            info.stream_digest: info
            for info in await asyncio.to_thread(self._cache.list_fact_streams)
        }
        sources.update(self.fact_streams.changes())
        for stream in self.fact_streams.deleted():
            sources.pop(stream, None)
        for stream, source in tuple(sources.items()):
            if source.owner_key_digest != owner_key:
                continue
            info = await self._own_fact_stream(stream)
            if info is None:
                continue
            await self._load_fact_subjects(info)
            for sequence in range(1, info.last_sequence + 1):
                self._deleted_facts.add((info.stream_digest, sequence))
                self._delete(_fact_item_path(self._root, info.stream_digest, sequence))
            info.last_sequence = 0
            self._sync_fact_stream(info)
            del self.fact_streams[stream]

    async def insert_operation(self, value: StoredOperation) -> None:
        existing = await self.get_operation(value.key_digest)
        stream_existing = await self._get_operation_by_stream_sequence(
            value.stream_digest,
            value.sequence,
        )
        if existing is not None or stream_existing is not None:
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        self.operations[value.key_digest] = value
        self._write(_operation_path(self._root, value), encode_operation(value))
        self._write(
            _operation_ref_path(self._root, value), {"key": value.key_digest.hex()}
        )

    async def _get_operation_by_stream_sequence(
        self,
        stream_digest: bytes,
        sequence: int,
    ) -> StoredOperation | None:
        for operation in self.operations.changes().values():
            if (
                operation.stream_digest == stream_digest
                and operation.sequence == sequence
            ):
                return operation
        value = await asyncio.to_thread(
            self._cache.get_operation_by_stream_sequence,
            stream_digest,
            sequence,
        )
        if value is not None and value.key_digest in self.operations.deleted():
            return None
        return value

    async def get_operation(self, key: bytes) -> StoredOperation | None:
        if key in self.operations.deleted():
            return None
        if key in self.operations:
            return self.operations[key]
        return await asyncio.to_thread(self._cache.get_operation, key)

    async def replace_operation(
        self, value: StoredOperation, *, expected_state: str
    ) -> bool:
        current = await self.get_operation(value.key_digest)
        if current is None or current.state != expected_state:
            return False
        validate_operation_replacement(current, value)
        self.operations[value.key_digest] = value
        self._write(_operation_path(self._root, value), encode_operation(value))
        return True

    async def list_operations(
        self, query: OperationQuery
    ) -> tuple[StoredOperation, ...]:
        if query.stream_digest is None:
            source = await asyncio.to_thread(self._cache.list_operations)
        else:
            source = await asyncio.to_thread(
                self._cache.list_operation_stream,
                query.stream_digest,
            )
        values_by_key = {item.key_digest: item for item in source}
        values_by_key.update(self.operations.changes())
        for key in self.operations.deleted():
            values_by_key.pop(key, None)
        values = [
            item
            for item in values_by_key.values()
            if (
                query.stream_digest is None or item.stream_digest == query.stream_digest
            )
            and (query.states is None or item.state in query.states)
            and (
                query.through_sequence is None
                or item.sequence <= query.through_sequence
            )
            and (query.compactable is None or item.compactable == query.compactable)
        ]
        values.sort(key=lambda item: (item.sequence, item.key_digest))
        if query.limit is not None:
            values = values[: query.limit]
        return tuple(values)

    async def scan_operations(self) -> tuple[StoredOperation, ...]:
        values = {
            item.key_digest: item
            for item in await asyncio.to_thread(self._cache.scan_operations)
        }
        values.update(self.operations.changes())
        for key in self.operations.deleted():
            values.pop(key, None)
        return tuple(values.values())

    async def scan_operations_page(
        self,
        *,
        after: OperationScanCursor | None,
        limit: int,
    ) -> tuple[StoredOperation, ...]:
        _require_scan_limit(limit)
        local = sorted(
            (
                value
                for key, value in self.operations.changes().items()
                if key not in self.operations.deleted()
                and (after is None or key > after.key_digest)
            ),
            key=lambda value: value.key_digest,
        )
        local_index = 0
        base_page: tuple[StoredOperation, ...] = ()
        base_index = 0
        base_after = after
        base_done = False
        result: list[StoredOperation] = []
        while len(result) < limit:
            if base_index >= len(base_page) and not base_done:
                base_page = await asyncio.to_thread(
                    self._cache.scan_operations_page,
                    after=base_after,
                    limit=limit,
                )
                base_index = 0
                if not base_page:
                    base_done = True
                else:
                    last = base_page[-1]
                    base_after = OperationScanCursor(last.key_digest)
                    base_done = len(base_page) < limit
            base_value = None if base_index >= len(base_page) else base_page[base_index]
            local_value = None if local_index >= len(local) else local[local_index]
            if base_value is None and local_value is None:
                break
            if local_value is None or (
                base_value is not None
                and base_value.key_digest < local_value.key_digest
            ):
                value = base_value
                base_index += 1
            else:
                value = local_value
                local_index += 1
                if base_value is not None and base_value.key_digest == value.key_digest:
                    base_index += 1
            if value is not None and value.key_digest not in self.operations.deleted():
                result.append(value)
        return tuple(result)

    async def delete_operations(
        self, query: OperationQuery
    ) -> tuple[StoredOperation, ...]:
        values = await self.list_operations(query)
        for value in values:
            self.operations[value.key_digest] = value
            del self.operations[value.key_digest]
            self._delete(_operation_path(self._root, value))
            self._delete(_operation_ref_path(self._root, value))
        return values

    async def _load_facts(
        self, info: _FactStreamInfo, sequences: Sequence[int]
    ) -> None:
        missing = tuple(
            sequence
            for sequence in sequences
            if (info.stream_digest, sequence) not in self._facts
            and (info.stream_digest, sequence) not in self._deleted_facts
        )
        if not missing:
            return
        values = await asyncio.to_thread(
            _read_fact_batch, self._root, info.stream_digest, missing
        )
        self._facts.update(values)

    async def _own_fact_stream(self, stream: bytes) -> _FactStreamInfo | None:
        if stream in self.fact_streams.deleted():
            return None
        info = self.fact_streams.get(stream)
        if info is None and stream not in self.fact_streams:
            info = await asyncio.to_thread(self._cache.get_fact_stream, stream)
        if info is None or stream in self._owned_fact_streams:
            return info
        owned = _FactStreamInfo(
            info.stream_digest,
            info.owner_key_digest,
            info.last_sequence,
            dict(info.subjects),
            info.subjects_loaded,
        )
        self.fact_streams[stream] = owned
        self._owned_fact_streams.add(stream)
        return owned

    async def _load_fact_subjects(self, info: _FactStreamInfo) -> None:
        if info.subjects_loaded:
            return
        await asyncio.to_thread(self._cache.load_fact_subjects, info)

    def _sync_fact_stream(
        self,
        info: _FactStreamInfo,
        *,
        subjects: Mapping[bytes, int] | None = None,
    ) -> None:
        if info.last_sequence == 0:
            self._delete(_fact_meta_path(self._root, info.stream_digest))
            for subject_digest in tuple(info.subjects):
                self._delete(
                    _fact_subject_path(self._root, info.stream_digest, subject_digest)
                )
            info.subjects.clear()
            return
        self._write(
            _fact_meta_path(self._root, info.stream_digest),
            {
                "stream": info.stream_digest.hex(),
                "owner_key": info.owner_key_digest.hex(),
                "last_sequence": info.last_sequence,
            },
        )
        for subject, sequence in ({} if subjects is None else subjects).items():
            self._write(
                _fact_subject_path(self._root, info.stream_digest, subject),
                {"sequence": sequence},
            )

    def _sync_record_index(
        self,
        previous: StoredRecord | None,
        current: StoredRecord | None,
    ) -> None:
        if (
            not self._cache.record_index_complete
            or _RECORD_INDEX_MARKER in self.deletes
        ):
            return
        previous_identity = _record_index_identity(previous)
        current_identity = _record_index_identity(current)
        if previous_identity == current_identity:
            return
        if previous_identity is not None:
            self._remove_record_index_entry(previous_identity)
        if current_identity is not None:
            self._add_record_index_entry(current_identity)

    def _record_index_node(
        self,
        kind: str,
        scope_digest: bytes,
        token: str,
    ) -> _RecordIndexNode | None:
        relative = _record_index_node_path(kind, scope_digest, token)
        if relative in self.deletes:
            return None
        pending = self.writes.get(relative)
        if pending is not None:
            return _decode_record_index_node_bytes(pending, expected_token=token)
        return _read_record_index_node(self._root, kind, scope_digest, token)

    def _store_record_index_node(
        self,
        kind: str,
        scope_digest: bytes,
        node: _RecordIndexNode,
    ) -> None:
        self._write(
            _record_index_node_path(kind, scope_digest, node.token),
            _record_index_node_payload(node),
        )

    def _add_record_index_entry(
        self,
        identity: tuple[str, bytes, str, bytes],
    ) -> None:
        kind, scope_digest, token, key_digest = identity
        current = self._record_index_node(kind, scope_digest, "")
        if current is None:
            current = _RecordIndexNode("", (), ())
        while True:
            if token == current.token:
                if key_digest in current.key_digests:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                updated = _RecordIndexNode(
                    current.token,
                    tuple(sorted((*current.key_digests, key_digest))),
                    current.children,
                )
                self._store_record_index_node(kind, scope_digest, updated)
                return
            if not token.startswith(current.token):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            child_token = _record_index_child(current, token)
            if child_token is None:
                leaf = _RecordIndexNode(token, (key_digest,), ())
                self._store_record_index_node(kind, scope_digest, leaf)
                updated = _RecordIndexNode(
                    current.token,
                    current.key_digests,
                    tuple(sorted((*current.children, token))),
                )
                self._store_record_index_node(kind, scope_digest, updated)
                return
            child = self._record_index_node(kind, scope_digest, child_token)
            if child is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            common = _record_index_common_prefix((token, child_token))
            if common == child_token:
                current = child
                continue
            if len(common) <= len(current.token):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if common == token:
                inserted = _RecordIndexNode(token, (key_digest,), (child_token,))
                self._store_record_index_node(kind, scope_digest, inserted)
                updated = _record_index_replace_child(current, child_token, token)
                self._store_record_index_node(kind, scope_digest, updated)
                return
            leaf = _RecordIndexNode(token, (key_digest,), ())
            branch = _RecordIndexNode(
                common,
                (),
                tuple(sorted((child_token, token))),
            )
            self._store_record_index_node(kind, scope_digest, leaf)
            self._store_record_index_node(kind, scope_digest, branch)
            updated = _record_index_replace_child(current, child_token, common)
            self._store_record_index_node(kind, scope_digest, updated)
            return

    def _remove_record_index_entry(
        self,
        identity: tuple[str, bytes, str, bytes],
    ) -> None:
        kind, scope_digest, token, key_digest = identity
        root = self._record_index_node(kind, scope_digest, "")
        if root is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        path = [root]
        current = root
        while current.token != token:
            if not token.startswith(current.token):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            child_token = _record_index_child(current, token)
            if child_token is None or not token.startswith(child_token):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            child = self._record_index_node(kind, scope_digest, child_token)
            if child is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            path.append(child)
            current = child
        if key_digest not in current.key_digests:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        updated = _RecordIndexNode(
            current.token,
            tuple(key for key in current.key_digests if key != key_digest),
            current.children,
        )
        if current.token == "":
            if updated.key_digests or updated.children:
                self._store_record_index_node(kind, scope_digest, updated)
            else:
                self._delete(_record_index_node_path(kind, scope_digest, ""))
            return
        if updated.key_digests or len(updated.children) >= 2:
            self._store_record_index_node(kind, scope_digest, updated)
            return
        removed_token = current.token
        replacement = updated.children[0] if updated.children else None
        self._delete(_record_index_node_path(kind, scope_digest, removed_token))
        for parent in reversed(path[:-1]):
            children = [child for child in parent.children if child != removed_token]
            if len(children) == len(parent.children):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if replacement is not None:
                children.append(replacement)
            next_parent = _RecordIndexNode(
                parent.token,
                parent.key_digests,
                tuple(sorted(children)),
            )
            if parent.token == "":
                if next_parent.key_digests or next_parent.children:
                    self._store_record_index_node(kind, scope_digest, next_parent)
                else:
                    self._delete(_record_index_node_path(kind, scope_digest, ""))
                return
            if next_parent.key_digests or len(next_parent.children) >= 2:
                self._store_record_index_node(kind, scope_digest, next_parent)
                return
            removed_token = parent.token
            replacement = next_parent.children[0] if next_parent.children else None
            self._delete(_record_index_node_path(kind, scope_digest, removed_token))

    def _write(self, relative: str | Path, value: Mapping[str, object] | bytes) -> None:
        relative = _relative_path(self._root, relative)
        self.deletes.discard(relative)
        self.writes[relative] = (
            value if isinstance(value, bytes) else _json_bytes(value)
        )

    def _delete(self, relative: str | Path) -> None:
        relative = _relative_path(self._root, relative)
        self.writes.pop(relative, None)
        self.deletes.add(relative)


__all__: list[str] = []
