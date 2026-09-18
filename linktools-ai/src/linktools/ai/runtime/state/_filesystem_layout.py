#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Filesystem state layout, cache, and index primitives."""

import asyncio
import hashlib
import json
import os
from collections.abc import Iterator, Mapping, MutableMapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

from linktools.core import environ

from ...errors import AIError, ErrorCode
from ...storage import sync_directory
from ._codec import decode_alias, decode_fact, decode_operation, decode_record
from ._store import (
    FactScanCursor,
    OperationScanCursor,
    RecordQuery,
    RecordScanCursor,
    StoredFact,
    StoredOperation,
    StoredRecord,
    validate_record_identity,
)

KeyT = TypeVar("KeyT")
MapValueT = TypeVar("MapValueT")
_logger = environ.get_logger("ai.runtime.state.filesystem")


@dataclass(slots=True)
class _FactStreamInfo:
    stream_digest: bytes
    owner_key_digest: bytes
    last_sequence: int
    subjects: dict[bytes, int]
    subjects_loaded: bool = True


@dataclass(frozen=True, slots=True)
class _RecordIndexNode:
    token: str
    key_digests: tuple[bytes, ...]
    children: tuple[str, ...]


class _FilesystemCache:
    def __init__(self, root: Path) -> None:
        self._root = root
        self._records: dict[bytes, StoredRecord | None] = {}
        self._aliases: dict[bytes, bytes | None] = {}
        self._sequences: dict[bytes, int] = {}
        self._fact_streams: dict[bytes, _FactStreamInfo | None] = {}
        self._operations: dict[bytes, StoredOperation | None] = {}
        self._record_kinds: tuple[str, ...] | None = None
        self._loaded_record_kinds: set[str] = set()
        self._aliases_complete = False
        self._fact_streams_complete = False
        self._operations_complete = False
        self._cache_hits = 0
        self._cache_misses = 0
        self._record_kind_scans = 0
        self._business_files_read = 0
        self._record_index_complete = _record_index_marker_valid(root)

    def get_record(self, key: bytes) -> StoredRecord | None:
        if key in self._records:
            self._cache_hits += 1
            return self._records[key]
        self._cache_misses += 1
        value_hex = key.hex()
        matches: list[StoredRecord] = []
        for kind in self._record_kind_names():
            path = self._root / "records" / kind / value_hex[:2] / f"{value_hex}.json"
            if not path.is_file():
                continue
            value = decode_record(_read_json(path))
            if value.key_digest != key or value.kind != kind:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            self._business_files_read += 1
            matches.append(value)
        if len(matches) > 1:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if matches:
            self._records[key] = matches[0]
            return matches[0]
        self._records[key] = None
        return None

    def list_records(self, query: RecordQuery) -> tuple[StoredRecord, ...]:
        indexed = self._list_indexed_records(query)
        if indexed is not None:
            return indexed
        kinds = (query.kind,) if query.kind is not None else self._record_kind_names()
        for kind in kinds:
            self._load_record_kind(kind)
        values = sorted(
            (
                value
                for value in self._records.values()
                if isinstance(value, StoredRecord)
                and value.kind in kinds
                and _matches_record(value, query)
            ),
            key=lambda record: (record.sort_key, record.key_digest),
        )
        if query.after_sort_key is not None and query.after_key_digest is not None:
            values = [
                value
                for value in values
                if (value.sort_key, value.key_digest)
                > (query.after_sort_key, query.after_key_digest)
            ]
        if query.limit is not None:
            values = values[: query.limit]
        return tuple(values)

    def _list_indexed_records(
        self,
        query: RecordQuery,
    ) -> tuple[StoredRecord, ...] | None:
        if not self._record_index_complete or not _record_query_indexable(query):
            return None
        if query.kind is None or query.limit is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        selected = _record_index_select(self._root, query)
        values: list[StoredRecord] = []
        for sort_key, key in selected:
            value = self._indexed_record(query.kind, key)
            if value.sort_key != sort_key or not _matches_record(value, query):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            values.append(value)
        return tuple(values)

    def _indexed_record(self, kind: str, key: bytes) -> StoredRecord:
        if key in self._records:
            self._cache_hits += 1
            value = self._records[key]
            if not isinstance(value, StoredRecord) or value.kind != kind:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            return value
        self._cache_misses += 1
        key_hex = key.hex()
        path = self._root / "records" / kind / key_hex[:2] / f"{key_hex}.json"
        if not path.is_file():
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        value = decode_record(_read_json(path))
        if value.key_digest != key or value.kind != kind:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        self._business_files_read += 1
        self._records[key] = value
        return value

    def get_alias(self, alias: bytes) -> bytes | None:
        if alias in self._aliases:
            self._cache_hits += 1
            return self._aliases[alias]
        self._cache_misses += 1
        value = None
        path = self._root / _alias_path(self._root, alias)
        if path.is_file():
            decoded = decode_alias(_read_json(path))
            if decoded.alias_digest != alias:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            value = decoded.record_key_digest
            self._business_files_read += 1
        self._aliases[alias] = value
        return value

    def list_aliases(self) -> tuple[tuple[bytes, bytes], ...]:
        if self._aliases_complete:
            return tuple(
                (alias, record_key)
                for alias, record_key in self._aliases.items()
                if record_key is not None
            )
        values: list[tuple[bytes, bytes]] = []
        for path in (self._root / "aliases").glob("*/*.json"):
            value = decode_alias(_read_json(path))
            self._aliases[value.alias_digest] = value.record_key_digest
            self._business_files_read += 1
            values.append((value.alias_digest, value.record_key_digest))
        self._aliases_complete = True
        return tuple(values)

    def get_sequence(self, key: bytes) -> int:
        if key in self._sequences:
            self._cache_hits += 1
            return self._sequences[key]
        self._cache_misses += 1
        value = 0
        path = self._root / _sequence_path(self._root, key)
        if path.is_file():
            stored_key, value = _read_sequence_metadata(path)
            if stored_key != key:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            self._business_files_read += 1
        self._sequences[key] = value
        return value

    def get_fact_stream(self, stream: bytes) -> _FactStreamInfo | None:
        if stream in self._fact_streams:
            self._cache_hits += 1
            return self._fact_streams[stream]
        self._cache_misses += 1
        path = self._root / _fact_meta_path(self._root, stream)
        if not path.is_file():
            self._fact_streams[stream] = None
            return None
        _require_layout_path(path, self._root, _fact_meta_path(self._root, stream))
        stored_stream, owner, last_sequence = _read_fact_metadata(path)
        if stored_stream != stream:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        value = _FactStreamInfo(stream, owner, last_sequence, {}, False)
        self._fact_streams[stream] = value
        self._business_files_read += 1
        return value

    def load_fact_subjects(self, info: _FactStreamInfo) -> None:
        if info.subjects_loaded:
            return
        subjects: dict[bytes, int] = {}
        root = self._root / _fact_directory(info.stream_digest) / "subjects"
        for ref in root.glob("*.ref"):
            subject = _layout_digest(ref.stem)
            _require_layout_path(
                ref,
                self._root,
                _fact_subject_path(self._root, info.stream_digest, subject),
            )
            sequence = _read_subject_sequence(ref)
            if not 1 <= sequence <= info.last_sequence:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            subjects[subject] = sequence
        for subject, sequence in subjects.items():
            if info.subjects.get(subject, 0) < sequence:
                info.subjects[subject] = sequence
        info.subjects_loaded = True
        self._business_files_read += len(subjects)

    def list_fact_streams(self) -> tuple[_FactStreamInfo, ...]:
        if self._fact_streams_complete:
            return tuple(
                value for value in self._fact_streams.values() if value is not None
            )
        values: list[_FactStreamInfo] = []
        for path in (self._root / "facts").glob("*/*/meta.json"):
            stream, _owner, _last_sequence = _read_fact_metadata(path)
            info = self.get_fact_stream(stream)
            if info is not None:
                values.append(info)
        self._fact_streams_complete = True
        return tuple(values)

    def get_operation(self, key: bytes) -> StoredOperation | None:
        if key in self._operations:
            self._cache_hits += 1
            return self._operations[key]
        self._cache_misses += 1
        value = None
        key_hex = key.hex()
        path = self._root / "operations" / "by-key" / key_hex[:2] / f"{key_hex}.json"
        if path.is_file():
            value = decode_operation(_read_json(path))
            if value.key_digest != key:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            self._business_files_read += 1
        self._operations[key] = value
        return value

    def list_operations(self) -> tuple[StoredOperation, ...]:
        if self._operations_complete:
            return tuple(
                value for value in self._operations.values() if value is not None
            )
        values: list[StoredOperation] = []
        for path in (self._root / "operations/by-key").glob("*/*.json"):
            value = decode_operation(_read_json(path))
            self._operations[value.key_digest] = value
            self._business_files_read += 1
            values.append(value)
        self._operations_complete = True
        return tuple(values)

    def get_operation_by_stream_sequence(
        self,
        stream_digest: bytes,
        sequence: int,
    ) -> StoredOperation | None:
        stream = stream_digest.hex()
        path = (
            self._root
            / "operations"
            / "streams"
            / stream[:2]
            / stream
            / f"{sequence:020d}.ref"
        )
        if not path.is_file():
            return None
        _require_layout_path(
            path,
            self._root,
            f"operations/streams/{stream[:2]}/{stream}/{sequence:020d}.ref",
        )
        key = _read_operation_ref(path)
        operation = self.get_operation(key)
        if (
            operation is None
            or operation.stream_digest != stream_digest
            or operation.sequence != sequence
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return operation

    def list_operation_stream(
        self, stream_digest: bytes
    ) -> tuple[StoredOperation, ...]:
        stream = stream_digest.hex()
        root = self._root / "operations" / "streams" / stream[:2] / stream
        values: list[StoredOperation] = []
        if not root.is_dir():
            return ()
        for path in sorted(root.glob("*.ref")):
            sequence = _layout_sequence_name(path.stem)
            value = self.get_operation_by_stream_sequence(stream_digest, sequence)
            if value is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            values.append(value)
        return tuple(values)

    def scan_records(self) -> tuple[StoredRecord, ...]:
        for kind in self._record_kind_names():
            self._load_record_kind(kind)
        return tuple(value for value in self._records.values() if value is not None)

    def scan_records_page(
        self,
        *,
        after: RecordScanCursor | None,
        limit: int,
    ) -> tuple[StoredRecord, ...]:
        _require_scan_limit(limit)
        values: list[StoredRecord] = []
        for kind in self._record_kind_names():
            if after is not None and kind < after.kind:
                continue
            root = self._root / "records" / kind
            if not root.is_dir():
                continue
            for shard in sorted(path for path in root.iterdir() if path.is_dir()):
                for path in sorted(shard.glob("*.json")):
                    key_hex = path.stem
                    key = _layout_digest(key_hex)
                    if after is not None and (kind, key) <= (
                        after.kind,
                        after.key_digest,
                    ):
                        continue
                    value = decode_record(_read_json(path))
                    _require_layout_path(
                        path,
                        self._root,
                        f"records/{kind}/{key_hex[:2]}/{key_hex}.json",
                    )
                    if value.kind != kind or value.key_digest != key:
                        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                    self._records[key] = value
                    self._business_files_read += 1
                    values.append(value)
                    if len(values) == limit:
                        return tuple(values)
        return tuple(values)

    def scan_facts(self) -> tuple[StoredFact, ...]:
        values: list[StoredFact] = []
        for info in self.list_fact_streams():
            loaded = _read_fact_batch(
                self._root,
                info.stream_digest,
                tuple(range(1, info.last_sequence + 1)),
            )
            values.extend(loaded.values())
        return tuple(values)

    def scan_facts_page(
        self,
        *,
        after: FactScanCursor | None,
        limit: int,
    ) -> tuple[StoredFact, ...]:
        _require_scan_limit(limit)
        facts_root = self._root / "facts"
        if not facts_root.is_dir():
            return ()
        values: list[StoredFact] = []
        for shard in sorted(path for path in facts_root.iterdir() if path.is_dir()):
            for stream_dir in sorted(path for path in shard.iterdir() if path.is_dir()):
                stream = _layout_digest(stream_dir.name)
                if after is not None and stream < after.stream_digest:
                    continue
                meta_path = stream_dir / "meta.json"
                if not meta_path.is_file():
                    continue
                _require_layout_path(
                    meta_path,
                    self._root,
                    _fact_meta_path(self._root, stream),
                )
                info = self.get_fact_stream(stream)
                if info is None:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                first = 1
                if after is not None and stream == after.stream_digest:
                    first = after.sequence + 1
                if first > info.last_sequence:
                    continue
                sequences = tuple(
                    range(
                        first,
                        min(
                            info.last_sequence + 1,
                            first + limit - len(values),
                        ),
                    )
                )
                batch = _read_fact_batch(self._root, info.stream_digest, sequences)
                for sequence in sequences:
                    try:
                        value = batch[(info.stream_digest, sequence)]
                    except KeyError as error:
                        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
                    self._business_files_read += 1
                    values.append(value)
                    if len(values) == limit:
                        return tuple(values)
        return tuple(values)

    def scan_operations(self) -> tuple[StoredOperation, ...]:
        return self.list_operations()

    def scan_operations_page(
        self,
        *,
        after: OperationScanCursor | None,
        limit: int,
    ) -> tuple[StoredOperation, ...]:
        _require_scan_limit(limit)
        values: list[StoredOperation] = []
        root = self._root / "operations" / "by-key"
        if not root.is_dir():
            return ()
        for shard in sorted(path for path in root.iterdir() if path.is_dir()):
            for path in sorted(shard.glob("*.json")):
                key_hex = path.stem
                key = _layout_digest(key_hex)
                if after is not None and key <= after.key_digest:
                    continue
                value = decode_operation(_read_json(path))
                _require_layout_path(
                    path,
                    self._root,
                    f"operations/by-key/{key_hex[:2]}/{key_hex}.json",
                )
                if value.key_digest != key:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                self._operations[key] = value
                self._business_files_read += 1
                values.append(value)
                if len(values) == limit:
                    return tuple(values)
        return tuple(values)

    def _record_kind_names(self) -> tuple[str, ...]:
        if self._record_kinds is None:
            root = self._root / "records"
            self._record_kinds = (
                tuple(sorted(path.name for path in root.iterdir() if path.is_dir()))
                if root.is_dir()
                else ()
            )
            self._record_kind_scans += 1
        return self._record_kinds

    def _load_record_kind(self, kind: str) -> None:
        if kind in self._loaded_record_kinds:
            self._cache_hits += 1
            return
        self._cache_misses += 1
        root = self._root / "records" / kind
        for path in root.glob("*/*.json"):
            value = decode_record(_read_json(path))
            if value.kind != kind:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            _require_layout_path(
                path,
                self._root,
                f"records/{kind}/{value.key_digest.hex()[:2]}/{value.key_digest.hex()}.json",
            )
            self._business_files_read += 1
            self._records[value.key_digest] = value
        self._loaded_record_kinds.add(kind)
        self._record_kind_scans += 1

    @property
    def record_index_complete(self) -> bool:
        return self._record_index_complete

    def disable_record_index(self) -> None:
        self._record_index_complete = False

    def set_record(
        self,
        key: bytes,
        value: StoredRecord | None,
        *,
        old_kind: str | None = None,
    ) -> None:
        self._records[key] = value
        if isinstance(value, StoredRecord) and self._record_kinds is not None:
            self._record_kinds = tuple(sorted(set(self._record_kinds) | {value.kind}))

    def set_alias(self, key: bytes, value: bytes | None) -> None:
        self._aliases[key] = value

    def set_sequence(self, key: bytes, value: int) -> None:
        self._sequences[key] = value

    def set_fact_stream(self, key: bytes, value: _FactStreamInfo | None) -> None:
        self._fact_streams[key] = value

    def set_operation(self, key: bytes, value: StoredOperation | None) -> None:
        self._operations[key] = value

    def _log_summary(self, event: str, value: int) -> None:
        _logger.debug(
            "filesystem cache summary: event=%s value=%s cache_hit=%s cache_miss=%s "
            "record_kind_scan=%s business_files_read=%s",
            event,
            value,
            self._cache_hits,
            self._cache_misses,
            self._record_kind_scans,
            self._business_files_read,
        )


@dataclass(slots=True)
class _FilesystemIndex:
    records: dict[bytes, StoredRecord]
    aliases: dict[bytes, bytes]
    sequences: dict[bytes, int]
    fact_streams: dict[bytes, _FactStreamInfo]
    operations: dict[bytes, StoredOperation]
    cache: _FilesystemCache


class _CowMap(MutableMapping[KeyT, MapValueT]):
    def __init__(self, base: Mapping[KeyT, MapValueT]) -> None:
        self._base = base
        self._changes: dict[KeyT, MapValueT] = {}
        self._deleted: set[KeyT] = set()

    def __getitem__(self, key: KeyT) -> MapValueT:
        if key in self._changes:
            return self._changes[key]
        if key in self._deleted:
            raise KeyError(key)
        return self._base[key]

    def __setitem__(self, key: KeyT, value: MapValueT) -> None:
        self._deleted.discard(key)
        self._changes[key] = value

    def __delitem__(self, key: KeyT) -> None:
        if key not in self and key not in self._changes:
            raise KeyError(key)
        self._changes.pop(key, None)
        self._deleted.add(key)

    def __iter__(self) -> Iterator[KeyT]:
        keys = set(self._base) | set(self._changes)
        return iter(keys - self._deleted)

    def __len__(self) -> int:
        return sum(1 for _ in self)

    def changes(self) -> Mapping[KeyT, MapValueT]:
        return self._changes

    def deleted(self) -> frozenset[KeyT]:
        return frozenset(self._deleted)

    def apply_to(self, target: MutableMapping[KeyT, MapValueT]) -> None:
        for key in self._deleted:
            target.pop(key, None)
        target.update(self._changes)

def _matches_record(record: StoredRecord, query: RecordQuery) -> bool:
    return (
        (
            query.partition_digest is None
            or record.partition_digest == query.partition_digest
        )
        and (query.scope_digest is None or record.scope_digest == query.scope_digest)
        and (query.parent_digest is None or record.parent_digest == query.parent_digest)
        and (query.kind is None or record.kind == query.kind)
        and (
            query.sort_key_prefix is None
            or record.sort_key.startswith(query.sort_key_prefix)
        )
        and (query.states is None or record.state in query.states)
    )


def _relative_path(root: Path, value: str | Path) -> str:
    if isinstance(value, Path):
        return value.relative_to(root).as_posix()
    return value


_RECORD_INDEX_MARKER = "record-index/complete"
_RECORD_INDEX_VERSION = "1"
_RECORD_INDEX_NODE_VERSION = 1


def _record_index_marker_valid(root: Path) -> bool:
    marker = root / _RECORD_INDEX_MARKER
    if not marker.exists():
        return False
    try:
        value = marker.read_text(encoding="utf-8")
    except OSError as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
    if value != _RECORD_INDEX_VERSION:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return True


def _record_query_indexable(query: RecordQuery) -> bool:
    return (
        query.kind is not None
        and query.scope_digest is not None
        and query.partition_digest is None
        and query.parent_digest is None
        and query.states is None
        and query.limit is not None
    )


def _record_index_identity(
    record: StoredRecord | None,
) -> tuple[str, bytes, str, bytes] | None:
    if record is None or record.scope_digest is None:
        return None
    validate_record_identity(record)
    return record.kind, record.scope_digest, record.sort_key, record.key_digest


def _record_index_node_path(
    kind: str,
    scope_digest: bytes,
    token: str,
) -> str:
    digest = hashlib.sha256(token.encode("ascii")).hexdigest()
    return f"record-index/{kind}/{scope_digest.hex()}/nodes/{digest[:2]}/{digest}.json"


def _record_index_node_payload(node: _RecordIndexNode) -> Mapping[str, object]:
    return {
        "version": _RECORD_INDEX_NODE_VERSION,
        "token": node.token,
        "keys": [key.hex() for key in node.key_digests],
        "children": list(node.children),
    }


def _decode_record_index_node(
    raw: Mapping[str, object],
    *,
    expected_token: str,
) -> _RecordIndexNode:
    _require_layout_keys(raw, frozenset({"version", "token", "keys", "children"}))
    token = raw["token"]
    keys = raw["keys"]
    children = raw["children"]
    if (
        raw["version"] != _RECORD_INDEX_NODE_VERSION
        or token != expected_token
        or not isinstance(token, str)
        or not token.isascii()
        or not isinstance(keys, list)
        or not isinstance(children, list)
        or any(not isinstance(child, str) for child in children)
        or children != sorted(set(children))
    ):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    key_digests = tuple(_layout_digest(key) for key in keys)
    if key_digests != tuple(sorted(set(key_digests))):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    next_characters: set[str] = set()
    for child in children:
        if (
            len(child) <= len(token)
            or not child.isascii()
            or not child.startswith(token)
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        next_character = child[len(token)]
        if next_character in next_characters:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        next_characters.add(next_character)
    if token and not key_digests and len(children) < 2:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    if not token and not key_digests and not children:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return _RecordIndexNode(token, key_digests, tuple(children))


def _decode_record_index_node_bytes(
    value: bytes,
    *,
    expected_token: str,
) -> _RecordIndexNode:
    try:
        raw = json.loads(value.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
    if not isinstance(raw, Mapping):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return _decode_record_index_node(raw, expected_token=expected_token)


def _read_record_index_node_path(
    path: Path,
    *,
    expected_token: str,
) -> _RecordIndexNode:
    try:
        raw = _read_json(path)
    except (OSError, TypeError, ValueError) as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
    return _decode_record_index_node(raw, expected_token=expected_token)


def _read_record_index_node(
    root: Path,
    kind: str,
    scope_digest: bytes,
    token: str,
) -> _RecordIndexNode | None:
    path = root / _record_index_node_path(kind, scope_digest, token)
    if not path.exists():
        return None
    return _read_record_index_node_path(path, expected_token=token)


def _record_index_child(node: _RecordIndexNode, token: str) -> str | None:
    if len(token) <= len(node.token):
        return None
    target = token[len(node.token)]
    matches = [child for child in node.children if child[len(node.token)] == target]
    if len(matches) > 1:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return None if not matches else matches[0]


def _record_index_replace_child(
    node: _RecordIndexNode,
    previous: str,
    current: str,
) -> _RecordIndexNode:
    children = [child for child in node.children if child != previous]
    if len(children) == len(node.children):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    children.append(current)
    return _RecordIndexNode(
        node.token,
        node.key_digests,
        tuple(sorted(children)),
    )


def _record_index_common_prefix(values: Sequence[str]) -> str:
    if not values:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    first = min(values)
    last = max(values)
    limit = min(len(first), len(last))
    index = 0
    while index < limit and first[index] == last[index]:
        index += 1
    return first[:index]


def _record_index_select(
    root: Path,
    query: RecordQuery,
) -> tuple[tuple[str, bytes], ...]:
    if query.kind is None or query.scope_digest is None or query.limit is None:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    prefix = "" if query.sort_key_prefix is None else query.sort_key_prefix
    node = _read_record_index_node(root, query.kind, query.scope_digest, "")
    if node is None:
        return ()
    while not node.token.startswith(prefix):
        if not prefix.startswith(node.token):
            return ()
        child_token = _record_index_child(node, prefix)
        if child_token is None or not (
            prefix.startswith(child_token) or child_token.startswith(prefix)
        ):
            return ()
        child = _read_record_index_node(
            root, query.kind, query.scope_digest, child_token
        )
        if child is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        node = child

    selected: list[tuple[str, bytes]] = []
    after_sort_key = query.after_sort_key
    after_key_digest = query.after_key_digest

    def visit(current: _RecordIndexNode) -> None:
        if len(selected) >= query.limit:
            return
        for key_digest in current.key_digests:
            if after_sort_key is None or (
                current.token,
                key_digest,
            ) > (after_sort_key, after_key_digest):
                selected.append((current.token, key_digest))
                if len(selected) >= query.limit:
                    return
        for child_token in current.children:
            if (
                after_sort_key is not None
                and child_token < after_sort_key
                and not after_sort_key.startswith(child_token)
            ):
                continue
            child = _read_record_index_node(
                root, query.kind, query.scope_digest, child_token
            )
            if child is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            visit(child)
            if len(selected) >= query.limit:
                return

    visit(node)
    return tuple(selected)


def _build_record_index_nodes(
    records: Sequence[StoredRecord],
) -> dict[str, _RecordIndexNode]:
    groups: dict[tuple[str, bytes], dict[str, set[bytes]]] = {}
    for record in records:
        identity = _record_index_identity(record)
        if identity is None:
            continue
        kind, scope_digest, token, key_digest = identity
        by_token = groups.setdefault((kind, scope_digest), {})
        by_token.setdefault(token, set()).add(key_digest)

    result: dict[str, _RecordIndexNode] = {}
    for (kind, scope_digest), by_token in groups.items():
        entries = tuple(
            (token, tuple(sorted(keys))) for token, keys in sorted(by_token.items())
        )

        def build(
            prefix: str,
            values: Sequence[tuple[str, tuple[bytes, ...]]],
        ) -> str:
            exact = next((keys for token, keys in values if token == prefix), ())
            descendants: dict[str, list[tuple[str, tuple[bytes, ...]]]] = {}
            for token, keys in values:
                if token == prefix:
                    continue
                if not token.startswith(prefix) or len(token) <= len(prefix):
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                descendants.setdefault(token[len(prefix)], []).append((token, keys))
            children: list[str] = []
            for child_values in descendants.values():
                child_prefix = _record_index_common_prefix(
                    tuple(token for token, _keys in child_values)
                )
                build(child_prefix, tuple(child_values))
                children.append(child_prefix)
            node = _RecordIndexNode(prefix, tuple(exact), tuple(sorted(children)))
            relative = _record_index_node_path(kind, scope_digest, prefix)
            if relative in result and result[relative] != node:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            result[relative] = node
            return prefix

        build("", entries)
    return result


def _sync_record_index_tree(root: Path) -> None:
    if not root.exists():
        return
    for directory, _children, _files in os.walk(root, topdown=False):
        sync_directory(Path(directory))


def _record_path(record: StoredRecord) -> str:
    key = record.key_digest.hex()
    return f"records/{record.kind}/{key[:2]}/{key}.json"


def _alias_path(root: Path, alias: bytes) -> str:
    value = alias.hex()
    return f"aliases/{value[:2]}/{value}.json"


def _sequence_path(root: Path, key: bytes) -> str:
    value = key.hex()
    return f"sequences/{value[:2]}/{value}.json"


def _fact_directory(stream: bytes) -> str:
    value = stream.hex()
    return f"facts/{value[:2]}/{value}"


def _fact_meta_path(root: Path, stream: bytes) -> str:
    return f"{_fact_directory(stream)}/meta.json"


def _fact_item_path(root: Path, stream: bytes, sequence: int) -> Path:
    return root / _fact_directory(stream) / "items" / f"{sequence:020d}.json"


def _fact_subject_path(root: Path, stream: bytes, subject: bytes) -> str:
    return f"{_fact_directory(stream)}/subjects/{subject.hex()}.ref"


def _operation_path(root: Path, value: StoredOperation) -> str:
    key = value.key_digest.hex()
    return f"operations/by-key/{key[:2]}/{key}.json"


def _operation_ref_path(root: Path, value: StoredOperation) -> str:
    stream = value.stream_digest.hex()
    return f"operations/streams/{stream[:2]}/{stream}/{value.sequence:020d}.ref"


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _json_bytes(value: Mapping[str, object]) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _read_json(path: Path) -> Mapping[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError("JSON root must be an object")  # noqa: TRY004
    return value


def _require_layout_keys(value: Mapping[str, object], expected: frozenset[str]) -> None:
    if set(value.keys()) != expected:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)


def _layout_digest(value: object) -> bytes:
    if not isinstance(value, str):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    try:
        result = bytes.fromhex(value)
    except ValueError as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
    if len(result) != 32 or result.hex() != value:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return result


def _layout_nonnegative_int(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return value


def _layout_positive_int(value: object) -> int:
    result = _layout_nonnegative_int(value)
    if result < 1:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return result


def _layout_sequence_name(value: str) -> int:
    if len(value) != 20 or not value.isascii() or not value.isdigit():
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    sequence = int(value)
    if sequence < 1 or f"{sequence:020d}" != value:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return sequence


def _read_sequence_metadata(path: Path) -> tuple[bytes, int]:
    try:
        raw = _read_json(path)
    except (OSError, TypeError, ValueError) as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
    _require_layout_keys(raw, frozenset({"key", "value"}))
    return _layout_digest(raw["key"]), _layout_nonnegative_int(raw["value"])


def _read_fact_metadata(path: Path) -> tuple[bytes, bytes, int]:
    try:
        raw = _read_json(path)
    except (OSError, TypeError, ValueError) as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
    _require_layout_keys(raw, frozenset({"stream", "owner_key", "last_sequence"}))
    return (
        _layout_digest(raw["stream"]),
        _layout_digest(raw["owner_key"]),
        _layout_positive_int(raw["last_sequence"]),
    )


def _read_subject_sequence(path: Path) -> int:
    try:
        raw = _read_json(path)
    except (OSError, TypeError, ValueError) as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
    _require_layout_keys(raw, frozenset({"sequence"}))
    return _layout_positive_int(raw["sequence"])


def _read_operation_ref(path: Path) -> bytes:
    try:
        raw = _read_json(path)
    except (OSError, TypeError, ValueError) as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
    _require_layout_keys(raw, frozenset({"key"}))
    return _layout_digest(raw["key"])


def _read_generation_value(path: Path) -> int:
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
    if not raw.isascii() or not raw.isdigit():
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    value = int(raw)
    if value < 0 or str(value) != raw:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return value


def _read_fact_batch(
    root: Path,
    stream: bytes,
    sequences: Sequence[int],
) -> dict[tuple[bytes, int], StoredFact]:
    values: dict[tuple[bytes, int], StoredFact] = {}
    for sequence in sequences:
        try:
            fact = decode_fact(_read_json(_fact_item_path(root, stream, sequence)))
        except FileNotFoundError as error:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
        if fact.stream_digest != stream or fact.sequence != sequence:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        values[(stream, sequence)] = fact
    return values


def _require_scan_limit(limit: int) -> None:
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise ValueError("scan page limit must be positive")


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_json_bytes(value))
    _sync_file(path)


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")
    _sync_file(path)


def _sync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _require_layout_path(path: Path, root: Path, expected: str) -> None:
    try:
        actual = path.relative_to(root).as_posix()
    except ValueError as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
    if actual != expected:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)


def _track_physical_task(
    tasks: set[asyncio.Task[None]],
    task: asyncio.Task[None],
    label: str,
) -> None:
    tasks.add(task)

    def consume(done: asyncio.Task[None]) -> None:
        try:
            done.result()
        except asyncio.CancelledError:
            pass
        except BaseException:  # noqa: BLE001
            _logger.exception("detached %s failed", label)
        finally:
            tasks.discard(done)

    task.add_done_callback(consume)


__all__: list[str] = []
