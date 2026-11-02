#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Read-only decoding of the legacy filesystem State layout."""

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path


from ..errors import AIError, ErrorCode
from .codec import decode_fact, decode_operation, decode_record
from .state import (
    FactScanCursor,
    OperationScanCursor,
    RecordScanCursor,
    StoredFact,
    StoredOperation,
    StoredRecord,
)


@dataclass(slots=True)
class _FactStreamInfo:
    stream_digest: bytes
    owner_key_digest: bytes
    last_sequence: int


class _FilesystemCache:
    def __init__(self, root: Path) -> None:
        self._root = root
        self._fact_streams: dict[bytes, _FactStreamInfo | None] = {}
        self._operations: dict[bytes, StoredOperation | None] = {}
        self._record_kinds: tuple[str, ...] | None = None


    def get_fact_stream(self, stream: bytes) -> _FactStreamInfo | None:
        if stream in self._fact_streams:
            return self._fact_streams[stream]
        path = self._root / _fact_meta_path(self._root, stream)
        if not path.is_file():
            self._fact_streams[stream] = None
            return None
        _require_layout_path(path, self._root, _fact_meta_path(self._root, stream))
        stored_stream, owner, last_sequence = _read_fact_metadata(path)
        if stored_stream != stream:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        value = _FactStreamInfo(stream, owner, last_sequence)
        self._fact_streams[stream] = value
        return value


    def get_operation(self, key: bytes) -> StoredOperation | None:
        if key in self._operations:
            return self._operations[key]
        value = None
        key_hex = key.hex()
        path = self._root / "operations" / "by-key" / key_hex[:2] / f"{key_hex}.json"
        if path.is_file():
            value = decode_operation(_read_json(path))
            if value.key_digest != key:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        self._operations[key] = value
        return value


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
                    values.append(value)
                    if len(values) == limit:
                        return tuple(values)
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
                    values.append(value)
                    if len(values) == limit:
                        return tuple(values)
        return tuple(values)


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
        return self._record_kinds


def _fact_directory(stream: bytes) -> str:
    value = stream.hex()
    return f"facts/{value[:2]}/{value}"


def _fact_meta_path(root: Path, stream: bytes) -> str:
    return f"{_fact_directory(stream)}/meta.json"


def _fact_item_path(root: Path, stream: bytes, sequence: int) -> Path:
    return root / _fact_directory(stream) / "items" / f"{sequence:020d}.json"


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


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


def _require_layout_path(path: Path, root: Path, expected: str) -> None:
    try:
        actual = path.relative_to(root).as_posix()
    except ValueError as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
    if actual != expected:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)


__all__: list[str] = []
