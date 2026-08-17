#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Granular, journaled filesystem implementation of StateStore."""

import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import TypeVar

from linktools.core import environ

from ...errors import AIError, ErrorCode
from ...storage import FilesystemJournal, FilesystemMutationLock, sync_directory
from ._codec import (
    decode_alias,
    decode_fact,
    decode_operation,
    decode_record,
    encode_alias,
    encode_fact,
    encode_operation,
    encode_record,
)
from ._memory import _MemoryTransaction
from ._store import (
    StateCallback,
    StoredAlias,
    StoredFact,
    StoredOperation,
    StoredRecord,
    active_state_transaction,
    bind_state_transaction,
    reset_state_transaction,
)

ValueT = TypeVar("ValueT")
_logger = environ.get_logger("ai.runtime.state.filesystem")


class FilesystemStateStore:
    """A domain-local granular StateStore with crash-safe commit markers."""

    def __init__(self, root: str | Path, *, namespace: str, tenant_id: str, runtime_domain: str) -> None:
        self._root = Path(root).expanduser().resolve()
        self._namespace = namespace
        self._tenant_id = tenant_id
        self._runtime_domain = runtime_domain
        self._lock = FilesystemMutationLock(self._root / "state.lock")
        self._journal = FilesystemJournal(
            self._root,
            error_code=ErrorCode.STORAGE_INTEGRITY_ERROR,
        )
        self._closed = False
        self._initialized = False
        self._active_depth = 0

    @property
    def root(self) -> Path:
        return self._root

    async def initialize(self) -> None:
        if self._closed:
            raise AIError(ErrorCode.STORAGE_CLOSED)
        self._root.mkdir(parents=True, exist_ok=True)
        for name in (
            "records",
            "aliases",
            "facts",
            "sequences",
            "operations/by-key",
            "operations/streams",
            ".txn/stage",
        ):
            (self._root / name).mkdir(parents=True, exist_ok=True)
        manifest = self._root / "manifest.json"
        expected = {
            "format": "linktools-ai-state",
            "layout_version": 1,
            "namespace_digest": _digest(self._namespace),
            "tenant_digest": _digest(self._tenant_id),
            "runtime_domain": self._runtime_domain,
        }
        if manifest.exists():
            try:
                actual = json.loads(manifest.read_text(encoding="utf-8"))
            except (OSError, ValueError) as error:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
            if actual != expected:
                raise AIError(ErrorCode.STORAGE_VERSION_UNSUPPORTED)
        else:
            _write_json(manifest, expected)
        generation = self._root / "generation"
        if not generation.exists():
            _write_text(generation, "0")
        sync_directory(self._root)
        await self._recover()
        self._initialized = True
        _logger.info("filesystem StateStore initialized: domain=%s root=%s", self._runtime_domain, self._root)

    async def close(self) -> None:
        self._closed = True
        self._initialized = False
        _logger.debug("filesystem StateStore closed: domain=%s", self._runtime_domain)

    async def read(self, fn: StateCallback[ValueT]) -> ValueT:
        self._ensure_ready()
        active = active_state_transaction(self)
        if active is not None:
            return await fn(active)
        for _ in range(3):
            before = self._generation()
            if (self._root / ".txn" / "commit").exists():
                await self._recover()
                continue
            maps = await self._load_maps()
            result = await fn(
                _MemoryTransaction(
                    dict(maps[0]),
                    dict(maps[1]),
                    dict(maps[2]),
                    dict(maps[3]),
                    dict(maps[4]),
                )
            )
            after = self._generation()
            if before == after and not (self._root / ".txn" / "commit").exists():
                return result
        async with self._lock:
            await self._recover()
            maps = await self._load_maps()
            return await fn(_MemoryTransaction(*maps))

    async def mutate(self, fn: StateCallback[ValueT]) -> ValueT:
        self._ensure_ready()
        active = active_state_transaction(self)
        if active is not None:
            self._active_depth += 1
            try:
                return await fn(active)
            finally:
                self._active_depth -= 1
        await self._lock.__aenter__()
        token: object | None = None
        try:
            await self._recover()
            maps = await self._load_maps()
            transaction = _MemoryTransaction(dict(maps[0]), dict(maps[1]), dict(maps[2]), dict(maps[3]), dict(maps[4]))
            token = bind_state_transaction(self, transaction)
            self._active_depth = 1
            result = await fn(transaction)
            await self._commit(maps, transaction)
            return result
        finally:
            if token is not None:
                reset_state_transaction(token)
            self._active_depth = 0
            await self._lock.__aexit__(None, None, None)

    async def validate_integrity(self) -> None:
        self._ensure_ready()
        maps = await self._load_maps()
        transaction = _MemoryTransaction(*maps)
        await transaction.validate_integrity()

    def _ensure_ready(self) -> None:
        if self._closed:
            raise AIError(ErrorCode.STORAGE_CLOSED)
        if not self._initialized:
            raise AIError(ErrorCode.STORAGE_DEPENDENCY_NOT_READY)

    def _generation(self) -> int:
        try:
            return int((self._root / "generation").read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error

    async def _load_maps(
        self,
    ) -> tuple[
        dict[bytes, StoredRecord],
        dict[bytes, bytes],
        dict[bytes, int],
        dict[tuple[bytes, int], StoredFact],
        dict[bytes, StoredOperation],
    ]:
        try:
            return await self._load_maps_unchecked()
        except AIError:
            raise
        except (OSError, TypeError, ValueError, KeyError) as error:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error

    async def _load_maps_unchecked(
        self,
    ) -> tuple[
        dict[bytes, StoredRecord],
        dict[bytes, bytes],
        dict[bytes, int],
        dict[tuple[bytes, int], StoredFact],
        dict[bytes, StoredOperation],
    ]:
        records: dict[bytes, StoredRecord] = {}
        aliases: dict[bytes, bytes] = {}
        sequences: dict[bytes, int] = {}
        facts: dict[tuple[bytes, int], StoredFact] = {}
        operations: dict[bytes, StoredOperation] = {}
        for path in (self._root / "records").glob("*/*/*.json"):
            value = decode_record(_read_json(path))
            _require_layout_path(
                path,
                self._root,
                f"records/{value.kind}/{value.key_digest.hex()[:2]}/{value.key_digest.hex()}.json",
            )
            if value.key_digest in records:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            records[value.key_digest] = value
        for path in (self._root / "aliases").glob("*/*.json"):
            value = decode_alias(_read_json(path))
            _require_layout_path(
                path,
                self._root,
                f"aliases/{value.alias_digest.hex()[:2]}/{value.alias_digest.hex()}.json",
            )
            if value.alias_digest in aliases:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            aliases[value.alias_digest] = value.record_key_digest
        for path in (self._root / "sequences").glob("*/*.json"):
            raw = _read_json(path)
            key = bytes.fromhex(str(raw["key"]))
            if len(key) != 32 or key in sequences:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            value = int(raw["value"])
            if value < 0:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            _require_layout_path(
                path,
                self._root,
                f"sequences/{key.hex()[:2]}/{key.hex()}.json",
            )
            sequences[key] = value
        for path in (self._root / "facts").glob("*/*/*.json"):
            value = decode_fact(_read_json(path))
            key = (value.stream_digest, value.sequence)
            _require_layout_path(
                path,
                self._root,
                f"facts/{value.stream_digest.hex()[:2]}/{value.stream_digest.hex()}/{value.sequence:020d}.json",
            )
            if key in facts:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            facts[key] = value
        for path in (self._root / "operations/by-key").glob("*/*.json"):
            value = decode_operation(_read_json(path))
            _require_layout_path(
                path,
                self._root,
                f"operations/by-key/{value.key_digest.hex()[:2]}/{value.key_digest.hex()}.json",
            )
            if value.key_digest in operations:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            operations[value.key_digest] = value
        references: dict[tuple[bytes, int], bytes] = {}
        for path in (self._root / "operations/streams").glob("*/*/*.ref"):
            stream = bytes.fromhex(path.parent.name)
            sequence = int(path.stem)
            raw = _read_json(path)
            key = bytes.fromhex(str(raw["key"]))
            _require_layout_path(
                path,
                self._root,
                f"operations/streams/{stream.hex()[:2]}/{stream.hex()}/{sequence:020d}.ref",
            )
            if len(stream) != 32 or len(key) != 32 or (stream, sequence) in references:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            references[(stream, sequence)] = key
        expected_references = {(value.stream_digest, value.sequence): value.key_digest for value in operations.values()}
        if references != expected_references:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        transaction = _MemoryTransaction(records, aliases, sequences, facts, operations)
        expected_files = set(_serialized_state(transaction))
        actual_files = {
            path.relative_to(self._root).as_posix()
            for directory in (
                self._root / "records",
                self._root / "aliases",
                self._root / "facts",
                self._root / "sequences",
                self._root / "operations",
            )
            for path in directory.rglob("*")
            if path.is_file()
        }
        if actual_files != expected_files:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return records, aliases, sequences, facts, operations

    async def _commit(
        self,
        before: tuple[
            Mapping[bytes, StoredRecord],
            Mapping[bytes, bytes],
            Mapping[bytes, int],
            Mapping[tuple[bytes, int], StoredFact],
            Mapping[bytes, StoredOperation],
        ],
        after: _MemoryTransaction,
    ) -> None:
        base = self._generation()
        desired = _serialized_state(after)
        previous = _serialized_state_values(before)
        if desired == previous:
            _logger.debug("filesystem StateStore mutation was a no-op: generation=%s", base)
            return
        target = base + 1
        plan = self._journal.stage(
            desired,
            previous,
            base_generation=base,
            target_generation=target,
        )
        self._journal.publish(plan)
        _write_text(self._root / "generation", str(target))
        sync_directory(self._root)
        self._journal.complete()
        _logger.debug("filesystem StateStore mutation committed: generation=%s", target)

    async def _recover(self) -> None:
        self._journal.recover(
            self._generation,
            lambda target: _write_text(self._root / "generation", str(target)),
        )


def _serialized_state(transaction: _MemoryTransaction) -> dict[str, bytes]:
    values: dict[str, bytes] = {}
    for value in transaction.records.values():
        values[f"records/{value.kind}/{value.key_digest.hex()[:2]}/{value.key_digest.hex()}.json"] = _json_bytes(
            encode_record(value)
        )
    for alias, record in transaction.aliases.items():
        values[f"aliases/{alias.hex()[:2]}/{alias.hex()}.json"] = _json_bytes(encode_alias(StoredAlias(alias, record)))
    for key, value in transaction.sequences.items():
        values[f"sequences/{key.hex()[:2]}/{key.hex()}.json"] = _json_bytes({"key": key.hex(), "value": value})
    for value in transaction.facts.values():
        stream = value.stream_digest.hex()
        values[f"facts/{stream[:2]}/{stream}/{value.sequence:020d}.json"] = _json_bytes(encode_fact(value))
    for value in transaction.operations.values():
        key = value.key_digest.hex()
        stream = value.stream_digest.hex()
        values[f"operations/by-key/{key[:2]}/{key}.json"] = _json_bytes(encode_operation(value))
        values[f"operations/streams/{stream[:2]}/{stream}/{value.sequence:020d}.ref"] = _json_bytes({"key": key})
    return values


def _serialized_state_values(
    values: tuple[
        Mapping[bytes, StoredRecord],
        Mapping[bytes, bytes],
        Mapping[bytes, int],
        Mapping[tuple[bytes, int], StoredFact],
        Mapping[bytes, StoredOperation],
    ],
) -> dict[str, bytes]:
    transaction = _MemoryTransaction(
        dict(values[0]), dict(values[1]), dict(values[2]), dict(values[3]), dict(values[4])
    )
    return _serialized_state(transaction)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _json_bytes(value: Mapping[str, object]) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _read_json(path: Path) -> Mapping[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


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


__all__ = ["FilesystemStateStore"]
