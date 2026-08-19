#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""In-memory implementation of the Runtime StateStore primitives."""

import asyncio
from collections.abc import Mapping, Sequence
from dataclasses import replace
from datetime import datetime, timezone
from typing import TypeVar

from linktools.core import environ

from ...errors import AIError, ErrorCode
from ._store import (
    FactQuery,
    OperationQuery,
    RecordQuery,
    RecordReplacement,
    StateCallback,
    StateGroupCallback,
    StateStorageGroup,
    StateTransaction,
    StoredAlias,
    StoredFact,
    StoredOperation,
    StoredRecord,
    active_state_transaction,
    active_state_group_transaction,
    bind_state_scope,
    reset_state_transaction,
    validate_record_identity,
    validate_record_replacement,
)

ValueT = TypeVar("ValueT")
_logger = environ.get_logger("ai.runtime.state.memory")


class _MemoryGroupTransaction:
    def __init__(
        self,
        group: "MemoryStateStorageGroup",
        transactions: Mapping["MemoryStateStore", StateTransaction],
    ) -> None:
        self._group = group
        self._transactions = transactions

    def transaction(self, store: "MemoryStateStore") -> StateTransaction:
        if store.storage_group is not self._group:
            raise RuntimeError("store does not belong to this StateStorageGroup")
        try:
            return self._transactions[store]
        except KeyError as error:
            raise RuntimeError("store was not enlisted in the StateStorageGroup transaction") from error


class MemoryStateStorageGroup:
    """Atomic group coordinator for independent in-memory logical stores."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()

    async def read(self, store: "MemoryStateStore", fn: StateCallback[ValueT]) -> ValueT:
        self._ensure_member(store)
        active = active_state_transaction(store)
        if active is not None:
            return await fn(active)
        async with self._lock:
            return await fn(self._transaction(store))

    async def mutate(
        self,
        stores: Sequence["MemoryStateStore"],
        fn: StateGroupCallback[ValueT],
    ) -> ValueT:
        members = tuple(dict.fromkeys(stores))
        if not members:
            raise ValueError("StateStorageGroup mutation requires a store")
        for store in members:
            self._ensure_member(store)
        active = active_state_transaction(members[0])
        if active is not None:
            group_transaction = active_state_group_transaction(self, members)
            return await fn(group_transaction)
        async with self._lock:
            transaction_now = datetime.now(timezone.utc)
            transactions = {
                store: self._transaction(store, now=transaction_now)
                for store in members
            }
            group_transaction = _MemoryGroupTransaction(self, transactions)
            token = bind_state_scope(self, transactions)
            try:
                result = await fn(group_transaction)
                for store, transaction in transactions.items():
                    store._apply_transaction(transaction)
                _logger.debug(
                    "state storage group committed: backend=memory stores=%s",
                    len(transactions),
                )
                return result
            finally:
                reset_state_transaction(token)

    def _transaction(
        self,
        store: "MemoryStateStore",
        *,
        now: datetime | None = None,
    ) -> "_MemoryTransaction":
        return _MemoryTransaction(
            dict(store._records),
            dict(store._aliases),
            dict(store._sequences),
            dict(store._facts),
            dict(store._operations),
            now=now,
        )

    def _ensure_member(self, store: "MemoryStateStore") -> None:
        if store.storage_group is not self:
            raise RuntimeError("store does not belong to this StateStorageGroup")
        store._ensure_ready()


class MemoryStateStore:
    """Atomic process-local StateStore used by tests and volatile routes."""

    def __init__(self, group: MemoryStateStorageGroup | None = None) -> None:
        self._records: dict[bytes, StoredRecord] = {}
        self._aliases: dict[bytes, bytes] = {}
        self._sequences: dict[bytes, int] = {}
        self._facts: dict[tuple[bytes, int], StoredFact] = {}
        self._operations: dict[bytes, StoredOperation] = {}
        self._lock = asyncio.Lock()
        self._closed = False
        self._initialized = False
        self._storage_group = group or MemoryStateStorageGroup()

    @property
    def storage_group(self) -> StateStorageGroup:
        return self._storage_group

    async def initialize(self) -> None:
        if self._closed:
            raise AIError(ErrorCode.STORAGE_CLOSED)
        self._initialized = True
        _logger.debug("memory StateStore initialized")

    async def close(self) -> None:
        self._closed = True
        self._initialized = False
        _logger.debug("memory StateStore closed")

    async def read(self, fn: StateCallback[ValueT]) -> ValueT:
        self._ensure_ready()
        active = active_state_transaction(self)
        if active is not None:
            return await fn(active)
        return await self._storage_group.read(self, fn)

    async def mutate(self, fn: StateCallback[ValueT]) -> ValueT:
        self._ensure_ready()
        active = active_state_transaction(self)
        if active is not None:
            return await fn(active)
        return await self._storage_group.mutate((self,), lambda group: fn(group.transaction(self)))

    async def validate_integrity(self) -> None:
        self._ensure_ready()

        async def check(transaction: StateTransaction) -> None:
            _validate_transaction_integrity(transaction)

        await self.read(check)

    def _ensure_ready(self) -> None:
        if self._closed:
            raise AIError(ErrorCode.STORAGE_CLOSED)
        if not self._initialized:
            raise AIError(ErrorCode.STORAGE_DEPENDENCY_NOT_READY)

    def _apply_transaction(self, transaction: "_MemoryTransaction") -> None:
        self._records = transaction.records
        self._aliases = transaction.aliases
        self._sequences = transaction.sequences
        self._facts = transaction.facts
        self._operations = transaction.operations


class _MemoryTransaction:
    def __init__(
        self,
        records: dict[bytes, StoredRecord],
        aliases: dict[bytes, bytes],
        sequences: dict[bytes, int],
        facts: dict[tuple[bytes, int], StoredFact],
        operations: dict[bytes, StoredOperation],
        *,
        now: datetime | None = None,
    ) -> None:
        self.records = records
        self.aliases = aliases
        self.sequences = sequences
        self.facts = facts
        self.operations = operations
        self.guarded_record_keys: set[bytes] = set()
        self._now = now

    async def now(self) -> datetime:
        if self._now is None:
            self._now = datetime.now(timezone.utc)
        return self._now

    async def validate_integrity(self) -> None:
        _validate_transaction_integrity(self)

    async def get_record(self, key: bytes) -> StoredRecord | None:
        return self.records.get(key)

    async def get_records(self, keys: Sequence[bytes]) -> Mapping[bytes, StoredRecord]:
        return {key: self.records[key] for key in keys if key in self.records}

    async def insert_record(self, record: StoredRecord) -> None:
        await self.insert_records((record,))

    async def insert_records(self, records: Sequence[StoredRecord]) -> None:
        values = tuple(records)
        keys = [record.key_digest for record in values]
        if len(keys) != len(set(keys)):
            raise ValueError("insert_records contains duplicate keys")
        for record in sorted(values, key=lambda value: value.key_digest):
            validate_record_identity(record)
            if record.key_digest in self.records:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            self.records[record.key_digest] = record
            self.guarded_record_keys.add(record.key_digest)

    async def guard_record(
        self,
        key: bytes,
        *,
        expected_storage_version: int,
    ) -> StoredRecord | None:
        current = self.records.get(key)
        if current is None or current.storage_version != expected_storage_version:
            if key in self.guarded_record_keys and current is not None:
                return current
            return None
        current = replace(current, storage_version=expected_storage_version + 1)
        self.records[key] = current
        self.guarded_record_keys.add(key)
        return current

    async def replace_record(self, record: StoredRecord, *, expected_storage_version: int) -> bool:
        try:
            await self.replace_records((RecordReplacement(record, expected_storage_version),))
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
        candidates: list[StoredRecord] = []
        for replacement in sorted(values, key=lambda value: value.record.key_digest):
            current = self.records.get(replacement.record.key_digest)
            if current is None or current.storage_version != replacement.expected_storage_version:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            validate_record_replacement(current, replacement.record)
            validate_record_identity(replacement.record)
            if replacement.record.storage_version != replacement.expected_storage_version + 1:
                raise ValueError("replacement must increment storage_version exactly once")
            candidates.append(replacement.record)
        for record in candidates:
            self.records[record.key_digest] = record
            self.guarded_record_keys.add(record.key_digest)

    async def update_record_lease(
        self,
        key: bytes,
        *,
        expected_storage_version: int,
        lease_owner: str | None,
        lease_fence: int,
        lease_expires_at: datetime | None,
    ) -> bool:
        current = self.records.get(key)
        if current is None or current.storage_version != expected_storage_version:
            return False
        if lease_fence < 0 or lease_expires_at is not None and lease_expires_at.tzinfo is None:
            raise ValueError("record lease is invalid")
        self.records[key] = replace(
            current,
            storage_version=expected_storage_version + 1,
            lease_owner=lease_owner,
            lease_fence=lease_fence,
            lease_expires_at=lease_expires_at,
        )
        self.guarded_record_keys.add(key)
        return True

    async def delete_record(self, key: bytes, *, expected_storage_version: int | None = None) -> bool:
        current = self.records.get(key)
        if current is None:
            return False
        expected = current.storage_version if expected_storage_version is None else expected_storage_version
        guarded = await self.guard_record(key, expected_storage_version=expected)
        if guarded is None:
            return False
        for alias, record_key in tuple(self.aliases.items()):
            if record_key == key:
                del self.aliases[alias]
        for fact_key, fact in tuple(self.facts.items()):
            if fact.owner_key_digest == key:
                del self.facts[fact_key]
        del self.records[key]
        self.guarded_record_keys.discard(key)
        return True

    async def list_records(self, query: RecordQuery) -> tuple[StoredRecord, ...]:
        values = [record for record in self.records.values() if _matches_record(record, query)]
        values.sort(key=lambda record: (record.sort_key, record.key_digest))
        if query.after_sort_key is not None and query.after_key_digest is not None:
            values = [
                record
                for record in values
                if (record.sort_key, record.key_digest) > (query.after_sort_key, query.after_key_digest)
            ]
        if query.limit is not None:
            values = values[: query.limit]
        return tuple(values)

    async def resolve_alias(self, alias: bytes) -> bytes | None:
        return (await self.resolve_aliases((alias,))).get(alias)

    async def resolve_aliases(self, aliases: Sequence[bytes]) -> Mapping[bytes, bytes]:
        values = {alias: self.aliases[alias] for alias in dict.fromkeys(aliases) if alias in self.aliases}
        for value in values.values():
            if value not in self.records:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return values

    async def insert_alias(self, alias: StoredAlias) -> None:
        await self.insert_aliases((alias,))

    async def insert_aliases(self, aliases: Sequence[StoredAlias]) -> None:
        for alias in sorted(aliases, key=lambda value: value.alias_digest):
            current = self.aliases.get(alias.alias_digest)
            if current is not None and current != alias.record_key_digest:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            if alias.record_key_digest not in self.guarded_record_keys:
                raise RuntimeError("alias owner must be guarded in the current transaction")
            self.aliases[alias.alias_digest] = alias.record_key_digest

    async def get_sequence(self, key: bytes) -> int:
        return self.sequences.get(key, 0)

    async def get_sequences(self, keys: Sequence[bytes]) -> Mapping[bytes, int]:
        return {key: self.sequences.get(key, 0) for key in keys}

    async def next_sequence(self, key: bytes) -> int:
        value = self.sequences.get(key, 0) + 1
        self.sequences[key] = value
        return value

    async def reserve_sequence(self, key: bytes, count: int) -> int:
        return (await self.reserve_sequences({key: count}))[key]

    async def reserve_sequences(self, requests: Mapping[bytes, int]) -> Mapping[bytes, int]:
        if any(count < 1 for count in requests.values()):
            raise ValueError("sequence reservation count must be positive")
        values = {
            key: self.sequences.get(key, 0) + requests[key]
            for key in sorted(requests)
        }
        self.sequences.update(values)
        return values

    async def advance_sequence(self, key: bytes, expected: int) -> int:
        current = self.sequences.get(key, 0)
        if current != expected:
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        return await self.next_sequence(key)

    async def delete_sequence(self, key: bytes) -> None:
        await self.delete_sequences((key,))

    async def delete_sequences(self, keys: Sequence[bytes]) -> None:
        for key in sorted(set(keys)):
            self.sequences.pop(key, None)

    async def insert_fact(self, fact: StoredFact) -> None:
        key = (fact.stream_digest, fact.sequence)
        if key in self.facts:
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        if fact.owner_key_digest not in self.guarded_record_keys:
            raise RuntimeError("fact owner must be guarded in the current transaction")
        self.facts[key] = fact

    async def insert_facts(self, facts: Sequence[StoredFact]) -> None:
        for fact in facts:
            await self.insert_fact(fact)

    async def list_facts(self, query: FactQuery) -> tuple[StoredFact, ...]:
        values = [fact for fact in self.facts.values() if fact.stream_digest == query.stream_digest]
        if query.after_sequence is not None:
            values = [fact for fact in values if fact.sequence > query.after_sequence]
        if query.subject_digest is not None:
            values = [fact for fact in values if fact.subject_digest == query.subject_digest]
        values.sort(key=lambda fact: fact.sequence)
        if query.latest:
            return () if not values else (values[-1],)
        if query.latest_per_subject:
            latest: dict[bytes | None, StoredFact] = {}
            for fact in values:
                latest[fact.subject_digest] = fact
            values = sorted(latest.values(), key=lambda fact: fact.sequence)
        if query.limit is not None:
            values = values[: query.limit]
        return tuple(values)

    async def delete_fact_streams(self, owner_key: bytes) -> None:
        for key, fact in tuple(self.facts.items()):
            if fact.owner_key_digest == owner_key:
                del self.facts[key]

    async def insert_operation(self, value: StoredOperation) -> None:
        if value.key_digest in self.operations:
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        if any(
            operation.stream_digest == value.stream_digest and operation.sequence == value.sequence
            for operation in self.operations.values()
        ):
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        self.operations[value.key_digest] = value

    async def get_operation(self, key: bytes) -> StoredOperation | None:
        return self.operations.get(key)

    async def replace_operation(self, value: StoredOperation, *, expected_state: str) -> bool:
        current = self.operations.get(value.key_digest)
        if current is None or current.state != expected_state:
            return False
        self.operations[value.key_digest] = value
        return True

    async def list_operations(self, query: OperationQuery) -> tuple[StoredOperation, ...]:
        values = [
            operation
            for operation in self.operations.values()
            if (query.stream_digest is None or operation.stream_digest == query.stream_digest)
            and (query.states is None or operation.state in query.states)
            and (query.through_sequence is None or operation.sequence <= query.through_sequence)
        ]
        values.sort(key=lambda operation: (operation.sequence, operation.key_digest))
        if query.limit is not None:
            values = values[: query.limit]
        return tuple(values)

    async def delete_operations(self, query: OperationQuery) -> tuple[StoredOperation, ...]:
        values = await self.list_operations(query)
        for value in values:
            del self.operations[value.key_digest]
        return values


def _matches_record(record: StoredRecord, query: RecordQuery) -> bool:
    if query.partition_digest is not None and record.partition_digest != query.partition_digest:
        return False
    if query.scope_digest is not None and record.scope_digest != query.scope_digest:
        return False
    if query.parent_digest is not None and record.parent_digest != query.parent_digest:
        return False
    if query.kind is not None and record.kind != query.kind:
        return False
    return query.states is None or record.state in query.states


def _validate_transaction_integrity(transaction: _MemoryTransaction) -> None:
    if any(record_key not in transaction.records for record_key in transaction.aliases.values()):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    for fact in transaction.facts.values():
        if fact.owner_key_digest not in transaction.records:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    fact_sequences: dict[bytes, list[int]] = {}
    for fact in transaction.facts.values():
        fact_sequences.setdefault(fact.stream_digest, []).append(fact.sequence)
    for sequences in fact_sequences.values():
        if sorted(sequences) != list(range(1, max(sequences) + 1)):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)


__all__ = ["MemoryStateStorageGroup", "MemoryStateStore"]
