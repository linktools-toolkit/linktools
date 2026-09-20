#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""In-memory Runtime StateStore and storage-group lifecycle."""

import asyncio
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from typing import TypeVar

from linktools.core import environ

from ...errors import AIError, ErrorCode
from ._memory_transaction import _MemoryTransaction, _validate_transaction_integrity
from ._store import (
    StateCallback,
    StateGroupCallback,
    StateStorageGroup,
    StateTransaction,
    StoredFact,
    StoredOperation,
    StoredRecord,
    active_state_group_transaction,
    active_state_transaction,
    bind_state_scope,
    reset_state_transaction,
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
            raise RuntimeError(
                "store was not enlisted in the StateStorageGroup transaction"
            ) from error


class MemoryStateStorageGroup:
    """Atomic group coordinator for independent in-memory logical stores."""

    def __init__(self, *, read_only: bool = False) -> None:
        self._lock = asyncio.Lock()
        self._read_only = read_only

    async def read(
        self,
        store: "MemoryStateStore",
        fn: StateCallback[ValueT],
    ) -> ValueT:
        self._ensure_member(store)
        active = active_state_transaction(store)
        if active is not None:
            return await fn(active)
        async with self._lock:
            transaction = self._transaction(store)
            token = bind_state_scope(
                self,
                {store: transaction},
                writable=False,
            )
            try:
                readonly = active_state_transaction(store)
                if readonly is None:
                    raise RuntimeError(
                        "read-only StateTransaction scope was not bound"
                    )
                return await fn(readonly)
            finally:
                reset_state_transaction(token)

    async def mutate(
        self,
        stores: Sequence["MemoryStateStore"],
        fn: StateGroupCallback[ValueT],
    ) -> ValueT:
        if self._read_only:
            raise AIError(ErrorCode.STORAGE_READ_ONLY)
        members = tuple(dict.fromkeys(stores))
        if not members:
            raise ValueError("StateStorageGroup mutation requires a store")
        for store in members:
            self._ensure_member(store)
        active = active_state_transaction(members[0], writable=True)
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
    ) -> _MemoryTransaction:
        return _MemoryTransaction(
            store._records,
            store._aliases,
            store._sequences,
            store._facts,
            store._operations,
            store._operation_streams,
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
        self._operation_streams: dict[bytes, dict[int, bytes]] = {}
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
        active = active_state_transaction(self, writable=True)
        if active is not None:
            return await fn(active)
        return await self._storage_group.mutate(
            (self,),
            lambda group: fn(group.transaction(self)),
        )

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

    def _apply_transaction(self, transaction: _MemoryTransaction) -> None:
        transaction.records.apply_to(self._records)
        transaction.aliases.apply_to(self._aliases)
        transaction.sequences.apply_to(self._sequences)
        transaction.facts.apply_to(self._facts)
        transaction.operations.apply_to(self._operations)
        for stream, changes in transaction.operation_stream_changes.items():
            values = self._operation_streams.setdefault(stream, {})
            for sequence, key in changes.items():
                if key is None:
                    values.pop(sequence, None)
                else:
                    values[sequence] = key
            if not values:
                self._operation_streams.pop(stream, None)


__all__ = ["MemoryStateStorageGroup", "MemoryStateStore"]
