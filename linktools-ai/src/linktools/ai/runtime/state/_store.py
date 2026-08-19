#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Backend-neutral persistence primitives for Runtime state.

The module deliberately contains no domain state machines.  It defines the
small physical contract shared by memory, filesystem, and SQL stores.
"""

import asyncio
import hashlib
from collections.abc import Awaitable, Callable, Mapping, Sequence
from contextvars import ContextVar, Token
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Generic, Protocol, TypeVar

from ...core import JsonValue, canonical_json_bytes

ValueT = TypeVar("ValueT")


class StateLockOrderError(RuntimeError):
    """Raised when a state callback requests an invalid synchronization edge."""


class StateTransactionNestingError(RuntimeError):
    """Raised when a callback requests an invalid nested state operation."""


@dataclass(frozen=True, slots=True)
class _ActiveStateScope:
    group: "StateStorageGroup"
    task: "asyncio.Task[object]"
    transactions: Mapping["StateStore", "StateTransaction"]
    writable: bool


_active_state_scope: ContextVar[_ActiveStateScope | None] = ContextVar(
    "linktools_ai_active_state_transaction",
    default=None,
)


def active_state_transaction(
    store: "StateStore",
    *,
    writable: bool = False,
) -> "StateTransaction | None":
    """Return the active transaction, rejecting invalid scope reuse."""
    return _active_transaction(store, writable=writable)


def _active_transaction(
    store: "StateStore",
    *,
    writable: bool,
) -> "StateTransaction | None":
    active = _active_state_scope.get()
    if active is None:
        return None
    task = asyncio.current_task()
    if active.task is not task:
        raise StateTransactionNestingError(
            "a child task cannot reuse its parent StateStore transaction"
        )
    if writable and not active.writable:
        raise StateTransactionNestingError(
            "a read-only StateStore callback cannot mutate state"
        )
    try:
        return active.transactions[store]
    except KeyError as error:
        if store.storage_group is active.group:
            raise StateTransactionNestingError(
                "a StateStore was not enlisted in the active group transaction"
            ) from error
        raise StateTransactionNestingError(
            "a task cannot access two StateStorageGroups at once"
        ) from error


def active_state_scope() -> _ActiveStateScope | None:
    """Return the current scope for synchronization-order checks."""
    return _active_state_scope.get()


def bind_state_transaction(store: "StateStore", transaction: "StateTransaction") -> Token:
    """Bind a compatibility single-store scope around one transaction."""
    return bind_state_scope(
        store.storage_group,
        {store: transaction},
    )


def bind_state_scope(
    group: "StateStorageGroup",
    transactions: Mapping["StateStore", "StateTransaction"],
    *,
    writable: bool = True,
) -> Token:
    task = asyncio.current_task()
    if task is None:
        raise RuntimeError("StateStore mutation requires an asyncio task")
    bound = (
        transactions
        if writable
        else {
            store: _ReadOnlyStateTransaction(transaction)
            for store, transaction in transactions.items()
        }
    )
    return _active_state_scope.set(_ActiveStateScope(group, task, bound, writable))


class _BoundStateGroupTransaction:
    def __init__(
        self,
        group: "StateStorageGroup",
        transactions: Mapping["StateStore", "StateTransaction"],
    ) -> None:
        self._group = group
        self._transactions = transactions

    def transaction(self, store: "StateStore") -> "StateTransaction":
        if store.storage_group is not self._group:
            raise RuntimeError("store does not belong to this StateStorageGroup")
        try:
            return self._transactions[store]
        except KeyError as error:
            raise RuntimeError("store was not enlisted in the StateStorageGroup transaction") from error


def active_state_group_transaction(
    group: "StateStorageGroup",
    stores: Sequence["StateStore"],
) -> "StateGroupTransaction":
    active = _active_state_scope.get()
    if active is None:
        raise RuntimeError("no active StateStorageGroup transaction")
    task = asyncio.current_task()
    if active.task is not task:
        raise StateTransactionNestingError(
            "a child task cannot reuse its parent StateStorageGroup transaction"
        )
    if not active.writable:
        raise StateTransactionNestingError(
            "a read-only StateStore callback cannot mutate state"
        )
    if active.group is not group:
        raise StateTransactionNestingError(
            "a task cannot mutate two StateStorageGroups at once"
        )
    for store in stores:
        if store not in active.transactions:
            raise StateTransactionNestingError(
                "store was not enlisted in the StateStorageGroup transaction"
            )
    return _BoundStateGroupTransaction(group, active.transactions)


def reset_state_transaction(token: Token) -> None:
    _active_state_scope.reset(token)


def _digest(value: JsonValue) -> bytes:
    return hashlib.sha256(canonical_json_bytes(value)).digest()


def _require_digest(value: bytes, name: str) -> bytes:
    if not isinstance(value, bytes) or len(value) != 32:
        raise ValueError(f"{name} must be a 32-byte digest")
    return value


@dataclass(frozen=True, slots=True)
class StoredRecord:
    """Current mutable state of one logical resource."""

    key_digest: bytes
    partition_digest: bytes
    scope_digest: bytes | None
    parent_digest: bytes | None
    kind: str
    sort_key: str
    state: str | None
    storage_version: int
    lease_owner: str | None
    lease_fence: int
    lease_expires_at: datetime | None
    data: Mapping[str, JsonValue]

    def __post_init__(self) -> None:
        _require_digest(self.key_digest, "key_digest")
        _require_digest(self.partition_digest, "partition_digest")
        if self.scope_digest is not None:
            _require_digest(self.scope_digest, "scope_digest")
        if self.parent_digest is not None:
            _require_digest(self.parent_digest, "parent_digest")
        if not isinstance(self.kind, str) or not 0 < len(self.kind) <= 32:
            raise ValueError("record kind must contain at most 32 characters")
        if self.kind in {".", ".."} or any(character in self.kind for character in "/\\"):
            raise ValueError("record kind contains a path separator")
        if (
            not isinstance(self.sort_key, str)
            or not 0 < len(self.sort_key) <= 128
            or self.sort_key.isascii() is False
        ):
            raise ValueError("record sort_key must contain 1..128 ASCII characters")
        if not isinstance(self.storage_version, int) or self.storage_version < 0:
            raise ValueError("storage_version must be non-negative")
        if not isinstance(self.lease_fence, int) or self.lease_fence < 0:
            raise ValueError("lease_fence must be non-negative")
        if self.lease_expires_at is not None and self.lease_expires_at.tzinfo is None:
            raise ValueError("lease_expires_at must be timezone-aware")
        if not isinstance(self.data, Mapping):
            raise TypeError("record data must be a mapping")


@dataclass(frozen=True, slots=True)
class StoredAlias:
    alias_digest: bytes
    record_key_digest: bytes

    def __post_init__(self) -> None:
        _require_digest(self.alias_digest, "alias_digest")
        _require_digest(self.record_key_digest, "record_key_digest")


@dataclass(frozen=True, slots=True)
class StoredFact:
    stream_digest: bytes
    sequence: int
    owner_key_digest: bytes
    kind: str
    subject_digest: bytes | None
    state: str | None
    data: Mapping[str, JsonValue]

    def __post_init__(self) -> None:
        _require_digest(self.stream_digest, "stream_digest")
        _require_digest(self.owner_key_digest, "owner_key_digest")
        if self.subject_digest is not None:
            _require_digest(self.subject_digest, "subject_digest")
        if not isinstance(self.sequence, int) or self.sequence < 0:
            raise ValueError("fact sequence must be non-negative")
        if not isinstance(self.kind, str) or not 0 < len(self.kind) <= 32:
            raise ValueError("fact kind must contain at most 32 characters")


@dataclass(frozen=True, slots=True)
class StoredOperation:
    key_digest: bytes
    stream_digest: bytes
    sequence: int
    state: str
    compactable: bool
    data: Mapping[str, JsonValue]

    def __post_init__(self) -> None:
        _require_digest(self.key_digest, "key_digest")
        _require_digest(self.stream_digest, "stream_digest")
        if not isinstance(self.sequence, int) or self.sequence < 0:
            raise ValueError("operation sequence must be non-negative")
        if not isinstance(self.state, str) or not self.state:
            raise ValueError("operation state is required")


@dataclass(frozen=True, slots=True)
class Observed(Generic[ValueT]):
    """A value paired with its physical CAS token."""

    value: ValueT
    storage_version: int

    def __post_init__(self) -> None:
        if not isinstance(self.storage_version, int) or self.storage_version < 0:
            raise ValueError("storage_version must be non-negative")


@dataclass(frozen=True, slots=True)
class RecordReplacement:
    record: StoredRecord
    expected_storage_version: int

    def __post_init__(self) -> None:
        if self.expected_storage_version < 0:
            raise ValueError("expected_storage_version must be non-negative")


@dataclass(frozen=True, slots=True)
class RecordQuery:
    partition_digest: bytes | None = None
    scope_digest: bytes | None = None
    parent_digest: bytes | None = None
    kind: str | None = None
    states: frozenset[str] | None = None
    after_sort_key: str | None = None
    after_key_digest: bytes | None = None
    limit: int | None = None

    def __post_init__(self) -> None:
        if not any(value is not None for value in (self.partition_digest, self.scope_digest, self.parent_digest)):
            raise ValueError("RecordQuery requires a query dimension")
        for name, value in (
            ("partition_digest", self.partition_digest),
            ("scope_digest", self.scope_digest),
            ("parent_digest", self.parent_digest),
        ):
            if value is not None:
                _require_digest(value, name)
        if self.kind is not None:
            if (
                not isinstance(self.kind, str)
                or not 0 < len(self.kind) <= 32
                or self.kind in {".", ".."}
                or any(character in self.kind for character in "/\\")
            ):
                raise ValueError("record kind contains a path separator")
        if self.after_sort_key is not None and self.after_key_digest is None:
            raise ValueError("after_key_digest is required with after_sort_key")
        if self.after_key_digest is not None:
            _require_digest(self.after_key_digest, "after_key_digest")
        _validate_limit(self.limit)


@dataclass(frozen=True, slots=True)
class FactQuery:
    stream_digest: bytes
    after_sequence: int | None = None
    limit: int | None = None
    subject_digest: bytes | None = None
    latest: bool = False
    latest_per_subject: bool = False

    def __post_init__(self) -> None:
        _require_digest(self.stream_digest, "stream_digest")
        if self.after_sequence is not None and self.after_sequence < 0:
            raise ValueError("after_sequence must be non-negative")
        if self.subject_digest is not None:
            _require_digest(self.subject_digest, "subject_digest")
        if self.latest and self.latest_per_subject:
            raise ValueError("latest and latest_per_subject cannot both be true")
        if self.latest_per_subject and self.subject_digest is not None:
            raise ValueError("latest_per_subject cannot be combined with subject_digest")
        _validate_limit(self.limit)


@dataclass(frozen=True, slots=True)
class OperationQuery:
    stream_digest: bytes | None = None
    states: frozenset[str] | None = None
    through_sequence: int | None = None
    compactable: bool | None = None
    limit: int | None = None

    def __post_init__(self) -> None:
        if self.stream_digest is not None:
            _require_digest(self.stream_digest, "stream_digest")
        if self.through_sequence is not None and self.through_sequence < 0:
            raise ValueError("through_sequence must be non-negative")
        if self.compactable is not None and not isinstance(self.compactable, bool):
            raise TypeError("compactable must be a bool or None")
        _validate_limit(self.limit)


class StateTransaction(Protocol):
    async def now(self) -> datetime: ...

    async def get_record(self, key: bytes) -> StoredRecord | None: ...
    async def get_records(self, keys: Sequence[bytes]) -> Mapping[bytes, StoredRecord]: ...
    async def insert_record(self, record: StoredRecord) -> None: ...
    async def insert_records(self, records: Sequence[StoredRecord]) -> None: ...
    async def guard_record(
        self,
        key: bytes,
        *,
        expected_storage_version: int,
    ) -> StoredRecord | None: ...
    async def replace_record(self, record: StoredRecord, *, expected_storage_version: int) -> bool: ...
    async def replace_records(self, replacements: Sequence["RecordReplacement"]) -> None: ...
    async def update_record_lease(
        self,
        key: bytes,
        *,
        expected_storage_version: int,
        lease_owner: str | None,
        lease_fence: int,
        lease_expires_at: datetime | None,
    ) -> bool: ...
    async def delete_record(self, key: bytes, *, expected_storage_version: int | None = None) -> bool: ...
    async def list_records(self, query: RecordQuery) -> tuple[StoredRecord, ...]: ...
    async def scan_records(self) -> tuple[StoredRecord, ...]: ...
    async def resolve_alias(self, alias: bytes) -> bytes | None: ...
    async def resolve_aliases(self, aliases: Sequence[bytes]) -> Mapping[bytes, bytes]: ...
    async def insert_alias(self, alias: StoredAlias) -> None: ...
    async def insert_aliases(self, aliases: Sequence[StoredAlias]) -> None: ...
    async def get_sequence(self, key: bytes) -> int: ...
    async def get_sequences(self, keys: Sequence[bytes]) -> Mapping[bytes, int]: ...
    async def next_sequence(self, key: bytes) -> int: ...
    async def reserve_sequence(self, key: bytes, count: int) -> int: ...
    async def reserve_sequences(self, reservations: Mapping[bytes, int]) -> Mapping[bytes, int]: ...
    async def advance_sequence(self, key: bytes, expected: int) -> int: ...
    async def delete_sequence(self, key: bytes) -> None: ...
    async def delete_sequences(self, keys: Sequence[bytes]) -> None: ...
    async def insert_fact(self, fact: StoredFact) -> None: ...
    async def insert_facts(self, facts: Sequence[StoredFact]) -> None: ...
    async def list_facts(self, query: FactQuery) -> tuple[StoredFact, ...]: ...
    async def scan_facts(self) -> tuple[StoredFact, ...]: ...
    async def delete_fact_streams(self, owner_key: bytes) -> None: ...
    async def insert_operation(self, value: StoredOperation) -> None: ...
    async def get_operation(self, key: bytes) -> StoredOperation | None: ...
    async def replace_operation(self, value: StoredOperation, *, expected_state: str) -> bool: ...
    async def list_operations(self, query: OperationQuery) -> tuple[StoredOperation, ...]: ...
    async def scan_operations(self) -> tuple[StoredOperation, ...]: ...
    async def delete_operations(self, query: OperationQuery) -> tuple[StoredOperation, ...]: ...


class _ReadOnlyStateTransaction(StateTransaction):
    def __init__(self, transaction: StateTransaction) -> None:
        self._transaction = transaction

    async def now(self) -> datetime:
        return await self._transaction.now()

    async def get_record(self, key: bytes) -> StoredRecord | None:
        return await self._transaction.get_record(key)

    async def get_records(self, keys: Sequence[bytes]) -> Mapping[bytes, StoredRecord]:
        return await self._transaction.get_records(keys)

    async def guard_record(
        self,
        key: bytes,
        *,
        expected_storage_version: int,
    ) -> StoredRecord | None:
        del key, expected_storage_version
        self._reject("guard_record")
        return None

    async def list_records(self, query: RecordQuery) -> tuple[StoredRecord, ...]:
        return await self._transaction.list_records(query)

    async def scan_records(self) -> tuple[StoredRecord, ...]:
        return await self._transaction.scan_records()

    async def resolve_alias(self, alias: bytes) -> bytes | None:
        return await self._transaction.resolve_alias(alias)

    async def resolve_aliases(self, aliases: Sequence[bytes]) -> Mapping[bytes, bytes]:
        return await self._transaction.resolve_aliases(aliases)

    async def get_sequence(self, key: bytes) -> int:
        return await self._transaction.get_sequence(key)

    async def get_sequences(self, keys: Sequence[bytes]) -> Mapping[bytes, int]:
        return await self._transaction.get_sequences(keys)

    async def list_facts(self, query: FactQuery) -> tuple[StoredFact, ...]:
        return await self._transaction.list_facts(query)

    async def scan_facts(self) -> tuple[StoredFact, ...]:
        return await self._transaction.scan_facts()

    async def get_operation(self, key: bytes) -> StoredOperation | None:
        return await self._transaction.get_operation(key)

    async def list_operations(self, query: OperationQuery) -> tuple[StoredOperation, ...]:
        return await self._transaction.list_operations(query)

    async def scan_operations(self) -> tuple[StoredOperation, ...]:
        return await self._transaction.scan_operations()

    def _reject(self, operation: str) -> None:
        raise StateTransactionNestingError(
            f"read-only StateTransaction cannot perform {operation}"
        )

    async def insert_record(self, record: StoredRecord) -> None:
        del record
        self._reject("insert_record")

    async def insert_records(self, records: Sequence[StoredRecord]) -> None:
        del records
        self._reject("insert_records")

    async def replace_record(self, record: StoredRecord, *, expected_storage_version: int) -> bool:
        del record, expected_storage_version
        self._reject("replace_record")
        return False

    async def replace_records(self, replacements: Sequence[RecordReplacement]) -> None:
        del replacements
        self._reject("replace_records")

    async def update_record_lease(
        self,
        key: bytes,
        *,
        expected_storage_version: int,
        lease_owner: str | None,
        lease_fence: int,
        lease_expires_at: datetime | None,
    ) -> bool:
        del key, expected_storage_version, lease_owner, lease_fence, lease_expires_at
        self._reject("update_record_lease")
        return False

    async def delete_record(self, key: bytes, *, expected_storage_version: int | None = None) -> bool:
        del key, expected_storage_version
        self._reject("delete_record")
        return False

    async def insert_alias(self, alias: StoredAlias) -> None:
        del alias
        self._reject("insert_alias")

    async def insert_aliases(self, aliases: Sequence[StoredAlias]) -> None:
        del aliases
        self._reject("insert_aliases")

    async def next_sequence(self, key: bytes) -> int:
        del key
        self._reject("next_sequence")
        return 0

    async def reserve_sequence(self, key: bytes, count: int) -> int:
        del key, count
        self._reject("reserve_sequence")
        return 0

    async def reserve_sequences(self, reservations: Mapping[bytes, int]) -> Mapping[bytes, int]:
        del reservations
        self._reject("reserve_sequences")
        return {}

    async def advance_sequence(self, key: bytes, expected: int) -> int:
        del key, expected
        self._reject("advance_sequence")
        return 0

    async def delete_sequence(self, key: bytes) -> None:
        del key
        self._reject("delete_sequence")

    async def delete_sequences(self, keys: Sequence[bytes]) -> None:
        del keys
        self._reject("delete_sequences")

    async def insert_fact(self, fact: StoredFact) -> None:
        del fact
        self._reject("insert_fact")

    async def insert_facts(self, facts: Sequence[StoredFact]) -> None:
        del facts
        self._reject("insert_facts")

    async def delete_fact_streams(self, owner_key: bytes) -> None:
        del owner_key
        self._reject("delete_fact_streams")

    async def insert_operation(self, value: StoredOperation) -> None:
        del value
        self._reject("insert_operation")

    async def replace_operation(self, value: StoredOperation, *, expected_state: str) -> bool:
        del value, expected_state
        self._reject("replace_operation")
        return False

    async def delete_operations(self, query: OperationQuery) -> tuple[StoredOperation, ...]:
        del query
        self._reject("delete_operations")
        return ()


StateCallback = Callable[[StateTransaction], Awaitable[ValueT]]
StateGroupCallback = Callable[["StateGroupTransaction"], Awaitable[ValueT]]


class StateGroupTransaction(Protocol):
    def transaction(self, store: "StateStore") -> StateTransaction: ...


class StateStorageGroup(Protocol):
    async def read(self, store: "StateStore", fn: StateCallback[ValueT]) -> ValueT: ...

    async def mutate(
        self,
        stores: Sequence["StateStore"],
        fn: StateGroupCallback[ValueT],
    ) -> ValueT: ...


class StateStore(Protocol):
    """Physical StateStore contract with retry-safe, transaction-local callbacks.

    Mutation callbacks must not perform external side effects outside the
    supplied StateTransaction.
    """

    async def initialize(self) -> None: ...
    async def close(self) -> None: ...
    @property
    def storage_group(self) -> StateStorageGroup: ...
    async def read(self, fn: StateCallback[ValueT]) -> ValueT: ...
    async def mutate(self, fn: StateCallback[ValueT]) -> ValueT: ...

    async def validate_integrity(self) -> None: ...


def record_key_digest(
    namespace: str,
    tenant_id: str,
    runtime_domain: str,
    kind: str,
    identity: JsonValue,
) -> bytes:
    return _digest(["record", namespace, tenant_id, runtime_domain, kind, identity])


def partition_digest(namespace: str, tenant_id: str, runtime_domain: str, kind: str) -> bytes:
    return _digest(["partition", namespace, tenant_id, runtime_domain, kind])


def scope_digest(
    namespace: str,
    tenant_id: str,
    runtime_domain: str,
    kind: str,
    relation: str,
    identity: JsonValue,
) -> bytes:
    return _digest(["scope", namespace, tenant_id, runtime_domain, kind, relation, identity])


def parent_digest(
    namespace: str,
    tenant_id: str,
    runtime_domain: str,
    kind: str,
    relation: str,
    identity: JsonValue,
) -> bytes:
    return _digest(["parent", namespace, tenant_id, runtime_domain, kind, relation, identity])


def alias_digest(
    namespace: str,
    tenant_id: str,
    runtime_domain: str,
    relation: str,
    identity: JsonValue,
) -> bytes:
    return _digest(["alias", namespace, tenant_id, runtime_domain, relation, identity])


def stream_digest(
    namespace: str,
    tenant_id: str,
    runtime_domain: str,
    relation: str,
    identity: JsonValue,
) -> bytes:
    return _digest(["stream", namespace, tenant_id, runtime_domain, relation, identity])


def sequence_key(
    namespace: str,
    tenant_id: str,
    runtime_domain: str,
    relation: str,
    identity: JsonValue,
) -> bytes:
    return _digest(["sequence", namespace, tenant_id, runtime_domain, relation, identity])


def operation_key(
    namespace: str,
    tenant_id: str,
    runtime_domain: str,
    identity: JsonValue,
) -> bytes:
    return _digest(["operation", namespace, tenant_id, runtime_domain, identity])


def subject_digest(identity: JsonValue) -> bytes:
    return _digest(["subject", identity])


def sortable_id(value: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("sortable id must be non-empty")
    return "i:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def sortable_timestamp(value: datetime, suffix: str) -> str:
    if value.tzinfo is None:
        raise ValueError("sortable timestamp must be timezone-aware")
    micros = int(value.astimezone(timezone.utc).timestamp() * 1_000_000)
    if micros < 0 or micros >= 1 << 63:
        raise ValueError("sortable timestamp is outside the supported range")
    if not suffix:
        raise ValueError("sortable timestamp suffix is required")
    return "t:" + f"{micros:016x}" + ":" + hashlib.sha256(suffix.encode("utf-8")).hexdigest()


def sortable_identity(identity: JsonValue) -> str:
    return "i:" + hashlib.sha256(canonical_json_bytes(identity)).hexdigest()


def encode_sort_key(value: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("sort key is required")
    if len(value) > 128 or not value.isascii():
        raise ValueError("sort key must contain 1..128 ASCII characters")
    return value


def decode_sort_key(value: str) -> str:
    return encode_sort_key(value)


def validate_record_identity(record: StoredRecord) -> None:
    """Validate physical invariants after decoding a backend row or file."""
    values = (
        ("key_digest", record.key_digest),
        ("partition_digest", record.partition_digest),
        ("scope_digest", record.scope_digest),
        ("parent_digest", record.parent_digest),
    )
    for name, value in values:
        if value is not None and (not isinstance(value, bytes) or len(value) != 32):
            raise ValueError(f"record {name} is invalid")
    encode_sort_key(record.sort_key)
    try:
        canonical_json_bytes(record.data)
    except (TypeError, ValueError) as error:
        raise ValueError("record data is not canonical JSON") from error


def validate_record_replacement(current: StoredRecord, candidate: StoredRecord) -> None:
    """Keep immutable physical identity columns stable across a CAS."""
    if (
        current.key_digest != candidate.key_digest
        or current.partition_digest != candidate.partition_digest
        or current.scope_digest != candidate.scope_digest
        or current.parent_digest != candidate.parent_digest
        or current.kind != candidate.kind
        or current.sort_key != candidate.sort_key
    ):
        raise ValueError("record physical identity cannot change")


def validate_operation_replacement(
    current: StoredOperation,
    candidate: StoredOperation,
) -> None:
    """Keep immutable operation identity stable across a state replacement."""
    if (
        current.key_digest != candidate.key_digest
        or current.stream_digest != candidate.stream_digest
        or current.sequence != candidate.sequence
        or current.compactable != candidate.compactable
    ):
        raise ValueError("operation physical identity cannot change")


def _validate_limit(limit: int | None) -> None:
    if limit is not None and (not isinstance(limit, int) or not 0 < limit <= 1000):
        raise ValueError("limit must be between 1 and 1000")


__all__ = [
    "FactQuery",
    "Observed",
    "OperationQuery",
    "RecordQuery",
    "RecordReplacement",
    "StateGroupTransaction",
    "StateLockOrderError",
    "StateStore",
    "StateStorageGroup",
    "StateTransactionNestingError",
    "StateTransaction",
    "StoredAlias",
    "StoredFact",
    "StoredOperation",
    "StoredRecord",
    "alias_digest",
    "decode_sort_key",
    "encode_sort_key",
    "operation_key",
    "parent_digest",
    "partition_digest",
    "record_key_digest",
    "scope_digest",
    "sequence_key",
    "sortable_id",
    "sortable_identity",
    "sortable_timestamp",
    "stream_digest",
    "subject_digest",
    "validate_operation_replacement",
    "validate_record_identity",
    "validate_record_replacement",
]
