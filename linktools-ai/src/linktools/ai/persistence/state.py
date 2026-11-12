#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Backend-neutral persistence primitives for Runtime state.

The module deliberately contains no domain state machines.  It defines the
small physical contract shared by memory, filesystem, and SQL stores.
"""

import asyncio
import hashlib
import math
from collections.abc import Awaitable, Callable, Mapping, Sequence
from contextvars import ContextVar, Token
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Generic, Literal, Protocol, TypeVar, runtime_checkable

from ..core import validate_capability_id, validate_capability_revision, validate_capability_write_key, ImmutableJsonMapping, JsonValue, canonical_json_bytes, validate_idempotency_key, validate_persistence_namespace, validate_resource_id, validate_tenant_id
from ..errors import AIError
from ._archive_read import ArchiveFrontierReader, ArchiveNodeRead
from ._record_key import encode_key_components as _encode_key_components, RECORD_KEY_MAX_BYTES, validate_record_key

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession
    from sqlalchemy.sql import Select
    from ..storage import SqlStorageContext, SqlValue

ValueT = TypeVar("ValueT")

OPERATION_KEY_MAX_BYTES = 768
STREAM_KEY_MAX_BYTES = 768
STORE_KEY_MAX_BYTES = 128
_STORE_SOURCE_MAX_BYTES = 768
RELATION_KEY_MAX_BYTES = 768
PARENT_KEY_MAX_BYTES = RECORD_KEY_MAX_BYTES


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
    record_reader: "GroupRecordReader | None" = None


_active_state_scope: ContextVar[_ActiveStateScope | None] = ContextVar(
    "linktools_ai_active_state_transaction",
    default=None,
)

_run_history_lock_stack: ContextVar[tuple[str, ...]] = ContextVar(
    "linktools_ai_run_history_lock_stack",
    default=(),
)


def enter_run_history_lock(agent_run_id: str) -> Token[tuple[str, ...]]:
    """Mark a local run-history lock as held for runtime I/O guards."""
    if not agent_run_id:
        raise ValueError("agent_run_id is required")
    stack = _run_history_lock_stack.get()
    if stack and stack[-1] != agent_run_id:
        raise StateLockOrderError("one task cannot hold multiple run history locks")
    return _run_history_lock_stack.set(stack + (agent_run_id,))


def exit_run_history_lock(token: Token[tuple[str, ...]]) -> None:
    """Restore the run-history lock marker after leaving a local lock."""
    _run_history_lock_stack.reset(token)


def require_no_run_history_lock(operation: str) -> None:
    """Fail before durable I/O is attempted while a run lock is held."""
    if _run_history_lock_stack.get():
        raise StateLockOrderError(
            f"{operation} cannot run while a RunHistoryLock is held"
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


def is_active_state_transaction(transaction: "StateTransaction") -> bool:
    """Recognize the underlying transaction without exposing read-only wrappers."""
    active = _active_state_scope.get()
    return active is not None and active.task is asyncio.current_task() and any(
        value is transaction
        or isinstance(value, _ReadOnlyStateTransaction) and value._transaction is transaction
        for value in active.transactions.values()
    )


def bind_state_scope(
    group: "StateStorageGroup",
    transactions: Mapping["StateStore", "StateTransaction"],
    *,
    writable: bool = True,
    record_reader: "GroupRecordReader | None" = None,
) -> Token:
    task = asyncio.current_task()
    if task is None:
        raise RuntimeError("StateStore mutation requires an asyncio task")
    bound = (
        transactions
        if writable
        else {
            store: (_ReadOnlyFactReader(transaction, transaction)
                    if isinstance(transaction, FactReader)
                    else _ReadOnlyStateTransaction(transaction))
            for store, transaction in transactions.items()
        }
    )
    return _active_state_scope.set(_ActiveStateScope(group, task, bound, writable, record_reader))


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


class _BoundGroupRecordReader(_BoundStateGroupTransaction):
    def __init__(
        self,
        group: "StateStorageGroup",
        transactions: Mapping["StateStore", "StateTransaction"],
        reader: "GroupRecordReader",
    ) -> None:
        super().__init__(group, transactions)
        self._reader = reader

    async def get_records(
        self, requests: Mapping["StateStore", Sequence[bytes]],
    ) -> Mapping["StateStore", Mapping[bytes, "StoredRecord"]]:
        active_state_group_transaction(self._group, tuple(requests), writable=False)
        active = active_state_scope()
        if active is None or active.record_reader is not self._reader:
            raise StateTransactionNestingError("group record reader is outside its transaction")
        return await self._reader.get_records(requests)


def active_state_group_transaction(
    group: "StateStorageGroup",
    stores: Sequence["StateStore"],
    *,
    writable: bool = True,
) -> "StateGroupTransaction":
    active = _active_state_scope.get()
    if active is None:
        raise RuntimeError("no active StateStorageGroup transaction")
    task = asyncio.current_task()
    if active.task is not task:
        raise StateTransactionNestingError(
            "a child task cannot reuse its parent StateStorageGroup transaction"
        )
    if writable and not active.writable:
        raise StateTransactionNestingError(
            "a read-only StateStore callback cannot mutate state"
        )
    if active.group is not group:
        raise StateTransactionNestingError(
            "a task cannot access two StateStorageGroups at once"
        )
    for store in stores:
        if store not in active.transactions:
            raise StateTransactionNestingError(
                "store was not enlisted in the StateStorageGroup transaction"
            )
    if active.record_reader is not None:
        return _BoundGroupRecordReader(group, active.transactions, active.record_reader)
    return _BoundStateGroupTransaction(group, active.transactions)


def reset_state_transaction(token: Token) -> None:
    _active_state_scope.reset(token)


def _require_operation_key(value: bytes) -> bytes:
    if not isinstance(value, bytes) or not 1 <= len(value) <= OPERATION_KEY_MAX_BYTES:
        raise ValueError(f"operation key must contain 1..{OPERATION_KEY_MAX_BYTES} bytes")
    return value


def _require_subject_id(value: str) -> str:
    try:
        return validate_resource_id(value)
    except AIError as error:
        raise ValueError("subject_id must be a valid resource identifier") from error


def _require_nonnegative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _require_optional_string(value: object, name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string or None")
    return value


def _require_state(value: object, name: str, *, optional: bool) -> str | None:
    if value is None:
        if optional:
            return None
        raise TypeError(f"{name} must be a string")
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not optional and not value:
        raise ValueError(f"{name} is required")
    if len(value) > 64:
        raise ValueError(f"{name} must contain at most 64 characters")
    return value


def _require_optional_aware_datetime(value: object, name: str) -> datetime | None:
    if value is None:
        return None
    if not isinstance(value, datetime):
        raise TypeError(f"{name} must be a datetime or None")
    if value.tzinfo is None:
        raise ValueError(f"{name} must be timezone-aware")
    return value


def _require_json_value(value: object, name: str) -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{name} contains a non-finite float")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _require_json_value(item, f"{name}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{name} contains a non-string object key")
            _require_json_value(item, f"{name}.{key}")
        return
    raise TypeError(f"{name} contains a non-JSON value")


def _require_data_mapping(value: object, name: str) -> Mapping[str, JsonValue]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    for key, item in value.items():
        if not isinstance(key, str):
            raise TypeError(f"{name} contains a non-string object key")
        _require_json_value(item, f"{name}.{key}")
    return value


@dataclass(frozen=True, slots=True)
class StoredRecord:
    """Current state with a physical CAS revision independent of domain data."""

    key: bytes = field(repr=False)
    scope_key: bytes | None = field(repr=False)
    parent_key: bytes | None = field(repr=False)
    kind: str
    sort_key: bytes = field(repr=False)
    state: str | None
    storage_revision: int
    lease_owner: str | None
    lease_fence: int
    lease_expires_at: datetime | None
    data: Mapping[str, JsonValue] = field(repr=False)
    subject_id: str | None = None
    payload_bytes: bytes | None = None

    def __post_init__(self) -> None:
        from ._record_owner import record_owner_table, record_owner_values

        record_owner_values(record_owner_table(self.key), {
            "kind": self.kind, "subject_id": self.subject_id, "parent_key": self.parent_key, "state": self.state,
            "lease_owner": self.lease_owner, "lease_fence": self.lease_fence,
            "lease_expires_at": self.lease_expires_at, "payload_bytes": self.payload_bytes,
        })
        if self.payload_bytes is not None and type(self.payload_bytes) is not bytes:
            raise TypeError("record binary payload must be frozen bytes")
        validate_record_key(self.key)
        if self.scope_key is not None:
            validate_relation_key(self.scope_key)
        if self.parent_key is not None:
            validate_parent_key(self.parent_key)
        if self.subject_id is not None:
            _require_subject_id(self.subject_id)
        if not isinstance(self.kind, str) or not 0 < len(self.kind) <= 32:
            raise ValueError("record kind must contain at most 32 characters")
        if self.kind in {".", ".."} or any(character in self.kind for character in "/\\"):
            raise ValueError("record kind contains a path separator")
        validate_sort_key(self.sort_key)
        _require_state(self.state, "record state", optional=True)
        _require_nonnegative_int(self.storage_revision, "storage_revision")
        _require_optional_string(self.lease_owner, "lease_owner")
        _require_nonnegative_int(self.lease_fence, "lease_fence")
        _require_optional_aware_datetime(self.lease_expires_at, "lease_expires_at")
        if not isinstance(self.data, ImmutableJsonMapping):
            _require_data_mapping(self.data, "record data")
            object.__setattr__(self, "data", ImmutableJsonMapping(self.data))


@dataclass(frozen=True, slots=True)
class StoredFact:
    stream_key: bytes = field(repr=False)
    sequence: int
    owner_key: bytes = field(repr=False)
    kind: str
    subject_id: str | None
    state: str | None
    data: Mapping[str, JsonValue] = field(repr=False)

    def __post_init__(self) -> None:
        validate_stream_key(self.stream_key)
        validate_record_key(self.owner_key)
        if self.subject_id is not None:
            _require_subject_id(self.subject_id)
        _require_nonnegative_int(self.sequence, "fact sequence")
        if not isinstance(self.kind, str) or not 0 < len(self.kind) <= 32:
            raise ValueError("fact kind must contain at most 32 characters")
        _require_state(self.state, "fact state", optional=True)
        if not isinstance(self.data, ImmutableJsonMapping):
            _require_data_mapping(self.data, "fact data")
            object.__setattr__(self, "data", ImmutableJsonMapping(self.data))


@dataclass(frozen=True, slots=True)
class StoredOperation:
    key: bytes = field(repr=False)
    stream_key: bytes = field(repr=False)
    sequence: int
    state: str
    compactable: bool
    data: Mapping[str, JsonValue] = field(repr=False)

    def __post_init__(self) -> None:
        _require_operation_key(self.key)
        validate_stream_key(self.stream_key)
        _require_nonnegative_int(self.sequence, "operation sequence")
        _require_state(self.state, "operation state", optional=False)
        if not isinstance(self.compactable, bool):
            raise TypeError("operation compactable must be a bool")
        if not isinstance(self.data, ImmutableJsonMapping):
            _require_data_mapping(self.data, "operation data")
            object.__setattr__(self, "data", ImmutableJsonMapping(self.data))


@dataclass(frozen=True, slots=True)
class RecordScanCursor:
    kind: str
    key: bytes = field(repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.kind, str) or not self.kind:
            raise ValueError("record scan cursor kind is required")
        validate_record_key(self.key)


@dataclass(frozen=True, slots=True)
class FactScanCursor:
    stream_key: bytes = field(repr=False)
    sequence: int

    def __post_init__(self) -> None:
        validate_stream_key(self.stream_key)
        if isinstance(self.sequence, bool) or not isinstance(self.sequence, int):
            raise TypeError("fact scan cursor sequence must be an integer")
        if self.sequence < 1:
            raise ValueError("fact scan cursor sequence must be positive")


@dataclass(frozen=True, slots=True)
class OperationScanCursor:
    key: bytes = field(repr=False)

    def __post_init__(self) -> None:
        _require_operation_key(self.key)


@dataclass(frozen=True, slots=True)
class Observed(Generic[ValueT]):
    """A value paired with its physical CAS token."""

    value: ValueT
    storage_revision: int

    def __post_init__(self) -> None:
        _require_nonnegative_int(self.storage_revision, "storage_revision")


@dataclass(frozen=True, slots=True)
class RecordReplacement:
    record: StoredRecord
    expected_storage_revision: int

    def __post_init__(self) -> None:
        _require_nonnegative_int(self.expected_storage_revision, "expected_storage_revision")


@dataclass(frozen=True, slots=True)
class RecordQuery:
    """Ordered records; an after-sort without a key excludes that entire sort value."""

    scope_key: bytes | None = field(default=None, repr=False)
    parent_key: bytes | None = field(default=None, repr=False)
    kind: str | None = None
    states: frozenset[str] | None = None
    sort_key_prefix: bytes | None = field(default=None, repr=False)
    after_sort_key: bytes | None = field(default=None, repr=False)
    after_key: bytes | None = field(default=None, repr=False)
    limit: int | None = None

    def __post_init__(self) -> None:
        if self.kind is None and self.scope_key is None and self.parent_key is None:
            raise ValueError("RecordQuery requires a query dimension")
        if self.scope_key is not None:
            validate_relation_key(self.scope_key)
        if self.parent_key is not None:
            validate_parent_key(self.parent_key)
        if self.kind is not None and (
            not isinstance(self.kind, str)
            or not 0 < len(self.kind) <= 32
            or self.kind in {".", ".."}
            or any(character in self.kind for character in "/\\")
        ):
            raise ValueError("record kind contains a path separator")
        if self.sort_key_prefix is not None:
            validate_sort_key(self.sort_key_prefix)
        if self.after_sort_key is None and self.after_key is not None:
            raise ValueError("record cursor key requires a sort key")
        if self.after_sort_key is not None:
            validate_sort_key(self.after_sort_key)
        if self.after_key is not None:
            validate_record_key(self.after_key)
        _validate_limit(self.limit)


@dataclass(frozen=True, slots=True)
class FactQuery:
    stream_key: bytes = field(repr=False)
    after_sequence: int | None = None
    limit: int | None = None
    subject_id: str | None = None
    latest: bool = False
    latest_per_subject: bool = False

    def __post_init__(self) -> None:
        validate_stream_key(self.stream_key)
        if self.after_sequence is not None:
            _require_nonnegative_int(self.after_sequence, "after_sequence")
        if self.subject_id is not None:
            _require_subject_id(self.subject_id)
        if not isinstance(self.latest, bool) or not isinstance(self.latest_per_subject, bool):
            raise TypeError("latest flags must be bool values")
        if self.latest and self.latest_per_subject:
            raise ValueError("latest and latest_per_subject cannot both be true")
        if self.latest_per_subject and self.subject_id is not None:
            raise ValueError("latest_per_subject cannot be combined with subject_id")
        _validate_limit(self.limit)


@dataclass(frozen=True, slots=True)
class OperationQuery:
    stream_key: bytes | None = field(default=None, repr=False)
    states: frozenset[str] | None = None
    through_sequence: int | None = None
    compactable: bool | None = None
    limit: int | None = None
    after_sequence: int | None = None

    def __post_init__(self) -> None:
        if self.after_sequence is not None:
            _require_nonnegative_int(self.after_sequence, "after_sequence")
        if self.stream_key is not None:
            validate_stream_key(self.stream_key)
        if self.through_sequence is not None:
            _require_nonnegative_int(self.through_sequence, "through_sequence")
        if self.compactable is not None and not isinstance(self.compactable, bool):
            raise TypeError("compactable must be a bool or None")
        _validate_limit(self.limit)


@dataclass(frozen=True, slots=True)
class RecordLeaseTransition:
    """Conditional lease change using the backend's current time.

    Acquire requires an available lease and advances the expected fence once.
    Renew, release and guard require the matching live owner and fence.
    Guard_available requires an available lease and its exact current owner
    (including None) and fence. Both guards leave ownership, fence and expiry
    unchanged, while allowing backend-owned physical revision advancement for
    optimistic write protection. Ordinary record
    reads are snapshots, not authorization to execute work. The returned expiry
    may already have passed when a transaction commits or its response arrives;
    callers still need fencing/idempotency at their effect boundary. Durations
    use timedelta microsecond precision and must remain positive after rounding.
    """

    operation: Literal["acquire", "renew", "release", "guard", "guard_available"]
    owner: str | None
    fence: int
    lease_seconds: int | float | None = None

    def __post_init__(self) -> None:
        if self.operation not in ("acquire", "renew", "release", "guard", "guard_available"):
            raise ValueError("invalid lease operation")
        if not (self.operation == "guard_available" and self.owner is None):
            if not isinstance(self.owner, str) or not self.owner.strip():
                raise ValueError("lease owner is required")
        _require_nonnegative_int(self.fence, "fence")
        if self.operation in ("release", "guard", "guard_available"):
            if self.lease_seconds is not None:
                raise ValueError(f"{self.operation} does not accept a lease duration")
        else:
            duration = self.lease_seconds
            if (isinstance(duration, bool) or not isinstance(duration, (int, float))
                    or (isinstance(duration, float) and not math.isfinite(duration))
                    or duration <= 0):
                raise ValueError("lease_seconds must be finite and positive")
            try:
                delta = timedelta(seconds=duration)
            except OverflowError as error:
                raise ValueError("lease_seconds exceeds timedelta precision") from error
            if delta <= timedelta(0):
                raise ValueError("lease_seconds must remain positive at microsecond precision")

    def apply(self, record: StoredRecord, now: datetime) -> StoredRecord | None:
        """Calculate a candidate while the backend protects the record's revision."""
        if now.tzinfo is None:
            raise ValueError("lease time must be timezone aware")
        if record.lease_fence != self.fence:
            return None
        live = (record.lease_owner is not None
                and (record.lease_expires_at is None or record.lease_expires_at > now))
        if self.operation in ("acquire", "guard_available"):
            if live or (self.operation == "guard_available" and record.lease_owner != self.owner):
                return None
        elif (not live or record.lease_owner != self.owner
              or record.lease_expires_at is None):
            return None
        if self.operation in ("guard", "guard_available"):
            return replace(record, storage_revision=record.storage_revision + 1)
        return replace(
            record,
            storage_revision=record.storage_revision + 1,
            lease_owner=None if self.operation == "release" else self.owner,
            lease_fence=self.fence + (self.operation == "acquire"),
            lease_expires_at=(None if self.lease_seconds is None
                              else now + timedelta(seconds=self.lease_seconds)),
        )


class StateTransaction(Protocol):
    async def now(self) -> datetime: ...

    async def get_record(self, key: bytes) -> StoredRecord | None: ...
    async def get_records(self, keys: Sequence[bytes]) -> Mapping[bytes, StoredRecord]: ...
    async def insert_record(self, record: StoredRecord) -> None: ...
    async def insert_records(self, records: Sequence[StoredRecord]) -> None: ...
    async def insert_immutable_records(self, records: Sequence[StoredRecord]) -> None:
        """Insert or confirm byte-exact immutable records without a preceding read.

        Any failure poisons the enclosing mutation, even if the callback catches it.
        """
        ...
    async def guard_record(
        self, key: bytes, *, expected_storage_revision: int,
    ) -> StoredRecord | None:
        """Protect a record once per transaction and return its source-backed view."""
        ...
    async def guard_record_revision(
        self, key: bytes, *, expected_storage_revision: int,
    ) -> int | None:
        """Protect a record without loading or accepting its payload.

        Return the provisional physical revision, or None on a missing/stale key.
        Repeat guards of an already protected key return its current transaction
        revision without another increment. This receipt proves no payload facts
        and becomes confirmed only when the whole mutation group commits.
        """
        ...
    async def replace_record(self, record: StoredRecord, *, expected_storage_revision: int) -> bool: ...
    async def replace_records(self, replacements: Sequence["RecordReplacement"]) -> None:
        """CAS known records without preloading; immutable identities must match.

        A conditional conflict in a multi-record batch aborts the entire group,
        even when caught by the callback. A single zero-match CAS is recoverable.
        Caller receipts advance only after confirmed group commit.
        """
        ...
    async def transition_record_lease(
        self,
        key: bytes,
        *,
        expected_storage_revision: int,
        transition: RecordLeaseTransition,
    ) -> StoredRecord | None:
        """CAS a lease using fresh backend time after obtaining write protection.

        A condition mismatch returns None. SQL may instead abort the mutation
        with STORAGE_CONFLICT when a condition stops holding during the CAS
        wait; even a caught conflict must roll back that mutation. Successful
        physical revision increments are backend owned, not domain revisions.
        Both guards preserve the lease and prove their predicate only at the
        fresh check, not across later awaits or commit completion.
        """
        ...

    async def update_record_lease(
        self,
        key: bytes,
        *,
        expected_storage_revision: int,
        lease_owner: str | None,
        lease_fence: int,
        lease_expires_at: datetime | None,
    ) -> bool: ...
    async def delete_record(self, key: bytes, *, expected_storage_revision: int | None = None) -> bool: ...
    async def delete_records(self, keys: Sequence[bytes]) -> None: ...
    async def list_records(self, query: RecordQuery) -> tuple[StoredRecord, ...]: ...
    async def scan_records(self) -> tuple[StoredRecord, ...]: ...
    async def scan_records_page(
        self,
        *,
        after: RecordScanCursor | None,
        limit: int,
    ) -> tuple[StoredRecord, ...]: ...
    async def get_sequence(self, key: bytes) -> int: ...
    async def get_sequences(self, keys: Sequence[bytes]) -> Mapping[bytes, int]: ...
    async def scan_sequences(self) -> Mapping[bytes, int]: ...
    async def scan_sequences_page(
        self,
        *,
        after: bytes | None,
        limit: int,
    ) -> Mapping[bytes, int]: ...
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
    async def scan_facts_page(
        self,
        *,
        after: FactScanCursor | None,
        limit: int,
    ) -> tuple[StoredFact, ...]: ...
    async def delete_fact_streams(self, owner_key: bytes) -> None: ...
    async def insert_operation(self, value: StoredOperation) -> None: ...
    async def get_operation(self, key: bytes) -> StoredOperation | None: ...
    async def replace_operation(self, value: StoredOperation, *, expected_state: str) -> bool: ...
    async def list_operations(self, query: OperationQuery) -> tuple[StoredOperation, ...]: ...
    async def scan_operations(self) -> tuple[StoredOperation, ...]: ...
    async def scan_operations_page(
        self,
        *,
        after: OperationScanCursor | None,
        limit: int,
    ) -> tuple[StoredOperation, ...]: ...
    async def delete_operations(self, query: OperationQuery) -> tuple[StoredOperation, ...]: ...


@runtime_checkable
class RecordSubjectReader(Protocol):
    """Optional exact-subject reads within one record scope and kind.

    Return matching records in ascending (sort_key, key) order from the
    transaction's current view, including its own writes. Subject duplicates
    do not duplicate rows, but multiple records for one subject are retained.
    Empty input returns an empty tuple without source reads. Parameter limits
    may require several statements without strengthening isolation. Return
    None only when this capability is unavailable, never for an empty match.
    """

    async def read_record_subjects(
        self, *, scope_key: bytes, kind: str, subject_ids: Sequence[str],
    ) -> tuple[StoredRecord, ...] | None: ...


@runtime_checkable
class FactReader(Protocol):
    """Optional exact-sequence reads in the current state transaction.

    Return only found facts in this store and stream, once per requested
    sequence, in ascending sequence order. Input order and duplicates do not
    affect the result. Missing sequences are omitted; empty input returns an
    empty tuple without source reads. Reuse the transaction's read view,
    including its own writes. Parameter limits may require multiple statements;
    this operation does not strengthen isolation or bound returned bytes.
    """

    async def get_facts(
        self, stream_key: bytes, sequences: Sequence[int],
    ) -> tuple[StoredFact, ...]: ...


@runtime_checkable
class SqlStateRowReader(Protocol):
    """Read immutable native rows in the active state transaction's snapshot."""

    async def read_sql_rows(
        self, context: "SqlStorageContext", statement: "Select",
    ) -> tuple[Mapping[str, object], ...]: ...


@runtime_checkable
class SqlStateRecordReader(Protocol):
    """Join immutable sources to scoped records within the current read boundary.

    Results retain the physical record separately from joined columns. Raw
    scans and transaction caches continue to expose the persisted record.
    """

    def cached_records(self, keys: Sequence[bytes]) -> Mapping[bytes, StoredRecord | None]: ...

    async def read_joined_records(
        self, context: "SqlStorageContext", statement: "Select", *, keys: Sequence[bytes] = (),
    ) -> tuple[tuple[StoredRecord, Mapping[str, object]], ...]: ...


@runtime_checkable
class SqlStateFactReader(Protocol):
    """Join immutable sources to physical facts in the current read boundary.

    Select all physical fact columns under their original names and give
    joined columns distinct labels. The caller owns store/stream predicates,
    ordering and pagination. Results keep the persisted fact separate from
    joined source columns; projections never replace physical fact data.
    Each call executes only the supplied read-only Select, without follow-up
    reads or writes.
    """

    async def read_joined_facts(
        self, context: "SqlStorageContext", statement: "Select",
    ) -> tuple[tuple[StoredFact, Mapping[str, "SqlValue"]], ...]: ...


@runtime_checkable
class SqlStateTransaction(Protocol):
    """Execute prepared SQL inside the current writable storage callback.

    The callback must only write through the supplied session. It must not
    begin, commit or close a transaction, nor perform external operations.
    Receipts are provisional until the enclosing StateStore mutation confirms.
    """

    async def run_sql(
        self, context: "SqlStorageContext",
        callback: Callable[["AsyncSession"], Awaitable[ValueT]],
    ) -> ValueT: ...


@runtime_checkable
class SqlMutationObserver(Protocol):
    """Observe the real enclosing SQL commit, never a nested callback return."""

    def observe_mutation(self, callback: Callable[[bool], None]) -> None: ...


class _ReadOnlyStateTransaction(StateTransaction):
    def __init__(self, transaction: StateTransaction) -> None:
        self._transaction = transaction

    async def now(self) -> datetime:
        return await self._transaction.now()

    async def get_record(self, key: bytes) -> StoredRecord | None:
        return await self._transaction.get_record(key)

    async def get_records(self, keys: Sequence[bytes]) -> Mapping[bytes, StoredRecord]:
        return await self._transaction.get_records(keys)

    def cached_records(self, keys: Sequence[bytes]) -> Mapping[bytes, StoredRecord | None]:
        if not isinstance(self._transaction, SqlStateRecordReader):
            raise StateTransactionNestingError("joined record reads require SQL storage")
        return self._transaction.cached_records(keys)

    async def read_sql_rows(
        self, context: "SqlStorageContext", statement: "Select",
    ) -> tuple[Mapping[str, object], ...]:
        if not isinstance(self._transaction, SqlStateRowReader):
            raise StateTransactionNestingError("native row reads require SQL storage")
        return await self._transaction.read_sql_rows(context, statement)

    async def read_joined_records(
        self, context: "SqlStorageContext", statement: "Select", *, keys: Sequence[bytes] = (),
    ) -> tuple[tuple[StoredRecord, Mapping[str, object]], ...]:
        if not isinstance(self._transaction, SqlStateRecordReader):
            raise StateTransactionNestingError("joined record reads require SQL storage")
        return await self._transaction.read_joined_records(context, statement, keys=keys)

    async def read_joined_facts(
        self, context: "SqlStorageContext", statement: "Select",
    ) -> tuple[tuple[StoredFact, Mapping[str, "SqlValue"]], ...]:
        if not isinstance(self._transaction, SqlStateFactReader):
            raise StateTransactionNestingError("joined fact reads require SQL storage")
        return await self._transaction.read_joined_facts(context, statement)

    async def read_record_subjects(
        self, *, scope_key: bytes, kind: str, subject_ids: Sequence[str],
    ) -> tuple[StoredRecord, ...] | None:
        if isinstance(self._transaction, RecordSubjectReader):
            return await self._transaction.read_record_subjects(
                scope_key=scope_key, kind=kind, subject_ids=subject_ids,
            )
        return None

    async def read_archive_frontiers(
        self, *, scope_key: bytes, requests: Sequence[ArchiveNodeRead],
        content_keys: Sequence[bytes] = (),
    ) -> Mapping[bytes, StoredRecord] | None:
        if isinstance(self._transaction, ArchiveFrontierReader):
            return await self._transaction.read_archive_frontiers(
                scope_key=scope_key, requests=requests, content_keys=content_keys,
            )
        return None

    async def guard_record(
        self, key: bytes, *, expected_storage_revision: int,
    ) -> StoredRecord | None:
        del key, expected_storage_revision
        self._reject("guard_record")
        return None

    async def guard_record_revision(
        self, key: bytes, *, expected_storage_revision: int,
    ) -> int | None:
        del key, expected_storage_revision
        self._reject("guard_record_revision")
        return None

    async def list_records(self, query: RecordQuery) -> tuple[StoredRecord, ...]:
        return await self._transaction.list_records(query)

    async def scan_records(self) -> tuple[StoredRecord, ...]:
        return await self._transaction.scan_records()

    async def scan_records_page(
        self,
        *,
        after: RecordScanCursor | None,
        limit: int,
    ) -> tuple[StoredRecord, ...]:
        return await self._transaction.scan_records_page(after=after, limit=limit)


    async def get_sequence(self, key: bytes) -> int:
        return await self._transaction.get_sequence(key)

    async def get_sequences(self, keys: Sequence[bytes]) -> Mapping[bytes, int]:
        return await self._transaction.get_sequences(keys)

    async def scan_sequences(self) -> Mapping[bytes, int]:
        return await self._transaction.scan_sequences()

    async def scan_sequences_page(
        self,
        *,
        after: bytes | None,
        limit: int,
    ) -> Mapping[bytes, int]:
        return await self._transaction.scan_sequences_page(after=after, limit=limit)

    async def list_facts(self, query: FactQuery) -> tuple[StoredFact, ...]:
        return await self._transaction.list_facts(query)

    async def scan_facts(self) -> tuple[StoredFact, ...]:
        return await self._transaction.scan_facts()

    async def scan_facts_page(
        self,
        *,
        after: FactScanCursor | None,
        limit: int,
    ) -> tuple[StoredFact, ...]:
        return await self._transaction.scan_facts_page(after=after, limit=limit)

    async def get_operation(self, key: bytes) -> StoredOperation | None:
        return await self._transaction.get_operation(key)

    async def list_operations(self, query: OperationQuery) -> tuple[StoredOperation, ...]:
        return await self._transaction.list_operations(query)

    async def scan_operations(self) -> tuple[StoredOperation, ...]:
        return await self._transaction.scan_operations()

    async def scan_operations_page(
        self,
        *,
        after: OperationScanCursor | None,
        limit: int,
    ) -> tuple[StoredOperation, ...]:
        return await self._transaction.scan_operations_page(
            after=after,
            limit=limit,
        )

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

    async def insert_immutable_records(self, records: Sequence[StoredRecord]) -> None:
        del records
        self._reject("insert_immutable_records")

    async def replace_record(self, record: StoredRecord, *, expected_storage_revision: int) -> bool:
        del record, expected_storage_revision
        self._reject("replace_record")
        return False

    async def replace_records(self, replacements: Sequence[RecordReplacement]) -> None:
        del replacements
        self._reject("replace_records")

    async def transition_record_lease(
        self,
        key: bytes,
        *,
        expected_storage_revision: int,
        transition: RecordLeaseTransition,
    ) -> StoredRecord | None:
        del key, expected_storage_revision, transition
        self._reject("transition_record_lease")
        return None

    async def update_record_lease(
        self,
        key: bytes,
        *,
        expected_storage_revision: int,
        lease_owner: str | None,
        lease_fence: int,
        lease_expires_at: datetime | None,
    ) -> bool:
        del key, expected_storage_revision, lease_owner, lease_fence, lease_expires_at
        self._reject("update_record_lease")
        return False

    async def delete_record(self, key: bytes, *, expected_storage_revision: int | None = None) -> bool:
        del key, expected_storage_revision
        self._reject("delete_record")
        return False

    async def delete_records(self, keys: Sequence[bytes]) -> None:
        del keys
        self._reject("delete_records")


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


class _ReadOnlyFactReader(_ReadOnlyStateTransaction):
    def __init__(self, transaction: StateTransaction, reader: FactReader) -> None:
        super().__init__(transaction)
        self._fact_reader = reader

    async def get_facts(
        self, stream_key: bytes, sequences: Sequence[int],
    ) -> tuple[StoredFact, ...]:
        return await self._fact_reader.get_facts(stream_key, sequences)


StateCallback = Callable[[StateTransaction], Awaitable[ValueT]]
StateGroupCallback = Callable[["StateGroupTransaction"], Awaitable[ValueT]]


class StateGroupTransaction(Protocol):
    def transaction(self, store: "StateStore") -> StateTransaction: ...


@runtime_checkable
class GroupRecordReader(Protocol):
    """Optional exact-key reads across enlisted stores in the current callback.

    Returned mappings contain found records only. Requested but absent keys are
    observed misses, not unsupported reads. Implementations reuse each store's
    transaction-local read view and preserve its updates after own writes.
    Parameter limits may require multiple statements; this operation does not
    strengthen the backend's transaction isolation or bound returned bytes.
    """

    async def get_records(
        self, requests: Mapping["StateStore", Sequence[bytes]],
    ) -> Mapping["StateStore", Mapping[bytes, StoredRecord]]: ...


@runtime_checkable
class StateStorageGroup(Protocol):
    async def open_partition(
        self, key: bytes, *, create: bool = False,
    ) -> "StateStore | None":
        """Borrow an isolated logical store owned by this existing group.

        Keys are opaque binary storage identities of 1..768 bytes. Open outside
        transaction callbacks; opening never changes record keys or transaction boundaries.
        Read-only opens do not create partitions. The group owns their lifetime.
        """
        ...

    @property
    def opened_archive(self) -> tuple[str, "StateStore"] | None:
        """Return an already opened archive handle without I/O or enlistment.

        Callbacks must still verify that its store belongs to their active
        transaction. An absent handle never authorizes opening storage there.
        """
        ...

    async def open_archive(
        self, *, create: bool = False,
    ) -> tuple[str, "StateStore"] | None:
        """Open group-owned archive metadata and its stable opaque key scope.

        Open outside state callbacks, before preparing or publishing owners.
        Read-only opens never create identity or provision storage. Restoring the
        original resource preserves its scope; independent clones must import
        and retain content into a new resource. Identity does not grant GC
        exclusivity and cannot detect manually copied resources. Once opened,
        this store is automatically enlisted in subsequent group callbacks so
        owner and archive writes share one physical commit.
        """
        ...

    async def scan_stores(self) -> tuple["StateStore", ...]:
        """Enumerate all persisted group partitions, not just active routes.

        Destructive collection still requires externally proven offline
        exclusivity for every writer's prepare-through-publication window.
        """
        ...

    async def read(self, store: "StateStore", fn: StateCallback[ValueT]) -> ValueT: ...

    async def read_many(
        self,
        stores: Sequence["StateStore"],
        fn: StateGroupCallback[ValueT],
    ) -> ValueT:
        """Read enlisted stores in one consistent, task-local callback scope."""
        ...

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
    async def read_record(self, key: bytes) -> StoredRecord | None: ...
    async def read_records(self, keys: Sequence[bytes]) -> Mapping[bytes, StoredRecord]: ...
    async def read(self, fn: StateCallback[ValueT]) -> ValueT: ...
    async def mutate(self, fn: StateCallback[ValueT]) -> ValueT: ...

    async def validate_integrity(self) -> None: ...


def record_key(
    namespace: str,
    tenant_id: str,
    runtime_domain: str,
    kind: str,
    identity: Sequence[str],
) -> bytes:
    """Encode one exact globally scoped record identity with explicit owner parts."""
    if not all(isinstance(value, str) and value for value in (runtime_domain, kind)):
        raise ValueError("record domain and kind must be non-empty strings")
    if not isinstance(identity, Sequence) or isinstance(identity, (str, bytes)):
        raise TypeError("record identity must be a flat sequence of strings")
    if not identity:
        raise ValueError("record identity must contain at least one component")
    # Group-owned archive records use the reserved empty binding, not a tenant.
    if not (namespace == tenant_id == "" and runtime_domain == "archive"):
        validate_persistence_namespace(namespace)
        validate_tenant_id(tenant_id)
    return _encode_key_components(
        ("record", namespace, tenant_id, runtime_domain, kind, *identity),
        RECORD_KEY_MAX_BYTES,
    )


def validate_parent_key(value: bytes) -> bytes:
    """A parent can carry either a relation address or a complete record reference."""
    if type(value) is not bytes or not 1 <= len(value) <= PARENT_KEY_MAX_BYTES:
        raise ValueError(f"parent key must contain 1..{PARENT_KEY_MAX_BYTES} bytes")
    return value


def state_store_key(namespace: str, tenant_id: str, partition: str) -> bytes:
    """Compact the complete ordinary or configured capability partition once."""
    if not isinstance(partition, str) or not partition:
        raise ValueError("store partition must be a non-empty string")
    return hashlib.sha256(_encode_key_components(
        ("state_store", validate_persistence_namespace(namespace), validate_tenant_id(tenant_id), partition),
        _STORE_SOURCE_MAX_BYTES,
    )).digest()


def capability_store_key(namespace: str, tenant_id: str, capability_id: str) -> bytes:
    """Compact the complete dynamic capability partition in its own family."""
    return hashlib.sha256(_encode_key_components(
        ("capability-store", validate_persistence_namespace(namespace),
         validate_tenant_id(tenant_id), validate_capability_id(capability_id)),
        _STORE_SOURCE_MAX_BYTES,
    )).digest()


def standalone_store_key(partition: str) -> bytes:
    """Compact one standalone SQL store's complete declared partition once."""
    if not isinstance(partition, str) or not partition:
        raise ValueError("store partition must be a non-empty string")
    return hashlib.sha256(_encode_key_components(("standalone", partition), _STORE_SOURCE_MAX_BYTES)).digest()


def validate_store_key(value: bytes) -> bytes:
    """Accept bounded opaque keys without imposing built-in tuple syntax."""
    if type(value) is not bytes or not 1 <= len(value) <= STORE_KEY_MAX_BYTES:
        raise ValueError(f"store key must contain 1..{STORE_KEY_MAX_BYTES} bytes")
    return value


def scope_key(
    namespace: str,
    tenant_id: str,
    runtime_domain: str,
    kind: str,
    relation: str,
    identity: Sequence[str],
) -> bytes:
    """Encode one globally scoped, flat grouping identity without hashing."""
    return _relation_key("scope", namespace, tenant_id, runtime_domain, kind, relation, identity)


def parent_key(
    namespace: str,
    tenant_id: str,
    runtime_domain: str,
    kind: str,
    relation: str,
    identity: Sequence[str],
) -> bytes:
    """Encode one globally scoped, flat parent identity without hashing."""
    return _relation_key("parent", namespace, tenant_id, runtime_domain, kind, relation, identity)


def _relation_key(
    role: str,
    namespace: str,
    tenant_id: str,
    runtime_domain: str,
    kind: str,
    relation: str,
    identity: Sequence[str],
) -> bytes:
    if not all(isinstance(value, str) and value for value in (runtime_domain, kind, relation)):
        raise ValueError("relation domain, kind, and name must be non-empty strings")
    if not isinstance(identity, Sequence) or isinstance(identity, (str, bytes)):
        raise TypeError("relation identity must be a flat sequence of strings")
    if not identity:
        raise ValueError("relation identity must contain at least one component")
    return _encode_key_components(
        (role, validate_persistence_namespace(namespace), validate_tenant_id(tenant_id),
         runtime_domain, kind, relation, *identity),
        RELATION_KEY_MAX_BYTES,
    )


def validate_relation_key(value: bytes) -> bytes:
    """Validate a bounded relation address, including reused binary identities."""
    if type(value) is not bytes or not 1 <= len(value) <= RELATION_KEY_MAX_BYTES:
        raise ValueError(f"relation key must contain 1..{RELATION_KEY_MAX_BYTES} bytes")
    return value


def stream_key(
    namespace: str,
    tenant_id: str,
    runtime_domain: str,
    relation: str,
    identity: Sequence[str],
) -> bytes:
    """Encode a globally scoped stream from explicit, exact UTF-8 owner parts."""
    if not all(isinstance(value, str) and value for value in (runtime_domain, relation)):
        raise ValueError("stream domain and relation must be non-empty strings")
    if not isinstance(identity, Sequence) or isinstance(identity, (str, bytes)):
        raise TypeError("stream identity must be a flat sequence of strings")
    return _encode_key_components(
        ("stream", validate_persistence_namespace(namespace), validate_tenant_id(tenant_id),
         runtime_domain, relation, *identity),
        STREAM_KEY_MAX_BYTES,
    )


def validate_stream_key(value: bytes) -> bytes:
    """Validate the complete bounded stream/counter address, including custom keys."""
    if type(value) is not bytes or not 1 <= len(value) <= STREAM_KEY_MAX_BYTES:
        raise ValueError(f"stream key must contain 1..{STREAM_KEY_MAX_BYTES} bytes")
    return value


def capability_receipt_key(
    namespace: str, tenant_id: str, capability_id: str, revision: int,
    run_id: str, write_key: str,
) -> bytes:
    """Address the retained capability write identity in its operation family."""
    return b"\x02" + _encode_key_components(
        ("capability-write", validate_persistence_namespace(namespace),
         validate_tenant_id(tenant_id), validate_capability_id(capability_id),
         str(validate_capability_revision(revision)), validate_resource_id(run_id),
         validate_capability_write_key(write_key)),
        OPERATION_KEY_MAX_BYTES - 1,
    )


def operation_key(
    namespace: str,
    tenant_id: str,
    runtime_domain: str,
    identity: str,
) -> bytes:
    """Address an ordinary ledger ID independently of StateStore partitioning."""
    return b"\x01" + _encode_key_components(
        (validate_persistence_namespace(namespace), validate_tenant_id(tenant_id),
         runtime_domain, validate_idempotency_key(identity)),
        OPERATION_KEY_MAX_BYTES - 1,
    )


def sortable_timestamp(value: datetime, key: bytes) -> bytes:
    if value.tzinfo is None:
        raise ValueError("sortable timestamp must be timezone-aware")
    delta = value.astimezone(timezone.utc) - datetime(1970, 1, 1, tzinfo=timezone.utc)
    micros = (delta.days * 86400 + delta.seconds) * 1_000_000 + delta.microseconds
    if micros < 0 or micros >= 1 << 63:
        raise ValueError("sortable timestamp is outside the supported range")
    return validate_sort_key(micros.to_bytes(8, "big") + validate_record_key(key))


def validate_sort_key(value: bytes) -> bytes:
    """Validate opaque ordering data without imposing identity-key bounds."""
    if type(value) is not bytes or not value:
        raise ValueError("sort key must contain nonempty bytes")
    return value


def validate_record_identity(record: StoredRecord) -> None:
    """Validate physical invariants after decoding a backend row or file."""
    validate_record_key(record.key)
    if record.scope_key is not None:
        validate_relation_key(record.scope_key)
    if record.parent_key is not None:
        validate_parent_key(record.parent_key)
    if record.subject_id is not None:
        _require_subject_id(record.subject_id)
    validate_sort_key(record.sort_key)
    if type(record.data) is ImmutableJsonMapping:
        return
    try:
        canonical_json_bytes(dict(record.data))
    except (TypeError, ValueError) as error:
        raise ValueError("record data is not canonical JSON") from error


def validate_record_replacement(current: StoredRecord, candidate: StoredRecord) -> None:
    """Keep immutable physical identity columns stable across a CAS."""
    if (
        current.key != candidate.key
        or current.scope_key != candidate.scope_key
        or current.parent_key != candidate.parent_key
        or current.subject_id != candidate.subject_id
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
        current.key != candidate.key
        or current.stream_key != candidate.stream_key
        or current.sequence != candidate.sequence
        or current.compactable != candidate.compactable
    ):
        raise ValueError("operation physical identity cannot change")


def _validate_limit(limit: int | None) -> None:
    if limit is None:
        return
    if isinstance(limit, bool) or not isinstance(limit, int) or not 0 < limit <= 1000:
        raise ValueError("limit must be between 1 and 1000")


STATE_GROUP_METADATA_KEY = _encode_key_components(("state-group-metadata",), STORE_KEY_MAX_BYTES)
STATE_GROUP_ARCHIVE_KEY = _encode_key_components(("state-group-archive",), STORE_KEY_MAX_BYTES)


__all__ = ['STORE_KEY_MAX_BYTES', 'validate_store_key', 'capability_store_key', 'standalone_store_key', 'STATE_GROUP_METADATA_KEY', 'STATE_GROUP_ARCHIVE_KEY', 'STREAM_KEY_MAX_BYTES', 'validate_stream_key', 'capability_receipt_key', 'RECORD_KEY_MAX_BYTES', 'PARENT_KEY_MAX_BYTES', 'validate_record_key', 'validate_parent_key', 'RELATION_KEY_MAX_BYTES', 'validate_relation_key', 'OPERATION_KEY_MAX_BYTES', 'SqlMutationObserver', 'SqlStateRowReader', 'RecordSubjectReader', 'FactReader', 'FactQuery', 'FactScanCursor', 'GroupRecordReader', 'Observed', 'OperationQuery', 'OperationScanCursor', 'RecordLeaseTransition', 'RecordQuery', 'RecordReplacement', 'RecordScanCursor', 'StateGroupTransaction', 'StateLockOrderError', 'StateStorageGroup', 'StateStore', 'StateTransaction', 'SqlStateTransaction', 'SqlStateRecordReader', 'SqlStateFactReader', 'StateTransactionNestingError', 'StoredFact', 'StoredOperation', 'StoredRecord', 'active_state_scope', 'is_active_state_transaction', 'validate_sort_key', 'enter_run_history_lock', 'exit_run_history_lock', 'operation_key', 'parent_key', 'record_key', 'require_no_run_history_lock', 'scope_key', 'sortable_timestamp', 'state_store_key', 'stream_key', 'validate_operation_replacement', 'validate_record_identity', 'validate_record_replacement']
