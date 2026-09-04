#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Terminal execution read models backed by Runtime StateStore facts."""

import asyncio
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from uuid import uuid4

from linktools.core import environ

from ...core import JsonValue
from ...errors import AIError, ErrorCode
from ._codec import _decode_enveloped_domain
from ._contracts import ExecutionHistorySealRecord
from ._store import (
    FactQuery,
    StateStore,
    StateTransaction,
    StoredFact,
    StoredRecord,
    partition_digest,
    record_key_digest,
    sequence_key,
    sortable_identity,
    stream_digest,
)

_logger = environ.get_logger("ai.runtime.state.readmodel")
_CHUNK_SIZE = 128
_LEASE_SECONDS = 30
_MODEL_VERSION = 1


class ExecutionReadModelStatus(str, Enum):
    __str__ = str.__str__
    __format__ = str.__format__
    BUILDING = "BUILDING"
    COMPLETE = "COMPLETE"


@dataclass(frozen=True, slots=True)
class ExecutionReadModelRecord:
    execution_id: str
    tenant_id: str
    source_digest: str
    model_version: int
    status: ExecutionReadModelStatus
    trace_count: int
    history_count: int
    transcript_count: int
    revision: int

    def __post_init__(self) -> None:
        values = (
            self.model_version,
            self.trace_count,
            self.history_count,
            self.transcript_count,
            self.revision,
        )
        if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
            raise TypeError("execution read model counters must be integers")
        if self.model_version != _MODEL_VERSION:
            raise ValueError("unsupported execution read model version")
        if self.revision < 0 or any(
            count < 0
            for count in (
                self.trace_count,
                self.history_count,
                self.transcript_count,
            )
        ):
            raise ValueError("execution read model counts are invalid")


@dataclass(frozen=True, slots=True)
class ExecutionReadModelBuild:
    execution_id: str
    tenant_id: str
    source_digest: str
    trace_items: tuple[Mapping[str, JsonValue], ...]
    history_items: tuple[Mapping[str, JsonValue], ...]
    transcript_items: tuple[Mapping[str, JsonValue], ...]


class _ExecutionBuildFlights:
    def __init__(self) -> None:
        self._guard = asyncio.Lock()
        self._flights: dict[str, asyncio.Task[ExecutionReadModelRecord]] = {}

    async def run(
        self,
        execution_id: str,
        builder: Callable[[], Awaitable[ExecutionReadModelRecord]],
    ) -> ExecutionReadModelRecord:
        async with self._guard:
            task = self._flights.get(execution_id)
            if task is None:
                task = asyncio.create_task(builder())
                self._flights[execution_id] = task
                task.add_done_callback(
                    lambda completed: asyncio.create_task(
                        self._remove(execution_id, completed)
                    )
                )
        try:
            return await asyncio.shield(task)
        finally:
            if task.done():
                async with self._guard:
                    if self._flights.get(execution_id) is task:
                        self._flights.pop(execution_id, None)

    async def _remove(
        self,
        execution_id: str,
        task: asyncio.Task[ExecutionReadModelRecord],
    ) -> None:
        async with self._guard:
            if self._flights.get(execution_id) is task:
                self._flights.pop(execution_id, None)


class ExecutionReadModelRepository:
    """Build and page derived terminal execution streams."""

    def __init__(self, store: StateStore, *, namespace: str, tenant_id: str) -> None:
        self._store = store
        self._namespace = namespace
        self._tenant_id = tenant_id
        self._flights = _ExecutionBuildFlights()

    async def ensure(
        self,
        execution_id: str,
        *,
        tenant_id: str,
        builder: Callable[[], Awaitable[ExecutionReadModelBuild]],
    ) -> ExecutionReadModelRecord:
        if tenant_id != self._tenant_id:
            raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)
        seal = await self._load_history_seal(execution_id)
        if seal is None:
            raise AIError(ErrorCode.EXECUTION_HISTORY_UNAVAILABLE)

        async def build_and_publish() -> ExecutionReadModelRecord:
            existing = await self.get_complete(
                execution_id,
                tenant_id=tenant_id,
            )
            if existing is not None:
                return existing
            build = await builder()
            if (
                build.execution_id != execution_id
                or build.tenant_id != tenant_id
                or not build.source_digest
            ):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            owner, fence, claimed = await self._claim(execution_id)
            if claimed is not None:
                if claimed.source_digest != build.source_digest:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                return claimed
            current_seal = await self._load_history_seal(execution_id)
            if current_seal != seal:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            await self._write_build(build, owner, fence)
            result = ExecutionReadModelRecord(
                execution_id,
                tenant_id,
                build.source_digest,
                _MODEL_VERSION,
                ExecutionReadModelStatus.COMPLETE,
                len(build.trace_items),
                len(build.history_items),
                len(build.transcript_items),
                1,
            )
            _logger.info(
                "execution read model completed: execution=%s trace=%s history=%s transcript=%s",
                execution_id,
                result.trace_count,
                result.history_count,
                result.transcript_count,
            )
            return result

        return await self._flights.run(execution_id, build_and_publish)

    async def get_complete(
        self,
        execution_id: str,
        *,
        tenant_id: str,
    ) -> ExecutionReadModelRecord | None:
        if tenant_id != self._tenant_id:
            return None
        if await self._load_history_seal(execution_id) is None:
            return None
        record = await self._store.read(
            lambda transaction: transaction.get_record(self._record_key(execution_id))
        )
        if record is None:
            return None
        self._validate_stored_record(record, execution_id)
        if self._stored_model_version(record) != _MODEL_VERSION:
            _logger.error(
                "execution read model version rejected: execution=%s",
                execution_id,
            )
            raise AIError(ErrorCode.STORAGE_VERSION_UNSUPPORTED)
        value = self._decode_record(record)
        self._validate_owner(value, execution_id)
        return value if value.status is ExecutionReadModelStatus.COMPLETE else None

    async def page(
        self,
        execution_id: str,
        *,
        tenant_id: str,
        stream_name: str,
        offset: int,
        limit: int,
    ) -> tuple[ExecutionReadModelRecord, tuple[Mapping[str, JsonValue], ...]]:
        if tenant_id != self._tenant_id:
            raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)
        if (
            isinstance(offset, bool)
            or not isinstance(offset, int)
            or isinstance(limit, bool)
            or not isinstance(limit, int)
            or offset < 0
            or limit < 1
        ):
            raise AIError(ErrorCode.PAGE_LIMIT_INVALID)
        if await self._load_history_seal(execution_id) is None:
            raise AIError(ErrorCode.EXECUTION_HISTORY_UNAVAILABLE)

        async def read(
            transaction: StateTransaction,
        ) -> tuple[ExecutionReadModelRecord, tuple[Mapping[str, JsonValue], ...]]:
            record = await transaction.get_record(self._record_key(execution_id))
            if record is None:
                raise AIError(ErrorCode.EXECUTION_HISTORY_UNAVAILABLE)
            self._validate_stored_record(record, execution_id)
            if self._stored_model_version(record) != _MODEL_VERSION:
                _logger.error(
                    "execution read model version rejected: execution=%s",
                    execution_id,
                )
                raise AIError(ErrorCode.STORAGE_VERSION_UNSUPPORTED)
            value = self._decode_record(record)
            self._validate_owner(value, execution_id)
            if value.status is not ExecutionReadModelStatus.COMPLETE:
                raise AIError(ErrorCode.EXECUTION_HISTORY_UNAVAILABLE)
            count = self._stream_count(value, stream_name)
            if offset > count:
                raise AIError(ErrorCode.CURSOR_INVALID)
            if offset == count:
                return value, ()
            first_chunk = offset // _CHUNK_SIZE
            local_offset = offset % _CHUNK_SIZE
            after_sequence = first_chunk
            selected: list[Mapping[str, JsonValue]] = []
            while len(selected) < limit and after_sequence < (count + _CHUNK_SIZE - 1) // _CHUNK_SIZE:
                facts = await transaction.list_facts(
                    FactQuery(
                        self._stream(execution_id, stream_name),
                        after_sequence=after_sequence,
                        limit=1,
                    )
                )
                if not facts:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                fact = facts[0]
                if (
                    fact.owner_key_digest != self._record_key(execution_id)
                    or fact.kind != f"execution_read_{stream_name}"
                    or fact.sequence != after_sequence + 1
                ):
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                after_sequence = fact.sequence
                items = fact.data.get("items")
                if not isinstance(items, list):
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                for item in items[local_offset:]:
                    if not isinstance(item, Mapping):
                        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                    selected.append(item)
                    if len(selected) == limit:
                        break
                local_offset = 0
            return value, tuple(selected)

        return await self._store.read(read)

    async def _claim(
        self,
        execution_id: str,
    ) -> tuple[str, int, ExecutionReadModelRecord | None]:
        owner = uuid4().hex
        while True:
            result = await self._store.mutate(
                lambda transaction: self._claim_in_transaction(
                    transaction,
                    execution_id,
                    owner,
                )
            )
            if result[2] is not None:
                return result
            if result[0] == owner:
                return result
            current = await self._store.read(
                lambda transaction: transaction.get_record(
                    self._record_key(execution_id)
                )
            )
            if current is None or current.lease_expires_at is None:
                await asyncio.sleep(0.05)
                continue
            remaining = (
                current.lease_expires_at - datetime.now(timezone.utc)
            ).total_seconds()
            await asyncio.sleep(max(0.01, min(0.25, remaining)))

    async def _load_history_seal(
        self,
        execution_id: str,
    ) -> ExecutionHistorySealRecord | None:
        async def read(transaction: StateTransaction) -> ExecutionHistorySealRecord | None:
            record = await transaction.get_record(self._history_seal_key(execution_id))
            if record is None:
                return None
            if record.kind != "execution_history_seal":
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            try:
                value = _decode_enveloped_domain(
                    record.data,
                    ExecutionHistorySealRecord,
                )
            except (TypeError, ValueError, KeyError, AIError) as error:
                if isinstance(error, AIError):
                    raise
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
            if (
                value.execution_id != execution_id
                or value.tenant_id != self._tenant_id
            ):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            return value

        return await self._store.read(read)

    async def _claim_in_transaction(
        self,
        transaction: StateTransaction,
        execution_id: str,
        owner: str,
    ) -> tuple[str, int, ExecutionReadModelRecord | None]:
        key = self._record_key(execution_id)
        current = await transaction.get_record(key)
        now = await transaction.now()
        if current is None:
            value = self._building_value(execution_id)
            await transaction.insert_record(
                self._stored_record(value, lease_owner=owner, lease_fence=1, expires=now)
            )
            return owner, 1, None
        self._validate_stored_record(current, execution_id)
        if self._stored_model_version(current) != _MODEL_VERSION:
            _logger.error(
                "execution read model version rejected: execution=%s",
                execution_id,
            )
            raise AIError(ErrorCode.STORAGE_VERSION_UNSUPPORTED)
        value = self._decode_record(current)
        self._validate_owner(value, execution_id)
        if value.status is ExecutionReadModelStatus.COMPLETE:
            return owner, current.lease_fence, value
        if (
            current.lease_owner is not None
            and current.lease_expires_at is not None
            and current.lease_expires_at > now
        ):
            return current.lease_owner, current.lease_fence, None
        next_value = self._building_value(execution_id)
        next_fence = current.lease_fence + 1
        candidate = self._stored_record(
            next_value,
            storage_version=current.storage_version + 1,
            lease_owner=owner,
            lease_fence=next_fence,
            expires=now,
        )
        if not await transaction.replace_record(
            candidate,
            expected_storage_version=current.storage_version,
        ):
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        await transaction.delete_fact_streams(key)
        await transaction.delete_sequences(
            tuple(
                self._sequence_key(execution_id, stream_name)
                for stream_name in ("trace", "history", "transcript")
            )
        )
        _logger.warning("execution read model rebuild claimed: execution=%s", execution_id)
        return owner, next_fence, None

    async def _write_build(
        self,
        build: ExecutionReadModelBuild,
        owner: str,
        fence: int,
    ) -> None:
        streams = (
            ("trace", build.trace_items),
            ("history", build.history_items),
            ("transcript", build.transcript_items),
        )
        chunk_count = max(
            ((len(values) + _CHUNK_SIZE - 1) // _CHUNK_SIZE for _, values in streams),
            default=0,
        )
        if chunk_count == 0:
            await self._write_ordinal(build, (), owner, fence, complete=True)
            return
        for ordinal in range(chunk_count):
            start = ordinal * _CHUNK_SIZE
            prepared = tuple(
                (
                    stream_name,
                    {"items": [dict(item) for item in values[start : start + _CHUNK_SIZE]]},
                )
                for stream_name, values in streams
                if start < len(values)
            )
            await self._write_ordinal(
                build,
                prepared,
                owner,
                fence,
                complete=ordinal == chunk_count - 1,
            )

    async def _write_ordinal(
        self,
        build: ExecutionReadModelBuild,
        chunks: Sequence[tuple[str, Mapping[str, JsonValue]]],
        owner: str,
        fence: int,
        *,
        complete: bool,
    ) -> None:
        async def mutate(transaction: StateTransaction) -> None:
            key = self._record_key(build.execution_id)
            current = await transaction.get_record(key)
            if current is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            self._validate_stored_record(current, build.execution_id)
            value = self._decode_record(current)
            self._validate_owner(value, build.execution_id)
            if (
                value.status is not ExecutionReadModelStatus.BUILDING
                or current.lease_owner != owner
                or current.lease_fence != fence
            ):
                raise AIError(ErrorCode.STORAGE_CONFLICT)

            sequence_keys = {
                self._sequence_key(build.execution_id, stream_name): 1
                for stream_name, _data in chunks
            }
            high_waters = (
                await transaction.reserve_sequences(sequence_keys)
                if sequence_keys
                else {}
            )
            facts = tuple(
                StoredFact(
                    self._stream(build.execution_id, stream_name),
                    high_waters[self._sequence_key(build.execution_id, stream_name)],
                    key,
                    f"execution_read_{stream_name}",
                    None,
                    None,
                    data,
                )
                for stream_name, data in chunks
            )
            if complete:
                next_value = ExecutionReadModelRecord(
                    build.execution_id,
                    build.tenant_id,
                    build.source_digest,
                    _MODEL_VERSION,
                    ExecutionReadModelStatus.COMPLETE,
                    len(build.trace_items),
                    len(build.history_items),
                    len(build.transcript_items),
                    value.revision + 1,
                )
                candidate = self._stored_record(
                    next_value,
                    storage_version=current.storage_version + 1,
                    lease_owner=None,
                    lease_fence=fence,
                    expires=None,
                )
            else:
                now = await transaction.now()
                candidate = self._stored_record(
                    value,
                    storage_version=current.storage_version + 1,
                    lease_owner=owner,
                    lease_fence=fence,
                    expires=now,
                )
            if not await transaction.replace_record(
                candidate,
                expected_storage_version=current.storage_version,
            ):
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            if facts:
                await transaction.insert_facts(facts)

        await self._store.mutate(mutate)

    def _building_value(self, execution_id: str) -> ExecutionReadModelRecord:
        return ExecutionReadModelRecord(
            execution_id,
            self._tenant_id,
            "",
            _MODEL_VERSION,
            ExecutionReadModelStatus.BUILDING,
            0,
            0,
            0,
            0,
        )

    def _stored_record(
        self,
        value: ExecutionReadModelRecord,
        *,
        storage_version: int = 0,
        lease_owner: str | None,
        lease_fence: int,
        expires: datetime | None,
    ) -> StoredRecord:
        return StoredRecord(
            self._record_key(value.execution_id),
            partition_digest(
                self._namespace,
                self._tenant_id,
                "execution",
                "execution_read_model",
            ),
            None,
            None,
            "execution_read_model",
            sortable_identity(value.execution_id),
            value.status.value,
            storage_version,
            lease_owner,
            lease_fence,
            None if expires is None else expires + timedelta(seconds=_LEASE_SECONDS),
            {
                "execution_id": value.execution_id,
                "tenant_id": value.tenant_id,
                "source_digest": value.source_digest,
                "model_version": value.model_version,
                "status": value.status.value,
                "trace_count": value.trace_count,
                "history_count": value.history_count,
                "transcript_count": value.transcript_count,
                "revision": value.revision,
            },
        )

    def _record_key(self, execution_id: str) -> bytes:
        return record_key_digest(
            self._namespace,
            self._tenant_id,
            "execution",
            "execution_read_model",
            execution_id,
        )

    def _history_seal_key(self, execution_id: str) -> bytes:
        return record_key_digest(
            self._namespace,
            self._tenant_id,
            "execution",
            "execution_history_seal",
            execution_id,
        )

    def _stream(self, execution_id: str, stream_name: str) -> bytes:
        return stream_digest(
            self._namespace,
            self._tenant_id,
            "execution",
            f"execution_read_{stream_name}",
            execution_id,
        )

    def _sequence_key(self, execution_id: str, stream_name: str) -> bytes:
        return sequence_key(
            self._namespace,
            self._tenant_id,
            "execution",
            f"execution_read_{stream_name}",
            execution_id,
        )

    def _decode_record(self, record: StoredRecord) -> ExecutionReadModelRecord:
        data = record.data
        expected = {
            "execution_id",
            "tenant_id",
            "source_digest",
            "model_version",
            "status",
            "trace_count",
            "history_count",
            "transcript_count",
            "revision",
        }
        if not expected.issubset(data):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        try:
            execution_id = data["execution_id"]
            tenant_id = data["tenant_id"]
            source_digest = data["source_digest"]
            status = data["status"]
            if not all(
                isinstance(value, str)
                for value in (execution_id, tenant_id, source_digest, status)
            ):
                raise TypeError("read model string field has the wrong type")
            counters = (
                data["model_version"],
                data["trace_count"],
                data["history_count"],
                data["transcript_count"],
                data["revision"],
            )
            if any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in counters
            ):
                raise TypeError("read model counter has the wrong type")
            return ExecutionReadModelRecord(
                execution_id,
                tenant_id,
                source_digest,
                counters[0],
                ExecutionReadModelStatus(status),
                counters[1],
                counters[2],
                counters[3],
                counters[4],
            )
        except (KeyError, TypeError, ValueError) as error:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error

    def _stored_model_version(self, record: StoredRecord) -> int:
        value = record.data.get("model_version")
        if isinstance(value, bool) or not isinstance(value, int):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return value

    def _stream_count(self, value: ExecutionReadModelRecord, stream_name: str) -> int:
        try:
            return {
                "trace": value.trace_count,
                "history": value.history_count,
                "transcript": value.transcript_count,
            }[stream_name]
        except KeyError as error:
            raise ValueError("unknown execution read model stream") from error

    def _validate_owner(
        self,
        value: ExecutionReadModelRecord,
        execution_id: str,
    ) -> None:
        if value.execution_id != execution_id or value.tenant_id != self._tenant_id:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

    def _validate_stored_record(
        self,
        record: StoredRecord,
        execution_id: str,
    ) -> None:
        if (
            record.key_digest != self._record_key(execution_id)
            or record.partition_digest
            != partition_digest(
                self._namespace,
                self._tenant_id,
                "execution",
                "execution_read_model",
            )
            or record.kind != "execution_read_model"
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)


__all__ = [
    "ExecutionReadModelBuild",
    "ExecutionReadModelRecord",
    "ExecutionReadModelRepository",
    "ExecutionReadModelStatus",
]
