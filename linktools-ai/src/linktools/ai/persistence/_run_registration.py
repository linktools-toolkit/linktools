#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run registration ownership independent of optional checkpoint capture."""

import json
from dataclasses import dataclass, replace
from datetime import datetime
from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractAsyncContextManager
from typing import TYPE_CHECKING

from pydantic import TypeAdapter
from pydantic_ai_harness.step_persistence import RunRecord

from ..core import canonical_sha256, validate_page_limit
from ..errors import AIError, ErrorCode
from ..steps import validate_native_step_ids
from .codec import decode_envelope, encode_envelope, _encode_step_envelope
from ._official_step_reader import OfficialStepReader
from ._step_contracts import AgentRunRecord
from .state import StateStore, StateTransaction, StoredRecord, record_key, active_state_transaction, StateStorageGroup, RecordScanCursor

if TYPE_CHECKING:
    from ._step_store import SqlStepStore


_RUN = TypeAdapter(RunRecord)


@dataclass(frozen=True, slots=True)
class RunRegistrationCleanupResult:
    deleted: tuple[str, ...] = ()
    missing: tuple[str, ...] = ()
    retained: tuple[str, ...] = ()
    blocked: tuple[str, ...] = ()


class RunRegistrationStore:
    """One immutable registration per run, shared by domain-local references.

    SQL uses the native run table itself. A non-SQL owner uses one canonical
    record in its selected State partition; consumers never copy that record.
    Admitted external metadata is independent from lifecycle commit evidence.
    """

    def __init__(self, store: StateStore, *, namespace: str, tenant_id: str,
                 native: "SqlStepStore | None" = None) -> None:
        if native is not None and native.store_key != native.scope_key(namespace=namespace, tenant_id=tenant_id):
            raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)
        self._store = store
        self._namespace = namespace
        self._tenant_id = tenant_id
        self._native = native

    def key(self, run_id: str) -> bytes:
        validate_native_step_ids(run_id=run_id)
        return record_key(self._namespace, self._tenant_id, "execution", "step_run", (run_id,))

    @property
    def native(self) -> "SqlStepStore | None":
        return self._native

    async def get(self, run_id: str) -> RunRecord | None:
        return (await self.get_many((run_id,))).get(run_id)

    async def get_many(self, run_ids: Sequence[str]) -> Mapping[str, RunRecord]:
        keys = {self.key(run_id): run_id for run_id in run_ids}
        if self._native is not None:
            return await self._native.get_runs(tuple(keys.values()))
        if not keys:
            return {}
        rows = await self._store.read(lambda transaction: transaction.get_records(tuple(keys)))
        result = {}
        for key, stored in rows.items():
            if stored is None:
                continue
            if stored.kind != "step_run":
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            try:
                run = _RUN.validate_python(decode_envelope(stored.data).value)
            except ValueError as error:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
            if run.run_id != keys[key]:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            result[run.run_id] = run
        return result

    async def purge_unreferenced(
        self, candidates: tuple[str, ...], *, exclusive: Callable[[tuple[StateStorageGroup, ...]], AbstractAsyncContextManager[None]],
        complete_reference_stores: tuple[StateStore, ...] | None = None, limit: int = 100,
    ) -> RunRegistrationCleanupResult:
        """Reclaim non-SQL metadata only under explicit complete-source authority.

        The inventory declares every live archive source borrowing this owner.
        Before yielding, the callback must drain and exclude all source users:
        readers, snapshot exporters, and writers from preparation through
        publication. It must hold that exclusion across all supplied groups
        until context exit; stopping only new writes is insufficient.
        A regular snapshot guard, omitted inventory, or SQL owner cannot authorize
        this collection. Domain release alone never proves shared unreachability.
        """
        from ._capability_marker import has_capability_partition_marker

        validate_page_limit(limit)
        selected = tuple(dict.fromkeys(candidates))
        if not selected:
            return RunRegistrationCleanupResult()
        keys = {identity: self.key(identity) for identity in selected}
        if self._native is not None or not complete_reference_stores:
            return RunRegistrationCleanupResult(blocked=selected)
        groups = tuple(dict.fromkeys(store.storage_group for store in complete_reference_stores))
        if (self._store.storage_group not in groups or not callable(exclusive)
                or any(not isinstance(group, StateStorageGroup) for group in groups)):
            return RunRegistrationCleanupResult(blocked=selected)
        async with exclusive(groups):
            discovered = [store for group in groups for store in await group.scan_stores()]
            partitions = tuple(dict.fromkeys((*complete_reference_stores, *discovered)))
            if any([await store.read(has_capability_partition_marker) for store in partitions]):
                return RunRegistrationCleanupResult(blocked=selected)
            live = set()
            for store in partitions:
                cursor = None
                while True:
                    page = await store.read(lambda tx: tx.scan_records_page(after=cursor, limit=1000))
                    for record in page:
                        if record.kind == "agent_run":
                            reference = RunRegistrationReference.from_data(record.data)
                            if reference.run_id in keys:
                                live.add(reference.run_id)
                    if len(page) < 1000:
                        break
                    cursor = RecordScanCursor(page[-1].kind, page[-1].key)

            async def collect(transaction: StateTransaction) -> RunRegistrationCleanupResult:
                rows = await transaction.get_records(tuple(keys.values()))
                deleted, missing, retained, blocked = [], [], [], []
                for identity, key in keys.items():
                    if identity in live:
                        retained.append(identity)
                    elif len(deleted) + len(missing) >= limit:
                        blocked.append(identity)
                    elif key not in rows:
                        missing.append(identity)
                    else:
                        record = rows[key]
                        if record.kind != "step_run":
                            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                        try:
                            run = _RUN.validate_python(decode_envelope(record.data).value)
                        except ValueError as error:
                            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
                        if run.run_id != identity:
                            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                        if not await transaction.delete_record(key, expected_storage_revision=record.storage_revision):
                            raise AIError(ErrorCode.STORAGE_CONFLICT)
                        deleted.append(identity)
                return RunRegistrationCleanupResult(tuple(deleted), tuple(missing), tuple(retained), tuple(blocked))

            return await self._store.mutate(collect)

    def joins(self, store: StateStore) -> bool:
        if self._native is not None:
            from ._sql import SqlStateStore
            return (isinstance(store, SqlStateStore)
                    and store.context.engine is self._native.storage_context.engine)
        return store.storage_group is self._store.storage_group

    def transaction_stores(self, store: StateStore) -> tuple[StateStore, ...]:
        return (self._store,) if self._native is None and self.joins(store) else ()

    async def _write(self, transaction: StateTransaction, run: RunRecord) -> None:
        key = self.key(run.run_id)
        data = encode_envelope(json.loads(_RUN.dump_json(run)))
        current = await transaction.get_record(key)
        if current is not None:
            if current.kind != "step_run" or current.data != data:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            return
        await transaction.insert_record(StoredRecord(
            key=key, scope_key=None, parent_key=None, kind="step_run",
            sort_key=run.started_at.isoformat().encode("utf-8"), state=None,
            storage_revision=0, lease_owner=None, lease_fence=0,
            lease_expires_at=None, data=data,
        ))

    async def get_in_transaction(self, transaction: StateTransaction, run_id: str) -> RunRecord | None:
        """Resolve an unprepared admission inside its already-enlisted owner."""
        key = self.key(run_id)
        if self._native is not None:
            return await self._native.get_run_in_transaction(transaction, run_id=run_id)
        owner = active_state_transaction(self._store, writable=True)
        if owner is None:
            raise AIError(ErrorCode.STORAGE_DEPENDENCY_NOT_READY)
        stored = await owner.get_record(key)
        if stored is None:
            return None
        if stored.kind != "step_run":
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        try:
            run = _RUN.validate_python(decode_envelope(stored.data).value)
        except ValueError as error:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
        if run.run_id != run_id:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return run

    async def publish_in_transaction(self, transaction: StateTransaction, run: RunRecord) -> None:
        """Publish with the domain owner; never confirm provisional native data."""
        if self._native is not None:
            await self._native.import_run_in_transaction(transaction, run)
        else:
            owner = active_state_transaction(self._store, writable=True)
            if owner is None:
                raise AIError(ErrorCode.STORAGE_DEPENDENCY_NOT_READY)
            await self._write(owner, run)

    async def publish(self, run: RunRecord) -> None:
        """Import metadata for a consumer in an independent storage group."""
        validate_native_step_ids(run_id=run.run_id, conversation_id=run.conversation_id,
                                 parent_run_id=run.parent_run_id)
        if self._native is not None:
            await self._native.import_run(run)
        else:
            await self._store.mutate(lambda transaction: self._write(transaction, run))


@dataclass(frozen=True, slots=True)
class RunRegistrationReference:
    """Domain membership retaining only actual differences from its run owner."""

    run_id: str
    registration_id: str | None
    fields: Mapping[str, str | None]
    metadata: Mapping[str, str]
    removed_metadata: tuple[str, ...]
    projection_digest: str

    @classmethod
    def capture(cls, source: RunRecord, projection: AgentRunRecord) -> "RunRegistrationReference":
        if source.run_id != projection.agent_run_id:
            raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)
        original = OfficialStepReader.project_run(source)
        changes = {
            name: getattr(projection, name)
            for name in ("agent_conversation_id", "parent_agent_run_id", "agent_id")
            if getattr(original, name) != getattr(projection, name)
        }
        if original.started_at.isoformat() != projection.started_at.isoformat():
            changes["started_at"] = projection.started_at.isoformat()
        return cls(source.run_id, source.registration_id, changes,
                   {key: value for key, value in projection.metadata.items()
                    if original.metadata.get(key) != value},
                   tuple(sorted(original.metadata.keys() - projection.metadata.keys())),
                   canonical_sha256(_encode_step_envelope(projection)))

    def to_data(self) -> dict[str, object]:
        return encode_envelope({"type": "run_reference", "payload": {
            "run_id": self.run_id, "registration_id": self.registration_id,
            "fields": dict(self.fields), "metadata": dict(self.metadata),
            "removed_metadata": list(self.removed_metadata),
            "projection_digest": self.projection_digest,
        }})

    @classmethod
    def from_data(cls, data: Mapping[str, object]) -> "RunRegistrationReference":
        value = decode_envelope(data).value
        payload = value.get("payload")
        if (set(value) != {"type", "payload"} or value["type"] != "run_reference"
                or not isinstance(payload, Mapping)
                or set(payload) != {"run_id", "registration_id", "fields", "metadata", "removed_metadata", "projection_digest"}):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        validate_native_step_ids(run_id=payload["run_id"])
        registration, fields, metadata, removed = (payload[name] for name in
            ("registration_id", "fields", "metadata", "removed_metadata"))
        if (registration is not None and not isinstance(registration, str)
                or not isinstance(fields, Mapping)
                or set(fields) - {"agent_conversation_id", "parent_agent_run_id", "agent_id", "started_at"}
                or any(value is not None and not isinstance(value, str) for value in fields.values())
                or not isinstance(metadata, Mapping)
                or any(not isinstance(key, str) or not isinstance(value, str) for key, value in metadata.items())
                or not isinstance(removed, (list, tuple))
                or any(not isinstance(key, str) for key in removed)
                or list(removed) != sorted(set(removed)) or set(removed).intersection(metadata)):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        digest = payload["projection_digest"]
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return cls(payload["run_id"], registration, dict(fields), dict(metadata), tuple(removed), digest)

    def project(self, source: RunRecord) -> AgentRunRecord:
        if (source.run_id != self.run_id or source.registration_id != self.registration_id):
            raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)
        original = OfficialStepReader.project_run(source)
        changes = dict(self.fields)
        if set(changes) - {"agent_conversation_id", "parent_agent_run_id", "agent_id", "started_at"}:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if "started_at" in changes:
            try:
                timestamp = datetime.fromisoformat(changes["started_at"])
                if timestamp.tzinfo is None:
                    raise ValueError("run timestamp requires a timezone")
                changes["started_at"] = timestamp
            except (TypeError, ValueError) as error:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
        metadata = {key: value for key, value in original.metadata.items() if key not in self.removed_metadata}
        metadata.update(self.metadata)
        projection = replace(original, metadata=metadata, **changes)
        # Commit to the consumed projection, not source-only or overridden data.
        if canonical_sha256(_encode_step_envelope(projection)) != self.projection_digest:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return projection


__all__ = ["RunRegistrationStore", "RunRegistrationReference", "RunRegistrationCleanupResult"]
