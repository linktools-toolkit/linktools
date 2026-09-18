#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Memory, artifact, and evaluation repository implementations."""

import json
from dataclasses import replace
from ...core import OperationKind, OperationLedgerInput, OperationLedgerRecord, OperationStatus, Page, ResourceKind
from ...errors import AIError, ErrorCode
from ._contracts import ArtifactRecord, EvaluationRecord, MemoryRecord
from ._plan import RuntimeDomain
from ._store import StateStore, StateTransaction, StoredRecord
from ._repository_common import (
    ResourceRepository as _ResourceRepository,
    insert_operation as _insert_operation,
    projected_record as _projected_record,
    record_cursor as _record_cursor,
    replace_checked as _replace_checked,
    require_tenant as _require_tenant,
    reserve_operation as _reserve_operation,
    validate_page_limit as _validate_page_limit,
)

class EvaluationRepositoryImpl(_ResourceRepository[EvaluationRecord]):
    def __init__(self, store: StateStore, *, namespace: str, tenant_id: str) -> None:
        super().__init__(
            store,
            namespace=namespace,
            tenant_id=tenant_id,
            domain=RuntimeDomain.EVALUATION,
            kind="evaluation",
            resource_kind=ResourceKind.EVALUATION,
            value_type=EvaluationRecord,
        )

    async def list_by_execution(
        self, execution_id: str, *, tenant_id: str
    ) -> tuple[EvaluationRecord, ...]:
        if tenant_id != self._tenant_id:
            return ()
        records = await self._records(
            "evaluation",
            scope=self._scope("evaluation", "execution", execution_id),
        )
        return tuple(
            [await self._decode(record, EvaluationRecord) for record in records]
        )


class MemoryRepositoryImpl(_ResourceRepository[MemoryRecord]):
    def __init__(
        self, store: StateStore, *, namespace: str, tenant_id: str
    ) -> None:
        super().__init__(
            store,
            namespace=namespace,
            tenant_id=tenant_id,
            domain=RuntimeDomain.MEMORY,
            kind="memory",
            resource_kind=ResourceKind.MEMORY,
            value_type=MemoryRecord,
        )

    def _stored(
        self,
        kind: str,
        identity: object,
        value: object,
        *,
        scope: bytes | None = None,
        parent: bytes | None = None,
        state: str | None = None,
    ) -> StoredRecord:
        if kind != "memory" or not isinstance(value, MemoryRecord):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        path = value.metadata.get("path")
        if (
            not isinstance(path, str)
            or not path
            or not path.isascii()
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return replace(
            super()._stored(
                kind,
                identity,
                value,
                scope=scope,
                parent=parent,
                state=state,
            ),
            sort_key=path,
        )

    async def apply_write(
        self,
        record: MemoryRecord,
        *,
        expected_revision: int | None,
        operation: OperationLedgerInput,
    ) -> tuple[MemoryRecord | None, bool]:
        _require_tenant(record, self._tenant_id)
        _require_tenant(operation, self._tenant_id)
        if (
            operation.resource_kind is not ResourceKind.MEMORY
            or operation.resource_id != record.memory_id
            or operation.operation_kind is not OperationKind.MEMORY_WRITE
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

        async def mutate(
            transaction: StateTransaction,
        ) -> tuple[MemoryRecord | None, bool]:
            operation_record, replayed = await _reserve_operation(
                transaction,
                self,
                operation,
            )
            if replayed:
                return None, True
            if operation_record is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            key = self._key("memory", record.memory_id)
            current = await transaction.get_record(key)
            if current is None:
                if expected_revision not in (None, 0):
                    raise AIError(ErrorCode.STORAGE_CONFLICT)
                next_value = replace(record, revision=1)
                await transaction.insert_record(
                    self._stored("memory", record.memory_id, next_value)
                )
            else:
                value = await self._decode(current, MemoryRecord)
                if (
                    isinstance(current.storage_version, bool)
                    or current.storage_version < 0
                    or value.revision < 1
                ):
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                if expected_revision != value.revision:
                    raise AIError(ErrorCode.STORAGE_CONFLICT)
                next_value = replace(record, revision=value.revision + 1)
                await _replace_checked(
                    transaction,
                    _projected_record(self, current, next_value),
                    current.storage_version,
                )
            await _insert_operation(
                transaction,
                self,
                operation,
                operation_record.sequence,
            )
            return next_value, False

        return await self._store.mutate(mutate)

    async def list(
        self,
        *,
        tenant_id: str,
        memory_scope_digest: str,
        prefix: str,
        cursor: str | None,
        limit: int,
    ) -> Page[MemoryRecord]:
        if tenant_id != self._tenant_id:
            return Page(())
        _validate_page_limit(limit)
        records = await self._records(
            "memory",
            scope=self._scope("memory", "memory_scope", memory_scope_digest),
            sort_key_prefix=prefix or None,
            cursor=cursor,
            limit=limit + 1,
        )
        values = tuple(
            [await self._decode(record, MemoryRecord) for record in records[:limit]]
        )
        next_cursor = (
            _record_cursor(records[limit - 1]) if len(records) > limit else None
        )
        return Page(values, next_cursor)

    async def apply_delete(
        self,
        memory_id: str,
        *,
        tenant_id: str,
        expected_revision: int | None,
        operation: OperationLedgerInput,
    ) -> tuple[bool, bool]:
        if tenant_id != self._tenant_id:
            raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)
        _require_tenant(operation, self._tenant_id)
        if (
            operation.resource_kind is not ResourceKind.MEMORY
            or operation.resource_id != memory_id
            or operation.operation_kind is not OperationKind.MEMORY_DELETE
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

        async def mutate(transaction: StateTransaction) -> tuple[bool, bool]:
            operation_record, replayed = await _reserve_operation(
                transaction,
                self,
                operation,
            )
            if replayed:
                return _memory_delete_replay_result(operation_record), True
            if operation_record is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            key = self._key("memory", memory_id)
            current = await transaction.get_record(key)
            if current is None:
                if expected_revision not in (None, 0):
                    raise AIError(ErrorCode.STORAGE_CONFLICT)
                await _insert_operation(
                    transaction,
                    self,
                    operation,
                    operation_record.sequence,
                )
                return False, False
            # A zero revision represents a completed missing-read observation.
            if expected_revision == 0:
                value = await self._decode(current, MemoryRecord)
                if (
                    isinstance(current.storage_version, bool)
                    or current.storage_version < 0
                    or value.revision < 1
                ):
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                await _insert_operation(
                    transaction,
                    self,
                    operation,
                    operation_record.sequence,
                )
                return False, False
            value = await self._decode(current, MemoryRecord)
            if (
                isinstance(current.storage_version, bool)
                or current.storage_version < 0
                or value.revision < 1
            ):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if expected_revision is None or value.revision != expected_revision:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            if not await transaction.delete_record(
                key,
                expected_storage_version=current.storage_version,
            ):
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            await _insert_operation(
                transaction,
                self,
                operation,
                operation_record.sequence,
            )
            return True, False

        return await self._store.mutate(mutate)


class ArtifactRepositoryImpl(_ResourceRepository[ArtifactRecord]):
    def __init__(self, store: StateStore, *, namespace: str, tenant_id: str) -> None:
        super().__init__(
            store,
            namespace=namespace,
            tenant_id=tenant_id,
            domain=RuntimeDomain.ARTIFACT,
            kind="artifact",
            resource_kind=ResourceKind.ARTIFACT,
            value_type=ArtifactRecord,
        )

    async def put_metadata(self, record: ArtifactRecord) -> ArtifactRecord:
        _require_tenant(record, self._tenant_id)

        async def mutate(transaction: StateTransaction) -> ArtifactRecord:
            key = self._key("artifact", record.artifact_id)
            current = await transaction.get_record(key)
            if current is not None:
                existing = await self._decode(current, ArtifactRecord)
                if existing == record:
                    return existing
                raise AIError(ErrorCode.IDEMPOTENCY_CONFLICT)
            await transaction.insert_record(
                self._stored("artifact", record.artifact_id, record)
            )
            return record

        return await self._store.mutate(mutate)

    async def get_metadata(
        self, artifact_id: str, *, tenant_id: str
    ) -> ArtifactRecord | None:
        return await self.get(artifact_id, tenant_id=tenant_id)

    async def list_by_execution(
        self, execution_id: str, *, tenant_id: str, cursor: str | None, limit: int
    ) -> Page[ArtifactRecord]:
        if tenant_id != self._tenant_id:
            return Page(())
        _validate_page_limit(limit)
        records = await self._records(
            "artifact",
            scope=self._scope("artifact", "execution", execution_id),
            cursor=cursor,
            limit=limit + 1,
        )
        values = tuple(
            [await self._decode(record, ArtifactRecord) for record in records[:limit]]
        )
        next_cursor = (
            _record_cursor(records[limit - 1]) if len(records) > limit else None
        )
        return Page(values, next_cursor)


def _memory_delete_replay_result(operation: OperationLedgerRecord | None) -> bool:
    if (
        operation is None
        or operation.status is not OperationStatus.SUCCEEDED
        or operation.result_ref is None
    ):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    try:
        value = json.loads(operation.result_ref)
        if (
            not isinstance(value, dict)
            or not {"version", "result"}.issubset(value)
            or isinstance(value.get("version"), bool)
            or not isinstance(value.get("version"), int)
            or value.get("version") < 1
        ):
            raise ValueError("memory delete receipt is invalid")
        if value["version"] != 1:
            raise AIError(ErrorCode.STORAGE_VERSION_UNSUPPORTED)
        if not isinstance(value.get("result"), dict):
            raise ValueError("memory delete receipt is invalid")
        result = value["result"]
        if (
            not {"file", "version", "status"}.issubset(result)
            or not isinstance(result["file"], str)
            or result["version"] is not None
            or result["status"] not in {"deleted", "not_found"}
        ):
            raise ValueError("memory delete receipt is invalid")
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
    return result["status"] == "deleted"


__all__ = [
    "ArtifactRepositoryImpl",
    "EvaluationRepositoryImpl",
    "MemoryRepositoryImpl",
]
