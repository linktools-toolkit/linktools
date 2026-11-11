#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Persistent memory operations and captured-input restoration."""

from __future__ import annotations

import json
import re
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone

from linktools.core import environ
from pydantic_ai_harness.memory import (
    MemoryConflictError,
    MemoryFile,
    MemoryMutation,
    MemoryOperation,
    MemoryOperationConflictError,
)

from ..core import (
    JsonValue,
    OperationKind,
    OperationLedgerInput,
    OperationStatus,
    ResourceKind,
    canonical_sha256,
    validate_tenant_id,
    validate_resource_id,
)
from ..errors import AIError, ErrorCode
from ..storage import ObjectStore, StoredPayload
from ..persistence.objects import read_runtime_object
from ..persistence.contracts import MemoryRecord, MemoryRepositories

_logger = environ.get_logger("ai.memory")
_MAX_CONTENT_CHARS = 65_536
_MEMORY_VERSION = re.compile(r"m1:[0-9a-f]{64}")
_STORE_SEGMENT = re.compile(r"[A-Za-z0-9_.-]{1,200}")


@dataclass(frozen=True, slots=True)
class _MutationReceipt:
    path: str
    version: str | None
    existed: bool
    kind: str

    def encode(self) -> str:
        status = (
            "deleted"
            if self.kind == "delete" and self.existed
            else "not_found"
            if self.kind == "delete"
            else "updated"
            if self.existed
            else "created"
        )
        return json.dumps(
            {
                "version": 1,
                "result": {
                    "file": self.path,
                    "version": self.version,
                    "status": status,
                },
            },
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )

    @classmethod
    def decode(cls, value: str) -> _MutationReceipt:
        try:
            raw = json.loads(value)
            if (
                not isinstance(raw, dict)
                or isinstance(raw.get("version"), bool)
                or not isinstance(raw.get("version"), int)
                or raw.get("version") != 1
                or not {"version", "result"}.issubset(raw)
            ):
                raise ValueError("memory receipt is malformed")
            result = raw["result"]
            if not isinstance(result, dict) or not {
                "file",
                "version",
                "status",
            }.issubset(result):
                raise ValueError("memory receipt result is malformed")
            path = _normalize_path(result["file"])
            version = result["version"]
            status = result["status"]
            if status not in {"created", "updated", "deleted", "not_found"}:
                raise ValueError("memory receipt status is invalid")
            if status in {"created", "updated"}:
                if (
                    not isinstance(version, str)
                    or _MEMORY_VERSION.fullmatch(version) is None
                ):
                    raise ValueError("memory receipt version is invalid")
                kind = "write"
            else:
                if version is not None:
                    raise ValueError("delete receipt cannot carry a version")
                kind = "delete"
            return cls(
                path=path,
                version=version,
                existed=status not in {"created", "not_found"},
                kind=kind,
            )
        except AIError:
            raise
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error

    def require_target(self, path: str, kind: str) -> None:
        if self.path != path or self.kind != kind:
            raise MemoryOperationConflictError(
                "memory operation targets a different mutation"
            )

    def mutation(self, *, replayed: bool) -> MemoryMutation:
        return MemoryMutation(self.version, replayed, self.existed)


class PersistentMemoryStore:
    """Borrow persistence resources for one admitted memory scope."""

    def __init__(
        self,
        state: MemoryRepositories,
        *,
        object_store: ObjectStore,
        namespace: str,
        tenant_id: str,
        execution_id: str,
        memory_scope: str,
        transient: bool = False,
    ) -> None:
        try:
            validate_tenant_id(tenant_id)
            if transient:
                validate_resource_id(execution_id)
            MemoryRecord.scope_components(memory_scope, execution_id if transient else None)
        except AIError as error:
            raise ValueError("memory store identity is invalid") from error
        self._state = state
        self._object_store = object_store
        self._namespace = namespace
        self._tenant_id = tenant_id
        self._memory_scope = memory_scope
        self._transient_execution_id = execution_id if transient else None

    async def capture(self) -> Mapping[str, JsonValue]:
        """Read a consistent scope view, preserving each file's version."""
        records = await self._state.records.capture_scope(
            tenant_id=self._tenant_id, memory_scope=self._memory_scope,
            transient_execution_id=self._transient_execution_id,
        )
        result: dict[str, JsonValue] = {}
        for record in records:
            path = record.metadata["path"]
            _validate_record(record, path, self._memory_scope, self._transient_execution_id)
            result[path] = {"content": await self._content(record), "version": _record_version(record)}
        return result

    async def seed_capture(self, snapshot: Mapping[str, JsonValue]) -> None:
        """Seed captured versions once per path without reverting later mutations.

        Each file and its receipt commit together. A partially completed seed
        can be retried with the same capture after cancellation or an unknown
        outcome; the original receipts remain authoritative.
        """
        for path, value in snapshot.items():
            _normalize_path(path)
            if not isinstance(value, Mapping):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            content, version = value["content"], value["version"]
            _validate_content(content)
            if not isinstance(version, str) or _MEMORY_VERSION.fullmatch(version) is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            operation = MemoryOperation(id="capture:" + path, fingerprint=canonical_sha256(value))
            receipt = await self._get_receipt(operation, captured_version=version)
            if receipt is not None:
                receipt.require_target(path, "write")
                continue
            now = datetime.now(timezone.utc)
            memory_id = _memory_id(self._memory_scope, path, self._transient_execution_id)
            record = MemoryRecord(
                memory_id, self._memory_scope, StoredPayload.inline_text(content),
                {"path": path, "version": version, "operation_id": _operation_id(
                    self._memory_scope, operation.id, self._transient_execution_id,
                )},
                0, now, now, transient_execution_id=self._transient_execution_id,
            )
            await self._state.records.apply_write(
                record, expected_revision=None,
                operation=_operation_input(
                    operation, self._memory_scope, self._tenant_id,
                    _MutationReceipt(path, version, False, "write"), OperationKind.MEMORY_WRITE, memory_id,
                    transient_execution_id=self._transient_execution_id,
                ),
            )

    async def read(self, path: str, *, max_chars: int) -> MemoryFile | None:
        logical_path = _normalize_path(path)
        if (
            not isinstance(max_chars, int)
            or isinstance(max_chars, bool)
            or max_chars <= 0
        ):
            raise ValueError("max_chars must be positive")
        record = await self._record(logical_path)
        if record is None:
            return None
        content = await self._content(record)
        version = _record_version(record)
        if version is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        operation_id = record.metadata.get("operation_id")
        return MemoryFile(
            content=content[:max_chars],
            version=version,
            operation_id=operation_id if isinstance(operation_id, str) else None,
            truncated=len(content) > max_chars,
        )

    async def get_operation(self, operation: MemoryOperation) -> MemoryMutation | None:
        receipt = await self._get_receipt(operation)
        if receipt is None:
            return None
        return receipt.mutation(replayed=True)

    async def write(
        self,
        path: str,
        content: str,
        *,
        expected_version: str | None,
        operation: MemoryOperation | None = None,
    ) -> MemoryMutation:
        logical_path = _normalize_path(path)
        _validate_content(content)
        _validate_expected_version(expected_version)
        operation = _ensure_operation(operation, "write", logical_path, content)
        replay = await self._get_receipt(operation)
        if replay is not None:
            replay.require_target(logical_path, "write")
            return replay.mutation(replayed=True)

        current = await self._record(logical_path)
        _check_version(current, expected_version)
        version = _version_token(
            self._namespace,
            self._tenant_id,
            self._memory_scope,
            logical_path,
            operation.id,
            transient_execution_id=self._transient_execution_id,
        )
        stored_content = StoredPayload.inline_text(content)
        now = datetime.now(timezone.utc)
        next_record = MemoryRecord(
            _memory_id(self._memory_scope, logical_path, self._transient_execution_id),
            self._memory_scope,
            stored_content,
            {
                "path": logical_path,
                "version": version,
                "operation_id": _operation_id(
                    self._memory_scope,
                    operation.id,
                    self._transient_execution_id,
                ),
            },
            0,
            now if current is None else current.created_at,
            now,
            transient_execution_id=self._transient_execution_id,
        )
        receipt = _MutationReceipt(
            logical_path,
            version,
            current is not None,
            "write",
        )
        try:
            stored, replayed = await self._state.records.apply_write(
                next_record,
                expected_revision=None if current is None else current.revision,
                operation=_operation_input(
                    operation,
                    self._memory_scope,
                    self._tenant_id,
                    receipt,
                    OperationKind.MEMORY_WRITE,
                    next_record.memory_id,
                    transient_execution_id=self._transient_execution_id,
                ),
            )
        except AIError as error:
            raise await self._map_mutation_error(error, operation) from error
        if replayed:
            replay = await self._get_receipt(operation)
            if replay is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            replay.require_target(logical_path, "write")
            return replay.mutation(replayed=True)
        if stored is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        _logger.debug(
            "memory write committed: path=%s",
            logical_path,
        )
        return receipt.mutation(replayed=False)

    async def delete(
        self,
        path: str,
        *,
        expected_version: str | None,
        operation: MemoryOperation | None = None,
    ) -> MemoryMutation:
        logical_path = _normalize_path(path)
        _validate_expected_version(expected_version)
        operation = _ensure_operation(operation, "delete", logical_path, None)
        replay = await self._get_receipt(operation)
        if replay is not None:
            replay.require_target(logical_path, "delete")
            return replay.mutation(replayed=True)

        current = await self._record(logical_path)
        _check_version(current, expected_version)
        receipt = _MutationReceipt(
            logical_path,
            None,
            current is not None,
            "delete",
        )
        try:
            deleted, replayed = await self._state.records.apply_delete(
                _memory_id(self._memory_scope, logical_path, self._transient_execution_id),
                tenant_id=self._tenant_id,
                expected_revision=0 if current is None else current.revision,
                operation=_operation_input(
                    operation,
                    self._memory_scope,
                    self._tenant_id,
                    receipt,
                    OperationKind.MEMORY_DELETE,
                    _memory_id(self._memory_scope, logical_path, self._transient_execution_id),
                    transient_execution_id=self._transient_execution_id,
                ),
            )
        except AIError as error:
            raise await self._map_mutation_error(error, operation) from error
        if replayed:
            replay = await self._get_receipt(operation)
            if replay is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            replay.require_target(logical_path, "delete")
            return replay.mutation(replayed=True)
        if deleted != (current is not None):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        _logger.debug(
            "memory delete committed: path=%s",
            logical_path,
        )
        return receipt.mutation(replayed=False)

    async def list_paths(self, prefix: str = "", *, limit: int) -> list[str]:
        normalized_prefix = _normalize_prefix(prefix)
        if not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0:
            raise ValueError("limit must be positive")
        paths: list[str] = []
        cursor: str | None = None
        while len(paths) < limit:
            page = await self._state.records.list(
                tenant_id=self._tenant_id,
                memory_scope=self._memory_scope,
                transient_execution_id=self._transient_execution_id,
                prefix=normalized_prefix,
                cursor=cursor,
                limit=min(1000, limit - len(paths)),
            )
            batch = _record_paths(list(page.items))
            if any(not path.startswith(normalized_prefix) for path in batch):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            paths.extend(batch)
            if page.next_cursor is None:
                break
            cursor = page.next_cursor
        return paths

    async def _get_receipt(
        self,
        operation: MemoryOperation,
        *,
        captured_version: str | None = None,
    ) -> _MutationReceipt | None:
        _validate_operation(operation)
        operation_id = _operation_id(self._memory_scope, operation.id, self._transient_execution_id)
        record = await self._state.operations.get(
            operation_id,
            tenant_id=self._tenant_id,
        )
        if record is None:
            return None
        if record.request_digest != operation.fingerprint:
            raise MemoryOperationConflictError(
                f"operation id {operation.id!r} was reused with different arguments"
            )
        if (
            record.operation_id != operation_id
            or record.resource_kind is not ResourceKind.MEMORY
            or record.status is not OperationStatus.SUCCEEDED
            or record.result_ref is None
            or record.execution_id is not None
            or record.result_digest is not None
            or record.error_code is not None
            or not record.compactable
            or isinstance(record.sequence, bool)
            or not isinstance(record.sequence, int)
            or record.sequence < 1
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        receipt = _MutationReceipt.decode(record.result_ref)
        expected_kind = (
            OperationKind.MEMORY_WRITE
            if receipt.kind == "write"
            else OperationKind.MEMORY_DELETE
        )
        if (
            record.resource_id != _memory_id(self._memory_scope, receipt.path, self._transient_execution_id)
            or record.operation_kind is not expected_kind
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if receipt.version is not None and receipt.version != (captured_version or _version_token(
            self._namespace,
            self._tenant_id,
            self._memory_scope,
            receipt.path,
            operation.id,
            transient_execution_id=self._transient_execution_id,
        )):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return receipt

    async def _map_mutation_error(
        self,
        error: AIError,
        operation: MemoryOperation,
    ) -> Exception:
        if error.code is not ErrorCode.STORAGE_CONFLICT:
            return error
        existing = await self._state.operations.get(
            _operation_id(self._memory_scope, operation.id, self._transient_execution_id),
            tenant_id=self._tenant_id,
        )
        if existing is not None:
            return MemoryOperationConflictError(
                f"operation id {operation.id!r} conflicted with an existing mutation"
            )
        return MemoryConflictError("memory version conflict")

    async def _record(self, logical_path: str) -> MemoryRecord | None:
        record = await self._state.records.get(
            _memory_id(self._memory_scope, logical_path, self._transient_execution_id),
            tenant_id=self._tenant_id,
        )
        if record is not None:
            _validate_record(
                record,
                logical_path,
                self._memory_scope,
                self._transient_execution_id,
            )
        return record

    async def _content(self, record: MemoryRecord) -> str:
        try:
            if record.content.kind == "inline":
                content = record.content.decode()
                if not isinstance(content, str):
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            elif record.content.ref is not None:
                content = (
                    await read_runtime_object(
                        self._object_store,
                        record.content.ref,
                    )
                ).decode("utf-8")
            else:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        except AIError:
            raise
        except (UnicodeDecodeError, TypeError, ValueError, OSError) as error:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
        if len(content) > _MAX_CONTENT_CHARS or "\x00" in content:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return content


def _record_paths(records: list[MemoryRecord]) -> list[str]:
    paths: list[str] = []
    for record in records:
        path = record.metadata.get("path")
        if not isinstance(path, str):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        normalized = _normalize_path(path)
        if normalized != path:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        paths.append(normalized)
    if paths != sorted(paths):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return paths


def _normalize_path(path: str) -> str:
    if (
        not isinstance(path, str)
        or not path
        or path.startswith("/")
        or "\\" in path
        or "\x00" in path
    ):
        raise ValueError("memory path is invalid")
    parts = path.split("/")
    if any(not _STORE_SEGMENT.fullmatch(part) or ".." in part for part in parts):
        raise ValueError("memory path is invalid")
    return "/".join(parts)


def _normalize_prefix(prefix: str) -> str:
    if prefix == "":
        return ""
    if not isinstance(prefix, str):
        raise ValueError("memory prefix is invalid")
    _normalize_path(prefix.removesuffix("/"))
    return prefix


def _validate_content(content: str) -> None:
    if (
        not isinstance(content, str)
        or len(content) > _MAX_CONTENT_CHARS
        or "\x00" in content
    ):
        raise ValueError("memory content is invalid")
    try:
        content.encode("utf-8", errors="strict")
    except UnicodeEncodeError as error:
        raise ValueError("memory content is invalid") from error


def _validate_operation(operation: MemoryOperation) -> None:
    if (
        not isinstance(operation, MemoryOperation)
        or not isinstance(operation.id, str)
        or not operation.id
        or not isinstance(operation.fingerprint, str)
        or not re.fullmatch(r"[0-9a-f]{64}", operation.fingerprint)
    ):
        raise ValueError("memory operation is invalid")


def _ensure_operation(
    operation: MemoryOperation | None,
    kind: str,
    path: str,
    content: str | None,
) -> MemoryOperation:
    if operation is not None:
        _validate_operation(operation)
        return operation
    return MemoryOperation(
        id=uuid.uuid4().hex,
        fingerprint=canonical_sha256({"kind": kind, "path": path, "content": content}),
    )


def _record_version(record: MemoryRecord) -> str | None:
    value = record.metadata.get("version")
    return (
        value if isinstance(value, str) and _MEMORY_VERSION.fullmatch(value) else None
    )



def _validate_record(
    record: MemoryRecord,
    logical_path: str,
    memory_scope: str,
    transient_execution_id: str | None = None,
) -> None:
    if (
        record.memory_id != _memory_id(memory_scope, logical_path, transient_execution_id)
        or record.memory_scope != memory_scope
        or record.transient_execution_id != transient_execution_id
        or not isinstance(record.revision, int)
        or isinstance(record.revision, bool)
        or record.revision < 1
        or _record_version(record) is None
        or "storage_revision" in record.metadata
        or record.metadata.get("path") != logical_path
        or not isinstance(record.metadata.get("operation_id"), str)
    ):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)


def _version_token(
    namespace: str,
    tenant_id: str,
    memory_scope: str,
    path: str,
    identity: str,
    *,
    transient_execution_id: str | None = None,
) -> str:
    return "m1:" + canonical_sha256(
        {
            "namespace": namespace,
            "tenant_id": tenant_id,
            "scope": memory_scope,
            "transient_execution_id": transient_execution_id,
            "path": path,
            "identity": identity,
        }
    )


def _memory_id(
    memory_scope: str, logical_path: str, transient_execution_id: str | None = None,
) -> str:
    # Full-length paths and scopes do not fit the bounded record address together.
    return canonical_sha256({
        "memory_scope": memory_scope,
        "transient_execution_id": transient_execution_id,
        "path": logical_path,
    })


def _operation_id(
    memory_scope: str, operation_id: str, transient_execution_id: str | None = None,
) -> str:
    # Capture operation IDs include the full path, beyond ordinary caller ID bounds.
    return canonical_sha256({
        "memory_scope": memory_scope,
        "transient_execution_id": transient_execution_id,
        "operation_id": operation_id,
    })


def _operation_input(
    operation: MemoryOperation,
    memory_scope: str,
    tenant_id: str,
    receipt: _MutationReceipt,
    kind: OperationKind,
    resource_id: str,
    *,
    transient_execution_id: str | None = None,
) -> OperationLedgerInput:
    now = datetime.now(timezone.utc)
    return OperationLedgerInput(
        _operation_id(memory_scope, operation.id, transient_execution_id),
        tenant_id,
        ResourceKind.MEMORY,
        resource_id,
        None,
        kind,
        OperationStatus.SUCCEEDED,
        operation.fingerprint,
        receipt.encode(),
        None,
        None,
        True,
        now,
        now,
    )


def _check_version(record: MemoryRecord | None, expected: str | None) -> None:
    if record is None:
        if expected is not None:
            raise MemoryConflictError("memory version conflict")
        return
    actual = _record_version(record)
    if actual is None:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    if expected != actual:
        raise MemoryConflictError("memory version conflict")


def _validate_expected_version(value: str | None) -> None:
    if value is not None and (
        not isinstance(value, str) or _MEMORY_VERSION.fullmatch(value) is None
    ):
        raise ValueError("memory version is invalid")


__all__ = [
    "PersistentMemoryStore",
]
