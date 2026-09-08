#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime persistence adapter for Harness memory."""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

from linktools.core import environ
from pydantic_ai_harness.memory import (
    MemoryConflictError,
    MemoryFile,
    MemoryMutation,
    MemoryOperation,
    MemoryOperationConflictError,
    MemorySearchMatch,
    MemorySearchResult,
    MemoryStore,
)

from ..core import (
    OperationKind,
    OperationLedgerInput,
    OperationStatus,
    ResourceKind,
    canonical_sha256,
    validate_memory_scope,
    validate_tenant_id,
)
from ..errors import AIError, ErrorCode
from ..storage import ObjectStore, PayloadPolicy, StoredPayload, payload_fits_inline
from ._object import RuntimeObjectKeyFactory, put_runtime_object, read_runtime_object
from .state import MemoryRecord, MemoryState, RuntimeDomain

_logger = environ.get_logger("ai.runtime.memory")
_MAX_CONTENT_CHARS = 65_536
_MEMORY_VERSION = re.compile(r"m2:[0-9a-f]{64}")
_STORE_SEGMENT = re.compile(r"[A-Za-z0-9_.-]{1,200}")


@dataclass(frozen=True, slots=True)
class _MutationReceipt:
    path: str
    version: str | None
    existed: bool
    kind: str


class RuntimeMemoryStore:
    """Persist Harness memory files inside one Runtime memory scope."""

    def __init__(
        self,
        state: MemoryState,
        *,
        object_store: ObjectStore,
        namespace: str,
        tenant_id: str,
        execution_id: str,
        memory_scope: str,
        transient: bool = False,
        payload_policy: PayloadPolicy | None = None,
    ) -> None:
        try:
            validate_tenant_id(tenant_id)
            validate_memory_scope(memory_scope)
        except AIError as error:
            raise ValueError("memory store identity is invalid") from error
        self._state = state
        self._object_store = object_store
        self._namespace = namespace
        self._tenant_id = tenant_id
        self._execution_id = execution_id
        self._payload_policy = payload_policy or PayloadPolicy()
        scope_digest = canonical_sha256(memory_scope)
        self._memory_scope_digest = (
            canonical_sha256(
                {"execution_id": execution_id, "memory_scope_digest": scope_digest}
            )
            if transient
            else scope_digest
        )

    async def read(self, path: str, *, max_chars: int) -> MemoryFile | None:
        logical_path = _normalize_path(path)
        if not isinstance(max_chars, int) or isinstance(max_chars, bool) or max_chars <= 0:
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
        return MemoryMutation(
            version=receipt.version,
            replayed=True,
            existed=receipt.existed,
        )

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
            _require_receipt(replay, logical_path, "write")
            return MemoryMutation(replay.version, True, replay.existed)

        current = await self._record(logical_path)
        _check_version(current, expected_version)
        expected_storage_version = (
            None if current is None else _record_storage_version(current)
        )
        version = _version_token(
            self._namespace,
            self._tenant_id,
            self._memory_scope_digest,
            logical_path,
            operation.id,
        )
        stored_content = StoredPayload.inline_text(content)
        if not payload_fits_inline(stored_content, self._payload_policy):
            reference = await put_runtime_object(
                self._object_store,
                RuntimeObjectKeyFactory(self._namespace),
                RuntimeDomain.MEMORY,
                self._tenant_id,
                content.encode("utf-8"),
            )
            stored_content = StoredPayload.object(reference)
        now = datetime.now(timezone.utc)
        next_record = MemoryRecord(
            _memory_id(self._memory_scope_digest, logical_path),
            self._tenant_id,
            self._memory_scope_digest,
            stored_content,
            {
                "path": logical_path,
                "version": version,
                "operation_id": _operation_id(
                    self._memory_scope_digest,
                    operation.id,
                ),
            },
            0,
            now if current is None else current.created_at,
            now,
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
                expected_storage_version=expected_storage_version,
                operation=_operation_input(
                    operation,
                    self._memory_scope_digest,
                    self._tenant_id,
                    receipt,
                    OperationKind.MEMORY_WRITE,
                    next_record.memory_id,
                ),
            )
        except AIError as error:
            raise await self._map_mutation_error(error, operation) from error
        if replayed:
            replay = await self._get_receipt(operation)
            if replay is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            _require_receipt(replay, logical_path, "write")
            return MemoryMutation(replay.version, True, replay.existed)
        if stored is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        _logger.debug(
            "memory write committed: scope=%s path=%s",
            self._memory_scope_digest,
            logical_path,
        )
        return MemoryMutation(version, False, current is not None)

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
            _require_receipt(replay, logical_path, "delete")
            return MemoryMutation(replay.version, True, replay.existed)

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
                _memory_id(self._memory_scope_digest, logical_path),
                tenant_id=self._tenant_id,
                expected_revision=0 if current is None else current.revision,
                expected_storage_version=(
                    None if current is None else _record_storage_version(current)
                ),
                operation=_operation_input(
                    operation,
                    self._memory_scope_digest,
                    self._tenant_id,
                    receipt,
                    OperationKind.MEMORY_DELETE,
                    _memory_id(self._memory_scope_digest, logical_path),
                ),
            )
        except AIError as error:
            raise await self._map_mutation_error(error, operation) from error
        if replayed:
            replay = await self._get_receipt(operation)
            if replay is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            _require_receipt(replay, logical_path, "delete")
            return MemoryMutation(replay.version, True, replay.existed)
        if deleted != (current is not None):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        _logger.debug(
            "memory delete committed: scope=%s path=%s",
            self._memory_scope_digest,
            logical_path,
        )
        return MemoryMutation(None, False, current is not None)

    async def list_paths(self, prefix: str = "", *, limit: int) -> list[str]:
        normalized_prefix = _normalize_prefix(prefix)
        if not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0:
            raise ValueError("limit must be positive")
        records, _ = await self._list_records(limit=None)
        paths: list[str] = []
        for record in records:
            path = record.metadata.get("path")
            if not isinstance(path, str):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            normalized = _normalize_path(path)
            if normalized != path:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if normalized.startswith(normalized_prefix):
                paths.append(normalized)
        return sorted(paths)[:limit]

    async def _get_receipt(
        self,
        operation: MemoryOperation,
    ) -> _MutationReceipt | None:
        _validate_operation(operation)
        operation_id = _operation_id(self._memory_scope_digest, operation.id)
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
            or record.tenant_id != self._tenant_id
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
        receipt = _decode_receipt(record.result_ref)
        expected_kind = (
            OperationKind.MEMORY_WRITE
            if receipt.kind == "write"
            else OperationKind.MEMORY_DELETE
        )
        if (
            record.resource_id
            != _memory_id(self._memory_scope_digest, receipt.path)
            or record.operation_kind is not expected_kind
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if receipt.version is not None and receipt.version != _version_token(
            self._namespace,
            self._tenant_id,
            self._memory_scope_digest,
            receipt.path,
            operation.id,
        ):
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
            _operation_id(self._memory_scope_digest, operation.id),
            tenant_id=self._tenant_id,
        )
        if existing is not None:
            return MemoryOperationConflictError(
                f"operation id {operation.id!r} conflicted with an existing mutation"
            )
        return MemoryConflictError("memory version conflict")

    async def _list_records(
        self,
        *,
        limit: int | None,
    ) -> tuple[list[MemoryRecord], bool]:
        records: list[MemoryRecord] = []
        cursor: str | None = None
        has_more = False
        while limit is None or len(records) < limit:
            page_limit = 200 if limit is None else min(200, limit - len(records))
            page = await self._state.records.list(
                tenant_id=self._tenant_id,
                memory_scope_digest=self._memory_scope_digest,
                cursor=cursor,
                limit=page_limit,
            )
            records.extend(page.items)
            if page.next_cursor is None:
                break
            if page.next_cursor == cursor or not page.items:
                has_more = True
                break
            if limit is not None and len(records) >= limit:
                has_more = True
                break
            cursor = page.next_cursor
        return records if limit is None else records[:limit], has_more

    async def _record(self, logical_path: str) -> MemoryRecord | None:
        record = await self._state.records.get(
            _memory_id(self._memory_scope_digest, logical_path),
            tenant_id=self._tenant_id,
        )
        if record is not None:
            _validate_record(
                record,
                logical_path,
                self._memory_scope_digest,
                self._tenant_id,
            )
        return record

    async def _content(self, record: MemoryRecord) -> str:
        try:
            if record.content.kind == "inline":
                content = record.content.decode()
                if not isinstance(content, str):
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            elif record.content.ref is not None:
                content = (await read_runtime_object(
                    self._object_store,
                    record.content.ref,
                )).decode("utf-8")
            else:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        except AIError:
            raise
        except (UnicodeDecodeError, TypeError, ValueError, OSError) as error:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
        if len(content) > _MAX_CONTENT_CHARS or "\x00" in content:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return content


def _normalize_path(path: str) -> str:
    if not isinstance(path, str) or not path or path.startswith("/") or "\\" in path or "\x00" in path:
        raise ValueError("memory path is invalid")
    parts = path.split("/")
    if any(
        not _STORE_SEGMENT.fullmatch(part) or ".." in part
        for part in parts
    ):
        raise ValueError("memory path is invalid")
    return "/".join(parts)


def _normalize_prefix(prefix: str) -> str:
    if prefix == "":
        return ""
    normalized = prefix.removesuffix("/")
    return f"{_normalize_path(normalized)}/"


def normalize_memory_file(file: str) -> str:
    """Normalize a model-facing memory filename for legacy internal callers."""
    value = file.strip() if isinstance(file, str) else ""
    if value and not value.endswith(".md"):
        value = f"{value}.md"
    if not value or "/" in value or "\\" in value:
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
    try:
        return _normalize_path(value)
    except ValueError as error:
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID) from error


def _validate_content(content: str) -> None:
    if not isinstance(content, str) or len(content) > _MAX_CONTENT_CHARS or "\x00" in content:
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
        fingerprint=canonical_sha256(
            {"kind": kind, "path": path, "content": content}
        ),
    )


def memory_operation_fingerprint(
    action: str,
    file: str,
    content: str | None,
    old_text: str | None,
    append: bool,
) -> str:
    """Retain the prior fingerprint helper for Runtime-private compatibility."""
    return canonical_sha256(
        {
            "action": action,
            "file": file,
            "content": content,
            "old_text": old_text,
            "append": append,
        }
    )


def _record_version(record: MemoryRecord) -> str | None:
    value = record.metadata.get("version")
    return value if isinstance(value, str) and _MEMORY_VERSION.fullmatch(value) else None


def _record_storage_version(record: MemoryRecord) -> int:
    value = record.metadata.get("storage_version")
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return value


def _validate_record(
    record: MemoryRecord,
    logical_path: str,
    scope_digest: str,
    tenant_id: str,
) -> None:
    if (
        record.tenant_id != tenant_id
        or record.memory_id != _memory_id(scope_digest, logical_path)
        or record.memory_scope_digest != scope_digest
        or not isinstance(record.revision, int)
        or isinstance(record.revision, bool)
        or record.revision < 1
        or _record_version(record) is None
        or _record_storage_version(record) != record.revision
        or record.metadata.get("path") != logical_path
        or not isinstance(record.metadata.get("operation_id"), str)
    ):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)


def _version_token(
    namespace: str,
    tenant_id: str,
    scope_digest: str,
    path: str,
    identity: str,
) -> str:
    return "m2:" + canonical_sha256(
        {
            "namespace": namespace,
            "tenant_id": tenant_id,
            "scope": scope_digest,
            "path": path,
            "identity": canonical_sha256(identity),
        }
    )


def _memory_id(scope_digest: str, logical_path: str) -> str:
    return hashlib.sha256(f"{scope_digest}\0{logical_path}".encode()).hexdigest()


def _operation_id(scope_digest: str, operation_id: str) -> str:
    return hashlib.sha256(f"{scope_digest}\0{operation_id}".encode()).hexdigest()


def _operation_input(
    operation: MemoryOperation,
    scope_digest: str,
    tenant_id: str,
    receipt: _MutationReceipt,
    kind: OperationKind,
    resource_id: str,
) -> OperationLedgerInput:
    now = datetime.now(timezone.utc)
    return OperationLedgerInput(
        _operation_id(scope_digest, operation.id),
        tenant_id,
        ResourceKind.MEMORY,
        resource_id,
        None,
        kind,
        OperationStatus.SUCCEEDED,
        operation.fingerprint,
        _encode_receipt(receipt),
        None,
        None,
        True,
        now,
        now,
    )


def _encode_receipt(receipt: _MutationReceipt) -> str:
    status = (
        "deleted"
        if receipt.kind == "delete" and receipt.existed
        else "not_found"
        if receipt.kind == "delete"
        else "updated"
        if receipt.existed
        else "created"
    )
    return json.dumps(
        {
            "version": 2,
            "result": {
                "file": receipt.path,
                "version": receipt.version,
                "status": status,
            },
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _decode_receipt(value: str) -> _MutationReceipt:
    try:
        raw = json.loads(value)
        if not isinstance(raw, dict) or raw.get("version") != 2 or set(raw) != {"version", "result"}:
            raise ValueError("memory receipt is malformed")
        result = raw["result"]
        if not isinstance(result, dict) or set(result) != {"file", "version", "status"}:
            raise ValueError("memory receipt result is malformed")
        path = _normalize_path(result["file"])
        version = result["version"]
        status = result["status"]
        if status not in {"created", "appended", "updated", "deleted", "not_found"}:
            raise ValueError("memory receipt status is invalid")
        if status in {"created", "appended", "updated"}:
            if not isinstance(version, str) or _MEMORY_VERSION.fullmatch(version) is None:
                raise ValueError("memory receipt version is invalid")
            kind = "write"
        else:
            if version is not None:
                raise ValueError("delete receipt cannot carry a version")
            kind = "delete"
        return _MutationReceipt(
            path=path,
            version=version,
            existed=status not in {"created", "not_found"},
            kind=kind,
        )
    except AIError:
        raise
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error


def _require_receipt(receipt: _MutationReceipt, path: str, kind: str) -> None:
    if receipt.path != path or receipt.kind != kind:
        raise MemoryOperationConflictError("memory operation targets a different mutation")


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
    "MemoryFile",
    "MemoryMutation",
    "MemoryOperation",
    "MemorySearchMatch",
    "MemorySearchResult",
    "MemoryStore",
    "RuntimeMemoryStore",
    "memory_operation_fingerprint",
    "normalize_memory_file",
]
