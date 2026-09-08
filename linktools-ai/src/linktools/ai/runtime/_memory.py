#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime-owned memory tools and their persistence adapter."""

import hashlib
import json
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Protocol

from linktools.core import environ

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
_MAX_SEARCH_QUERY_CHARS = 1_000
_MAX_SEARCH_TERMS = 32
_MAX_SEARCH_FILES = 1_000
_MAX_SEARCH_MATCHES = 10
_MAX_SEARCH_OUTPUT_CHARS = 4_000
_MEMORY_FILE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,79}")
_MEMORY_VERSION = re.compile(r"m2:[0-9a-f]{64}")
_SHA256 = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True, slots=True)
class MemoryFile:
    content: str
    version: str
    storage_version: int = 0
    operation_id: str | None = None
    truncated: bool = False


@dataclass(frozen=True, slots=True)
class MemoryMutation:
    file: str
    version: str | None
    replayed: bool
    existed: bool
    status: str = "updated"


@dataclass(frozen=True, slots=True)
class MemoryOperation:
    id: str
    fingerprint: str
    action: str | None = None
    file: str | None = None
    content: str | None = None
    old_text: str | None = None
    append: bool | None = None


@dataclass(frozen=True, slots=True)
class MemorySearchMatch:
    file: str
    snippet: str
    score: float


@dataclass(frozen=True, slots=True)
class MemorySearchResult:
    matches: list[MemorySearchMatch]
    scanned: int
    truncated: bool


class MemoryStore(Protocol):
    async def read(self, file: str, *, max_chars: int) -> MemoryFile | None: ...

    async def get_operation(
        self,
        operation: MemoryOperation,
    ) -> MemoryMutation | None: ...

    async def write(
        self,
        file: str,
        content: str,
        *,
        expected_version: str | None,
        operation: MemoryOperation | None = None,
        append: bool = True,
    ) -> MemoryMutation: ...

    async def delete(
        self,
        file: str,
        *,
        expected_version: str | None,
        operation: MemoryOperation | None = None,
    ) -> MemoryMutation: ...

    async def search(
        self,
        query: str,
        *,
        limit: int = _MAX_SEARCH_MATCHES,
    ) -> MemorySearchResult: ...


class RuntimeMemoryStore:
    """Bind memory operations to one immutable tenant and logical scope."""

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

    async def read(self, file: str, *, max_chars: int) -> MemoryFile | None:
        if not isinstance(max_chars, int) or isinstance(max_chars, bool) or max_chars <= 0:
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        logical_file = _normalize_file(file)
        record = await self._record(logical_file)
        if record is None:
            return None
        content = await self._content(record)
        token = _record_version(record)
        operation_id = record.metadata.get("operation_id")
        if token is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return MemoryFile(
            content[:max_chars],
            token,
            _record_storage_version(record),
            operation_id if isinstance(operation_id, str) else None,
            len(content) > max_chars,
        )

    async def get_operation(self, operation: MemoryOperation) -> MemoryMutation | None:
        _validate_operation(operation)
        record = await self._state.operations.get(
            _operation_id(self._memory_scope_digest, operation.id),
            tenant_id=self._tenant_id,
        )
        if record is None:
            return None
        if (
            record.operation_id
            != _operation_id(self._memory_scope_digest, operation.id)
            or record.tenant_id != self._tenant_id
            or record.resource_kind is not ResourceKind.MEMORY
            or record.request_digest != operation.fingerprint
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
            if record.request_digest != operation.fingerprint:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        mutation = _decode_receipt(record.result_ref)
        if mutation.version is not None and mutation.version != _version_token(
            self._namespace,
            self._tenant_id,
            self._memory_scope_digest,
            mutation.file,
            operation.id,
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        expected_resource_id = _memory_id(
            self._memory_scope_digest,
            mutation.file,
        )
        expected_kind = (
            OperationKind.MEMORY_DELETE
            if mutation.status in {"deleted", "not_found"}
            else OperationKind.MEMORY_WRITE
        )
        if (
            record.resource_id != expected_resource_id
            or record.operation_kind is not expected_kind
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return mutation

    async def write(
        self,
        file: str,
        content: str,
        *,
        expected_version: str | None,
        operation: MemoryOperation | None = None,
        append: bool = True,
    ) -> MemoryMutation:
        logical_file = _normalize_file(file)
        _validate_content(content)
        _validate_expected_version(expected_version)
        if not isinstance(append, bool):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        operation = _ensure_operation(
            operation,
            action="write",
            file=logical_file,
            content=content,
            expected_version=expected_version,
            append=append,
        )
        replay = await self._replay(
            operation,
            logical_file=logical_file,
            operation_kind=OperationKind.MEMORY_WRITE,
        )
        if replay is not None:
            return replay
        current = await self._record(logical_file)
        _check_version(current, expected_version)
        expected_storage_version = (
            None if current is None else _record_storage_version(current)
        )
        existing_content = "" if current is None else await self._content(current)
        if append:
            next_content, status = _append_content(
                existing_content,
                content,
                existed=current is not None,
            )
        else:
            next_content, status = content, "updated"
        _validate_operation_effect(
            operation,
            logical_file=logical_file,
            existing_content=existing_content,
            next_content=next_content,
            append=append,
            existed=current is not None,
        )
        token = _version_token(
            self._namespace,
            self._tenant_id,
            self._memory_scope_digest,
            logical_file,
            operation.id,
        )
        stored_content = StoredPayload.inline_text(next_content)
        if not payload_fits_inline(stored_content, self._payload_policy):
            reference = await put_runtime_object(
                self._object_store,
                RuntimeObjectKeyFactory(self._namespace),
                RuntimeDomain.MEMORY,
                self._tenant_id,
                next_content.encode("utf-8"),
            )
            stored_content = StoredPayload.object(reference)
        now = datetime.now(timezone.utc)
        next_record = MemoryRecord(
            _memory_id(self._memory_scope_digest, logical_file),
            self._tenant_id,
            self._memory_scope_digest,
            stored_content,
            {
                "path": logical_file,
                "version": token,
                "operation_id": _operation_id(
                    self._memory_scope_digest,
                    operation.id,
                ),
            },
            0,
            now if current is None else current.created_at,
            now,
        )
        mutation = MemoryMutation(
            logical_file,
            token,
            False,
            current is not None,
            status,
        )
        operation_input = _operation_input(
            operation,
            self._memory_scope_digest,
            self._tenant_id,
            mutation,
            OperationKind.MEMORY_WRITE,
            next_record.memory_id,
        )
        try:
            stored, replayed = await self._state.records.apply_write(
                next_record,
                expected_revision=None if current is None else current.revision,
                expected_storage_version=expected_storage_version,
                operation=operation_input,
            )
        except AIError as error:
            raise _map_memory_error(error) from error
        if replayed:
            replay = await self._replay(
                operation,
                logical_file=logical_file,
                operation_kind=OperationKind.MEMORY_WRITE,
            )
            if replay is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            return replay
        if stored is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        _logger.debug(
            "memory write committed: scope=%s file=%s status=%s",
            self._memory_scope_digest,
            logical_file,
            status,
        )
        return mutation

    async def delete(
        self,
        file: str,
        *,
        expected_version: str | None,
        operation: MemoryOperation | None = None,
    ) -> MemoryMutation:
        logical_file = _normalize_file(file)
        if logical_file == "MEMORY.md":
            raise AIError(ErrorCode.AUTHORIZATION_DENIED)
        _validate_expected_version(expected_version)
        operation = _ensure_operation(
            operation,
            action="delete",
            file=logical_file,
            content=None,
            expected_version=expected_version,
            append=False,
        )
        replay = await self._replay(
            operation,
            logical_file=logical_file,
            operation_kind=OperationKind.MEMORY_DELETE,
        )
        if replay is not None:
            return replay
        current = await self._record(logical_file)
        _check_version(current, expected_version)
        mutation = MemoryMutation(
            logical_file,
            None,
            False,
            current is not None,
            "deleted" if current is not None else "not_found",
        )
        operation_input = _operation_input(
            operation,
            self._memory_scope_digest,
            self._tenant_id,
            mutation,
            OperationKind.MEMORY_DELETE,
            _memory_id(self._memory_scope_digest, logical_file),
        )
        try:
            deleted, replayed = await self._state.records.apply_delete(
                _memory_id(self._memory_scope_digest, logical_file),
                tenant_id=self._tenant_id,
                expected_revision=0 if current is None else current.revision,
                expected_storage_version=(
                    None if current is None else _record_storage_version(current)
                ),
                operation=operation_input,
            )
        except AIError as error:
            raise _map_memory_error(error) from error
        if replayed:
            replay = await self._replay(
                operation,
                logical_file=logical_file,
                operation_kind=OperationKind.MEMORY_DELETE,
            )
            if replay is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            return replay
        if deleted != (current is not None):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        _logger.debug(
            "memory delete committed: scope=%s file=%s status=%s",
            self._memory_scope_digest,
            logical_file,
            mutation.status,
        )
        return mutation

    async def search(
        self,
        query: str,
        *,
        limit: int = _MAX_SEARCH_MATCHES,
    ) -> MemorySearchResult:
        if (
            not isinstance(query, str)
            or not query.strip()
            or len(query) > _MAX_SEARCH_QUERY_CHARS
            or not isinstance(limit, int)
            or isinstance(limit, bool)
            or not 1 <= limit <= _MAX_SEARCH_MATCHES
        ):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        terms = tuple(dict.fromkeys(query.lower().split()))
        if len(terms) > _MAX_SEARCH_TERMS:
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        records, truncated = await self._list_records(limit=_MAX_SEARCH_FILES)
        candidates: list[MemorySearchMatch] = []
        for record in records:
            file = record.metadata.get("path")
            if not isinstance(file, str):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            logical_file = _normalize_file(file)
            if logical_file != file:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            _validate_record(
                record,
                logical_file,
                self._memory_scope_digest,
                self._tenant_id,
            )
            content = await self._content(record)
            file_haystack = file.lower()
            content_haystack = content.lower()
            score = sum(
                file_haystack.count(term) + content_haystack.count(term)
                for term in terms
            )
            if score == 0:
                continue
            index = min(
                (
                    content_haystack.find(term)
                    for term in terms
                    if content_haystack.find(term) >= 0
                ),
                default=0,
            )
            snippet = content[max(0, index - 120) : max(0, index - 120) + 400]
            candidates.append(MemorySearchMatch(file, snippet, float(score)))
        candidates.sort(key=lambda value: (-value.score, value.file))
        selected: list[MemorySearchMatch] = []
        used = 0
        for value in candidates[:limit]:
            size = len(value.file) + len(value.snippet)
            if used + size > _MAX_SEARCH_OUTPUT_CHARS:
                truncated = True
                break
            selected.append(value)
            used += size
        if len(selected) < len(candidates):
            truncated = True
        return MemorySearchResult(selected, len(records), truncated)

    async def _list_records(
        self,
        *,
        limit: int | None = None,
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

    async def _record(self, logical_file: str) -> MemoryRecord | None:
        record = await self._state.records.get(
            _memory_id(self._memory_scope_digest, logical_file),
            tenant_id=self._tenant_id,
        )
        if record is not None:
            _validate_record(
                record,
                logical_file,
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

    async def _replay(
        self,
        operation: MemoryOperation | None,
        *,
        logical_file: str,
        operation_kind: OperationKind,
    ) -> MemoryMutation | None:
        if operation is None:
            return None
        mutation = await self.get_operation(operation)
        if mutation is None:
            return None
        actual_kind = (
            OperationKind.MEMORY_DELETE
            if mutation.status in {"deleted", "not_found"}
            else OperationKind.MEMORY_WRITE
        )
        if mutation.file != logical_file or actual_kind is not operation_kind:
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        return mutation


def _normalize_file(file: str) -> str:
    if not isinstance(file, str):
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
    value = file.strip()
    if (
        not value
        or "/" in value
        or "\\" in value
        or value.startswith("/")
        or "\x00" in value
    ):
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
    if not value.endswith(".md"):
        value += ".md"
    if ".." in value or _MEMORY_FILE.fullmatch(value) is None:
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
    return value


def normalize_memory_file(file: str) -> str:
    """Return the canonical logical filename used by memory commands."""
    return _normalize_file(file)


def _validate_content(content: str) -> None:
    if (
        not isinstance(content, str)
        or len(content) > _MAX_CONTENT_CHARS
        or "\x00" in content
    ):
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
    try:
        content.encode("utf-8", errors="strict")
    except UnicodeEncodeError as error:
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID) from error


def _validate_operation(operation: MemoryOperation) -> None:
    if (
        not isinstance(operation, MemoryOperation)
        or not isinstance(operation.id, str)
        or not operation.id
        or not isinstance(operation.fingerprint, str)
        or _SHA256.fullmatch(operation.fingerprint) is None
    ):
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
    if operation.action is None:
        if any(
            value is not None
            for value in (
                operation.file,
                operation.content,
                operation.old_text,
                operation.append,
            )
        ):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        return
    if (
        operation.action not in {"write", "delete"}
        or not isinstance(operation.file, str)
        or _normalize_file(operation.file) != operation.file
        or operation.content is not None
        and not isinstance(operation.content, str)
        or operation.old_text is not None
        and not isinstance(operation.old_text, str)
        or operation.append is not None
        and not isinstance(operation.append, bool)
    ):
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
    if operation.action == "write" and not isinstance(operation.content, str):
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
    if operation.action == "delete" and (
        operation.content is not None
        or operation.old_text is not None
        or operation.append is not False
    ):
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID)


def _append_content(
    existing: str,
    content: str,
    *,
    existed: bool,
) -> tuple[str, str]:
    if not content.strip():
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
    if existing == "":
        value = content if content.endswith("\n") else content + "\n"
        status = "appended" if existed else "created"
    else:
        value = existing.rstrip() + "\n" + content
        if not value.endswith("\n"):
            value += "\n"
        status = "appended"
    if len(value) > _MAX_CONTENT_CHARS:
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
    return value, status


def _validate_operation_effect(
    operation: MemoryOperation,
    *,
    logical_file: str,
    existing_content: str,
    next_content: str,
    append: bool,
    existed: bool,
) -> None:
    if operation.action is None:
        return
    if (
        operation.action != "write"
        or operation.file != logical_file
        or operation.append != append
    ):
        raise AIError(ErrorCode.STORAGE_CONFLICT)
    if operation.content is None:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    if append:
        expected_content, _ = _append_content(
            existing_content,
            operation.content,
            existed=existed,
        )
    elif operation.old_text is None:
        expected_content = operation.content
    else:
        if (
            not operation.old_text
            or existing_content.count(operation.old_text) != 1
        ):
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        expected_content = existing_content.replace(
            operation.old_text,
            operation.content,
        )
    if expected_content != next_content:
        raise AIError(ErrorCode.STORAGE_CONFLICT)


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
    logical_file: str,
    scope_digest: str,
    tenant_id: str,
) -> None:
    if (
        record.tenant_id != tenant_id
        or record.memory_id != _memory_id(scope_digest, logical_file)
        or record.memory_scope_digest != scope_digest
        or not isinstance(record.revision, int)
        or isinstance(record.revision, bool)
        or record.revision < 1
        or _record_version(record) is None
        or _record_storage_version(record) != record.revision
        or record.metadata.get("path") != logical_file
        or _SHA256.fullmatch(str(record.metadata.get("operation_id"))) is None
    ):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)


def _version_token(
    namespace: str,
    tenant_id: str,
    scope_digest: str,
    file: str,
    identity: str,
) -> str:
    return "m2:" + canonical_sha256(
        {
            "namespace": namespace,
            "tenant_id": tenant_id,
            "scope": scope_digest,
            "file": file,
            "identity": canonical_sha256(identity),
        }
    )


def _memory_id(scope_digest: str, logical_file: str) -> str:
    return hashlib.sha256(f"{scope_digest}\0{logical_file}".encode()).hexdigest()


def _operation_id(scope_digest: str, operation_id: str) -> str:
    return hashlib.sha256(f"{scope_digest}\0{operation_id}".encode()).hexdigest()


def _operation_input(
    operation: MemoryOperation,
    scope_digest: str,
    tenant_id: str,
    mutation: MemoryMutation,
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
        _encode_receipt(mutation),
        None,
        None,
        True,
        now,
        now,
    )


def _encode_receipt(mutation: MemoryMutation) -> str:
    return json.dumps(
        {
            "version": 2,
            "result": {
                "file": mutation.file,
                "version": mutation.version,
                "status": mutation.status,
            },
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _decode_receipt(value: str) -> MemoryMutation:
    try:
        raw = json.loads(value)
        if not isinstance(raw, dict):
            raise ValueError("memory receipt is not an object")
        version = raw.get("version")
        if (
            "version" not in raw
            or isinstance(version, bool)
            or not isinstance(version, int)
            or version < 1
        ):
            raise ValueError("memory receipt version is malformed")
        if version != 2:
            raise AIError(ErrorCode.STORAGE_VERSION_UNSUPPORTED)
        result = raw["result"]
        if (
            set(raw) != {"version", "result"}
            or not isinstance(result, dict)
            or set(result) != {"file", "version", "status"}
            or not isinstance(result["file"], str)
            or not isinstance(result["status"], str)
            or result["status"]
            not in {"created", "appended", "updated", "deleted", "not_found"}
            or result["version"] is not None
            and not isinstance(result["version"], str)
        ):
            raise ValueError("memory receipt version is invalid")
        status = result["status"]
        file = result["file"]
        try:
            canonical_file = _normalize_file(file)
        except AIError as error:
            raise ValueError("memory receipt file is invalid") from error
        if canonical_file != file:
            raise ValueError("memory receipt file is not canonical")
        version = result["version"]
        if status in {"created", "appended", "updated"}:
            if (
                not isinstance(version, str)
                or _MEMORY_VERSION.fullmatch(version) is None
            ):
                raise ValueError("memory receipt token is invalid")
        elif version is not None:
            raise ValueError("deleted memory receipt cannot have a token")
        return MemoryMutation(
            file,
            version,
            True,
            status not in {"created", "not_found"},
            status,
        )
    except AIError:
        raise
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error


def _check_version(record: MemoryRecord | None, expected: str | None) -> None:
    if record is None:
        if expected is not None:
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        return
    actual = _record_version(record)
    if actual is None or expected != actual:
        raise AIError(ErrorCode.STORAGE_CONFLICT)


def _validate_expected_version(value: str | None) -> None:
    if value is not None and (
        not isinstance(value, str) or _MEMORY_VERSION.fullmatch(value) is None
    ):
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID)


def _map_memory_error(error: AIError) -> AIError:
    if error.code is ErrorCode.STORAGE_CONFLICT:
        return AIError(ErrorCode.STORAGE_CONFLICT)
    return error


def _ensure_operation(
    operation: MemoryOperation | None,
    *,
    action: str,
    file: str,
    content: str | None,
    expected_version: str | None,
    append: bool,
) -> MemoryOperation:
    if operation is not None:
        _validate_operation(operation)
        if operation.action is None:
            expected = memory_operation_fingerprint(
                action,
                file,
                content,
                None,
                append,
            )
        else:
            if operation.action != action or operation.file != file:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            operation_append = (
                append if operation.append is None else operation.append
            )
            if operation_append != append:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            expected = memory_operation_fingerprint(
                operation.action,
                operation.file,
                operation.content,
                operation.old_text,
                operation_append,
            )
        if operation.fingerprint != expected:
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        return operation
    return MemoryOperation(
        uuid.uuid4().hex,
        memory_operation_fingerprint(action, file, content, None, append),
        action,
        file,
        content,
        None,
        append,
    )


def memory_operation_fingerprint(
    action: str,
    file: str,
    content: str | None,
    old_text: str | None,
    append: bool,
) -> str:
    return canonical_sha256(
        {
            "action": action,
            "file": file,
            "content": content,
            "old_text": old_text,
            "append": append,
        }
    )


__all__ = [
    "MemoryFile",
    "MemoryMutation",
    "MemoryOperation",
    "MemorySearchMatch",
    "MemorySearchResult",
    "MemoryStore",
    "memory_operation_fingerprint",
    "normalize_memory_file",
    "RuntimeMemoryStore",
]
