#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime-backed Harness memory storage."""

import asyncio
import hashlib
import json
import re
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
    SearchableMemoryStore,
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
from ..runtime import (
    RuntimeDomain,
    RuntimeObjectKeyFactory,
    put_runtime_object,
    read_runtime_object,
)
from ..runtime.state import MemoryRecord, MemoryState
from ..storage import ObjectStore, PayloadPolicy, StoredPayload, payload_fits_inline

_logger = environ.get_logger("ai.adapter.memory")
_MEMORY_PATH_SEGMENT = re.compile(r"[A-Za-z0-9_.-]{1,200}")


def _memory_working_scope_digest(execution_id: str, memory_scope: str) -> str:
    logical_scope_digest = canonical_sha256(memory_scope)
    return canonical_sha256({"execution_id": execution_id, "memory_scope_digest": logical_scope_digest})


class RuntimeMemoryStore(SearchableMemoryStore):
    """Map Harness memory paths to the bound Runtime memory repositories."""

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
        payload_policy: "PayloadPolicy | None" = None,
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
        self._transient = transient
        self._payload_policy = payload_policy or PayloadPolicy()
        logical_scope_digest = canonical_sha256(memory_scope)
        self._memory_scope_digest = (
            _memory_working_scope_digest(execution_id, memory_scope)
            if transient
            else logical_scope_digest
        )
        self._lock = asyncio.Lock()

    async def read(self, path: str, *, max_chars: int) -> "MemoryFile | None":
        if max_chars <= 0:
            raise ValueError("max_chars must be positive")
        logical_path = self._path(path)
        record = await self._record(logical_path)
        if record is None:
            return None
        content = await self._content(record)
        operation_id = record.metadata.get("operation_id")
        return MemoryFile(content[:max_chars], _version(record.revision), operation_id if isinstance(operation_id, str) else None, len(content) > max_chars)

    async def get_operation(self, operation: MemoryOperation) -> MemoryMutation | None:
        record = await self._state.operations.get(
            _operation_id(self._memory_scope_digest, operation.id),
            tenant_id=self._tenant_id,
        )
        if record is None:
            return None
        if record.resource_kind is not ResourceKind.MEMORY or record.request_digest != operation.fingerprint:
            raise MemoryOperationConflictError("memory operation fingerprint conflict")
        if record.status is not OperationStatus.SUCCEEDED or record.result_ref is None:
            raise MemoryOperationConflictError("memory operation is not replayable")
        mutation = _decode_mutation(record.result_ref)
        return MemoryMutation(mutation.version, True, mutation.existed)

    async def write(
        self,
        path: str,
        content: str,
        *,
        expected_version: str | None,
        operation: MemoryOperation | None = None,
    ) -> MemoryMutation:
        logical_path = self._path(path)
        async with self._lock:
            replay = await self._replay(operation)
            if replay is not None:
                return replay
            current = await self._record(logical_path)
            _check_version(current, expected_version)
            inline = StoredPayload.inline_text(content)
            stored_content = inline
            if not payload_fits_inline(inline, self._payload_policy):
                blob = await put_runtime_object(
                    self._object_store,
                    RuntimeObjectKeyFactory(self._namespace),
                    RuntimeDomain.MEMORY,
                    self._tenant_id,
                    content.encode("utf-8"),
                )
                stored_content = StoredPayload.object(blob)
            now = datetime.now(timezone.utc)
            next_record = MemoryRecord(
                _memory_id(self._memory_scope_digest, logical_path),
                self._tenant_id,
                self._memory_scope_digest,
                stored_content,
                {"path": logical_path, **({} if operation is None else {"operation_id": operation.id})},
                0 if current is None else current.revision + 1,
                now if current is None else current.created_at,
                now,
            )
            mutation = MemoryMutation(_version(next_record.revision), False, current is not None)
            operation_input = _operation_input(
                operation,
                self._memory_scope_digest,
                self._tenant_id,
                mutation,
                OperationKind.MEMORY_WRITE,
                _memory_id(self._memory_scope_digest, logical_path),
            )
            try:
                stored, replayed = await self._state.records.put_with_operation(
                    next_record,
                    expected_revision=None if current is None else current.revision,
                    operation=operation_input,
                )
            except AIError as error:
                mapped = await self._map_mutation_error(error, operation)
                raise mapped from error
            if replayed:
                replay = await self._replay(operation)
                if replay is None:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                return replay
            if stored is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            mutation = MemoryMutation(_version(stored.revision), False, current is not None)
            _logger.debug(
                "runtime memory written: memory_scope_digest=%s path_digest=%s replayed=%s",
                self._memory_scope_digest,
                canonical_sha256(logical_path),
                False,
            )
            return mutation

    async def delete(
        self,
        path: str,
        *,
        expected_version: str | None,
        operation: MemoryOperation | None = None,
    ) -> MemoryMutation:
        logical_path = self._path(path)
        async with self._lock:
            replay = await self._replay(operation)
            if replay is not None:
                return replay
            current = await self._record(logical_path)
            _check_version(current, expected_version)
            if current is None:
                mutation = MemoryMutation(None, False, False)
            else:
                mutation = MemoryMutation(None, False, True)
            operation_input = _operation_input(
                operation,
                self._memory_scope_digest,
                self._tenant_id,
                mutation,
                OperationKind.MEMORY_DELETE,
                _memory_id(self._memory_scope_digest, logical_path),
            )
            try:
                deleted, replayed = await self._state.records.delete_with_operation(
                    _memory_id(self._memory_scope_digest, logical_path),
                    tenant_id=self._tenant_id,
                    expected_revision=None if current is None else current.revision,
                    operation=operation_input,
                )
            except AIError as error:
                mapped = await self._map_mutation_error(error, operation)
                raise mapped from error
            if replayed:
                replay = await self._replay(operation)
                if replay is None:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                return replay
            if deleted != (current is not None):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            _logger.debug(
                "runtime memory deleted: memory_scope_digest=%s path_digest=%s replayed=%s",
                self._memory_scope_digest,
                canonical_sha256(logical_path),
                False,
            )
            return mutation

    async def list_paths(self, prefix: str = "", *, limit: int) -> "list[str]":
        if limit <= 0:
            raise ValueError("limit must be positive")
        normalized_prefix = self._prefix(prefix)
        records = await self._list_records()
        result = sorted(
            (item.metadata["path"] for item in records if isinstance(item.metadata.get("path"), str)),
        )
        return [path for path in result if path.startswith(normalized_prefix)][:limit]

    async def search(
        self,
        prefix: str,
        query: str,
        *,
        limit: int,
        max_files: int,
        max_chars: int,
        max_file_chars: int,
    ) -> MemorySearchResult:
        normalized_prefix = self._prefix(prefix)
        if not query.split() or limit <= 0 or max_files <= 0 or max_chars <= 0 or max_file_chars <= 0:
            return MemorySearchResult([], 0, False)
        records = await self._list_records()
        matching_records = [
            record
            for record in records
            if isinstance(record.metadata.get("path"), str)
            and record.metadata["path"].startswith(normalized_prefix)
        ]
        matching_records.sort(key=lambda record: str(record.metadata["path"]))
        files_truncated = len(matching_records) > max_files
        records = matching_records[:max_files]
        candidates: list[tuple[float, str, str]] = []
        scanned = 0
        content_truncated = False
        terms = tuple(term.lower() for term in query.split())
        for record in records[:max_files]:
            logical_path = record.metadata.get("path")
            if not isinstance(logical_path, str):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            path = logical_path
            scanned += 1
            full_content = await self._content(record)
            content_truncated = content_truncated or len(full_content) > max_file_chars
            content = full_content[:max_file_chars]
            searchable = f"{path}\n{content}".lower()
            score = float(sum(searchable.count(term) for term in terms))
            if score == 0:
                continue
            index = min((content.lower().find(term) for term in terms if content.lower().find(term) >= 0), default=0)
            start = max(0, index - 120)
            snippet = content[start:start + min(400, max_chars)]
            candidates.append((score, path, snippet))
        candidates.sort(key=lambda item: (-item[0], item[1]))
        matches: list[MemorySearchMatch] = []
        remaining = max_chars
        for score, path, snippet in candidates[:limit]:
            available = remaining - len(path)
            if available <= 0:
                break
            visible = snippet[:available]
            matches.append(MemorySearchMatch(path, visible, score))
            remaining -= len(path) + len(visible)
        return MemorySearchResult(matches, scanned, files_truncated or content_truncated or len(matches) < len(candidates))

    async def _list_records(self, *, limit: int | None = None) -> list[MemoryRecord]:
        records: list[MemoryRecord] = []
        cursor = None
        while limit is None or len(records) < limit:
            page_limit = 200 if limit is None else min(200, limit - len(records))
            page = await self._state.records.list(
                tenant_id=self._tenant_id,
                memory_scope_digest=self._memory_scope_digest,
                cursor=cursor,
                limit=page_limit,
            )
            records.extend(page.items)
            if page.next_cursor is None or page.next_cursor == cursor or not page.items:
                break
            cursor = page.next_cursor
        return records if limit is None else records[:limit]

    async def _record(self, logical_path: str) -> MemoryRecord | None:
        return await self._state.records.get(
            _memory_id(self._memory_scope_digest, logical_path),
            tenant_id=self._tenant_id,
        )

    async def _content(self, record: MemoryRecord) -> str:
        if record.content.kind == "inline":
            content = record.content.decode()
            if not isinstance(content, str):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            return content
        reference = record.content.ref
        if reference is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return (await read_runtime_object(self._object_store, reference)).decode("utf-8")

    async def _replay(self, operation: MemoryOperation | None) -> MemoryMutation | None:
        return None if operation is None else await self.get_operation(operation)

    async def _map_mutation_error(self, error: AIError, operation: MemoryOperation | None) -> Exception:
        if error.code is not ErrorCode.STORAGE_CONFLICT:
            return error
        if operation is not None:
            existing = await self._state.operations.get(
                _operation_id(self._memory_scope_digest, operation.id),
                tenant_id=self._tenant_id,
            )
            if existing is not None:
                if existing.request_digest != operation.fingerprint:
                    return MemoryOperationConflictError("memory operation fingerprint conflict")
                return MemoryOperationConflictError("memory operation conflict")
        return MemoryConflictError("memory version conflict")

    def _path(self, path: str) -> str:
        if not isinstance(path, str) or not path or any(
            not _MEMORY_PATH_SEGMENT.fullmatch(segment) for segment in path.split("/")
        ):
            raise ValueError("memory path is invalid")
        return path

    def _prefix(self, prefix: str) -> str:
        if prefix == "":
            return ""
        normalized = prefix.rstrip("/")
        if not normalized:
            raise ValueError("memory prefix is invalid")
        self._path(normalized)
        return f"{normalized}/"


def _operation_input(operation: "MemoryOperation | None", namespace_digest: str, tenant_id: str, mutation: MemoryMutation, kind: OperationKind, resource_id: str) -> "OperationLedgerInput | None":
    if operation is None:
        return None
    now = datetime.now(timezone.utc)
    return OperationLedgerInput(
        _operation_id(namespace_digest, operation.id),
        tenant_id,
        ResourceKind.MEMORY,
        resource_id,
        None,
        kind,
        OperationStatus.SUCCEEDED,
        operation.fingerprint,
        _encode_mutation(mutation),
        None,
        None,
        True,
        now,
        now,
    )


def _memory_id(namespace_digest: str, logical_path: str) -> str:
    return hashlib.sha256(f"{namespace_digest}\0{logical_path}".encode()).hexdigest()


def _operation_id(namespace_digest: str, operation_id: str) -> str:
    return hashlib.sha256(f"{namespace_digest}\0{operation_id}".encode()).hexdigest()


def _version(revision: int) -> str:
    return f"v{revision}"


def _check_version(record: MemoryRecord | None, expected_version: str | None) -> None:
    if record is None:
        if expected_version is not None:
            raise MemoryConflictError("memory version conflict")
        return
    if expected_version != _version(record.revision):
        raise MemoryConflictError("memory version conflict")


def _encode_mutation(mutation: MemoryMutation) -> str:
    return json.dumps({"version": mutation.version, "existed": mutation.existed}, separators=(",", ":"), sort_keys=True)


def _decode_mutation(value: str) -> MemoryMutation:
    try:
        raw = json.loads(value)
        if not isinstance(raw, dict):
            raise ValueError
        return MemoryMutation(None if raw.get("version") is None else str(raw["version"]), False, bool(raw["existed"]))
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error


__all__ = ["RuntimeMemoryStore"]
