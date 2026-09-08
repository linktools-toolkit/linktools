#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Execution-scoped read_attachment resolution and source freezing."""

import hashlib
import mimetypes
from collections.abc import Awaitable, Callable
from pathlib import PurePosixPath
from typing import TYPE_CHECKING

from ..capability import WorkspaceAccess
from ..core import ExecutionStatus, JsonValue
from ..errors import AIError, ErrorCode
from ..storage import ObjectRef, TransientObjectStore
from ._attachment_tool_state import attachment_tool_return
from ._object import RuntimeObjectKeyFactory, read_runtime_object
from .state import (
    AttachmentEntry,
    AttachmentPresentation,
    AttachmentResult,
    AttachmentSourceRecord,
    ContentRef,
    RuntimeDomain,
    managed_attachment_locator,
    managed_attachment_path,
)
from .state._attachment_repository import AttachmentRepository

if TYPE_CHECKING:
    from ._local import LocalExecutionBackend
    from .state import ExecutionRecord

_AttachmentCommitted = Callable[[str, AttachmentResult], Awaitable[None]]


class AttachmentReadRuntime:
    """Resolve read_attachment against one live Execution and its frozen facts."""

    def __init__(
        self,
        backend: "LocalExecutionBackend",
        *,
        execution_id: str,
        tenant_id: str,
        agent_run_sequence: int,
        on_committed: _AttachmentCommitted | None = None,
    ) -> None:
        if not isinstance(execution_id, str) or not execution_id:
            raise ValueError("execution_id is required")
        if not isinstance(tenant_id, str) or not tenant_id:
            raise ValueError("tenant_id is required")
        if (
            isinstance(agent_run_sequence, bool)
            or not isinstance(agent_run_sequence, int)
            or agent_run_sequence < 0
        ):
            raise ValueError("agent_run_sequence is invalid")
        self._backend = backend
        self._execution_id = execution_id
        self._tenant_id = tenant_id
        self._agent_run_sequence = agent_run_sequence
        self._on_committed = on_committed
        self._repository = AttachmentRepository(
            backend._execution.executions.state_store,
            namespace=backend._namespace,
            tenant_id=tenant_id,
        )
        self._object_keys = RuntimeObjectKeyFactory(backend._namespace)
        self._cache: dict[str, tuple[AttachmentEntry, bytes]] = {}

    async def read(
        self,
        access: WorkspaceAccess,
        path: str,
    ) -> dict[str, JsonValue]:
        if not isinstance(access, WorkspaceAccess):
            raise TypeError("access must be WorkspaceAccess")
        execution = await self._execution()
        cached = self._cache.get(path)
        if cached is not None:
            entry, body = cached
            _verify_body(entry, body)
            return self._result(entry)
        if not isinstance(path, str) or not path:
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        if path.startswith("virtual:"):
            entry = await self._managed_entry(execution, path)
            body = await self._read_content(entry.content)
        else:
            relative, media_type = _ordinary_path(path)
            source = await self._repository.get_source(
                self._execution_id,
                relative,
                tenant_id=self._tenant_id,
            )
            if source is None:
                body = await access.read_bytes(relative)
                if not body:
                    raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
                owner_key = self._repository.source_key(
                    self._execution_id,
                    relative,
                )
                content = await self._store_content(
                    body,
                    owner_scope=f"attachment-source:{owner_key}",
                )
                candidate = AttachmentSourceRecord(
                    1,
                    self._execution_id,
                    relative,
                    AttachmentEntry(
                        managed_attachment_path("e", owner_key, 0),
                        PurePosixPath(relative).name,
                        media_type,
                        AttachmentPresentation(None, None),
                        content,
                    ),
                )
                _key, source = await self._repository.freeze_source(candidate)
                if source == candidate:
                    entry = candidate.entry
                    _verify_body(entry, body)
                else:
                    entry = source.entry
                    body = await self._read_content(entry.content)
            else:
                entry = source.entry
                body = await self._read_content(entry.content)
        _verify_body(entry, body)
        self._cache[path] = (entry, body)
        return self._result(entry)

    def _result(self, entry: AttachmentEntry) -> dict[str, JsonValue]:
        return attachment_tool_return(
            {
                "status": "read",
                "path": entry.path,
                "name": entry.name,
                "media_type": entry.media_type,
                "size": entry.content.object.size,
            },
            AttachmentResult(1, entry),
            on_committed=self._on_committed,
        )

    async def _execution(self) -> "ExecutionRecord":
        execution = await self._backend._execution.executions.get(
            self._execution_id,
            tenant_id=self._tenant_id,
        )
        if execution is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if (
            execution.status is not ExecutionStatus.STARTED
            or execution.agent_run_sequence != self._agent_run_sequence
        ):
            raise AIError(ErrorCode.CAPABILITY_POLICY_CONFLICT)
        return execution

    async def _managed_entry(
        self,
        execution: "ExecutionRecord",
        path: str,
    ) -> AttachmentEntry:
        try:
            kind, owner_key, slot = managed_attachment_locator(path)
        except (TypeError, ValueError) as error:
            raise AIError(ErrorCode.AUTHORIZATION_DENIED) from error
        if kind == "p":
            entries = tuple(
                entry for entry in execution.attachment_manifest if entry.path == path
            )
            if len(entries) != 1:
                raise AIError(ErrorCode.AUTHORIZATION_DENIED)
            return entries[0]
        if kind == "e":
            if slot != 0:
                raise AIError(ErrorCode.AUTHORIZATION_DENIED)
            source = await self._repository.get_source_by_key(
                owner_key,
                tenant_id=self._tenant_id,
            )
            if (
                source is None
                or source.execution_id != self._execution_id
                or source.entry.path != path
            ):
                raise AIError(ErrorCode.AUTHORIZATION_DENIED)
            return source.entry
        raise AIError(ErrorCode.AUTHORIZATION_DENIED)

    async def _read_content(self, content: ContentRef) -> bytes:
        if content.domain != RuntimeDomain.EXECUTION.value:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        store = self._backend._execution_objects
        if isinstance(store, TransientObjectStore):
            if content.owner_scope is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            store = store.scoped(
                f"runtime:{RuntimeDomain.EXECUTION.value}:{content.owner_scope}"
            )
        return await read_runtime_object(store, content.object)

    async def _store_content(
        self,
        data: bytes,
        *,
        owner_scope: str,
    ) -> ContentRef:
        if not data:
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        digest = hashlib.sha256(data).hexdigest()
        store = self._backend._execution_objects
        if isinstance(store, TransientObjectStore):
            store = store.scoped(
                f"runtime:{RuntimeDomain.EXECUTION.value}:{owner_scope}"
            )
        key = self._object_keys.key(
            RuntimeDomain.EXECUTION,
            self._tenant_id,
            digest,
        )
        stat = await store.stat(key)
        if stat is not None:
            if stat.digest != digest or stat.size != len(data):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            reference = ObjectRef(store.store_id, key, stat.digest, stat.size)
        else:
            async def chunks():
                yield data

            created = await store.put(
                key,
                chunks(),
                expected_size=len(data),
                expected_digest=digest,
            )
            reference = ObjectRef(store.store_id, key, created.digest, created.size)
        return ContentRef(
            RuntimeDomain.EXECUTION.value,
            owner_scope,
            reference,
        )


def _ordinary_path(path: str) -> tuple[str, str]:
    if (
        path.startswith("file:")
        or "://" in path
        or "\\" in path
        or "\x00" in path
    ):
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
    value = PurePosixPath(path)
    if value.is_absolute() or any(part in {"", ".", ".."} for part in value.parts):
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
    relative = value.as_posix()
    if relative != path:
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
    media_type, _encoding = mimetypes.guess_type(relative, strict=False)
    if media_type is None:
        raise AIError(
            ErrorCode.REQUEST_FIELD_INVALID,
            safe_details={"field": "path", "reason": "media_type_unknown"},
        )
    return relative, media_type


def _verify_body(entry: AttachmentEntry, body: bytes) -> None:
    reference = entry.content.object
    if (
        not isinstance(body, bytes)
        or not body
        or len(body) != reference.size
        or hashlib.sha256(body).hexdigest() != reference.digest
    ):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)


__all__ = ["AttachmentReadRuntime"]
