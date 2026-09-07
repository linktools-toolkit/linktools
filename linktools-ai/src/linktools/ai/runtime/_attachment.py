#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime attachment upload and release service."""

import hashlib
from typing import TYPE_CHECKING

from ..core import Principal, canonical_sha256, validate_idempotency_key
from ..errors import AIError, ErrorCode
from ._object import RuntimeObjectKeyFactory, put_runtime_object
from .service_api import AttachmentInfo
from .state import (
    AttachmentPresentation,
    AttachmentSemanticEntry,
    AttachmentUploadRecord,
    ContentRef,
    RuntimeDomain,
    managed_attachment_locator,
    managed_attachment_path,
)
from .state._attachment_repository import AttachmentRepository

if TYPE_CHECKING:
    from .state import RuntimeState


class DefaultAttachmentService:
    """Own application uploads without granting them to an Execution."""

    def __init__(
        self,
        repository: AttachmentRepository,
        state: "RuntimeState",
        object_key_factory: RuntimeObjectKeyFactory,
    ) -> None:
        self._repository = repository
        self._state = state
        self._object_key_factory = object_key_factory

    async def upload(
        self,
        data: bytes,
        *,
        media_type: str,
        name: str | None = None,
        principal: Principal,
        idempotency_key: str,
    ) -> AttachmentInfo:
        if not isinstance(data, bytes) or not data:
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        if not isinstance(media_type, str) or not media_type:
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        if name is not None and (not isinstance(name, str) or not name):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        validate_idempotency_key(idempotency_key)
        if principal.tenant_id != self._state.tenant_id:
            raise AIError(ErrorCode.AUTHORIZATION_DENIED)

        owner_key = self._repository.upload_key(principal, idempotency_key)
        digest = hashlib.sha256(data).hexdigest()
        size = len(data)
        path = managed_attachment_path("u", owner_key, 0)
        presentation = AttachmentPresentation(None, None)
        descriptor = AttachmentSemanticEntry(
            path,
            name,
            media_type,
            presentation,
            digest,
            size,
        )
        intent_digest = canonical_sha256(
            {
                "version": 1,
                "name": name,
                "media_type": media_type,
                "digest": digest,
                "size": size,
            }
        )

        existing = await self._repository.get_upload(
            owner_key,
            tenant_id=principal.tenant_id,
        )
        if existing is not None:
            if (
                existing.owner_principal != principal
                or existing.intent_digest != intent_digest
                or existing.descriptor != descriptor
            ):
                raise AIError(ErrorCode.IDEMPOTENCY_CONFLICT)
            return _attachment_info(existing.descriptor)

        owner_scope = f"attachment-upload:{owner_key}"
        store = self._state.working_object_store(
            RuntimeDomain.EXECUTION,
            owner_scope=owner_scope,
        )
        reference = await put_runtime_object(
            store,
            self._object_key_factory,
            RuntimeDomain.EXECUTION,
            principal.tenant_id,
            data,
        )
        candidate = AttachmentUploadRecord(
            1,
            principal,
            intent_digest,
            descriptor,
            ContentRef(
                RuntimeDomain.EXECUTION.value,
                owner_scope,
                reference,
            ),
            "HELD",
        )
        _owner_key, current = await self._repository.reserve_upload(
            principal,
            idempotency_key,
            candidate,
        )
        if current.intent_digest != intent_digest or current.descriptor != descriptor:
            raise AIError(ErrorCode.IDEMPOTENCY_CONFLICT)
        return _attachment_info(current.descriptor)

    async def release(
        self,
        path: str,
        *,
        principal: Principal,
    ) -> None:
        try:
            kind, owner_key, slot = managed_attachment_locator(path)
        except (TypeError, ValueError) as error:
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID) from error
        if kind != "u" or slot != 0:
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        if principal.tenant_id != self._state.tenant_id:
            raise AIError(ErrorCode.AUTHORIZATION_DENIED)
        current = await self._repository.release_upload(
            owner_key,
            principal=principal,
        )
        if current.descriptor.path != path:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        await self._state._release_object_scope(
            RuntimeDomain.EXECUTION,
            owner_scope=f"attachment-upload:{owner_key}",
        )


def _attachment_info(value: AttachmentSemanticEntry) -> AttachmentInfo:
    return AttachmentInfo(
        value.path,
        value.name,
        value.media_type,
        value.size,
    )


__all__ = ["DefaultAttachmentService"]
