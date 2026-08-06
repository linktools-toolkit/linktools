#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Blob state and object-store lifecycle coordination."""

from linktools.core import environ

from ...domain.blob import BlobObject, BlobReference, BlobState
from ...foundation.digest import sha256_digest
from ...foundation.errors import ErrorCode, LinktoolsAIError

logger = environ.get_logger("ai.application.services.blob")


class BlobService:
    def __init__(self, repository: object, object_store: object) -> None:
        self._repository = repository
        self._object_store = object_store

    async def stage(self, blob: BlobObject, content: bytes) -> BlobObject:
        if blob.state is not BlobState.STAGING:
            raise LinktoolsAIError(ErrorCode.STORAGE_CONFLICT, "blob is not in staging state")
        if blob.size != len(content) or blob.digest != sha256_digest(content):
            raise LinktoolsAIError(ErrorCode.STORAGE_INTEGRITY_ERROR, "blob content digest or size does not match")
        await self._object_store.put(blob.object_key, content)
        logger.info("blob staged id=%s size=%s", blob.blob_id, blob.size)
        return await self._repository.create_staging(blob)

    async def commit_reference(self, blob_id: str, reference: object) -> object:
        committed = await self._repository.commit(blob_id)
        if not committed.can_reference():
            raise LinktoolsAIError(ErrorCode.STORAGE_CONFLICT, "blob is not committed")
        if isinstance(reference, BlobReference) and reference.tenant_id != committed.tenant_id:
            raise LinktoolsAIError(ErrorCode.AUTHORIZATION_DENIED, "blob reference tenant does not match")
        return await self._repository.add_reference(reference)

    async def delete_reference(self, reference_id: str) -> object:
        return await self._repository.delete_reference(reference_id)

    async def acquire_delete_lease(self, blob_id: str) -> object:
        return await self._repository.acquire_delete_lease(blob_id)

    async def complete_delete(self, blob_id: str, lease_id: str) -> object:
        blob = await self._repository.complete_delete(blob_id, lease_id)
        if blob.state is BlobState.DELETED:
            await self._object_store.delete(blob.object_key)
        return blob

    async def open(self, blob: BlobObject) -> bytes:
        return await self._object_store.get(blob.object_key)


__all__ = ["BlobService"]
