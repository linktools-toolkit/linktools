#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"SQLite blob persistence adapter boundary."


class BlobStore:
    def __init__(self, operation: object) -> None:
        self._operation = operation

    async def create_staging(self, blob: object) -> object:
        return await self._operation.create_staging(blob)

    async def commit(self, blob_id: str) -> object:
        return await self._operation.commit(blob_id)

    async def add_reference(self, reference: object) -> object:
        return await self._operation.add_reference(reference)

    async def delete_reference(self, reference_id: str) -> object:
        return await self._operation.delete_reference(reference_id)

    async def acquire_delete_lease(self, blob_id: str) -> object:
        return await self._operation.acquire_delete_lease(blob_id)

    async def complete_delete(self, blob_id: str, lease_id: str) -> object:
        return await self._operation.complete_delete(blob_id, lease_id)


__all__ = ["BlobStore"]
