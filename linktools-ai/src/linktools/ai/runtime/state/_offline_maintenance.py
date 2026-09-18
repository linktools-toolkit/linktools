#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Offline destructive Runtime storage maintenance."""

from contextlib import AbstractAsyncContextManager, AsyncExitStack
from typing import Protocol

from ...errors import AIError, ErrorCode
from ...storage import ObjectStoreMaintenance


class OfflineExclusiveStorage(Protocol):
    def offline_exclusivity(self) -> AbstractAsyncContextManager[None]: ...


class _RuntimeStorageInspection(Protocol):
    def object_maintenance_stores(self) -> tuple[ObjectStoreMaintenance, ...]: ...

    async def _compact_objects(self) -> int: ...


class OfflineRuntimeStorageMaintenance:
    """Run destructive object collection under an explicit exclusive guard."""

    def __init__(
        self,
        inspection: _RuntimeStorageInspection,
        exclusive_guard: OfflineExclusiveStorage | None = None,
    ) -> None:
        self._inspection = inspection
        self._exclusive_guard = exclusive_guard

    async def compact_objects(self) -> int:
        if self._exclusive_guard is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        async with self._exclusive_guard.offline_exclusivity():
            async with AsyncExitStack() as stack:
                for object_store in self._inspection.object_maintenance_stores():
                    await stack.enter_async_context(
                        object_store.offline_exclusivity()
                    )
                return await self._inspection._compact_objects()


__all__ = ["OfflineExclusiveStorage", "OfflineRuntimeStorageMaintenance"]
