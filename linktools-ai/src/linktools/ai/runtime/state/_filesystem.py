#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Filesystem-backed RuntimeState repository composition."""

from typing import TYPE_CHECKING

from ._memory import FilesystemRuntime as _MemoryFilesystemRuntime, build_filesystem_runtime as _build_filesystem_runtime

if TYPE_CHECKING:
    from ._contracts import RuntimeDomain
    from ...storage import FilesystemWriterLock


class FilesystemRuntime(_MemoryFilesystemRuntime):
    """Durable filesystem runtime owned by the RuntimeState subsystem."""


def build_filesystem_runtime(
    root: str,
    *,
    namespace: str,
    tenant_id: str = "",
    persist: "frozenset[RuntimeDomain] | RuntimeDomain | None" = None,
    writer_lock: "FilesystemWriterLock | None" = None,
) -> FilesystemRuntime:
    """Build a filesystem runtime with state-owned persistence resources."""
    return _build_filesystem_runtime(
        root,
        namespace=namespace,
        tenant_id=tenant_id,
        persist=persist,
        writer_lock=writer_lock,
        runtime_cls=FilesystemRuntime,
    )


__all__ = ["FilesystemRuntime", "build_filesystem_runtime"]
