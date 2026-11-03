#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Read clean legacy filesystem state without initializing a state writer."""

import hashlib
from collections.abc import AsyncIterator
from pathlib import Path

from ..core import canonical_sha256, validate_persistence_namespace, validate_tenant_id
from ..errors import AIError, ErrorCode
from ..storage import FilesystemObjectStore, ObjectRef, ObjectStore, namespace_digest
from ._filesystem_reader import _FilesystemOfflineSource, _read_owned
from ._filesystem_layout import _FilesystemCache
from ._plan import PersistenceDomain
from .snapshots import _SnapshotArchive, _SnapshotExportSource, _export_snapshot
from ._snapshot import SnapshotLimits
from .state import FactScanCursor, OperationScanCursor, RecordScanCursor, StoredFact, StoredOperation, StoredRecord

_LEGACY_DOMAINS = frozenset({
    "conversation", "execution", "memory", "artifact", "task", "evaluation",
    "recovery", "trace", "checkpoint",
})


class _FilesystemSnapshotPartition:
    def __init__(self, root: Path) -> None:
        self._cache = _FilesystemCache(root)

    async def get_operation(self, key: bytes) -> StoredOperation | None:
        return await _read_owned(lambda: self._cache.get_operation(key))

    async def scan_records_page(self, *, after: RecordScanCursor | None, limit: int) -> tuple[StoredRecord, ...]:
        return await _read_owned(lambda: self._cache.scan_records_page(after=after, limit=limit))

    async def scan_facts_page(self, *, after: FactScanCursor | None, limit: int) -> tuple[StoredFact, ...]:
        return await _read_owned(lambda: self._cache.scan_facts_page(after=after, limit=limit))

    async def scan_operations_page(self, *, after: OperationScanCursor | None, limit: int) -> tuple[StoredOperation, ...]:
        return await _read_owned(lambda: self._cache.scan_operations_page(after=after, limit=limit))


class FilesystemSnapshotReader:
    """Read clean legacy filesystem state into a portable storage snapshot.

    This offline source owner accepts the standard grouped filesystem layout,
    not Runtime routes or snapshot archives. It never repairs committed journals
    or interprets opaque capability partitions. Source objects are borrowed.
    Keep this instance until ``close`` succeeds so failed lock cleanup can be
    retried. Construction does not open or create source files.
    """

    def __init__(
        self, root: str | Path, *, namespace: str, tenant_id: str = "default",
        source_object_store: ObjectStore | None = None,
    ) -> None:
        validate_persistence_namespace(namespace)
        validate_tenant_id(tenant_id)
        self._closed = False
        self._root = Path(root).expanduser().resolve()
        self._namespace = namespace
        self._tenant_id = tenant_id
        self._source_object_store = source_object_store
        self._scope = canonical_sha256({"namespace": namespace, "tenant_id": tenant_id})[:32]
        self._group = _FilesystemOfflineSource(
            self._root, namespace=namespace, tenant_id=tenant_id,
            scope_digest=self._scope,
        )

    async def close(self) -> None:
        """Release retained offline locks; retry this instance if cleanup fails."""
        if not self._closed:
            await self._group.close()
            self._closed = True

    async def export_snapshot(self, *, object_store: ObjectStore, limits: SnapshotLimits) -> ObjectRef:
        if self._closed:
            raise AIError(ErrorCode.STORAGE_CLOSED)
        if not isinstance(limits, SnapshotLimits):
            raise TypeError("limits must be SnapshotLimits")
        async with self._group.offline_exclusivity():
            inventory = self._group.describe()
            members = inventory.members
            if any(member.runtime_domain not in _LEGACY_DOMAINS | {"group_archive"}
                   or member.partition_key is not None for member in members):
                raise AIError(ErrorCode.SNAPSHOT_UNSUPPORTED, "Opaque capability partitions require their original migration owner")
            roots = {member.runtime_domain: member.root for member in members}
            if not _LEGACY_DOMAINS.issubset(roots):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            suffix = Path(namespace_digest(self._namespace)) / hashlib.sha256(("tenant:" + self._tenant_id).encode()).hexdigest()
            for name in _LEGACY_DOMAINS:
                if roots[name] != self._root / name / suffix:
                    raise AIError(ErrorCode.SNAPSHOT_UNSUPPORTED, "The legacy source is not a standard grouped filesystem root")
            recovery = _FilesystemSnapshotPartition(roots["recovery"])
            if (await recovery.scan_records_page(after=None, limit=1)
                    or await recovery.scan_facts_page(after=None, limit=1)
                    or await recovery.scan_operations_page(after=None, limit=1)):
                raise AIError(ErrorCode.SNAPSHOT_UNSUPPORTED,
                              "Retired recovery data requires its original reader")
            readers = {domain: _FilesystemSnapshotPartition(roots[domain.value]) for domain in PersistenceDomain}
            source_stores = {}
            for domain in PersistenceDomain:
                route_root = (self._root if domain in {
                    PersistenceDomain.EXECUTION, PersistenceDomain.TRACE, PersistenceDomain.CHECKPOINT,
                } else self._root / domain.value)
                source_stores[domain] = (self._source_object_store if self._source_object_store is not None
                                         else FilesystemObjectStore(route_root / "objects"))
            archive_store = (self._source_object_store if self._source_object_store is not None
                             else FilesystemObjectStore(self._root / ".state-archive-objects"))

            async def archives() -> AsyncIterator[_SnapshotArchive]:
                archive_root = roots.get("group_archive")
                if archive_root is not None:
                    assert inventory.archive_scope is not None
                    yield _SnapshotArchive(inventory.archive_scope,
                                           min(PersistenceDomain, key=lambda value: value.value),
                                           _FilesystemSnapshotPartition(archive_root))

            source = _SnapshotExportSource(
                namespace=self._namespace, tenant_id=self._tenant_id, readers=readers,
                archives=archives(), object_stores=source_stores,
                archive_object_stores={domain: archive_store for domain in PersistenceDomain},
            )
            return await _export_snapshot(source, object_store=object_store, limits=limits)


__all__ = ["FilesystemSnapshotReader"]
