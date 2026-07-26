#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime persistence: the composition root that wires each domain's
persistence/ adapters into a single ``Storage`` (+ the SqlAlchemy/Sqlite
convenience subclasses) plus the high-level StorageUnitOfWork Protocol.

This package lives at the runtime layer (NOT under storage/) so the storage
kernel stays free of any domain import: ``import linktools.ai.storage`` pulls
only the storage-kernel (object / cache / blob / coordination / backends) and
never any domain package. Each domain's own persistence/ subpackage owns its
backend adapters; this package composes them.

Public symbols (mirroring the pre-Phase-7 ``linktools.ai.storage`` surface):
- ``Storage``, ``FilesystemStorage`` -- the core composition + FS reference.
- SQLAlchemy composition -- loaded lazily on access (SQLAlchemy is optional
  dependency).
- ``StorageFeatures`` -- runtime composition of the capability enums.
- ``StorageUnitOfWork``, ``StorageTransactionManager`` -- the cross-store UoW
  Protocol.
"""

__all__: "list[str]" = [
    "Storage",
    "FilesystemStorage",
    "SqlAlchemyStorageAdapter",
    "StorageFeatures",
    "StorageUnitOfWork",
    "StorageTransactionManager",
]


def __getattr__(name: str):
    if name in {"Storage", "FilesystemStorage"}:
        from .facade import FilesystemStorage, Storage

        return {
            "Storage": Storage,
            "FilesystemStorage": FilesystemStorage,
        }[name]
    if name == "StorageFeatures":
        from .features import StorageFeatures

        return {
            "StorageFeatures": StorageFeatures,
        }[name]
    if name in {"StorageUnitOfWork", "StorageTransactionManager"}:
        from .protocols import StorageTransactionManager, StorageUnitOfWork

        return {
            "StorageUnitOfWork": StorageUnitOfWork,
            "StorageTransactionManager": StorageTransactionManager,
        }[name]
    if name == "SqlAlchemyStorageAdapter":
        try:
            from .sqlalchemy import SqlAlchemyStorageAdapter
        except ModuleNotFoundError as exc:
            if exc.name and exc.name.split(".")[0] in {"sqlalchemy", "aiosqlite"}:
                raise ImportError(
                    "SQLAlchemy storage requires optional dependencies. "
                    "Install with: pip install 'linktools-ai[sqlite]' "
                    "or pip install 'linktools-ai[sqlalchemy]'."
                ) from exc
            raise
        return SqlAlchemyStorageAdapter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
