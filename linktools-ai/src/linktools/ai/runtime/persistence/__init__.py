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
- ``SqlAlchemyStorage`` -- loaded lazily on access (SQLAlchemy is an optional
  dependency).
- ``StorageFeatures``, ``FILE_STORAGE_FEATURES``, ``SQLALCHEMY_STORAGE_FEATURES``
  -- runtime composition of the capability enums.
- ``StorageUnitOfWork``, ``StorageTransactionManager`` -- the cross-store UoW
  Protocol.
"""

__all__: "list[str]" = [
    "Storage",
    "FilesystemStorage",
    "SqlAlchemyStorage",
    "StorageFeatures",
    "FILE_STORAGE_FEATURES",
    "SQLALCHEMY_STORAGE_FEATURES",
    "StorageUnitOfWork",
    "StorageTransactionManager",
]


def __getattr__(name: str):
    if name in {"Storage", "FilesystemStorage", "FILE_STORAGE_FEATURES"}:
        from .facade import FILE_STORAGE_FEATURES, FilesystemStorage, Storage

        return {
            "Storage": Storage,
            "FilesystemStorage": FilesystemStorage,
            "FILE_STORAGE_FEATURES": FILE_STORAGE_FEATURES,
        }[name]
    if name in {"StorageFeatures", "SQLALCHEMY_STORAGE_FEATURES"}:
        from .features import SQLALCHEMY_STORAGE_FEATURES, StorageFeatures

        return {
            "StorageFeatures": StorageFeatures,
            "SQLALCHEMY_STORAGE_FEATURES": SQLALCHEMY_STORAGE_FEATURES,
        }[name]
    if name in {"StorageUnitOfWork", "StorageTransactionManager"}:
        from .protocols import StorageTransactionManager, StorageUnitOfWork

        return {
            "StorageUnitOfWork": StorageUnitOfWork,
            "StorageTransactionManager": StorageTransactionManager,
        }[name]
    if name == "SqlAlchemyStorage":
        try:
            from .sqlalchemy import SqlAlchemyStorage
        except ModuleNotFoundError as exc:
            if exc.name and exc.name.split(".")[0] in {"sqlalchemy", "aiosqlite"}:
                raise ImportError(
                    "SqlAlchemyStorage requires optional dependencies. "
                    "Install with: pip install 'linktools-ai[sqlite]' "
                    "or pip install 'linktools-ai[sqlalchemy]'."
                ) from exc
            raise
        return SqlAlchemyStorage
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
