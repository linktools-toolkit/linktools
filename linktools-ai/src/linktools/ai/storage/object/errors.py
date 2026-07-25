#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""storage.object error hierarchy.

Backends raise these to signal storage-kernel outcomes (not-found, CAS
precondition failure, idempotency conflict, unsupported operation). They are
deliberately generic -- no domain vocabulary -- so the storage kernel stays
ignorant of the domains consuming it."""

from __future__ import annotations

from ...errors import LinktoolsAIError


class StorageObjectError(LinktoolsAIError):
    """Base for storage-object failures."""


class StorageObjectNotFoundError(StorageObjectError):
    """The key does not exist in the backend."""


class StoragePreconditionFailedError(StorageObjectError):
    """A CAS precondition (if_match / if_none_match) was not met."""


class StorageIdempotencyConflictError(StorageObjectError):
    """An idempotency key was replayed with DIFFERENT content (a conflict,
    not a silent overwrite)."""


class StorageTransactionNotSupportedError(StorageObjectError):
    """The backend does not support the requested (multi-object) transaction."""
