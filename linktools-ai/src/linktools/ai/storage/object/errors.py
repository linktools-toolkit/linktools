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


class StorageIntegrityError(StorageObjectError):
    """On-disk state contradicts what the operation expected (e.g. a journal
    recovery found a published version dir whose metadata does not match the
    intent, or an idempotency record points at a version that is missing).
    Raised by the filesystem backend's crash-recovery path so a corruption
    is surfaced instead of silently swallowed or papered over with stale
    current-state reads."""


class StorageHashCollisionError(StorageObjectError):
    """A key_hash (SHA-256) lookup returned a row whose plaintext key does NOT
    match the queried key. SHA-256 collisions are astronomically unlikely, so a
    real collision signals either a hash function bug, a corrupted index, or an
    adversary with a broken hash. Surfaces as a distinct error (not NotFound)
    so the caller cannot accidentally read / overwrite the WRONG object -- the
    digest in the message is the colliding digest, never the plaintext key."""

    def __init__(self, *, namespace: str, digest: str) -> None:
        self.namespace = namespace
        self.digest = digest
        super().__init__(
            f"hash collision in {namespace!r}: digest {digest!r} matched a row "
            f"with a different plaintext key; refusing to read or overwrite the "
            f"wrong object"
        )


class StorageTransactionNotSupportedError(StorageObjectError):
    """The backend does not support the requested (multi-object) transaction."""


class StorageTransactionClosedError(StorageObjectError):
    """A transaction-bound child backend was used AFTER its context manager
    exited. The child owns transaction-local state that is no longer valid;
    callers must issue all reads/writes inside the ``async with`` block."""


class StorageRecoveryError(StorageObjectError):
    """Recovery could not resolve an operation journal entry, or a journal
    cleanup failed. The backend enters a fail-closed state: subsequent
    operations re-raise this error rather than serving potentially-inconsistent
    state. A corrupt or unresolvable operation directory is RETAINED on disk
    so it can be inspected, never silently dropped."""
