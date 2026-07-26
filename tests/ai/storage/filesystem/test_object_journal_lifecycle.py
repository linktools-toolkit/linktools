#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Filesystem Object journal lifecycle contract.

The active operations directory is a recovery queue: COMMITTED/ABORTED
operations are removed once durable, FAILED/CORRUPT operations are retained
and the backend goes fail-closed, and the steady-state directory is empty so
a read's recovery sweep does not scan historical journals."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from hashlib import sha256

import pytest

from linktools.ai.storage.backends.filesystem.object import (
    FilesystemObjectBackend,
    _OpState,
    _OperationIntent,
    _VersionIntent,
)
from linktools.ai.storage.backends.filesystem.secure_directory import (
    FilesystemSecurityMode,
)
from linktools.ai.storage.object.errors import StorageRecoveryError
from linktools.ai.storage.object.models import StorageKey, WriteOptions


def _key(v: str) -> StorageKey:
    return StorageKey(v)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _backend(tmp_path, *, mode=FilesystemSecurityMode.SECURE_POSIX):
    return FilesystemObjectBackend(root=tmp_path / "root", mode=mode)


def _active(backend) -> "tuple[str, ...]":
    return backend._sd.list_names(".storage", "operations")


def _seed_committed(backend, op_id: str, *, state=_OpState.COMMITTED) -> None:
    """Write a fully-formed intent + state on disk, emulating a crashed op."""
    intent = _OperationIntent(
        operation="put",
        operation_id=op_id,
        request_hash="rh",
        idempotency_key=None,
        new_revision=1,
        versions=[
            _VersionIntent(
                key_value="/a",
                version=1,
                tombstone=False,
                etag="x",
                content_type=None,
                size=1,
                modified_at=_now(),
                metadata={},
                commit_revision=1,
                content_sha256=sha256(b"v1").hexdigest(),
                operation_id=op_id,
            )
        ],
    )
    backend._sd.ensure_directory(*backend._operation_dir_components(op_id))
    backend._sd.atomic_write(
        *backend._operation_intent_components(op_id),
        content=intent.to_json(),
    )
    backend._set_state(op_id, state)


# --- normal journal cleanup ---------------------------------------------------


def test_committed_operation_removed_from_active_directory(tmp_path):
    """A normal put/delete/move removes its operation directory on commit."""
    backend = _backend(tmp_path)

    async def _run():
        await backend.raw_put_checked(
            _key("/a"), b"v1", options=WriteOptions(), request_hash="h"
        )
        await backend.raw_put_checked(
            _key("/b"), b"v2", options=WriteOptions(), request_hash="h2"
        )
        await backend.raw_delete_checked(
            _key("/a"), options=WriteOptions(), request_hash="h3"
        )

    asyncio.run(_run())
    assert _active(backend) == ()


def test_aborted_operation_removed_from_active_directory(tmp_path):
    """A PREPARED op that never published a version is aborted AND its
    directory removed by the next recovery sweep."""
    backend = _backend(tmp_path)
    _seed_committed(backend, "aborted-op", state=_OpState.PREPARED)
    assert "aborted-op" in _active(backend)

    fresh = _backend(tmp_path)

    async def _run():
        await fresh.raw_put_checked(
            _key("/trigger"), b"x", options=WriteOptions(), request_hash="rt"
        )

    asyncio.run(_run())
    assert _active(fresh) == ()


# --- recovery journal cleanup -------------------------------------------------


def test_committed_recovery_entry_removed(tmp_path):
    """A COMMITTED entry left on disk (crash after the COMMITTED marker, before
    directory removal) is removed by the next recovery sweep."""
    backend = _backend(tmp_path)
    _seed_committed(backend, "leftover-committed", state=_OpState.COMMITTED)

    fresh = _backend(tmp_path)

    async def _run():
        await fresh.raw_stat(_key("/anything"))

    asyncio.run(_run())
    assert _active(fresh) == ()


def test_aborted_recovery_entry_removed(tmp_path):
    """An ABORTED entry left on disk is removed by the next recovery sweep."""
    backend = _backend(tmp_path)
    _seed_committed(backend, "leftover-aborted", state=_OpState.ABORTED)

    fresh = _backend(tmp_path)

    async def _run():
        await fresh.raw_stat(_key("/anything"))

    asyncio.run(_run())
    assert _active(fresh) == ()


# --- failed/corrupt journal retained + fail-closed -----------------------------


def test_failed_recovery_entry_retained(tmp_path):
    """A corrupt intent (invalid JSON) is retained on disk and the backend
    enters fail-closed state: subsequent operations re-raise."""
    backend = _backend(tmp_path)
    op_id = "corrupt-op"
    backend._sd.ensure_directory(*backend._operation_dir_components(op_id))
    backend._sd.atomic_write(
        *backend._operation_intent_components(op_id),
        content=b"{not valid json",
    )
    backend._set_state(op_id, _OpState.PREPARED)

    fresh = _backend(tmp_path)

    async def _run():
        with pytest.raises(StorageRecoveryError):
            await fresh.raw_stat(_key("/anything"))

    asyncio.run(_run())
    # Corrupt entry retained.
    assert op_id in _active(fresh)
    # Backend stays fail-closed on subsequent ops.
    with pytest.raises(StorageRecoveryError):
        asyncio.run(fresh.raw_get(_key("/x")))


def test_cleanup_failure_fails_closed(tmp_path):
    """If the directory cleanup itself fails, the backend goes fail-closed
    and the operation directory is retained (not silently leaked)."""
    backend = _backend(tmp_path)
    _seed_committed(backend, "cleanup-target", state=_OpState.COMMITTED)

    # Inject a cleanup failure: remove_tree raises.
    original = backend._sd.remove_tree

    def _boom(*components: str) -> None:
        raise OSError("simulated cleanup failure")

    fresh = _backend(tmp_path)
    fresh._sd.remove_tree = _boom  # type: ignore[assignment]

    async def _run():
        with pytest.raises(StorageRecoveryError):
            await fresh.raw_stat(_key("/anything"))

    asyncio.run(_run())
    # Directory retained because cleanup could not complete.
    assert "cleanup-target" in _active(fresh)
    # Restore for sanity (other backends unaffected).
    backend._sd.remove_tree = original  # type: ignore[assignment]


# --- no historical scan growth --------------------------------------------------


def test_stable_read_scans_empty_active_directory(tmp_path):
    """After a completed mutation, a read triggers a recovery sweep over an
    EMPTY active directory and returns the live state without error."""
    backend = _backend(tmp_path)

    async def _seed():
        await backend.raw_put_checked(
            _key("/a"), b"v1", options=WriteOptions(), request_hash="h"
        )

    asyncio.run(_seed())
    assert _active(backend) == ()

    async def _read():
        info = await backend.raw_stat(_key("/a"))
        assert info is not None
        assert info.key == _key("/a")

    asyncio.run(_read())
    assert _active(backend) == ()


def test_100k_completed_mutations_leave_no_active_journals(tmp_path):
    """100,000 completed mutations must leave zero active operation
    directories. Uses TRUSTED_LOCAL mode (no per-op fsync) so the volume is
    exercisable in-test; the lifecycle contract (dir removed on commit) is
    identical across modes."""
    backend = _backend(tmp_path, mode=FilesystemSecurityMode.TRUSTED_LOCAL)
    total = 100_000

    async def _run():
        for i in range(total):
            await backend.raw_put_checked(
                _key(f"/k/{i}"),
                b"x",
                options=WriteOptions(),
                request_hash=f"h{i}",
            )

    asyncio.run(_run())
    # The assertion the spec demands: zero active journals regardless of how
    # many mutations ran.
    assert _active(backend) == ()


# --- intent strong validation ---------------------------------------------------


def test_intent_operation_id_mismatch_fails_closed(tmp_path):
    """An intent whose operation_id differs from its directory name is
    inconsistent; recovery fails closed and retains the directory."""
    backend = _backend(tmp_path)
    op_id = "dir-name"
    intent = _OperationIntent(
        operation="put",
        operation_id="different-name",
        request_hash="rh",
        idempotency_key=None,
        new_revision=1,
        versions=[
            _VersionIntent(
                key_value="/a",
                version=1,
                tombstone=False,
                etag="x",
                content_type=None,
                size=1,
                modified_at=_now(),
                metadata={},
                commit_revision=1,
                content_sha256=sha256(b"v1").hexdigest(),
                operation_id="different-name",
            )
        ],
    )
    backend._sd.ensure_directory(*backend._operation_dir_components(op_id))
    backend._sd.atomic_write(
        *backend._operation_intent_components(op_id),
        content=intent.to_json(),
    )
    backend._set_state(op_id, _OpState.PREPARED)

    fresh = _backend(tmp_path)

    async def _run():
        with pytest.raises(StorageRecoveryError):
            await fresh.raw_stat(_key("/anything"))

    asyncio.run(_run())
    assert op_id in _active(fresh)


def test_move_intent_shape_validation(tmp_path):
    """Move intent must be exactly one non-tombstone target + one source
    tombstone; any other shape is rejected by from_json."""
    base = dict(
        operation="move",
        operation_id="op-1",
        request_hash="rh",
        idempotency_key=None,
        new_revision=2,
    )
    target = dict(
        key_value="/dst",
        version=1,
        tombstone=False,
        etag="x",
        content_type=None,
        size=1,
        modified_at=_now(),
        metadata={},
        commit_revision=2,
        content_sha256="x",
        operation_id="op-1",
    )
    tombstone = dict(
        key_value="/src",
        version=2,
        tombstone=True,
        etag="",
        content_type=None,
        size=0,
        modified_at=_now(),
        metadata={},
        commit_revision=2,
        operation_id="op-1",
    )

    # Two tombstones (no target) -> rejected.
    with pytest.raises(StorageRecoveryError):
        _OperationIntent.from_json(
            json.dumps({**base, "versions": [tombstone, dict(tombstone)]}).encode()
        )
    # Two non-tombstones (no source tombstone) -> rejected.
    with pytest.raises(StorageRecoveryError):
        _OperationIntent.from_json(
            json.dumps({**base, "versions": [target, dict(target)]}).encode()
        )
    # Three versions -> rejected.
    with pytest.raises(StorageRecoveryError):
        _OperationIntent.from_json(
            json.dumps(
                {**base, "versions": [target, tombstone, dict(tombstone)]}
            ).encode()
        )
    # One valid target + one valid tombstone -> accepted.
    intent = _OperationIntent.from_json(
        json.dumps({**base, "versions": [target, tombstone]}).encode()
    )
    assert intent.operation == "move"
    assert len(intent.versions) == 2
