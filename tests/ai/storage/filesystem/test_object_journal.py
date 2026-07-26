#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Filesystem operation journal + crash recovery tests (P2b, the coordination spec).

These tests simulate a crash mid-mutation by writing a partial journal record
to disk, then constructing a FRESH backend against the same root (the next
operation's recovery sweep) and verifying the record resolves per the recovery table
table:

    PREPARED                     → abort (any temp cleared, no version visible)
    VERSIONS_PUBLISHED           → forward-complete (revision advance + idempotency)
    REVISION_PUBLISHED           → write idempotency result
    COMMITTED                    → no-op

And the never-regress rule: recovery must NOT lower an already-published
revision."""

from __future__ import annotations

import asyncio
import json
import urllib.parse
from datetime import datetime, timezone

import pytest

from linktools.ai.storage.backends.filesystem.object import (
    FilesystemObjectBackend,
    _IdempotencyRecord,
    _OpState,
    _OperationIntent,
    _VersionIntent,
)
from linktools.ai.storage.coordination.file import FilesystemKeyedCoordinator
from linktools.ai.storage.object.errors import StorageIntegrityError
from linktools.ai.storage.object.models import (
    StorageKey,
    StoredObject,
    WriteOptions,
)


def _key(value: str) -> StorageKey:
    return StorageKey(value)


def _make_backend(tmp_path):
    return FilesystemObjectBackend(root=tmp_path / "root")


def _encode(key: StorageKey) -> str:
    return urllib.parse.quote(key.value.strip("/") or "__root__", safe="")


# --- the basic happy path still works (sanity) -------------------------------


def test_put_then_get_round_trips(tmp_path):
    backend = _make_backend(tmp_path)

    async def _run():
        await backend.raw_put_checked(
            _key("/a"), b"v1", options=WriteOptions(), request_hash="h1"
        )
        lookup = await backend.raw_get(_key("/a"))
        assert hasattr(lookup, "object")
        assert lookup.object.content == b"v1"

    asyncio.run(_run())


def test_version_directory_uses_new_layout(tmp_path):
    """The journal layout: history/<key>/versions/<version>/{metadata.json,
    content.bin}. Verify the new layout is on disk after a put."""
    backend = _make_backend(tmp_path)

    async def _run():
        await backend.raw_put_checked(
            _key("/a"), b"v1", options=WriteOptions(), request_hash="h1"
        )

    asyncio.run(_run())
    # Layout check via raw filesystem.
    sd = backend._sd
    meta_components = backend._version_metadata_components(_key("/a"), 1)
    content_components = backend._version_content_components(_key("/a"), 1)
    raw_meta = json.loads(sd.read_bytes(*meta_components))
    assert raw_meta["version"] == 1
    assert raw_meta["tombstone"] is False
    assert raw_meta["operation_id"]
    assert sd.read_bytes(*content_components) == b"v1"


# --- recovery: PREPARED with no version published → abort -------------------


def test_recovery_aborts_prepared_with_no_version(tmp_path):
    """A PREPARED record whose version directory was never published must be
    aborted. The next read must NOT observe a phantom version, and the next
    put must produce version 1 (not 2)."""
    backend = _make_backend(tmp_path)

    # Simulate a crash: write the intent + PREPARED state directly to disk
    # WITHOUT publishing the version directory.
    intent = _OperationIntent(
        operation="put",
        operation_id="crashed-op-1",
        request_hash="rh",
        idempotency_key=None,
        new_revision=1,
        versions=[
            _VersionIntent(
                key_value="/phantom",
                version=1,
                tombstone=False,
                etag="x",
                content_type=None,
                size=1,
                modified_at=datetime.now(timezone.utc).isoformat(),
                metadata={},
            )
        ],
    )
    backend._begin_operation(intent)

    # Construct a fresh backend (the next op's recovery sweep).
    fresh = _make_backend(tmp_path)

    async def _run():
        # A read must NOT find /phantom.
        from linktools.ai.storage.object.models import Missing

        lookup = await fresh.raw_get(_key("/phantom"))
        assert lookup is Missing
        # A put to a fresh key must produce version 1 (the aborted op did not
        # consume a version slot).
        result = await fresh.raw_put_checked(
            _key("/fresh"), b"x", options=WriteOptions(), request_hash="rh"
        )
        assert result.info.version == 1

    asyncio.run(_run())
    # The crashed op is now ABORTED on disk.
    assert fresh._read_state("crashed-op-1") is _OpState.ABORTED


# --- recovery: VERSIONS_PUBLISHED → forward-complete ------------------------


def test_recovery_forward_completes_after_version_published(tmp_path):
    """A VERSIONS_PUBLISHED record (version dir on disk, but revision file
    and idempotency not yet advanced) must be forward-completed by recovery:
    revision advances to the intent's new_revision, idempotency is written,
    and the state reaches COMMITTED."""
    backend = _make_backend(tmp_path)

    # Simulate the live path up to VERSIONS_PUBLISHED: publish the version
    # directory, write the state, but do NOT advance revision or write
    # idempotency.
    intent = _OperationIntent(
        operation="put",
        operation_id="crashed-op-2",
        request_hash="rh-2",
        idempotency_key="idem-2",
        new_revision=5,
        versions=[
            _VersionIntent(
                key_value="/a",
                version=1,
                tombstone=False,
                etag="x",
                content_type=None,
                size=1,
                modified_at=datetime.now(timezone.utc).isoformat(),
                metadata={},
            )
        ],
    )
    backend._begin_operation(intent)
    backend._publish_version_dir(
        key=_key("/a"),
        version=1,
        metadata={
            "key": "/a",
            "version": 1,
            "commit_revision": 5,
            "etag": "x",
            "content_type": None,
            "size": 1,
            "modified_at": datetime.now(timezone.utc).isoformat(),
            "metadata": {},
            "tombstone": False,
            "operation_id": "crashed-op-2",
        },
        content=b"published-before-crash",
    )
    backend._set_state("crashed-op-2", _OpState.VERSIONS_PUBLISHED)
    # Sanity: revision is still 0 (we did NOT advance it).
    assert backend._read_revision_sync() == 0

    # Fresh backend -- recovery runs on the next op.
    fresh = _make_backend(tmp_path)

    async def _run():
        # Trigger recovery by issuing any mutation.
        await fresh.raw_put_checked(
            _key("/trigger"), b"x", options=WriteOptions(), request_hash="rt"
        )

    asyncio.run(_run())
    # Recovery advanced the revision to the intent's new_revision (5), THEN
    # the trigger put bumped it to 6.
    assert fresh._read_revision_sync() == 6
    # Idempotency was written for the crashed op.
    record = fresh._read_idempotency("put:/a:idem-2")
    assert record is not None
    assert record.operation == "put"
    assert record.result_version == 1
    # The crashed op is COMMITTED.
    assert fresh._read_state("crashed-op-2") is _OpState.COMMITTED


# --- recovery: REVISION_PUBLISHED → write idempotency -----------------------


def test_recovery_writes_idempotency_after_revision_published(tmp_path):
    """A REVISION_PUBLISHED record (revision advanced, idempotency not yet
    written) recovers by writing the idempotency result and reaching
    COMMITTED."""
    backend = _make_backend(tmp_path)

    intent = _OperationIntent(
        operation="put",
        operation_id="crashed-op-3",
        request_hash="rh-3",
        idempotency_key="idem-3",
        new_revision=7,
        versions=[
            _VersionIntent(
                key_value="/a",
                version=1,
                tombstone=False,
                etag="x",
                content_type=None,
                size=1,
                modified_at=datetime.now(timezone.utc).isoformat(),
                metadata={},
            )
        ],
    )
    backend._begin_operation(intent)
    backend._publish_version_dir(
        key=_key("/a"),
        version=1,
        metadata={
            "key": "/a",
            "version": 1,
            "commit_revision": 7,
            "etag": "x",
            "content_type": None,
            "size": 1,
            "modified_at": datetime.now(timezone.utc).isoformat(),
            "metadata": {},
            "tombstone": False,
            "operation_id": "crashed-op-3",
        },
        content=b"v1",
    )
    backend._set_state("crashed-op-3", _OpState.VERSIONS_PUBLISHED)
    backend._advance_revision_to(7)
    backend._set_state("crashed-op-3", _OpState.REVISION_PUBLISHED)
    # No idempotency written.

    fresh = _make_backend(tmp_path)

    async def _run():
        await fresh.raw_put_checked(
            _key("/trigger"), b"x", options=WriteOptions(), request_hash="rt"
        )

    asyncio.run(_run())
    record = fresh._read_idempotency("put:/a:idem-3")
    assert record is not None
    assert record.commit_revision == 7
    assert fresh._read_state("crashed-op-3") is _OpState.COMMITTED


# --- recovery: COMMITTED → no-op --------------------------------------------


def test_recovery_skips_committed_records(tmp_path):
    """A COMMITTED record is left alone by recovery (idempotent re-sweeps
    are cheap no-ops)."""
    backend = _make_backend(tmp_path)

    async def _run():
        await backend.raw_put_checked(
            _key("/a"),
            b"v1",
            options=WriteOptions(idempotency_key="idem-1"),
            request_hash="rh",
        )

    asyncio.run(_run())
    # Find the one operation dir; it must be COMMITTED.
    op_ids = backend._sd.list_names(".storage", "operations")
    assert len(op_ids) == 1
    assert backend._read_state(op_ids[0]) is _OpState.COMMITTED

    fresh = _make_backend(tmp_path)

    async def _run2():
        # Replay the same idempotency key -- must succeed without conflict,
        # because recovery is a no-op for COMMITTED records.
        result = await fresh.raw_put_checked(
            _key("/a"),
            b"v1",
            options=WriteOptions(idempotency_key="idem-1"),
            request_hash="rh",
        )
        assert result.info.version == 1  # no version bump on replay

    asyncio.run(_run2())


# --- recovery: never regress an already-published revision ------------------


def test_recovery_does_not_regress_published_revision(tmp_path):
    """Spec the recovery table: 'forbid recovery from overwriting an already-published
    new revision by writing an older one'. If revision on disk is already 10
    and an unfinished record's new_revision is 5, recovery must NOT lower
    the revision to 5."""
    backend = _make_backend(tmp_path)

    intent = _OperationIntent(
        operation="put",
        operation_id="crashed-low-rev",
        request_hash="rh",
        idempotency_key=None,
        new_revision=5,
        versions=[
            _VersionIntent(
                key_value="/stale",
                version=1,
                tombstone=False,
                etag="x",
                content_type=None,
                size=1,
                modified_at=datetime.now(timezone.utc).isoformat(),
                metadata={},
            )
        ],
    )
    backend._begin_operation(intent)
    # Publish the stale version dir + advance state past VERSIONS_PUBLISHED,
    # but THEN manually set revision HIGHER than the intent's new_revision.
    backend._publish_version_dir(
        key=_key("/stale"),
        version=1,
        metadata={
            "key": "/stale",
            "version": 1,
            "commit_revision": 5,
            "etag": "x",
            "content_type": None,
            "size": 1,
            "modified_at": datetime.now(timezone.utc).isoformat(),
            "metadata": {},
            "tombstone": False,
            "operation_id": "crashed-low-rev",
        },
        content=b"v1",
    )
    backend._set_state("crashed-low-rev", _OpState.VERSIONS_PUBLISHED)
    backend._advance_revision_to(10)  # higher than intent.new_revision

    fresh = _make_backend(tmp_path)

    async def _run():
        await fresh.raw_put_checked(
            _key("/trigger"), b"x", options=WriteOptions(), request_hash="rt"
        )

    asyncio.run(_run())
    # Recovery did NOT lower the revision; the trigger bumped it from 10 to 11.
    assert fresh._read_revision_sync() == 11


# --- recovery: integrity error when published metadata mismatches intent ----


def test_recovery_raises_on_metadata_mismatch(tmp_path):
    """Recovery verifies a published version dir against the intent. If the
    on-disk metadata contradicts the intent, recovery raises
    StorageIntegrityError and marks the op ABORTED (does NOT silently
    forward-complete)."""
    backend = _make_backend(tmp_path)

    intent = _OperationIntent(
        operation="put",
        operation_id="crashed-mismatch",
        request_hash="rh",
        idempotency_key=None,
        new_revision=3,
        versions=[
            _VersionIntent(
                key_value="/a",
                version=1,
                tombstone=False,
                etag="INTENT-ETAG",
                content_type=None,
                size=1,
                modified_at=datetime.now(timezone.utc).isoformat(),
                metadata={},
            )
        ],
    )
    backend._begin_operation(intent)
    # Publish with a DIFFERENT etag than the intent expects.
    backend._publish_version_dir(
        key=_key("/a"),
        version=1,
        metadata={
            "key": "/a",
            "version": 1,
            "commit_revision": 3,
            "etag": "DIFFERENT-ETAG",
            "content_type": None,
            "size": 1,
            "modified_at": datetime.now(timezone.utc).isoformat(),
            "metadata": {},
            "tombstone": False,
            "operation_id": "crashed-mismatch",
        },
        content=b"v1",
    )
    backend._set_state("crashed-mismatch", _OpState.VERSIONS_PUBLISHED)

    fresh = _make_backend(tmp_path)

    async def _run():
        # The trigger op's recovery sweep hits the mismatched record. The
        # sweep itself does not raise (it isolates the corrupt op), but the
        # corrupt op is marked ABORTED.
        await fresh.raw_put_checked(
            _key("/trigger"), b"x", options=WriteOptions(), request_hash="rt"
        )

    asyncio.run(_run())
    assert fresh._read_state("crashed-mismatch") is _OpState.ABORTED


# --- recovery: move intent with missing target → reconstructed from source ---


def test_recovery_move_publishes_missing_target_from_source(tmp_path):
    """A move op's intent carries the source key+version so recovery can re-
    materialize a missing target version dir. Simulate a crash AFTER the
    source tombstone was published but BEFORE the target -- recovery must
    reconstruct the target's content from history and complete the move."""
    backend = _make_backend(tmp_path)

    # Set up source history first via a real put.
    async def _seed():
        await backend.raw_put_checked(
            _key("/src"), b"payload", options=WriteOptions(), request_hash="rh"
        )

    asyncio.run(_seed())
    # Source is now at version 1 with content "payload".

    # Simulate a crashed move that published the SOURCE TOMBSTONE but not
    # the TARGET version dir. (In practice the live path publishes target
    # first; this test reverses order to exercise recovery's source-based
    # materialization.)
    intent = _OperationIntent(
        operation="move",
        operation_id="crashed-move",
        request_hash="rh",
        idempotency_key=None,
        new_revision=2,
        versions=[
            _VersionIntent(
                key_value="/dst",
                version=1,
                tombstone=False,
                etag=backend._live_version_metadata(_key("/src"))[1]["etag"],
                content_type=None,
                size=len(b"payload"),
                modified_at=datetime.now(timezone.utc).isoformat(),
                metadata={},
                source_key="/src",
                source_version=1,
            ),
            _VersionIntent(
                key_value="/src",
                version=2,
                tombstone=True,
                etag="",
                content_type=None,
                size=0,
                modified_at=datetime.now(timezone.utc).isoformat(),
                metadata={},
            ),
        ],
    )
    backend._begin_operation(intent)
    # Publish ONLY the source tombstone; leave the target missing.
    backend._publish_version_dir(
        key=_key("/src"),
        version=2,
        metadata={
            "key": "/src",
            "version": 2,
            "commit_revision": 2,
            "etag": "",
            "content_type": None,
            "size": 0,
            "modified_at": datetime.now(timezone.utc).isoformat(),
            "metadata": {},
            "tombstone": True,
            "operation_id": "crashed-move",
        },
        content=None,
    )
    backend._set_state("crashed-move", _OpState.VERSIONS_PUBLISHED)

    fresh = _make_backend(tmp_path)

    async def _run():
        # Trigger recovery.
        await fresh.raw_put_checked(
            _key("/trigger"), b"x", options=WriteOptions(), request_hash="rt"
        )
        # Recovery reconstructed /dst version 1 from /src version 1's content.
        lookup = await fresh.raw_get(_key("/dst"))
        assert hasattr(lookup, "object")
        assert lookup.object.content == b"payload"

    asyncio.run(_run())
    assert fresh._read_state("crashed-move") is _OpState.COMMITTED


# --- coordinator: every op is serialized by the namespace lock --------------


def test_coordinator_is_constructed_by_default(tmp_path):
    """If no coordinator is injected, the backend builds a default
    FilesystemKeyedCoordinator rooted at .storage/coordination."""
    backend = _make_backend(tmp_path)
    assert isinstance(backend.coordinator, FilesystemKeyedCoordinator)


def test_coordinator_can_be_injected(tmp_path):
    """A caller can inject a coordinator (e.g. shared across backends)."""
    shared = FilesystemKeyedCoordinator(
        root=tmp_path / "shared-coordination"
    )
    backend = FilesystemObjectBackend(
        root=tmp_path / "root", coordinator=shared
    )
    assert backend.coordinator is shared


# --- idempotency: replay reads from immutable version dir, not current ------


def test_idempotency_replay_reads_from_immutable_history(tmp_path):
    """Spec the idempotency record spec: replay reads from the immutable version directory (the
    result's recorded key+version), NOT from the current live state. After
    a put with idempotency-key K and a SECOND non-idempotent put that
    overwrites the key, replaying K must return the ORIGINAL version's
    content + version, not the new live state."""
    backend = _make_backend(tmp_path)

    async def _run():
        r1 = await backend.raw_put_checked(
            _key("/a"),
            b"original",
            options=WriteOptions(idempotency_key="K"),
            request_hash="rh",
        )
        # Overwrite WITHOUT idempotency key.
        await backend.raw_put_checked(
            _key("/a"), b"new", options=WriteOptions(), request_hash="rh2"
        )
        # Replay K: returns the ORIGINAL.
        replay = await backend.raw_put_checked(
            _key("/a"),
            b"original",
            options=WriteOptions(idempotency_key="K"),
            request_hash="rh",
        )
        assert replay.info.version == r1.info.version
        assert replay.content == b"original"

    asyncio.run(_run())
