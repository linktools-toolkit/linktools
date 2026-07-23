#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Blob orphan sweeper -- the core orphan contract + the backend-agnostic sweep
implementation.

A blob becomes an orphan candidate when it is written but the record
transaction that would pin it fails (or the record is later deleted). The core
contract: an unreferenced blob is only
deletable once it is past a safety window (default 24h); the sweep runs no more
frequently than the sweep interval (default 6h). The window protects a blob
written by a transaction that has not yet committed its record -- deleting it
eagerly would corrupt a still-in-flight put.

The contract config (:class:`OrphanSweepConfig`) is core business policy and
lives here; the enumeration + delete is adapter-specific (an external object
store sweeps its own way). This module provides the backend-agnostic sweep that
walks the blob and record trees via the orphan-enumeration extension
(``iter_digests_with_mtime`` / ``iter_referenced_digests``) that the in-repo
Filesystem + SqlAlchemy backends provide. The base ArtifactBlobStore /
ArtifactRecordStore Protocols are intentionally minimal (not every backend can
list its blobs); a backend is sweepable only if it implements that extension.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..artifact.coordination import ArtifactDigestCoordinator
    from ..observability.metrics import ObservabilityMetrics
    from .protocols import ArtifactBlobStore, ArtifactRecordStore

from ..errors import ArtifactRecordCorruptError


@dataclass(frozen=True, slots=True)
class OrphanSweepConfig:
    """Core orphan policy. Defaults: a 24h grace window before
    an unreferenced blob is deletable, swept at most every 6h.

    ``grace_period`` is enforced by :func:`sweep_orphan_blobs` (a blob inside
    the window is never deleted). ``sweep_interval`` is NOT enforced by the
    sweep function -- it is the upper bound on how often a SCHEDULER calls the
    sweep, and is declared here so the policy lives in one place. A caller that
    runs the sweep on a timer reads ``sweep_interval`` to size that timer.
    """

    grace_period: timedelta = timedelta(hours=24)
    sweep_interval: timedelta = timedelta(hours=6)


@dataclass(frozen=True, slots=True)
class OrphanSweepStats:
    """Outcome of one sweep pass, for observability."""

    deleted: int
    kept_within_grace: int
    in_use: int


async def sweep_orphan_blobs(
    blob_store: "ArtifactBlobStore",
    record_store: "ArtifactRecordStore",
    coordinator: "ArtifactDigestCoordinator",
    config: "OrphanSweepConfig | None" = None,
    *,
    now: "datetime | None" = None,
    metrics: "ObservabilityMetrics | None" = None,
) -> OrphanSweepStats:
    """Delete blobs that no record references AND that are past the grace
    window, coordinating per digest against in-flight puts.

    The scan builds candidates (blobs past the grace window), but the FINAL
    delete decision is made UNDER the per-digest lock: re-stat the blob (another
    sweeper may have removed it since the scan), then re-check
    ``record_store.is_digest_referenced`` against the LIVE reference set -- NOT
    the snapshot taken at scan start. A put that pins this blob while the sweeper
    waited for the lock creates its record under the same lock, so the re-check
    sees the pin and the blob is kept. (Re-checking age under the lock adds no
    signal here: ``put_if_absent`` is idempotent, so a dedup re-pin does not
    advance the blob mtime -- ``is_digest_referenced`` is the definitive pin
    check.)

    Returns counts of each disposition so a caller can observe sweep progress.
    Idempotent: re-running with the same state deletes nothing more.

    When ``metrics`` is wired, each deleted orphan increments
    ``artifact_orphan_total`` and each delete-side failure increments
    ``artifact_orphan_cleanup_failure_total``. The sweeper keeps going on a
    single delete failure so one bad blob cannot stall the sweep; the failure
    is counted for observability instead of being swallowed silently.
    """
    from datetime import timezone

    cfg = config or OrphanSweepConfig()
    moment = now if now is not None else datetime.now(timezone.utc)

    deleted = 0
    kept_within_grace = 0
    in_use = 0
    async for digest, modified_at in blob_store.iter_digests_with_mtime():
        age = moment - modified_at
        if age < cfg.grace_period:
            # Inside the safety window regardless of reference state: an
            # in-flight transaction may still commit a record pinning it.
            kept_within_grace += 1
            continue
        # Past grace: candidate. The delete decision is re-evaluated under the
        # per-digest lock so it reflects the live reference set, not the scan
        # snapshot.
        async with coordinator.hold(digest):
            # Re-stat: the blob may have been deleted since the scan.
            if await blob_store.stat(digest=digest) is None:
                continue
            # The definitive pin check, under the lock: a record created by a
            # concurrent put (which holds the same lock to create it) is visible
            # here, so a just-pinned blob is never deleted.
            try:
                referenced = await record_store.is_digest_referenced(digest)
            except ArtifactRecordCorruptError:
                if metrics is not None:
                    metrics.counter("artifact_orphan_sweep_failure_total")
                raise
            if referenced:
                in_use += 1
                continue
            try:
                await blob_store.delete(digest=digest)
            except Exception:
                if metrics is not None:
                    metrics.counter("artifact_orphan_cleanup_failure_total")
                continue
            deleted += 1
            if metrics is not None:
                metrics.counter("artifact_orphan_total")

    return OrphanSweepStats(
        deleted=deleted, kept_within_grace=kept_within_grace, in_use=in_use
    )


__all__: "list[str]" = [
    "OrphanSweepConfig",
    "OrphanSweepStats",
    "sweep_orphan_blobs",
]
