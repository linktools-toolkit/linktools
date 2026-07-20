#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Blob orphan sweeper -- the core orphan contract + the asset-backed reference
sweep implementation.

A blob becomes an orphan candidate when it is written but the record
transaction that would pin it fails (or the record is later deleted). The core
contract (plan Storage/Asset/Artifact section): an unreferenced blob is only
deletable once it is past a safety window (default 24h); the sweep runs no more
frequently than the sweep interval (default 6h). The window protects a blob
written by a transaction that has not yet committed its record -- deleting it
eagerly would corrupt a still-in-flight put.

The contract config (:class:`OrphanSweepConfig`) is core business policy and
lives here; the enumeration + delete is adapter-specific (an external object
store sweeps its own way). This module provides the asset-backed reference
sweep that walks the blob and record trees via the asset-backed adapters'
enumeration methods; a downstream adapter implements the same policy against
its own backend.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .artifact_backends import AssetBackedArtifactBlobStore, AssetBackedArtifactRecordStore


@dataclass(frozen=True, slots=True)
class OrphanSweepConfig:
    """Core orphan policy. Defaults follow the plan: a 24h grace window before
    an unreferenced blob is deletable, swept at most every 6h.

    ``grace_period`` is enforced by ``sweep_asset_backed_orphan_blobs`` (a blob
    inside the window is never deleted). ``sweep_interval`` is NOT enforced by
    the sweep function -- it is the upper bound on how often a SCHEDULER calls
    the sweep, and is declared here so the policy lives in one place. A caller
    that runs the sweep on a timer reads ``sweep_interval`` to size that timer.
    """

    grace_period: timedelta = timedelta(hours=24)
    sweep_interval: timedelta = timedelta(hours=6)


@dataclass(frozen=True, slots=True)
class OrphanSweepStats:
    """Outcome of one sweep pass, for observability."""

    deleted: int
    kept_within_grace: int
    in_use: int


async def sweep_asset_backed_orphan_blobs(
    blob_store: "AssetBackedArtifactBlobStore",
    record_store: "AssetBackedArtifactRecordStore",
    config: "OrphanSweepConfig | None" = None,
    *,
    now: "datetime | None" = None,
    metrics: "Any | None" = None,
) -> OrphanSweepStats:
    """Delete asset-backed blobs that no record references AND that are past
    the grace window. Blobs still in use (referenced by a live record) are never
    touched; unreferenced blobs within the grace window are kept (an in-flight
    transaction may still pin them).

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

    # The set of digests pinned by at least one record. A blob not in this set
    # is an orphan candidate.
    referenced: "set[str]" = set()
    async for digest in record_store.iter_referenced_digests():
        referenced.add(digest)

    deleted = 0
    kept_within_grace = 0
    in_use = 0
    async for digest, modified_at in blob_store.iter_digests_with_mtime():
        if digest in referenced:
            in_use += 1
            continue
        age = moment - modified_at
        if age < cfg.grace_period:
            # Unreferenced but inside the safety window -- an in-flight
            # transaction may still commit a record pinning it. Leave it.
            kept_within_grace += 1
            continue
        try:
            await blob_store.delete(digest=digest)
        except Exception:
            if metrics is not None:
                metrics.counter("artifact_orphan_cleanup_failure_total")
            # A single failed delete must not stall the sweep; the blob stays
            # an orphan candidate and the next sweep re-attempts it.
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
    "sweep_asset_backed_orphan_blobs",
]
