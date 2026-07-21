#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Replay: re-run a captured :class:`RunSnapshot` to produce a NEW execution,
never mutating history. Before re-running, the snapshot is validated: every
referenced artifact must exist and hash to its id, so a tampered or truncated
snapshot is refused."""

from ..artifact import ArtifactIntegrityError
from .models import EvalCase, EvalExecution, EvalTarget

# RunSnapshot is imported lazily (string annotation) to avoid an import cycle
# with snapshot.py at module load; both live in the same package.


class SnapshotValidationError(Exception):
    """Raised when a snapshot's artifacts are missing or fail integrity checks."""


async def validate_snapshot(snapshot, artifact_store, *, tenant_id: str) -> None:
    """Validate a RunSnapshot before replay. Every artifact it references (run
    record, run definition, input, output, events, resource snapshots) must
    resolve for this tenant AND hash back to its content-addressed id -- get()
    re-hashes on read, so a tampered or truncated artifact is refused (not just
    a missing one)."""
    artifact_ids: "list[str]" = [
        snapshot.run_record_artifact_id,
        snapshot.run_definition_artifact_id,
        snapshot.input_artifact_id,
    ]
    if snapshot.output_artifact_id is not None:
        artifact_ids.append(snapshot.output_artifact_id)
    artifact_ids.extend(snapshot.event_artifact_ids)
    for snap_ref in snapshot.resource_snapshots:
        artifact_ids.append(snap_ref.artifact_id)
    for artifact_id in artifact_ids:
        try:
            content = await artifact_store.get(artifact_id=artifact_id, tenant_id=tenant_id)
        except ArtifactIntegrityError as exc:
            raise SnapshotValidationError(
                f"snapshot artifact integrity failed: {artifact_id}: {exc}"
            ) from exc
        if content is None:
            raise SnapshotValidationError(
                f"snapshot artifact missing or not tenant-owned: {artifact_id}"
            )


async def replay(
    snapshot,
    executor,
    target: EvalTarget,
    artifact_store,
    *,
    tenant_id: str,
) -> EvalExecution:
    """Re-run a snapshot: validate it, then drive ``executor`` against a case
    reconstructed from the snapshot's input. The executor mints a fresh run, so
    history is never modified -- replay always creates a new execution."""
    await validate_snapshot(snapshot, artifact_store, tenant_id=tenant_id)
    case = EvalCase(
        id=f"replay-{snapshot.run_id}",
        input_artifact_id=snapshot.input_artifact_id,
    )
    return await executor.execute(target, case)


__all__: "list[str]" = ["SnapshotValidationError", "validate_snapshot", "replay"]
