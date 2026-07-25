#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""One-shot data migration for a SQLite deployment's artifact blob root."""


import hashlib
from dataclasses import dataclass, field
from pathlib import Path

__all__ = [
    "migrate_sqlite_artifact_root",
    "SqliteArtifactRootMigrationReport",
]


@dataclass
class SqliteArtifactRootMigrationReport:
    """Result of :func:`migrate_sqlite_artifact_root`. ``copied`` are digests
    moved old -> new; ``missing`` are digests referenced by a record but absent
    in the old blobs dir; ``unreferenced`` are blobs in the old dir no record
    references (the operator decides whether to clean them -- they may belong to
    a different database that shared the old dir)."""

    copied: "list[str]" = field(default_factory=list)
    missing: "list[str]" = field(default_factory=list)
    unreferenced: "list[str]" = field(default_factory=list)


async def migrate_sqlite_artifact_root(
    *,
    engine,
    old_blobs_root: "str | Path",
    new_blobs_root: "str | Path",
) -> SqliteArtifactRootMigrationReport:
    """Copy a database's referenced artifact blobs from the old shared
    ``blobs/`` directory into the new per-database ``<db>.artifacts/blobs`` root,
    verifying each blob's SHA256. The old directory is NOT deleted (it may be
    shared with other databases that read it too -- there is no way to tell
    which blobs belong to which db, so auto-deleting could orphan another db).

    Steps:
    1. scan ``ai_artifact_records.sha256`` for the set of referenced digests;
    2. copy each referenced blob from old -> new (``<xx>/<digest>`` layout);
    3. verify SHA256 on every copy (a mismatch means the old blob is corrupt --
       fail closed rather than propagate bad bytes);
    4. leave the old directory in place;
    5. return the unreferenced + missing digests so the operator can clean up
       once every database that shared the old dir has migrated.

    A blob already present in the new root is left as-is (re-running the tool
    after a partial migration is safe)."""
    from sqlalchemy import text

    old = Path(old_blobs_root)
    new = Path(new_blobs_root)
    new.mkdir(parents=True, exist_ok=True)

    async with engine.begin() as conn:
        rows = await conn.execute(
            text("SELECT DISTINCT sha256 FROM ai_artifact_records")
        )
        referenced = sorted({r[0] for r in rows if r[0]})

    report = SqliteArtifactRootMigrationReport()
    for digest in referenced:
        src = old / digest[:2] / digest
        if not src.is_file():
            report.missing.append(digest)
            continue
        dst = new / digest[:2] / digest
        if dst.is_file():
            continue  # already migrated (idempotent re-run)
        data = src.read_bytes()
        actual = hashlib.sha256(data).hexdigest()
        if actual != digest:
            raise RuntimeError(
                f"migrate_sqlite_artifact_root: sha256 mismatch for {digest} "
                f"(old blob hashes to {actual}) -- refusing to copy a corrupt blob"
            )
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(data)
        report.copied.append(digest)

    referenced_set = set(referenced)
    if old.is_dir():
        for shard in sorted(old.iterdir()):
            if shard.is_dir():
                for blob in sorted(shard.iterdir()):
                    if blob.name not in referenced_set:
                        report.unreferenced.append(blob.name)
    return report
