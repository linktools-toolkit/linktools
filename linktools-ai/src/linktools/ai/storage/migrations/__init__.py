#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""One-shot layout migrations for the asset domain.

The runtime code reads only the new schema (``ai_assets`` / ``ai_asset_*`` tables;
``{root}/assets`` directory). These functions migrate EXISTING data written under
the old names so a deployment does not have to rebuild its store from scratch.
They are explicit, one-shot tools -- the runtime never auto-migrates (it cannot
tell, in the shared-directory case, which blobs belong to which database)."""


import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path

__all__ = [
    "migrate_asset_layout",
    "migrate_sql_asset_tables",
    "migrate_sql_asset_tables_downgrade",
    "migrate_sqlite_artifact_root",
    "SqliteArtifactRootMigrationReport",
]


def migrate_asset_layout(root: "str | Path") -> None:
    """Rename a FilesystemAssetBackend root's old ``resources`` directory to
    ``assets`` in place.

    Rules:
    * ``resources`` exists, ``assets`` does not -> atomic ``os.replace``.
    * ``resources`` does not exist, ``assets`` exists -> no-op.
    * neither exists -> create ``assets``.
    * both exist -> fail closed (a human must resolve the split -- silently
      merging or picking one could lose data).

    The atomic rename means a crash mid-migration cannot leave the store half-
    renamed: ``os.replace`` is atomic on the same filesystem."""
    root_path = Path(root)
    resources = root_path / "resources"
    assets = root_path / "assets"
    if resources.exists() and assets.exists():
        raise RuntimeError(
            f"migrate_asset_layout({root!r}): both 'resources' and 'assets' "
            f"exist -- resolve the split manually before migrating"
        )
    if resources.exists():
        os.replace(resources, assets)
        return
    if assets.exists():
        return
    assets.mkdir(parents=True, exist_ok=True)


async def migrate_sql_asset_tables(engine) -> None:
    """Rename the old asset tables on an EXISTING SQLAlchemy database to the
    new names, idempotently. Renames:

    * ``ai_resources``            -> ``ai_assets``
    * ``ai_resource_idempotency`` -> ``ai_asset_idempotency``
    * ``ai_resource_revision``    -> ``ai_asset_revision``

    A table that already has the new name (or never had the old one) is a no-op
    for that table. The index/constraint names derived from SQLAlchemy's naming
    convention are recreated by the ORM metadata on the next ``create_all``; this
    migration only moves the data-bearing tables. Run once against a live engine.
    To revert, call :func:`migrate_sql_asset_tables_downgrade` (the same rename
    in reverse)."""
    from sqlalchemy import text

    renames = (
        ("ai_resources", "ai_assets"),
        ("ai_resource_idempotency", "ai_asset_idempotency"),
        ("ai_resource_revision", "ai_asset_revision"),
    )
    async with engine.begin() as conn:
        for old, new in renames:
            old_exists = await conn.execute(
                text(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name=:n"
                ),
                {"n": old},
            )
            if old_exists.first() is None:
                continue
            new_exists = await conn.execute(
                text(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name=:n"
                ),
                {"n": new},
            )
            if new_exists.first() is not None:
                # Both present -- refuse rather than risk clobbering.
                raise RuntimeError(
                    f"migrate_sql_asset_tables: both {old!r} and {new!r} exist -- "
                    f"resolve manually"
                )
            await conn.execute(text(f"ALTER TABLE {old} RENAME TO {new}"))


async def migrate_sql_asset_tables_downgrade(engine) -> None:
    """Reverse of :func:`migrate_sql_asset_tables`: rename the new asset tables
    back to the old resource names. Exists only for rollback -- runtime code
    never calls it. Same idempotency + fail-closed-on-both-present rules as the
    upgrade direction."""
    from sqlalchemy import text

    renames = (
        ("ai_assets", "ai_resources"),
        ("ai_asset_idempotency", "ai_resource_idempotency"),
        ("ai_asset_revision", "ai_resource_revision"),
    )
    async with engine.begin() as conn:
        for old, new in renames:
            old_exists = await conn.execute(
                text(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name=:n"
                ),
                {"n": old},
            )
            if old_exists.first() is None:
                continue
            new_exists = await conn.execute(
                text(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name=:n"
                ),
                {"n": new},
            )
            if new_exists.first() is not None:
                raise RuntimeError(
                    f"migrate_sql_asset_tables_downgrade: both {old!r} and "
                    f"{new!r} exist -- resolve manually"
                )
            await conn.execute(text(f"ALTER TABLE {old} RENAME TO {new}"))


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
