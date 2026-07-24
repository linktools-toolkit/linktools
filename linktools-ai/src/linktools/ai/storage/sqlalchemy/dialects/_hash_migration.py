#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Dialect-specific implementation of the ai_assets/ai_asset_idempotency
path_hash/key_hash live migration (spec section 7.4). Lives under
``dialects/`` -- the one package core is allowed to branch on a dialect
name -- so :mod:`storage.migrations` itself stays dialect-neutral and just
delegates to :func:`run_hash_index_migration` / :func:`run_hash_index_migration_downgrade`."""

import hashlib

# Rows re-hashed per batch during backfill -- bounded so a huge existing
# table does not hold one giant transaction/result set in memory.
_HASH_BACKFILL_BATCH = 1000


def _row_hash(value: str) -> bytes:
    return hashlib.sha256(value.encode("utf-8")).digest()


async def run_hash_index_migration(engine) -> None:
    """Live migration for an EXISTING ``ai_assets`` / ``ai_asset_idempotency``
    database created before ``path_hash`` / ``key_hash`` existed: MySQL's
    index key-length limit is exceeded by the full ``path``/``key`` column
    under a multi-byte charset, so the unique index moved to a sha256 hash of
    each. Steps, in the spec's order:

    1. add the ``path_hash`` / ``key_hash`` columns, nullable;
    2. backfill them in batches (Python-side sha256, portable across every
       dialect -- no dialect-specific hash function);
    3. verify no NULLs remain;
    4. tighten the columns to NOT NULL;
    5. create the new hash-based unique constraint;
    6. drop the old full-column unique constraint;
    7. create the (MySQL-length-capped) query index on the raw column.

    Idempotent: a column/constraint/index already present is left alone, so
    a partially-applied migration (e.g. crashed mid-run) can simply be
    re-run. SQLite cannot ALTER a column's nullability or add/drop a named
    constraint in place, so its path goes through a create-copy-swap of each
    table; MySQL/PostgreSQL use direct ``ALTER TABLE`` statements."""
    dialect = engine.dialect.name
    if dialect == "sqlite":
        await _migrate_sqlite(engine)
    else:
        await _migrate_alter(engine, dialect)


async def run_hash_index_migration_downgrade(engine) -> None:
    """Reverse of :func:`run_hash_index_migration`: restore the full-column
    unique constraints and drop the hash columns + query index. Exists only
    for rollback -- runtime code never calls it."""
    dialect = engine.dialect.name
    if dialect == "sqlite":
        await _migrate_sqlite_downgrade(engine)
    else:
        await _migrate_alter_downgrade(engine, dialect)


async def _backfill_hash_column(conn, *, table: str, id_col: str, value_col: str, hash_col: str) -> None:
    from sqlalchemy import text

    while True:
        rows = (
            await conn.execute(
                text(
                    f"SELECT {id_col}, {value_col} FROM {table} "
                    f"WHERE {hash_col} IS NULL LIMIT :n"
                ),
                {"n": _HASH_BACKFILL_BATCH},
            )
        ).fetchall()
        if not rows:
            return
        for row_id, value in rows:
            await conn.execute(
                text(f"UPDATE {table} SET {hash_col} = :h WHERE {id_col} = :id"),
                {"h": _row_hash(value), "id": row_id},
            )


async def _assert_no_nulls(conn, *, table: str, hash_col: str) -> None:
    from sqlalchemy import text

    remaining = (
        await conn.execute(text(f"SELECT COUNT(*) FROM {table} WHERE {hash_col} IS NULL"))
    ).scalar_one()
    if remaining:
        raise RuntimeError(
            f"run_hash_index_migration: {remaining} row(s) in {table} still "
            f"have a NULL {hash_col} after backfill -- refusing to tighten to NOT NULL"
        )


async def _migrate_alter(engine, dialect: str) -> None:
    from sqlalchemy import text

    from ..models import ASSET_IDEMPOTENCY_CONSTRAINT, ASSET_PATH_CONSTRAINT

    binary_type = "VARBINARY(32)" if dialect == "mysql" else "BYTEA"
    async with engine.begin() as conn:
        cols = (await conn.execute(text("SELECT * FROM ai_assets LIMIT 0"))).keys()
        if "path_hash" not in cols:
            await conn.execute(text(f"ALTER TABLE ai_assets ADD COLUMN path_hash {binary_type} NULL"))
        idem_cols = (
            await conn.execute(text("SELECT * FROM ai_asset_idempotency LIMIT 0"))
        ).keys()
        if "key_hash" not in idem_cols:
            await conn.execute(
                text(f"ALTER TABLE ai_asset_idempotency ADD COLUMN key_hash {binary_type} NULL")
            )

        await _backfill_hash_column(
            conn, table="ai_assets", id_col="id", value_col="path", hash_col="path_hash"
        )
        await _backfill_hash_column(
            conn, table="ai_asset_idempotency", id_col="id", value_col="key", hash_col="key_hash"
        )
        await _assert_no_nulls(conn, table="ai_assets", hash_col="path_hash")
        await _assert_no_nulls(conn, table="ai_asset_idempotency", hash_col="key_hash")

        if dialect == "mysql":
            await conn.execute(text(f"ALTER TABLE ai_assets MODIFY path_hash {binary_type} NOT NULL"))
            await conn.execute(
                text(f"ALTER TABLE ai_asset_idempotency MODIFY key_hash {binary_type} NOT NULL")
            )
        else:
            await conn.execute(text("ALTER TABLE ai_assets ALTER COLUMN path_hash SET NOT NULL"))
            await conn.execute(
                text("ALTER TABLE ai_asset_idempotency ALTER COLUMN key_hash SET NOT NULL")
            )

        await conn.execute(
            text(f"ALTER TABLE ai_assets ADD CONSTRAINT {ASSET_PATH_CONSTRAINT} UNIQUE (path_hash)")
        )
        await conn.execute(
            text(
                f"ALTER TABLE ai_asset_idempotency ADD CONSTRAINT "
                f"{ASSET_IDEMPOTENCY_CONSTRAINT} UNIQUE (key_hash)"
            )
        )
        if dialect == "mysql":
            await conn.execute(text("ALTER TABLE ai_assets DROP INDEX path"))
            await conn.execute(text("ALTER TABLE ai_asset_idempotency DROP INDEX key"))
        else:
            await conn.execute(text("DROP INDEX IF EXISTS ix_ai_assets_path"))
            await conn.execute(text("DROP INDEX IF EXISTS ix_ai_asset_idempotency_key"))

        prefix_len = 191
        if dialect == "mysql":
            await conn.execute(
                text(f"CREATE INDEX ix_ai_assets_path_prefix ON ai_assets (path({prefix_len}))")
            )
            await conn.execute(
                text(
                    f"CREATE INDEX ix_ai_asset_idempotency_key_prefix ON "
                    f"ai_asset_idempotency (key({prefix_len}))"
                )
            )
        else:
            await conn.execute(text("CREATE INDEX IF NOT EXISTS ix_ai_assets_path_prefix ON ai_assets (path)"))
            await conn.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS ix_ai_asset_idempotency_key_prefix "
                    "ON ai_asset_idempotency (key)"
                )
            )


async def _migrate_alter_downgrade(engine, dialect: str) -> None:
    from sqlalchemy import text

    from ..models import ASSET_IDEMPOTENCY_CONSTRAINT, ASSET_PATH_CONSTRAINT

    async with engine.begin() as conn:
        await conn.execute(text(f"ALTER TABLE ai_assets DROP CONSTRAINT {ASSET_PATH_CONSTRAINT}"))
        await conn.execute(
            text(f"ALTER TABLE ai_asset_idempotency DROP CONSTRAINT {ASSET_IDEMPOTENCY_CONSTRAINT}")
        )
        await conn.execute(text(f"ALTER TABLE ai_assets ADD CONSTRAINT {ASSET_PATH_CONSTRAINT} UNIQUE (path)"))
        await conn.execute(
            text(
                f"ALTER TABLE ai_asset_idempotency ADD CONSTRAINT "
                f"{ASSET_IDEMPOTENCY_CONSTRAINT} UNIQUE (key)"
            )
        )
        await conn.execute(text("ALTER TABLE ai_assets DROP COLUMN path_hash"))
        await conn.execute(text("ALTER TABLE ai_asset_idempotency DROP COLUMN key_hash"))


async def _table_columns(conn, table: str) -> "list[str]":
    from sqlalchemy import text

    return list((await conn.execute(text(f"SELECT * FROM {table} LIMIT 0"))).keys())


async def _migrate_sqlite(engine) -> None:
    """SQLite cannot add a UNIQUE constraint, drop a constraint, or tighten a
    column to NOT NULL in place -- each requires rebuilding the table.
    Recreates ``ai_assets``/``ai_asset_idempotency`` under the final target
    schema (matching ``models.py`` exactly), copies every row across with the
    hash backfilled inline, then swaps the old table out atomically within
    the migration's own transaction."""
    from sqlalchemy import text

    async with engine.begin() as conn:
        asset_cols = await _table_columns(conn, "ai_assets")
        if "path_hash" not in asset_cols:
            rows = (await conn.execute(text("SELECT * FROM ai_assets"))).mappings().all()
            await conn.execute(text("ALTER TABLE ai_assets RENAME TO ai_assets_old"))
            await conn.execute(
                text(
                    """
                    CREATE TABLE ai_assets (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        path VARCHAR(1024) NOT NULL,
                        path_hash BLOB NOT NULL,
                        kind VARCHAR(32) NOT NULL,
                        etag VARCHAR(64) NOT NULL,
                        version INTEGER NOT NULL,
                        content_type VARCHAR(255),
                        size INTEGER NOT NULL,
                        content BLOB NOT NULL,
                        modified_at DATETIME NOT NULL,
                        metadata_json TEXT NOT NULL,
                        deleted_at DATETIME,
                        whiteout_version INTEGER,
                        CONSTRAINT uq_ai_assets_tenant_path UNIQUE (path_hash)
                    )
                    """
                )
            )
            for row in rows:
                values = dict(row)
                values["path_hash"] = _row_hash(values["path"])
                columns = ", ".join(values)
                placeholders = ", ".join(f":{k}" for k in values)
                await conn.execute(
                    text(f"INSERT INTO ai_assets ({columns}) VALUES ({placeholders})"),
                    values,
                )
            await conn.execute(text("DROP TABLE ai_assets_old"))
            await conn.execute(
                text("CREATE INDEX ix_ai_assets_path_prefix ON ai_assets (path)")
            )

        idem_cols = await _table_columns(conn, "ai_asset_idempotency")
        if "key_hash" not in idem_cols:
            rows = (
                await conn.execute(text("SELECT * FROM ai_asset_idempotency"))
            ).mappings().all()
            await conn.execute(
                text("ALTER TABLE ai_asset_idempotency RENAME TO ai_asset_idempotency_old")
            )
            await conn.execute(
                text(
                    """
                    CREATE TABLE ai_asset_idempotency (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        key_hash BLOB NOT NULL,
                        key VARCHAR(1024) NOT NULL,
                        request_hash VARCHAR(64) NOT NULL,
                        result_json TEXT,
                        CONSTRAINT uq_ai_asset_idempotency_tenant_key UNIQUE (key_hash)
                    )
                    """
                )
            )
            for row in rows:
                values = dict(row)
                values["key_hash"] = _row_hash(values["key"])
                columns = ", ".join(values)
                placeholders = ", ".join(f":{k}" for k in values)
                await conn.execute(
                    text(f"INSERT INTO ai_asset_idempotency ({columns}) VALUES ({placeholders})"),
                    values,
                )
            await conn.execute(text("DROP TABLE ai_asset_idempotency_old"))
            await conn.execute(
                text(
                    "CREATE INDEX ix_ai_asset_idempotency_key_prefix "
                    "ON ai_asset_idempotency (key)"
                )
            )


async def _migrate_sqlite_downgrade(engine) -> None:
    """Reverse of :func:`_migrate_sqlite`: rebuild both tables under the
    pre-migration (full-column-unique, no hash column) schema."""
    from sqlalchemy import text

    async with engine.begin() as conn:
        rows = (await conn.execute(text("SELECT * FROM ai_assets"))).mappings().all()
        await conn.execute(text("ALTER TABLE ai_assets RENAME TO ai_assets_new"))
        await conn.execute(
            text(
                """
                CREATE TABLE ai_assets (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    path VARCHAR(1024) NOT NULL,
                    kind VARCHAR(32) NOT NULL,
                    etag VARCHAR(64) NOT NULL,
                    version INTEGER NOT NULL,
                    content_type VARCHAR(255),
                    size INTEGER NOT NULL,
                    content BLOB NOT NULL,
                    modified_at DATETIME NOT NULL,
                    metadata_json TEXT NOT NULL,
                    deleted_at DATETIME,
                    whiteout_version INTEGER,
                    CONSTRAINT uq_ai_assets_tenant_path UNIQUE (path)
                )
                """
            )
        )
        for row in rows:
            values = {k: v for k, v in dict(row).items() if k != "path_hash"}
            columns = ", ".join(values)
            placeholders = ", ".join(f":{k}" for k in values)
            await conn.execute(
                text(f"INSERT INTO ai_assets ({columns}) VALUES ({placeholders})"), values
            )
        await conn.execute(text("DROP TABLE ai_assets_new"))

        idem_rows = (
            await conn.execute(text("SELECT * FROM ai_asset_idempotency"))
        ).mappings().all()
        await conn.execute(
            text("ALTER TABLE ai_asset_idempotency RENAME TO ai_asset_idempotency_new")
        )
        await conn.execute(
            text(
                """
                CREATE TABLE ai_asset_idempotency (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    key VARCHAR(512) NOT NULL,
                    request_hash VARCHAR(64) NOT NULL,
                    result_json TEXT,
                    CONSTRAINT uq_ai_asset_idempotency_tenant_key UNIQUE (key)
                )
                """
            )
        )
        for row in idem_rows:
            values = {k: v for k, v in dict(row).items() if k != "key_hash"}
            columns = ", ".join(values)
            placeholders = ", ".join(f":{k}" for k in values)
            await conn.execute(
                text(f"INSERT INTO ai_asset_idempotency ({columns}) VALUES ({placeholders})"),
                values,
            )
        await conn.execute(text("DROP TABLE ai_asset_idempotency_new"))


__all__: "list[str]" = ["run_hash_index_migration", "run_hash_index_migration_downgrade"]
