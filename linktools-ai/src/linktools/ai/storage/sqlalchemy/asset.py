#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SqlAlchemyAssetBackend: DB-backed AssetBackend.

Concurrency model:

- Revision counter bumps atomically via a portable UPDATE + SELECT loop
  (no dialect-specific upsert, no RETURNING -- MySQL lacks RETURNING on
  UPDATE): seeds ``id=1`` on the first call, then server-side arithmetic
  guarantees two concurrent writers always produce distinct revisions (one
  blocks on the row lock, then re-evaluates against the new value). See
  :meth:`_bump_revision`.
- Asset updates use a conditional WHERE clause on ``version``:
  ``UPDATE ... WHERE path = :path AND version = :expected``. ``rowcount == 0``
  means a concurrent writer committed first (lost update prevented). Callers
  without a precondition retry the SELECT-UPDATE loop.
- ``If-Match`` enters the same UPDATE WHERE clause as ``AND etag = :if_match``,
  so the precondition is enforced by the DB rather than a Python
  pre-read that can race.
- An idempotent no-op PUT (identical content already live) does not return on
  the strength of the Python SELECT alone: a zero-effect ``UPDATE ... SET
  version = version WHERE path = :path AND version = :expected AND etag =
  :expected_etag`` (:meth:`_confirm_unchanged`) re-validates the row is STILL
  exactly what was read before handing it back. ``rowcount == 0`` means a
  concurrent writer changed the row after the read, so the caller retries
  instead of returning an already-stale result.

Each checked write (raw_put_checked / raw_delete_checked) runs precondition +
idempotency + mutate in ONE transaction. The unique ``path`` constraint
backstops the INSERT race; the conditional UPDATE backstops the UPDATE race.
Reads use their own short-lived session.
"""

import hashlib
import json
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from hashlib import sha256
from typing import AsyncIterator, Callable, Mapping

from sqlalchemy import or_, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from .dialects import SqlAlchemyDialectStrategy, resolve_dialect_strategy
from .models import AssetRow, AssetIdempotencyRow, AssetRevisionRow
from ...asset.models import (
    Depth,
    Found,
    IdempotencyRecord,
    Masked,
    Missing,
    MoveResult,
    Asset,
    AssetInfo,
    AssetLookupInfo,
    AssetKind,
    AssetPage,
    WriteOptions,
)
from ...asset.path import AssetPath, _require_persistable_path, matches_asset_depth
from ...errors import (
    IdempotencyConflictError,
    AssetPathHashCollisionError,
    AssetPreconditionFailedError,
)


def _idempotency_key_hash(key: str) -> bytes:
    """The AssetIdempotencyRow unique-index key, for the same MySQL
    index-key-length reason as :func:`asset_path_hash`."""
    return hashlib.sha256(key.encode("utf-8")).digest()


def asset_path_hash(path: AssetPath) -> bytes:
    """The AssetRow unique-index key: MySQL's index key-length limit is
    exceeded by ``path``'s VARCHAR(1024) under a multi-byte charset, so the
    full path cannot itself be the unique-constraint column on that dialect.
    A collision (same hash, different path) is a real possibility over a
    32-byte digest at scale; every insert/conflict path in this module
    re-verifies the full ``path`` against the row it collided with and
    raises :class:`AssetPathHashCollisionError` rather than silently
    treating two different paths as the same row."""
    return hashlib.sha256(path.value.encode("utf-8")).digest()


def _row_to_info(row: AssetRow) -> AssetInfo:
    return AssetInfo(
        path=AssetPath(row.path),
        kind=AssetKind(row.kind),
        etag=row.etag,
        version=row.version,
        content_type=row.content_type,
        size=row.size,
        modified_at=row.modified_at,
        metadata=json.loads(row.metadata_json),
    )


def _dict_to_info(values: "Mapping[str, object]") -> AssetInfo:
    """Build a AssetInfo from a column dict (the shape returned by
    _conditional_update_row). Used in place of _row_to_info when we hold column
    values from RETURNING rather than a AssetRow instance, to avoid the
    identity-map staleness that update().returning(AssetRow) trips over."""
    return AssetInfo(
        path=AssetPath(values["path"]),
        kind=AssetKind(values["kind"]),
        etag=values["etag"],
        version=values["version"],
        content_type=values["content_type"],
        size=values["size"],
        modified_at=values["modified_at"],
        metadata=json.loads(values["metadata_json"]),
    )


def _idempotency_result_to_info(result_json: "str | None") -> "AssetInfo | None":
    if result_json is None:
        return None
    raw = json.loads(result_json)
    return AssetInfo(
        path=AssetPath(raw["path"]),
        kind=AssetKind(raw["kind"]),
        etag=raw["etag"],
        version=raw["version"],
        content_type=raw["content_type"],
        size=raw["size"],
        modified_at=datetime.fromisoformat(raw["modified_at"]),
        metadata=raw["metadata"],
    )


# Bounded retry for the unconditional-put/delete conflict loop. Two concurrent
# writers cannot loop forever: each retry sees the winner's commit and either
# succeeds on the next conditional UPDATE or hits the precondition path. 8 is
# generous -- in practice 1-2 attempts suffice under any realistic schedule.
_CONFLICT_RETRIES = 8

# Candidate fetch window for depth-filtered listing. The ONE/INFINITY filter
# runs in Python (to stay dialect-portable), so rows are pulled in ordered
# batches past the cursor until enough qualify; 256 is comfortably above any
# realistic single-listing fanout at the current data scale.
_LIST_BATCH = 256


class SqlAlchemyAssetBackend:
    def __init__(
        self,
        *,
        session_factory: "Callable[[], AsyncSession]",
        session: "AsyncSession | None" = None,
        strategy: "SqlAlchemyDialectStrategy | None" = None,
    ) -> None:
        self._session_factory = session_factory
        # When ``session`` is bound (UoW participation), reads reuse it without
        # closing and writes flush into the surrounding transaction without
        # begin/commit/rollback -- the UoW owns the atomic scope. Unbound, each
        # call opens a short session and commits/rolls back itself.
        self._session = session
        # Stable id the AssetStore overrides with the canonical primary/overlay
        # tag; defaults to the backend's origin so it is never blank.
        self.backend_id = "sqlalchemy"
        # The dialect strategy classifies integrity violations so the insert
        # path can react to a known unique-key conflict portably. Resolved here
        # when the caller (e.g. a test) does not inject one; the adapter passes
        # its already-resolved strategy to avoid resolving twice.
        self._strategy = strategy or resolve_dialect_strategy(session_factory)

    @asynccontextmanager
    async def _read_session(self) -> "AsyncIterator[AsyncSession]":
        if self._session is not None:
            yield self._session
            return
        async with self._session_factory() as session:
            yield session

    @asynccontextmanager
    async def _write_session(self) -> "AsyncIterator[AsyncSession]":
        if self._session is not None:
            # Bound: the UoW owns begin/commit/rollback; flush happens via the
            # callers' existing awaits. We MUST NOT begin/commit/rollback here.
            yield self._session
            return
        async with self._session_factory() as session:
            async with session.begin():
                yield session

    async def _get_row(
        self, session: AsyncSession, path: AssetPath
    ) -> "AssetRow | None":
        result = await session.execute(
            select(AssetRow).where(AssetRow.path == path.value)
        )
        return result.scalar_one_or_none()

    async def _check_hash_collision(
        self, session: AsyncSession, path: AssetPath
    ) -> None:
        """Called after an INSERT conflict on the ``path_hash`` unique index
        when no row exists at the exact ``path`` we tried to write (the
        normal "this path already has a row" case is handled by the caller's
        retry against ``_get_row``, which matches on the full path). If a
        DIFFERENT path collides on the same hash, raise rather than let the
        retry loop spin until it exhausts its attempt budget with no
        actionable error."""
        colliding = await session.execute(
            select(AssetRow.path).where(AssetRow.path_hash == asset_path_hash(path))
        )
        row = colliding.scalar_one_or_none()
        if row is not None and row != path.value:
            raise AssetPathHashCollisionError(
                f"path {path.value!r} hashes to the same path_hash as "
                f"existing path {row!r}"
            )

    async def raw_get(self, path: AssetPath, *, include_content: bool = True):
        _require_persistable_path(path)
        async with self._read_session() as session:
            row = await self._get_row(session, path)
            if row is None:
                return Missing()
            if row.deleted_at is not None:
                return Masked(path=path, version=row.whiteout_version or 0)
            content = row.content if include_content else b""
            return Found(asset=Asset(info=_row_to_info(row), content=content))

    async def raw_stat(self, path: AssetPath) -> "AssetLookupInfo | None":
        """Metadata-only stat: SELECT every column EXCEPT content.
        Loading a potentially-large blob just to read its etag/version is
        wasteful; projecting the metadata columns only keeps stat() cheap. A
        masked (deleted_at) row is treated as absent -- stat is for live
        assets; whiteout lineage is the province of raw_get."""
        async with self._read_session() as session:
            result = await session.execute(
                select(
                    AssetRow.path,
                    AssetRow.kind,
                    AssetRow.etag,
                    AssetRow.version,
                    AssetRow.content_type,
                    AssetRow.size,
                    AssetRow.modified_at,
                    AssetRow.metadata_json,
                )
                .where(AssetRow.path == path.value)
                .where(AssetRow.deleted_at.is_(None))
            )
            row = result.one_or_none()
        if row is None:
            return None
        return _dict_to_info(row._asdict())

    async def raw_list(
        self, path: AssetPath, *, depth: Depth, limit: int, cursor: "str | None"
    ) -> AssetPage:
        """Depth-filtered keyset pagination using the shared
        :func:`matches_asset_depth` so ZERO/ONE/INFINITY mean the same here as
        in the Memory and Filesystem backends.

        ZERO targets only the base row itself (``path = :base``). ONE and
        INFINITY select the base plus every descendant (``path = :base OR path
        LIKE :prefix``) and narrow ONE to direct children in Python -- no
        database-specific string function is used to express "direct child", so
        the query is portable across SQLite/MySQL/PostgreSQL.

        Because the ONE filter runs in Python, a single ``LIMIT`` could under-
        return when many deeper descendants fill the window before enough
        direct children arrive. Candidates are therefore pulled in ordered
        batches past the cursor until ``limit + 1`` qualifying rows accumulate
        or the candidate set is exhausted; the limit-th path becomes
        next_cursor."""
        if depth is Depth.ZERO:
            async with self._read_session() as session:
                result = await session.execute(
                    select(AssetRow).where(
                        AssetRow.path == path.value,
                        AssetRow.deleted_at.is_(None),
                    )
                )
                row = result.scalar_one_or_none()
            items = [_row_to_info(row)] if row is not None else []
            return AssetPage(items=tuple(items), cursor=None)

        prefix = path.value.rstrip("/") + "/"
        escaped_prefix = (
            prefix.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        )
        candidate_clause = or_(
            AssetRow.path == path.value,
            AssetRow.path.like(f"{escaped_prefix}%", escape="\\"),
        )
        items: "list[AssetInfo]" = []
        scan_cursor = cursor
        while True:
            conditions = [candidate_clause, AssetRow.deleted_at.is_(None)]
            if scan_cursor is not None:
                conditions.append(AssetRow.path > scan_cursor)
            async with self._read_session() as session:
                result = await session.execute(
                    select(AssetRow)
                    .where(*conditions)
                    .order_by(AssetRow.path)
                    .limit(_LIST_BATCH)
                )
                rows = result.scalars().all()
            if not rows:
                break
            last_scanned = rows[-1].path
            for row in rows:
                info = _row_to_info(row)
                if matches_asset_depth(path, info.path, depth):
                    items.append(info)
                    if len(items) > limit:
                        break
            if len(items) > limit:
                break
            if len(rows) < _LIST_BATCH:
                break
            scan_cursor = last_scanned
        next_cursor = items[limit - 1].path.value if len(items) > limit else None
        return AssetPage(items=tuple(items[:limit]), cursor=next_cursor)

    # ------------------------------------------------------------------
    # Revision counter: atomic increment
    # ------------------------------------------------------------------

    async def _bump_revision(self, session: AsyncSession) -> int:
        """Atomic increment of the single revision counter (row id=1).

        ``UPDATE ... SET value = value + 1`` does the server-side arithmetic so
        two concurrent writers always produce distinct revisions (one blocks on
        the row lock, then re-evaluates against the new value). The first-ever
        bump -- when the row does not yet exist -- seeds it inside a savepoint;
        if a concurrent seed already landed the PK conflict is absorbed and the
        increment retries against the now-existing row. Plain UPDATE + SELECT
        keeps this portable across SQLite/MySQL/PostgreSQL (no dialect-specific
        upsert, no RETURNING -- MySQL lacks RETURNING on UPDATE)."""
        for _ in range(_CONFLICT_RETRIES):
            stmt = (
                update(AssetRevisionRow)
                .where(AssetRevisionRow.id == 1)
                .values(value=AssetRevisionRow.value + 1)
                .execution_options(synchronize_session=False)
            )
            result = await session.execute(stmt)
            if result.rowcount:
                value = (
                    await session.execute(
                        select(AssetRevisionRow.value).where(AssetRevisionRow.id == 1)
                    )
                ).scalar_one()
                return value
            try:
                async with session.begin_nested():
                    session.add(AssetRevisionRow(id=1, value=1))
                return 1
            except IntegrityError:
                continue
        raise AssetPreconditionFailedError(
            f"revision bump conflict after {_CONFLICT_RETRIES} retries"
        )

    # ------------------------------------------------------------------
    # PUT: conditional UPDATE on version + If-Match in WHERE
    # ------------------------------------------------------------------

    async def _confirm_unchanged(
        self,
        session: AsyncSession,
        path: AssetPath,
        *,
        expected_version: int,
        expected_etag: str,
    ) -> bool:
        """No-op version fence: ``UPDATE ... SET version = version WHERE path
        = :path AND version = :expected AND etag = :expected_etag``. An
        idempotent PUT whose content already matches the live row must NOT
        return early on the strength of a Python-side SELECT alone -- that
        SELECT could be arbitrarily stale by the time the caller observes the
        result. This zero-effect UPDATE re-validates the row is STILL exactly
        what was read, atomically, at the moment of return: rowcount == 1
        means the DB confirms the read was current; rowcount == 0 means a
        concurrent writer changed the row between the read and this fence, so
        the caller must retry rather than hand back a result that was already
        stale when it was produced."""
        stmt = (
            update(AssetRow)
            .where(
                AssetRow.path == path.value,
                AssetRow.version == expected_version,
                AssetRow.etag == expected_etag,
            )
            .values(version=AssetRow.version)
            .execution_options(synchronize_session=False)
        )
        result = await session.execute(stmt)
        return result.rowcount == 1

    async def _conditional_update_row(
        self,
        session: AsyncSession,
        path: AssetPath,
        expected_version: int,
        content: bytes,
        content_type: "str | None",
        metadata: "Mapping[str, object]",
        *,
        new_version: int,
        if_match: "str | None",
    ) -> "dict | None":
        """Conditional UPDATE on ``version`` with optional If-Match in the WHERE
        clause. Returns a dict of the post-update column values, or None when 0
        rows matched (a concurrent writer committed first, or the etag
        precondition failed).

        Portability: the UPDATE is issued WITHOUT ``RETURNING`` (MySQL lacks
        UPDATE...RETURNING); success is read from ``rowcount`` and the new column
        values are re-read with a SELECT in the same transaction (the row lock
        the UPDATE acquired is held until commit, so the SELECT observes exactly
        this writer's new values, not a concurrent writer's). The SELECT projects
        individual columns to bypass the identity map (``synchronize_session=
        False`` would otherwise hand back a stale cached instance)."""
        conditions = [
            AssetRow.path == path.value,
            AssetRow.version == expected_version,
        ]
        if if_match is not None:
            # push the etag precondition into the UPDATE WHERE so the DB
            # -- not a Python pre-read -- enforces it. Two concurrent writers
            # both holding the same stale if_match cannot both pass: only one
            # UPDATE matches the etag before the row's etag changes.
            conditions.append(AssetRow.etag == if_match)
        stmt = (
            update(AssetRow)
            .where(*conditions)
            .values(
                kind="file",
                etag=sha256(content).hexdigest(),
                version=new_version,
                content_type=content_type,
                size=len(content),
                content=content,
                modified_at=datetime.now(timezone.utc),
                metadata_json=json.dumps(dict(metadata)),
                deleted_at=None,
                whiteout_version=None,
            )
            .execution_options(synchronize_session=False)
        )
        result = await session.execute(stmt)
        if result.rowcount != 1:
            return None
        refreshed = await session.execute(
            select(
                AssetRow.path,
                AssetRow.kind,
                AssetRow.etag,
                AssetRow.version,
                AssetRow.content_type,
                AssetRow.size,
                AssetRow.modified_at,
                AssetRow.metadata_json,
            ).where(AssetRow.path == path.value)
        )
        return refreshed.one()._asdict()

    async def _put_once(
        self,
        session: AsyncSession,
        path: AssetPath,
        content: bytes,
        content_type: "str | None",
        metadata: "Mapping[str, object]",
        *,
        if_match: "str | None",
        if_none_match: bool,
        bump_revision: bool = True,
    ) -> "AssetInfo | None":
        """One attempt: SELECT then either INSERT (unique-constraint atomicity)
        or conditional UPDATE ``WHERE version = :expected [AND etag = :if_match]``.

        Returns the new ``AssetInfo`` on success.

        Returns None on a *retry-able* conflict: a concurrent INSERT won the
        unique-path race (no precondition set), or a concurrent UPDATE bumped
        the version first (no If-Match set). The caller retries.

        Raises ``AssetPreconditionFailedError`` on a hard precondition
        failure: If-Match on a missing asset, If-None-Match on an existing
        asset, IntegrityError on INSERT under If-None-Match, or a conditional
        UPDATE that missed because the etag no longer matches If-Match.

        ``bump_revision``: when False, the caller (raw_move) owns the single
        revision bump for the whole composite operation and directs this helper
        to skip its per-step bumps. The no-op short-circuit path never bumps
        regardless of the flag (an idempotent no-op PUT must not bump).
        """
        row = await self._get_row(session, path)
        if row is None:
            if if_match is not None:
                # If-Match on a missing asset is a precondition failure.
                raise AssetPreconditionFailedError(
                    f"if-match precondition failed: {path}"
                )
            # INSERT path. The dialect strategy executes a conflict-detecting
            # INSERT (SQLite/PostgreSQL ON CONFLICT DO NOTHING; MySQL SAVEPOINT)
            # so a concurrent INSERT that wins the path-unique race is reported
            # as a conflict WITHOUT poisoning the surrounding transaction -- the
            # write cannot leak out of the UoW. A conflict under If-None-Match
            # is a precondition failure; otherwise the caller retries via the
            # UPDATE-existing path against the new row. Any non-unique
            # IntegrityError propagates unchanged (the strategy re-raises OTHER).
            conflict = await self._strategy.execute_conflict_insert(
                session,
                AssetRow,
                {
                    "path": path.value,
                    "path_hash": asset_path_hash(path),
                    "kind": "file",
                    "etag": sha256(content).hexdigest(),
                    "version": 1,
                    "content_type": content_type,
                    "size": len(content),
                    "content": content,
                    "modified_at": datetime.now(timezone.utc),
                    "metadata_json": json.dumps(dict(metadata)),
                    "deleted_at": None,
                    "whiteout_version": None,
                },
                index_elements=["path_hash"],
            )
            if conflict:
                await self._check_hash_collision(session, path)
                if if_none_match:
                    raise AssetPreconditionFailedError(
                        f"asset already exists: {path}"
                    )
                return None
            if bump_revision:
                await self._bump_revision(session)
            return AssetInfo(
                path=path,
                kind=AssetKind.FILE,
                etag=sha256(content).hexdigest(),
                version=1,
                content_type=content_type,
                size=len(content),
                modified_at=datetime.now(timezone.utc),
                metadata=dict(metadata),
            )
        else:
            # Row exists. If-None-Match demands it not exist.
            if if_none_match and row.deleted_at is None:
                raise AssetPreconditionFailedError(
                    f"asset already exists: {path}"
                )
            # no-op short-circuit: identical content + content_type +
            # metadata + live state is an idempotent no-op PUT, which must NOT
            # bump version/revision. The Python comparison above is only a
            # candidate check -- the row may have changed since our SELECT, so
            # the DB-level no-op version fence below is what actually confirms
            # the read is still current before returning it as truth.
            if (
                row.deleted_at is None
                and row.content == content
                and row.content_type == content_type
                and json.loads(row.metadata_json) == dict(metadata)
            ):
                # If-Match is still enforced even on a no-op: a stale etag means
                # the caller's view of the asset is outdated, which must be
                # surfaced as a precondition failure regardless of
                # whether the PUT would have changed anything.
                if if_match is not None and row.etag != if_match:
                    raise AssetPreconditionFailedError(
                        f"if-match precondition failed: {path}"
                    )
                confirmed = await self._confirm_unchanged(
                    session,
                    path,
                    expected_version=row.version,
                    expected_etag=row.etag,
                )
                if not confirmed:
                    return None  # retry-able: the row changed after our SELECT
                return _row_to_info(row)
            # conditional UPDATE on version. new_version is computed in
            # Python from the SELECTed row, but the conditional WHERE makes the
            # assignment safe: if another writer bumped version first, our
            # UPDATE matches 0 rows.
            expected_version = row.version
            new_version = max(row.version or 0, row.whiteout_version or 0) + 1
            updated = await self._conditional_update_row(
                session,
                path,
                expected_version,
                content,
                content_type,
                metadata,
                new_version=new_version,
                if_match=if_match,
            )
            if updated is None:
                if if_match is not None:
                    # the etag precondition failed inside the DB WHERE.
                    raise AssetPreconditionFailedError(
                        f"if-match precondition failed: {path}"
                    )
                return None  # retry-able conflict
            if bump_revision:
                await self._bump_revision(session)
            return _dict_to_info(updated)

    async def _put_with_retry(
        self,
        session: AsyncSession,
        path: AssetPath,
        content: bytes,
        content_type: "str | None",
        metadata: "Mapping[str, object]",
        *,
        if_match: "str | None",
        if_none_match: bool,
        bump_revision: bool = True,
    ) -> AssetInfo:
        """SELECT-then-conditional-UPDATE loop. Precondition failures raise
        immediately from ``_put_once`` (no retry). Retry-able conflicts (only
        reachable with no precondition set) loop until the conditional UPDATE
        matches, preserving the external "always wins" contract for unconditional
        puts without losing updates. ``bump_revision`` is forwarded to
        ``_put_once`` so composite operations (raw_move) can own the single
        revision bump themselves."""
        for _ in range(_CONFLICT_RETRIES):
            info = await self._put_once(
                session,
                path,
                content,
                content_type,
                metadata,
                if_match=if_match,
                if_none_match=if_none_match,
                bump_revision=bump_revision,
            )
            if info is not None:
                return info
            # Retry-able conflict (no precondition set). Expire the identity-map
            # cache so the next iteration's SELECT sees the winner's commit.
            session.expire_all()
        raise AssetPreconditionFailedError(
            f"asset update conflict after {_CONFLICT_RETRIES} retries: {path}"
        )

    async def raw_put(
        self,
        path: AssetPath,
        content: bytes,
        *,
        content_type: "str | None",
        metadata: "Mapping[str, object]",
    ):
        async with self._write_session() as session:
            info = await self._put_with_retry(
                session,
                path,
                content,
                content_type,
                metadata,
                if_match=None,
                if_none_match=False,
            )
        return info

    async def raw_put_checked(
        self,
        path: AssetPath,
        content: bytes,
        *,
        options: WriteOptions,
        request_hash: str,
    ) -> Asset:
        """Atomic precondition + idempotency + put in ONE transaction.

        The If-Match precondition enters the UPDATE WHERE clause, so the etag
        check is enforced by the DB rather than a Python pre-read. Concurrent
        writers both holding the same stale If-Match cannot both succeed: the
        conditional UPDATE serializes them at the row lock. If-None-Match is
        enforced up-front (live-row existence check) and atomically by the
        unique-path constraint: a concurrent INSERT that wins the path race
        between the check and the insert is classified ASSET_KEY by the savepoint
        helper and retried (or, under If-None-Match, raised as a precondition
        failure). Non-unique IntegrityErrors are never blanket-converted -- they
        propagate unchanged."""
        _require_persistable_path(path)
        idem_key = f"put:{options.idempotency_key}" if options.idempotency_key else None
        async with self._write_session() as session:
            if idem_key is not None:
                idem_result = await session.execute(
                    select(AssetIdempotencyRow).where(AssetIdempotencyRow.key == idem_key)
                )
                idem_row = idem_result.scalar_one_or_none()
                if idem_row is not None:
                    if idem_row.request_hash != request_hash:
                        raise IdempotencyConflictError(
                            f"idempotency key {options.idempotency_key!r} reused with a different request"
                        )
                    cached_info = _idempotency_result_to_info(
                        idem_row.result_json
                    )
                    row = await self._get_row(session, path)
                    content_bytes = (
                        row.content
                        if (row is not None and row.deleted_at is None)
                        else content
                    )
                    return Asset(info=cached_info, content=content_bytes)
            info = await self._put_with_retry(
                session,
                path,
                content,
                options.content_type,
                options.metadata,
                if_match=options.if_match,
                if_none_match=options.if_none_match,
            )
            if idem_key is not None:
                await self._save_idempotency_row(
                    session, idem_key, request_hash, info
                )
            return Asset(info=info, content=content)

    # ------------------------------------------------------------------
    # DELETE: same conditional pattern (If-Match in WHERE for consistency)
    # ------------------------------------------------------------------

    async def _conditional_delete_row(
        self,
        session: AsyncSession,
        path: AssetPath,
        expected_version: int,
        *,
        if_match: "str | None",
    ) -> "bool":
        """Conditional UPDATE that marks a live row as masked: ``WHERE path =
        :path AND version = :expected [AND etag = :if_match] AND deleted_at IS
        NULL``. Returns True if a row was masked, False on 0-row match (already
        masked, missing, or precondition failure)."""
        conditions = [
            AssetRow.path == path.value,
            AssetRow.version == expected_version,
            AssetRow.deleted_at.is_(None),
        ]
        if if_match is not None:
            conditions.append(AssetRow.etag == if_match)
        stmt = (
            update(AssetRow)
            .where(*conditions)
            .values(
                deleted_at=datetime.now(timezone.utc),
                whiteout_version=expected_version + 1,
                content=b"",
            )
            .execution_options(synchronize_session=False)
        )
        result = await session.execute(stmt)
        return result.rowcount > 0

    async def _ensure_tombstone(self, session: AsyncSession, path: AssetPath) -> bool:
        """Insert a tombstone for a missing path via the dialect's
        conflict-detecting insert. Returns True if a concurrent create/delete
        already has a row at this path (the caller re-reads and re-runs its
        conditional mask); False if the tombstone was inserted. A non-unique
        IntegrityError propagates unchanged."""
        conflict = await self._strategy.execute_conflict_insert(
            session,
            AssetRow,
            {
                "path": path.value,
                "path_hash": asset_path_hash(path),
                "kind": "file",
                "etag": "",
                "version": 0,
                "content_type": None,
                "size": 0,
                "content": b"",
                "modified_at": datetime.now(timezone.utc),
                "metadata_json": "{}",
                "deleted_at": datetime.now(timezone.utc),
                "whiteout_version": 1,
            },
            index_elements=["path_hash"],
        )
        if conflict:
            await self._check_hash_collision(session, path)
        return conflict

    async def _apply_delete_unconditional(
        self, session: AsyncSession, path: AssetPath
    ) -> "AssetInfo | None":
        """Unconditional delete. Loops:
        SELECT then mask-the-live-row via conditional UPDATE on version. If a
        concurrent writer bumps version first, our UPDATE misses and we retry
        against the new committed state. A row that is already masked gets its
        whiteout counter bumped atomically; a row that doesn't exist gets a
        tombstone inserted (unique-path constraint backstops concurrent creates)."""
        for _ in range(_CONFLICT_RETRIES):
            row = await self._get_row(session, path)
            if row is None:
                # No row at all: seed a tombstone so future reads see Masked. A
                # concurrent create/delete may land a row first; the
                # conflict-detecting insert reports that so we retry and handle
                # the now-present row (mask it if live, idempotent if masked).
                if await self._ensure_tombstone(session, path):
                    session.expire_all()
                    continue
                await self._bump_revision(session)
                return None
            if row.deleted_at is not None:
                # Already masked: atomically bump the whiteout counter so the
                # lineage version keeps advancing.
                stmt = (
                    update(AssetRow)
                    .where(
                        AssetRow.path == path.value,
                        AssetRow.version == row.version,
                    )
                    .values(whiteout_version=(AssetRow.whiteout_version or 0) + 1)
                    .execution_options(synchronize_session=False)
                )
                await session.execute(stmt)
                await self._bump_revision(session)
                return None
            # Live row: conditional mask on version.If a concurrent writer bumps
            # version first, our UPDATE matches 0 rows and we retry.
            removed_info = _row_to_info(row)
            masked = await self._conditional_delete_row(
                session, path, row.version, if_match=None
            )
            if masked:
                await self._bump_revision(session)
                return removed_info
            # Lost the race; expire and retry against the new committed state.
            session.expire_all()
        return None

    async def raw_delete(self, path: AssetPath) -> "AssetInfo | None":
        async with self._write_session() as session:
            removed_info = await self._apply_delete_unconditional(session, path)
        return removed_info

    async def raw_delete_checked(
        self,
        path: AssetPath,
        *,
        options: WriteOptions,
        request_hash: str,
    ) -> None:
        """Atomic precondition + idempotency + delete in ONE transaction. If-Match
        is pushed into the UPDATE WHERE (consistent with raw_put_checked):
        two concurrent deletes holding the same stale If-Match cannot
        both succeed."""
        _require_persistable_path(path)
        idem_key = (
            f"delete:{options.idempotency_key}" if options.idempotency_key else None
        )
        async with self._write_session() as session:
            if idem_key is not None:
                idem_result = await session.execute(
                    select(AssetIdempotencyRow).where(AssetIdempotencyRow.key == idem_key)
                )
                idem_row = idem_result.scalar_one_or_none()
                if idem_row is not None:
                    if idem_row.request_hash != request_hash:
                        raise IdempotencyConflictError(
                            f"idempotency key {options.idempotency_key!r} reused with a different request"
                        )
                    return  # idempotent replay: delete returns None
            # SELECT-then-conditional-mask loop. On conflict with if_match,
            # raise; on conflict without if_match, retry; if the row is
            # already masked or absent, treat as idempotent success.
            removed_info: "AssetInfo | None" = None
            masked_any = False
            for _ in range(_CONFLICT_RETRIES):
                row = await self._get_row(session, path)
                if row is None or row.deleted_at is not None:
                    # Missing or already masked. If-Match requires a live row.
                    if options.if_match is not None:
                        raise AssetPreconditionFailedError(
                            f"if-match precondition failed: {path}"
                        )
                    # No-op for the caller, but ensure a tombstone exists so
                    # subsequent reads see Masked. A concurrent create/delete
                    # may have landed a row first; retry to mask/reconcile it.
                    if row is None:
                        if await self._ensure_tombstone(session, path):
                            session.expire_all()
                            continue
                        await self._bump_revision(session)
                    masked_any = True
                    break
                removed_info = _row_to_info(row)
                masked = await self._conditional_delete_row(
                    session, path, row.version, if_match=options.if_match
                )
                if masked:
                    await self._bump_revision(session)
                    masked_any = True
                    break
                # Conflict: either if_match failed (precondition failure) or
                # a concurrent writer bumped version first (retry).
                if options.if_match is not None:
                    raise AssetPreconditionFailedError(
                        f"if-match precondition failed: {path}"
                    )
                session.expire_all()
            if not masked_any:
                raise AssetPreconditionFailedError(
                    f"asset delete conflict after {_CONFLICT_RETRIES} retries: {path}"
                )
            if idem_key is not None:
                await self._save_idempotency_row(
                    session, idem_key, request_hash, removed_info
                )

    # ------------------------------------------------------------------
    # MOVE: ONE transaction
    # ------------------------------------------------------------------

    async def raw_move_checked(
        self,
        source: AssetPath,
        target: AssetPath,
        *,
        options: WriteOptions,
        request_hash: str,
    ) -> MoveResult:
        """Atomic MOVE in ONE transaction: idempotency check, load+lock source,
        validate, write target, whiteout source, bump revision once, idempotency
        save. All steps commit or roll back together, so a concurrent reader can
        never observe the intermediate states a decomposed put+delete would
        expose (target written while source still live = duplicate; source
        masked while target missing = data loss) and a replayed idempotency key
        cannot execute the move twice. The revision counter bumps exactly once
        -- observable proof of single-transaction atomicity (a put+delete
        decomposition would bump twice).

        Source precondition: must exist and be live. Target preconditions:
        options.if_match / options.if_none_match are enforced against the
        target row via _put_with_retry (same code path as raw_put_checked, so
        If-Match enters the conditional UPDATE WHERE)."""
        _require_persistable_path(source)
        _require_persistable_path(target)
        idem_key = f"move:{options.idempotency_key}" if options.idempotency_key else None
        async with self._write_session() as session:
            if idem_key is not None:
                idem_result = await session.execute(
                    select(AssetIdempotencyRow).where(AssetIdempotencyRow.key == idem_key)
                )
                idem_row = idem_result.scalar_one_or_none()
                if idem_row is not None:
                    if idem_row.request_hash != request_hash:
                        raise IdempotencyConflictError(
                            f"idempotency key {options.idempotency_key!r} reused with a different request"
                        )
                    cached_info = _idempotency_result_to_info(idem_row.result_json)
                    content_row = await self._get_row(session, target)
                    bytes_ = (
                        content_row.content
                        if (content_row is not None and content_row.deleted_at is None)
                        else b""
                    )
                    return Asset(
                        info=cached_info
                        or AssetInfo(
                            path=target, kind=AssetKind.FILE, etag="",
                            version=0, content_type=None, size=0,
                            modified_at=datetime.now(timezone.utc), metadata={},
                        ),
                        content=bytes_,
                    )
            source_row = await self._get_row(session, source)
            if source_row is None or source_row.deleted_at is not None:
                raise AssetPreconditionFailedError(
                    f"cannot move missing asset: {source}"
                )
            source_content = source_row.content
            source_info = _row_to_info(source_row)

            # Write target (INSERT or conditional UPDATE). Runs in this
            # transaction's snapshot: no concurrent writer can interleave.
            # bump_revision=False: raw_move owns the single revision bump
            # for the whole composite operation -- the
            # target write and source mask together count as ONE state
            # change, so the counter advances exactly once.
            target_info = await self._put_with_retry(
                session,
                target,
                source_content,
                source_info.content_type,
                dict(source_info.metadata),
                if_match=options.if_match,
                if_none_match=options.if_none_match,
                bump_revision=False,
            )

            # Whiteout source: conditional mask on the version we read.
            # Inside one transaction the source row cannot have changed
            # since our SELECT, so the conditional UPDATE always matches;
            # masked=False would indicate a bug or external mutation.
            masked = await self._conditional_delete_row(
                session, source, source_row.version, if_match=None
            )
            if not masked:
                raise AssetPreconditionFailedError(
                    f"source changed during move: {source}"
                )

            # One revision bump for the whole move.
            await self._bump_revision(session)
            if idem_key is not None:
                await self._save_idempotency_row(
                    session, idem_key, request_hash, target_info
                )
            return Asset(info=target_info, content=source_content)

    # ------------------------------------------------------------------
    # Idempotency + revision readers (unchanged behavior)
    # ------------------------------------------------------------------

    async def _save_idempotency_row(
        self,
        session: AsyncSession,
        key: str,
        request_hash: str,
        info: "AssetInfo | None",
    ) -> None:
        result_json = None
        if info is not None:
            result_json = json.dumps(
                {
                    "path": info.path.value,
                    "kind": info.kind.value,
                    "etag": info.etag,
                    "version": info.version,
                    "content_type": info.content_type,
                    "size": info.size,
                    "modified_at": info.modified_at.isoformat(),
                    "metadata": dict(info.metadata),
                }
            )
        # INSERT the idempotency record. A concurrent same-key insert surfaces
        # as a conflict: re-read and fingerprint-check -- same request_hash is
        # an idempotent replay (the winner's record stands), a different one is a
        # key-reuse conflict. Any non-unique IntegrityError propagates unchanged
        # (the strategy re-raises OTHER) rather than being swallowed.
        key_hash = _idempotency_key_hash(key)
        conflict = await self._strategy.execute_conflict_insert(
            session,
            AssetIdempotencyRow,
            {
                "key": key,
                "key_hash": key_hash,
                "request_hash": request_hash,
                "result_json": result_json,
            },
            index_elements=["key_hash"],
        )
        if not conflict:
            return
        existing = await session.execute(
            select(AssetIdempotencyRow).where(AssetIdempotencyRow.key_hash == key_hash)
        )
        row = existing.scalar_one_or_none()
        if row is not None and row.key != key:
            raise AssetPathHashCollisionError(
                f"idempotency key {key!r} hashes to the same key_hash as "
                f"existing key {row.key!r}"
            )
        if row is not None and row.request_hash != request_hash:
            raise IdempotencyConflictError(
                f"idempotency key {key!r} reused with a different request"
            )

    async def revision(self) -> str:
        async with self._read_session() as session:
            row = await session.get(AssetRevisionRow, 1)
            return str(row.value if row is not None else 0)
