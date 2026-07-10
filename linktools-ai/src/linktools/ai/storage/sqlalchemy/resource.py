#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SqlAlchemyResourceBackend: DB-backed ResourceBackend.

Concurrency model:

- Revision counter bumps atomically via ``UPDATE ai_resource_revision
  SET value = value + 1 WHERE id = 1 RETURNING value``. Server-side
  arithmetic so two concurrent writers always produce distinct revisions.
- Resource updates use a conditional WHERE clause on ``version``:
  ``UPDATE ... WHERE path = :path AND version = :expected``. ``rowcount == 0``
  means a concurrent writer committed first (lost update prevented). Callers
  without a precondition retry the SELECT-UPDATE loop.
- ``If-Match`` enters the same UPDATE WHERE clause as ``AND etag = :if_match``,
  so the precondition is enforced by the DB rather than a Python
  pre-read that can race.

Each checked write (raw_put_checked / raw_delete_checked) runs precondition +
idempotency + mutate in ONE transaction. The unique ``path`` constraint
backstops the INSERT race; the conditional UPDATE backstops the UPDATE race.
Reads use their own short-lived session.
"""

import json
from datetime import datetime, timezone
from hashlib import sha256
from typing import Callable, Mapping

from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from .models import ResourceRow, IdempotencyRow, RevisionRow
from ..resource.models import Depth, Found, IdempotencyRecord, Masked, Missing, MoveResult, Resource, ResourceInfo, ResourceLookupInfo, ResourceKind, ResourcePage, WriteOptions
from ..resource.path import ResourcePath
from ...errors import IdempotencyConflictError, ResourcePreconditionFailedError


def _row_to_info(row: ResourceRow) -> ResourceInfo:
    return ResourceInfo(
        path=ResourcePath(row.path),
        kind=ResourceKind(row.kind),
        etag=row.etag,
        version=row.version,
        content_type=row.content_type,
        size=row.size,
        modified_at=row.modified_at,
        metadata=json.loads(row.metadata_json),
    )


def _dict_to_info(values: "Mapping[str, object]") -> ResourceInfo:
    """Build a ResourceInfo from a column dict (the shape returned by
    _conditional_update_row). Used in place of _row_to_info when we hold column
    values from RETURNING rather than a ResourceRow instance, to avoid the
    identity-map staleness that update().returning(ResourceRow) trips over."""
    return ResourceInfo(
        path=ResourcePath(values["path"]),
        kind=ResourceKind(values["kind"]),
        etag=values["etag"],
        version=values["version"],
        content_type=values["content_type"],
        size=values["size"],
        modified_at=values["modified_at"],
        metadata=json.loads(values["metadata_json"]),
    )


def _idempotency_result_to_info(result_json: "str | None") -> "ResourceInfo | None":
    if result_json is None:
        return None
    raw = json.loads(result_json)
    return ResourceInfo(
        path=ResourcePath(raw["path"]),
        kind=ResourceKind(raw["kind"]),
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


class SqlAlchemyResourceBackend:
    def __init__(self, *, session_factory: "Callable[[], AsyncSession]", readonly: bool = False) -> None:
        self.readonly = readonly
        self._session_factory = session_factory

    async def _get_row(self, session: AsyncSession, path: ResourcePath) -> "ResourceRow | None":
        result = await session.execute(select(ResourceRow).where(ResourceRow.path == path.value))
        return result.scalar_one_or_none()

    async def raw_get(self, path: ResourcePath, *, include_content: bool = True):
        async with self._session_factory() as session:
            row = await self._get_row(session, path)
            if row is None:
                return Missing()
            if row.deleted_at is not None:
                return Masked(path=path, version=row.whiteout_version or 0)
            content = row.content if include_content else b""
            return Found(resource=Resource(info=_row_to_info(row), content=content))

    async def raw_stat(self, path: ResourcePath) -> "ResourceLookupInfo | None":
        """Metadata-only stat: SELECT every column EXCEPT content.
        Loading a potentially-large blob just to read its etag/version is
        wasteful; projecting the metadata columns only keeps stat() cheap. A
        masked (deleted_at) row is treated as absent -- stat is for live
        resources; whiteout lineage is the province of raw_get."""
        async with self._session_factory() as session:
            result = await session.execute(
                select(
                    ResourceRow.path, ResourceRow.kind, ResourceRow.etag, ResourceRow.version,
                    ResourceRow.content_type, ResourceRow.size, ResourceRow.modified_at,
                    ResourceRow.metadata_json,
                )
                .where(ResourceRow.path == path.value)
                .where(ResourceRow.deleted_at.is_(None))
            )
            row = result.one_or_none()
        if row is None:
            return None
        return _dict_to_info(row._asdict())

    async def raw_propfind(self, path: ResourcePath, *, depth: Depth, limit: int, cursor: "str | None") -> ResourcePage:
        """Keyset pagination: ``WHERE path > :cursor ORDER BY path
        LIMIT :limit+1``. Pushing the depth=ONE filter into SQL (``NOT LIKE
        prefix + '%/%'``) keeps the LIMIT honest -- a Python-side depth filter
        applied after LIMIT could silently under-return. Fetching limit+1 rows
        lets us detect "more available" without a second count query: when we
        get limit+1, the (limit+1)th path becomes next_cursor."""
        prefix = path.value.rstrip("/") + "/"
        escaped_prefix = prefix.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        conditions = [
            ResourceRow.path.like(f"{escaped_prefix}%", escape="\\"),
            ResourceRow.deleted_at.is_(None),
        ]
        if depth == Depth.ONE:
            # Exclude grand-children and deeper: a child of `/agents/` matches
            # `/agents/%` but NOT `/agents/%/%` (which requires at least one
            # further slash). The leading prefix is already escaped; the
            # trailing `%/` are wildcards/escaped-slash per LIKE-with-escape.
            conditions.append(~ResourceRow.path.like(f"{escaped_prefix}%/%", escape="\\"))
        if cursor is not None:
            conditions.append(ResourceRow.path > cursor)
        async with self._session_factory() as session:
            result = await session.execute(
                select(ResourceRow)
                .where(*conditions)
                .order_by(ResourceRow.path)
                .limit(limit + 1)
            )
            items = [_row_to_info(row) for row in result.scalars()]
        next_cursor = items[limit].path.value if len(items) > limit else None
        return ResourcePage(items=tuple(items[:limit]), cursor=next_cursor)

    # ------------------------------------------------------------------
    # Revision counter: atomic increment
    # ------------------------------------------------------------------

    async def _bump_revision(self, session: AsyncSession) -> int:
        """Atomic ``UPDATE ... SET value = value + 1 RETURNING value``.

        Server-side arithmetic guarantees two concurrent writers always produce
        distinct revisions: one UPDATE blocks on the row lock, then re-evaluates
        against the new value. Replaces the Python-side ``row.value += 1`` read-
        modify-write that lost updates under contention.

        On the first-ever revision the counter row doesn't exist yet; the seed
        INSERT runs inside a SAVEPOINT so a concurrent creator (which got there
        first) doesn't abort the outer transaction when its pk constraint fires.
        """
        bump_stmt = (
            update(RevisionRow)
            .where(RevisionRow.id == 1)
            .values(value=RevisionRow.value + 1)
            .returning(RevisionRow.value)
            .execution_options(synchronize_session=False)
        )
        result = await session.execute(bump_stmt)
        new_value = result.scalar_one_or_none()
        if new_value is not None:
            return new_value
        # First-ever revision: seed id=1 inside a SAVEPOINT. A concurrent creator
        # that won the race makes our INSERT raise IntegrityError here; the
        # savepoint absorbs it and the outer transaction stays usable.
        try:
            async with session.begin_nested():
                session.add(RevisionRow(id=1, value=1))
                await session.flush()
            return 1
        except IntegrityError:
            # Lost the seed race: another transaction inserted id=1 between our
            # UPDATE (0 rows) and our INSERT. The row exists now -- re-bump.
            result = await session.execute(bump_stmt)
            return result.scalar_one()

    # ------------------------------------------------------------------
    # PUT: conditional UPDATE on version + If-Match in WHERE
    # ------------------------------------------------------------------

    async def _conditional_update_row(
        self,
        session: AsyncSession,
        path: ResourcePath,
        expected_version: int,
        content: bytes,
        content_type: "str | None",
        metadata: "Mapping[str, object]",
        *,
        new_version: int,
        if_match: "str | None",
    ) -> "dict | None":
        """Conditional UPDATE on ``version`` with optional If-Match
        in the WHERE clause. Returns a dict of the post-update
        column values, or None when 0 rows matched (a concurrent writer
        committed first, or the etag precondition failed).

        Returns columns rather than the ORM row because SQLAlchemy, with
        ``synchronize_session=False``, hands back the *cached* identity-map
        instance on ``update().returning(ResourceRow)`` without refreshing its
        attributes -- so the caller would see stale (pre-update) values. READS
        via individual columns bypass the identity map entirely.
        """
        conditions = [ResourceRow.path == path.value, ResourceRow.version == expected_version]
        if if_match is not None:
            # push the etag precondition into the UPDATE WHERE so the DB
            # -- not a Python pre-read -- enforces it. Two concurrent writers
            # both holding the same stale if_match cannot both pass: only one
            # UPDATE matches the etag before the row's etag changes.
            conditions.append(ResourceRow.etag == if_match)
        stmt = (
            update(ResourceRow)
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
            .returning(
                ResourceRow.path, ResourceRow.kind, ResourceRow.etag, ResourceRow.version,
                ResourceRow.content_type, ResourceRow.size, ResourceRow.modified_at,
                ResourceRow.metadata_json,
            )
            .execution_options(synchronize_session=False)
        )
        result = await session.execute(stmt)
        row = result.one_or_none()
        if row is None:
            return None
        return row._asdict()

    async def _insert_new_row(
        self,
        session: AsyncSession,
        path: ResourcePath,
        content: bytes,
        content_type: "str | None",
        metadata: "Mapping[str, object]",
    ) -> ResourceRow:
        """INSERT a fresh row at version=1. The caller wraps this in a SAVEPOINT
        so a concurrent INSERT (which won the unique-path race) surfaces as
        IntegrityError that the caller can translate without aborting its outer
        transaction."""
        row = ResourceRow(
            path=path.value,
            kind="file",
            etag=sha256(content).hexdigest(),
            version=1,
            content_type=content_type,
            size=len(content),
            content=content,
            modified_at=datetime.now(timezone.utc),
            metadata_json=json.dumps(dict(metadata)),
            deleted_at=None,
            whiteout_version=None,
        )
        session.add(row)
        await session.flush()
        return row

    async def _put_once(
        self,
        session: AsyncSession,
        path: ResourcePath,
        content: bytes,
        content_type: "str | None",
        metadata: "Mapping[str, object]",
        *,
        if_match: "str | None",
        if_none_match: bool,
        bump_revision: bool = True,
    ) -> "ResourceInfo | None":
        """One attempt: SELECT then either INSERT (unique-constraint atomicity)
        or conditional UPDATE ``WHERE version = :expected [AND etag = :if_match]``.

        Returns the new ``ResourceInfo`` on success.

        Returns None on a *retry-able* conflict: a concurrent INSERT won the
        unique-path race (no precondition set), or a concurrent UPDATE bumped
        the version first (no If-Match set). The caller retries.

        Raises ``ResourcePreconditionFailedError`` on a hard precondition
        failure: If-Match on a missing resource, If-None-Match on an existing
        resource, IntegrityError on INSERT under If-None-Match, or a conditional
        UPDATE that missed because the etag no longer matches If-Match.

        ``bump_revision``: when False, the caller (raw_move) owns the single
        revision bump for the whole composite operation and directs this helper
        to skip its per-step bumps. The no-op short-circuit path never bumps
        regardless of the flag (an idempotent no-op PUT must not bump).
        """
        row = await self._get_row(session, path)
        if row is None:
            if if_match is not None:
                # If-Match on a missing resource is a precondition failure.
                raise ResourcePreconditionFailedError(f"if-match precondition failed: {path}")
            # INSERT path: unique-path constraint is the atomicity backstop.
            try:
                async with session.begin_nested():
                    new_row = await self._insert_new_row(session, path, content, content_type, metadata)
                    if bump_revision:
                        await self._bump_revision(session)
                return _row_to_info(new_row)
            except IntegrityError:
                # Concurrent INSERT won the path race. Under If-None-Match this
                # is a precondition failure (resource now exists); otherwise the
                # caller retries via the UPDATE-existing path.
                if if_none_match:
                    raise ResourcePreconditionFailedError(f"resource already exists: {path}")
                return None
        else:
            # Row exists. If-None-Match demands it not exist.
            if if_none_match and row.deleted_at is None:
                raise ResourcePreconditionFailedError(f"resource already exists: {path}")
            # no-op short-circuit: identical content + content_type +
            # metadata + live state is an idempotent no-op PUT, which must NOT
            # bump version/revision. Python comparison is a tiny race window
            # (another writer between our SELECT and return); the consequence is
            # returning slightly stale info for a same-content PUT, which is
            # benign and matches the prior implementation's behavior.
            if (
                row.deleted_at is None
                and row.content == content
                and row.content_type == content_type
                and json.loads(row.metadata_json) == dict(metadata)
            ):
                # If-Match is still enforced even on a no-op: a stale etag means
                # the caller's view of the resource is outdated, which must be
                # surfaced as a precondition failure regardless of
                # whether the PUT would have changed anything.
                if if_match is not None and row.etag != if_match:
                    raise ResourcePreconditionFailedError(f"if-match precondition failed: {path}")
                return _row_to_info(row)
            # conditional UPDATE on version. new_version is computed in
            # Python from the SELECTed row, but the conditional WHERE makes the
            # assignment safe: if another writer bumped version first, our
            # UPDATE matches 0 rows.
            expected_version = row.version
            new_version = max(row.version or 0, row.whiteout_version or 0) + 1
            updated = await self._conditional_update_row(
                session, path, expected_version, content, content_type, metadata,
                new_version=new_version, if_match=if_match,
            )
            if updated is None:
                if if_match is not None:
                    # the etag precondition failed inside the DB WHERE.
                    raise ResourcePreconditionFailedError(f"if-match precondition failed: {path}")
                return None  # retry-able conflict
            if bump_revision:
                await self._bump_revision(session)
            return _dict_to_info(updated)

    async def _put_with_retry(
        self,
        session: AsyncSession,
        path: ResourcePath,
        content: bytes,
        content_type: "str | None",
        metadata: "Mapping[str, object]",
        *,
        if_match: "str | None",
        if_none_match: bool,
        bump_revision: bool = True,
    ) -> ResourceInfo:
        """SELECT-then-conditional-UPDATE loop. Precondition failures raise
        immediately from ``_put_once`` (no retry). Retry-able conflicts (only
        reachable with no precondition set) loop until the conditional UPDATE
        matches, preserving the external "always wins" contract for unconditional
        puts without losing updates. ``bump_revision`` is forwarded to
        ``_put_once`` so composite operations (raw_move) can own the single
        revision bump themselves."""
        for _ in range(_CONFLICT_RETRIES):
            info = await self._put_once(
                session, path, content, content_type, metadata,
                if_match=if_match, if_none_match=if_none_match,
                bump_revision=bump_revision,
            )
            if info is not None:
                return info
            # Retry-able conflict (no precondition set). Expire the identity-map
            # cache so the next iteration's SELECT sees the winner's commit.
            session.expire_all()
        raise ResourcePreconditionFailedError(
            f"resource update conflict after {_CONFLICT_RETRIES} retries: {path}"
        )

    async def raw_put(self, path: ResourcePath, content: bytes, *, content_type: "str | None", metadata: "Mapping[str, object]"):
        async with self._session_factory() as session:
            async with session.begin():
                info = await self._put_with_retry(
                    session, path, content, content_type, metadata,
                    if_match=None, if_none_match=False,
                )
            return info

    async def raw_put_checked(
        self,
        path: ResourcePath,
        content: bytes,
        *,
        options: WriteOptions,
        request_hash: str,
    ) -> Resource:
        """Atomic precondition + idempotency + put in ONE transaction.

        The If-Match precondition enters the UPDATE WHERE clause,
        so the etag check is enforced by the DB rather than a Python pre-read.
        Concurrent writers both holding the same stale If-Match cannot both
        succeed: the conditional UPDATE serializes them at the row lock.
        If-None-Match is enforced both up-front (live-row existence check) and
        atomically by the unique-path constraint (a concurrent INSERT that lands
        between our check and our INSERT surfaces as IntegrityError, translated
        to a precondition failure by the except below).
        """
        idem_key = f"put:{options.idempotency_key}" if options.idempotency_key else None
        try:
            async with self._session_factory() as session:
                async with session.begin():
                    if idem_key is not None:
                        idem_result = await session.execute(select(IdempotencyRow).where(IdempotencyRow.key == idem_key))
                        idem_row = idem_result.scalar_one_or_none()
                        if idem_row is not None:
                            if idem_row.request_hash != request_hash:
                                raise IdempotencyConflictError(
                                    f"idempotency key {options.idempotency_key!r} reused with a different request"
                                )
                            cached_info = _idempotency_result_to_info(idem_row.result_json)
                            row = await self._get_row(session, path)
                            content_bytes = row.content if (row is not None and row.deleted_at is None) else content
                            return Resource(info=cached_info, content=content_bytes)
                    info = await self._put_with_retry(
                        session, path, content, options.content_type, options.metadata,
                        if_match=options.if_match, if_none_match=options.if_none_match,
                    )
                    if idem_key is not None:
                        await self._save_idempotency_row(session, idem_key, request_hash, info)
                    return Resource(info=info, content=content)
        except IntegrityError as exc:
            # Concurrent put won the path race between our (empty) precondition
            # read and our insert: the unique-path constraint caught it. Surface
            # it as a precondition failure so callers see a deterministic error
            # regardless of how the two transactions interleaved.
            raise ResourcePreconditionFailedError(
                f"resource already exists (concurrent write): {path}"
            ) from exc

    # ------------------------------------------------------------------
    # DELETE: same conditional pattern (If-Match in WHERE for consistency)
    # ------------------------------------------------------------------

    async def _conditional_delete_row(
        self,
        session: AsyncSession,
        path: ResourcePath,
        expected_version: int,
        *,
        if_match: "str | None",
    ) -> "bool":
        """Conditional UPDATE that marks a live row as masked: ``WHERE path =
        :path AND version = :expected [AND etag = :if_match] AND deleted_at IS
        NULL``. Returns True if a row was masked, False on 0-row match (already
        masked, missing, or precondition failure)."""
        conditions = [
            ResourceRow.path == path.value,
            ResourceRow.version == expected_version,
            ResourceRow.deleted_at.is_(None),
        ]
        if if_match is not None:
            conditions.append(ResourceRow.etag == if_match)
        stmt = (
            update(ResourceRow)
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

    async def _apply_delete_unconditional(self, session: AsyncSession, path: ResourcePath) -> "ResourceInfo | None":
        """Unconditional delete used by the legacy raw_delete path. Loops:
        SELECT then mask-the-live-row via conditional UPDATE on version. If a
        concurrent writer bumps version first, our UPDATE misses and we retry
        against the new committed state. A row that is already masked gets its
        whiteout counter bumped atomically; a row that doesn't exist gets a
        tombstone inserted (unique-path constraint backstops concurrent creates)."""
        for _ in range(_CONFLICT_RETRIES):
            row = await self._get_row(session, path)
            if row is None:
                # No row at all: seed a tombstone so future reads see Masked.
                session.add(ResourceRow(
                    path=path.value, kind="file", etag="", version=0, content_type=None, size=0,
                    content=b"", modified_at=datetime.now(timezone.utc), metadata_json="{}",
                    deleted_at=datetime.now(timezone.utc), whiteout_version=1,
                ))
                await self._bump_revision(session)
                return None
            if row.deleted_at is not None:
                # Already masked: atomically bump the whiteout counter so the
                # lineage version keeps advancing (matches legacy semantics).
                stmt = (
                    update(ResourceRow)
                    .where(ResourceRow.path == path.value, ResourceRow.version == row.version)
                    .values(whiteout_version=(ResourceRow.whiteout_version or 0) + 1)
                    .execution_options(synchronize_session=False)
                )
                await session.execute(stmt)
                await self._bump_revision(session)
                return None
            # Live row: conditional mask on version.If a concurrent writer bumps
            # version first, our UPDATE matches 0 rows and we retry.
            removed_info = _row_to_info(row)
            masked = await self._conditional_delete_row(session, path, row.version, if_match=None)
            if masked:
                await self._bump_revision(session)
                return removed_info
            # Lost the race; expire and retry against the new committed state.
            session.expire_all()
        return None

    async def raw_delete(self, path: ResourcePath) -> "ResourceInfo | None":
        async with self._session_factory() as session:
            async with session.begin():
                removed_info = await self._apply_delete_unconditional(session, path)
            return removed_info

    async def raw_delete_checked(
        self,
        path: ResourcePath,
        *,
        options: WriteOptions,
        request_hash: str,
    ) -> None:
        """Atomic precondition + idempotency + delete in ONE transaction. If-Match
        is pushed into the UPDATE WHERE (consistent with raw_put_checked):
        two concurrent deletes holding the same stale If-Match cannot
        both succeed."""
        idem_key = f"delete:{options.idempotency_key}" if options.idempotency_key else None
        async with self._session_factory() as session:
            async with session.begin():
                if idem_key is not None:
                    idem_result = await session.execute(select(IdempotencyRow).where(IdempotencyRow.key == idem_key))
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
                removed_info: "ResourceInfo | None" = None
                masked_any = False
                for _ in range(_CONFLICT_RETRIES):
                    row = await self._get_row(session, path)
                    if row is None or row.deleted_at is not None:
                        # Missing or already masked. If-Match requires a live row.
                        if options.if_match is not None:
                            raise ResourcePreconditionFailedError(f"if-match precondition failed: {path}")
                        # No-op for the caller, but ensure a tombstone exists so
                        # subsequent reads see Masked (matches legacy semantics).
                        if row is None:
                            session.add(ResourceRow(
                                path=path.value, kind="file", etag="", version=0, content_type=None,
                                size=0, content=b"", modified_at=datetime.now(timezone.utc),
                                metadata_json="{}", deleted_at=datetime.now(timezone.utc), whiteout_version=1,
                            ))
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
                        raise ResourcePreconditionFailedError(f"if-match precondition failed: {path}")
                    session.expire_all()
                if not masked_any:
                    raise ResourcePreconditionFailedError(
                        f"resource delete conflict after {_CONFLICT_RETRIES} retries: {path}"
                    )
                if idem_key is not None:
                    await self._save_idempotency_row(session, idem_key, request_hash, removed_info)

    # ------------------------------------------------------------------
    # MOVE: ONE transaction
    # ------------------------------------------------------------------

    async def raw_move(
        self,
        source: ResourcePath,
        target: ResourcePath,
        *,
        options: WriteOptions,
    ) -> MoveResult:
        """Atomic MOVE in ONE transaction: load+lock source,
        validate, write target, whiteout source, bump revision once. All four
        mutations commit or roll back together, so a concurrent reader can
        never observe the intermediate states a decomposed put+delete would
        expose (target written while source still live = duplicate; source
        masked while target missing = data loss). The revision counter bumps
        exactly once -- observable proof of single-transaction atomicity (a
        put+delete decomposition would bump twice).

        Source precondition: must exist and be live. Target preconditions:
        options.if_match / options.if_none_match are enforced against the
        target row via _put_with_retry (same code path as raw_put_checked, so
        If-Match enters the conditional UPDATE WHERE)."""
        async with self._session_factory() as session:
            async with session.begin():
                source_row = await self._get_row(session, source)
                if source_row is None or source_row.deleted_at is not None:
                    raise ResourcePreconditionFailedError(f"cannot move missing resource: {source}")
                source_content = source_row.content
                source_info = _row_to_info(source_row)

                # Write target (INSERT or conditional UPDATE). Runs in this
                # transaction's snapshot: no concurrent writer can interleave.
                # bump_revision=False: raw_move owns the single revision bump
                # for the whole composite operation -- the
                # target write and source mask together count as ONE state
                # change, so the counter advances exactly once.
                target_info = await self._put_with_retry(
                    session, target, source_content, source_info.content_type, dict(source_info.metadata),
                    if_match=options.if_match, if_none_match=options.if_none_match,
                    bump_revision=False,
                )

                # Whiteout source: conditional mask on the version we read.
                # Inside one transaction the source row cannot have changed
                # since our SELECT, so the conditional UPDATE always matches;
                # masked=False would indicate a bug or external mutation.
                masked = await self._conditional_delete_row(session, source, source_row.version, if_match=None)
                if not masked:
                    raise ResourcePreconditionFailedError(f"source changed during move: {source}")

                # One revision bump for the whole move.
                await self._bump_revision(session)
                return Resource(info=target_info, content=source_content)

    # ------------------------------------------------------------------
    # Idempotency + revision readers (unchanged behavior)
    # ------------------------------------------------------------------

    async def _save_idempotency_row(
        self,
        session: AsyncSession,
        key: str,
        request_hash: str,
        info: "ResourceInfo | None",
    ) -> None:
        result_json = None
        if info is not None:
            result_json = json.dumps({
                "path": info.path.value, "kind": info.kind.value, "etag": info.etag,
                "version": info.version, "content_type": info.content_type,
                "size": info.size, "modified_at": info.modified_at.isoformat(),
                "metadata": dict(info.metadata),
            })
        result = await session.execute(select(IdempotencyRow).where(IdempotencyRow.key == key))
        row = result.scalar_one_or_none()
        if row is None:
            session.add(IdempotencyRow(key=key, request_hash=request_hash, result_json=result_json))
        else:
            row.request_hash = request_hash
            row.result_json = result_json

    async def revision(self) -> int:
        async with self._session_factory() as session:
            row = await session.get(RevisionRow, 1)
            return row.value if row is not None else 0

    async def get_idempotency(self, key: str) -> "IdempotencyRecord | None":
        async with self._session_factory() as session:
            result = await session.execute(select(IdempotencyRow).where(IdempotencyRow.key == key))
            row = result.scalar_one_or_none()
            if row is None:
                return None
            result_info = None
            if row.result_json is not None:
                raw = json.loads(row.result_json)
                result_info = ResourceInfo(
                    path=ResourcePath(raw["path"]), kind=ResourceKind(raw["kind"]), etag=raw["etag"],
                    version=raw["version"], content_type=raw["content_type"], size=raw["size"],
                    modified_at=datetime.fromisoformat(raw["modified_at"]), metadata=raw["metadata"],
                )
            return IdempotencyRecord(key=row.key, request_hash=row.request_hash, result=result_info)

    async def put_idempotency(self, record: IdempotencyRecord) -> None:
        result_json = None
        if record.result is not None:
            result_json = json.dumps({
                "path": record.result.path.value, "kind": record.result.kind.value, "etag": record.result.etag,
                "version": record.result.version, "content_type": record.result.content_type,
                "size": record.result.size, "modified_at": record.result.modified_at.isoformat(),
                "metadata": dict(record.result.metadata),
            })
        async with self._session_factory() as session:
            async with session.begin():
                result = await session.execute(select(IdempotencyRow).where(IdempotencyRow.key == record.key))
                row = result.scalar_one_or_none()
                if row is None:
                    session.add(IdempotencyRow(key=record.key, request_hash=record.request_hash, result_json=result_json))
                else:
                    row.request_hash = record.request_hash
                    row.result_json = result_json
