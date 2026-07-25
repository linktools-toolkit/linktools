#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SqlAlchemyObjectBackend: DB-backed ObjectWriterBackend + history + tx.

Concurrency model (ported from the legacy asset SQLAlchemy backend's proven
pattern -- see :mod:`storage.sqlalchemy.asset`): the revision counter bumps
atomically via a portable UPDATE + SELECT loop (no dialect-specific upsert, no
RETURNING -- MySQL lacks RETURNING on UPDATE); a write's CAS precondition
(If-Match / version) enters the UPDATE's WHERE clause so the DB -- not a
Python pre-read -- enforces it; an idempotent no-op PUT re-validates via a
zero-effect UPDATE fence (:meth:`_confirm_unchanged`) before returning, so it
never hands back an already-stale read. Every checked op writes exactly one
``StorageObjectVersionRow`` (append-only, never updated) except a genuine
no-op replay, which writes none -- history is intrinsic to the write path,
not reconstructed after the fact.

Unlike the legacy asset backend, this backend DOES implement
``TransactionalObjectBackend``: ``transaction()`` binds every checked op
inside it to one session + one open transaction, so multiple keys commit or
roll back together and share a single revision bump (the SAME
``session.begin()``/rollback machinery the legacy backend already leans on
for its own atomic single-op writes, extended to span multiple calls)."""

from __future__ import annotations

import hashlib
import json
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from hashlib import sha256
from typing import AsyncIterator

from sqlalchemy import or_, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from ...object.errors import (
    StorageIdempotencyConflictError,
    StorageObjectNotFoundError,
    StoragePreconditionFailedError,
)
from ...object.models import (
    Depth,
    Found,
    Masked,
    Missing,
    ObjectInfo,
    ObjectPage,
    ObjectVersionPage,
    StorageKey,
    StoredObject,
    WriteOptions,
)
from ...sqlalchemy.dialects import resolve_dialect_strategy
from .models import (
    Base,
    StorageObjectIdempotencyRow,
    StorageObjectRevisionRow,
    StorageObjectRow,
    StorageObjectVersionRow,
)


def _key_hash(key: StorageKey) -> bytes:
    return hashlib.sha256(key.value.encode("utf-8")).digest()


def _idempotency_key_hash(idem_key: str) -> bytes:
    return hashlib.sha256(idem_key.encode("utf-8")).digest()


def _row_to_info(row: StorageObjectRow) -> ObjectInfo:
    return ObjectInfo(
        key=StorageKey(row.key),
        etag=row.etag,
        version=row.version,
        commit_revision=row.commit_revision,
        content_type=row.content_type,
        size=row.size,
        modified_at=row.modified_at,
        metadata=json.loads(row.metadata_json),
    )


def _matches_depth(prefix: StorageKey, candidate: StorageKey, depth: "Depth") -> bool:
    if not candidate.is_under(prefix):
        return False
    if depth is Depth.INFINITY:
        return True
    if prefix.is_root:
        rel_depth = len(candidate._segments)
    elif candidate.value == prefix.value:
        rel_depth = 0
    else:
        rel_depth = len(candidate._segments) - len(prefix._segments)
    return rel_depth == 0 if depth is Depth.ZERO else rel_depth <= 1


# Bounded retry for the conflict loop (SELECT-then-conditional-UPDATE). Two
# concurrent writers cannot loop forever: each retry sees the winner's commit
# and either succeeds on the next conditional UPDATE or hits a precondition.
_CONFLICT_RETRIES = 8

_LIST_BATCH = 256


class SqlAlchemyObjectBackend:
    """Implements ObjectReaderBackend + ObjectWriterBackend +
    TransactionalObjectBackend + VersionedObjectBackend.

    When ``session`` is bound (external Unit-of-Work participation, e.g. a
    caller composing this backend into a multi-store transactional scope it
    owns), every read AND write permanently reuses that session instead of
    opening/closing its own -- writes flush (not commit/rollback; the
    surrounding UoW owns that), so an object mutation commits or rolls back
    together with every other store sharing the session. A bound backend
    does not support its OWN ``transaction()`` (the UoW already is one)."""

    def __init__(self, *, session_factory, strategy=None, session=None) -> None:
        self._session_factory = session_factory
        self.backend_id = "primary"
        self._strategy = strategy or resolve_dialect_strategy(session_factory)
        self._tx_session: "AsyncSession | None" = session
        self._tx_revision: "int | None" = None

    # --- session plumbing ----------------------------------------------------

    @asynccontextmanager
    async def _read_session(self) -> "AsyncIterator[AsyncSession]":
        if self._tx_session is not None:
            yield self._tx_session
            return
        async with self._session_factory() as session:
            yield session

    @asynccontextmanager
    async def _write_session(self) -> "AsyncIterator[AsyncSession]":
        if self._tx_session is not None:
            yield self._tx_session
            return
        async with self._session_factory() as session:
            async with session.begin():
                yield session

    @asynccontextmanager
    async def transaction(self) -> "AsyncIterator[None]":
        if self._tx_session is not None:
            raise StorageObjectNotFoundError("nested transaction() is not supported")
        async with self._session_factory() as session:
            async with session.begin():
                self._tx_session = session
                self._tx_revision = None
                try:
                    yield
                finally:
                    self._tx_session = None
                    self._tx_revision = None

    async def _get_row(
        self, session: AsyncSession, key: StorageKey
    ) -> "StorageObjectRow | None":
        result = await session.execute(
            select(StorageObjectRow).where(StorageObjectRow.key_hash == _key_hash(key))
        )
        return result.scalar_one_or_none()

    # --- ObjectReaderBackend -------------------------------------------------

    async def raw_get(self, key: StorageKey, *, include_content: bool = True):
        async with self._read_session() as session:
            row = await self._get_row(session, key)
            if row is None:
                return Missing
            if row.tombstone:
                return Masked(key=key, version=row.version, commit_revision=row.commit_revision)
            content = row.content if include_content else b""
            return Found(StoredObject(info=_row_to_info(row), content=content))

    async def raw_stat(self, key: StorageKey) -> "ObjectInfo | None":
        async with self._read_session() as session:
            row = await self._get_row(session, key)
            if row is None or row.tombstone:
                return None
            return _row_to_info(row)

    async def raw_list(
        self, prefix: StorageKey, *, depth: "Depth", limit: int, cursor: "str | None"
    ) -> ObjectPage:
        if depth is Depth.ZERO:
            async with self._read_session() as session:
                row = await self._get_row(session, prefix)
            items = [_row_to_info(row)] if (row is not None and not row.tombstone) else []
            return ObjectPage(items=tuple(items), next_cursor=None)

        prefix_str = prefix.value.rstrip("/") + "/"
        escaped = prefix_str.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        candidate_clause = or_(
            StorageObjectRow.key == prefix.value,
            StorageObjectRow.key.like(f"{escaped}%", escape="\\"),
        )
        items: "list[ObjectInfo]" = []
        scan_cursor = cursor
        while True:
            conditions = [candidate_clause, StorageObjectRow.tombstone.is_(False)]
            if scan_cursor is not None:
                conditions.append(StorageObjectRow.key > scan_cursor)
            async with self._read_session() as session:
                result = await session.execute(
                    select(StorageObjectRow)
                    .where(*conditions)
                    .order_by(StorageObjectRow.key)
                    .limit(_LIST_BATCH)
                )
                rows = result.scalars().all()
            if not rows:
                break
            last_scanned = rows[-1].key
            for row in rows:
                info = _row_to_info(row)
                if _matches_depth(prefix, info.key, depth):
                    items.append(info)
                    if len(items) > limit:
                        break
            if len(items) > limit:
                break
            if len(rows) < _LIST_BATCH:
                break
            scan_cursor = last_scanned
        next_cursor = items[limit - 1].key.value if len(items) > limit else None
        return ObjectPage(items=tuple(items[:limit]), next_cursor=next_cursor)

    # --- revision --------------------------------------------------------------

    async def _bump_revision_row(self, session: AsyncSession) -> int:
        for _ in range(_CONFLICT_RETRIES):
            stmt = (
                update(StorageObjectRevisionRow)
                .where(StorageObjectRevisionRow.id == 1)
                .values(value=StorageObjectRevisionRow.value + 1)
                .execution_options(synchronize_session=False)
            )
            result = await session.execute(stmt)
            if result.rowcount:
                return (
                    await session.execute(
                        select(StorageObjectRevisionRow.value).where(
                            StorageObjectRevisionRow.id == 1
                        )
                    )
                ).scalar_one()
            try:
                async with session.begin_nested():
                    session.add(StorageObjectRevisionRow(id=1, value=1))
                return 1
            except IntegrityError:
                continue
        raise StoragePreconditionFailedError(
            f"revision bump conflict after {_CONFLICT_RETRIES} retries"
        )

    async def _bump_revision(self, session: AsyncSession) -> int:
        """Within an active transaction() the whole multi-op tx shares ONE
        bump (cached on first mutation); outside one, every checked op bumps
        its own single-op transaction independently."""
        if self._tx_session is not None:
            if self._tx_revision is None:
                self._tx_revision = await self._bump_revision_row(session)
            return self._tx_revision
        return await self._bump_revision_row(session)

    async def revision(self) -> str:
        async with self._read_session() as session:
            row = await session.get(StorageObjectRevisionRow, 1)
            return str(row.value if row is not None else 0)

    # --- history -----------------------------------------------------------

    async def _record_version(
        self,
        session: AsyncSession,
        *,
        key: StorageKey,
        etag: str,
        version: int,
        content_type: "str | None",
        size: int,
        content: bytes,
        modified_at: datetime,
        metadata_json: str,
        tombstone: bool,
        commit_revision: int,
    ) -> ObjectInfo:
        """Append one history row and return the ObjectInfo for this write,
        built directly from the values just written -- NOT from a re-SELECT.
        A conditional UPDATE runs with ``synchronize_session=False`` (so the
        DB-level WHERE clause -- not a Python pre-read -- enforces the CAS
        precondition), which leaves any identity-mapped ``StorageObjectRow``
        for this key stale; a follow-up ORM SELECT-by-identity would silently
        hand back that stale cached instance instead of the row just written.
        ``expire_all()`` clears the identity map so any LATER read in this
        session (e.g. a subsequent op inside the same transaction()) re-fetches
        from the DB rather than the stale cache."""
        session.add(
            StorageObjectVersionRow(
                key=key.value,
                key_hash=_key_hash(key),
                version=version,
                etag=etag,
                content_type=content_type,
                size=size,
                content=None if tombstone else content,
                modified_at=modified_at,
                metadata_json=metadata_json,
                tombstone=tombstone,
                commit_revision=commit_revision,
            )
        )
        # Explicit flush (rather than relying on the next statement's
        # autoflush) -- mixing a pending ORM add() with a subsequent raw Core
        # execute() (as raw_move_checked's source tombstone UPDATE does) can
        # trigger autoflush outside the async greenlet context under
        # aiosqlite, raising MissingGreenlet.
        await session.flush()
        session.expire_all()
        return ObjectInfo(
            key=key,
            etag=etag,
            version=version,
            commit_revision=commit_revision,
            content_type=content_type,
            size=size,
            modified_at=modified_at,
            metadata=json.loads(metadata_json),
        )

    # --- idempotency ---------------------------------------------------------

    async def _read_idempotency(
        self, session: AsyncSession, idem_key: str
    ) -> "StorageObjectIdempotencyRow | None":
        result = await session.execute(
            select(StorageObjectIdempotencyRow).where(
                StorageObjectIdempotencyRow.key_hash == _idempotency_key_hash(idem_key)
            )
        )
        return result.scalar_one_or_none()

    async def _save_idempotency(
        self,
        session: AsyncSession,
        idem_key: str,
        request_hash: str,
        info: "ObjectInfo | None",
    ) -> None:
        result_json = None
        if info is not None:
            result_json = json.dumps(
                {
                    "key": info.key.value,
                    "etag": info.etag,
                    "version": info.version,
                    "commit_revision": info.commit_revision,
                    "content_type": info.content_type,
                    "size": info.size,
                    "modified_at": info.modified_at.isoformat(),
                    "metadata": dict(info.metadata),
                }
            )
        session.add(
            StorageObjectIdempotencyRow(
                key_hash=_idempotency_key_hash(idem_key),
                key=idem_key,
                request_hash=request_hash,
                result_json=result_json,
            )
        )

    @staticmethod
    def _idempotency_result_to_info(result_json: "str | None") -> "ObjectInfo | None":
        if result_json is None:
            return None
        raw = json.loads(result_json)
        return ObjectInfo(
            key=StorageKey(raw["key"]),
            etag=raw["etag"],
            version=raw["version"],
            commit_revision=raw["commit_revision"],
            content_type=raw["content_type"],
            size=raw["size"],
            modified_at=datetime.fromisoformat(raw["modified_at"]),
            metadata=raw["metadata"],
        )

    # --- PUT -------------------------------------------------------------------

    async def _confirm_unchanged(
        self, session: AsyncSession, key: StorageKey, *, expected_version: int, expected_etag: str
    ) -> bool:
        stmt = (
            update(StorageObjectRow)
            .where(
                StorageObjectRow.key_hash == _key_hash(key),
                StorageObjectRow.version == expected_version,
                StorageObjectRow.etag == expected_etag,
            )
            .values(version=StorageObjectRow.version)
            .execution_options(synchronize_session=False)
        )
        result = await session.execute(stmt)
        return result.rowcount == 1

    async def _put_once(
        self,
        session: AsyncSession,
        key: StorageKey,
        content: bytes,
        *,
        if_match: "str | None",
        if_none_match: "bool | None",
        content_type: "str | None",
        metadata_json: str,
    ) -> "ObjectInfo | None":
        row = await self._get_row(session, key)
        now = datetime.now(timezone.utc)
        etag = sha256(content).hexdigest()
        if row is None:
            if if_match is not None:
                raise StoragePreconditionFailedError(f"if_match failed: {key.value!r} does not exist")
            commit_revision = await self._bump_revision(session)
            conflict = await self._strategy.execute_conflict_insert(
                session,
                StorageObjectRow,
                {
                    "key": key.value,
                    "key_hash": _key_hash(key),
                    "etag": etag,
                    "version": 1,
                    "content_type": content_type,
                    "size": len(content),
                    "content": content,
                    "modified_at": now,
                    "metadata_json": metadata_json,
                    "tombstone": False,
                    "commit_revision": commit_revision,
                },
                index_elements=["key_hash"],
            )
            if conflict:
                return None  # retry-able: a concurrent writer just inserted
            return await self._record_version(
                session,
                key=key,
                etag=etag,
                version=1,
                content_type=content_type,
                size=len(content),
                content=content,
                modified_at=now,
                metadata_json=metadata_json,
                tombstone=False,
                commit_revision=commit_revision,
            )
        if if_none_match and not row.tombstone:
            raise StoragePreconditionFailedError(f"if_none_match failed: {key.value!r} already exists")
        if (
            not row.tombstone
            and row.content == content
            and row.content_type == content_type
            and row.metadata_json == metadata_json
        ):
            if if_match is not None and row.etag != if_match:
                raise StoragePreconditionFailedError(f"if_match failed: {key.value!r} etag mismatch")
            confirmed = await self._confirm_unchanged(
                session, key, expected_version=row.version, expected_etag=row.etag
            )
            if not confirmed:
                return None  # retry-able: the row changed after our read
            return _row_to_info(row)
        if if_match is not None and (row.tombstone or row.etag != if_match):
            raise StoragePreconditionFailedError(f"if_match failed: {key.value!r} etag mismatch")
        expected_version = row.version
        new_version = row.version + 1
        commit_revision = await self._bump_revision(session)
        conditions = [
            StorageObjectRow.key_hash == _key_hash(key),
            StorageObjectRow.version == expected_version,
        ]
        if if_match is not None:
            conditions.append(StorageObjectRow.etag == if_match)
        stmt = (
            update(StorageObjectRow)
            .where(*conditions)
            .values(
                etag=etag,
                version=new_version,
                content_type=content_type,
                size=len(content),
                content=content,
                modified_at=now,
                metadata_json=metadata_json,
                tombstone=False,
                commit_revision=commit_revision,
            )
            .execution_options(synchronize_session=False)
        )
        result = await session.execute(stmt)
        if result.rowcount != 1:
            return None  # retry-able: lost the race
        return await self._record_version(
            session,
            key=key,
            etag=etag,
            version=new_version,
            content_type=content_type,
            size=len(content),
            content=content,
            modified_at=now,
            metadata_json=metadata_json,
            tombstone=False,
            commit_revision=commit_revision,
        )

    async def _put_with_retry(
        self,
        session: AsyncSession,
        key: StorageKey,
        content: bytes,
        *,
        if_match: "str | None",
        if_none_match: "bool | None",
        content_type: "str | None",
        metadata_json: str,
    ) -> ObjectInfo:
        for _ in range(_CONFLICT_RETRIES):
            info = await self._put_once(
                session,
                key,
                content,
                if_match=if_match,
                if_none_match=if_none_match,
                content_type=content_type,
                metadata_json=metadata_json,
            )
            if info is not None:
                return info
            session.expire_all()
        raise StoragePreconditionFailedError(
            f"object update conflict after {_CONFLICT_RETRIES} retries: {key.value}"
        )

    async def raw_put_checked(
        self, key: StorageKey, content: bytes, *, options: WriteOptions, request_hash: str
    ) -> StoredObject:
        idem_key = f"put:{key.value}:{options.idempotency_key}" if options.idempotency_key else None
        metadata_json = json.dumps(dict(options.metadata or {}), sort_keys=True)
        async with self._write_session() as session:
            if idem_key is not None:
                idem_row = await self._read_idempotency(session, idem_key)
                if idem_row is not None:
                    if idem_row.request_hash != request_hash:
                        raise StorageIdempotencyConflictError(
                            f"idempotency key {options.idempotency_key!r} replayed with a different request"
                        )
                    info = self._idempotency_result_to_info(idem_row.result_json)
                    row = await self._get_row(session, key)
                    content_bytes = row.content if (row is not None and not row.tombstone) else content
                    return StoredObject(info=info, content=content_bytes)
            info = await self._put_with_retry(
                session,
                key,
                content,
                if_match=options.if_match,
                if_none_match=options.if_none_match,
                content_type=options.content_type,
                metadata_json=metadata_json,
            )
            if idem_key is not None:
                await self._save_idempotency(session, idem_key, request_hash, info)
            return StoredObject(info=info, content=content)

    # --- DELETE ------------------------------------------------------------

    async def _raw_delete_once(
        self, session: AsyncSession, key: StorageKey, *, if_match: "str | None"
    ) -> "bool | None":
        """Returns True on a live row masked, False on an already-missing/
        already-tombstoned key (idempotent no-op), None on a retry-able race."""
        row = await self._get_row(session, key)
        if row is None or row.tombstone:
            if if_match is not None:
                raise StoragePreconditionFailedError(f"if_match failed: {key.value!r} does not exist")
            return False
        if if_match is not None and row.etag != if_match:
            raise StoragePreconditionFailedError(f"if_match failed: {key.value!r} etag mismatch")
        commit_revision = await self._bump_revision(session)
        new_version = row.version + 1
        now = datetime.now(timezone.utc)
        stmt = (
            update(StorageObjectRow)
            .where(
                StorageObjectRow.key_hash == _key_hash(key),
                StorageObjectRow.version == row.version,
            )
            .values(
                version=new_version,
                tombstone=True,
                content=b"",
                etag="",
                modified_at=now,
                commit_revision=commit_revision,
            )
            .execution_options(synchronize_session=False)
        )
        result = await session.execute(stmt)
        if result.rowcount != 1:
            return None
        await self._record_version(
            session,
            key=key,
            etag="",
            version=new_version,
            content_type=None,
            size=0,
            content=b"",
            modified_at=now,
            metadata_json="{}",
            tombstone=True,
            commit_revision=commit_revision,
        )
        return True

    async def raw_delete_checked(
        self, key: StorageKey, *, options: WriteOptions, request_hash: str
    ) -> None:
        idem_key = f"delete:{key.value}:{options.idempotency_key}" if options.idempotency_key else None
        async with self._write_session() as session:
            if idem_key is not None:
                idem_row = await self._read_idempotency(session, idem_key)
                if idem_row is not None:
                    if idem_row.request_hash != request_hash:
                        raise StorageIdempotencyConflictError(
                            f"idempotency key {options.idempotency_key!r} replayed with a different request"
                        )
                    return None
            for _ in range(_CONFLICT_RETRIES):
                outcome = await self._raw_delete_once(session, key, if_match=options.if_match)
                if outcome is not None:
                    break
                session.expire_all()
            else:
                raise StoragePreconditionFailedError(
                    f"object delete conflict after {_CONFLICT_RETRIES} retries: {key.value}"
                )
            if idem_key is not None:
                await self._save_idempotency(session, idem_key, request_hash, None)
            return None

    # --- MOVE (one transaction) ---------------------------------------------

    async def raw_move_checked(
        self, source: StorageKey, target: StorageKey, *, options: WriteOptions, request_hash: str
    ) -> StoredObject:
        idem_key = (
            f"move:{source.value}:{target.value}:{options.idempotency_key}"
            if options.idempotency_key
            else None
        )
        async with self._write_session() as session:
            if idem_key is not None:
                idem_row = await self._read_idempotency(session, idem_key)
                if idem_row is not None:
                    if idem_row.request_hash != request_hash:
                        raise StorageIdempotencyConflictError(
                            f"idempotency key {options.idempotency_key!r} replayed with a different request"
                        )
                    info = self._idempotency_result_to_info(idem_row.result_json)
                    if info is None:
                        raise StorageObjectNotFoundError(source.value)
                    row = await self._get_row(session, target)
                    content_bytes = row.content if (row is not None and not row.tombstone) else b""
                    return StoredObject(info=info, content=content_bytes)

            source_row = await self._get_row(session, source)
            if source_row is None or source_row.tombstone:
                raise StorageObjectNotFoundError(source.value)
            source_content = source_row.content
            source_info = _row_to_info(source_row)
            # Captured now, not read off `source_row` again later: the target
            # write's _record_version() call below expires the whole session,
            # and touching an expired ORM attribute outside an awaited
            # context raises MissingGreenlet under aiosqlite.
            source_expected_version = source_row.version

            target_row = await self._get_row(session, target)
            if options.if_none_match and target_row is not None and not target_row.tombstone:
                raise StoragePreconditionFailedError(f"if_none_match failed: {target.value!r} already exists")
            if options.if_match is not None:
                if target_row is None or target_row.tombstone or target_row.etag != options.if_match:
                    raise StoragePreconditionFailedError(f"if_match failed: {target.value!r} etag mismatch")

            commit_revision = await self._bump_revision(session)
            now = datetime.now(timezone.utc)

            if target_row is None:
                conflict = await self._strategy.execute_conflict_insert(
                    session,
                    StorageObjectRow,
                    {
                        "key": target.value,
                        "key_hash": _key_hash(target),
                        "etag": source_info.etag,
                        "version": 1,
                        "content_type": source_info.content_type,
                        "size": source_info.size,
                        "content": source_content,
                        "modified_at": now,
                        "metadata_json": json.dumps(dict(source_info.metadata)),
                        "tombstone": False,
                        "commit_revision": commit_revision,
                    },
                    index_elements=["key_hash"],
                )
                if conflict:
                    raise StoragePreconditionFailedError(
                        f"move target changed concurrently: {target.value!r}"
                    )
                new_target_version = 1
            else:
                new_target_version = target_row.version + 1
                stmt = (
                    update(StorageObjectRow)
                    .where(
                        StorageObjectRow.key_hash == _key_hash(target),
                        StorageObjectRow.version == target_row.version,
                    )
                    .values(
                        etag=source_info.etag,
                        version=new_target_version,
                        content_type=source_info.content_type,
                        size=source_info.size,
                        content=source_content,
                        modified_at=now,
                        metadata_json=json.dumps(dict(source_info.metadata)),
                        tombstone=False,
                        commit_revision=commit_revision,
                    )
                    .execution_options(synchronize_session=False)
                )
                result = await session.execute(stmt)
                if result.rowcount != 1:
                    raise StoragePreconditionFailedError(
                        f"move target changed concurrently: {target.value!r}"
                    )
            target_metadata_json = json.dumps(dict(source_info.metadata))
            info = await self._record_version(
                session,
                key=target,
                etag=source_info.etag,
                version=new_target_version,
                content_type=source_info.content_type,
                size=source_info.size,
                content=source_content,
                modified_at=now,
                metadata_json=target_metadata_json,
                tombstone=False,
                commit_revision=commit_revision,
            )

            source_new_version = source_expected_version + 1
            source_stmt = (
                update(StorageObjectRow)
                .where(
                    StorageObjectRow.key_hash == _key_hash(source),
                    StorageObjectRow.version == source_expected_version,
                )
                .values(
                    version=source_new_version,
                    tombstone=True,
                    content=b"",
                    etag="",
                    modified_at=now,
                    commit_revision=commit_revision,
                )
                .execution_options(synchronize_session=False)
            )
            source_result = await session.execute(source_stmt)
            if source_result.rowcount != 1:
                raise StoragePreconditionFailedError(f"source changed during move: {source.value!r}")
            await self._record_version(
                session,
                key=source,
                etag="",
                version=source_new_version,
                content_type=None,
                size=0,
                content=b"",
                modified_at=now,
                metadata_json="{}",
                tombstone=True,
                commit_revision=commit_revision,
            )

            if idem_key is not None:
                await self._save_idempotency(session, idem_key, request_hash, info)
            return StoredObject(info=info, content=source_content)

    # --- VersionedObjectBackend ----------------------------------------------

    async def raw_get_version(self, key: StorageKey, version: int) -> "StoredObject | None":
        async with self._read_session() as session:
            result = await session.execute(
                select(StorageObjectVersionRow).where(
                    StorageObjectVersionRow.key_hash == _key_hash(key),
                    StorageObjectVersionRow.version == version,
                )
            )
            row = result.scalar_one_or_none()
        if row is None or row.tombstone:
            return None
        return StoredObject(
            info=ObjectInfo(
                key=key,
                etag=row.etag,
                version=row.version,
                commit_revision=row.commit_revision,
                content_type=row.content_type,
                size=row.size,
                modified_at=row.modified_at,
                metadata=json.loads(row.metadata_json),
            ),
            content=row.content or b"",
        )

    async def raw_get_at_revision(self, key: StorageKey, revision: int) -> "StoredObject | None":
        async with self._read_session() as session:
            result = await session.execute(
                select(StorageObjectVersionRow)
                .where(
                    StorageObjectVersionRow.key_hash == _key_hash(key),
                    StorageObjectVersionRow.commit_revision <= revision,
                )
                .order_by(StorageObjectVersionRow.version.desc())
                .limit(1)
            )
            row = result.scalar_one_or_none()
        if row is None or row.tombstone:
            return None
        return StoredObject(
            info=ObjectInfo(
                key=key,
                etag=row.etag,
                version=row.version,
                commit_revision=row.commit_revision,
                content_type=row.content_type,
                size=row.size,
                modified_at=row.modified_at,
                metadata=json.loads(row.metadata_json),
            ),
            content=row.content or b"",
        )

    async def raw_list_versions(
        self, key: StorageKey, *, limit: int = 100, cursor: "str | None" = None
    ) -> ObjectVersionPage:
        conditions = [StorageObjectVersionRow.key_hash == _key_hash(key)]
        if cursor is not None:
            conditions.append(StorageObjectVersionRow.version > int(cursor))
        async with self._read_session() as session:
            result = await session.execute(
                select(StorageObjectVersionRow)
                .where(*conditions)
                .order_by(StorageObjectVersionRow.version)
                .limit(limit + 1)
            )
            rows = result.scalars().all()
        page_rows = rows[:limit]
        items = tuple(
            ObjectInfo(
                key=key,
                etag=row.etag,
                version=row.version,
                commit_revision=row.commit_revision,
                content_type=row.content_type,
                size=row.size,
                modified_at=row.modified_at,
                metadata=json.loads(row.metadata_json),
            )
            for row in page_rows
        )
        next_cursor = str(page_rows[-1].version) if len(rows) > limit else None
        return ObjectVersionPage(items=items, next_cursor=next_cursor)

    async def raw_list_at_revision(self, prefix: StorageKey, revision: int) -> "tuple[ObjectInfo, ...]":
        async with self._read_session() as session:
            result = await session.execute(
                select(StorageObjectVersionRow)
                .where(StorageObjectVersionRow.commit_revision <= revision)
                .order_by(StorageObjectVersionRow.key_hash, StorageObjectVersionRow.version)
            )
            rows = result.scalars().all()
        latest_by_key: "dict[bytes, StorageObjectVersionRow]" = {}
        for row in rows:
            latest_by_key[row.key_hash] = row  # last write per key_hash wins (ordered by version)
        out: "list[ObjectInfo]" = []
        for row in latest_by_key.values():
            if row.tombstone:
                continue
            key = StorageKey(row.key)
            if not key.is_under(prefix):
                continue
            out.append(
                ObjectInfo(
                    key=key,
                    etag=row.etag,
                    version=row.version,
                    commit_revision=row.commit_revision,
                    content_type=row.content_type,
                    size=row.size,
                    modified_at=row.modified_at,
                    metadata=json.loads(row.metadata_json),
                )
            )
        out.sort(key=lambda i: i.key.value)
        return tuple(out)


class SqlAlchemyObjectStore:
    """Convenience: an ObjectStore pre-wired to a SqlAlchemyObjectBackend.

    Takes a caller-built ``session_factory`` (never a URL/engine -- core
    parses no DSN and constructs no engine; that is the downstream/SQLite-
    helper's job, per the adapter boundary)."""

    def __init__(self, *, session_factory) -> None:
        from ...object.store import ObjectStore

        self._session_factory = session_factory
        self._backend = SqlAlchemyObjectBackend(session_factory=session_factory)
        self._store = ObjectStore(primary=self._backend)
        self._schema_ready = False

    async def _ensure_schema(self) -> None:
        if self._schema_ready:
            return
        async with self._session_factory() as session:
            conn = await session.connection()
            await conn.run_sync(Base.metadata.create_all)
            await session.commit()
        self._schema_ready = True

    @property
    def backend(self) -> SqlAlchemyObjectBackend:
        return self._backend

    async def get(self, key: StorageKey) -> "StoredObject | None":
        await self._ensure_schema()
        return await self._store.get(key)

    async def stat(self, key: StorageKey) -> "ObjectInfo | None":
        await self._ensure_schema()
        return await self._store.stat(key)

    async def list(self, prefix: StorageKey, **kwargs) -> ObjectPage:
        await self._ensure_schema()
        return await self._store.list(prefix, **kwargs)

    async def revision(self) -> str:
        await self._ensure_schema()
        return await self._store.revision()

    async def put(self, key: StorageKey, content: bytes, **kwargs) -> StoredObject:
        await self._ensure_schema()
        return await self._store.put(key, content, **kwargs)

    async def delete(self, key: StorageKey, **kwargs) -> None:
        await self._ensure_schema()
        await self._store.delete(key, **kwargs)

    async def move(self, source: StorageKey, target: StorageKey, **kwargs) -> StoredObject:
        await self._ensure_schema()
        return await self._store.move(source, target, **kwargs)

    @asynccontextmanager
    async def transaction(self) -> "AsyncIterator[None]":
        await self._ensure_schema()
        async with self._backend.transaction():
            yield

    async def get_version(self, key: StorageKey, version: int) -> "StoredObject | None":
        await self._ensure_schema()
        return await self._backend.raw_get_version(key, version)

    async def list_versions(
        self, key: StorageKey, *, limit: int = 100, cursor: "str | None" = None
    ) -> ObjectVersionPage:
        await self._ensure_schema()
        return await self._backend.raw_list_versions(key, limit=limit, cursor=cursor)


__all__: "list[str]" = ["SqlAlchemyObjectBackend", "SqlAlchemyObjectStore"]
