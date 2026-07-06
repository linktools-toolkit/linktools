#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SqlAlchemyResourceBackend: DB-backed ResourceBackend. Each write (raw_put/
raw_delete) runs read-precondition-check -> mutate -> bump-revision -> commit in
one transaction (spec docs/linktools-ai.md section 16). Reads (raw_get/raw_propfind/
revision/get_idempotency) use their own short-lived session."""

import json
from datetime import datetime, timezone
from hashlib import sha256
from typing import Callable, Mapping

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from .models import ResourceRow, IdempotencyRow, RevisionRow
from ..resource.models import Depth, Found, IdempotencyRecord, Masked, Missing, Resource, ResourceInfo, ResourceKind, ResourcePage
from ..resource.path import ResourcePath


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

    async def raw_propfind(self, path: ResourcePath, *, depth: Depth, limit: int, cursor: "str | None") -> ResourcePage:
        # NOTE: cursor-based continuation is not yet implemented in Phase 1 -- `cursor`
        # is accepted for forward API compatibility but ignored; results are simply
        # truncated to `limit`. Real pagination is deferred to a later phase.
        prefix = path.value.rstrip("/") + "/"
        escaped_prefix = prefix.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        async with self._session_factory() as session:
            result = await session.execute(
                select(ResourceRow)
                .where(ResourceRow.path.like(f"{escaped_prefix}%", escape="\\"))
                .where(ResourceRow.deleted_at.is_(None))
            )
            items = []
            for row in result.scalars():
                rest = row.path[len(prefix):]
                if depth == Depth.ONE and "/" in rest:
                    continue
                items.append(_row_to_info(row))
            items.sort(key=lambda info: info.path.value)
            return ResourcePage(items=tuple(items[:limit]), cursor=None)

    async def _bump_revision(self, session: AsyncSession) -> None:
        row = await session.get(RevisionRow, 1)
        if row is None:
            session.add(RevisionRow(id=1, value=1))
        else:
            row.value += 1

    async def raw_put(self, path: ResourcePath, content: bytes, *, content_type: "str | None", metadata: "Mapping[str, object]"):
        async with self._session_factory() as session:
            async with session.begin():
                row = await self._get_row(session, path)
                if row is None:
                    version = 1
                    row = ResourceRow(path=path.value)
                    session.add(row)
                else:
                    # Version must be monotonic across both the live lineage and any
                    # prior whiteout: a path that was deleted and is now being
                    # recreated must not reuse a version number already observed by
                    # a reader before the delete.
                    version = max(row.version or 0, row.whiteout_version or 0) + 1
                row.kind = "file"
                row.etag = sha256(content).hexdigest()
                row.version = version
                row.content_type = content_type
                row.size = len(content)
                row.content = content
                row.modified_at = datetime.now(timezone.utc)
                row.metadata_json = json.dumps(dict(metadata))
                row.deleted_at = None
                row.whiteout_version = None
                await self._bump_revision(session)
                await session.flush()
                info = _row_to_info(row)
            return info

    async def raw_delete(self, path: ResourcePath) -> "ResourceInfo | None":
        async with self._session_factory() as session:
            async with session.begin():
                row = await self._get_row(session, path)
                removed_info = None
                if row is not None and row.deleted_at is None:
                    removed_info = _row_to_info(row)
                    prior_version = row.version
                    row.deleted_at = datetime.now(timezone.utc)
                    row.whiteout_version = prior_version + 1
                    row.content = b""
                elif row is None:
                    row = ResourceRow(
                        path=path.value, kind="file", etag="", version=0, content_type=None, size=0,
                        content=b"", modified_at=datetime.now(timezone.utc), metadata_json="{}",
                        deleted_at=datetime.now(timezone.utc), whiteout_version=1,
                    )
                    session.add(row)
                else:
                    row.whiteout_version = (row.whiteout_version or 0) + 1
                await self._bump_revision(session)
            return removed_info

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
