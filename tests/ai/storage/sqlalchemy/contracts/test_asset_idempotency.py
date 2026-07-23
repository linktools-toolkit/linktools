#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SqlAlchemy asset idempotency contract, parametrized over the supported
dialects. Covers the idempotency scenarios: same key + same fingerprint
is idempotent (cached result returned), same key + different fingerprint
conflicts, and a non-unique IntegrityError is never mis-mapped to an asset
conflict (it must always re-raise)."""

import pytest
from sqlalchemy.exc import IntegrityError

from linktools.ai.asset.models import WriteOptions
from linktools.ai.asset.path import AssetPath
from linktools.ai.asset.store import AssetStore
from linktools.ai.errors import IdempotencyConflictError

pytestmark = pytest.mark.asyncio


def _put_opts(**kw):
    return WriteOptions(**kw)


async def test_idempotency_same_key_same_fingerprint_is_idempotent(sql_asset_backend):
    store = AssetStore(primary=sql_asset_backend)
    path = AssetPath("/contract/idem-same.txt")

    first = await store.put(path, b"x", options=_put_opts(idempotency_key="k1"))
    second = await store.put(path, b"x", options=_put_opts(idempotency_key="k1"))
    # Same request -> the cached result is returned; no second write.
    assert first.info.etag == second.info.etag


async def test_idempotency_same_key_different_fingerprint_conflicts(sql_asset_backend):
    store = AssetStore(primary=sql_asset_backend)
    path = AssetPath("/contract/idem-diff.txt")

    await store.put(path, b"x", options=_put_opts(idempotency_key="k2"))
    with pytest.raises(IdempotencyConflictError):
        await store.put(path, b"y", options=_put_opts(idempotency_key="k2"))


async def test_non_unique_integrity_error_is_not_swallowed(sql_asset_backend):
    # Insert a row directly with a NULL on a NOT-NULL column to provoke a
    # non-unique IntegrityError; it must surface (not be converted to an asset
    # conflict) -- a non-unique violation must always re-raise.
    from linktools.ai.storage.sqlalchemy.models import AssetRow

    async with sql_asset_backend._session_factory() as session:
        async with session.begin():
            with pytest.raises(IntegrityError):
                await sql_asset_backend._strategy.execute_conflict_insert(
                    session,
                    AssetRow,
                    {
                        "path": "/contract/bad.txt",
                        "kind": "file",
                        "etag": "e",
                        "version": 1,
                        "content_type": None,
                        "size": 0,
                        "content": b"",
                        "modified_at": None,  # NOT NULL violation
                        "metadata_json": "{}",
                        "deleted_at": None,
                        "whiteout_version": None,
                    },
                    index_elements=["path"],
                )
