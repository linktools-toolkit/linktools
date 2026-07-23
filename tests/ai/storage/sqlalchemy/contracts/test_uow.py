#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SqlAlchemy Unit-of-Work contract, parametrized over the supported dialects.
Covers the scenario "幂等记录和 Asset mutation 一起 rollback": when a UoW
transaction aborts, BOTH the asset row AND its idempotency record roll back
together (they share one transaction), so a later replay with the same
idempotency key re-executes instead of returning a stale cached result."""

import asyncio

import pytest
import pytest_asyncio

from linktools.ai.asset.models import WriteOptions
from linktools.ai.asset.path import AssetPath
from linktools.ai.storage import SqlAlchemyStorage
from linktools.ai.storage.sqlalchemy.models import Base

# Reuse the dialect-parametrized builder + DSN-skip logic from the shared
# contracts conftest so this file exercises SQLite always and MySQL/PostgreSQL
# when their TEST_*_DSN env vars are set.
from .conftest import _build, _DIALECTS

pytestmark = pytest.mark.asyncio


@pytest_asyncio.fixture(params=_DIALECTS)
async def sql_storage(request, tmp_path):
    engine, session_factory = await _build(request.param, tmp_path)
    storage = SqlAlchemyStorage(
        session_factory=session_factory, blobs_root=tmp_path / "blobs"
    )
    yield storage
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
    except Exception:
        pass
    await engine.dispose()


async def test_idempotency_record_rolls_back_with_asset_mutation(sql_storage):
    # Within one UoW: write an asset WITH an idempotency key, then abort. Both
    # the asset row and the idempotency record must disappear together -- a
    # later replay with the same key must execute again, not return the cached
    # (rolled-back) result.
    path = AssetPath("/contract/uow-rollback.txt")

    async def _abort():
        async with sql_storage.transaction() as tx:
            await tx.assets.put(path, b"first", options=WriteOptions(idempotency_key="uow-k1"))
            # Visible inside the UoW before abort.
            assert (await tx.assets.get(path)) is not None
            raise RuntimeError("abort the unit of work")

    with pytest.raises(RuntimeError, match="abort"):
        await _abort()

    # After rollback: the asset is gone...
    assert await sql_storage.assets.get(path) is None
    # ...and the idempotency record is gone too -- the next call with the same
    # key is a fresh execution (different content succeeds), proving the record
    # did not survive the aborted transaction.
    await sql_storage.assets.put(path, b"second", options=WriteOptions(idempotency_key="uow-k1"))
    assert (await sql_storage.assets.get(path)).content == b"second"


async def test_idempotency_record_commits_with_asset_mutation(sql_storage):
    # The converse: a committed UoW persists both the asset and the idempotency
    # record, so a replay with the same key AND same content returns the cached
    # result (no second write -- the etag is unchanged).
    path = AssetPath("/contract/uow-commit.txt")

    async with sql_storage.transaction() as tx:
        first = await tx.assets.put(path, b"v1", options=WriteOptions(idempotency_key="uow-k2"))

    second = await sql_storage.assets.put(path, b"v1", options=WriteOptions(idempotency_key="uow-k2"))
    assert first.info.etag == second.info.etag
