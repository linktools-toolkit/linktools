#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SqlAlchemy asset backend CRUD contract, parametrized over the supported
dialects (SQLite always; MySQL/PostgreSQL when their TEST_*_DSN env vars are
set). The concurrency, idempotency, and UoW scenarios live in their own files
per (test_asset_concurrency.py, test_asset_idempotency.py, test_uow.py);
this file holds the basic create/get/update/delete round-trip."""

import pytest

from linktools.ai.asset.models import WriteOptions
from linktools.ai.asset.path import AssetPath
from linktools.ai.asset.store import AssetStore

pytestmark = pytest.mark.asyncio


def _put_opts(**kw):
    return WriteOptions(**kw)


async def test_create_get_update_delete_roundtrip(sql_asset_backend):
    store = AssetStore(primary=sql_asset_backend)
    path = AssetPath("/contract/roundtrip.txt")

    await store.put(path, b"v1")
    assert (await store.get(path)).content == b"v1"

    info = await sql_asset_backend.raw_stat(path)
    await store.put(path, b"v2", options=_put_opts(if_match=info.etag))
    assert (await store.get(path)).content == b"v2"

    await store.delete(path)
    assert await store.get(path) is None
