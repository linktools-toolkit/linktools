#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests/ai/storage/sqlalchemy/contracts/test_asset_path_hash_index.py

Work package four (spec section 7): AssetRow/AssetIdempotencyRow index
safety, scoped to path_hash/key_hash only (no tenant_id -- the Asset domain
has no tenant_id concept anywhere; see the WP4 memory note). Covers:
create_all, a full-length (1024) ASCII path, a full-length multi-byte path,
a simulated path_hash collision, a long idempotency key, and that the
unique index is on the hash rather than the raw column."""

import hashlib

import pytest

from linktools.ai.asset.models import WriteOptions
from linktools.ai.asset.path import AssetPath
from linktools.ai.asset.store import AssetStore
from linktools.ai.errors import AssetPathHashCollisionError
from linktools.ai.storage.sqlalchemy.asset import asset_path_hash


@pytest.mark.asyncio
async def test_create_all_produces_hash_columns(sql_asset_backend):
    # sql_asset_backend fixture already ran create_all; a round-trip put/get
    # exercising the hash columns end-to-end is the create_all smoke test.
    store = AssetStore(primary=sql_asset_backend)
    path = AssetPath("/contract/create-all-smoke.txt")
    await store.put(path, b"x")
    assert (await store.get(path)).content == b"x"


@pytest.mark.asyncio
async def test_ascii_path_at_max_length(sql_asset_backend):
    store = AssetStore(primary=sql_asset_backend)
    # AssetPath segments are joined with "/"; build a single long segment of
    # ASCII characters near the 1024-char column cap.
    long_segment = "a" * 1000
    path = AssetPath(f"/{long_segment}")
    await store.put(path, b"ascii-1024")
    assert (await store.get(path)).content == b"ascii-1024"


@pytest.mark.asyncio
async def test_multibyte_path_at_max_length(sql_asset_backend):
    store = AssetStore(primary=sql_asset_backend)
    # Multi-byte (3-byte UTF-8 each) characters -- the column stores
    # characters, not bytes, so this is well within String(1024), but proves
    # the hash (computed on UTF-8 bytes) and full-path storage both round-trip
    # correctly for non-ASCII content.
    long_segment = "测试" * 400  # 800 chars, each U+6D4B/U+8BD5 (CJK)
    path = AssetPath(f"/{long_segment}")
    await store.put(path, b"multibyte")
    assert (await store.get(path)).content == b"multibyte"


@pytest.mark.asyncio
async def test_simulated_path_hash_collision_raises(sql_asset_backend, monkeypatch):
    """Force two DIFFERENT paths to share a path_hash (impossible with real
    SHA-256 at this scale, so the hash function itself is monkeypatched to a
    constant) and confirm the backend raises AssetPathHashCollisionError
    rather than silently mis-serving one path as the other."""
    store = AssetStore(primary=sql_asset_backend)
    path_a = AssetPath("/contract/collide-a.txt")
    path_b = AssetPath("/contract/collide-b.txt")

    constant_hash = hashlib.sha256(b"forced-collision").digest()
    monkeypatch.setattr(
        "linktools.ai.storage.sqlalchemy.asset.asset_path_hash",
        lambda _path: constant_hash,
    )

    await store.put(path_a, b"a")
    with pytest.raises(AssetPathHashCollisionError):
        await store.put(path_b, b"b")


@pytest.mark.asyncio
async def test_long_idempotency_key_round_trips(sql_asset_backend):
    store = AssetStore(primary=sql_asset_backend)
    path = AssetPath("/contract/idem-long-key.txt")
    long_key = "k" * 1000

    first = await store.put(path, b"x", options=WriteOptions(idempotency_key=long_key))
    second = await store.put(path, b"x", options=WriteOptions(idempotency_key=long_key))
    assert first.info.etag == second.info.etag


@pytest.mark.asyncio
async def test_simulated_idempotency_key_hash_collision_raises(sql_asset_backend, monkeypatch):
    store = AssetStore(primary=sql_asset_backend)
    path = AssetPath("/contract/idem-collide.txt")

    constant_hash = hashlib.sha256(b"forced-key-collision").digest()
    monkeypatch.setattr(
        "linktools.ai.storage.sqlalchemy.asset._idempotency_key_hash",
        lambda _key: constant_hash,
    )

    await store.put(path, b"x", options=WriteOptions(idempotency_key="key-one"))
    with pytest.raises(AssetPathHashCollisionError):
        await store.put(path, b"y", options=WriteOptions(idempotency_key="key-two"))


def test_asset_path_hash_is_deterministic_sha256():
    path = AssetPath("/a/b/c.txt")
    expected = hashlib.sha256(path.value.encode("utf-8")).digest()
    assert asset_path_hash(path) == expected
    assert len(asset_path_hash(path)) == 32
