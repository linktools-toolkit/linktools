#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Asset storage and command checks."""

import asyncio
from pathlib import Path

import pytest

from linktools.ai.foundation.errors import InvalidAssetError, AssetConflictError
from linktools.ai.asset import (
    AssetContent,
    AssetContentInfo,
    AssetIndex,
    AssetLoader,
    StrictConfigReader,
    compute_asset_etag,
    parse_json_text,
    parse_markdown_text,
)
from linktools.ai.asset.persistence.local import LocalAssetBackend, PrefixAssetPathAdapter
from linktools.ai.asset.persistence.sqlalchemy import SqlAlchemyAssetBackend
from linktools.ai.asset.store import AssetStore
from linktools.ai.storage.database import build_sqlite_storage


def asset_content(path: str, raw: bytes, version: int = 1) -> AssetContent:
    return AssetContent(
        AssetContentInfo(path, path.split("/", 1)[0], version, compute_asset_etag(raw)),
        raw,
    )


def test_asset_parsers_and_strict_null_semantics() -> None:
    assert parse_json_text('{"name": "demo"}') == {"name": "demo"}
    frontmatter, body = parse_markdown_text("---\nname: demo\n---\nbody")
    assert frontmatter == {"name": "demo"}
    assert body.endswith("body")
    reader = StrictConfigReader({}, allowed={"retries"}, context="asset")
    assert reader.non_negative_int("retries", 2) == 2
    with pytest.raises(InvalidAssetError):
        StrictConfigReader({"retries": None}, allowed={"retries"}, context="asset").non_negative_int("retries")


def test_local_backend_path_adapter_and_atomic_contract(tmp_path: Path) -> None:
    async def run() -> None:
        backend = LocalAssetBackend(tmp_path, path_adapter=PrefixAssetPathAdapter({"mcp": "adapter"}))
        await backend.initialize_storage()
        entry = asset_content("mcp/demo.json", b"{}")
        await backend.put(entry)
        assert (tmp_path / "adapter/demo.json").read_bytes() == b"{}"
        assert await backend.get("mcp/demo.json") == entry
        assert [item.path for item in await backend.list_info()] == ["mcp/demo.json"]
        with pytest.raises(AssetConflictError):
            await backend.get("../escape")

    asyncio.run(run())

def test_asset_index_reloads_changed_files(tmp_path: Path) -> None:
    async def run() -> None:
        (tmp_path / "one.json").write_text('{"value": 1}', encoding="utf-8")
        loader = AssetLoader.from_filesystem(tmp_path)

        class Codec:
            def decode(self, item_id: str, raw: str) -> dict[str, object]:
                return parse_json_text(raw)

        index = AssetIndex(index_source := AssetIndex.source_from_loader(loader), Codec(), suffix=".json")
        assert await index.list_ids() == ("one",)
        assert await index.get("one") == {"value": 1}
        (tmp_path / "one.json").write_text('{"value": 2}', encoding="utf-8")
        assert await index.get("one") == {"value": 2}
        assert index_source.identity('{"value": 2}')

    asyncio.run(run())


def test_sql_backend_history_and_composed_reads(tmp_path: Path) -> None:
    async def run() -> None:
        database = build_sqlite_storage(tmp_path / "assets.db")
        backend = SqlAlchemyAssetBackend(database.session_factory)
        await backend.initialize_storage(database.engine)
        store = AssetStore(backend, writer=backend)
        first = asset_content("agent/demo.json", b"one", 1)
        second = asset_content("agent/demo.json", b"two", 2)
        await store.put(first)
        first_revision = await store.current_revision()
        await store.put(second)
        assert (await store.get_at_revision("agent/demo.json", first_revision)).content == b"one"
        assert (await store.get_at_version("agent/demo.json", 1)).content == b"one"
        assert (await store.get("agent/demo.json")).content == b"two"
        await store.delete("agent/demo.json")
        assert (await store.list_versions("agent/demo.json"))[0].deleted is True
        await database.engine.dispose()

    asyncio.run(run())
