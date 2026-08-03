#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""LocalSpecBackend: plain filesystem-directory spec persistence."""

import pytest

from linktools.ai.errors import SpecConflictError
from linktools.ai.spec.document import SpecDocument, SpecDocumentInfo, compute_spec_etag
from linktools.ai.spec.persistence.local import LocalSpecBackend
from linktools.ai.storage.multi import BatchStorageWriter, StorageReader, StorageWriter


def doc(path, body, *, version=1, kind="agent"):
    return SpecDocument(
        SpecDocumentInfo(path, kind, version, compute_spec_etag(body)),
        body,
    )


def test_local_backend_implements_reader_writer_batch():
    assert isinstance(LocalSpecBackend("/tmp"), StorageReader)
    assert isinstance(LocalSpecBackend("/tmp"), StorageWriter)
    assert isinstance(LocalSpecBackend("/tmp"), BatchStorageWriter)


@pytest.mark.asyncio
async def test_local_put_get_roundtrip_writes_file_to_tree(tmp_path):
    backend = LocalSpecBackend(tmp_path / "spec")
    await backend.initialize_storage()
    await backend.put(doc("agent/writer.md", b"hello"))
    # The spec path maps directly to a file under root.
    assert (tmp_path / "spec" / "agent" / "writer.md").read_bytes() == b"hello"
    got = await backend.get("agent/writer.md")
    assert got is not None
    assert got.content == b"hello"
    assert got.info.path == "agent/writer.md"


@pytest.mark.asyncio
async def test_local_metadata_derived_from_file(tmp_path):
    backend = LocalSpecBackend(tmp_path / "spec")
    await backend.initialize_storage()
    await backend.put(doc("agent/writer.md", b"hello"))
    info = await backend.stat("agent/writer.md")
    assert info is not None
    assert info.kind == "agent"
    assert info.version == 1
    assert info.active is True
    assert info.etag == compute_spec_etag(b"hello")


@pytest.mark.asyncio
async def test_local_kind_defaults_to_spec_for_top_level_path(tmp_path):
    backend = LocalSpecBackend(tmp_path / "spec")
    await backend.initialize_storage()
    await backend.put(doc("readme", b"top-level", kind="spec"))
    info = await backend.stat("readme")
    assert info is not None
    assert info.kind == "spec"


@pytest.mark.asyncio
async def test_local_list_info_scans_tree_and_sorts(tmp_path):
    backend = LocalSpecBackend(tmp_path / "spec")
    await backend.initialize_storage()
    await backend.put(doc("agent/b", b"b"))
    await backend.put(doc("agent/a", b"a"))
    await backend.put(doc("skill/x", b"x"))
    infos = await backend.list_info()
    paths = [info.path for info in infos]
    assert paths == ["agent/a", "agent/b", "skill/x"]


@pytest.mark.asyncio
async def test_local_list_info_kind_filter(tmp_path):
    backend = LocalSpecBackend(tmp_path / "spec")
    await backend.initialize_storage()
    await backend.put(doc("agent/a", b"a"))
    await backend.put(doc("skill/b", b"b"))
    infos = await backend.list_info(kind="skill")
    assert [info.path for info in infos] == ["skill/b"]


@pytest.mark.asyncio
async def test_local_delete_removes_file_and_is_idempotent(tmp_path):
    backend = LocalSpecBackend(tmp_path / "spec")
    await backend.initialize_storage()
    await backend.put(doc("agent/a", b"a"))
    await backend.delete("agent/a")
    assert not (tmp_path / "spec" / "agent" / "a").exists()
    # Deleting again is a no-op (no error).
    await backend.delete("agent/a")
    assert await backend.get("agent/a") is None


@pytest.mark.asyncio
async def test_local_get_returns_none_for_missing(tmp_path):
    backend = LocalSpecBackend(tmp_path / "spec")
    await backend.initialize_storage()
    assert await backend.get("ghost") is None


@pytest.mark.asyncio
async def test_local_apply_batch_writes_and_deletes(tmp_path):
    backend = LocalSpecBackend(tmp_path / "spec")
    await backend.initialize_storage()
    await backend.put(doc("keep", b"keep"))
    await backend.put(doc("del", b"del"))
    await backend.apply_batch(
        (doc("agent/new", b"new"), doc("keep", b"keep2")),
        ("del",),
    )
    assert (await backend.get("agent/new")).content == b"new"
    assert (await backend.get("keep")).content == b"keep2"
    assert await backend.get("del") is None


@pytest.mark.asyncio
async def test_local_apply_batch_put_overrides_delete_same_path(tmp_path):
    backend = LocalSpecBackend(tmp_path / "spec")
    await backend.initialize_storage()
    await backend.put(doc("x", b"v1"))
    await backend.apply_batch((doc("x", b"v2"),), ("x",))
    assert (await backend.get("x")).content == b"v2"


@pytest.mark.asyncio
async def test_local_reset_replaces_full_tree(tmp_path):
    backend = LocalSpecBackend(tmp_path / "spec")
    await backend.initialize_storage()
    await backend.put(doc("old", b"old"))
    await backend.reset((doc("agent/new", b"new"),))
    assert await backend.get("old") is None
    assert (await backend.get("agent/new")).content == b"new"


@pytest.mark.asyncio
async def test_local_rejects_path_traversal(tmp_path):
    backend = LocalSpecBackend(tmp_path / "spec")
    await backend.initialize_storage()
    with pytest.raises(SpecConflictError, match="escapes root"):
        await backend.put(doc("../escape", b"x"))
    with pytest.raises(SpecConflictError, match="escapes root"):
        await backend.delete("../escape")


# -- path adapter ----------------------------------------------------------


def _adapter_backend(tmp_path):
    """LocalSpecBackend whose spec `mcp/...` maps to disk `adapter/...`
    (the capabilities-editor-era naming quirk the adapter exists to absorb)."""
    from linktools.ai.spec.persistence.local import PrefixSpecPathAdapter

    return LocalSpecBackend(
        tmp_path / "spec", path_adapter=PrefixSpecPathAdapter({"mcp": "adapter"})
    )


@pytest.mark.asyncio
async def test_adapter_put_get_writes_to_remapped_disk_dir(tmp_path):
    backend = _adapter_backend(tmp_path)
    await backend.initialize_storage()
    await backend.put(doc("mcp/server.md", b"hi", kind="mcp"))
    # The spec path `mcp/...` lands on disk under `adapter/...`.
    assert (tmp_path / "spec" / "adapter" / "server.md").read_bytes() == b"hi"
    # ...and round-trips back through the spec path.
    got = await backend.get("mcp/server.md")
    assert got is not None
    assert got.content == b"hi"
    assert got.info.path == "mcp/server.md"
    assert got.info.kind == "mcp"


@pytest.mark.asyncio
async def test_adapter_unmapped_kind_passes_through(tmp_path):
    backend = _adapter_backend(tmp_path)
    await backend.initialize_storage()
    await backend.put(doc("agent/writer.md", b"x"))
    # `agent` is not in the mapping, so it lands at `agent/...` unchanged.
    assert (tmp_path / "spec" / "agent" / "writer.md").read_bytes() == b"x"
    assert (await backend.get("agent/writer.md")).content == b"x"


@pytest.mark.asyncio
async def test_adapter_stat_translates_path(tmp_path):
    backend = _adapter_backend(tmp_path)
    await backend.initialize_storage()
    await backend.put(doc("mcp/server.md", b"hi", kind="mcp"))
    info = await backend.stat("mcp/server.md")
    assert info is not None
    assert info.path == "mcp/server.md"
    assert info.kind == "mcp"


@pytest.mark.asyncio
async def test_adapter_list_info_translates_back_to_spec_path(tmp_path):
    backend = _adapter_backend(tmp_path)
    await backend.initialize_storage()
    await backend.put(doc("mcp/server.md", b"hi", kind="mcp"))
    await backend.put(doc("agent/writer.md", b"x"))
    infos = await backend.list_info()
    paths = {i.path for i in infos}
    assert "mcp/server.md" in paths
    assert "agent/writer.md" in paths
    assert not any(p.startswith("adapter/") for p in paths)


@pytest.mark.asyncio
async def test_adapter_list_info_kind_filter_uses_spec_kind(tmp_path):
    backend = _adapter_backend(tmp_path)
    await backend.initialize_storage()
    await backend.put(doc("mcp/server.md", b"hi", kind="mcp"))
    await backend.put(doc("agent/writer.md", b"x"))
    # `kind="mcp"` filters by the SPEC kind, even though on disk it's `adapter`.
    mcp_only = await backend.list_info(kind="mcp")
    assert [i.path for i in mcp_only] == ["mcp/server.md"]


@pytest.mark.asyncio
async def test_adapter_get_many_translates_paths(tmp_path):
    backend = _adapter_backend(tmp_path)
    await backend.initialize_storage()
    await backend.put(doc("mcp/a.md", b"a", kind="mcp"))
    await backend.put(doc("mcp/b.md", b"b", kind="mcp"))
    loaded = await backend.get_many(("mcp/a.md", "mcp/b.md", "mcp/missing.md"))
    assert set(loaded) == {"mcp/a.md", "mcp/b.md"}
    assert loaded["mcp/a.md"].content == b"a"


@pytest.mark.asyncio
async def test_adapter_delete_removes_remapped_file(tmp_path):
    backend = _adapter_backend(tmp_path)
    await backend.initialize_storage()
    await backend.put(doc("mcp/server.md", b"hi", kind="mcp"))
    await backend.delete("mcp/server.md")
    assert not (tmp_path / "spec" / "adapter" / "server.md").exists()
    assert await backend.get("mcp/server.md") is None


@pytest.mark.asyncio
async def test_adapter_reset_and_apply_batch_round_trip(tmp_path):
    backend = _adapter_backend(tmp_path)
    await backend.initialize_storage()
    await backend.put(doc("mcp/old.md", b"old", kind="mcp"))
    # reset replaces the tree; the old mcp file is gone, the new one lands on disk.
    await backend.reset((doc("mcp/new.md", b"new", kind="mcp"),))
    assert await backend.get("mcp/old.md") is None
    assert (tmp_path / "spec" / "adapter" / "new.md").read_bytes() == b"new"
    # apply_batch: put + delete, put-wins for same path.
    await backend.apply_batch(
        puts=(doc("mcp/new.md", b"v2", kind="mcp"),),
        deletes=("mcp/new.md",),
    )
    assert (await backend.get("mcp/new.md")).content == b"v2"


@pytest.mark.asyncio
async def test_adapter_still_rejects_path_traversal(tmp_path):
    # The adapter runs before the escape check, so a traversal path is still
    # rejected regardless of the mapping.
    backend = _adapter_backend(tmp_path)
    await backend.initialize_storage()
    with pytest.raises(SpecConflictError, match="escapes root"):
        await backend.put(doc("../escape", b"x"))
