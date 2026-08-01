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
