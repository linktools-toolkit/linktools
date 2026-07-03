import asyncio

import pytest

from linktools.ai.resource_store.file import FileBackend
from linktools.ai.resource_store.protocols import DeleteOp, MoveOp, PutOp, ResourceBackend


def test_backend_satisfies_protocol(tmp_path):
    assert isinstance(FileBackend(tmp_path), ResourceBackend)


def test_put_writes_to_disk_and_get_reads_it_back(tmp_path):
    async def run():
        backend = FileBackend(tmp_path)
        await backend.put("/skill/a/SKILL.md", "hello")
        assert (tmp_path / "skill" / "a" / "SKILL.md").read_text(encoding="utf-8") == "hello"
        result = await backend.get("/skill/a/SKILL.md")
        assert result is not None
        assert result.content == "hello"
        assert result.version == 1

    asyncio.run(run())


def test_put_same_content_twice_is_idempotent(tmp_path):
    async def run():
        backend = FileBackend(tmp_path)
        await backend.put("/skill/a/SKILL.md", "hello")
        second = await backend.put("/skill/a/SKILL.md", "hello")
        assert second.version == 1

    asyncio.run(run())


def test_get_returns_none_for_missing_file(tmp_path):
    async def run():
        backend = FileBackend(tmp_path)
        assert await backend.get("/skill/missing/SKILL.md") is None

    asyncio.run(run())


def test_delete_removes_file(tmp_path):
    async def run():
        backend = FileBackend(tmp_path)
        await backend.put("/skill/a/SKILL.md", "hello")
        assert await backend.delete("/skill/a/SKILL.md") is True
        assert not (tmp_path / "skill" / "a" / "SKILL.md").exists()
        assert await backend.get("/skill/a/SKILL.md") is None

    asyncio.run(run())


def test_delete_nonexistent_file_is_idempotent_no_op(tmp_path):
    async def run():
        backend = FileBackend(tmp_path)
        assert await backend.delete("/skill/never/SKILL.md") is False

    asyncio.run(run())


def test_move_relocates_file(tmp_path):
    async def run():
        backend = FileBackend(tmp_path)
        await backend.put("/skill/a/SKILL.md", "hello")
        moved = await backend.move("/skill/a/SKILL.md", "/skill/b/SKILL.md")
        assert moved is not None
        assert moved.content == "hello"
        assert not (tmp_path / "skill" / "a" / "SKILL.md").exists()
        assert (tmp_path / "skill" / "b" / "SKILL.md").read_text(encoding="utf-8") == "hello"

    asyncio.run(run())


def test_propfind_lists_files_under_prefix(tmp_path):
    async def run():
        backend = FileBackend(tmp_path)
        await backend.put("/skill/a/SKILL.md", "a")
        await backend.put("/skill/b/SKILL.md", "b")
        await backend.put("/mcp/x/mcp.yaml", "x")
        results = await backend.propfind("/skill/")
        paths = {r.path for r in results}
        assert paths == {"/skill/a/SKILL.md", "/skill/b/SKILL.md"}

    asyncio.run(run())


def test_get_by_name_matches_filename(tmp_path):
    async def run():
        backend = FileBackend(tmp_path)
        await backend.put("/skill/a/SKILL.md", "a")
        await backend.put("/skill/b/SKILL.md", "b")
        await backend.put("/skill/b/notes.md", "not a match")
        results = await backend.get_by_name("skill", "SKILL.md")
        paths = {r.path for r in results}
        assert paths == {"/skill/a/SKILL.md", "/skill/b/SKILL.md"}

    asyncio.run(run())


def test_apply_batch_ordered_operations(tmp_path):
    async def run():
        backend = FileBackend(tmp_path)
        await backend.put("/skill/a/SKILL.md", "a")
        results = await backend.apply_batch([
            PutOp(path="/skill/b/SKILL.md", content="b"),
            DeleteOp(path="/skill/a/SKILL.md"),
            MoveOp(src_path="/skill/b/SKILL.md", dst_path="/skill/c/SKILL.md"),
        ])
        result_paths = {r.path for r in results}
        assert result_paths == {"/skill/c/SKILL.md"}
        assert await backend.get("/skill/a/SKILL.md") is None
        assert await backend.get("/skill/c/SKILL.md") is not None

    asyncio.run(run())


def test_readonly_backend_rejects_put(tmp_path):
    async def run():
        backend = FileBackend(tmp_path, readonly=True)
        with pytest.raises(PermissionError):
            await backend.put("/skill/a/SKILL.md", "hello")

    asyncio.run(run())


def test_readonly_backend_rejects_delete(tmp_path):
    async def run():
        backend = FileBackend(tmp_path, readonly=True)
        with pytest.raises(PermissionError):
            await backend.delete("/skill/a/SKILL.md")

    asyncio.run(run())


def test_readonly_backend_rejects_move(tmp_path):
    async def run():
        backend = FileBackend(tmp_path, readonly=True)
        with pytest.raises(PermissionError):
            await backend.move("/skill/a/SKILL.md", "/skill/b/SKILL.md")

    asyncio.run(run())


def test_readonly_backend_rejects_apply_batch(tmp_path):
    async def run():
        backend = FileBackend(tmp_path, readonly=True)
        with pytest.raises(PermissionError):
            await backend.apply_batch([PutOp(path="/skill/a/SKILL.md", content="a")])

    asyncio.run(run())


def test_readonly_backend_allows_reads(tmp_path):
    async def run():
        writable = FileBackend(tmp_path)
        await writable.put("/skill/a/SKILL.md", "hello")
        readonly = FileBackend(tmp_path, readonly=True)
        result = await readonly.get("/skill/a/SKILL.md")
        assert result is not None
        assert result.content == "hello"

    asyncio.run(run())


def test_get_revision_is_always_zero_for_file_backend(tmp_path):
    async def run():
        backend = FileBackend(tmp_path)
        await backend.put("/skill/a/SKILL.md", "hello")
        # FileBackend has no distributed revision concept -- it's always 0, since
        # DatabaseBackend (Task 5) owns the only real revision counter.
        assert await backend.get_revision() == 0

    asyncio.run(run())
