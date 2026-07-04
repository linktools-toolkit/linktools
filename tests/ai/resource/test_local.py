import asyncio
from datetime import datetime, timedelta

from linktools.ai.resource.local import InMemoryResourceBackend
from linktools.ai.resource.protocols import DeleteOp, MoveOp, PutOp, ResourceBackend


def test_backend_satisfies_protocol():
    assert isinstance(InMemoryResourceBackend(), ResourceBackend)


def test_put_then_get_roundtrip():
    async def run():
        backend = InMemoryResourceBackend()
        put = await backend.put("/skill/a/SKILL.md", "hello")
        assert put.version == 1
        got = await backend.get("/skill/a/SKILL.md")
        assert got is not None
        assert got.content == "hello"
        assert got.version == 1

    asyncio.run(run())


def test_put_same_content_is_idempotent_no_version_bump():
    async def run():
        backend = InMemoryResourceBackend()
        await backend.put("/skill/a/SKILL.md", "hello")
        second = await backend.put("/skill/a/SKILL.md", "hello")
        assert second.version == 1  # unchanged content -> no version bump

    asyncio.run(run())


def test_put_different_content_bumps_version():
    async def run():
        backend = InMemoryResourceBackend()
        await backend.put("/skill/a/SKILL.md", "hello")
        second = await backend.put("/skill/a/SKILL.md", "world")
        assert second.version == 2

    asyncio.run(run())


def test_get_returns_none_for_missing_path():
    async def run():
        backend = InMemoryResourceBackend()
        assert await backend.get("/skill/missing/SKILL.md") is None

    asyncio.run(run())


def test_delete_then_get_returns_none():
    async def run():
        backend = InMemoryResourceBackend()
        await backend.put("/skill/a/SKILL.md", "hello")
        deleted = await backend.delete("/skill/a/SKILL.md")
        assert deleted is True
        assert await backend.get("/skill/a/SKILL.md") is None

    asyncio.run(run())


def test_delete_already_deleted_path_is_idempotent_no_op():
    async def run():
        backend = InMemoryResourceBackend()
        await backend.put("/skill/a/SKILL.md", "hello")
        await backend.delete("/skill/a/SKILL.md")
        second = await backend.delete("/skill/a/SKILL.md")
        assert second is False  # nothing to delete, no error

    asyncio.run(run())


def test_delete_nonexistent_path_returns_false_no_error():
    async def run():
        backend = InMemoryResourceBackend()
        assert await backend.delete("/skill/never-existed/SKILL.md") is False

    asyncio.run(run())


def test_move_relocates_content():
    async def run():
        backend = InMemoryResourceBackend()
        await backend.put("/skill/a/SKILL.md", "hello")
        moved = await backend.move("/skill/a/SKILL.md", "/skill/b/SKILL.md")
        assert moved is not None
        assert moved.content == "hello"
        assert await backend.get("/skill/a/SKILL.md") is None
        assert (await backend.get("/skill/b/SKILL.md")).content == "hello"

    asyncio.run(run())


def test_move_already_applied_is_idempotent():
    async def run():
        backend = InMemoryResourceBackend()
        await backend.put("/skill/a/SKILL.md", "hello")
        await backend.move("/skill/a/SKILL.md", "/skill/b/SKILL.md")
        # Retry the same move -- src is gone, dst already holds the content.
        second = await backend.move("/skill/a/SKILL.md", "/skill/b/SKILL.md")
        assert second is not None
        assert second.content == "hello"

    asyncio.run(run())


def test_move_nonexistent_src_returns_none():
    async def run():
        backend = InMemoryResourceBackend()
        assert await backend.move("/skill/never/SKILL.md", "/skill/dst/SKILL.md") is None

    asyncio.run(run())


def test_list_pattern_matches_prefix():
    async def run():
        backend = InMemoryResourceBackend()
        await backend.put("/skill/a/SKILL.md", "a")
        await backend.put("/skill/b/SKILL.md", "b")
        await backend.put("/mcp/x/mcp.yaml", "x")
        results = await backend.list(pattern="/skill/*")
        paths = {r.path for r in results}
        assert paths == {"/skill/a/SKILL.md", "/skill/b/SKILL.md"}

    asyncio.run(run())


def test_list_excludes_deleted_resources_by_default():
    async def run():
        backend = InMemoryResourceBackend()
        await backend.put("/skill/a/SKILL.md", "a")
        await backend.delete("/skill/a/SKILL.md")
        results = await backend.list(pattern="/skill/*")
        assert results == []

    asyncio.run(run())


def test_list_pattern_matches_filename_across_namespace():
    async def run():
        backend = InMemoryResourceBackend()
        await backend.put("/skill/a/SKILL.md", "a")
        await backend.put("/skill/b/SKILL.md", "b")
        await backend.put("/skill/b/notes.md", "not a match")
        results = await backend.list(pattern="/skill/*/SKILL.md")
        paths = {r.path for r in results}
        assert paths == {"/skill/a/SKILL.md", "/skill/b/SKILL.md"}

    asyncio.run(run())


def test_list_pattern_matches_glob_across_namespace():
    async def run():
        backend = InMemoryResourceBackend()
        await backend.put("/skill/a/SKILL.md", "a")
        await backend.put("/mcp/b/mcp.yaml", "b")
        await backend.put("/skill/b/notes.md", "not a match")
        results = await backend.list(pattern="*/SKILL.md")
        paths = {r.path for r in results}
        assert paths == {"/skill/a/SKILL.md"}

    asyncio.run(run())


def test_get_at_version_returns_historical_content():
    async def run():
        backend = InMemoryResourceBackend()
        await backend.put("/skill/a/SKILL.md", "v1")
        await backend.put("/skill/a/SKILL.md", "v2")
        v1 = await backend.get("/skill/a/SKILL.md", 1)
        assert v1 is not None
        assert v1.content == "v1"

    asyncio.run(run())


def test_list_with_no_filters_returns_everything():
    async def run():
        backend = InMemoryResourceBackend()
        await backend.put("/skill/a/SKILL.md", "a")
        await backend.put("/skill/b/SKILL.md", "b")
        results = await backend.list()
        assert len(results) == 2

    asyncio.run(run())


def test_list_since_filters_older_rows():
    async def run():
        backend = InMemoryResourceBackend()
        await backend.put("/skill/a/SKILL.md", "a")
        future = datetime.now() + timedelta(days=1)
        results = await backend.list(since=future)
        assert results == []

    asyncio.run(run())




def test_apply_batch_mixed_operations():
    async def run():
        backend = InMemoryResourceBackend()
        await backend.put("/skill/a/SKILL.md", "a")
        await backend.put("/skill/b/SKILL.md", "b")
        results = await backend.apply_batch([
            PutOp(path="/skill/c/SKILL.md", content="c"),
            DeleteOp(path="/skill/a/SKILL.md"),
            MoveOp(src_path="/skill/b/SKILL.md", dst_path="/skill/d/SKILL.md"),
        ])
        result_paths = {r.path for r in results}
        assert result_paths == {"/skill/c/SKILL.md", "/skill/d/SKILL.md"}
        assert await backend.get("/skill/a/SKILL.md") is None
        assert await backend.get("/skill/d/SKILL.md") is not None

    asyncio.run(run())


def test_revision_increments_on_write():
    async def run():
        backend = InMemoryResourceBackend()
        before = await backend.revision()
        await backend.put("/skill/a/SKILL.md", "a")
        after = await backend.revision()
        assert after == before + 1

    asyncio.run(run())
