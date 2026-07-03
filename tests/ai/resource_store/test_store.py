import asyncio

import pytest

from linktools.ai.resource_store.local import InMemoryResourceBackend
from linktools.ai.resource_store.protocols import DeleteOp, MoveOp, PutOp
from linktools.ai.resource_store.store import ResourceStore


def test_store_requires_at_least_one_backend():
    with pytest.raises(ValueError):
        ResourceStore(backends=[])


def test_single_backend_get_put_roundtrip():
    async def run():
        backend = InMemoryResourceBackend()
        store = ResourceStore(backends=[backend])
        await store.put("/skill/a/SKILL.md", "hello")
        result = await store.get("/skill/a/SKILL.md")
        assert result is not None
        assert result.content == "hello"

    asyncio.run(run())


def test_get_falls_through_to_second_backend_when_first_misses():
    async def run():
        primary = InMemoryResourceBackend()
        fallback = InMemoryResourceBackend()
        await fallback.put("/skill/only-in-fallback/SKILL.md", "fallback content")
        store = ResourceStore(backends=[primary, fallback])

        result = await store.get("/skill/only-in-fallback/SKILL.md")
        assert result is not None
        assert result.content == "fallback content"

    asyncio.run(run())


def test_get_prefers_first_backend_when_both_have_the_path():
    async def run():
        primary = InMemoryResourceBackend()
        fallback = InMemoryResourceBackend()
        await primary.put("/skill/a/SKILL.md", "primary content")
        await fallback.put("/skill/a/SKILL.md", "fallback content")
        store = ResourceStore(backends=[primary, fallback])

        result = await store.get("/skill/a/SKILL.md")
        assert result.content == "primary content"

    asyncio.run(run())


def test_writes_always_target_the_first_backend_only():
    async def run():
        primary = InMemoryResourceBackend()
        fallback = InMemoryResourceBackend()
        store = ResourceStore(backends=[primary, fallback])

        await store.put("/skill/a/SKILL.md", "hello")

        assert await primary.get("/skill/a/SKILL.md") is not None
        assert await fallback.get("/skill/a/SKILL.md") is None

    asyncio.run(run())


def test_propfind_unions_across_backends_keyed_by_path():
    async def run():
        primary = InMemoryResourceBackend()
        fallback = InMemoryResourceBackend()
        await primary.put("/skill/a/SKILL.md", "primary a")
        await fallback.put("/skill/a/SKILL.md", "fallback a")  # shadowed by primary
        await fallback.put("/skill/b/SKILL.md", "fallback only b")
        store = ResourceStore(backends=[primary, fallback])

        results = await store.propfind("/skill/")
        by_path = {r.path: r.content for r in results}
        assert by_path == {
            "/skill/a/SKILL.md": "primary a",  # primary wins the collision
            "/skill/b/SKILL.md": "fallback only b",  # fallback-only path still surfaces
        }

    asyncio.run(run())


def test_get_by_name_unions_across_backends_keyed_by_path():
    async def run():
        primary = InMemoryResourceBackend()
        fallback = InMemoryResourceBackend()
        await primary.put("/skill/a/SKILL.md", "primary a")
        await fallback.put("/skill/b/SKILL.md", "fallback only b")
        store = ResourceStore(backends=[primary, fallback])

        results = await store.get_by_name("skill", "SKILL.md")
        paths = {r.path for r in results}
        assert paths == {"/skill/a/SKILL.md", "/skill/b/SKILL.md"}

    asyncio.run(run())


def test_delete_and_move_target_first_backend_only():
    async def run():
        primary = InMemoryResourceBackend()
        fallback = InMemoryResourceBackend()
        await primary.put("/skill/a/SKILL.md", "hello")
        store = ResourceStore(backends=[primary, fallback])

        deleted = await store.delete("/skill/a/SKILL.md")
        assert deleted is True

        await primary.put("/skill/b/SKILL.md", "world")
        moved = await store.move("/skill/b/SKILL.md", "/skill/c/SKILL.md")
        assert moved is not None
        assert await fallback.get("/skill/c/SKILL.md") is None

    asyncio.run(run())


def test_put_many_delete_many_move_many_call_apply_batch_not_a_loop():
    async def run():
        primary = InMemoryResourceBackend()
        store = ResourceStore(backends=[primary])

        await store.put_many([PutOp(path="/skill/a/SKILL.md", content="a"), PutOp(path="/skill/b/SKILL.md", content="b")])
        assert (await primary.get("/skill/a/SKILL.md")).content == "a"
        assert (await primary.get("/skill/b/SKILL.md")).content == "b"

        await store.delete_many(["/skill/a/SKILL.md"])
        assert await primary.get("/skill/a/SKILL.md") is None

        await store.move_many([("/skill/b/SKILL.md", "/skill/c/SKILL.md")])
        assert await primary.get("/skill/c/SKILL.md") is not None

    asyncio.run(run())


def test_apply_batch_mixed_ops_hits_first_backend():
    async def run():
        primary = InMemoryResourceBackend()
        fallback = InMemoryResourceBackend()
        store = ResourceStore(backends=[primary, fallback])

        results = await store.apply_batch([
            PutOp(path="/skill/a/SKILL.md", content="a"),
            PutOp(path="/skill/b/SKILL.md", content="b"),
            DeleteOp(path="/skill/a/SKILL.md"),
        ])
        result_paths = {r.path for r in results}
        assert result_paths == {"/skill/b/SKILL.md"}
        assert await fallback.get("/skill/b/SKILL.md") is None

    asyncio.run(run())


def test_get_revision_reads_first_backend():
    async def run():
        primary = InMemoryResourceBackend()
        store = ResourceStore(backends=[primary])
        await store.put("/skill/a/SKILL.md", "hello")
        assert await store.get_revision() == await primary.get_revision()

    asyncio.run(run())


def test_refresh_does_not_raise_with_in_memory_backend():
    async def run():
        store = ResourceStore(backends=[InMemoryResourceBackend()])
        await store.refresh()  # no-op for a backend with no internal cache to invalidate

    asyncio.run(run())
