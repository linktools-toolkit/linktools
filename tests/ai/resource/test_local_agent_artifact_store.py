import asyncio

from linktools.ai.resource.protocols import AgentArtifactStore, ArtifactRef
from linktools.ai.resource.local import LocalAgentArtifactStore


def test_satisfies_protocol(tmp_path):
    assert isinstance(LocalAgentArtifactStore(root=tmp_path), AgentArtifactStore)


def test_get_missing_returns_none(tmp_path):
    store = LocalAgentArtifactStore(root=tmp_path)
    ref = ArtifactRef(domain="session", scope="s1", kind="todos", path="todos.json")
    assert asyncio.run(store.get(ref)) is None


def test_put_then_get_roundtrip(tmp_path):
    store = LocalAgentArtifactStore(root=tmp_path)
    ref = ArtifactRef(domain="session", scope="s1", kind="todos", path="todos.json")
    content = b'{"todos": []}'

    async def _run():
        meta = await store.put(ref, content, idempotency_key="key-1")
        fetched = await store.get(ref)
        return meta, fetched

    meta, fetched = asyncio.run(_run())
    assert fetched == content
    assert meta.ref == ref
    assert meta.backend == "local"
    assert meta.status == "stored"
    assert meta.size_bytes == len(content)
    assert meta.checksum  # non-empty
    assert (tmp_path / ref.key).read_bytes() == content


def test_put_overwrites_existing(tmp_path):
    store = LocalAgentArtifactStore(root=tmp_path)
    ref = ArtifactRef(domain="session", scope="s1", kind="todos", path="todos.json")

    async def _run():
        await store.put(ref, b"first", idempotency_key="k1")
        await store.put(ref, b"second", idempotency_key="k2")
        return await store.get(ref)

    assert asyncio.run(_run()) == b"second"
