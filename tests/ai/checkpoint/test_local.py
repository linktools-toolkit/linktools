import asyncio

from linktools.ai.checkpoint.local import FileCheckpointStore
from linktools.ai.checkpoint.protocols import CheckpointStore


def test_satisfies_protocol(tmp_path):
    assert isinstance(FileCheckpointStore(root=tmp_path), CheckpointStore)


def test_save_then_restore_roundtrip(tmp_path):
    store = FileCheckpointStore(root=tmp_path)

    async def _run():
        checkpoint_id = await store.save("session-1", 1, b'{"messages": []}')
        return checkpoint_id, await store.restore(checkpoint_id)

    checkpoint_id, restored = asyncio.run(_run())
    assert restored == b'{"messages": []}'
    assert checkpoint_id == "session-1:1"


def test_list_returns_checkpoint_ids_in_save_order(tmp_path):
    store = FileCheckpointStore(root=tmp_path)

    async def _run():
        await store.save("session-1", 1, b"a")
        await store.save("session-1", 2, b"b")
        await store.save("session-2", 1, b"c")  # different session, must not appear
        return await store.list("session-1")

    ids = asyncio.run(_run())
    assert ids == ["session-1:1", "session-1:2"]


def test_restore_unknown_checkpoint_raises_key_error(tmp_path):
    import pytest

    store = FileCheckpointStore(root=tmp_path)

    async def _run():
        await store.restore("no-such-session:1")

    with pytest.raises(KeyError):
        asyncio.run(_run())
