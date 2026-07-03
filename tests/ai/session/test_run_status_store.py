import asyncio

import pytest

from linktools.ai.session.local import InMemoryRunStatusStore
from linktools.ai.session.protocols import RunStatus, RunStatusStore


def test_satisfies_protocol():
    assert isinstance(InMemoryRunStatusStore(), RunStatusStore)


def test_start_then_get_returns_running():
    store = InMemoryRunStatusStore()

    async def _run():
        await store.start("run-1")
        return await store.get("run-1")

    status = asyncio.run(_run())
    assert status.state == "running"
    assert status.result is None
    assert status.error is None


def test_update_overwrites_status():
    store = InMemoryRunStatusStore()

    async def _run():
        await store.start("run-1")
        await store.update("run-1", RunStatus(state="done", result={"ok": True}))
        return await store.get("run-1")

    status = asyncio.run(_run())
    assert status.state == "done"
    assert status.result == {"ok": True}


def test_get_unknown_run_id_raises_key_error():
    store = InMemoryRunStatusStore()

    async def _run():
        await store.get("does-not-exist")

    with pytest.raises(KeyError):
        asyncio.run(_run())
