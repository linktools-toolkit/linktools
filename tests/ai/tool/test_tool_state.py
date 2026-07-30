import pytest

from linktools.ai.agent.tool.persistence.memory import LocalToolStateBackend
from linktools.ai.agent.tool.models import ToolOperation, ToolOperationStatus
from linktools.ai.agent.tool.store import ToolStateStore


@pytest.mark.asyncio
async def test_tool_state_replays_completed_idempotent_operation():
    store = LocalToolStateBackend()
    operation = ToolOperation("op", "tenant", "run", "call", "idem", "tool", "hash", "binding", ToolOperationStatus.PREPARED)
    await store.prepare(operation)
    claim = await store.claim("op", owner="worker")
    result = await store.complete("op", owner="worker", fence=claim.fence, result={"ok": True})
    replay = await store.prepare(operation)
    assert result.status is ToolOperationStatus.COMPLETED
    assert replay.result == {"ok": True}


@pytest.mark.asyncio
async def test_tool_state_rejects_stale_fence():
    store = LocalToolStateBackend()
    await store.prepare(ToolOperation("op", None, "run", "call", "idem", "tool", "hash", "binding", ToolOperationStatus.PREPARED))
    claim = await store.claim("op", owner="worker")
    with pytest.raises(ValueError):
        await store.complete("op", owner="other", fence=claim.fence, result=None)
