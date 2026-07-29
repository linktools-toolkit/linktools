import pytest

from linktools.ai.tasks.models import TaskExecution, TaskPlan
from linktools.ai.tasks.persistence.local import LocalTaskBackend
from linktools.ai.tasks.store import TaskStore


@pytest.mark.asyncio
async def test_task_store_uses_one_fenced_claim_path():
    store = TaskStore(LocalTaskBackend())
    await store.save_plan(TaskPlan("plan", ()))
    await store.add_execution(TaskExecution("execution", "plan", "node", "ready"))
    claimed = await store.claim("execution", owner="worker")
    with pytest.raises(ValueError):
        await store.complete("execution", owner="other", fence=claimed.fence, result=None)
    result = await store.complete("execution", owner="worker", fence=claimed.fence, result="ok")
    assert result.status == "completed"
