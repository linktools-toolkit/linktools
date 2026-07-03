import asyncio

import pytest

from linktools.ai.swarm.local import FileTaskQueue, InMemoryTaskQueue
from linktools.ai.swarm.protocols import Task, TaskQueue


@pytest.fixture(params=["memory", "file"])
def queue(request, tmp_path):
    if request.param == "memory":
        return InMemoryTaskQueue()
    return FileTaskQueue(root=tmp_path)


def test_queue_satisfies_protocol(queue):
    assert isinstance(queue, TaskQueue)


def test_add_and_list(queue):
    async def _run():
        await queue.add([Task(task_id="a", payload=1), Task(task_id="b", payload=2)])
        return await queue.list()

    tasks = asyncio.run(_run())
    assert {t.task_id for t in tasks} == {"a", "b"}


def test_claim_respects_dependencies(queue):
    async def _run():
        await queue.add([
            Task(task_id="a", payload=1),
            Task(task_id="b", payload=2, depends_on=("a",)),
        ])
        first = await queue.claim("agent1")
        second = await queue.claim("agent2")  # b depends on unfinished a
        await queue.complete("a", {"ok": True})
        third = await queue.claim("agent2")
        return first, second, third

    first, second, third = asyncio.run(_run())
    assert first.task_id == "a"
    assert second is None
    assert third.task_id == "b"


def test_claim_returns_none_when_no_pending_tasks(queue):
    result = asyncio.run(queue.claim("agent1"))
    assert result is None


def test_fail_marks_task_failed_and_not_reclaimable(queue):
    async def _run():
        await queue.add([Task(task_id="a", payload=1)])
        claimed = await queue.claim("agent1")
        await queue.fail(claimed.task_id, "boom")
        again = await queue.claim("agent2")
        failed = await queue.list(status="failed")
        return again, failed

    again, failed = asyncio.run(_run())
    assert again is None
    assert len(failed) == 1
    assert failed[0].error == "boom"


def test_concurrent_claims_do_not_double_claim_same_task(queue):
    async def _run():
        await queue.add([Task(task_id="a", payload=1)])
        results = await asyncio.gather(*[queue.claim(f"agent{i}") for i in range(10)])
        return results

    results = asyncio.run(_run())
    claimed = [r for r in results if r is not None]
    assert len(claimed) == 1
