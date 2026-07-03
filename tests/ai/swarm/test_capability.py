import asyncio

from linktools.ai.swarm.capability import SwarmCapability
from linktools.ai.swarm.local import InMemoryTaskQueue
from linktools.ai.swarm.protocols import Task


def test_claim_task_returns_task_payload():
    queue = InMemoryTaskQueue()
    asyncio.run(queue.add([Task(task_id="a", payload={"do": "x"})]))
    cap = SwarmCapability(task_queue=queue, agent_id="agent-1")
    toolset = cap.get_toolset()

    async def _run():
        return await toolset.tools["claim_task"].function()

    result = asyncio.run(_run())
    assert result["task_id"] == "a"
    assert result["payload"] == {"do": "x"}


def test_claim_task_returns_none_marker_when_queue_empty():
    queue = InMemoryTaskQueue()
    cap = SwarmCapability(task_queue=queue, agent_id="agent-1")
    toolset = cap.get_toolset()

    async def _run():
        return await toolset.tools["claim_task"].function()

    result = asyncio.run(_run())
    assert result == {"task_id": None}


def test_complete_task_marks_done_in_queue():
    queue = InMemoryTaskQueue()
    asyncio.run(queue.add([Task(task_id="a", payload=1)]))
    cap = SwarmCapability(task_queue=queue, agent_id="agent-1")
    toolset = cap.get_toolset()

    async def _run():
        await toolset.tools["claim_task"].function()
        await toolset.tools["complete_task"].function("a", {"answer": 42})
        return await queue.list(status="done")

    done = asyncio.run(_run())
    assert len(done) == 1
    assert done[0].result == {"answer": 42}


def test_fail_task_marks_failed_in_queue():
    queue = InMemoryTaskQueue()
    asyncio.run(queue.add([Task(task_id="a", payload=1)]))
    cap = SwarmCapability(task_queue=queue, agent_id="agent-1")
    toolset = cap.get_toolset()

    async def _run():
        await toolset.tools["claim_task"].function()
        await toolset.tools["fail_task"].function("a", "could not do it")
        return await queue.list(status="failed")

    failed = asyncio.run(_run())
    assert len(failed) == 1
    assert failed[0].error == "could not do it"


def test_list_tasks_returns_dicts():
    queue = InMemoryTaskQueue()
    asyncio.run(queue.add([Task(task_id="a", payload=1), Task(task_id="b", payload=2)]))
    cap = SwarmCapability(task_queue=queue, agent_id="agent-1")
    toolset = cap.get_toolset()

    async def _run():
        return await toolset.tools["list_tasks"].function()

    result = asyncio.run(_run())
    assert {t["task_id"] for t in result} == {"a", "b"}
