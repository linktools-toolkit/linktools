from datetime import datetime, timezone

import pytest

from linktools.ai.execution.trace_collector import SemanticTraceCollector
from linktools.ai.execution.models import RunStatus, RunUsage


class FakeTraceStore:
    def __init__(self):
        self.steps = []

    async def append_trace_steps(self, run_id, *, expected_sequence, steps):
        assert expected_sequence == len(self.steps)
        self.steps.extend(steps)
        return len(self.steps)


@pytest.mark.asyncio
async def test_collector_flushes_each_completed_step_and_keeps_bounded_state():
    store = FakeTraceStore()
    collector = SemanticTraceCollector("run", store, 0)
    await collector.model_request_succeeded({"request": {"messages": []}, "response": "ok"})
    await collector.tool_result({"call_id": "c1", "result": {"ok": True}})
    assert collector._pending == []
    assert len(store.steps) == 2
    snapshot = await collector.build_snapshot(resume_messages=(), final_output={"ok": True}, status=RunStatus.COMPLETED, usage=RunUsage(), revision=1)
    assert snapshot.trace_end_sequence == 2
