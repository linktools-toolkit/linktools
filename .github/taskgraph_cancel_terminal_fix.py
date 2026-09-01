#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def replace_once(path: str, old: str, new: str) -> None:
    target = ROOT / path
    text = target.read_text()
    if text.count(old) != 1:
        raise RuntimeError(f"{path}: replacement anchor mismatch")
    target.write_text(text.replace(old, new, 1))


replace_once(
    "linktools-ai/src/linktools/ai/task/_service_impl.py",
    '''        if claimed:\n            try:\n                if not _terminal(view.status):\n                    view = await self._persistence.tasks.cancel_graph(\n                        graph_id,\n                        tenant_id=tenant_id,\n                    )\n                view = await self._persistence.tasks.get_graph(\n''',
    '''        if claimed:\n            try:\n                view = await self._persistence.tasks.cancel_graph(\n                    graph_id,\n                    tenant_id=tenant_id,\n                )\n                view = await self._persistence.tasks.get_graph(\n''',
)

path = ROOT / "tests/ai/test_task_review_closure.py"
text = path.read_text()
anchor = '''@pytest.mark.asyncio\nasync def test_explicit_cancel_preserves_blocked_terminal_node() -> None:\n'''
test = '''@pytest.mark.asyncio\nasync def test_service_cancel_overrides_failed_graph_with_active_node() -> None:\n    state = RuntimeState.in_memory()\n    await state.initialize(namespace="task-service-cancel-override", tenant_id="tenant")\n    try:\n        repository = state.task.tasks\n        graph = TaskGraph("service-cancel-override", (TaskNode("failed"), TaskNode("active")))\n        await repository.create_graph(graph, tenant_id="tenant")\n        lease = await repository.claim(\n            graph.graph_id,\n            "failed",\n            tenant_id="tenant",\n            owner="worker",\n            lease_seconds=30,\n        )\n        await repository.fail(\n            lease,\n            tenant_id="tenant",\n            error_code=ErrorCode.TASK_NODE_FAILED.value,\n            error_digest="c" * 64,\n        )\n        before = await repository.snapshot_graph(graph.graph_id, tenant_id="tenant")\n        assert before is not None\n        assert before.status is TaskStatus.FAILED\n        assert {item.node_id: item.status for item in before.node_states}["active"] is TaskStatus.READY\n\n        service = DefaultTaskService(state.task, _AllowAuthorization())\n        view = await service.cancel_graph(\n            graph.graph_id,\n            CancelGraphRequest(\n                trusted_workspace_principal("tenant"),\n                "service-cancel-override-0001",\n            ),\n        )\n\n        assert view.status is TaskStatus.CANCELLED\n        after = await repository.snapshot_graph(graph.graph_id, tenant_id="tenant")\n        assert after is not None\n        states = {item.node_id: item.status for item in after.node_states}\n        assert after.status is TaskStatus.CANCELLED\n        assert states["failed"] is TaskStatus.FAILED\n        assert states["active"] is TaskStatus.CANCELLED\n    finally:\n        await state.close()\n\n\n'''
if text.count(anchor) != 1:
    raise RuntimeError("test insertion anchor mismatch")
path.write_text(text.replace(anchor, test + anchor, 1))

print("terminal cancel closure applied")
