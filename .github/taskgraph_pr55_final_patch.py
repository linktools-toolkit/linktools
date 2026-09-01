#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def replace_once(path: str, old: str, new: str) -> None:
    target = ROOT / path
    text = target.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"expected one patch anchor in {path}, found {count}")
    target.write_text(text.replace(old, new), encoding="utf-8")


def run(*args: str) -> None:
    subprocess.run(args, cwd=ROOT, check=True)


def main() -> None:
    replace_once(
        "tests/ai/test_durable_contract_repairs.py",
        '''class _LostTaskWaiter:\n    def owns_graph(self, graph_id: str, *, tenant_id: str) -> bool:\n        del graph_id, tenant_id\n        return False\n\n    async def wait_graph_activity(self, graph_id: str, *, tenant_id: str) -> None:\n        del graph_id, tenant_id\n        raise AssertionError("lost owner must not wait for local activity")\n''',
        '''class _LostTaskWaiter:\n    def owns_graph(self, graph_id: str, *, tenant_id: str) -> bool:\n        del graph_id, tenant_id\n        return False\n\n    def graph_activity_generation(\n        self,\n        graph_id: str,\n        *,\n        tenant_id: str,\n    ) -> "int | None":\n        del graph_id, tenant_id\n        return None\n\n    async def wait_graph_activity(\n        self,\n        graph_id: str,\n        *,\n        tenant_id: str,\n        after_generation: "int | None" = None,\n    ) -> None:\n        del graph_id, tenant_id, after_generation\n        raise AssertionError("lost owner must not wait for local activity")\n''',
    )
    replace_once(
        "tests/ai/test_durable_contract_repairs.py",
        '''class _TerminalTaskWaiter:\n    def __init__(self, error: "AIError | None" = None) -> None:\n        self.error = error\n        self.started = asyncio.Event()\n        self.release = asyncio.Event()\n        self.owned = True\n\n    def owns_graph(self, graph_id: str, *, tenant_id: str) -> bool:\n        del graph_id, tenant_id\n        return self.owned\n\n    async def wait_graph_activity(self, graph_id: str, *, tenant_id: str) -> None:\n        del graph_id, tenant_id\n        self.started.set()\n        if self.error is not None:\n            raise self.error\n        await self.release.wait()\n        self.owned = False\n''',
        '''class _TerminalTaskWaiter:\n    def __init__(self, error: "AIError | None" = None) -> None:\n        self.error = error\n        self.started = asyncio.Event()\n        self.release = asyncio.Event()\n        self.owned = True\n\n    def owns_graph(self, graph_id: str, *, tenant_id: str) -> bool:\n        del graph_id, tenant_id\n        return self.owned\n\n    def graph_activity_generation(\n        self,\n        graph_id: str,\n        *,\n        tenant_id: str,\n    ) -> int:\n        del graph_id, tenant_id\n        return 0\n\n    async def wait_graph_activity(\n        self,\n        graph_id: str,\n        *,\n        tenant_id: str,\n        after_generation: "int | None" = None,\n    ) -> None:\n        del graph_id, tenant_id, after_generation\n        self.started.set()\n        if self.error is not None:\n            raise self.error\n        await self.release.wait()\n        self.owned = False\n''',
    )
    replace_once(
        "tests/ai/test_task_reliable_review_regressions.py",
        '''@pytest.mark.asyncio\nasync def test_explicit_cancel_cleans_running_node_without_local_scheduler_ownership() -> None:\n''',
        '''@pytest.mark.asyncio\nasync def test_explicit_cancel_keeps_failed_node_but_marks_graph_cancelled() -> None:\n    state = RuntimeState.in_memory()\n    await state.initialize(namespace="task-cancel-after-failure", tenant_id="tenant")\n    try:\n        repository = state.task.tasks\n        graph = TaskGraph(\n            "cancel-after-failure",\n            (TaskNode("failed"), TaskNode("active")),\n        )\n        await repository.create_graph(graph, tenant_id="tenant")\n        lease = await repository.claim(\n            graph.graph_id,\n            "failed",\n            tenant_id="tenant",\n            owner="worker",\n            lease_seconds=30,\n        )\n        await repository.fail(\n            lease,\n            tenant_id="tenant",\n            error_code=ErrorCode.TASK_NODE_FAILED.value,\n            error_digest="a" * 64,\n        )\n\n        before = await repository.snapshot_graph(graph.graph_id, tenant_id="tenant")\n        assert before is not None\n        assert before.status is TaskStatus.PENDING\n\n        view = await repository.cancel_graph(graph.graph_id, tenant_id="tenant")\n        snapshot = await repository.snapshot_graph(graph.graph_id, tenant_id="tenant")\n        assert snapshot is not None\n        by_id = {node.node_id: node for node in snapshot.node_states}\n        assert view.status is TaskStatus.CANCELLED\n        assert snapshot.status is TaskStatus.CANCELLED\n        assert by_id["failed"].status is TaskStatus.FAILED\n        assert by_id["active"].status is TaskStatus.CANCELLED\n    finally:\n        await state.close()\n\n\n@pytest.mark.asyncio\nasync def test_explicit_cancel_cleans_running_node_without_local_scheduler_ownership() -> None:\n''',
    )

    run("git", "diff", "--check")
    run(
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "tests/ai/test_durable_contract_repairs.py::test_task_wait_returns_current_terminal_snapshot_without_local_wait",
        "tests/ai/test_durable_contract_repairs.py::test_task_wait_prefers_terminal_truth_after_scheduler_failure",
        "tests/ai/test_task_reliable_review_regressions.py::test_explicit_cancel_keeps_failed_node_but_marks_graph_cancelled",
        "tests/ai/test_task_runtime_recovery.py::test_sqlite_runtime_open_recovers_expired_task_lease",
        "tests/ai/test_task_sqlite_cas_convergence.py::test_sqlite_public_runtime_task_failure_blocks_dependency",
    )
    run(sys.executable, "manage.py", "check", "linktools-ai")
    run(sys.executable, "manage.py", "build", "linktools-ai")
    run(sys.executable, "manage.py", "verify", "linktools-ai")

    run("git", "config", "user.name", "github-actions[bot]")
    run("git", "config", "user.email", "41898282+github-actions[bot]@users.noreply.github.com")
    run("git", "add", "tests/ai/test_durable_contract_repairs.py", "tests/ai/test_task_reliable_review_regressions.py")
    run("git", "diff", "--cached", "--check")
    run("git", "commit", "-m", "test(ai-task): close cancellation observation regressions")
    run("git", "push", "origin", "HEAD:feat/ai-taskgraph-reliable-mixed-nodes")


if __name__ == "__main__":
    main()
