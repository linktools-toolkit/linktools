from datetime import datetime, timezone

import pytest

from linktools.ai.execution.models import RunApproval, RunSnapshot
from linktools.ai.execution.persistence.local import LocalExecutionBackend
from linktools.ai.execution.store import ExecutionStore
from linktools.ai.execution.commands import ClaimRun, CompleteRun, DecideRunApproval, FailRun, PauseRun, StartRun
from linktools.ai.execution.run import RunDefinition, RunKind, RunStatus, RunUsage, RunnableType


def definition(name: str) -> RunDefinition:
    return RunDefinition(name, RunnableType.AGENT, "agent-spec.v1", {"id": name}, name)


@pytest.mark.asyncio
async def test_local_constructor_is_lazy_and_session_context_uses_snapshot(tmp_path):
    root = tmp_path / "data"
    store = ExecutionStore(LocalExecutionBackend(root))
    assert not root.exists()
    await store.create_session(session_id="s", user_id="u", tenant_id="t")
    run = await store.start_run(
        StartRun("r", "s", RunKind.USER_TURN, definition("agent"), "hello")
    )
    claimed = await store.claim_run(ClaimRun(run.id, "worker", datetime.now(timezone.utc), __import__("datetime").timedelta(minutes=5)))
    now = datetime.now(timezone.utc)
    snapshot = RunSnapshot("run-snapshot.v1", "r", 1, ({"role": "user", "content": "hello"},), {"ok": True}, RunStatus.COMPLETED, RunUsage(), 0, now)
    await store.complete_run(CompleteRun("r", "worker", claimed.lease.fence, snapshot))
    assert await store.load_session_context("s") == snapshot.resume_messages


@pytest.mark.asyncio
async def test_local_trace_is_append_only_and_numbered(tmp_path):
    store = ExecutionStore(LocalExecutionBackend(tmp_path / "data"))
    await store.create_session(session_id="s", user_id=None, tenant_id=None)
    await store.start_run(StartRun("r", "s", RunKind.BACKGROUND, definition("agent"), ""))
    sequence = await store.append_trace_steps(
        "r", expected_sequence=0,
        steps=(type("Step", (), {"kind": "model_interaction", "payload": {"request": "x"}, "created_at": datetime.now(timezone.utc)})(),),
    )
    assert sequence == 1
    assert (tmp_path / "data/execution/runs/r/trace/00000000000000000001.json").exists()
    assert len(await store.list_trace_steps("r")) == 1


@pytest.mark.asyncio
async def test_local_rejects_unbounded_page_size(tmp_path):
    store = ExecutionStore(LocalExecutionBackend(tmp_path / "data"))
    await store.create_session(session_id="s", user_id=None, tenant_id=None)
    with pytest.raises(ValueError):
        await store.list_session_turns("s", limit=101)


@pytest.mark.asyncio
async def test_child_runs_do_not_create_session_turns_and_failed_runs_keep_snapshot(tmp_path):
    store = ExecutionStore(LocalExecutionBackend(tmp_path / "data"))
    await store.create_session(session_id="s", user_id=None, tenant_id=None)
    await store.start_run(StartRun("root", "s", RunKind.USER_TURN, definition("agent"), "p"))
    child = await store.start_run(StartRun("child", "s", RunKind.SUBAGENT, definition("child"), "", root_run_id="root", parent_run_id="root"))
    assert child.session_turn_sequence is None
    assert (await store.list_session_turns("s")).items[0].run_id == "root"
    claimed = await store.claim_run(ClaimRun("root", "worker", datetime.now(timezone.utc), __import__("datetime").timedelta(minutes=5)))
    snapshot = RunSnapshot("run-snapshot.v1", "root", 1, (), None, RunStatus.FAILED, RunUsage(), 0, datetime.now(timezone.utc))
    await store.fail_run(FailRun("root", "worker", claimed.lease.fence, snapshot))
    assert await store.get_snapshot("root") == snapshot
    assert (await store.get_session("s")).latest_completed_run_id is None


@pytest.mark.asyncio
async def test_pause_and_approval_decision_share_execution_record(tmp_path):
    store = ExecutionStore(LocalExecutionBackend(tmp_path / "data"))
    await store.create_session(session_id="s", user_id=None, tenant_id=None)
    run = await store.start_run(StartRun("r", "s", RunKind.USER_TURN, definition("agent"), "p"))
    claimed = await store.claim_run(ClaimRun("r", "worker", datetime.now(timezone.utc), __import__("datetime").timedelta(minutes=5)))
    approval = RunApproval("approval", "call", "tool", {"x": 1})
    snapshot = RunSnapshot("run-snapshot.v1", "r", 1, (), None, RunStatus.PAUSED, RunUsage(), 0, datetime.now(timezone.utc))
    await store.pause_run(PauseRun("r", "worker", claimed.lease.fence, snapshot, approval))
    decided = await store.decide_approval(DecideRunApproval("r", "approval", "allow", "user:u"))
    assert decided.pending_approval.decision == "allow"
