from datetime import datetime, timezone

import pytest

from linktools.ai.execution.models import RunApproval, RunDefinitionSnapshot, RunKind, RunSnapshot, RunStatus, RunUsage
from linktools.ai.execution.persistence.local import LocalExecutionStore


@pytest.mark.asyncio
async def test_local_constructor_is_lazy_and_session_context_uses_snapshot(tmp_path):
    root = tmp_path / "data"
    store = LocalExecutionStore(root)
    assert not root.exists()
    await store.create_session(session_id="s", user_id="u", tenant_id="t")
    run = await store.start_run(
        run_id="r",
        session_id="s",
        kind=RunKind.USER_TURN,
        definition=RunDefinitionSnapshot("agent"),
        user_prompt="hello",
    )
    claimed = await store.claim_run(run.id, owner="worker", expected_fence=run.execution_fence)
    now = datetime.now(timezone.utc)
    snapshot = RunSnapshot("run-snapshot.v1", "r", 1, ({"role": "user", "content": "hello"},), {"ok": True}, RunStatus.COMPLETED, RunUsage(), 0, now)
    await store.complete_run("r", owner="worker", fence=claimed.execution_fence, snapshot=snapshot)
    assert await store.load_session_context("s") == snapshot.resume_messages


@pytest.mark.asyncio
async def test_local_trace_is_append_only_and_numbered(tmp_path):
    store = LocalExecutionStore(tmp_path / "data")
    await store.create_session(session_id="s", user_id=None, tenant_id=None)
    await store.start_run(run_id="r", session_id="s", kind=RunKind.BACKGROUND, definition=RunDefinitionSnapshot("agent"))
    sequence = await store.append_trace_steps(
        "r", expected_sequence=0,
        steps=(type("Step", (), {"kind": "model_interaction", "payload": {"request": "x"}, "created_at": datetime.now(timezone.utc)})(),),
    )
    assert sequence == 1
    assert (tmp_path / "data/execution/runs/r/trace/00000000000000000001.json").exists()
    assert len(await store.list_trace_steps("r")) == 1


@pytest.mark.asyncio
async def test_local_rejects_unbounded_page_size(tmp_path):
    store = LocalExecutionStore(tmp_path / "data")
    await store.create_session(session_id="s", user_id=None, tenant_id=None)
    with pytest.raises(ValueError):
        await store.list_session_turns("s", limit=101)


@pytest.mark.asyncio
async def test_child_runs_do_not_create_session_turns_and_failed_runs_keep_snapshot(tmp_path):
    store = LocalExecutionStore(tmp_path / "data")
    await store.create_session(session_id="s", user_id=None, tenant_id=None)
    await store.start_run(run_id="root", session_id="s", kind=RunKind.USER_TURN, definition=RunDefinitionSnapshot("agent"), user_prompt="p")
    child = await store.start_run(run_id="child", session_id="s", kind=RunKind.SUBAGENT, parent_run_id="root", root_run_id="root", definition=RunDefinitionSnapshot("child"))
    assert child.session_turn_sequence is None
    assert (await store.list_session_turns("s")).items[0].run_id == "root"
    claimed = await store.claim_run("root", owner="worker")
    snapshot = RunSnapshot("run-snapshot.v1", "root", 1, (), None, RunStatus.FAILED, RunUsage(), 0, datetime.now(timezone.utc))
    await store.fail_run("root", owner="worker", fence=claimed.execution_fence, snapshot=snapshot)
    assert await store.get_snapshot("root") == snapshot
    assert (await store.get_session("s")).latest_completed_run_id is None


@pytest.mark.asyncio
async def test_pause_and_approval_decision_share_execution_record(tmp_path):
    store = LocalExecutionStore(tmp_path / "data")
    await store.create_session(session_id="s", user_id=None, tenant_id=None)
    run = await store.start_run(run_id="r", session_id="s", kind=RunKind.USER_TURN, definition=RunDefinitionSnapshot("agent"), user_prompt="p")
    claimed = await store.claim_run("r", owner="worker")
    approval = RunApproval("approval", "call", "tool", {"x": 1})
    snapshot = RunSnapshot("run-snapshot.v1", "r", 1, (), None, RunStatus.PAUSED, RunUsage(), 0, datetime.now(timezone.utc))
    await store.pause_run("r", owner="worker", fence=claimed.execution_fence, snapshot=snapshot, pending_approval=approval)
    decided = await store.decide_approval("r", approval_id="approval", decision="allow", decided_by="user:u")
    assert decided.pending_approval.decision == "allow"
