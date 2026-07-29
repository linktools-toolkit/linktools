import asyncio
import threading
from datetime import datetime, timezone

import pytest

from linktools.ai.execution.domain import RunApproval, RunError
from linktools.ai.execution.snapshots import AgentSnapshotData
from linktools.ai.execution.persistence.local import LocalExecutionBackend
from linktools.ai.execution.persistence import local as local_module
from linktools.ai.errors import StorageConflictError
from linktools.ai.execution.store import ExecutionStore
from linktools.ai.execution.commands import AbortExecution, ClaimExecution, CompleteExecution, DecideApproval, FailExecution, PauseExecution, ResumeExecution, StartExecution
from linktools.ai.execution.domain import RunDefinition, RunKind, RunStatus, RunUsage, RunnableType


def definition(name: str) -> RunDefinition:
    return RunDefinition(name, RunnableType.AGENT, "agent-spec.v1", {"id": name}, name)


@pytest.mark.asyncio
async def test_concurrent_same_owner_session_creation_is_idempotent(tmp_path):
    store = LocalExecutionBackend(tmp_path / "data")
    sessions = await asyncio.gather(
        *(
            store.create_session(
                session_id="s", user_id="u", tenant_id="t"
            )
            for _ in range(8)
        )
    )
    assert {session.id for session in sessions} == {"s"}
    assert {session.user_id for session in sessions} == {"u"}
    assert {session.tenant_id for session in sessions} == {"t"}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("first", "second"),
    [
        (("u", "tenant-a"), ("u", "tenant-b")),
        (("user-a", "t"), ("user-b", "t")),
    ],
)
async def test_concurrent_session_creation_fences_ownership(
    tmp_path, first, second
):
    store = LocalExecutionBackend(tmp_path / "data")
    results = await asyncio.gather(
        store.create_session(
            session_id="s", user_id=first[0], tenant_id=first[1]
        ),
        store.create_session(
            session_id="s", user_id=second[0], tenant_id=second[1]
        ),
        return_exceptions=True,
    )
    assert sum(isinstance(item, StorageConflictError) for item in results) == 1


@pytest.mark.asyncio
async def test_local_constructor_is_lazy_and_session_context_uses_snapshot(tmp_path):
    root = tmp_path / "data"
    store = LocalExecutionBackend(root)
    assert not root.exists()
    await store.create_session(session_id="s", user_id="u", tenant_id="t")
    run = await store.start_run(
        StartExecution("r", "s", RunKind.USER_TURN, definition("agent"), "hello")
    )
    assert run.input == "hello"
    claimed = await store.claim_run(ClaimExecution(run.id, "worker", datetime.now(timezone.utc), __import__("datetime").timedelta(minutes=5)))
    now = datetime.now(timezone.utc)
    snapshot = AgentSnapshotData(({"role": "user", "content": "hello"},), {"ok": True}, RunUsage(), 0)
    await store.complete_run(CompleteExecution("r", "worker", claimed.lease.fence, snapshot))
    assert await store.load_session_context("s") == snapshot.resume_messages


@pytest.mark.asyncio
async def test_local_trace_is_append_only_and_numbered(tmp_path):
    store = LocalExecutionBackend(tmp_path / "data")
    await store.create_session(session_id="s", user_id=None, tenant_id=None)
    await store.start_run(StartExecution("r", "s", RunKind.BACKGROUND, definition("agent"), ""))
    sequence = await store.append_trace_steps(
        "r", expected_sequence=0,
        steps=(type("Step", (), {"kind": "model_interaction", "payload": {"request": "x"}, "created_at": datetime.now(timezone.utc)})(),),
    )
    assert sequence == 1
    assert (tmp_path / "data/execution/runs/r/trace/00000000000000000001.json").exists()
    assert len(await store.list_trace_steps("r")) == 1


@pytest.mark.asyncio
async def test_local_rejects_unbounded_page_size(tmp_path):
    store = LocalExecutionBackend(tmp_path / "data")
    await store.create_session(session_id="s", user_id=None, tenant_id=None)
    with pytest.raises(ValueError):
        await store.list_session_turns("s", limit=101)


@pytest.mark.asyncio
async def test_child_runs_do_not_create_session_turns_and_failed_runs_keep_snapshot(tmp_path):
    store = LocalExecutionBackend(tmp_path / "data")
    await store.create_session(session_id="s", user_id=None, tenant_id=None)
    await store.start_run(StartExecution("root", "s", RunKind.USER_TURN, definition("agent"), "p"))
    child = await store.start_run(StartExecution("child", "s", RunKind.SUBAGENT, definition("child"), "", root_execution_id="root", parent_execution_id="root"))
    assert child.session_turn_sequence is None
    assert (await store.list_session_turns("s")).items[0].run_id == "root"
    claimed = await store.claim_run(ClaimExecution("root", "worker", datetime.now(timezone.utc), __import__("datetime").timedelta(minutes=5)))
    snapshot = AgentSnapshotData((), None, RunUsage(), 0)
    await store.fail_run(FailExecution("root", "worker", claimed.lease.fence, snapshot))
    assert (await store.get_snapshot("root")).final_output is None
    assert (await store.get_session("s")).latest_completed_run_id is None


@pytest.mark.asyncio
async def test_pause_and_approval_decision_share_execution_record(tmp_path):
    store = LocalExecutionBackend(tmp_path / "data")
    await store.create_session(session_id="s", user_id=None, tenant_id=None)
    run = await store.start_run(StartExecution("r", "s", RunKind.USER_TURN, definition("agent"), "p"))
    claimed = await store.claim_run(ClaimExecution("r", "worker", datetime.now(timezone.utc), __import__("datetime").timedelta(minutes=5)))
    approval = RunApproval("approval", "call", "tool", {"x": 1})
    snapshot = AgentSnapshotData((), None, RunUsage(), 0)
    await store.pause_run(PauseExecution("r", "worker", claimed.lease.fence, snapshot, approval))
    decided = await store.decide_approval(DecideApproval("r", "approval", "allow", "user:u"))
    assert decided.approval.decision == "allow"
    await store.resume_run(ResumeExecution("r"))
    reclaimed = await store.claim_run(
        ClaimExecution(
            "r",
            "worker",
            datetime.now(timezone.utc),
            __import__("datetime").timedelta(minutes=5),
        )
    )
    completed = await store.complete_run(
        CompleteExecution(
            "r", "worker", reclaimed.lease.fence, snapshot
        )
    )
    assert completed.approval == decided.approval


@pytest.mark.asyncio
async def test_abort_run_persists_error_without_a_snapshot(tmp_path):
    store = LocalExecutionBackend(tmp_path / "data")
    await store.create_session(session_id="s", user_id=None, tenant_id=None)
    await store.start_run(StartExecution("r", "s", RunKind.USER_TURN, definition("agent"), "p"))
    claimed = await store.claim_run(ClaimExecution("r", "worker", datetime.now(timezone.utc), __import__("datetime").timedelta(minutes=5)))
    aborted = await store.abort_run(AbortExecution("r", "worker", claimed.lease.fence, RunError("RuntimeError", "boom"), 3))
    assert aborted.status is RunStatus.FAILED
    assert aborted.error == RunError("RuntimeError", "boom")
    assert aborted.lease.owner is None
    assert aborted.trace_sequence == 3
    assert await store.get_snapshot("r") is None


def _fail_after_publication(monkeypatch, write_point: int) -> None:
    real_replace = local_module.os.replace
    publications = 0

    def replace_with_failure(source, target):
        nonlocal publications
        real_replace(source, target)
        if str(source).endswith(".staged.json"):
            publications += 1
            if publications == write_point:
                raise OSError(f"injected publication failure {write_point}")

    monkeypatch.setattr(local_module.os, "replace", replace_with_failure)


@pytest.mark.asyncio
@pytest.mark.parametrize("write_point", range(1, 5))
async def test_start_run_journal_recovers_every_publication_point(
    tmp_path, monkeypatch, write_point
):
    root = tmp_path / "data"
    backend = LocalExecutionBackend(root)
    store = backend
    await store.create_session(session_id="s", user_id="u", tenant_id="t")
    _fail_after_publication(monkeypatch, write_point)

    with pytest.raises(OSError):
        await store.start_run(
            StartExecution(
                "r", "s", RunKind.USER_TURN, definition("agent"), "hello"
            )
        )

    recovered = LocalExecutionBackend(root)
    # Session is deliberately queried first: its aggregate journal must be
    # sufficient to recover without a global run-directory scan.
    session = await recovered.get_session("s")
    run = await recovered.get_run("r")
    if write_point < 3:
        assert run is None
        assert session.next_turn_sequence == 1
    else:
        assert run is not None
        assert session.next_turn_sequence == 2
        assert (await recovered.list_session_turns("s")).items[0].run_id == "r"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("terminal", "write_point"),
    [
        *((RunStatus.PAUSED, point) for point in range(1, 5)),
        *((RunStatus.COMPLETED, point) for point in range(1, 6)),
    ],
)
async def test_terminal_journal_recovers_every_publication_point(
    tmp_path, monkeypatch, terminal, write_point
):
    root = tmp_path / f"{terminal.value}-{write_point}"
    backend = LocalExecutionBackend(root)
    store = backend
    await store.create_session(session_id="s", user_id="u", tenant_id="t")
    await store.start_run(
        StartExecution(
            "r", "s", RunKind.USER_TURN, definition("agent"), "hello"
        )
    )
    claimed = await store.claim_run(
        ClaimExecution(
            "r",
            "worker",
            datetime.now(timezone.utc),
            __import__("datetime").timedelta(minutes=5),
        )
    )
    snapshot = AgentSnapshotData(
        ({"role": "user", "content": "hello"},),
        {"ok": True},
        RunUsage(),
        0,
    )
    _fail_after_publication(monkeypatch, write_point)

    with pytest.raises(OSError):
        if terminal is RunStatus.PAUSED:
            await store.pause_run(
                PauseExecution(
                    "r",
                    "worker",
                    claimed.lease.fence,
                    snapshot,
                    RunApproval("approval", "call", "tool", {"x": 1}),
                )
            )
        else:
            await store.complete_run(
                CompleteExecution(
                    "r", "worker", claimed.lease.fence, snapshot
                )
            )

    recovered = LocalExecutionBackend(root)
    run = await recovered.get_run("r")
    if write_point < (4 if terminal is RunStatus.PAUSED else 4):
        assert run.status is RunStatus.RUNNING
    else:
        assert run.status is terminal
        assert (await recovered.get_snapshot("r")).status is terminal
        turn = (await recovered.list_session_turns("s")).items[0]
        assert turn.status is terminal
        if terminal is RunStatus.COMPLETED:
            assert (await recovered.get_session("s")).latest_completed_run_id == "r"


@pytest.mark.asyncio
async def test_published_manifest_with_missing_turn_is_corruption(
    tmp_path, monkeypatch
):
    root = tmp_path / "data"
    store = LocalExecutionBackend(root)
    await store.create_session(session_id="s", user_id="u", tenant_id="t")
    _fail_after_publication(monkeypatch, 3)
    with pytest.raises(OSError):
        await store.start_run(
            StartExecution(
                "r", "s", RunKind.USER_TURN, definition("agent"), "hello"
            )
        )
    (root / "execution/sessions/s/turns/00000000000000000001.json").unlink()

    with pytest.raises(local_module.StorageCorruptionError):
        await LocalExecutionBackend(root).get_run("r")


@pytest.mark.asyncio
async def test_live_publication_and_recovery_share_aggregate_lock(
    tmp_path, monkeypatch
):
    root = tmp_path / "data"
    store = LocalExecutionBackend(root)
    await store.create_session(session_id="s", user_id="u", tenant_id="t")
    real_replace = local_module.os.replace
    publication_started = threading.Event()
    allow_publication = threading.Event()

    def blocking_replace(source, target):
        if (
            str(source).endswith(".staged.json")
            and not publication_started.is_set()
        ):
            publication_started.set()
            assert allow_publication.wait(timeout=5)
        return real_replace(source, target)

    monkeypatch.setattr(local_module.os, "replace", blocking_replace)
    publisher = asyncio.create_task(
        store.start_run(
            StartExecution(
                "r", "s", RunKind.USER_TURN, definition("agent"), "hello"
            )
        )
    )
    assert await asyncio.to_thread(publication_started.wait, 5)
    reader = asyncio.create_task(store.get_session("s"))
    await asyncio.sleep(0)
    assert not reader.done()
    allow_publication.set()
    await publisher
    session = await reader
    assert session.next_turn_sequence == 2
    assert (await store.get_run("r")).status is RunStatus.PENDING


@pytest.mark.asyncio
async def test_run_side_recovery_holds_session_then_run_locks(
    tmp_path, monkeypatch
):
    root = tmp_path / "data"
    initial = LocalExecutionBackend(root)
    await initial.create_session(session_id="s", user_id="u", tenant_id="t")
    _fail_after_publication(monkeypatch, 3)
    with pytest.raises(OSError):
        await initial.start_run(
            StartExecution(
                "r", "s", RunKind.USER_TURN, definition("agent"), "hello"
            )
        )

    backend = LocalExecutionBackend(root)
    recovered = backend
    real_recover = backend._recover_journal
    recovery_started = threading.Event()
    allow_recovery = threading.Event()

    def blocking_recovery(path):
        recovery_started.set()
        assert allow_recovery.wait(timeout=5)
        return real_recover(path)

    monkeypatch.setattr(backend, "_recover_journal", blocking_recovery)
    run_reader = asyncio.create_task(recovered.get_run("r"))
    assert await asyncio.to_thread(recovery_started.wait, 5)
    session_reader = asyncio.create_task(recovered.get_session("s"))
    await asyncio.sleep(0)
    assert not session_reader.done()
    allow_recovery.set()
    assert (await run_reader).status is RunStatus.PENDING
    assert (await session_reader).next_turn_sequence == 2
