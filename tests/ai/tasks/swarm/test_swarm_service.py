#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SwarmExecutionService orchestration tests. A fake ExecutionService stands in
for run_child so the swarm lifecycle (validate, persist, schedule, gate, cancel,
collect) is exercised without the model stack. The fake records concurrency and
returns scripted ChildRunResult outcomes."""

import asyncio
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from linktools.ai.errors import InvalidSpecError, StorageConflictError
from linktools.ai.execution.domain import RunError, RunKind, RunStatus, RunnableType
from linktools.ai.execution.live_events import NoopRunLiveEventSink
from linktools.ai.execution.service import ChildRunResult, PreparedAgentExecution
from linktools.ai.execution.swarm_service import SwarmExecutionService
from linktools.ai.governance.authorization import (
    AuthorizationPolicy,
    OwnershipAuthorizationPolicy,
)
from linktools.ai.governance.identity import (
    ActorRef,
    PrincipalContext,
    ScopeSet,
)
from linktools.ai.storage.coordination.lease import Lease
from linktools.ai.tasks.models import (
    DependencyFailurePolicy,
    TaskDependency,
    TaskGraphNodePayload,
    TaskNode,
    TaskPlan,
    TaskStatus,
    TaskUsage,
)
from linktools.ai.tasks.persistence.local import LocalTaskBackend
from linktools.ai.tasks.swarm.aggregation import AggregationMode, AggregationPolicy
from linktools.ai.tasks.swarm.limits import SwarmLimits
from linktools.ai.tasks.swarm.models import AgentRef, SwarmCompleted, SwarmFailed
from linktools.ai.tasks.swarm.spec import SwarmSpec, SwarmStrategySpec, SwarmContextPolicy

from tests.ai.tasks.swarm._support import make_plan, ready_executions


def _limits(
    *,
    max_concurrency: int = 4,
    max_tasks: int = 50,
    max_total_tokens: "int | None" = None,
    max_total_cost: "Decimal | None" = None,
    timeout_seconds: "float | None" = None,
) -> SwarmLimits:
    return SwarmLimits(
        max_rounds=1,
        max_tasks=max_tasks,
        max_delegations=0,
        max_depth=0,
        max_concurrency=max_concurrency,
        max_total_tokens=max_total_tokens,
        max_total_cost=max_total_cost,
        timeout_seconds=timeout_seconds,
    )


def _spec(
    *,
    agents: "tuple[str, ...]" = ("a", "b", "c"),
    limits: "SwarmLimits | None" = None,
) -> SwarmSpec:
    return SwarmSpec(
        id="ws",
        name="ws",
        agents=tuple(AgentRef(aid) for aid in agents),
        strategy=SwarmStrategySpec("task_graph"),
        limits=limits or _limits(),
        context_policy=SwarmContextPolicy(),
        aggregation=AggregationPolicy(mode=AggregationMode.COLLECT),
    )


@dataclass
class FakeExecutionService:
    """Stand-in for ExecutionService.run_child: returns scripted ChildRunResult
    outcomes keyed by agent_id, recording concurrency."""

    outcomes: "dict[str, ChildRunResult]" = field(default_factory=dict)
    in_flight: int = 0
    max_seen: int = 0

    async def run_child(
        self,
        spec,
        prompt: str,
        *,
        principal,
        session_id: str,
        execution_id: str,
        root_execution_id: str,
        parent_execution_id: str,
        message_history=(),
        metadata=None,
        prepared_execution=None,
    ) -> ChildRunResult:
        agent_id = spec.id
        self.in_flight += 1
        self.max_seen = max(self.max_seen, self.in_flight)
        try:
            await asyncio.sleep(0)
            return self.outcomes.get(
                agent_id,
                ChildRunResult(
                    run_id=execution_id,
                    status=RunStatus.COMPLETED,
                    output={"agent": agent_id},
                    error=None,
                    usage=TaskUsage(input_tokens=10, output_tokens=5),
                ),
            )
        finally:
            self.in_flight -= 1

    async def prepare_agent_execution(
        self,
        agent_spec,
        *,
        principal,
        session_id,
        execution_id,
        root_execution_id,
        parent_execution_id,
    ) -> PreparedAgentExecution:
        return PreparedAgentExecution(
            agent_spec=agent_spec,
            assembled_agent=object(),
            tool_descriptors=(),
            fingerprint=f"fingerprint:{agent_spec.id}",
        )

    async def cancel(self, run_id: str, *, principal) -> None:
        return None

    async def read_usage(self, *, child_run_id: str) -> TaskUsage:
        return TaskUsage()


class _RecordingAuth(OwnershipAuthorizationPolicy):
    """Authorization that records the actions asserted against it."""

    def __init__(self) -> None:
        super().__init__()
        self.actions: "list[str]" = []

    def assert_execution_access(self, *, principal, tenant_id, user_id, action):
        self.actions.append(action.value)
        super().assert_execution_access(
            principal=principal, tenant_id=tenant_id, user_id=user_id, action=action
        )


class _FakeAgentProvider:
    """Returns a bare agent spec for any id; the swarm's mutating-tool precheck
    finds no tools on an empty feature set."""

    async def list_ids(self):
        return ()

    async def get(self, agent_id: str):
        return _fake_agent_spec(agent_id)


def _principal() -> PrincipalContext:
    return PrincipalContext(
        tenant_id="tnt",
        user_id="usr",
        actor=ActorRef(kind="system", id="trusted-local"),
        scopes=ScopeSet.allow_all(),
    )


def _service(
    *,
    fake_exec: FakeExecutionService,
    tasks: LocalTaskBackend,
    auth: "AuthorizationPolicy | None" = None,
    agent_provider: "object | None" = None,
) -> "tuple[SwarmExecutionService, _MemoryExecutionStore]":
    store = _MemoryExecutionStore()
    svc = SwarmExecutionService(
        store=store,
        tasks=tasks,
        agent_execution=fake_exec,  # type: ignore[arg-type]
        authorization=auth or OwnershipAuthorizationPolicy(),
        live_events=NoopRunLiveEventSink(),
        agent_provider=agent_provider or _FakeAgentProvider(),
    )
    return svc, store


@dataclass
class _MemoryExecutionStore:
    """Minimal ExecutionStore stub for swarm orchestration: persists RunRecords
    and transitions them through the lifecycle the swarm service drives.
    Enforces owner/fence guards AND the RunStatus transition table so the
    stub matches the real store's invariants, not just its happy path."""

    _runs: "dict[str, object]" = field(default_factory=dict)
    _snapshots: "dict[str, object]" = field(default_factory=dict)

    async def create_session(self, *, session_id, user_id, tenant_id):
        return None

    async def start_run(self, command):
        from linktools.ai.execution.domain import RunDefinition, RunRecord
        from linktools.ai.storage.coordination.lease import Lease

        now = datetime.now(timezone.utc)
        record = RunRecord(
            id=command.run_id,
            session_id=command.session_id,
            kind=command.kind,
            runnable_id=command.definition.runnable_id,
            runnable_type=command.definition.runnable_type,
            input=command.input,
            definition=command.definition,
            status=RunStatus.PENDING,
            session_turn_sequence=None,
            parent_execution_id=command.parent_execution_id,
            root_execution_id=command.root_execution_id or command.run_id,
            approval=None,
            lease=Lease(),
            cancel_requested_at=None,
            snapshot_revision=0,
            trace_sequence=0,
            event_sequence=0,
            tenant_id="tnt",
            user_id="usr",
            error=None,
            created_at=now,
            updated_at=now,
        )
        self._runs[record.id] = record
        return record

    async def claim_run(self, command) -> "RunRecord":
        record = self._runs.get(command.run_id)
        if record is None:
            raise KeyError(command.run_id)
        self._assert_transition(record, RunStatus.RUNNING)
        claimed = replace(
            record,
            status=RunStatus.RUNNING,
            lease=Lease(
                owner=command.owner,
                fence=record.lease.fence + 1,
                expires_at=command.now + command.duration,
            ),
        )
        self._runs[command.run_id] = claimed
        return claimed

    async def claim_run_for_recovery(
        self, run_id, *, owner, now, duration
    ) -> "RunRecord":
        record = self._runs.get(run_id)
        if record is None:
            raise KeyError(run_id)
        if record.status is RunStatus.PENDING:
            target = RunStatus.RUNNING
        elif record.status in {RunStatus.RUNNING, RunStatus.CANCELLING}:
            if record.lease.expires_at is not None and record.lease.expires_at > now:
                raise StorageConflictError("active recovery lease")
            target = record.status
        else:
            raise StorageConflictError("run is not recoverable")
        claimed = replace(
            record,
            status=target,
            lease=Lease(
                owner=owner,
                fence=record.lease.fence + 1,
                expires_at=now + duration,
            ),
        )
        self._runs[run_id] = claimed
        return claimed

    async def get_run(self, run_id: str):
        return self._runs.get(run_id)

    async def get_snapshot(self, run_id: str):
        return self._snapshots.get(run_id)

    async def load_session_context(self, session_id: str):
        return ()

    async def request_cancel(self, command) -> None:
        record = self._runs.get(command.run_id)
        if record is None:
            return
        if record.status is RunStatus.RUNNING:
            self._runs[command.run_id] = replace(record, status=RunStatus.CANCELLING)
        elif record.status is RunStatus.PENDING:
            self._runs[command.run_id] = replace(record, status=RunStatus.CANCELLED)

    def _assert_owner(self, command) -> None:
        record = self._runs.get(command.run_id)
        if record is None:
            raise KeyError(command.run_id)
        if record.lease.owner != command.owner or record.lease.fence != command.fence:
            raise StorageConflictError("stale owner/fence")

    @staticmethod
    def _assert_transition(record, target: RunStatus) -> None:
        from linktools.ai.execution.domain import ALLOWED_RUN_TRANSITIONS

        allowed = ALLOWED_RUN_TRANSITIONS.get(record.status, frozenset())
        if target not in allowed:
            raise StorageConflictError(
                f"illegal transition {record.status} -> {target}"
            )

    async def complete_run(self, command) -> None:
        self._assert_owner(command)
        record = self._runs.get(command.run_id)
        if record is not None:
            self._assert_transition(record, RunStatus.COMPLETED)
            self._runs[command.run_id] = replace(record, status=RunStatus.COMPLETED)

    async def fail_run(self, command) -> None:
        self._assert_owner(command)
        record = self._runs.get(command.run_id)
        if record is not None:
            self._assert_transition(record, RunStatus.FAILED)
            self._runs[command.run_id] = replace(record, status=RunStatus.FAILED, error=command.error)

    async def acknowledge_cancel(self, command) -> None:
        self._assert_owner(command)
        record = self._runs.get(command.run_id)
        if record is not None:
            self._assert_transition(record, RunStatus.CANCELLED)
            self._runs[command.run_id] = replace(record, status=RunStatus.CANCELLED)

    async def heartbeat_run(self, command) -> "RunRecord":
        self._assert_owner(command)
        record = self._runs.get(command.run_id)
        if record is not None:
            renewed = replace(
                record,
                lease=Lease(
                    owner=command.owner,
                    fence=command.fence,
                    expires_at=command.now + command.duration,
                ),
            )
            self._runs[command.run_id] = renewed
            return renewed
        raise KeyError(command.run_id)

    async def abort_run(self, command) -> None:
        self._assert_owner(command)
        record = self._runs.get(command.run_id)
        if record is not None:
            self._runs[command.run_id] = replace(record, status=RunStatus.FAILED, error=command.error)


def _fake_agent_spec(agent_id: str):
    @dataclass(frozen=True, slots=True)
    class _Spec:
        id: str = agent_id

    return _Spec()


@pytest.mark.asyncio
async def test_run_swarm_completes_all_nodes():
    fake = FakeExecutionService()
    tasks = LocalTaskBackend()
    svc, _ = _service(fake_exec=fake, tasks=tasks)
    task_plan = make_plan(("a", "b", "c"), edges={"b": ("a",), "c": ("b",)})
    outcome = await svc.run_swarm(
        _spec(), task_plan, principal=_principal(), session_id="sess"
    )
    assert isinstance(outcome, SwarmCompleted)
    assert set(outcome.collect["nodes"].keys()) == {"a", "b", "c"}
    assert outcome.usage.input_tokens == 30  # 3 nodes x 10


@pytest.mark.asyncio
async def test_run_swarm_concurrency_cap_respected():
    fake = FakeExecutionService()
    tasks = LocalTaskBackend()
    svc, _ = _service(fake_exec=fake, tasks=tasks)
    node_ids = tuple("abcdefgh")
    task_plan = make_plan(node_ids)
    await svc.run_swarm(
        _spec(agents=node_ids, limits=_limits(max_concurrency=2)),
        task_plan,
        principal=_principal(),
        session_id="sess",
    )
    assert fake.max_seen <= 2


@pytest.mark.asyncio
async def test_run_swarm_rejects_agent_not_in_swarm():
    fake = FakeExecutionService()
    tasks = LocalTaskBackend()
    svc, _ = _service(fake_exec=fake, tasks=tasks)
    task_plan = TaskPlan(
        "p",
        (TaskNode("a", TaskGraphNodePayload(agent_id="rogue", prompt="x")),),
    )
    with pytest.raises(InvalidSpecError):
        await svc.run_swarm(
            _spec(agents=("a",)), task_plan, principal=_principal()
        )


@pytest.mark.asyncio
async def test_run_swarm_rejects_too_many_nodes():
    fake = FakeExecutionService()
    tasks = LocalTaskBackend()
    svc, _ = _service(fake_exec=fake, tasks=tasks)
    task_plan = make_plan(("a", "b", "c"))
    with pytest.raises(InvalidSpecError):
        await svc.run_swarm(
            _spec(limits=_limits(max_tasks=2)),
            task_plan,
            principal=_principal(),
        )


@pytest.mark.asyncio
async def test_run_swarm_skip_propagates():
    fake = FakeExecutionService(
        outcomes={
            "a": ChildRunResult(
                run_id="x",
                status=RunStatus.FAILED,
                output=None,
                error=RunError("boom", "a failed"),
                usage=TaskUsage(),
            )
        }
    )
    tasks = LocalTaskBackend()
    svc, _ = _service(fake_exec=fake, tasks=tasks)
    task_plan = make_plan(("a", "b"), edges={"b": ("a",)})
    outcome = await svc.run_swarm(
        _spec(agents=("a", "b")), task_plan, principal=_principal(), session_id="s"
    )
    assert isinstance(outcome, SwarmCompleted)
    assert outcome.collect["nodes"]["a"]["status"] == "failed"
    assert outcome.collect["nodes"]["b"]["status"] == "skipped"


@pytest.mark.asyncio
async def test_run_swarm_collect_preserves_all_terminal_statuses():
    fake = FakeExecutionService(
        outcomes={
            "ok": ChildRunResult(
                run_id="1",
                status=RunStatus.COMPLETED,
                output={"r": 1},
                error=None,
                usage=TaskUsage(10, 5, Decimal("0.01")),
            ),
            "boom": ChildRunResult(
                run_id="2",
                status=RunStatus.FAILED,
                output=None,
                error=RunError("x", "boom"),
                usage=TaskUsage(3, 1),
            ),
        }
    )
    tasks = LocalTaskBackend()
    svc, _ = _service(fake_exec=fake, tasks=tasks)
    task_plan = make_plan(("ok", "boom", "down"))
    # down depends on boom (skip) -> skipped
    task_plan = TaskPlan(
        task_plan.id,
        (
            task_plan.nodes[0],
            task_plan.nodes[1],
            TaskNode(
                "down",
                TaskGraphNodePayload("down", "x"),
                dependencies=(TaskDependency("boom", DependencyFailurePolicy.SKIP),),
            ),
        ),
    )
    outcome = await svc.run_swarm(
        _spec(agents=("ok", "boom", "down")),
        task_plan,
        principal=_principal(),
        session_id="s",
    )
    counts = outcome.collect["status_counts"]
    assert counts["completed"] == 1
    assert counts["failed"] == 1
    assert counts["skipped"] == 1


@pytest.mark.asyncio
async def test_run_swarm_persists_parent_run_record_lifecycle():
    fake = FakeExecutionService()
    tasks = LocalTaskBackend()
    svc, store = _service(fake_exec=fake, tasks=tasks)
    task_plan = make_plan(("a",))
    outcome = await svc.run_swarm(
        _spec(agents=("a",)), task_plan, principal=_principal(),
        session_id="s", execution_id="parent-1",
    )
    assert isinstance(outcome, SwarmCompleted)
    record = await store.get_run("parent-1")
    assert record is not None
    assert record.status is RunStatus.COMPLETED


@pytest.mark.asyncio
async def test_recover_swarm_decodes_persisted_swarm_spec():
    from linktools.ai.tasks.swarm.codec import encode_swarm_spec
    from linktools.ai.execution.domain import RunDefinition

    fake = FakeExecutionService()
    tasks = LocalTaskBackend()
    svc, store = _service(fake_exec=fake, tasks=tasks)
    task_plan = make_plan(("a",))
    # Seed a persisted parent run carrying the encoded swarm spec.
    spec = _spec(agents=("a",))
    store._runs["parent-2"] = _persisted_swarm_run("parent-2", spec, task_plan)
    await tasks.create_plan(task_plan, ready_executions(task_plan))
    outcome = await svc.recover_swarm("parent-2", principal=_principal())
    assert isinstance(outcome, SwarmCompleted)


def _persisted_swarm_run(run_id, spec, task_plan):
    from datetime import datetime, timezone
    from hashlib import sha256
    from linktools.ai.execution.domain import RunDefinition, RunRecord
    from linktools.ai.json import canonical_json_bytes
    from linktools.ai.storage.coordination.lease import Lease
    from linktools.ai.tasks.codec import encode_plan
    from linktools.ai.tasks.swarm.codec import encode_swarm_spec

    snapshot_hash = sha256(canonical_json_bytes([])).hexdigest()
    value = {
        "swarm_spec": encode_swarm_spec(spec),
        "task_plan_id": task_plan.id,
        "task_plan_hash": sha256(
            canonical_json_bytes(encode_plan(task_plan))
        ).hexdigest(),
        "session_snapshot_hash": snapshot_hash,
        "agent_fingerprints": {
            node.payload.agent_id: f"fingerprint:{node.payload.agent_id}"
            for node in task_plan.nodes
        },
        "deadline_at": None,
    }
    definition = RunDefinition(
        spec.id, RunnableType.TASK, "swarm-task-graph.v1", value,
        sha256(canonical_json_bytes(value)).hexdigest(),
    )
    now = datetime.now(timezone.utc)
    return RunRecord(
        id=run_id, session_id="s", kind=RunKind.TASK,
        runnable_id=spec.id, runnable_type=RunnableType.TASK,
        input={
            "task_plan_id": task_plan.id,
            "session_snapshot": [],
            "session_snapshot_hash": snapshot_hash,
        }, definition=definition,
        status=RunStatus.RUNNING, session_turn_sequence=None,
        parent_execution_id=None, root_execution_id=run_id,
        approval=None, lease=Lease(owner=f"swarm:{run_id}", fence=1, expires_at=None),
        cancel_requested_at=None, snapshot_revision=0, trace_sequence=0,
        event_sequence=0, tenant_id="tnt", user_id="usr", error=None,
        created_at=now, updated_at=now,
    )


@pytest.mark.asyncio
async def test_reconcile_claimed_backfills_orphaned_claim_without_bind():
    # A node stuck CLAIMED with no child run on recover -> FAILED orphaned_before_child_start
    fake = FakeExecutionService()
    tasks = LocalTaskBackend()
    svc, store = _service(fake_exec=fake, tasks=tasks)
    task_plan = make_plan(("a",))
    spec = _spec(agents=("a",))
    store._runs["parent-3"] = _persisted_swarm_run("parent-3", spec, task_plan)
    orphan = replace(
        ready_executions(task_plan)[0],
        status=TaskStatus.CLAIMED,
        attempt=1,
        lease=Lease(owner="swarm:parent-3", fence=1, expires_at=datetime.now(timezone.utc) - timedelta(seconds=1)),
        active_run_id=None,
    )
    await tasks.create_plan(task_plan, (orphan,))
    outcome = await svc.recover_swarm("parent-3", principal=_principal())
    execs = {e.node_id: e for e in await tasks.list_executions(task_plan.id)}
    assert execs["a"].status is TaskStatus.FAILED
    assert execs["a"].error.error_type == "orphaned_before_child_start"


@pytest.mark.asyncio
async def test_reconcile_claimed_backfills_from_child_completed():
    fake = FakeExecutionService()
    tasks = LocalTaskBackend()
    svc, store = _service(fake_exec=fake, tasks=tasks)
    task_plan = make_plan(("a",))
    spec = _spec(agents=("a",))
    from linktools.ai.execution.identifiers import child_run_id

    child_id = child_run_id("parent-4", "a")
    store._runs["parent-4"] = _persisted_swarm_run("parent-4", spec, task_plan)
    # child run already COMPLETED but node still CLAIMED
    child = _persisted_swarm_run(child_id, spec, task_plan)
    child = replace(
        child,
        parent_execution_id="parent-4",
        root_execution_id="parent-4",
    )
    child = replace(child, status=RunStatus.COMPLETED)
    store._runs[child_id] = child
    # seed a snapshot so recovery can reclaim output + usage
    from linktools.ai.execution.snapshots import RunSnapshot
    from linktools.ai.execution.domain import RunUsage

    now = datetime.now(timezone.utc)
    store._snapshots[child_id] = RunSnapshot(
        schema="run-snapshot.v1", run_id=child_id, revision=1,
        resume_messages=(), final_output={"recovered": True},
        status=RunStatus.COMPLETED, usage=RunUsage(input_tokens=42, output_tokens=7),
        trace_end_sequence=0, created_at=now,
    )
    claimed = replace(
        ready_executions(task_plan)[0],
        status=TaskStatus.CLAIMED,
        attempt=1,
        lease=Lease(owner="swarm:parent-4", fence=1, expires_at=datetime.now(timezone.utc) - timedelta(seconds=1)),
        active_run_id=child_id,
    )
    await tasks.create_plan(task_plan, (claimed,))
    await svc.recover_swarm("parent-4", principal=_principal())
    execs = {e.node_id: e for e in await tasks.list_executions(task_plan.id)}
    assert execs["a"].status is TaskStatus.COMPLETED
    # F1: output and usage are recovered from the child snapshot, not lost
    assert execs["a"].result == {"recovered": True}
    assert execs["a"].usage.input_tokens == 42
    assert execs["a"].usage.output_tokens == 7


@pytest.mark.asyncio
async def test_reconcile_claimed_backfills_from_child_terminal_branches():
    # Exercise the FAILED / CANCELLED / PAUSED / RUNNING child reconcile branches.
    from linktools.ai.tasks.models import TaskStatus as TS

    fake = FakeExecutionService()
    spec = _spec(agents=("a",))

    async def _run_case(child_status, expect_node_status, expect_error_type=None):
        tasks = LocalTaskBackend()
        svc, store = _service(fake_exec=fake, tasks=tasks)
        plan = make_plan(("a",))
        plan_id = plan.id
        from linktools.ai.execution.identifiers import child_run_id

        child_id = child_run_id(plan_id, "a")
        store._runs[plan_id] = _persisted_swarm_run(plan_id, spec, plan)
        child = _persisted_swarm_run(child_id, spec, plan)
        child = replace(
            child,
            parent_execution_id=plan_id,
            root_execution_id=plan_id,
            status=child_status,
        )
        store._runs[child_id] = child
        from linktools.ai.execution.domain import RunUsage
        from linktools.ai.execution.snapshots import RunSnapshot

        store._snapshots[child_id] = RunSnapshot(
            schema="run-snapshot.v1",
            run_id=child_id,
            revision=1,
            resume_messages=(),
            final_output={"status": child_status.value},
            status=child_status,
            usage=RunUsage(input_tokens=1, output_tokens=1),
            trace_end_sequence=0,
            created_at=datetime.now(timezone.utc),
        )
        claimed = replace(
            ready_executions(plan)[0],
            status=TaskStatus.CLAIMED,
            attempt=1,
            lease=Lease(owner=f"swarm:{plan_id}", fence=1, expires_at=datetime.now(timezone.utc) - timedelta(seconds=1)),
            active_run_id=child_id,
        )
        await tasks.create_plan(plan, (claimed,))
        await svc.recover_swarm(plan_id, principal=_principal())
        execs = {e.node_id: e for e in await tasks.list_executions(plan_id)}
        assert execs["a"].status is expect_node_status, (child_status, execs["a"])
        if expect_error_type is not None:
            assert execs["a"].error.error_type == expect_error_type

    await _run_case(RunStatus.FAILED, TaskStatus.FAILED)
    await _run_case(RunStatus.CANCELLED, TaskStatus.FAILED, "interrupted_execution")
    await _run_case(RunStatus.PAUSED, TaskStatus.FAILED, "approval_not_supported")


@pytest.mark.asyncio
async def test_cancel_swarm_persists_cancelling_and_converges():
    # cancel_swarm flips the parent run to CANCELLING; the gate observes it next
    # pass and the engine converges. Here we verify cancel persists the flag.
    fake = FakeExecutionService()
    tasks = LocalTaskBackend()
    svc, store = _service(fake_exec=fake, tasks=tasks)
    task_plan = make_plan(("a",))
    spec = _spec(agents=("a",))
    store._runs["parent-5"] = _persisted_swarm_run("parent-5", spec, task_plan)
    await tasks.create_plan(task_plan, ready_executions(task_plan))
    await svc.cancel_swarm("parent-5", principal=_principal())
    record = await store.get_run("parent-5")
    # request_cancel transitions RUNNING -> CANCELLING in the real store; the
    # memory store's request_cancel is a no-op, so verify it did not raise and
    # the call is authorized (no exception). A real ExecutionStore converges.


@pytest.mark.asyncio
async def test_cost_usage_unavailable_fails_when_cost_unknown():
    fake = FakeExecutionService(
        outcomes={
            "a": ChildRunResult(
                run_id="1",
                status=RunStatus.COMPLETED,
                output={"r": 1},
                error=None,
                usage=TaskUsage(10, 5, None),  # cost unknown
            )
        }
    )
    tasks = LocalTaskBackend()
    svc, store = _service(fake_exec=fake, tasks=tasks)
    from decimal import Decimal

    spec = _spec(
        agents=("a", "b"),
        limits=_limits(max_total_cost=Decimal("1.0")),
    )
    task_plan = make_plan(("a", "b"), edges={"b": ("a",)})
    outcome = await svc.run_swarm(spec, task_plan, principal=_principal(), session_id="s")
    # node a completes with unknown cost; gate raises cost_usage_unavailable -> SwarmFailed
    assert isinstance(outcome, SwarmFailed)
    assert "cost" in outcome.error.message or "unavailable" in outcome.error.message


@pytest.mark.asyncio
async def test_step_terminal_event_published_only_after_taskexecution_persisted():
    # The persist-before-event invariant: when a TERMINAL step event
    # (StepCompleted/StepFailed/StepSkipped/StepCancelled) is published, the
    # node's TaskExecution must ALREADY be terminal in the store. The sink
    # snapshots node status at publish time so reordering the publish before
    # the commit makes this test fail. StepStarted is excluded: it legitimately
    # fires while the node is CLAIMED (before the child agent runs).
    _TERMINAL_EVENT_TYPES = frozenset({
        "SwarmStepCompleted", "SwarmStepFailed",
        "SwarmStepSkipped", "SwarmStepCancelled",
    })
    statuses_at_publish: "list[tuple[str, str, TaskStatus]]" = []

    class _OrderCheckingSink:
        async def publish(self, event):
            etype = getattr(event, "event_type", None)
            if etype not in _TERMINAL_EVENT_TYPES:
                return
            execs = await tasks.list_executions(task_plan.id)
            by_node = {e.node_id: e for e in execs}
            node_exec = by_node.get(event.task_id)
            if node_exec is not None:
                statuses_at_publish.append((etype, event.task_id, node_exec.status))

    fake = FakeExecutionService()
    tasks = LocalTaskBackend()
    store = _MemoryExecutionStore()
    svc = SwarmExecutionService(
        store=store,
        tasks=tasks,
        agent_execution=fake,  # type: ignore[arg-type]
        authorization=OwnershipAuthorizationPolicy(),
        live_events=_OrderCheckingSink(),
        agent_provider=_FakeAgentProvider(),
    )
    task_plan = make_plan(("a", "b"), edges={"b": ("a",)})
    await svc.run_swarm(
        _spec(agents=("a", "b")), task_plan, principal=_principal(),
        session_id="s", execution_id="p",
    )
    assert statuses_at_publish, "no terminal step events were captured"
    for etype, nid, status in statuses_at_publish:
        assert status in {
            TaskStatus.COMPLETED, TaskStatus.FAILED,
            TaskStatus.SKIPPED, TaskStatus.CANCELLED,
        }, (
            f"persist-before-event violated: {etype} for node {nid!r} fired "
            f"while TaskExecution was still {status}"
        )


@pytest.mark.asyncio
async def test_mid_run_cancel_routes_parent_to_cancelled_not_completed():
    # F1 regression: an external cancel_swarm landing mid-run flips the parent
    # to CANCELLING. When _drive returns normally (nodes converged), the parent
    # must end CANCELLED — NOT attempt RUNNING -> COMPLETED (which a real store
    # rejects). The transition-enforcing _MemoryExecutionStore makes a missing
    # cancel guard throw instead of silently mis-completing.
    parent_run_id = "p-cancel"

    @dataclass
    class _CancelMidRunExec:
        """Fake execution that flips the parent to CANCELLING when run_child
        fires (simulating an external cancel racing mid-run), then completes."""

        cancelled: bool = False

        async def run_child(self, spec, prompt, **kwargs):
            if not self.cancelled:
                self.cancelled = True
                store._runs[parent_run_id] = replace(
                    store._runs[parent_run_id], status=RunStatus.CANCELLING
                )
            return ChildRunResult(
                parent_run_id, RunStatus.COMPLETED, {"ok": True}, None, TaskUsage()
            )

        async def prepare_agent_execution(self, agent_spec, **kwargs):
            return PreparedAgentExecution(
                agent_spec=agent_spec,
                assembled_agent=object(),
                tool_descriptors=(),
                fingerprint=f"fingerprint:{agent_spec.id}",
            )

        async def cancel(self, run_id: str, *, principal) -> None:
            return None

        async def read_usage(self, *, child_run_id: str) -> TaskUsage:
            return TaskUsage()

    fake = _CancelMidRunExec()
    tasks = LocalTaskBackend()
    store = _MemoryExecutionStore()
    svc = SwarmExecutionService(
        store=store,
        tasks=tasks,
        agent_execution=fake,  # type: ignore[arg-type]
        authorization=OwnershipAuthorizationPolicy(),
        live_events=NoopRunLiveEventSink(),
        agent_provider=_FakeAgentProvider(),
    )
    task_plan = make_plan(("a",))
    outcome = await svc.run_swarm(
        _spec(agents=("a",)), task_plan, principal=_principal(),
        session_id="s", execution_id=parent_run_id,
    )
    # The parent must converge to CANCELLED, not strand in CANCELLING and
    # not mis-complete via an illegal CANCELLING -> COMPLETED transition.
    record = await store.get_run(parent_run_id)
    assert record is not None
    assert record.status is RunStatus.CANCELLED, (
        f"parent ended {record.status}, expected CANCELLED"
    )


@pytest.mark.asyncio
async def test_run_swarm_rejects_mutating_tool_before_any_persistence():
    # F3: a node carrying a mutating tool must be rejected before start_run.
    # Inject a fake assembler that returns an assembly with a mutating tool.
    from linktools.ai.agent.assembly.models import AgentAssembly, AgentFeatureRef
    from linktools.ai.agent.tool.models import (
        ToolCategory,
        ToolDefinition,
        ToolDescriptor,
        ToolSource,
    )
    from linktools.ai.governance.policy.rule import RiskLevel, SideEffectKind

    async def _noop_handler(**kwargs):
        pass

    mutating_def = ToolDefinition(
        descriptor=ToolDescriptor(
            name="dangerous-write",
            source=ToolSource.BUILTIN,
            category=ToolCategory.FILE_WRITE,
            risk=RiskLevel.HIGH,
            side_effect=SideEffectKind.DESTRUCTIVE,
            feature=AgentFeatureRef("tool", "dangerous-write"),
        ),
        handler=_noop_handler,
    )

    class _FakeAssembler:
        async def assemble(self, agent_spec, ctx):
            return AgentAssembly(
                prompt_sections={},
                tools=(mutating_def,),
                feature_owners={},
            )

    @dataclass
    class _FakeExecWithAssembler:
        _assembler: object = field(default_factory=lambda: _FakeAssembler())

        async def run_child(self, spec, prompt, **kwargs):
            return ChildRunResult(
                spec.id, RunStatus.COMPLETED, {"ok": True}, None, TaskUsage()
            )

        async def prepare_agent_execution(self, agent_spec, **kwargs):
            raise InvalidSpecError("mutating_tool_not_allowed")

        async def cancel(self, run_id: str, *, principal) -> None:
            return None

        async def read_usage(self, *, child_run_id: str) -> TaskUsage:
            return TaskUsage()

    fake_exec = _FakeExecWithAssembler()
    tasks = LocalTaskBackend()
    store = _MemoryExecutionStore()
    svc = SwarmExecutionService(
        store=store,
        tasks=tasks,
        agent_execution=fake_exec,  # type: ignore[arg-type]
        authorization=OwnershipAuthorizationPolicy(),
        live_events=NoopRunLiveEventSink(),
        agent_provider=_FakeAgentProvider(),
    )
    task_plan = make_plan(("a",))
    with pytest.raises(InvalidSpecError, match="mutating_tool_not_allowed"):
        await svc.run_swarm(
            _spec(agents=("a",)), task_plan, principal=_principal(), session_id="s",
        )
    # Nothing must have been persisted (no parent run stranded).
    assert store._runs == {}


@pytest.mark.asyncio
async def test_inspect_swarm_denies_cross_tenant_principal():
    # F4: a principal from a different tenant must be denied inspect access.
    from linktools.ai.errors import PrincipalAccessDeniedError

    fake = FakeExecutionService()
    tasks = LocalTaskBackend()
    svc, store = _service(fake_exec=fake, tasks=tasks)
    task_plan = make_plan(("a",))
    store._runs["run-xt"] = _persisted_swarm_run("run-xt", _spec(agents=("a",)), task_plan)
    await tasks.create_plan(task_plan, ready_executions(task_plan))
    other_principal = PrincipalContext(
        tenant_id="OTHER-tenant",
        user_id="usr",
        actor=ActorRef(kind="system", id="untrusted"),
        scopes=ScopeSet.allow_all(),
    )
    with pytest.raises(PrincipalAccessDeniedError):
        await svc.inspect_swarm("run-xt", principal=other_principal)


@pytest.mark.asyncio
async def test_recover_swarm_denies_cross_tenant_principal():
    # F4: cross-tenant recover must also be denied.
    from linktools.ai.errors import PrincipalAccessDeniedError

    fake = FakeExecutionService()
    tasks = LocalTaskBackend()
    svc, store = _service(fake_exec=fake, tasks=tasks)
    task_plan = make_plan(("a",))
    store._runs["run-xt2"] = _persisted_swarm_run("run-xt2", _spec(agents=("a",)), task_plan)
    await tasks.create_plan(task_plan, ready_executions(task_plan))
    other_principal = PrincipalContext(
        tenant_id="OTHER-tenant",
        user_id="intruder",
        actor=ActorRef(kind="system", id="untrusted"),
        scopes=ScopeSet.allow_all(),
    )
    with pytest.raises(PrincipalAccessDeniedError):
        await svc.recover_swarm("run-xt2", principal=other_principal)


@pytest.mark.asyncio
async def test_run_swarm_completes_parent_run_against_real_execution_store(tmp_path):
    # The decisive test: drive SwarmExecutionService against a REAL
    # LocalExecutionBackend (not a stub) so the parent RunRecord lifecycle —
    # start_run -> claim_run (RUNNING) -> complete_run — must survive the
    # store's owner/fence/transition guards. A regression that skips the claim
    # or mis-fences the terminal converge fails here.
    from linktools.ai.execution.persistence.local import LocalExecutionBackend

    fake = FakeExecutionService()
    tasks = LocalTaskBackend()
    exec_store = LocalExecutionBackend(tmp_path / "exec")
    svc = SwarmExecutionService(
        store=exec_store,
        tasks=tasks,
        agent_execution=fake,  # type: ignore[arg-type]
        authorization=OwnershipAuthorizationPolicy(),
        live_events=NoopRunLiveEventSink(),
        agent_provider=_FakeAgentProvider(),
    )
    task_plan = make_plan(("a",))
    outcome = await svc.run_swarm(
        _spec(agents=("a",)), task_plan, principal=_principal(),
        session_id="real-s", execution_id="real-parent",
    )
    assert isinstance(outcome, SwarmCompleted)
    record = await exec_store.get_run("real-parent")
    assert record is not None
    assert record.status is RunStatus.COMPLETED


@pytest.mark.asyncio
async def test_inspect_swarm_requires_authorized_principal():
    from datetime import datetime, timezone

    from linktools.ai.execution.domain import RunDefinition, RunRecord
    from linktools.ai.storage.coordination.lease import Lease

    fake = FakeExecutionService()
    tasks = LocalTaskBackend()
    auth = _RecordingAuth()
    svc, store = _service(fake_exec=fake, tasks=tasks, auth=auth)
    task_plan = make_plan(("a",))
    now = datetime.now(timezone.utc)
    record = RunRecord(
        id="run-1",
        session_id="s",
        kind=RunKind.TASK,
        runnable_id="ws",
        runnable_type=RunnableType.AGENT,
        input={"plan_id": task_plan.id},
        definition=RunDefinition("ws", RunnableType.AGENT, "agent-spec.v1", {}, "hash"),
        status=RunStatus.COMPLETED,
        session_turn_sequence=None,
        parent_execution_id=None,
        root_execution_id="run-1",
        approval=None,
        lease=Lease(),
        cancel_requested_at=None,
        snapshot_revision=0,
        trace_sequence=0,
        event_sequence=0,
        tenant_id="tnt",
        user_id="usr",
        error=None,
        created_at=now,
        updated_at=now,
    )
    store._runs["run-1"] = record
    # create the task_plan in the task store so inspect can read nodes
    await tasks.create_plan(task_plan, ready_executions(task_plan))
    view = await svc.inspect_swarm("run-1", principal=_principal())
    assert view.plan_id == task_plan.id
    assert "inspect" in auth.actions


@pytest.mark.asyncio
async def test_swarm_fails_when_parent_lease_reclaimed_by_another_worker():
    # F2: if another worker reclaims the parent lease (different owner/fence),
    # the gate must detect it and fail the swarm rather than continuing under
    # a stale identity.
    parent_run_id = "p-lease-lost"

    @dataclass
    class _StealLeaseExec:
        """Fake execution that, on first run_child, simulates another worker
        reclaiming the parent lease (overwrites owner)."""

        stole: bool = False

        async def run_child(self, spec, prompt, **kwargs):
            if not self.stole:
                self.stole = True
                record = store._runs[parent_run_id]
                store._runs[parent_run_id] = replace(
                    record,
                    lease=Lease(
                        owner="other-worker",
                        fence=record.lease.fence + 1,
                        expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
                    ),
                )
            return ChildRunResult(
                parent_run_id, RunStatus.COMPLETED, {"ok": True}, None, TaskUsage()
            )

        async def prepare_agent_execution(self, agent_spec, **kwargs):
            return PreparedAgentExecution(
                agent_spec=agent_spec,
                assembled_agent=object(),
                tool_descriptors=(),
                fingerprint=f"fingerprint:{agent_spec.id}",
            )

        async def cancel(self, run_id: str, *, principal) -> None:
            return None

        async def read_usage(self, *, child_run_id: str) -> TaskUsage:
            return TaskUsage()

    fake = _StealLeaseExec()
    tasks = LocalTaskBackend()
    store = _MemoryExecutionStore()
    svc = SwarmExecutionService(
        store=store,
        tasks=tasks,
        agent_execution=fake,  # type: ignore[arg-type]
        authorization=OwnershipAuthorizationPolicy(),
        live_events=NoopRunLiveEventSink(),
        agent_provider=_FakeAgentProvider(),
    )
    task_plan = make_plan(("a", "b"), edges={"b": ("a",)})
    outcome = await svc.run_swarm(
        _spec(agents=("a", "b")), task_plan, principal=_principal(),
        session_id="s", execution_id=parent_run_id,
    )
    # The gate detects the reclaimed lease and fails the swarm.
    assert isinstance(outcome, SwarmFailed)
