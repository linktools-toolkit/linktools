#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Session, Task, Trace, Schema, Local and Extension contracts. MT-40 MT-41 MT-42 MT-43 MT-44 MT-45 MT-46 MT-47 MT-48 MT-49 MT-50."""

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

pytest.importorskip("temporalio")

from linktools.ai.domain.schema import SchemaEntry
from linktools.ai.domain.session import Session, SessionStatus
from linktools.ai.domain.task import Job, TaskExecution, TaskNode, TaskPlan, TaskStatus
from linktools.ai.domain.trace import RunSnapshot, StopReason, TraceEvent, TraceKind
from linktools.ai.entrypoints.api import build_api
from linktools.ai.entrypoints.service import build_service
from linktools.ai.application.services.event import EventService
from linktools.ai.extension.registry import FeatureRegistry
from linktools.ai.foundation.errors import ErrorCode, LinktoolsAIError
from linktools.ai.foundation.json import canonical_json_bytes
from linktools.ai.local.index import SkillIndex
from linktools.ai.local.project import LocalProject
from linktools.ai.schema.registry import OutputSchemaRegistry
from linktools.ai.trace.api import InMemoryTraceRecorder
from linktools.ai.trace.snapshot import snapshot_digest, verify_snapshot
from linktools.ai.agent.deps import AgentDeps
from linktools.ai.bundles.generated import agent
from pydantic_ai.durable_exec.temporal import TemporalDurability


def test_session_and_task_fences_are_strict() -> None:
    session = Session(session_id="s", owner_id="o", project_id="p", agent_id="a", agent_revision=1, profile="local-coding")
    assert session.transition_to(SessionStatus.BUSY).status is SessionStatus.BUSY
    plan = TaskPlan(plan_id="p", tasks=(TaskNode(task_id="a"), TaskNode(task_id="b", dependencies=("a",))))
    assert plan.ready(frozenset()) == ("a",)
    execution = TaskExecution(task_id="a").claim("worker", datetime.now(timezone.utc), timedelta(minutes=1))
    with pytest.raises(LinktoolsAIError) as error:
        execution.complete("other", execution.fence, datetime.now(timezone.utc))
    assert error.value.code == ErrorCode.TASK_FENCE_STALE
    done = execution.complete("worker", execution.fence, datetime.now(timezone.utc), "result")
    assert done.status is TaskStatus.COMPLETED
    assert done.complete("worker", execution.fence, datetime.now(timezone.utc), "other") == done
    assert Job(job_id="j", plan=plan, executions=(done,)).aggregate() == "RUNNING"


def test_schema_drift_and_feature_freeze_are_fail_closed() -> None:
    registry = OutputSchemaRegistry()
    entry = SchemaEntry(schema_id="output", revision=1, fingerprint="a", python_type_path="builtins.dict", json_schema={"type": "object"})
    assert registry.register(entry) == entry
    with pytest.raises(LinktoolsAIError) as error:
        registry.register(entry.model_copy(update={"fingerprint": "b"}))
    assert error.value.code == ErrorCode.OUTPUT_SCHEMA_DRIFT
    features = FeatureRegistry()
    features.register("local", object())
    features.freeze()
    with pytest.raises(LinktoolsAIError) as error:
        features.register("other", object())
    assert error.value.code == ErrorCode.FEATURE_REGISTRY_FROZEN


@pytest.mark.asyncio
async def test_trace_is_monotonic_idempotent_and_snapshot_digest_is_verified() -> None:
    recorder = InMemoryTraceRecorder()
    event = TraceEvent(
        execution_id="e", sequence=1, run_id="r", kind=TraceKind.TERMINAL,
        timestamp=datetime.now(timezone.utc), status="SUCCEEDED",
    )
    assert await recorder.append(event) == event
    assert await recorder.append(event) == event
    values = {
        "snapshot_id": "s", "execution_id": "e", "run_id": "r", "input_digest": "i",
        "release_digest": "rel", "bundle_digest": "b", "model_plan_digest": "m",
        "prompt_digest": "p", "trace_start": 1, "trace_end": 1, "result_digest": None,
        "checkpoint_ref": None, "usage": {}, "stop_reason": StopReason.END_TURN.value,
    }
    snapshot = RunSnapshot(**values, digest=snapshot_digest(values))
    assert verify_snapshot(snapshot)
    assert snapshot.verify()


def test_local_project_and_private_agent_index_refresh_incrementally(tmp_path: Path) -> None:
    root = tmp_path / "project"
    skill = root / ".linktools" / "skills" / "demo"
    agents = skill / "agents"
    agents.mkdir(parents=True)
    (root / ".linktools" / "config.yaml").write_text("default_agent: builtin\n", encoding="utf-8")
    (skill / "SKILL.md").write_text("# Demo\n", encoding="utf-8")
    private = agents / "reviewer.md"
    private.write_text("review", encoding="utf-8")
    LocalProject.discover(root / "src")
    index = SkillIndex(skill.parent)
    first = index.refresh()
    assert index.resolve("demo").private_agents[0].agent_id == "reviewer"
    private.write_text("review changed", encoding="utf-8")
    assert index.refresh() > first
    assert index.resolve_agent("demo", "reviewer").content == "review changed"


def test_trace_snapshot_digest_is_canonical() -> None:
    first = snapshot_digest({"b": 1, "a": 2})
    second = snapshot_digest({"a": 2, "b": 1})
    assert first == second
    assert canonical_json_bytes({"a": 2, "b": 1}) == canonical_json_bytes({"b": 1, "a": 2})


def test_agent_deps_and_composition_roots_are_explicit() -> None:
    deps = AgentDeps(
        execution_id="e",
        tenant_principal_ref="tenant:subject",
        model_plan_id="model",
        budget_id="budget",
        prompt_snapshot_id="prompt",
    )
    assert deps.model_dump()["execution_id"] == "e"
    assert agent.name == "lt.generated.empty"
    assert TemporalDurability.from_agent(agent) is not None

    api = build_api(object(), object())
    assert api.routes == ()
    service = build_service(object(), ("workflow",), ("activity",))

    class Worker:
        def __init__(self) -> None:
            self.workflows = ()
            self.activities = ()

        def register_workflows(self, workflows: tuple[object, ...]) -> None:
            self.workflows = workflows

        def register_activities(self, activities: tuple[object, ...]) -> None:
            self.activities = activities

    worker = Worker()
    service.register(worker)
    assert worker.workflows == ("workflow",)
    assert worker.activities == ("activity",)


@pytest.mark.asyncio
async def test_durable_event_identity_is_execution_scoped() -> None:
    class Repository:
        def __init__(self) -> None:
            self.events = []

        async def append(self, event: object) -> object:
            self.events.append(event)
            return event

    repository = Repository()
    service = EventService(repository)
    timestamp = datetime.now(timezone.utc)
    first = await service.append("execution-a", "status", 1, timestamp, {"state": "RUNNING"}, source_id="workflow", source_phase="start")
    retry = await service.append("execution-a", "status", 1, timestamp, {"state": "RUNNING"}, source_id="workflow", source_phase="start")
    other = await service.append("execution-b", "status", 1, timestamp, {"state": "RUNNING"}, source_id="workflow", source_phase="start")
    assert first.event_id == retry.event_id
    assert first.event_id != other.event_id
