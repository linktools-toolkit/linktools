#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Terminal evaluation reconciliation keeps facts while avoiding repeated settlement."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from linktools.ai.core import TaskStatus
from linktools.ai.evaluation import ScoreAttemptView, TargetTrialRef
from linktools.ai.runtime._evaluation import RuntimeEvaluations
from linktools.ai.runtime.state._evaluation_records import EvaluationLaunchIntent
from linktools.ai.task import TaskGraph, TaskGraphAdmission, TaskGraphRequest, TaskGraphState, TaskGraphSubmission, TaskNode, TaskNodeView, TaskRef
from .test_evaluation_persistence import experiment


@pytest.mark.asyncio
@pytest.mark.parametrize("snapshot_released,current_released", ((False, False), (True, True), (False, True)))
async def test_terminal_reconciliation_preserves_score_collection_after_release(
    monkeypatch: pytest.MonkeyPatch, snapshot_released: bool, current_released: bool,
) -> None:
    owner = object.__new__(RuntimeEvaluations)
    record = experiment()
    principal = record.manifest.principal
    trial = TargetTrialRef(record.experiment_id, "trial")
    graph = TaskGraph("scorer", (TaskNode("score", task=TaskRef("score", 1)),))
    admission = TaskGraphAdmission.from_request(TaskGraphRequest(graph, principal, "start"))
    launch = EvaluationLaunchIntent("score:trial:score", trial, "score",
        TaskGraphSubmission("evaluation", admission, graph), None, confirmed=True, released=snapshot_released)
    snapshot = replace(record, gate="closed_budget", intents=(launch,))
    current = replace(snapshot, intents=(replace(launch, released=current_released),))
    state = TaskGraphState(graph.graph_id, TaskStatus.CANCELLED, graph.nodes,
        (TaskNodeView(graph.graph_id, "score", (), TaskStatus.CANCELLED, None, 0, None, None, None, None),))
    calls: list[str] = []
    reads = 0

    async def read(experiment_id, actor, *, allow_expired):
        nonlocal reads
        assert experiment_id == record.experiment_id and actor == principal and allow_expired
        reads += 1
        return snapshot if reads == 1 else current

    async def graph_state(intent, actor):
        assert intent == launch and actor == principal
        calls.append("read_graph")
        return state

    async def collect(value, intent, graph_state):
        nonlocal current
        assert graph_state is state
        calls.append("collect_score")
        result = ScoreAttemptView(record.experiment_id, None, trial, "score", TaskRef("score", 1),
                                  "not_attempted", reason="cancelled")
        current = replace(current, scores=(result,))

    async def close(key):
        assert key == (record.experiment_id, launch.slot_id)
        calls.append("close_scope")

    async def settle(experiment_id, slot_id, *, confirmed, released):
        nonlocal current
        assert experiment_id == record.experiment_id and slot_id == launch.slot_id
        assert confirmed and released
        calls.append("settle")
        current = replace(current, intents=(replace(current.intents[0], confirmed=True, released=True),))
        return current

    async def noop(*args, **kwargs):
        return None

    monkeypatch.setattr(owner, "_record", read)
    monkeypatch.setattr(owner, "_require_content", noop)
    monkeypatch.setattr(owner, "_graph_state", graph_state)
    monkeypatch.setattr(owner, "_resume_decision", noop)
    monkeypatch.setattr(owner, "_collect_score", collect)
    monkeypatch.setattr(owner, "_close_trial_scope", close)
    monkeypatch.setattr(owner, "_disposition", noop)
    owner._state = SimpleNamespace(settle_intent=settle)
    await owner._tick(record.experiment_id, None, principal)
    assert calls == ["read_graph", "collect_score", "close_scope"] + ([] if snapshot_released else ["settle"])
    assert current.intents[0].released
    assert current.scores[0].status == "not_attempted" and current.scores[0].reason == "cancelled"


@pytest.mark.asyncio
async def test_repeated_settlement_keeps_release_revision_and_timestamp() -> None:
    from linktools.ai.runtime import RuntimeStorage
    from .test_evaluation_persistence import intent

    storage = RuntimeStorage.in_memory()
    await storage.initialize(namespace="evaluation", tenant_id="tenant")
    try:
        repository = storage.evaluation.records
        record = experiment()
        launch = intent()
        await repository.reserve_experiment(record)
        await repository.register_launch_intent(record.experiment_id, launch, capacity=1)
        settled = await repository.settle_intent(record.experiment_id, launch.slot_id, confirmed=True, released=True)
        repeated = await repository.settle_intent(record.experiment_id, launch.slot_id, confirmed=False, released=False)
        assert repeated == settled
        assert repeated.intents[0].confirmed and repeated.intents[0].released
        assert repeated.revision == settled.revision and repeated.updated_at == settled.updated_at
    finally:
        await storage.close()
