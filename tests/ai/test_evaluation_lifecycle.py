#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Closed evaluation coordinators reject new work without changing durable facts."""

from linktools.ai.persistence import Persistence
from linktools.ai.task_engine import TaskEngine
from linktools.ai.evaluation_engine import Evaluations

import asyncio
from dataclasses import replace
from pathlib import Path

import pytest

from linktools.ai.core import TaskStatus, idempotency_key_digest
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.evaluation import (
    CandidateSpec, CaseRef, CaseSpec, DatasetRef, DatasetSpec, EvaluationSpec,
    HumanScoreRequest, RescoreRequest, ScoreAttemptView, ScoreBundle, ScorerSpec,
    StartEvaluationRequest,
)
from linktools.ai.persistence import PersistenceStorage
from linktools.ai.persistence.state import RecordQuery
from linktools.ai.task import Task, TaskRef

from .test_evaluation_consumers import CONTEXT, DIMENSION, PRINCIPAL, echo


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ("start", "rescore", "cancel", "reconcile", "human_score"))
async def test_closed_evaluations_reject_control_without_persisting(
    tmp_path: Path, operation: str,
) -> None:
    target = Task("lifecycle.target", echo, effect_policy="none")
    scorer = ScorerSpec("human", TaskRef.deferred_input(), (DIMENSION,))
    storage = PersistenceStorage.filesystem(tmp_path)
    async with (
        Persistence.open('evaluation-lifecycle', storage=storage, tenant_id='evaluation-consumers') as persistence_34,
        TaskEngine.open(target, persistence=persistence_34, context=CONTEXT) as engine,
        Evaluations.open(engine) as evaluations,
    ):
        dataset = await evaluations.publish_dataset(DatasetSpec(DatasetRef("lifecycle", 1), (
            CaseSpec.task(CaseRef("lifecycle", "one", 1), input={"answer": "yes"}),
        )), principal=PRINCIPAL, idempotency_key="dataset")
        request = StartEvaluationRequest(EvaluationSpec(dataset,
            (CandidateSpec("candidate", task=target.ref),), (scorer,)), PRINCIPAL, "start")
        run = await evaluations.start(request)

        async def waiting_score() -> ScoreAttemptView:
            while True:
                pending = (await run.scores()).items[0]
                if pending.scorer_execution is not None:
                    graph = await engine.get(pending.scorer_graph.graph_id, principal=PRINCIPAL)
                    state = await graph.state()
                    node = next(item for item in state.node_states if item.node_id == pending.scorer_node_id)
                    if node.status is TaskStatus.WAITING:
                        return pending
                await asyncio.sleep(0.01)

        pending = await asyncio.wait_for(waiting_score(), 10)
        await evaluations.close()
        repository = persistence_34.evaluation.records
        before = await repository.state_store.read(
            lambda transaction: transaction.list_records(RecordQuery(kind="evaluation")))
        assert len(before) == 1
        key = "closed-request"
        with pytest.raises(AIError) as raised:
            if operation == "start":
                await evaluations.start(replace(request, idempotency_key=key))
            elif operation == "rescore":
                await run.rescore(RescoreRequest((scorer,), key))
            elif operation == "cancel":
                await run.cancel(idempotency_key=key)
            elif operation == "reconcile":
                await evaluations.reconcile(run.experiment_id,
                                                    principal=PRINCIPAL, idempotency_key=key)
            else:
                await run.submit_human_score(HumanScoreRequest(pending.trial.trial_id,
                    pending.scorer_slot_id, pending.evidence_ref,
                    ScoreBundle(dimensions={"exact_match": 1.0}), key))
        assert raised.value.code is ErrorCode.RUNTIME_DEPENDENCY_NOT_READY
        assert not raised.value.retryable
        assert await repository.state_store.read(
            lambda transaction: transaction.list_records(RecordQuery(kind="evaluation"))) == before
        assert await persistence_34.evaluation.idempotency.get("evaluation.run", idempotency_key_digest(key),
                                                       tenant_id=PRINCIPAL.tenant_id) is None
        reopened = await evaluations.get(run.experiment_id, principal=PRINCIPAL)
        assert (await reopened.inspect()).completion == "running"
        assert (await reopened.scores()).items == (pending,)
        assert (await reopened.trials()).items[0].evidence_ref is not None
        assert (await evaluations.get_dataset(dataset, principal=PRINCIPAL)).ref == dataset
