#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Closed evaluation coordinators reject new work without changing durable facts."""

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
from linktools.ai.runtime import Runtime, RuntimeStorage
from linktools.ai.runtime.state._store import RecordQuery
from linktools.ai.task import Task, TaskRef

from .test_evaluation_consumers import CONTEXT, DIMENSION, PRINCIPAL, FixtureModels, echo


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ("start", "rescore", "cancel", "reconcile", "human_score"))
async def test_closed_evaluations_reject_control_without_persisting(
    tmp_path: Path, operation: str,
) -> None:
    target = Task("lifecycle.target", echo, effect_policy="none")
    scorer = ScorerSpec("human", TaskRef.deferred_input(), (DIMENSION,))
    storage = RuntimeStorage.filesystem(tmp_path)
    async with Runtime.open("evaluation-lifecycle", models=FixtureModels(), storage=storage,
                            context=CONTEXT) as runtime:
        engine = runtime.tasks.bind(target)
        dataset = await runtime.evaluations.publish_dataset(DatasetSpec(DatasetRef("lifecycle", 1), (
            CaseSpec.task(CaseRef("lifecycle", "one", 1), input={"answer": "yes"}),
        )), principal=PRINCIPAL, idempotency_key="dataset")
        request = StartEvaluationRequest(EvaluationSpec(dataset,
            (CandidateSpec("candidate", task=target.ref),), (scorer,)), PRINCIPAL, "start")
        run = await runtime.evaluations.start(request, engine=engine)

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
        await runtime.evaluations.close()
        repository = storage.evaluation.records
        before = await repository.state_store.read(
            lambda transaction: transaction.list_records(RecordQuery(kind="evaluation")))
        assert len(before) == 1
        key = "closed-request"
        with pytest.raises(AIError) as raised:
            if operation == "start":
                await runtime.evaluations.start(replace(request, idempotency_key=key), engine=engine)
            elif operation == "rescore":
                await run.rescore(RescoreRequest((scorer,), key), engine=engine)
            elif operation == "cancel":
                await run.cancel(idempotency_key=key)
            elif operation == "reconcile":
                await runtime.evaluations.reconcile(run.experiment_id, engine=engine,
                                                    principal=PRINCIPAL, idempotency_key=key)
            else:
                await run.submit_human_score(HumanScoreRequest(pending.trial.trial_id,
                    pending.scorer_slot_id, pending.evidence_ref,
                    ScoreBundle(dimensions={"exact_match": 1.0}), key))
        assert raised.value.code is ErrorCode.RUNTIME_DEPENDENCY_NOT_READY
        assert not raised.value.retryable
        assert await repository.state_store.read(
            lambda transaction: transaction.list_records(RecordQuery(kind="evaluation"))) == before
        assert await storage.evaluation.idempotency.get("evaluation.run", idempotency_key_digest(key),
                                                       tenant_id=PRINCIPAL.tenant_id) is None
        reopened = await runtime.evaluations.get(run.experiment_id, principal=PRINCIPAL)
        assert (await reopened.inspect()).completion == "running"
        assert (await reopened.scores()).items == (pending,)
        assert (await reopened.trials()).items[0].evidence_ref is not None
        assert (await runtime.evaluations.get_dataset(dataset, principal=PRINCIPAL)).ref == dataset
