#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Public evaluation composition borrows task, persistence, and asset owners."""

from pathlib import Path
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import pytest

from linktools.ai.asset import AssetKey, AssetStore, InMemoryAssetBackend
from linktools.ai.core import JsonValue, service_principal
from linktools.ai.evaluation import (
    CandidateSpec, CaseRef, CaseSpec, DatasetRef, DatasetSpec, DimensionContract,
    EvaluationPolicy, EvaluationSpec, ScoreBundle, ScorerSpec, ScoringInput,
    StartEvaluationRequest,
)
from linktools.ai.evaluation_engine import Evaluations, EvaluationTrialScope
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.storage import StorageOverlay
from linktools.ai.task import Task, TaskGraph, TaskNode, TaskNodeContext
from linktools.ai.task_engine import TaskEngine


async def echo(context: TaskNodeContext[None]) -> JsonValue:
    return context.input["answer"]


async def exact(context: TaskNodeContext[None]) -> JsonValue:
    sample = ScoringInput.from_mapping(context.input)
    return ScoreBundle(dimensions={"exact": float(sample.target_output == sample.expected)}).to_mapping()


@pytest.mark.asyncio
@pytest.mark.parametrize("backend,isolated", [("memory", False), ("filesystem", False), ("sqlite", False),
                                                ("filesystem", True), ("sqlite", True)])
async def test_evaluation_borrows_engine_and_persistence_for_complete_task_report(
    tmp_path: Path, backend: str, isolated: bool,
) -> None:
    storage = (PersistenceStorage.in_memory() if backend == "memory" else
               PersistenceStorage.filesystem(tmp_path / "state") if backend == "filesystem" else
               PersistenceStorage.sqlite(tmp_path / "state.sqlite"))
    principal = service_principal("tenant", "evaluator")
    target = Task("owner.echo", echo, effect_policy="none")
    judge = Task("owner.exact", exact, effect_policy="none")
    scorer = ScorerSpec("exact", judge.ref, (DimensionContract("exact", "boolean", "higher", 0, 1),))
    async with Persistence.open("evaluation-owner", storage=storage, tenant_id="tenant") as persistence:
        opened = []
        closed = []

        @asynccontextmanager
        async def trial_scope(scope: EvaluationTrialScope) -> AsyncIterator[TaskEngine]:
            opened.append(scope.slot_id)
            async with TaskEngine.open(target, judge, persistence=persistence) as selected:
                yield selected
            closed.append(scope.slot_id)

        async with TaskEngine.open(target, judge, persistence=persistence) as engine:
            async with Evaluations.open(engine) as evaluations:
                dataset = await evaluations.publish_dataset(
                    DatasetSpec(DatasetRef("cases", 1), cases=(
                        CaseSpec.task(CaseRef("cases", "one", 1), input={"answer": "yes"}, expected="yes"),
                    )), principal=principal, idempotency_key="dataset",
                )
                run = await evaluations.start(StartEvaluationRequest(
                    EvaluationSpec(dataset, (CandidateSpec("echo", task=target.ref),), (scorer,),
                                   policy=EvaluationPolicy(allow_volatile=backend == "memory")),
                    principal, "evaluation",
                ), trial_scope=trial_scope if isolated else None)
                assert (await run.wait(timeout_seconds=30)).result.completion == "complete"
                report = await run.create_report()
                assert report.scores[0].mean == 1.0
                experiment_id = run.experiment_id
            assert persistence.ready
            assert sorted(opened) == sorted(closed)
            assert bool(opened) == isolated
            graph = await engine.start(
                TaskGraph("after-evaluation", (TaskNode("echo", task=target.ref, input={"answer": "still open"}),)),
                principal=principal, idempotency_key="after",
            )
            await graph.wait(timeout_seconds=30)
            assert await graph.result("echo") == "still open"
            async with Evaluations.open(engine) as reopened:
                historical = await reopened.get(experiment_id, principal=principal)
                assert (await historical.preview_report()).scores[0].mean == 1.0
        assert persistence.ready
    assert not persistence.ready


@pytest.mark.asyncio
async def test_evaluation_reads_pinned_assets_without_owning_the_asset_store(tmp_path: Path) -> None:
    backend = InMemoryAssetBackend()
    assets = AssetStore(StorageOverlay(backend, writer=backend))
    await assets.initialize()
    try:
        key = AssetKey("fixture", "expected")
        await assets.put(key, b'"yes"')
        expected, = await assets.resolve_versions((key,))
        await assets.put(key, b'"new version"')
        principal = service_principal("tenant", "evaluator")
        target = Task("asset.echo", echo, effect_policy="none")
        judge = Task("asset.exact", exact, effect_policy="none")
        scorer = ScorerSpec("exact", judge.ref, (DimensionContract("exact", "boolean", "higher", 0, 1),))
        async with Persistence.open("asset-evaluation", storage=PersistenceStorage.filesystem(tmp_path), tenant_id="tenant") as persistence:
            async with TaskEngine.open(target, judge, persistence=persistence) as engine:
                async with Evaluations.open(engine, assets=(assets,)) as evaluations:
                    dataset = await evaluations.publish_dataset(
                        DatasetSpec(DatasetRef("assets", 1), cases=(
                            CaseSpec.task(CaseRef("assets", "one", 1), input={"answer": "yes"}, expected=expected),
                        )), principal=principal, idempotency_key="dataset",
                    )
                    run = await evaluations.start(StartEvaluationRequest(
                        EvaluationSpec(dataset, (CandidateSpec("echo", task=target.ref),), (scorer,)),
                        principal, "evaluation",
                    ))
                    assert (await run.wait(timeout_seconds=30)).result.completion == "complete"
                    assert (await run.create_report()).scores[0].mean == 1.0
                assert assets.ready
    finally:
        await assets.close()
