#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Score dimensions remain business data across durable snapshot boundaries."""

from linktools.ai.persistence import Persistence
from linktools.ai.snapshot import Snapshot, SnapshotLimits
from linktools.ai.task_engine import TaskEngine
from linktools.ai.evaluation_engine import Evaluations

from pathlib import Path
from .test_evaluation_retention import Offline

import pytest


from linktools.ai.core import JsonValue
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.evaluation import (
    CandidateSpec, CaseRef, CaseSpec, DatasetRef, DatasetSpec, DimensionContract,
    EvaluationSpec, ScoreBundle, ScoreNotApplicable, ScorerSpec, StartEvaluationRequest,
)
from linktools.ai.persistence import PersistenceDomain, PersistenceStorage
from linktools.ai.persistence.codec import decode_domain, encode_domain, iter_runtime_object_refs
from linktools.ai.storage import InMemoryObjectStore
from linktools.ai.task import Task, TaskNodeContext

from .test_evaluation_consumers import (
    CONTEXT, EVALUATION_COMPLETION_TIMEOUT_SECONDS, PRINCIPAL, echo,
)


def test_score_bundle_codec_preserves_business_fields_without_object_dependencies() -> None:
    score = ScoreBundle(
        dimensions={"kind": 1.0, "digest": 1.0, "size": 1.0,
                    "unavailable": ScoreNotApplicable(reason="no evidence")},
        rationale="The supplied evidence supports these scores",
        evidence_ids=("first", "second"),
        diagnostics=("reviewed",),
    )
    wire = encode_domain(score)

    assert decode_domain(wire, ScoreBundle) == score
    assert tuple(iter_runtime_object_refs(wire, default_domain=PersistenceDomain.EVALUATION)) == ()


def test_score_bundle_codec_rejects_malformed_mapping() -> None:
    wire = encode_domain(ScoreBundle(dimensions={"quality": 1.0}))
    dimensions = dict(wire["$mapping"])["dimensions"]
    dimensions["$mapping"] = "invalid"
    with pytest.raises(AIError) as raised:
        decode_domain(wire, ScoreBundle)
    assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


async def reference_named_dimensions(context: TaskNodeContext[None]) -> JsonValue:
    return ScoreBundle(dimensions={"kind": 1.0, "digest": 1.0, "size": 1.0}).to_mapping()


@pytest.mark.asyncio
async def test_evaluation_score_dimensions_survive_snapshot_export_and_restore(tmp_path: Path) -> None:
    root = tmp_path / "source"
    target = Task("score-snapshot.target", echo, effect_policy="none")
    scorer = Task("score-snapshot.score", reference_named_dimensions, effect_policy="none")
    expected = ScoreBundle(dimensions={"kind": 1.0, "digest": 1.0, "size": 1.0})
    async with (
        Persistence.open('score-snapshot', storage=PersistenceStorage.filesystem(root), tenant_id='evaluation-consumers') as persistence_62,
        TaskEngine.open(target, scorer, persistence=persistence_62, context=CONTEXT) as engine,
        Evaluations.open(engine) as evaluations,
    ):
        dataset = await evaluations.publish_dataset(
            DatasetSpec(DatasetRef("score-snapshot", 1), cases=(
                CaseSpec.task(CaseRef("score-snapshot", "one", 1), input={"answer": "yes"}),
            )), principal=PRINCIPAL, idempotency_key="publish-score-snapshot")
        declared = ScorerSpec("score", scorer.ref, dimensions=tuple(
            DimensionContract(name, "number", "higher") for name in expected.dimensions))
        run = await evaluations.start(
            StartEvaluationRequest(EvaluationSpec(dataset,
                (CandidateSpec("target", task=target.ref),), (declared,)),
                PRINCIPAL, "start-score-snapshot"))
        view = (await run.wait(timeout_seconds=EVALUATION_COMPLETION_TIMEOUT_SECONDS)).result
        assert view.completion == "complete"
        assert view.progress.valid_scores == 1
        assert (await run.scores()).items[0].score == expected
        report = await run.create_report()
        assert {item.dimension for item in report.scores} == set(expected.dimensions)
        experiment_id = run.experiment_id

    storage = Persistence(PersistenceStorage.filesystem(root))
    await storage.initialize(namespace="score-snapshot", tenant_id=PRINCIPAL.tenant_id, read_only=True)
    archive = InMemoryObjectStore("score-snapshot-archive")
    limits = SnapshotLimits(max_entries=4096, max_bytes=16 * 1024 * 1024)
    try:
        reference = await Snapshot.create(persistence=storage, exclusive=Offline(), object_store=archive, limits=limits)
    finally:
        await storage.close()

    restored_root = tmp_path / "restored"
    restored_snapshot = await Snapshot.restore(reference, object_store=archive, target=restored_root, namespace="score-snapshot", tenant_id=PRINCIPAL.tenant_id, limits=limits)
    async with (
        Persistence.open('score-snapshot', storage=PersistenceStorage.from_root(restored_snapshot.storage_root), tenant_id='evaluation-consumers') as persistence_93,
        TaskEngine.open(persistence=persistence_93, context=CONTEXT) as engine,
        Evaluations.open(engine) as restored_evaluations,
    ):
        run = await restored_evaluations.get(experiment_id, principal=PRINCIPAL)
        assert (await run.inspect()).completion == "complete"
        scores = (await run.scores()).items
        assert len(scores) == 1
        assert scores[0].score == expected
        assert (await run.create_report()).scores == report.scores
