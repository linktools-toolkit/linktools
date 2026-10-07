#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Score dimensions remain business data across durable snapshot boundaries."""

from pathlib import Path

import pytest

from linktools.ai.core import JsonValue
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.evaluation import (
    CandidateSpec, CaseRef, CaseSpec, DatasetRef, DatasetSpec, DimensionContract,
    EvaluationSpec, ScoreBundle, ScoreNotApplicable, ScorerSpec, StartEvaluationRequest,
)
from linktools.ai.runtime import Runtime
from linktools.ai.runtime.state import RuntimeDomain, RuntimeStorage, SnapshotLimits
from linktools.ai.runtime.state._codec import decode_domain, encode_domain, iter_runtime_object_refs
from linktools.ai.storage import InMemoryObjectStore
from linktools.ai.task import Task, TaskNodeContext

from .test_evaluation_consumers import (
    CONTEXT, EVALUATION_COMPLETION_TIMEOUT_SECONDS, PRINCIPAL, FixtureModels, echo,
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
    assert tuple(iter_runtime_object_refs(wire, default_domain=RuntimeDomain.EVALUATION)) == ()


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
    async with Runtime.open("score-snapshot", models=FixtureModels(),
                            storage=RuntimeStorage.filesystem(root), context=CONTEXT) as runtime:
        dataset = await runtime.evaluations.publish_dataset(
            DatasetSpec(DatasetRef("score-snapshot", 1), cases=(
                CaseSpec.task(CaseRef("score-snapshot", "one", 1), input={"answer": "yes"}),
            )), principal=PRINCIPAL, idempotency_key="publish-score-snapshot")
        declared = ScorerSpec("score", scorer.ref, dimensions=tuple(
            DimensionContract(name, "number", "higher") for name in expected.dimensions))
        run = await runtime.evaluations.start(
            StartEvaluationRequest(EvaluationSpec(dataset,
                (CandidateSpec("target", task=target.ref),), (declared,)),
                PRINCIPAL, "start-score-snapshot"), engine=runtime.tasks.bind(target, scorer))
        view = (await run.wait(timeout_seconds=EVALUATION_COMPLETION_TIMEOUT_SECONDS)).result
        assert view.completion == "complete"
        assert view.progress.valid_scores == 1
        assert (await run.scores()).items[0].score == expected
        report = await run.create_report()
        assert {item.dimension for item in report.scores} == set(expected.dimensions)
        experiment_id = run.experiment_id

    storage = RuntimeStorage.filesystem(root)
    await storage.initialize(namespace="score-snapshot", tenant_id=PRINCIPAL.tenant_id, read_only=True)
    archive = InMemoryObjectStore("score-snapshot-archive")
    limits = SnapshotLimits(max_entries=4096, max_bytes=16 * 1024 * 1024)
    try:
        reference = await storage.export_snapshot(object_store=archive, limits=limits)
    finally:
        await storage.close()

    restored_root = tmp_path / "restored"
    await RuntimeStorage.restore_snapshot(reference, object_store=archive, root=restored_root, limits=limits)
    async with Runtime.open("score-snapshot", models=FixtureModels(),
                            storage=RuntimeStorage.from_root(restored_root), context=CONTEXT) as restored:
        run = await restored.evaluations.get(experiment_id, principal=PRINCIPAL)
        assert (await run.inspect()).completion == "complete"
        scores = (await run.scores()).items
        assert len(scores) == 1
        assert scores[0].score == expected
        assert (await run.create_report()).scores == report.scores
