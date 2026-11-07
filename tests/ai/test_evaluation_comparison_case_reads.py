#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Comparison-local case batches preserve both datasets and fresh reads."""

import asyncio
from collections import Counter
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pytest
from sqlalchemy import event

from linktools.ai.core import AuthorizationAction, canonical_json_bytes
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.evaluation import (
    CandidateSlotRef, CandidateSpec, CaseRef, CaseSpec, ComparisonReadCutoff,
    ComparisonSpec, DatasetRef, DatasetSpec, EvaluationPolicy, EvaluationSpec,
    ScoreComparisonSelection, ScoreSelection, StartEvaluationRequest,
)
from linktools.ai.evaluation.engine import Evaluations
from linktools.ai.persistence import Persistence
from linktools.ai.persistence._evaluation_records import EvaluationCaseRecord
from linktools.ai.persistence.state import RecordQuery
from linktools.ai.task import Task
from linktools.ai.task.engine import TaskEngine

from .test_evaluation_consumers import CONTEXT, PRINCIPAL, echo, exact, rule_scorer
from .test_evaluation_evidence_read_reuse import Policy
from .test_evaluation_persistence import storage_for


def _case(dataset: str, name: str, revision: int = 1) -> CaseSpec:
    return CaseSpec.task(CaseRef(dataset, name, revision), input={"answer": name}, expected=name)


async def _comparison_count(persistence: Persistence) -> int:
    return len(await persistence.evaluation.records.state_store.read(
        lambda tx: tx.list_records(RecordQuery(kind="evaluation_comparison"))))


@asynccontextmanager
async def _comparison(
    path: Path, backend: str, left: DatasetSpec, right: DatasetSpec,
) -> AsyncIterator[tuple[Evaluations, Persistence, ComparisonSpec, Policy]]:
    target = Task("comparison-reads.target", echo, effect_policy="none")
    scorer = Task("comparison-reads.score", exact, effect_policy="none")
    policy = Policy()
    async with (
        Persistence.open("comparison-reads", storage=storage_for(backend, path),
                         tenant_id=PRINCIPAL.tenant_id) as persistence,
        TaskEngine.open(target, scorer, persistence=persistence, context=CONTEXT, authorization=policy) as engine,
        Evaluations.open(engine) as evaluations,
    ):
        await evaluations.publish_dataset(left, principal=PRINCIPAL, idempotency_key="left")
        if right != left:
            await evaluations.publish_dataset(right, principal=PRINCIPAL, idempotency_key="right")
        snapshots = []
        for name, dataset in (("left", left), ("right", right)):
            request = StartEvaluationRequest(EvaluationSpec(dataset.ref,
                (CandidateSpec("candidate", task=target.ref),), (rule_scorer(scorer),),
                policy=EvaluationPolicy(allow_volatile=True)), PRINCIPAL, name)
            record = await evaluations._reserve_start(request, engine=engine)
            snapshots.append((await evaluations._snapshot(record.experiment_id, PRINCIPAL))[1])
        selection = ScoreSelection("exact", "exact_match")
        spec = ComparisonSpec(
            CandidateSlotRef(snapshots[0].experiment_id, "candidate"),
            CandidateSlotRef(snapshots[1].experiment_id, "candidate"),
            (ScoreComparisonSelection(selection, selection),),
            cutoff=ComparisonReadCutoff(snapshots[0].cutoff, snapshots[1].cutoff),
        )
        yield evaluations, persistence, spec, policy


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "sqlite"))
@pytest.mark.parametrize("overlap", ("same", "reordered", "partial", "other_dataset"))
async def test_comparison_reads_each_case_identity_once_and_preserves_each_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend: str, overlap: str,
) -> None:
    left = DatasetSpec(DatasetRef("dataset", 1), tuple(_case("dataset", name) for name in ("c", "a", "b")))
    if overlap == "same":
        right = left
    elif overlap == "reordered":
        right = DatasetSpec(DatasetRef("dataset", 2), tuple(case.ref for case in reversed(left.cases)))
    elif overlap == "partial":
        right = DatasetSpec(DatasetRef("dataset", 2), (left.cases[1].ref, _case("dataset", "c", 2), _case("dataset", "d")))
    else:
        right = DatasetSpec(DatasetRef("other", 1), tuple(_case("other", name) for name in ("b", "a", "c")))
    left_refs = tuple(case.ref for case in left.cases)
    right_refs = tuple(case if isinstance(case, CaseRef) else case.ref for case in right.cases)
    expected_refs = tuple(dict.fromkeys((*left_refs, *right_refs)))
    async with _comparison(tmp_path, backend, left, right) as (evaluations, persistence, spec, _policy):
        repository = persistence.evaluation.records
        get_cases = repository.get_cases
        expected_left = [case.to_mapping() for case in await get_cases(left_refs)]
        expected_right = [case.to_mapping() for case in await get_cases(right_refs)]
        reads = []

        async def read(refs):
            reads.extend(refs)
            return await get_cases(refs)

        monkeypatch.setattr(repository, "get_cases", read)
        for _ in range(2):
            reads.clear()
            report = await evaluations.create_comparison_report(spec, principal=PRINCIPAL)
            assert report.cutoff == spec.cutoff
            assert set(reads) == set(expected_refs)
            assert len(reads) == len(expected_refs)
            assert tuple(pair.case_ref for pair in report.pairs) == expected_refs
            differences = [item for item in report.differences if item.path == "cases"]
            if expected_left == expected_right:
                assert differences == []
            else:
                difference, = differences
                assert difference.baseline == expected_left
                assert difference.candidate == expected_right


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "sqlite"))
@pytest.mark.parametrize("failure", ("missing", "expired", "cancelled"))
async def test_comparison_rechecks_required_case_content_before_publishing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend: str, failure: str,
) -> None:
    left = DatasetSpec(DatasetRef("dataset", 1), (_case("dataset", "a"),))
    right = DatasetSpec(DatasetRef("dataset", 2), (left.cases[0].ref, _case("dataset", "b")))
    async with _comparison(tmp_path, backend, left, right) as (evaluations, persistence, spec, _policy):
        before = await _comparison_count(persistence)
        await evaluations.create_comparison_report(spec, principal=PRINCIPAL)
        repository = persistence.evaluation.records
        count = await _comparison_count(persistence)
        assert count == before + 1
        ref = right.cases[-1].ref
        key = repository._key("evaluation_case", [ref.dataset_id, ref.case_id, ref.revision])
        if failure == "missing":
            await repository.state_store.mutate(lambda tx: tx.delete_records((key,)))
        elif failure == "expired":
            case = await repository.get_case(ref)
            now = datetime.now(timezone.utc)
            value = EvaluationCaseRecord(case, PRINCIPAL.principal_id, now, now)
            row = replace(repository._stored("evaluation_case", [ref.dataset_id, ref.case_id, ref.revision], value),
                          storage_revision=1)
            assert await repository.state_store.mutate(lambda tx: tx.replace_record(row, expected_storage_revision=0))
        else:
            async def cancelled(refs):
                raise asyncio.CancelledError
            monkeypatch.setattr(repository, "get_cases", cancelled)
        with pytest.raises(asyncio.CancelledError if failure == "cancelled" else AIError) as caught:
            await evaluations.create_comparison_report(spec, principal=PRINCIPAL)
        if failure != "cancelled":
            assert caught.value.code is (ErrorCode.STORAGE_NOT_FOUND if failure == "missing"
                                         else ErrorCode.EVALUATION_EVIDENCE_UNAVAILABLE)
        assert await _comparison_count(persistence) == count


@pytest.mark.asyncio
async def test_comparison_keeps_candidate_dataset_authorization(tmp_path: Path) -> None:
    left = DatasetSpec(DatasetRef("left", 1), (_case("left", "a"),))
    right = DatasetSpec(DatasetRef("right", 1), (_case("right", "a"),))
    async with _comparison(tmp_path, "memory", left, right) as (evaluations, persistence, spec, policy):
        count = await _comparison_count(persistence)
        policy.reject = lambda action, resource: (
            action is AuthorizationAction.EVALUATION_DATASET_READ and resource.id == right.ref.id)
        with pytest.raises(AIError) as caught:
            await evaluations.create_comparison_report(spec, principal=PRINCIPAL)
        assert caught.value.code is ErrorCode.AUTHORIZATION_DENIED
        assert await _comparison_count(persistence) == count


@pytest.mark.asyncio
@pytest.mark.parametrize("frozen", (False, True))
async def test_comparison_case_io_measurements(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, record_property, frozen: bool,
) -> None:
    dataset = DatasetSpec(DatasetRef("dataset", 1), tuple(_case("dataset", str(index)) for index in range(100)))
    async with _comparison(tmp_path, "sqlite", dataset, dataset) as (evaluations, persistence, spec, _policy):
        if not frozen:
            spec = replace(spec, cutoff=None)
        repository = persistence.evaluation.records
        engine = repository.state_store.context.engine.sync_engine
        counts = Counter()
        get_cases = repository.get_cases
        keys = {repository._key("evaluation_case", [case.ref.dataset_id, case.ref.case_id, case.ref.revision]).hex()
                for case in dataset.cases}

        async def read(refs):
            cases = await get_cases(refs)
            counts["case_records"] += len(cases)
            counts["case_json_bytes"] += sum(len(canonical_json_bytes(case.to_mapping())) for case in cases)
            return cases

        def capture(_connection, _cursor, sql, parameters, _context, _many):
            operation = sql.lstrip().split(None, 1)[0].upper()
            counts[operation] += 1
            if operation == "SELECT" and any(isinstance(value, str) and value in keys for value in parameters):
                counts["case_selects"] += 1

        def commit(_connection):
            counts["commits"] += 1

        monkeypatch.setattr(repository, "get_cases", read)
        event.listen(engine, "before_cursor_execute", capture)
        event.listen(engine, "commit", commit)
        try:
            report = await evaluations.create_comparison_report(spec, principal=PRINCIPAL)
        finally:
            event.remove(engine, "before_cursor_execute", capture)
            event.remove(engine, "commit", commit)
        record_property("comparison_case_io", dict(counts))
        assert len(report.pairs) == 100
        assert counts["case_selects"] <= (2 if frozen else 6)
        assert counts["case_records"] <= (100 if frozen else 300)
