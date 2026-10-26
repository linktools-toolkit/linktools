#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Polling evaluation reports does not publish durable report artifacts."""

from linktools.ai.persistence import Persistence
from linktools.ai.task_engine import TaskEngine
from linktools.ai.evaluation_engine import Evaluations

from pathlib import Path

import pytest

from linktools.ai.core import AuthorizationAction, Principal, ResourceRef, TenantAuthorizationPolicy
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.evaluation import (
    CandidateSpec, CaseRef, CaseSpec, DatasetRef, DatasetSpec, EvaluationSpec,
    StartEvaluationRequest,
)

from linktools.ai.persistence import PersistenceDomain
from linktools.ai.persistence.state import RecordQuery
from linktools.ai.task import Task

from .test_evaluation_consumers import (
    CONTEXT, EVALUATION_COMPLETION_TIMEOUT_SECONDS, PRINCIPAL, echo, exact, rule_scorer,
)
from .test_evaluation_persistence import storage_for


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("filesystem", "sqlite"))
async def test_repeated_report_previews_preserve_state_and_objects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend: str,
) -> None:
    target = Task("preview.target", echo, effect_policy="none")
    scorer = Task("preview.exact", exact, effect_policy="none")
    storage = storage_for(backend, tmp_path)
    async with (
        Persistence.open('report-preview', storage=storage, tenant_id='evaluation-consumers') as persistence_35,
        TaskEngine.open(target, scorer, persistence=persistence_35, context=CONTEXT) as engine,
        Evaluations.open(engine) as evaluations,
    ):
        dataset = await evaluations.publish_dataset(DatasetSpec(DatasetRef("preview", 1), (
            CaseSpec.task(CaseRef("preview", "one", 1), input={"answer": "yes"}, expected="yes"),
        )), principal=PRINCIPAL, idempotency_key="dataset")
        run = await evaluations.start(StartEvaluationRequest(EvaluationSpec(
            dataset, (CandidateSpec("current", task=target.ref),), (rule_scorer(scorer),),
        ), PRINCIPAL, "evaluate"))
        assert (await run.wait(timeout_seconds=EVALUATION_COMPLETION_TIMEOUT_SECONDS)).result.completion == "complete"
        await evaluations.close()
        records = persistence_35.evaluation.records
        before_record = await records.get(run.experiment_id, tenant_id=PRINCIPAL.tenant_id)
        before_rows = await records.state_store.read(lambda transaction: transaction.list_records(RecordQuery(kind="evaluation_report")))
        before_objects = {
            (item.key, item.digest, item.size)
            async for item in persistence_35.object_store(PersistenceDomain.EVALUATION).list_objects()
        }
        previews = [await run.preview_report() for _ in range(3)]
        assert await records.get(run.experiment_id, tenant_id=PRINCIPAL.tenant_id) == before_record
        assert await records.state_store.read(
            lambda transaction: transaction.list_records(RecordQuery(kind="evaluation_report"))
        ) == before_rows
        assert {
            (item.key, item.digest, item.size)
            async for item in persistence_35.object_store(PersistenceDomain.EVALUATION).list_objects()
        } == before_objects
        for preview in previews:
            assert preview.completion == "complete"
            assert preview.scores[0].mean == 1.0
            with pytest.raises(AIError) as missing:
                await evaluations.get_report(preview.report_id, principal=PRINCIPAL)
            assert missing.value.code is ErrorCode.STORAGE_NOT_FOUND
        saved = await run.create_report()
        assert await evaluations.get_report(saved.report_id, principal=PRINCIPAL) == saved
        assert all(
            preview.cutoff == saved.cutoff and preview.scores == saved.scores
            and preview.trials == saved.trials and preview.score_attempts == saved.score_attempts
            for preview in previews
        )

        async def deny_read(
            self: TenantAuthorizationPolicy, principal: Principal,
            action: AuthorizationAction, resource: ResourceRef,
        ) -> None:
            raise AIError(ErrorCode.AUTHORIZATION_DENIED)

        with monkeypatch.context() as patch:
            patch.setattr(TenantAuthorizationPolicy, "authorize", deny_read)
            for report in (run.preview_report, run.create_report):
                with pytest.raises(AIError) as denied:
                    await report()
                assert denied.value.code is ErrorCode.AUTHORIZATION_DENIED
