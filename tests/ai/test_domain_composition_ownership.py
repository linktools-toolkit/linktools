#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Exercise lifecycle ownership at the public domain composition boundaries."""

from pathlib import Path

import pytest

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.evaluation import (
    CandidateSpec, CaseRef, CaseSpec, DatasetRef, DatasetSpec, EvaluationSpec,
    StartEvaluationRequest,
)
from linktools.ai.evaluation.engine import Evaluations
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.runtime import Runtime
from linktools.ai.task import Task
from linktools.ai.task.engine import TaskEngine

from .test_evaluation_consumers import PRINCIPAL, echo, exact, rule_scorer


@pytest.mark.asyncio
@pytest.mark.parametrize("compose", [False, True])
async def test_evaluation_borrowers_keep_engine_and_shared_storage_open(
    tmp_path: Path, compose: bool,
) -> None:
    storage = PersistenceStorage.sqlite(tmp_path / "domains.sqlite")
    async with Persistence.open("domains", storage=storage) as persistence:
        async with TaskEngine.open(persistence=persistence) as peer:
            async with Runtime.open(persistence=persistence) if compose else TaskEngine.open(
                persistence=persistence,
            ) as scope:
                engine = await scope.tasks.bind() if compose else scope
                async with Evaluations.open(engine) as first:
                    if compose:
                        second = await scope.evaluations.bind(engine)
                        await second.close()
                    else:
                        async with Evaluations.open(engine):
                            pass
                    with pytest.raises(AIError) as task_error:
                        await engine.close()
                    assert task_error.value.code is ErrorCode.RUNTIME_DEPENDENCY_NOT_READY
                    with pytest.raises(AIError) as storage_error:
                        await persistence.close()
                    assert storage_error.value.code is ErrorCode.RUNTIME_DEPENDENCY_NOT_READY
                    # Failed early close does not poison the engine for a new borrower.
                    async with Evaluations.open(engine):
                        assert persistence.ready
                    await first.close()
                await engine.close()
                assert peer.persistence.ready
            assert peer.persistence.ready
        assert persistence.ready
    assert not persistence.ready


@pytest.mark.asyncio
async def test_standalone_evaluation_runs_and_reopens_without_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    def reject_runtime(*args: object, **kwargs: object) -> None:
        raise AssertionError("standalone evaluation must not construct Runtime")

    monkeypatch.setattr(Runtime, "__init__", reject_runtime)
    target = Task("domain.echo", echo, effect_policy="none")
    scorer = Task("domain.exact", exact, effect_policy="none")
    storage = PersistenceStorage.sqlite(tmp_path / "evaluation.sqlite")
    options = dict(namespace="standalone-evaluation", storage=storage,
                   tenant_id=PRINCIPAL.tenant_id, definitions=(target, scorer))
    async with Evaluations.open(**options) as evaluations:
        dataset = await evaluations.publish_dataset(
            DatasetSpec(DatasetRef("standalone", 1), cases=(
                CaseSpec.task(CaseRef("standalone", "case", 1),
                              input={"answer": "yes"}, expected="yes"),
            )),
            principal=PRINCIPAL, idempotency_key="publish",
        )
        request = StartEvaluationRequest(
            EvaluationSpec(dataset, (CandidateSpec("target", task=target.ref),),
                           (rule_scorer(scorer),)), PRINCIPAL, "start",
        )
        run = await evaluations.start(request)
        view = (await run.wait(timeout_seconds=10)).result
        assert view.completion == "complete"
        report = await run.create_report()
        assert report.scores[0].mean == 1.0

    async with Evaluations.open(**options) as reopened:
        assert await reopened.get_report(report.report_id, principal=PRINCIPAL) == report


@pytest.mark.asyncio
async def test_runtime_close_retries_after_external_evaluation_releases_engine() -> None:
    async with Runtime.open("domains", storage=PersistenceStorage.in_memory()) as runtime:
        engine = await runtime.tasks.bind()
        async with Evaluations.open(engine):
            with pytest.raises(AIError) as caught:
                await runtime.close()
            assert caught.value.code is ErrorCode.RUNTIME_DEPENDENCY_NOT_READY
            assert engine.persistence.ready
        await runtime.close()
        assert not engine.persistence.ready
