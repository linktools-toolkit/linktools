#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Typed evaluation numbers survive integral JSON transport representations."""

import json
from dataclasses import replace
from pathlib import Path

import pytest
import sqlalchemy.ext.asyncio

from linktools.ai.core import canonical_json_bytes
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.evaluation import (
    CandidateSpec, CaseRef, CaseSpec, DatasetRef, DatasetSpec, EvaluationSpec,
    StartEvaluationRequest,
)
from linktools.ai.persistence import PersistenceStorage
from linktools.ai.runtime import Runtime
from linktools.ai.task import Task

from .test_evaluation_consumers import PRINCIPAL, echo, exact, rule_scorer


def _transport_json(raw: str) -> object:
    # Emulate a JSON SQL/DAL transport emitting integral DOUBLEs without '.0'.
    # Untyped JSON cannot recover the lost float/int distinction from this wire.
    def number(raw_number: str) -> int | float:
        value = float(raw_number)
        return int(value) if value.is_integer() else value

    return json.loads(raw, parse_float=number)


@pytest.mark.asyncio
@pytest.mark.parametrize("normalize", (False, True))
@pytest.mark.parametrize("weight", (1.0, 1.5))
async def test_runtime_evaluation_round_trips_integral_json_numbers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, normalize: bool, weight: float,
) -> None:
    if normalize:
        create_engine = sqlalchemy.ext.asyncio.create_async_engine

        def transport_engine(*args: object, **kwargs: object):
            return create_engine(*args, **kwargs, json_deserializer=_transport_json)

        monkeypatch.setattr(sqlalchemy.ext.asyncio, "create_async_engine", transport_engine)

    target = Task("numeric.echo", echo, effect_policy="none")
    scorer = Task("numeric.exact", exact, effect_policy="none")
    spec = DatasetSpec(DatasetRef("numeric", 1), (
        CaseSpec.task(CaseRef("numeric", "one", 1),
                      input={"answer": "yes", "integer": 1, "boolean": True,
                             "whole": 1.0, "negative_zero": -0.0, "large": 1e20},
                      expected="yes", weight=weight),
    ))
    expected_case = None
    manifest_digest = None
    report = None
    experiment_id = None
    for reopened in (False, True):
        async with Runtime.open("numeric-transport", tenant_id=PRINCIPAL.tenant_id,
                                storage=PersistenceStorage.sqlite(tmp_path / "state.db")) as runtime:
            engine = await runtime.tasks.bind(target, scorer)
            evaluations = await runtime.evaluations.bind(engine)
            dataset = await evaluations.publish_dataset(spec, principal=PRINCIPAL, idempotency_key="publish")
            case = await evaluations.get_case(spec.cases[0].ref, principal=PRINCIPAL)
            assert case.weight == weight and type(case.weight) is float
            assert type(case.input.input["integer"]) is int
            assert case.input.input["boolean"] is True
            assert canonical_json_bytes(dict(case.input.input)) == canonical_json_bytes(dict(spec.cases[0].input.input))
            assert (await evaluations.list_cases(dataset, principal=PRINCIPAL)).items == (case,)
            if reopened:
                assert case == expected_case and case.digest == expected_case.digest
                assert await evaluations.get_report(report.report_id, principal=PRINCIPAL) == report
            else:
                expected_case = case
            with pytest.raises(AIError) as conflict:
                await evaluations.publish_dataset(replace(spec, cases=(replace(spec.cases[0], weight=2.0),)),
                                                  principal=PRINCIPAL, idempotency_key="publish")
            assert conflict.value.code is ErrorCode.IDEMPOTENCY_CONFLICT
            request = StartEvaluationRequest(EvaluationSpec(dataset,
                (CandidateSpec("target", task=target.ref),), (rule_scorer(scorer),)), PRINCIPAL, "start")
            run = await evaluations.start(request)
            assert (await evaluations.start(request)).experiment_id == run.experiment_id
            view = (await run.wait(timeout_seconds=30)).result
            assert view.completion == "complete" and view.progress.valid_scores == 1
            manifest = (await evaluations._state.get(run.experiment_id, tenant_id=PRINCIPAL.tenant_id)).manifest
            if reopened:
                assert run.experiment_id == experiment_id
                assert manifest.digest == manifest_digest
            else:
                experiment_id, manifest_digest = run.experiment_id, manifest.digest
                report = await run.create_report()
            assert report.scores[0].mean == report.scores[0].coverage == 1.0
            assert await evaluations.get_report(report.report_id, principal=PRINCIPAL) == report
