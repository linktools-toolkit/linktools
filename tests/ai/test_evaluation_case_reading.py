#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Each coordinator pass reads distinct immutable case identities afresh."""

from types import SimpleNamespace

import pytest

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.evaluation import CaseRef, TargetTrialRef
from linktools.ai.runtime._evaluation import RuntimeEvaluations


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", (None, "unavailable", "cancelled"))
async def test_coordinator_reads_distinct_case_versions_afresh(
    monkeypatch: pytest.MonkeyPatch, failure: str | None,
) -> None:
    import asyncio

    owner = object.__new__(RuntimeEvaluations)
    refs = (CaseRef("dataset", "case", 1), CaseRef("dataset", "case", 2),
            CaseRef("other", "case", 1))
    trials = tuple(SimpleNamespace(case_ref=ref, trial=TargetTrialRef("run", str(index)))
                   for index, ref in enumerate((refs[0], refs[0], refs[1], refs[2], refs[1])))
    record = SimpleNamespace(gate="open", intents=(), dispositions=(),
        manifest=SimpleNamespace(kind="score_only", candidates=(), scorers=()))
    reads: list[CaseRef] = []
    second_pass = False

    async def read_record(*args, **kwargs):
        return record

    async def read_trials(*args, **kwargs):
        return trials, {}

    async def read_case(ref):
        reads.append(ref)
        if second_pass and ref == refs[1]:
            if failure == "unavailable":
                raise AIError(ErrorCode.EVALUATION_EVIDENCE_UNAVAILABLE)
            if failure == "cancelled":
                raise asyncio.CancelledError
        return object()

    async def no_budget(*args, **kwargs):
        return False

    async def read_cases(references):
        return tuple([await read_case(ref) for ref in references])

    owner._state = SimpleNamespace(get_cases=read_cases)
    monkeypatch.setattr(owner, "_record", read_record)
    monkeypatch.setattr(owner, "_require_content", no_budget)
    monkeypatch.setattr(owner, "_budget_exhausted", no_budget)
    monkeypatch.setattr(owner, "_trials", read_trials)
    await owner._tick("run", object(), object())
    assert reads == list(refs)
    reads.clear()
    second_pass = True
    if failure is None:
        await owner._tick("run", object(), object())
        assert reads == list(refs)
    else:
        with pytest.raises(AIError if failure == "unavailable" else asyncio.CancelledError) as caught:
            await owner._tick("run", object(), object())
        if failure == "unavailable":
            assert caught.value.code is ErrorCode.EVALUATION_EVIDENCE_UNAVAILABLE
        assert reads == list(refs[:2])
