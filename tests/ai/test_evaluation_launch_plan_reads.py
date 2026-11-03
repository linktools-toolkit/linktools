#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Coordinator launch planning uses retained facts without execution hydration."""

import asyncio
from pathlib import Path

import pytest

from linktools.ai.core import AuthorizationAction
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.evaluation import RescoreRequest

from .test_evaluation_consumers import EVALUATION_COMPLETION_TIMEOUT_SECONDS, PRINCIPAL, rule_scorer
from .test_evaluation_evidence_read_reuse import completed


@pytest.mark.asyncio
@pytest.mark.parametrize("rescore", [False, True])
async def test_settled_launch_plan_needs_no_current_execution_read(
    tmp_path: Path, rescore: bool,
) -> None:
    async with completed(tmp_path, count=1) as (evaluations, storage, run, scorer, policy):
        if rescore:
            run = await run.rescore(RescoreRequest((rule_scorer(scorer),), "rescore"))
            assert (await run.wait(timeout_seconds=EVALUATION_COMPLETION_TIMEOUT_SECONDS)).result.completion == "complete"
        await asyncio.gather(*evaluations._watchers.values())
        before = await storage.evaluation.records.get(run.experiment_id, tenant_id=PRINCIPAL.tenant_id)
        policy.reject = lambda action, resource: action is AuthorizationAction.EXECUTION_READ

        # Run one real coordinator pass against retained facts. It has no new
        # target/scorer work and must not load current execution projections.
        await evaluations._tick(run.experiment_id, evaluations._engine_owner, PRINCIPAL)
        after = await storage.evaluation.records.get(run.experiment_id, tenant_id=PRINCIPAL.tenant_id)
        assert after == before
        assert after.evidence
        assert after.scores

        # Public current-state inspection still enforces execution authorization.
        with pytest.raises(AIError) as caught:
            await run.inspect()
        assert caught.value.code is ErrorCode.AUTHORIZATION_DENIED
