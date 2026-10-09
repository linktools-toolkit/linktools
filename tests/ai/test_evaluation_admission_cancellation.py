#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Cancellation owns the same native admission receipt as a concurrent start."""

import asyncio
from dataclasses import replace
from pathlib import Path

import pytest

from linktools.ai.core import (
    AuthorizationAction, JsonValue, Principal, ResourceRef, TenantAuthorizationPolicy,
    idempotency_key_digest, service_principal,
)
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.evaluation import (
    CandidateSpec, CaseRef, CaseSpec, DatasetRef, DatasetSpec, EvaluationSpec,
    StartEvaluationRequest,
)
from linktools.ai.model import ModelRegistry
from linktools.ai.runtime import Runtime
from linktools.ai.runtime.state._evaluation_records import EvaluationRecord
from linktools.ai.task import Task, TaskNodeContext

from .test_evaluation_consumers import (
    CONTEXT, EVALUATION_COMPLETION_TIMEOUT_SECONDS, PRINCIPAL, echo, exact, rule_scorer,
)
from .test_evaluation_persistence import storage_for


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("filesystem", "sqlite"))
@pytest.mark.parametrize("winner", ("cancel", "start"))
async def test_admission_cancellation_fences_both_start_orderings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend: str, winner: str,
) -> None:
    target_started = asyncio.Event()
    release_target = asyncio.Event()
    calls: list[str] = []

    async def target(context: TaskNodeContext[None]) -> JsonValue:
        calls.append("target")
        target_started.set()
        await release_target.wait()
        return context.input["answer"]

    async def score(context: TaskNodeContext[None]) -> JsonValue:
        calls.append("score")
        return await exact(context)

    target_task = Task("cancel-admission.target", target, effect_policy="none")
    scorer = Task("cancel-admission.score", score, effect_policy="none")
    storage = storage_for(backend, tmp_path)
    async with Runtime.open(
        "cancel-admission", models=ModelRegistry(), storage=storage, context=CONTEXT,
    ) as runtime:
        dataset = await runtime.evaluations.publish_dataset(DatasetSpec(DatasetRef("cancel", 1), (
            CaseSpec.task(CaseRef("cancel", "one", 1), input={"answer": "yes"}, expected="yes"),
        )), principal=PRINCIPAL, idempotency_key="dataset")
        request = StartEvaluationRequest(EvaluationSpec(
            dataset, (CandidateSpec("current", task=target_task.ref),), (rule_scorer(scorer),),
        ), PRINCIPAL, "same-admission")
        engine = runtime.tasks.bind(target_task, scorer)
        entered, release = asyncio.Event(), asyncio.Event()
        reserve = storage.evaluation.records.reserve_experiment
        delayed_gate = "open" if winner == "cancel" else "closed_cancel"

        async def paused(record: EvaluationRecord) -> EvaluationRecord:
            if record.gate == delayed_gate:
                entered.set()
                await release.wait()
            return await reserve(record)

        with monkeypatch.context() as patch:
            patch.setattr(storage.evaluation.records, "reserve_experiment", paused)
            delayed = asyncio.create_task(
                runtime.evaluations.start(request, engine=engine) if winner == "cancel" else
                runtime.evaluations.cancel_admission(request, engine=engine)
            )
            try:
                await asyncio.wait_for(entered.wait(), 10)
                if winner == "cancel":
                    first = await runtime.evaluations.cancel_admission(request, engine=engine)
                    assert (await first.inspect()).completion == "cancelled"
                else:
                    first = await runtime.evaluations.start(request, engine=engine)
                    await asyncio.wait_for(target_started.wait(), 10)
                release.set()
                second = await asyncio.wait_for(delayed, EVALUATION_COMPLETION_TIMEOUT_SECONDS)
            finally:
                release.set()
                if not delayed.done():
                    delayed.cancel()
                await asyncio.gather(delayed, return_exceptions=True)
        assert first.experiment_id == second.experiment_id
        assert (await second.wait(timeout_seconds=EVALUATION_COMPLETION_TIMEOUT_SECONDS)).result.completion == "cancelled"
        record = await storage.evaluation.records.get(first.experiment_id, tenant_id=PRINCIPAL.tenant_id)
        assert record is not None and record.gate == "closed_cancel"
        assert record.manifest.principal == PRINCIPAL
        assert calls == ([] if winner == "cancel" else ["target"])
        assert not any(item.scorer_slot_id is not None for item in record.intents)

        release_target.set()
        repeated = await runtime.evaluations.start(request, engine=engine)
        assert repeated.experiment_id == first.experiment_id
        assert (await repeated.wait(timeout_seconds=EVALUATION_COMPLETION_TIMEOUT_SECONDS)).result.completion == "cancelled"
        assert calls == ([] if winner == "cancel" else ["target"])
        repeated_cancel = await runtime.evaluations.cancel_admission(request, engine=engine)
        assert repeated_cancel.experiment_id == first.experiment_id
        assert (await repeated_cancel.inspect()).completion == "cancelled"


@pytest.mark.asyncio
async def test_admission_cancellation_preserves_authorization_and_request_identity(tmp_path: Path) -> None:
    class Authorization(TenantAuthorizationPolicy):
        denied: AuthorizationAction | None = None

        async def authorize(
            self, principal: Principal, action: AuthorizationAction, resource: ResourceRef,
        ) -> None:
            if action is self.denied:
                raise AIError(ErrorCode.AUTHORIZATION_DENIED)
            await super().authorize(principal, action, resource)

    authorization = Authorization(PRINCIPAL.tenant_id)
    target = Task("cancel-security.target", echo, effect_policy="none")
    scorer = Task("cancel-security.score", exact, effect_policy="none")
    storage = storage_for("filesystem", tmp_path)
    async with Runtime.open(
        "cancel-security", models=ModelRegistry(), storage=storage, context=CONTEXT,
        authorization=authorization,
    ) as runtime:
        dataset = await runtime.evaluations.publish_dataset(DatasetSpec(DatasetRef("cancel", 1), (
            CaseSpec.task(CaseRef("cancel", "one", 1), input={"answer": "yes"}, expected="yes"),
        )), principal=PRINCIPAL, idempotency_key="dataset")
        request = StartEvaluationRequest(EvaluationSpec(
            dataset, (CandidateSpec("current", task=target.ref),), (rule_scorer(scorer),),
        ), PRINCIPAL, "cancelled-admission")
        engine = runtime.tasks.bind(target, scorer)
        for action in (AuthorizationAction.EVALUATION_RUN, AuthorizationAction.EVALUATION_CANCEL):
            authorization.denied = action
            with pytest.raises(AIError) as denied:
                await runtime.evaluations.cancel_admission(request, engine=engine)
            assert denied.value.code is ErrorCode.AUTHORIZATION_DENIED
            assert await storage.evaluation.idempotency.get(
                "evaluation.run", idempotency_key_digest(request.idempotency_key), tenant_id=PRINCIPAL.tenant_id,
            ) is None
        authorization.denied = None
        run = await runtime.evaluations.cancel_admission(request, engine=engine)
        for principal in (
            service_principal(PRINCIPAL.tenant_id, "other-owner"),
            service_principal("foreign", PRINCIPAL.principal_id),
        ):
            with pytest.raises(AIError) as denied:
                await runtime.evaluations.cancel_admission(replace(request, principal=principal), engine=engine)
            assert denied.value.code is ErrorCode.AUTHORIZATION_DENIED
        changed = replace(request, spec=replace(request.spec, repetitions=2))
        for operation in (runtime.evaluations.start, runtime.evaluations.cancel_admission):
            with pytest.raises(AIError) as conflict:
                await operation(changed, engine=engine)
            assert conflict.value.code is ErrorCode.IDEMPOTENCY_CONFLICT
        assert (await run.inspect()).completion == "cancelled"

        completed_request = replace(request, idempotency_key="completed-admission")
        completed = await runtime.evaluations.start(completed_request, engine=engine)
        assert (await completed.wait(timeout_seconds=EVALUATION_COMPLETION_TIMEOUT_SECONDS)).result.completion == "complete"
        cancelled = await runtime.evaluations.cancel_admission(completed_request, engine=engine)
        assert cancelled.experiment_id == completed.experiment_id
        assert (await cancelled.inspect()).completion == "complete"
