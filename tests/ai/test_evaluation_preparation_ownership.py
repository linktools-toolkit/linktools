#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluation owns target preparation before private native payloads exist."""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import timedelta
from pathlib import Path

import pytest

from linktools.ai.core import service_principal
from linktools.ai.evaluation import (
    CandidateSpec, CaseRef, CaseSpec, DatasetRef, DatasetSpec, DimensionContract,
    EvaluationPolicy, EvaluationSpec, ScorerSpec, StartEvaluationRequest,
)
from linktools.ai.model import ModelRegistry
from linktools.ai.runtime import Runtime, RuntimeContext, RuntimeStorage
from linktools.ai.task import Task, TaskGraphSubmission


class Offline:
    @asynccontextmanager
    async def offline_exclusivity(self) -> AsyncIterator[None]:
        yield


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "filesystem"))
@pytest.mark.parametrize("boundary", ("before", "after"))
async def test_cancelled_preparation_releases_payload_and_survives_reopen(
    backend: str, boundary: str, tmp_path: Path,
) -> None:
    calls = []

    async def target(ctx):
        calls.append(ctx.input)
        return "ok"

    async def score(ctx):
        return {"dimensions": {"quality": 1.0}}

    owner = service_principal("tenant", "owner")
    tasks = (Task("target", target, effect_policy="none"), Task("score", score, effect_policy="none"))
    storage = RuntimeStorage.in_memory() if backend == "memory" else RuntimeStorage.filesystem(tmp_path)
    async with Runtime.open("preparation", models=ModelRegistry(), storage=storage,
                            context=RuntimeContext(None, tenant_id="tenant")) as runtime:
        engine = runtime.tasks.bind(*tasks)
        entered, release = asyncio.Event(), asyncio.Event()
        prepare = storage.task.admissions.prepare
        saved = []

        async def pause(submission: TaskGraphSubmission) -> TaskGraphSubmission:
            saved.append(submission)
            if boundary == "after":
                submission = await prepare(submission)
            entered.set()
            await release.wait()
            return await prepare(submission) if boundary == "before" else submission

        storage.task.admissions.prepare = pause
        dataset = await runtime.evaluations.publish_dataset(DatasetSpec(DatasetRef("dataset", 1), cases=(
            CaseSpec.task(CaseRef("dataset", "case", 1), input={"secret": "private target input"}),
        )), principal=owner, idempotency_key="publish")
        run = await runtime.evaluations.start(StartEvaluationRequest(EvaluationSpec(dataset,
            (CandidateSpec("target", task=tasks[0].ref),),
            (ScorerSpec("score", tasks[1].ref, (DimensionContract("quality", "number", "higher"),)),),
            policy=EvaluationPolicy(allow_volatile=True, content_retention_seconds=60,
                                    metadata_retention_seconds=60)), owner, "start"), engine=engine)
        await asyncio.wait_for(entered.wait(), 5)
        try:
            assert (await run.cancel(idempotency_key="cancel")).completion == "cancelled"
        finally:
            release.set()
        await asyncio.sleep(.1)
        record = await storage.evaluation.records.get(run.experiment_id, tenant_id="tenant")
        result = await runtime.evaluations.purge_expired(principal=owner,
            now=record.metadata_expires_at + timedelta(seconds=1), exclusive=Offline())
        assert result.evaluations == (run.experiment_id,)
        rows = await storage.task.admissions.state_store.read(lambda tx: tx.scan_records())
        assert not [row for row in rows if row.kind == "task_submission_payload"]
        assert await storage.task.admissions.submission_status(saved[0].ref) == "cancelled"
        assert calls == []
        assert not (await engine.start_prepared(saved[0])).admitted
    if backend == "filesystem":
        reopened = RuntimeStorage.filesystem(tmp_path)
        async with Runtime.open("preparation", models=ModelRegistry(), storage=reopened,
                                context=RuntimeContext(None, tenant_id="tenant")):
            assert await reopened.task.admissions.submission_status(saved[0].ref) == "cancelled"


@pytest.mark.asyncio
async def test_purge_discovers_capture_written_before_offline_quiescence(tmp_path: Path) -> None:
    from linktools.ai.runtime._runtime_identity import task_graph_binding_capture_key
    from linktools.ai.runtime.state import RuntimeDomain, input_capture_key

    async def target(ctx):
        pytest.fail("cancelled target must not start")

    async def score(ctx):
        return {"dimensions": {"quality": 1.0}}

    owner = service_principal("tenant", "owner")
    tasks = (Task("target", target, effect_policy="none"), Task("score", score, effect_policy="none"))
    storage = RuntimeStorage.filesystem(tmp_path)
    async with Runtime.open("preparation", models=ModelRegistry(), storage=storage,
                            context=RuntimeContext(None, tenant_id="tenant")) as runtime:
        engine = runtime.tasks.bind(*tasks)
        entered, release = asyncio.Event(), asyncio.Event()
        task_runtime = runtime._require_task_node_runtime()
        capture = task_runtime.capture_admission

        async def pause(admission, graph):
            entered.set()
            await release.wait()
            return await capture(admission, graph)

        task_runtime.capture_admission = pause
        dataset = await runtime.evaluations.publish_dataset(DatasetSpec(DatasetRef("dataset", 1), cases=(
            CaseSpec.task(CaseRef("dataset", "case", 1), input={"secret": "late private capture"}),
        )), principal=owner, idempotency_key="publish")
        run = await runtime.evaluations.start(StartEvaluationRequest(EvaluationSpec(dataset,
            (CandidateSpec("target", task=tasks[0].ref),),
            (ScorerSpec("score", tasks[1].ref, (DimensionContract("quality", "number", "higher"),)),),
            policy=EvaluationPolicy(allow_volatile=True, content_retention_seconds=60,
                                    metadata_retention_seconds=60)), owner, "start"), engine=engine)
        await asyncio.wait_for(entered.wait(), 5)
        record = await storage.evaluation.records.get(run.experiment_id, tenant_id="tenant")
        assert len(record.intents) == 1
        submission = record.intents[0].submission.ref

        class Quiesce:
            @asynccontextmanager
            async def offline_exclusivity(self) -> AsyncIterator[None]:
                release.set()
                await asyncio.gather(*tuple(runtime.evaluations._watchers.values()))
                yield

        result = await runtime.evaluations.purge_expired(principal=owner,
            now=record.metadata_expires_at + timedelta(seconds=1), exclusive=Quiesce())
        assert result.evaluations == (run.experiment_id,)
        objects = storage.object_store(RuntimeDomain.TASK)
        assert await objects.stat(input_capture_key(submission.namespace, submission.tenant_id,
                                                   "declaration", submission.graph_id)) is None
        assert await objects.stat(task_graph_binding_capture_key(submission.namespace, submission.tenant_id,
            submission.graph_id, submission.request_digest)) is None
        assert await storage.task.admissions.submission_status(submission) == "cancelled"


@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", ("before", "after"))
@pytest.mark.parametrize("action", ("cancel", "resume"))
async def test_interrupted_preparation_has_recoverable_owner(
    boundary: str, action: str, tmp_path: Path,
) -> None:
    calls = []

    async def target(ctx):
        calls.append(ctx.input)
        return "ok"

    async def score(ctx):
        return {"dimensions": {"quality": 1.0}}

    owner = service_principal("tenant", "owner")
    tasks = (Task("target", target, effect_policy="none"), Task("score", score, effect_policy="none"))
    storage = RuntimeStorage.filesystem(tmp_path)
    async with Runtime.open("preparation", models=ModelRegistry(), storage=storage,
                            context=RuntimeContext(None, tenant_id="tenant")) as runtime:
        engine = runtime.tasks.bind(*tasks)
        entered = asyncio.Event()
        prepare = storage.task.admissions.prepare
        saved = []

        async def interrupt(submission: TaskGraphSubmission) -> TaskGraphSubmission:
            saved.append(submission)
            if boundary == "after":
                submission = await prepare(submission)
            entered.set()
            await asyncio.Event().wait()
            return submission

        storage.task.admissions.prepare = interrupt
        dataset = await runtime.evaluations.publish_dataset(DatasetSpec(DatasetRef("dataset", 1), cases=(
            CaseSpec.task(CaseRef("dataset", "case", 1), input={"secret": "recoverable private input"}),
        )), principal=owner, idempotency_key="publish")
        run = await runtime.evaluations.start(StartEvaluationRequest(EvaluationSpec(dataset,
            (CandidateSpec("target", task=tasks[0].ref),),
            (ScorerSpec("score", tasks[1].ref, (DimensionContract("quality", "number", "higher"),)),),
            policy=EvaluationPolicy(allow_volatile=True, content_retention_seconds=60,
                                    metadata_retention_seconds=60)), owner, "start"), engine=engine)
        await asyncio.wait_for(entered.wait(), 5)
        await runtime.evaluations.close()
        experiment_id = run.experiment_id
        assert calls == []
    reopened = RuntimeStorage.filesystem(tmp_path)
    async with Runtime.open("preparation", models=ModelRegistry(), storage=reopened,
                            context=RuntimeContext(None, tenant_id="tenant")) as runtime:
        run = await runtime.evaluations.get(experiment_id, principal=owner)
        if action == "resume":
            await runtime.evaluations.reconcile(experiment_id, engine=runtime.tasks.bind(*tasks),
                                                principal=owner, idempotency_key="resume")
            assert (await run.wait(timeout_seconds=20)).completion == "complete"
            assert calls == [{"secret": "recoverable private input"}]
        else:
            assert (await run.cancel(idempotency_key="cancel")).completion == "cancelled"
            record = await reopened.evaluation.records.get(experiment_id, tenant_id="tenant")
            result = await runtime.evaluations.purge_expired(principal=owner,
                now=record.metadata_expires_at + timedelta(seconds=1), exclusive=Offline())
            assert result.evaluations == (experiment_id,)
            assert await reopened.task.admissions.submission_status(saved[0].ref) == "cancelled"
            rows = await reopened.task.admissions.state_store.read(lambda tx: tx.scan_records())
            assert not [row for row in rows if row.kind == "task_submission_payload"]
            assert calls == []
