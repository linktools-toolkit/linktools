#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluation admission owns its private inputs before any capture materialization."""

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path

from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from dataclasses import replace

import pytest

from linktools.ai.core import JsonValue, service_principal
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.evaluation import (
    CandidateSpec, CaseRef, CaseSpec, DatasetRef, DatasetSpec, DimensionContract,
    EvaluationPolicy, EvaluationSpec, GraphTargetSpec, ScorerSpec, ScoreBundle,
    StartEvaluationRequest, TaskCaseInput,
)
from linktools.ai.model import ModelRegistry
from linktools.ai.runtime import CaptureGraphRequest, CaptureInputRequest, Runtime, RuntimeContext, RuntimeStorage
from linktools.ai.runtime.state import RuntimeDomain
from linktools.ai.task import Task, TaskGraph, TaskGraphTemplate, TaskNode, TaskNodeResultRef, TaskRef, TaskNodeContext


PRINCIPAL = service_principal("template-owner", "owner")
CONTEXT = RuntimeContext(None, tenant_id=PRINCIPAL.tenant_id)
DIMENSION = DimensionContract("ok", "boolean", "higher", 0, 1)


class Offline:
    @asynccontextmanager
    async def offline_exclusivity(self) -> AsyncIterator[None]:
        yield


async def score(context: TaskNodeContext[None]) -> JsonValue:
    return ScoreBundle(dimensions={"ok": 1}).to_mapping()


async def task_objects(storage: RuntimeStorage) -> dict[str, str]:
    return {item.key: item.digest async for item in storage.object_store(RuntimeDomain.TASK).list_objects()}


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["scorer", "case", "reservation"])
async def test_failed_admission_does_not_persist_private_templates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str) -> None:
    storage = RuntimeStorage.filesystem(tmp_path)

    async def target(context: TaskNodeContext[None]) -> JsonValue:
        return dict(context.input)

    async with Runtime.open("admission", models=ModelRegistry(), storage=storage, context=CONTEXT) as runtime:
        task, scorer = Task("owner.target", target, effect_policy="none"), Task("owner.score", score, effect_policy="none")
        dataset = await runtime.evaluations.publish_dataset(DatasetSpec(DatasetRef("data", 1), cases=(
            CaseSpec.graph(CaseRef("data", "one", 1), inputs={"unknown" if failure == "case" else "target": TaskCaseInput(input={})}),
        )), principal=PRINCIPAL, idempotency_key="dataset")
        spec = EvaluationSpec(dataset, (CandidateSpec("candidate", graph_template=GraphTargetSpec(
            template=TaskGraphTemplate((TaskNode("target", task=task, input={"private": "sensitive template"}),)),
            selector="terminal_sinks")),), (ScorerSpec("score", TaskRef("unbound", 1) if failure == "scorer" else scorer.ref, (DIMENSION,)),),
            policy=EvaluationPolicy(content_retention_seconds=60, metadata_retention_seconds=60))
        before = await task_objects(storage)
        if failure == "reservation":
            async def reject(self, record):
                raise AIError(ErrorCode.STORAGE_UNAVAILABLE)
            monkeypatch.setattr(type(storage.evaluation.records), "reserve_experiment", reject)
        with pytest.raises(AIError):
            await runtime.evaluations.start(StartEvaluationRequest(spec, PRINCIPAL, "start"), engine=runtime.tasks.bind(task, scorer))
        assert await task_objects(storage) == before
        await runtime.evaluations.purge_expired(principal=PRINCIPAL, now=datetime.now(timezone.utc) + timedelta(days=30), exclusive=Offline())
        assert await task_objects(storage) == before


@pytest.mark.asyncio
async def test_graph_case_merges_preserve_captured_parameters_and_dependencies(tmp_path: Path) -> None:
    async def source(context: TaskNodeContext[None]) -> JsonValue:
        return "frozen"

    async def consume(context: TaskNodeContext[None]) -> JsonValue:
        return [await context.read_dependency("frozen"), dict(context.input)]

    async with Runtime.open("merge", models=ModelRegistry(), storage=RuntimeStorage.filesystem(tmp_path), context=CONTEXT) as runtime:
        producer, consumer = Task("merge.source", source, effect_policy="none"), Task("merge.consume", consume, effect_policy="none")
        scorer = Task("merge.score", score, effect_policy="none")
        engine = runtime.tasks.bind(producer, consumer, scorer)
        original = await engine.start(TaskGraph("source", (
            TaskNode("source", task=producer),
            TaskNode("consume", ("source",), task=consumer, input={"kept": 1}, input_refs={"frozen": TaskNodeResultRef("source")}),
        )), principal=PRINCIPAL, idempotency_key="source")
        await original.wait()
        execution = await original.execution("consume")
        capture = await runtime.executions.capture_input(execution.execution_id, CaptureInputRequest(PRINCIPAL, "input"))
        captured_graph = await engine.start(TaskGraph("captured", (TaskNode("consume", task=consumer, input_capture=capture),)),
            principal=PRINCIPAL, idempotency_key="captured")
        await captured_graph.wait()
        template = await runtime.tasks.capture_graph("captured", CaptureGraphRequest(PRINCIPAL, "template"))
        dataset = await runtime.evaluations.publish_dataset(DatasetSpec(DatasetRef("merge", 1), cases=tuple(
            CaseSpec.graph(CaseRef("merge", name, 1), inputs=inputs) for name, inputs in (
                ("omitted", {}), ("empty", {"consume": TaskCaseInput(input={})}),
                ("additive", {"consume": TaskCaseInput(input={"added": 2})}),
            )
        )), principal=PRINCIPAL, idempotency_key="dataset")
        spec = EvaluationSpec(dataset, (CandidateSpec("candidate", graph_template=GraphTargetSpec(capture=template, outputs={"answer": "consume"})),),
            (ScorerSpec("score", scorer.ref, (DIMENSION,)),))
        run = await runtime.evaluations.start(StartEvaluationRequest(spec, PRINCIPAL, "evaluate"), engine=engine)
        assert (await run.wait(timeout_seconds=30)).completion == "complete"
        for trial in (await run.trials()).items:
            evidence = await runtime.evaluations.read_evidence(trial.evidence_ref, principal=PRINCIPAL)
            output = evidence.target.outputs["answer"].value.value
            assert output == ["frozen", {"kept": 1, **({"added": 2} if trial.case_ref.case_id == "additive" else {})}]


@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", ["before", "after"])
async def test_reserved_capture_is_owned_across_cancel_and_purge(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, boundary: str) -> None:
    from linktools.ai.runtime._input_capture import RuntimeInputCaptures
    from linktools.ai.runtime.state import input_capture_key

    invocations = []

    async def target(context: TaskNodeContext[None]) -> JsonValue:
        invocations.append(dict(context.input))
        return dict(context.input)

    storage = RuntimeStorage.filesystem(tmp_path)
    async with Runtime.open("capture-owner", models=ModelRegistry(), storage=storage, context=CONTEXT) as runtime:
        task, scorer = Task("owner.target", target, effect_policy="none"), Task("owner.score", score, effect_policy="none")
        engine = runtime.tasks.bind(task, scorer)
        source = await engine.start(TaskGraph("source", (TaskNode("target", task=task, input={"secret": "retained"}),)),
            principal=PRINCIPAL, idempotency_key="source")
        await source.wait()
        execution = await source.execution("target")
        capture = await runtime.executions.capture_input(execution.execution_id, CaptureInputRequest(PRINCIPAL, "source-capture"))
        dataset = await runtime.evaluations.publish_dataset(DatasetSpec(DatasetRef("data", 1), cases=(
            CaseSpec.from_capture(CaseRef("data", "one", 1), capture=capture),
        )), principal=PRINCIPAL, idempotency_key="dataset")
        entered, release = asyncio.Event(), asyncio.Event()
        original = RuntimeInputCaptures.create_task_input
        saved = []

        async def paused(self, contract, **kwargs):
            saved.append((contract, kwargs))
            reference = None
            if boundary == "after":
                reference = await original(self, contract, **kwargs)
            entered.set()
            await release.wait()
            return reference if reference is not None else await original(self, contract, **kwargs)

        monkeypatch.setattr(RuntimeInputCaptures, "create_task_input", paused)
        run = await runtime.evaluations.start(StartEvaluationRequest(EvaluationSpec(dataset,
            (CandidateSpec("candidate", task=task.ref),), (ScorerSpec("score", scorer.ref, (DIMENSION,)),),
            policy=EvaluationPolicy(content_retention_seconds=60, metadata_retention_seconds=60)), PRINCIPAL, "start"), engine=engine)
        await asyncio.wait_for(entered.wait(), 5)
        record = await storage.evaluation.records.get(run.experiment_id, tenant_id=PRINCIPAL.tenant_id)
        assert len(record.owned_input_captures) == 1
        owned = record.owned_input_captures[0]
        key = input_capture_key(owned.namespace, owned.tenant_id, "task", owned.capture_id)
        assert (await storage.object_store(RuntimeDomain.TASK).stat(key) is not None) == (boundary == "after")
        assert (await run.cancel(idempotency_key="cancel")).completion == "cancelled"
        release.set()
        await asyncio.wait_for(runtime.evaluations._watchers[run.experiment_id], 5)
        result = await runtime.evaluations.purge_expired(principal=PRINCIPAL,
            now=record.metadata_expires_at + timedelta(seconds=1), exclusive=Offline())
        assert result.evaluations == (run.experiment_id,)
        assert await storage.object_store(RuntimeDomain.TASK).stat(key) is None
        assert invocations == [{"secret": "retained"}]
        assert await runtime.evaluations._captures.read_task(capture, principal=PRINCIPAL)
        with pytest.raises(AIError) as rejected:
            await original(runtime.evaluations._captures, saved[0][0], **saved[0][1])
        assert rejected.value.code is ErrorCode.INPUT_CAPTURE_UNAVAILABLE


@pytest.mark.asyncio
async def test_unmaterialized_capture_reservation_survives_snapshot_and_reconcile(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from linktools.ai.runtime._evaluation import RuntimeEvaluations
    from linktools.ai.runtime.state import SnapshotLimits, input_capture_key
    from linktools.ai.storage import InMemoryObjectStore

    async def target(context: TaskNodeContext[None]) -> JsonValue:
        return dict(context.input)

    root = tmp_path / "source"
    storage = RuntimeStorage.filesystem(root)
    task, scorer = Task("snapshot.target", target, effect_policy="none"), Task("snapshot.score", score, effect_policy="none")
    async with Runtime.open("snapshot", models=ModelRegistry(), storage=storage, context=CONTEXT) as runtime:
        engine = runtime.tasks.bind(task, scorer)
        source = await engine.start(TaskGraph("source", (TaskNode("target", task=task, input={"kept": 1}),)),
            principal=PRINCIPAL, idempotency_key="source")
        await source.wait()
        execution = await source.execution("target")
        captured = await runtime.executions.capture_input(execution.execution_id, CaptureInputRequest(PRINCIPAL, "source"))
        dataset = await runtime.evaluations.publish_dataset(DatasetSpec(DatasetRef("data", 1), cases=(
            CaseSpec.from_capture(CaseRef("data", "one", 1), capture=captured),
        )), principal=PRINCIPAL, idempotency_key="dataset")
        with monkeypatch.context() as patch:
            patch.setattr(RuntimeEvaluations, "_watch", lambda *args: None)
            run = await runtime.evaluations.start(StartEvaluationRequest(EvaluationSpec(dataset,
                (CandidateSpec("candidate", task=task.ref),), (ScorerSpec("score", scorer.ref, (DIMENSION,)),)),
                PRINCIPAL, "start"), engine=engine)
        experiment_id = run.experiment_id
        record = await storage.evaluation.records.get(experiment_id, tenant_id=PRINCIPAL.tenant_id)
        owned = record.owned_input_captures[0]
        key = input_capture_key(owned.namespace, owned.tenant_id, "task", owned.capture_id)
        assert await storage.object_store(RuntimeDomain.TASK).stat(key) is None
    snapshot = InMemoryObjectStore("snapshot-copy")
    limits = SnapshotLimits(max_entries=10000, max_bytes=20 * 1024 * 1024)
    readonly = RuntimeStorage.filesystem(root)
    await readonly.initialize(namespace="snapshot", tenant_id=PRINCIPAL.tenant_id, read_only=True)
    try:
        reference = await readonly.export_snapshot(object_store=snapshot, limits=limits)
    finally:
        await readonly.close()
    restored = tmp_path / "restored"
    await RuntimeStorage.restore_snapshot(reference, object_store=snapshot, root=restored, limits=limits)
    restored_storage = RuntimeStorage.from_root(restored)
    async with Runtime.open("snapshot", models=ModelRegistry(), storage=restored_storage, context=CONTEXT) as runtime:
        run = await runtime.evaluations.reconcile(experiment_id, engine=runtime.tasks.bind(task, scorer),
            principal=PRINCIPAL, idempotency_key="resume")
        assert (await run.wait(timeout_seconds=30)).completion == "complete"
        record = await restored_storage.evaluation.records.get(experiment_id, tenant_id=PRINCIPAL.tenant_id)
        assert record.owned_input_captures == (owned,)
        assert await restored_storage.object_store(RuntimeDomain.TASK).stat(key) is not None


@pytest.mark.asyncio
async def test_content_purge_redacts_inline_template_and_keeps_admission_identity(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from linktools.ai.runtime._evaluation import RuntimeEvaluations
    from linktools.ai.runtime.state._codec import decode_domain, encode_domain
    from linktools.ai.runtime.state._evaluation_records import EvaluationRecord

    async def target(context: TaskNodeContext[None]) -> JsonValue:
        return dict(context.input)

    storage = RuntimeStorage.filesystem(tmp_path)
    async with Runtime.open("redaction", models=ModelRegistry(), storage=storage, context=CONTEXT) as runtime:
        task, scorer = Task("redaction.target", target, effect_policy="none"), Task("redaction.score", score, effect_policy="none")
        dataset = await runtime.evaluations.publish_dataset(DatasetSpec(DatasetRef("data", 1), cases=(
            CaseSpec.graph(CaseRef("data", "one", 1), inputs={}),
        )), principal=PRINCIPAL, idempotency_key="dataset")
        with monkeypatch.context() as patch:
            patch.setattr(RuntimeEvaluations, "_watch", lambda *args: None)
            run = await runtime.evaluations.start(StartEvaluationRequest(EvaluationSpec(dataset,
                (CandidateSpec("candidate", graph_template=GraphTargetSpec(template=TaskGraphTemplate((
                    TaskNode("target", task=task, input={"private": "erase inline input"}),)), selector="terminal_sinks")),),
                (ScorerSpec("score", scorer.ref, (DIMENSION,)),),
                policy=EvaluationPolicy(content_retention_seconds=60)), PRINCIPAL, "start"), engine=runtime.tasks.bind(task, scorer))
        record = await storage.evaluation.records.get(run.experiment_id, tenant_id=PRINCIPAL.tenant_id)
        assert record.manifest.digest == record.manifest_digest
        await run.cancel(idempotency_key="cancel")
        result = await runtime.evaluations.purge_expired(principal=PRINCIPAL,
            now=record.content_expires_at + timedelta(seconds=1), exclusive=Offline())
        assert result.evaluations == (run.experiment_id,)
        deleted = await storage.evaluation.records.get(run.experiment_id, tenant_id=PRINCIPAL.tenant_id)
        assert deleted.manifest.candidates[0].graph_template.template is None
        assert deleted.manifest_digest == record.manifest_digest
        assert deleted.content_deleted_at is not None
        assert decode_domain(encode_domain(deleted), EvaluationRecord) == deleted
        with pytest.raises(ValueError):
            replace(deleted, content_deleted_at=None)
        with pytest.raises(AIError) as expired:
            await run.report()
        assert expired.value.code is ErrorCode.EVALUATION_EVIDENCE_UNAVAILABLE


@pytest.mark.asyncio
async def test_evaluation_purge_preserves_native_owners_of_derived_captures(tmp_path: Path) -> None:
    from linktools.ai.runtime.state import SnapshotLimits, input_capture_key
    from linktools.ai.storage import InMemoryObjectStore

    async def target(context: TaskNodeContext[None]) -> JsonValue:
        return dict(context.input)

    storage = RuntimeStorage.filesystem(tmp_path)
    async with Runtime.open("native-owner", models=ModelRegistry(), storage=storage, context=CONTEXT) as runtime:
        task, scorer = Task("native.target", target, effect_policy="none"), Task("native.score", score, effect_policy="none")
        engine = runtime.tasks.bind(task, scorer)
        source = await engine.start(TaskGraph("source", (TaskNode("target", task=task, input={"kept": 1}),)),
            principal=PRINCIPAL, idempotency_key="source")
        await source.wait()
        execution = await source.execution("target")
        captured = await runtime.executions.capture_input(execution.execution_id, CaptureInputRequest(PRINCIPAL, "source"))
        dataset = await runtime.evaluations.publish_dataset(DatasetSpec(DatasetRef("data", 1), cases=(
            CaseSpec.from_capture(CaseRef("data", "one", 1), capture=captured),
        )), principal=PRINCIPAL, idempotency_key="dataset")
        run = await runtime.evaluations.start(StartEvaluationRequest(EvaluationSpec(dataset,
            (CandidateSpec("candidate", task=task.ref),), (ScorerSpec("score", scorer.ref, (DIMENSION,)),),
            policy=EvaluationPolicy(content_retention_seconds=60)), PRINCIPAL, "start"), engine=engine)
        assert (await run.wait(timeout_seconds=30)).completion == "complete"
        trial = (await run.trials()).items[0]
        record = await storage.evaluation.records.get(run.experiment_id, tenant_id=PRINCIPAL.tenant_id)
        owned = record.owned_input_captures[0]
        result = await runtime.evaluations.purge_expired(principal=PRINCIPAL,
            now=record.content_expires_at + timedelta(seconds=1), exclusive=Offline())
        assert result.objects_retained >= 1
        key = input_capture_key(owned.namespace, owned.tenant_id, "task", owned.capture_id)
        assert await storage.object_store(RuntimeDomain.TASK).stat(key) is not None
        retained = await runtime.executions.capture_input(trial.subject.execution_id, CaptureInputRequest(PRINCIPAL, "retained"))
        assert dict((await runtime.evaluations._captures.read_task(retained, principal=PRINCIPAL)).input) == {"kept": 1}
    snapshot = InMemoryObjectStore("native-snapshot")
    readonly = RuntimeStorage.filesystem(tmp_path)
    await readonly.initialize(namespace="native-owner", tenant_id=PRINCIPAL.tenant_id, read_only=True)
    try:
        assert await readonly.export_snapshot(object_store=snapshot, limits=SnapshotLimits(max_entries=10000, max_bytes=20 * 1024 * 1024))
    finally:
        await readonly.close()
