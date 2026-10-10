#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluation admission owns its private inputs before any capture materialization."""

import asyncio
from collections.abc import AsyncIterator, Mapping
from pathlib import Path

from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from dataclasses import replace

import pytest

from linktools.ai.core import JsonValue, service_principal
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.evaluation import (
    AgentCaseInput, CandidateSpec, CaseRef, CaseSpec, DatasetRef, DatasetSpec, DimensionContract,
    EvaluationPolicy, EvaluationSpec, GraphTargetSpec, ScorerSpec, ScoreBundle,
    StartEvaluationRequest, TaskCaseInput,
)
from linktools.ai.model import ModelRegistry
from linktools.ai.runtime import CaptureGraphRequest, CaptureInputRequest, Runtime, RuntimeContext, RuntimeStorage
from linktools.ai.runtime.state import RuntimeDomain
from linktools.ai.task import Task, TaskGraph, TaskGraphTemplate, TaskNode, TaskNodeResultRef, TaskRef, TaskNodeContext

from .test_evaluation_consumers import EVALUATION_COMPLETION_TIMEOUT_SECONDS


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
        assert (await run.wait(timeout_seconds=EVALUATION_COMPLETION_TIMEOUT_SECONDS)).result.completion == "complete"
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
            patch.setattr(RuntimeEvaluations, "_watch", lambda *args, **kwargs: None)
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
        assert (await run.wait(timeout_seconds=EVALUATION_COMPLETION_TIMEOUT_SECONDS)).result.completion == "complete"
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
            patch.setattr(RuntimeEvaluations, "_watch", lambda *args, **kwargs: None)
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
            await run.create_report()
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
        assert (await run.wait(timeout_seconds=EVALUATION_COMPLETION_TIMEOUT_SECONDS)).result.completion == "complete"
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


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["input", "graph"])
async def test_function_task_capture_keeps_agent_shaped_business_json(tmp_path: Path, route: str) -> None:
    payload = {"kind": "agent-task-input", "prompt": "business literal", "session_id": "business-session",
               "memory_scope": "business-memory", "capture_context": {"business": True}, "files": ["business-file"]}

    async def target(context: TaskNodeContext[None]) -> JsonValue:
        return dict(context.input)

    async with Runtime.open("business-json", models=ModelRegistry(), storage=RuntimeStorage.filesystem(tmp_path), context=CONTEXT) as runtime:
        task, scorer = Task("business.target", target, effect_policy="none"), Task("business.score", score, effect_policy="none")
        engine = runtime.tasks.bind(task, scorer)
        source = await engine.start(TaskGraph("source", (TaskNode("target", task=task, input=payload),)),
            principal=PRINCIPAL, idempotency_key="source")
        await source.wait()
        if route == "input":
            execution = await source.execution("target")
            captured = await runtime.executions.capture_input(execution.execution_id, CaptureInputRequest(PRINCIPAL, "capture"))
            case = CaseSpec.from_capture(CaseRef("data", "one", 1), capture=captured)
            candidate = CandidateSpec("candidate", task=task.ref)
        else:
            captured = await runtime.tasks.capture_graph("source", CaptureGraphRequest(PRINCIPAL, "capture", context_policy="clean"))
            case = CaseSpec.graph(CaseRef("data", "one", 1), inputs={})
            candidate = CandidateSpec("candidate", graph_template=GraphTargetSpec(capture=captured, outputs={"answer": "target"}))
        dataset = await runtime.evaluations.publish_dataset(DatasetSpec(DatasetRef("data", 1), cases=(case,)),
            principal=PRINCIPAL, idempotency_key="dataset")
        for mode in ("fixed_input", "reproject_input"):
            run = await runtime.evaluations.start(StartEvaluationRequest(EvaluationSpec(dataset, (candidate,),
                (ScorerSpec("score", scorer.ref, (DIMENSION,)),), input_mode=mode), PRINCIPAL, mode), engine=engine)
            assert (await run.wait(timeout_seconds=EVALUATION_COMPLETION_TIMEOUT_SECONDS)).result.completion == "complete"
            trial = (await run.trials()).items[0]
            evidence = await runtime.evaluations.read_evidence(trial.evidence_ref, principal=PRINCIPAL)
            actual = evidence.target.output.value if route == "input" else evidence.target.outputs["answer"].value.value
            assert actual == payload


@pytest.mark.asyncio
@pytest.mark.parametrize(("capture_owner", "input_mode"), (
    ("case", "fixed_input"), ("case", "reproject_input"),
    ("template_graph", "fixed_input"), ("template_input", "fixed_input"),
))
async def test_captured_inputs_preserve_merged_fields_and_original_parameters(
    tmp_path: Path, capture_owner: str, input_mode: str,
) -> None:
    def normalize(value: Mapping[str, JsonValue]) -> Mapping[str, JsonValue]:
        return {**value, "number": value["number"] + 1}

    async def echo(context: TaskNodeContext[None]) -> JsonValue:
        return dict(context.input)

    target = Task("merge.capture", echo, normalize=normalize, effect_policy="none")
    judge = Task("merge.score", score, effect_policy="none")
    scorer = ScorerSpec("score", judge.ref, (DIMENSION,))
    async with Runtime.open("captured-case", models=ModelRegistry(), storage=RuntimeStorage.filesystem(tmp_path), context=CONTEXT) as runtime:
        engine = runtime.tasks.bind(target, judge)
        source = await engine.start(TaskGraph("source", (TaskNode("target", task=target, input={"number": 1}),)),
            principal=PRINCIPAL, idempotency_key="source")
        await source.wait()
        execution = await source.execution("target")
        captured = await runtime.executions.capture_input(execution.execution_id, CaptureInputRequest(PRINCIPAL, "source-input"))
        merged = {"number": 2 if input_mode == "fixed_input" else 1, "added": "template"}
        if capture_owner == "case":
            case_input = TaskCaseInput(capture=captured)
            graph_target = GraphTargetSpec(template=TaskGraphTemplate((TaskNode("target", task=target, input=merged),)),
                                           outputs={"answer": "target"})
        else:
            case_input = TaskCaseInput(input=merged)
            if capture_owner == "template_graph":
                graph_capture = await runtime.tasks.capture_graph(source.graph_id, CaptureGraphRequest(PRINCIPAL, "graph-input"))
                graph_target = GraphTargetSpec(capture=graph_capture, outputs={"answer": "target"})
            else:
                graph_target = GraphTargetSpec(template=TaskGraphTemplate((TaskNode("target", task=target, input_capture=captured),)),
                                               outputs={"answer": "target"})
        dataset = await runtime.evaluations.publish_dataset(DatasetSpec(DatasetRef("data", 1), (
            CaseSpec.graph(CaseRef("data", "one", 1), inputs={"target": case_input}),
        )), principal=PRINCIPAL, idempotency_key="dataset")
        candidate = CandidateSpec("candidate", graph_template=graph_target)
        run = await runtime.evaluations.start(StartEvaluationRequest(EvaluationSpec(dataset, (candidate,), (scorer,),
            input_mode=input_mode), PRINCIPAL, "evaluate"), engine=engine)
        assert (await run.wait(timeout_seconds=EVALUATION_COMPLETION_TIMEOUT_SECONDS)).result.completion == "complete"
        trial = (await run.trials()).items[0]
        graph = await engine.get(trial.graph_ref.graph_id, principal=PRINCIPAL)
        assert await graph.result("target") == {"number": 2, "added": "template"}
        replay_execution = await graph.execution("target")
        replay_capture = await runtime.executions.capture_input(replay_execution.execution_id, CaptureInputRequest(PRINCIPAL, "replay-input"))
        replay_dataset = await runtime.evaluations.publish_dataset(DatasetSpec(DatasetRef("replay", 1), (
            CaseSpec.from_capture(CaseRef("replay", "one", 1), capture=replay_capture),
        )), principal=PRINCIPAL, idempotency_key="replay-dataset")
        replay = await runtime.evaluations.start(StartEvaluationRequest(EvaluationSpec(replay_dataset,
            (CandidateSpec("candidate", task=target.ref),), (scorer,), input_mode="reproject_input"),
            PRINCIPAL, "reproject"), engine=engine)
        assert (await replay.wait(timeout_seconds=EVALUATION_COMPLETION_TIMEOUT_SECONDS)).result.completion == "complete"
        evidence = await runtime.evaluations.read_evidence((await replay.trials()).items[0].evidence_ref, principal=PRINCIPAL)
        assert evidence.target.output.value == {"number": 2, "added": "template"}
        conflicting = CandidateSpec("conflicting", graph_template=GraphTargetSpec(
            template=TaskGraphTemplate((TaskNode("target", task=target, input={"number": 3}),)), outputs={"answer": "target"}))
        with pytest.raises(AIError) as rejected:
            await runtime.evaluations.start(StartEvaluationRequest(EvaluationSpec(dataset, (conflicting,), (scorer,),
                input_mode=input_mode), PRINCIPAL, "conflict"), engine=engine)
        assert rejected.value.code is ErrorCode.EVALUATION_INCOMPATIBLE


@pytest.mark.asyncio
async def test_graph_input_capture_has_one_owner(tmp_path: Path) -> None:
    async def echo(context: TaskNodeContext[None]) -> JsonValue:
        return dict(context.input)

    target, judge = Task("capture.echo", echo, effect_policy="none"), Task("capture.score", score, effect_policy="none")
    async with Runtime.open("capture-owner", models=ModelRegistry(), storage=RuntimeStorage.filesystem(tmp_path), context=CONTEXT) as runtime:
        engine = runtime.tasks.bind(target, judge)
        captures = []
        for identity in ("first", "second"):
            graph = await engine.start(TaskGraph(identity, (TaskNode("target", task=target, input={identity: True}),)),
                principal=PRINCIPAL, idempotency_key=identity)
            await graph.wait()
            execution = await graph.execution("target")
            captures.append(await runtime.executions.capture_input(execution.execution_id, CaptureInputRequest(PRINCIPAL, identity)))
        dataset = await runtime.evaluations.publish_dataset(DatasetSpec(DatasetRef("data", 1), (
            CaseSpec.graph(CaseRef("data", "one", 1), inputs={"target": TaskCaseInput(capture=captures[0])}),
        )), principal=PRINCIPAL, idempotency_key="dataset")
        scorer = ScorerSpec("score", judge.ref, (DIMENSION,))
        for index, capture in enumerate(captures):
            candidate = CandidateSpec("candidate", graph_template=GraphTargetSpec(
                template=TaskGraphTemplate((TaskNode("target", task=target, input_capture=capture),)), outputs={"answer": "target"}))
            request = StartEvaluationRequest(EvaluationSpec(dataset, (candidate,), (scorer,)), PRINCIPAL, f"evaluate-{index}")
            if index:
                with pytest.raises(AIError) as rejected:
                    await runtime.evaluations.start(request, engine=engine)
                assert rejected.value.code is ErrorCode.EVALUATION_INCOMPATIBLE
            else:
                run = await runtime.evaluations.start(request, engine=engine)
                assert (await run.wait(timeout_seconds=EVALUATION_COMPLETION_TIMEOUT_SECONDS)).result.completion == "complete"
                evidence = await runtime.evaluations.read_evidence((await run.trials()).items[0].evidence_ref, principal=PRINCIPAL)
                assert evidence.target.outputs["answer"].value.value == {"first": True}


@pytest.mark.asyncio
async def test_agent_graph_case_preserves_template_options_and_checks_prompt_owner(tmp_path: Path) -> None:
    from linktools.ai.capability import CapabilityGroup
    from linktools.ai.runtime import AgentTaskInputContext
    from .test_evaluation_consumers import FixtureModels

    async def project(context: AgentTaskInputContext) -> str:
        return f"{context.prompt}:{context.input['suffix']}"

    models = FixtureModels()
    group = CapabilityGroup("case-agent")
    group.agent("default", model="default", allow_tools=(), allow_skills=(), allow_subagents=())
    async with Runtime.open("agent-case", models=models, storage=RuntimeStorage.filesystem(tmp_path), context=CONTEXT,
                            capabilities=(group,)) as runtime:
        target = runtime.tasks.from_agent("case.agent", runtime.agents.get(), build_input=project)
        judge = Task("case.score", score, effect_policy="none")
        engine = runtime.tasks.bind(target, judge)
        dataset = await runtime.evaluations.publish_dataset(DatasetSpec(DatasetRef("data", 1), (
            CaseSpec.graph(CaseRef("data", "one", 1), inputs={"target": AgentCaseInput(prompt="question")}),
        )), principal=PRINCIPAL, idempotency_key="dataset")
        for question in ("question", "conflict"):
            candidate = CandidateSpec("candidate", graph_template=GraphTargetSpec(template=TaskGraphTemplate((
                TaskNode("target", task=target, input={"prompt": question, "parameters": {"suffix": "template"}}),)), outputs={"answer": "target"}))
            request = StartEvaluationRequest(EvaluationSpec(dataset, (candidate,),
                (ScorerSpec("score", judge.ref, (DIMENSION,)),), policy=EvaluationPolicy(model_fixtures=(models.contract,))),
                PRINCIPAL, question)
            if question == "conflict":
                with pytest.raises(AIError) as rejected:
                    await runtime.evaluations.start(request, engine=engine)
                assert rejected.value.code is ErrorCode.EVALUATION_INCOMPATIBLE
            else:
                run = await runtime.evaluations.start(request, engine=engine)
                assert (await run.wait(timeout_seconds=EVALUATION_COMPLETION_TIMEOUT_SECONDS)).result.completion == "complete"
                trial = (await run.trials()).items[0]
                graph = await engine.get(trial.graph_ref.graph_id, principal=PRINCIPAL)
                assert (await graph.state(include_content=True)).nodes[0].input["parameters"] == {"suffix": "template"}
        assert models.prompts == ["question:template"]
