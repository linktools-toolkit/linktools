#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Retention removes owned bytes while preserving native fences and shared refs."""

from linktools.ai.persistence import Persistence
from linktools.ai.snapshot import Snapshot, SnapshotLimits
from linktools.ai.task_engine import TaskEngine
from linktools.ai.evaluation_engine import Evaluations

from linktools.ai.core import ExecutionContext

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime, timezone, tzinfo
from pathlib import Path

import pytest

from linktools.ai.core import AuthorizationAction, JsonValue, Principal, ResourceRef, canonical_json_bytes, canonical_sha256
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.agent import AgentInputCaptureRef
from linktools.ai.persistence.captures import input_capture_key
from linktools.ai.persistence.contracts import ArtifactRecord
from linktools.ai.persistence import PersistenceDomain, PersistenceStorage, PersistencePlan
from linktools.ai.storage import StorageRoute
from linktools.ai.persistence.codec import encode_domain
from linktools.ai.persistence._object_cleanup import purge_unreferenced_objects
from linktools.ai.storage import InMemoryObjectStore, ObjectRef

from .test_evaluation_consumers import EVALUATION_COMPLETION_TIMEOUT_SECONDS


NOW = datetime.now(timezone.utc)


class Offline:
    """The test owns the runtime and performs no concurrent mutation."""

    def __init__(self) -> None:
        self.active = False

    @asynccontextmanager
    async def offline_exclusivity(self) -> AsyncIterator[None]:
        self.active = True
        try:
            yield
        finally:
            self.active = False


class CheckedObjects(InMemoryObjectStore):
    def __init__(self, guard: Offline) -> None:
        super().__init__()
        self.guard = guard

    async def delete_object(self, key: str, *, expected_digest: str) -> bool:
        assert self.guard.active
        return await super().delete_object(key, expected_digest=expected_digest)


async def put(objects: InMemoryObjectStore, key: str, value: JsonValue) -> ObjectRef:
    data = canonical_json_bytes(value)

    async def chunks() -> AsyncIterator[bytes]:
        yield data

    stat = await objects.put(key, chunks(), expected_size=len(data), expected_digest=canonical_sha256(value))
    return ObjectRef(objects.store_id, stat.key, stat.digest, stat.size)


def storage_for(kind: str, path: Path, objects: InMemoryObjectStore) -> PersistenceStorage:
    if kind == "memory":
        return PersistenceStorage.in_memory()
    if kind == "filesystem":
        return PersistenceStorage.filesystem(path, object_store=objects)
    if kind == "sqlite":
        return PersistenceStorage.sqlite(path / "state.sqlite", object_store=objects)
    return PersistenceStorage.from_plan(PersistencePlan(
        execution=StorageRoute.sqlite(path / "execution.sqlite"),
        task=StorageRoute.filesystem(path / "task"),
        evaluation=StorageRoute.sqlite(path / "evaluation.sqlite"),
        artifact=StorageRoute.filesystem(path / "artifact"),
    ), object_store=objects)


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "filesystem", "sqlite", "mixed"))
async def test_object_cleanup_scans_all_domains_and_independent_capture_closures(backend: str, tmp_path: Path) -> None:
    guard = Offline()
    objects = CheckedObjects(guard)
    storage = storage_for(backend, tmp_path, objects)
    storage = Persistence(storage)
    await storage.initialize(namespace="evaluation-retention", tenant_id="tenant")
    try:
        shared = await put(objects, "shared-body", {"private": "referenced by another owner"})
        captured = await put(objects, "captured-body", {"private": "referenced only by capture"})
        orphan = await put(objects, "orphan", {"private": "evaluation-owned expired body"})
        await storage.artifact.records.put_metadata(ArtifactRecord(
            "artifact", "execution", "producer", "application/json", shared, NOW))
        foreign = await put(objects, input_capture_key("another-namespace", "another-tenant", "agent", "independent"), {
            "kind": "agent", "version": 1, "namespace": "another-namespace", "tenant_id": "another-tenant",
            "payload": encode_domain(captured),
        })
        await put(objects, "v1/input-capture/declaration/foreign", {
            "capture": encode_domain(AgentInputCaptureRef("another-namespace", "another-tenant", "independent", foreign.digest, None)),
        })
        candidates = tuple((PersistenceDomain.EVALUATION, ref) for ref in (shared, captured, orphan))
        result = await purge_unreferenced_objects(candidates,
            namespace="evaluation-retention", tenant_id="tenant", stores=storage._stores,
            object_stores={domain: objects for domain in PersistenceDomain}, exclusive=guard, limit=1)
        assert result.deleted == (candidates[-1],)
        assert result.retained == candidates[:2]
        assert await objects.stat(orphan.key) is None
        assert await objects.stat(shared.key) is not None
        assert await objects.stat(captured.key) is not None
        repeated = await purge_unreferenced_objects((candidates[-1],),
            namespace="evaluation-retention", tenant_id="tenant", stores=storage._stores,
            object_stores={domain: objects for domain in PersistenceDomain}, exclusive=guard)
        assert repeated.missing == (candidates[-1],)
    finally:
        await storage.close()


@pytest.mark.asyncio
async def test_object_cleanup_bounds_deletions_and_rejects_changed_bytes(tmp_path: Path) -> None:
    guard = Offline()
    objects = CheckedObjects(guard)
    storage = PersistenceStorage.in_memory()
    storage = Persistence(storage)
    await storage.initialize(namespace="evaluation-retention", tenant_id="tenant")
    try:
        refs = tuple([await put(objects, name, name) for name in ("first", "second")])
        candidates = tuple((PersistenceDomain.EVALUATION, ref) for ref in refs)
        result = await purge_unreferenced_objects(candidates,
            namespace="evaluation-retention", tenant_id="tenant", stores=storage._stores,
            object_stores={domain: objects for domain in PersistenceDomain}, exclusive=guard, limit=1)
        assert result.deleted == candidates[:1]
        assert await objects.stat(refs[1].key) is not None
        changed = ObjectRef(refs[1].store_id, refs[1].key, "0" * 64, refs[1].size)
        with pytest.raises(AIError) as raised:
            await purge_unreferenced_objects(((PersistenceDomain.EVALUATION, changed),),
                namespace="evaluation-retention", tenant_id="tenant", stores=storage._stores,
                object_stores={domain: objects for domain in PersistenceDomain}, exclusive=guard)
        assert raised.value.code is ErrorCode.STORAGE_CONFLICT
        assert await objects.stat(refs[1].key) is not None
    finally:
        await storage.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "filesystem", "sqlite", "mixed"))
async def test_public_expiry_purge_and_reopen_do_not_revive_private_evidence(backend: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from dataclasses import replace
    from datetime import timedelta
    from linktools.ai.core import service_principal
    from linktools.ai.evaluation import (
        CandidateSpec, CaseRef, CaseSpec, DatasetRef, DatasetSpec, DimensionContract,
        EvaluationPolicy, EvaluationSpec, EvidenceAttachmentRef, StartEvaluationRequest, ScorerSpec,
        CandidateSlotRef, ComparisonReadCutoff, ComparisonSpec, ScoreComparisonSelection, ScoreSelection,
    )
    from linktools.ai.persistence.objects import ObjectKeyFactory, put_runtime_object
    from linktools.ai.evaluation_engine import _service as evaluation_runtime
    from linktools.ai.persistence import _evaluation_repository as evaluation_repository
    from linktools.ai.task import Task, TaskNodeContext

    principal = service_principal("tenant", "owner")
    guard = Offline()
    objects = CheckedObjects(guard)

    async def target(context: TaskNodeContext[None]) -> JsonValue:
        return context.input["answer"]

    async def score(context: TaskNodeContext[None]) -> JsonValue:
        value = {"kind": "not_applicable", "reason": "private NA explanation"} if backend == "memory" else 1.0
        return {"dimensions": {"quality": value}, "rationale": "private score rationale"}

    tasks = (Task("retention.target", target, effect_policy="none"), Task("retention.score", score, effect_policy="none"))
    dataset_deadline = datetime.now(timezone.utc) + timedelta(seconds=60) if backend == "filesystem" else None
    policy = EvaluationPolicy(allow_volatile=backend == "memory",
        content_retention_seconds=None if dataset_deadline is not None else 60, metadata_retention_seconds=120)
    storage = storage_for(backend, tmp_path, objects)
    async with (
        Persistence.open('evaluation-retention', storage=storage, tenant_id='tenant') as persistence_179,
    ):
        async with (
            TaskEngine.open(*tasks, persistence=persistence_179, context=ExecutionContext(None)) as engine,
            Evaluations.open(engine) as evaluations,
        ):
            dataset = await evaluations.publish_dataset(DatasetSpec(DatasetRef("retained-source", 1), cases=(
                CaseSpec.task(CaseRef("retained-source", "one", 1), input={"answer": "answer"}, expected="answer"),
            )), principal=principal, idempotency_key="publish", content_expires_at=dataset_deadline)
            request = StartEvaluationRequest(EvaluationSpec(dataset, (CandidateSpec("target", task=tasks[0].ref),),
                (ScorerSpec("score", tasks[1].ref, dimensions=(DimensionContract("quality", "number", "higher"),), rubric="private rubric"),),
                policy=policy), principal, "start")
            run = await evaluations.start(request)
            assert (await run.wait(timeout_seconds=EVALUATION_COMPLETION_TIMEOUT_SECONDS)).result.completion == "complete"
            report = await run.create_report()
            trial = (await run.trials()).items[0]
            evidence = await evaluations.read_evidence(trial.evidence_ref, principal=principal)
            record = await persistence_179.evaluation.records.get(run.experiment_id, tenant_id="tenant")
            factory = ObjectKeyFactory("evaluation-retention")
            own = await put_runtime_object(persistence_179.object_store(PersistenceDomain.EVALUATION), factory,
                PersistenceDomain.EVALUATION, "tenant", b"private raw attachment")
            copied = replace(evidence, ref=replace(evidence.ref, evidence_id="retention-private-attachment"),
                             attachments=(EvidenceAttachmentRef("private", "application/octet-stream", own),))
            copied = replace(copied, ref=replace(copied.ref, digest=copied.digest))
            await persistence_179.evaluation.records.publish_evidence(copied)
            assert await evaluations.read_evidence_attachment(copied.ref, "private", principal=principal) == b"private raw attachment"
            future = (dataset_deadline or record.content_expires_at) + timedelta(seconds=1)

            class ExpiredDateTime(datetime):
                @classmethod
                def now(cls, tz: tzinfo | None = None) -> datetime:
                    return future if tz is not None else future.replace(tzinfo=None)

            monkeypatch.setattr(evaluation_runtime, "datetime", ExpiredDateTime)
            monkeypatch.setattr(evaluation_repository, "datetime", ExpiredDateTime)
            with pytest.raises(AIError) as unavailable:
                await evaluations.read_evidence(copied.ref, principal=principal)
            assert unavailable.value.code is ErrorCode.EVALUATION_EVIDENCE_UNAVAILABLE
            for read in (run.create_report, run.scores, run.trials):
                with pytest.raises(AIError) as unavailable:
                    await read()
                assert unavailable.value.code is ErrorCode.EVALUATION_EVIDENCE_UNAVAILABLE
            with pytest.raises(AIError) as unavailable:
                await evaluations.get_report(report.report_id, principal=principal)
            assert unavailable.value.code is ErrorCode.EVALUATION_EVIDENCE_UNAVAILABLE
            comparison = ComparisonSpec(CandidateSlotRef(run.experiment_id, "target"),
                CandidateSlotRef(run.experiment_id, "target"),
                (ScoreComparisonSelection(ScoreSelection("score", "quality"), ScoreSelection("score", "quality")),),
                cutoff=ComparisonReadCutoff(report.cutoff, report.cutoff))
            with pytest.raises(AIError) as unavailable:
                await evaluations.create_comparison_report(comparison, principal=principal)
            assert unavailable.value.code is ErrorCode.EVALUATION_EVIDENCE_UNAVAILABLE
            with pytest.raises(AIError) as unavailable:
                await evaluations.reconcile(run.experiment_id, principal=principal, idempotency_key="expired-resume")
            assert unavailable.value.code is ErrorCode.EVALUATION_EVIDENCE_UNAVAILABLE
            result = await evaluations.purge_expired(principal=principal, now=future, exclusive=guard)
            assert run.experiment_id in result.evaluations
            assert await persistence_179.object_store(PersistenceDomain.EVALUATION).stat(own.key) is None
            content_rows = await persistence_179._stores[PersistenceDomain.EVALUATION].read(lambda tx: tx.scan_records())
            content_state = repr(tuple(row.data for row in content_rows))
            assert "private score rationale" not in content_state
            assert "private NA explanation" not in content_state
            if dataset_deadline is None:
                assert await evaluations.get_dataset(dataset, principal=principal)
            else:
                with pytest.raises(AIError) as unavailable:
                    await evaluations.get_dataset(dataset, principal=principal)
                assert unavailable.value.code is ErrorCode.EVALUATION_EVIDENCE_UNAVAILABLE
            with pytest.raises(AIError) as unavailable:
                await evaluations.get_report(report.report_id, principal=principal)
            assert unavailable.value.code is ErrorCode.EVALUATION_EVIDENCE_UNAVAILABLE
            with pytest.raises(AIError) as unavailable:
                await evaluations.start(request)
            assert unavailable.value.code is ErrorCode.EVALUATION_EVIDENCE_UNAVAILABLE
            await evaluations.purge_expired(principal=principal,
                now=record.metadata_expires_at + timedelta(seconds=1), exclusive=guard)
            rows = await persistence_179._stores[PersistenceDomain.EVALUATION].read(lambda tx: tx.scan_records())
            persisted = repr(tuple(row.data for row in rows))
            assert "private score rationale" not in persisted
            assert "private rubric" not in persisted
    monkeypatch.undo()
    if backend != "memory":
        async with (
            Persistence.open('evaluation-retention', storage=storage_for(backend, tmp_path, objects), tenant_id='tenant') as persistence_258,
        ):
            async with (
                TaskEngine.open(*tasks, persistence=persistence_258, context=ExecutionContext(None)) as engine,
                Evaluations.open(engine) as reopened_evaluations,
            ):
                with pytest.raises(AIError) as unavailable:
                    await reopened_evaluations.start(request)
                assert unavailable.value.code is ErrorCode.EVALUATION_EVIDENCE_UNAVAILABLE


@pytest.mark.asyncio
async def test_expired_prepared_payload_is_fenced_before_deletion_and_cannot_resume(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from datetime import timedelta
    from linktools.ai.core import service_principal
    from linktools.ai.evaluation import (
        CandidateSpec, CaseRef, CaseSpec, DatasetRef, DatasetSpec, DimensionContract,
        EvaluationPolicy, EvaluationSpec, StartEvaluationRequest, ScorerSpec, TargetTrialRef, GraphTargetSpec, TaskCaseInput,
    )
    from linktools.ai.evaluation_engine._service import Evaluations
    from linktools.ai.persistence._evaluation_records import EvaluationLaunchIntent
    from linktools.ai.task import Task, TaskGraph, TaskGraphTemplate, TaskNode, TaskNodeContext

    called = []

    async def target(context: TaskNodeContext[None]) -> JsonValue:
        called.append(context.input)
        return None

    async def score(context: TaskNodeContext[None]) -> JsonValue:
        return {"dimensions": {"quality": 1.0}}

    principal = service_principal("tenant", "owner")
    storage = PersistenceStorage.filesystem(tmp_path)
    monkeypatch.setattr(Evaluations, "_watch", lambda *args, **kwargs: None)
    async with (
        Persistence.open('evaluation-retention', storage=storage, tenant_id='tenant') as persistence_291,
    ):
        tasks = (Task("retention.prepared", target, effect_policy="none"), Task("retention.score", score, effect_policy="none"))
        async with (
            TaskEngine.open(*tasks, persistence=persistence_291, context=ExecutionContext(None)) as engine,
            Evaluations.open(engine) as evaluations,
        ):
            dataset = await evaluations.publish_dataset(DatasetSpec(DatasetRef("prepared", 1), cases=(
                CaseSpec.graph(CaseRef("prepared", "one", 1), inputs={"target": TaskCaseInput(input={"answer": "source"})}),
            )), principal=principal, idempotency_key="publish")
            template = TaskGraphTemplate((TaskNode("target", task=tasks[0], input={"private": "private template payload"}),))
            run = await evaluations.start(StartEvaluationRequest(EvaluationSpec(dataset,
                (CandidateSpec("target", graph_template=GraphTargetSpec(template=template, outputs={"answer": "target"})),),
                (ScorerSpec("score", tasks[1].ref, dimensions=(DimensionContract("quality", "number", "higher"),)),),
                policy=EvaluationPolicy(content_retention_seconds=60, metadata_retention_seconds=60)), principal, "start"))
            record = await persistence_291.evaluation.records.get(run.experiment_id, tenant_id="tenant")
            declared = TaskGraph("unadmitted", (
                TaskNode("target", task=tasks[0], input={"secret": "unadmitted private prepared payload"}),
            ))
            submission = await engine.prepare_submission(declared, principal=principal, idempotency_key="prepared")
            trial = TargetTrialRef(run.experiment_id, record.manifest.trials[0].trial_id)
            await persistence_291.evaluation.records.register_launch_intent(run.experiment_id,
                EvaluationLaunchIntent("target:" + trial.trial_id, trial, None, submission, None), capacity=1)
            result = await evaluations.purge_expired(principal=principal,
                now=record.content_expires_at + timedelta(seconds=1), exclusive=Offline())
            assert result.evaluations == (run.experiment_id,)
            outcome = await engine.start_prepared(submission)
            assert not outcome.admitted
            repeated = await engine.prepare_submission(declared, principal=principal, idempotency_key="prepared")
            assert not (await engine.start_prepared(repeated)).admitted
            assert called == []
            task_rows = await persistence_291._stores[PersistenceDomain.TASK].read(lambda tx: tx.scan_records())
            assert "unadmitted private prepared payload" not in repr(tuple(row.data for row in task_rows))
            objects = persistence_291.object_store(PersistenceDomain.TASK)
            async for stat in objects.list_objects():
                data = b"".join([chunk async for chunk in objects.open(stat.key)])
                assert b"unadmitted private prepared payload" not in data
                assert b"private template payload" not in data


@pytest.mark.asyncio
async def test_literal_capture_deadline_deletes_owned_copy_without_deleting_borrowed_input(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from datetime import timedelta
    from linktools.ai.core import service_principal
    from linktools.ai.evaluation import CaseRef, CaseSpec, DatasetRef, DatasetSpec
    from linktools.ai.inputs import _input_capture as captures_module
    from linktools.ai.persistence.captures import input_capture_object_dependency

    principal = service_principal("tenant", "owner")
    deadline = datetime.now(timezone.utc) + timedelta(seconds=60)
    storage = PersistenceStorage.filesystem(tmp_path)
    async with (
        Persistence.open('evaluation-retention', storage=storage, tenant_id='tenant') as persistence_341,
        TaskEngine.open(persistence=persistence_341, context=ExecutionContext(None)) as engine,
        Evaluations.open(engine) as evaluations,
    ):
        original = await evaluations.publish_dataset(DatasetSpec(DatasetRef("original", 1), cases=(
            CaseSpec.agent(CaseRef("original", "one", 1), prompt="independently retained source"),
        )), principal=principal, idempotency_key="original")
        borrowed = (await evaluations.list_cases(original, principal=principal)).items[0].input
        spec = DatasetSpec(DatasetRef("expired", 1), cases=(
            CaseSpec.agent(CaseRef("expired", "owned", 1), prompt="private literal copy"),
            CaseSpec.from_capture(CaseRef("expired", "borrowed", 1), capture=borrowed),
        ))
        dataset = await evaluations.publish_dataset(spec, principal=principal,
            idempotency_key="expired", content_expires_at=deadline)
        owned = (await evaluations.list_cases(dataset, principal=principal)).items[0].input
        objects = persistence_341.object_store(PersistenceDomain.TASK)
        owned_key, _ = input_capture_object_dependency(owned)
        borrowed_key, _ = input_capture_object_dependency(borrowed)
        assert await objects.stat(owned_key) is not None
        assert await objects.stat(borrowed_key) is not None

        class ExpiredDateTime(datetime):
            @classmethod
            def now(cls, tz: tzinfo | None = None) -> datetime:
                value = deadline + timedelta(seconds=1)
                return value if tz is not None else value.replace(tzinfo=None)

        with monkeypatch.context() as clock:
            clock.setattr(captures_module, "datetime", ExpiredDateTime)
            with pytest.raises(AIError) as unavailable:
                await evaluations._captures.read_agent(owned, principal=principal)
            assert unavailable.value.code is ErrorCode.INPUT_CAPTURE_UNAVAILABLE
        for unauthorized in (Principal("local", "tenant", "local_trusted"), Principal("owner", "other-tenant", "service")):
            with pytest.raises(AIError) as denied:
                await evaluations.purge_expired(principal=unauthorized,
                    now=deadline + timedelta(seconds=1), exclusive=Offline())
            assert denied.value.code is ErrorCode.AUTHORIZATION_DENIED
            assert await objects.stat(owned_key) is not None
        result = await evaluations.purge_expired(principal=principal,
            now=deadline + timedelta(seconds=1), exclusive=Offline())
        assert dataset in result.datasets
        assert await objects.stat(owned_key) is None
        assert await objects.stat(borrowed_key) is not None
        assert (await evaluations._captures.read_agent(borrowed, principal=principal)).prompt == "independently retained source"
        with pytest.raises(AIError) as unavailable:
            await evaluations.publish_dataset(spec, principal=principal, idempotency_key="changed-publication",
                content_expires_at=deadline + timedelta(days=1))
        assert unavailable.value.code is ErrorCode.EVALUATION_EVIDENCE_UNAVAILABLE
        # The capture owner also blocks direct retries even without the dataset receipt.
        with pytest.raises(AIError) as unavailable:
            await evaluations._compiler.case(spec.cases[0], principal=principal)
        assert unavailable.value.code is ErrorCode.INPUT_CAPTURE_UNAVAILABLE
    async with (
        Persistence.open('evaluation-retention', storage=PersistenceStorage.filesystem(tmp_path), tenant_id='tenant') as persistence_391,
        TaskEngine.open(persistence=persistence_391, context=ExecutionContext(None)) as engine,
        Evaluations.open(engine) as evaluations,
    ):
        with pytest.raises(AIError) as unavailable:
            await evaluations._captures.read_agent(owned, principal=principal)
        assert unavailable.value.code is ErrorCode.INPUT_CAPTURE_UNAVAILABLE
        assert (await evaluations._captures.read_agent(borrowed, principal=principal)).prompt == "independently retained source"
    snapshot = InMemoryObjectStore("snapshot")
    limits = SnapshotLimits(max_entries=10000, max_bytes=10 * 1024 * 1024)
    readonly = PersistenceStorage.filesystem(tmp_path)
    readonly = Persistence(readonly)
    await readonly.initialize(namespace="evaluation-retention", tenant_id="tenant", read_only=True)
    try:
        snapshot_ref = await Snapshot.create(persistence=readonly, exclusive=Offline(), object_store=snapshot, limits=limits)
    finally:
        await readonly.close()
    async for stat in snapshot.list_objects():
        assert b"private literal copy" not in b"".join([chunk async for chunk in snapshot.open(stat.key)])
    restored_root = tmp_path / "restored"
    restored_snapshot = await Snapshot.restore(snapshot_ref, object_store=snapshot, target=restored_root, namespace="evaluation-retention", tenant_id="tenant", limits=limits)
    async with (
        Persistence.open('evaluation-retention', storage=PersistenceStorage.from_root(restored_snapshot.storage_root), tenant_id='tenant') as persistence_411,
        TaskEngine.open(persistence=persistence_411, context=ExecutionContext(None)) as engine,
        Evaluations.open(engine) as evaluations,
    ):
        with pytest.raises(AIError) as unavailable:
            await evaluations._compiler.case(spec.cases[0], principal=principal)
        assert unavailable.value.code is ErrorCode.INPUT_CAPTURE_UNAVAILABLE
        with pytest.raises(AIError) as unavailable:
            await evaluations.publish_dataset(spec, principal=principal, idempotency_key="restored-publication",
                content_expires_at=deadline + timedelta(days=1))
        assert unavailable.value.code is ErrorCode.EVALUATION_EVIDENCE_UNAVAILABLE
        assert (await evaluations._captures.read_agent(borrowed, principal=principal)).prompt == "independently retained source"


@pytest.mark.asyncio
async def test_cleanup_receipt_survives_failed_physical_delete_and_retry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from dataclasses import replace
    from datetime import timedelta
    from linktools.ai.core import service_principal
    from linktools.ai.evaluation import EvidenceAttachmentRef, EvidenceBundle, EvidenceRef, ExecutionSubjectRef, ExecutionTargetEvidence, TargetTrialRef
    from linktools.ai.evaluation_engine._retention import EvaluationRetention
    from tests.ai.test_evaluation_persistence import experiment

    class Authorization:
        async def authorize(self, principal: Principal, action: AuthorizationAction, resource: ResourceRef) -> None:
            return None

    class Captures:
        async def expire_inputs(self, *, principal: Principal, now: datetime, limit: int) -> tuple[ObjectRef, ...]:
            return ()

    async def cancel_submission(record, intent, principal, idempotency_key):
        raise AssertionError("no launch intent exists")

    guard = Offline()
    objects = CheckedObjects(guard)
    storage = PersistenceStorage.filesystem(tmp_path, object_store=objects)
    storage = Persistence(storage)
    await storage.initialize(namespace="evaluation", tenant_id="tenant")
    now = datetime.now(timezone.utc)
    record = replace(experiment(), content_expires_at=now + timedelta(seconds=60), metadata_expires_at=now + timedelta(seconds=60))
    try:
        await storage.evaluation.records.reserve_experiment(record)
        reference = await put(objects, "private-evidence", "private content")
        bundle = EvidenceBundle(EvidenceRef("evaluation", "tenant", "bundle", "0" * 64),
            TargetTrialRef(record.experiment_id, "trial"),
            ExecutionTargetEvidence(ExecutionSubjectRef("evaluation", "tenant", "source"), "succeeded"),
            attachments=(EvidenceAttachmentRef("raw", "text/plain", reference),))
        bundle = replace(bundle, ref=replace(bundle.ref, digest=bundle.digest))
        await storage.evaluation.records.publish_evidence(bundle)
        retention = EvaluationRetention(Authorization(), cancel_submission, Captures(),
            tenant_id=storage.tenant_id, repository=storage.evaluation.records,
            task_objects=storage.object_store(PersistenceDomain.TASK), maintenance=storage.snapshot_operations())
        original_delete = objects.delete_object

        async def fail_delete(key: str, *, expected_digest: str) -> bool:
            raise OSError("simulated object store outage")

        monkeypatch.setattr(objects, "delete_object", fail_delete)
        with pytest.raises(OSError):
            await retention.purge_expired(principal=service_principal("tenant", "owner"),
                now=now + timedelta(seconds=120), exclusive=guard)
        with pytest.raises(AIError) as unavailable:
            await storage.evaluation.records.get(record.experiment_id, tenant_id="tenant")
        assert unavailable.value.code is ErrorCode.EVALUATION_EVIDENCE_UNAVAILABLE
        assert await objects.stat(reference.key) is not None
        assert await storage.evaluation.records.pending_cleanup(owner_principal_id="owner", limit=100)
        monkeypatch.setattr(objects, "delete_object", original_delete)
    finally:
        await storage.close()
    reopened = PersistenceStorage.filesystem(tmp_path, object_store=objects)
    reopened = Persistence(reopened)
    await reopened.initialize(namespace="evaluation", tenant_id="tenant")
    try:
        result = await EvaluationRetention(Authorization(), cancel_submission, Captures(),
            tenant_id=reopened.tenant_id, repository=reopened.evaluation.records,
            task_objects=reopened.object_store(PersistenceDomain.TASK), maintenance=reopened.snapshot_operations()).purge_expired(
            principal=service_principal("tenant", "owner"), now=now + timedelta(seconds=120), exclusive=guard)
        assert result.objects_deleted == 1
        assert await objects.stat(reference.key) is None
        assert await reopened.evaluation.records.pending_cleanup(owner_principal_id="owner", limit=100) == ()
    finally:
        await reopened.close()
