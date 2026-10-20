#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Snapshot-local evidence reads keep current authorization and retention checks."""

from linktools.ai.runtime.state import RuntimeState
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from sqlalchemy import event

from linktools.ai.runtime import RuntimeFeatures
from linktools.ai.core import AuthorizationAction, Principal, ResourceRef, TenantAuthorizationPolicy
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.evaluation import (
    CandidateSpec, CaseRef, CaseSpec, DatasetRef, DatasetSpec, EvaluationSpec,
    EvidenceAttachmentRef, RescoreRequest, StartEvaluationRequest,
)
from linktools.ai.model import ModelRegistry
from linktools.ai.runtime import EvaluationRun, Runtime, RuntimeStorage
from linktools.ai.runtime import _evaluation as evaluation_module
from linktools.ai.runtime._object import RuntimeObjectKeyFactory, put_runtime_object
from linktools.ai.runtime.state import RuntimeDomain
from linktools.ai.runtime.state._store import RecordQuery
from linktools.ai.task import Task

from .test_evaluation_consumers import (
    CONTEXT, EVALUATION_COMPLETION_TIMEOUT_SECONDS, PRINCIPAL, echo, exact, rule_scorer,
)


class Policy:
    def __init__(self) -> None:
        self.default = TenantAuthorizationPolicy(PRINCIPAL.tenant_id)
        self.calls: list[tuple[AuthorizationAction, ResourceRef]] = []
        self.reject: Callable[[AuthorizationAction, ResourceRef], bool] = lambda action, resource: False

    async def authorize(self, principal: Principal, action: AuthorizationAction, resource: ResourceRef) -> None:
        self.calls.append((action, resource))
        if self.reject(action, resource):
            raise AIError(ErrorCode.AUTHORIZATION_DENIED)
        await self.default.authorize(principal, action, resource)


@asynccontextmanager
async def completed(path: Path, *, count: int = 2) -> AsyncIterator[tuple[Runtime, RuntimeStorage, EvaluationRun, Task, Policy]]:
    storage = RuntimeStorage.sqlite(path / "state.sqlite")
    policy = Policy()
    async with Runtime.open("evidence-read", storage=storage, models=ModelRegistry(),
                            context=CONTEXT, features=RuntimeFeatures("authorization", authorization=policy)) as runtime:
        target = Task("evidence-read.target", echo, effect_policy="none")
        scorer = Task("evidence-read.score", exact, effect_policy="none")
        dataset = await runtime.evaluations.publish_dataset(DatasetSpec(DatasetRef("dataset", 1), tuple(
            CaseSpec.task(CaseRef("dataset", f"case-{i}", 1), input={"answer": "yes"}, expected="yes")
            for i in range(count))), principal=PRINCIPAL, idempotency_key="dataset")
        run = await runtime.evaluations.start(StartEvaluationRequest(EvaluationSpec(dataset,
            (CandidateSpec("candidate", task=target.ref),), (rule_scorer(scorer),)), PRINCIPAL, "run"),
            engine=runtime.tasks.bind(target, scorer))
        assert (await run.wait(timeout_seconds=EVALUATION_COMPLETION_TIMEOUT_SECONDS)).result.completion == "complete"
        policy.calls.clear()
        yield runtime, runtime._state, run, scorer, policy


async def report_count(storage: RuntimeState) -> int:
    return len(await storage.evaluation.records.state_store.read(
        lambda tx: tx.list_records(RecordQuery(kind="evaluation_report"))))


@pytest.mark.asyncio
async def test_public_read_evidence_fetches_its_source_once_on_every_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with completed(tmp_path, count=1) as (runtime, storage, run, _scorer, policy):
        await runtime.evaluations.close()
        repository = storage.evaluation.records
        record = await repository.get(run.experiment_id, tenant_id=PRINCIPAL.tenant_id)
        get = repository.get
        calls = []

        async def capture(identity, *, tenant_id):
            calls.append(identity)
            return await get(identity, tenant_id=tenant_id)

        monkeypatch.setattr(repository, "get", capture)
        for expected_reads in (1, 2):
            bundle = await runtime.evaluations.read_evidence(record.evidence[0].evidence_ref, principal=PRINCIPAL)
            assert bundle.trial.target_experiment_id == run.experiment_id
            assert calls == [run.experiment_id] * expected_reads
        policy.reject = lambda action, resource: action is AuthorizationAction.EVALUATION_EVIDENCE_READ
        with pytest.raises(AIError) as caught:
            await runtime.evaluations.read_evidence(record.evidence[0].evidence_ref, principal=PRINCIPAL)
        assert caught.value.code is ErrorCode.AUTHORIZATION_DENIED
        assert len(calls) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("publish", (False, True))
async def test_snapshot_rechecks_authorization_after_all_evidence_reads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, publish: bool,
) -> None:
    async with completed(tmp_path, count=1) as (runtime, storage, run, _scorer, policy):
        await runtime.evaluations.close()
        get_cases = storage.evaluation.records.get_cases
        revoked = False

        async def revoke(refs):
            nonlocal revoked
            cases = await get_cases(refs)
            revoked = True
            policy.reject = lambda action, resource: (
                action is AuthorizationAction.EVALUATION_EVIDENCE_READ
                and resource.id == run.experiment_id
                and resource.owner_principal_id == PRINCIPAL.principal_id)
            return cases

        monkeypatch.setattr(storage.evaluation.records, "get_cases", revoke)
        with pytest.raises(AIError) as caught:
            await (run.create_report() if publish else run.preview_report())
        assert revoked and caught.value.code is ErrorCode.AUTHORIZATION_DENIED
        assert await report_count(storage) == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("rescore", (False, True))
async def test_snapshot_full_source_reads_are_bounded_and_source_owner_is_preserved(
    tmp_path: Path, rescore: bool,
) -> None:
    async with completed(tmp_path, count=3) as (runtime, storage, run, scorer, policy):
        source_id = run.experiment_id
        if rescore:
            run = await run.rescore(RescoreRequest((rule_scorer(scorer),), "rescore"),
                                   engine=runtime.tasks.bind(scorer))
            assert (await run.wait(timeout_seconds=EVALUATION_COMPLETION_TIMEOUT_SECONDS)).result.completion == "complete"
        await runtime.evaluations.close()
        source = await storage.evaluation.records.get(source_id, tenant_id=PRINCIPAL.tenant_id)
        source_key = storage.evaluation.records._key("evaluation", source_id).hex()
        statements = []

        def capture(_connection, _cursor, sql, parameters, _context, _many):
            if sql.startswith("SELECT") and source_key in parameters:
                statements.append(sql)

        engine = storage.evaluation.records.state_store.context.engine.sync_engine
        event.listen(engine, "before_cursor_execute", capture)
        policy.calls.clear()
        try:
            report = await run.preview_report()
        finally:
            event.remove(engine, "before_cursor_execute", capture)
        assert 1 <= len(statements) <= (6 if rescore else 2), "source reads must not grow with evidence count"
        assert len(report.trials) == 3 and report.scores[0].mean == 1.0
        assert await report_count(storage) == 0
        evidence_ids = {link.evidence_ref.evidence_id for link in source.evidence}
        evidence_calls = [(action, resource) for action, resource in policy.calls
                          if action is AuthorizationAction.EVALUATION_EVIDENCE_READ]
        assert {resource.id for _, resource in evidence_calls if resource.id in evidence_ids} == evidence_ids
        source_calls = [resource for _, resource in evidence_calls if resource.id == source_id]
        assert sum(resource.owner_principal_id is None for resource in source_calls) >= len(evidence_ids)
        assert sum(resource.owner_principal_id == PRINCIPAL.principal_id for resource in source_calls) >= len(evidence_ids)
        if rescore:
            assert all(resource.id != run.experiment_id for _, resource in evidence_calls)


@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", ("evidence", "source", "owner"))
async def test_snapshot_checks_each_evidence_policy_after_revocation(tmp_path: Path, boundary: str) -> None:
    async with completed(tmp_path) as (runtime, storage, run, _scorer, policy):
        await runtime.evaluations.close()
        record = await storage.evaluation.records.get(run.experiment_id, tenant_id=PRINCIPAL.tenant_id)
        second = record.evidence[1].evidence_ref.evidence_id
        revoked = False

        def reject(action, resource):
            nonlocal revoked
            if action is not AuthorizationAction.EVALUATION_EVIDENCE_READ:
                return False
            if resource.id == second:
                revoked = True
            return revoked and (
                boundary == "evidence" and resource.id == second
                or boundary == "source" and resource.id == run.experiment_id and resource.owner_principal_id is None
                or boundary == "owner" and resource.id == run.experiment_id and resource.owner_principal_id == PRINCIPAL.principal_id)

        policy.reject = reject
        with pytest.raises(AIError) as caught:
            await run.create_report()
        assert caught.value.code is ErrorCode.AUTHORIZATION_DENIED
        assert await report_count(storage) == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("deadline_field", ("metadata_expires_at", "content_expires_at"))
async def test_snapshot_checks_expiry_clock_for_each_cached_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, deadline_field: str,
) -> None:
    async with completed(tmp_path) as (runtime, storage, run, _scorer, _policy):
        await runtime.evaluations.close()
        now = datetime.now(timezone.utc)
        deadline = now + timedelta(hours=1)
        repository = storage.evaluation.records
        await repository.update(run.experiment_id, lambda record: replace(record, **{deadline_field: deadline}))
        read = repository.read_evidence
        calls = 0

        async def expire(ref):
            nonlocal calls, now
            bundle = await read(ref)
            calls += 1
            if calls == 2:
                now = deadline
            return bundle

        monkeypatch.setattr(repository, "read_evidence", expire)
        monkeypatch.setattr(evaluation_module, "_now", lambda: now)
        with pytest.raises(AIError) as caught:
            await run.create_report()
        assert caught.value.code is ErrorCode.EVALUATION_EVIDENCE_UNAVAILABLE
        assert calls == 2
        assert await report_count(storage) == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("mutation", ("purge", "missing", "revision"))
async def test_snapshot_revalidates_sources_before_releasing_content(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str,
) -> None:
    async with completed(tmp_path) as (runtime, storage, run, _scorer, _policy):
        await runtime.evaluations.close()
        repository = storage.evaluation.records
        initial = await repository.get(run.experiment_id, tenant_id=PRINCIPAL.tenant_id)
        read, get = repository.read_evidence, repository.get
        changed = False

        async def mutate(ref):
            nonlocal changed
            bundle = await read(ref)
            if not changed:
                changed = True
                if mutation == "purge":
                    await repository.update(run.experiment_id, lambda record: replace(
                        record, content_deleted_at=datetime.now(timezone.utc)))
                elif mutation == "revision":
                    await repository.close_reservation_gate(run.experiment_id, budget=True)
            return bundle

        async def missing(identity, *, tenant_id):
            if changed and mutation == "missing" and identity == run.experiment_id:
                return None
            return await get(identity, tenant_id=tenant_id)

        monkeypatch.setattr(repository, "read_evidence", mutate)
        monkeypatch.setattr(repository, "get", missing)
        if mutation == "revision":
            report = await run.preview_report()
            latest = await get(run.experiment_id, tenant_id=PRINCIPAL.tenant_id)
            assert latest.revision > initial.revision
            assert report.cutoff.state_revisions[f"evaluation:{run.experiment_id}"] == initial.revision
            assert report.completion == "complete" and report.scores[0].mean == 1.0
        else:
            with pytest.raises(AIError) as caught:
                await run.create_report()
            assert caught.value.code is (ErrorCode.STORAGE_NOT_FOUND if mutation == "missing"
                                          else ErrorCode.EVALUATION_EVIDENCE_UNAVAILABLE)
            assert await report_count(storage) == 0
            # Purge starts by revoking content, before the retained bundle is physically removed.
            assert await read(initial.evidence[0].evidence_ref) is not None


@pytest.mark.asyncio
async def test_snapshot_keeps_evidence_integrity_checks_with_reused_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with completed(tmp_path) as (runtime, storage, run, _scorer, _policy):
        await runtime.evaluations.close()
        repository = storage.evaluation.records
        read = repository.read_evidence
        calls = 0

        async def corrupt(ref):
            nonlocal calls
            bundle = await read(ref)
            calls += 1
            return replace(bundle, usage_complete=not bundle.usage_complete) if calls == 2 else bundle

        monkeypatch.setattr(repository, "read_evidence", corrupt)
        with pytest.raises(AIError) as caught:
            await run.create_report()
        assert caught.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
        assert await report_count(storage) == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("missing", (False, True))
async def test_snapshot_checks_retained_attachment_integrity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, missing: bool,
) -> None:
    async with completed(tmp_path, count=1) as (runtime, storage, run, _scorer, _policy):
        await runtime.evaluations.close()
        repository = storage.evaluation.records
        record = await repository.get(run.experiment_id, tenant_id=PRINCIPAL.tenant_id)
        bundle = await repository.read_evidence(record.evidence[0].evidence_ref)
        objects = storage.object_store(RuntimeDomain.EVALUATION)
        reference = await put_runtime_object(objects, RuntimeObjectKeyFactory("evidence-read"),
            RuntimeDomain.EVALUATION, PRINCIPAL.tenant_id, b"retained attachment")
        bundle = replace(bundle, ref=replace(bundle.ref, evidence_id="with-attachment"),
            attachments=(EvidenceAttachmentRef("raw", "text/plain", reference),))
        bundle = replace(bundle, ref=replace(bundle.ref, digest=bundle.digest))
        await repository.publish_evidence(bundle)
        await repository.update(run.experiment_id, lambda value: replace(value,
            evidence=(replace(value.evidence[0], evidence_ref=bundle.ref),)))
        stat = objects.stat
        checked = False

        async def changed(key):
            nonlocal checked
            result = await stat(key)
            if key != reference.key:
                return result
            checked = True
            return None if missing else replace(result, size=result.size + 1)

        monkeypatch.setattr(objects, "stat", changed)
        with pytest.raises(AIError) as caught:
            await run.create_report()
        assert checked
        assert caught.value.code is (ErrorCode.INPUT_CAPTURE_UNAVAILABLE if missing else ErrorCode.STORAGE_INTEGRITY_ERROR)
        assert await report_count(storage) == 0
