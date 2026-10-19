#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Current evaluation reads select manifest members before native hydration."""

from linktools.ai.runtime.state import RuntimeState
from collections.abc import AsyncIterator, Iterator
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from sqlalchemy import event

from linktools.ai.capability import RuntimeFeatures
from linktools.ai.core import (
    AuthorizationAction, ExecutionStatus, Principal, ResourceRef, TaskStatus, service_principal,
)
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.evaluation import (
    CandidateSpec, CaseRef, CaseSpec, DatasetRef, DatasetSpec, EvaluationSpec,
    EvidenceAttachmentRef, EvidenceBundle, EvidenceRef, RescoreRequest, StartEvaluationRequest,
)
from linktools.ai.model import ModelRegistry
from linktools.ai.runtime import ExecutionView, Runtime, RuntimeStorage
from linktools.ai.runtime import _evaluation as evaluation_module
from linktools.ai.runtime._object import RuntimeObjectKeyFactory, put_runtime_object
from linktools.ai.runtime.state import RuntimeDomain
from linktools.ai.runtime.state import _evaluation_repository as repository_module
from linktools.ai.runtime.state._evaluation_records import EvaluationCaseRecord
from linktools.ai.storage import ObjectRef, ObjectStat
from linktools.ai.task import Task, TaskGraphState

from .test_evaluation_consumers import (
    CONTEXT, EVALUATION_COMPLETION_TIMEOUT_SECONDS, PRINCIPAL, echo, exact, rule_scorer,
)
from .test_evaluation_evidence_read_reuse import Policy, completed, report_count
from .test_evaluation_retention import Offline


@contextmanager
def _sql_writes(storage: RuntimeState) -> Iterator[list[str]]:
    statements: list[str] = []

    def capture(_connection, _cursor, sql, _parameters, _context, _many) -> None:
        if sql.lstrip().split(None, 1)[0].upper() in {"INSERT", "UPDATE", "DELETE", "REPLACE"}:
            statements.append(sql)

    engine = storage.evaluation.records.state_store.context.engine.sync_engine
    event.listen(engine, "before_cursor_execute", capture)
    try:
        yield statements
    finally:
        event.remove(engine, "before_cursor_execute", capture)


async def _add_attachment(storage: RuntimeState, experiment_id: str, index: int) -> ObjectRef:
    repository = storage.evaluation.records
    record = await repository.get(experiment_id, tenant_id=PRINCIPAL.tenant_id)
    trial_id = record.manifest.trials[index].trial_id
    link = next(item for item in record.evidence if item.trial.trial_id == trial_id)
    bundle = await repository.read_evidence(link.evidence_ref)
    reference = await put_runtime_object(storage.object_store(RuntimeDomain.EVALUATION),
        RuntimeObjectKeyFactory("evidence-read"), RuntimeDomain.EVALUATION,
        PRINCIPAL.tenant_id, b"retained current-read attachment")
    bundle = replace(bundle, ref=replace(bundle.ref, evidence_id=f"attachment-{index}"),
        attachments=(EvidenceAttachmentRef("raw", "text/plain", reference),))
    bundle = replace(bundle, ref=replace(bundle.ref, digest=bundle.digest))
    await repository.publish_evidence(bundle)
    await repository.update(experiment_id, lambda value: replace(value,
        evidence=tuple(replace(item, evidence_ref=bundle.ref) if item.trial == link.trial else item
                       for item in value.evidence)))
    return reference


@pytest.mark.asyncio
@pytest.mark.parametrize("count", (2, 3))
async def test_current_pages_hydrate_only_the_manifest_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, count: int,
) -> None:
    async with completed(tmp_path, count=count) as (runtime, storage, run, _scorer, _policy):
        await runtime.evaluations.close()
        owner = runtime.evaluations
        record = await storage.evaluation.records.get(run.experiment_id, tenant_id=PRINCIPAL.tenant_id)
        state, inspect = owner._graph.state, owner._execution.inspect
        graph_reads: list[str] = []
        execution_reads: list[str] = []

        async def read_graph(graph_id: str, *, principal: Principal) -> TaskGraphState:
            graph_reads.append(graph_id)
            return await state(graph_id, principal=principal)

        async def read_execution(execution_id: str, *, principal: Principal) -> ExecutionView:
            execution_reads.append(execution_id)
            return await inspect(execution_id, principal=principal)

        async def no_object_write(
            key: str, chunks: AsyncIterator[bytes], *, expected_size: int, expected_digest: str,
        ) -> ObjectStat:
            pytest.fail("current pages must not write an object or snapshot")

        monkeypatch.setattr(owner._graph, "state", read_graph)
        monkeypatch.setattr(owner._execution, "inspect", read_execution)
        monkeypatch.setattr(owner._objects, "put", no_object_write)
        cursor = None
        seen = []
        with _sql_writes(storage) as writes:
            for index, plan in enumerate(record.manifest.trials):
                graph_reads.clear()
                execution_reads.clear()
                page = await run.current_trials(cursor=cursor, limit=1)
                assert len(page.items) == 1
                item = page.items[0]
                assert item.trial.trial_id == plan.trial_id and item.case_ref == plan.case_ref
                assert item.graph_ref is not None and item.subject is not None
                assert graph_reads == [item.graph_ref.graph_id]
                assert execution_reads == [item.subject.execution_id]
                assert (page.next_cursor is not None) == (index + 1 < count)
                seen.append(item)
                cursor = page.next_cursor
        assert not writes
        assert len(seen) == count and await report_count(storage) == 0
        assert await storage.evaluation.records.get(run.experiment_id, tenant_id=PRINCIPAL.tenant_id) == record


@pytest.mark.asyncio
async def test_current_reads_reload_native_recovery_state_without_updating_evaluation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with completed(tmp_path) as (runtime, storage, run, _scorer, _policy):
        await runtime.evaluations.close()
        owner = runtime.evaluations
        record = await storage.evaluation.records.get(run.experiment_id, tenant_id=PRINCIPAL.tenant_id)
        first = await run.current_trials(limit=1)
        original = first.items[0]
        state, inspect = owner._graph.state, owner._execution.inspect

        async def recovering_graph(graph_id: str, *, principal: Principal) -> TaskGraphState:
            value = await state(graph_id, principal=principal)
            return replace(value, status=TaskStatus.RECOVERY_REQUIRED, event_seq=value.event_seq + 1)

        async def recovering_execution(execution_id: str, *, principal: Principal) -> ExecutionView:
            value = await inspect(execution_id, principal=principal)
            return replace(value, status=ExecutionStatus.RECOVERY_REQUIRED, event_seq=value.event_seq + 1)

        monkeypatch.setattr(owner._graph, "state", recovering_graph)
        monkeypatch.setattr(owner._execution, "inspect", recovering_execution)
        refreshed = (await run.current_trials(limit=1)).items[0]
        direct = await run.read_trial(original.trial.trial_id)
        assert refreshed == direct
        assert original.graph_status is TaskStatus.SUCCEEDED
        assert refreshed.graph_status is TaskStatus.RECOVERY_REQUIRED
        assert refreshed.execution_status is ExecutionStatus.RECOVERY_REQUIRED
        assert refreshed.trial == original.trial and refreshed.evidence_ref == original.evidence_ref
        assert await storage.evaluation.records.get(run.experiment_id, tenant_id=PRINCIPAL.tenant_id) == record
        assert await report_count(storage) == 0


@pytest.mark.asyncio
async def test_current_cursor_survives_mutable_revision_and_reads_new_native_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with completed(tmp_path) as (runtime, storage, run, _scorer, _policy):
        await runtime.evaluations.close()
        repository = storage.evaluation.records
        before = await repository.get(run.experiment_id, tenant_id=PRINCIPAL.tenant_id)
        first = await run.current_trials(limit=1)
        after = await repository.close_reservation_gate(run.experiment_id, budget=True)
        assert after.revision > before.revision and after.manifest_digest == before.manifest_digest
        state = runtime.evaluations._graph.state

        async def recovering(graph_id: str, *, principal: Principal) -> TaskGraphState:
            return replace(await state(graph_id, principal=principal), status=TaskStatus.RECOVERY_REQUIRED)

        monkeypatch.setattr(runtime.evaluations._graph, "state", recovering)
        following = await run.current_trials(cursor=first.next_cursor, limit=1)
        assert following.items[0].trial.trial_id == before.manifest.trials[1].trial_id
        assert following.items[0].graph_status is TaskStatus.RECOVERY_REQUIRED
        assert following.next_cursor is None and await report_count(storage) == 0


@pytest.mark.asyncio
async def test_current_cursor_rejects_wrong_scope_manifest_and_position(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with completed(tmp_path) as (runtime, storage, run, scorer, policy):
        other = await run.rescore(RescoreRequest((rule_scorer(scorer),), "cursor-other"),
                                 engine=runtime.tasks.bind(scorer))
        assert (await other.wait(timeout_seconds=EVALUATION_COMPLETION_TIMEOUT_SECONDS)).result.completion == "complete"
        await runtime.evaluations.close()
        owner = runtime.evaluations
        first = await run.current_trials(limit=1)
        signer = owner._cursor_signer
        payload = signer.decode(first.next_cursor)
        following = await run.current_trials(cursor=first.next_cursor, limit=2)
        assert len(following.items) == 1 and following.next_cursor is None
        assert following.items[0].trial != first.items[0].trial

        async def allow(principal: Principal, action: AuthorizationAction, resource: ResourceRef) -> None:
            assert principal.tenant_id == PRINCIPAL.tenant_id

        monkeypatch.setattr(policy, "authorize", allow)
        for experiment_id, principal in (
            (run.experiment_id, service_principal(PRINCIPAL.tenant_id, "other-viewer")),
            (other.experiment_id, PRINCIPAL),
        ):
            with pytest.raises(AIError) as caught:
                await owner.current_trials(experiment_id, principal=principal, cursor=first.next_cursor, limit=1)
            assert caught.value.code is ErrorCode.CURSOR_INVALID
        for changed in (
            replace(payload, filter_digest="0" * 64),
            replace(payload, tenant_id="another-tenant"),
            replace(payload, resource_kind="evaluation.trials"),
            *(replace(payload, position=position) for position in ("-1", "2", "3", "not-an-index")),
        ):
            with pytest.raises(AIError) as caught:
                await run.current_trials(cursor=signer.encode(changed), limit=1)
            assert caught.value.code is ErrorCode.CURSOR_INVALID
        with monkeypatch.context() as changed_namespace:
            changed_namespace.setattr(owner, "_namespace", "another-namespace")
            with pytest.raises(AIError) as caught:
                await run.current_trials(cursor=first.next_cursor, limit=1)
            assert caught.value.code is ErrorCode.CURSOR_INVALID
        assert await report_count(storage) == 0


@pytest.mark.asyncio
async def test_rescore_current_pages_use_only_selected_source_trials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with completed(tmp_path, count=3) as (runtime, storage, source, scorer, policy):
        repository = storage.evaluation.records
        record = await repository.get(source.experiment_id, tenant_id=PRINCIPAL.tenant_id)
        plans = record.manifest.trials
        run = await source.rescore(RescoreRequest((rule_scorer(scorer),), "selected-rescore",
            trial_ids=(plans[2].trial_id, plans[0].trial_id)), engine=runtime.tasks.bind(scorer))
        assert (await run.wait(timeout_seconds=EVALUATION_COMPLETION_TIMEOUT_SECONDS)).result.completion == "complete"
        await runtime.evaluations.close()
        selected = await repository.get(run.experiment_id, tenant_id=PRINCIPAL.tenant_id)
        state = runtime.evaluations._graph.state
        reads: list[str] = []

        async def read_graph(graph_id: str, *, principal: Principal) -> TaskGraphState:
            reads.append(graph_id)
            return await state(graph_id, principal=principal)

        monkeypatch.setattr(runtime.evaluations._graph, "state", read_graph)
        policy.calls.clear()
        first = await run.current_trials(limit=1)
        assert reads == [first.items[0].graph_ref.graph_id]
        reads.clear()
        second = await run.current_trials(cursor=first.next_cursor, limit=1)
        assert reads == [second.items[0].graph_ref.graph_id]
        assert tuple(item.trial for item in (*first.items, *second.items)) == selected.manifest.source_trials
        assert [item.trial.trial_id for item in (*first.items, *second.items)] == [plans[0].trial_id, plans[2].trial_id]
        assert second.next_cursor is None
        assert await run.read_trial(plans[2].trial_id) == second.items[0]
        reads.clear()
        with pytest.raises(AIError) as caught:
            await run.read_trial(plans[1].trial_id)
        assert caught.value.code is ErrorCode.STORAGE_NOT_FOUND and not reads
        source_checks = [resource for action, resource in policy.calls
                         if action is AuthorizationAction.EVALUATION_EVIDENCE_READ
                         and resource.owner_principal_id is not None]
        assert source_checks and all(resource.id == source.experiment_id for resource in source_checks)
        assert all(resource.owner_principal_id == PRINCIPAL.principal_id for resource in source_checks)
        assert await report_count(storage) == 0


@pytest.mark.asyncio
async def test_direct_trial_and_case_reads_do_not_scan_pages_or_case_content(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with completed(tmp_path) as (runtime, storage, run, _scorer, _policy):
        await runtime.evaluations.close()
        owner = runtime.evaluations
        record = await storage.evaluation.records.get(run.experiment_id, tenant_id=PRINCIPAL.tenant_id)
        plan = record.manifest.trials[1]

        async def no_scan(*_args: object, **_kwargs: object) -> None:
            pytest.fail("single-resource reads must not scan a page or dataset")

        for name in ("current_trials", "list_cases", "_build_snapshot"):
            monkeypatch.setattr(owner, name, no_scan)
        monkeypatch.setattr(storage.evaluation.records, "get_cases", no_scan)
        state = owner._graph.state
        reads: list[str] = []

        async def selected_graph(graph_id: str, *, principal: Principal) -> TaskGraphState:
            reads.append(graph_id)
            return await state(graph_id, principal=principal)

        monkeypatch.setattr(owner._graph, "state", selected_graph)
        with _sql_writes(storage) as writes:
            trial = await run.read_trial(plan.trial_id)
            assert trial == await owner.read_trial(run.experiment_id, plan.trial_id, principal=PRINCIPAL)
            with monkeypatch.context() as case_read:
                case_read.setattr(owner, "get_dataset", no_scan)
                case = await owner.get_case(plan.case_ref, principal=PRINCIPAL)
            assert case.ref == plan.case_ref
            assert reads == [trial.graph_ref.graph_id] * 2
            reads.clear()
            with pytest.raises(AIError) as caught:
                await run.read_trial("unknown-trial")
            assert caught.value.code is ErrorCode.STORAGE_NOT_FOUND and not reads
        assert not writes and await report_count(storage) == 0


@pytest.mark.asyncio
async def test_current_reads_preserve_frozen_trial_and_score_pages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with completed(tmp_path) as (runtime, storage, run, _scorer, _policy):
        await runtime.evaluations.close()
        frozen_trials = await run.trials(limit=1)
        frozen_scores = await run.scores(limit=1)
        assert await report_count(storage) == 2
        current = await run.current_trials(limit=1)
        state = runtime.evaluations._graph.state

        async def recovering(graph_id: str, *, principal: Principal) -> TaskGraphState:
            return replace(await state(graph_id, principal=principal), status=TaskStatus.RECOVERY_REQUIRED)

        monkeypatch.setattr(runtime.evaluations._graph, "state", recovering)
        live = await run.current_trials(cursor=current.next_cursor, limit=1)
        frozen = await run.trials(cursor=frozen_trials.next_cursor, limit=2)
        scores = await run.scores(cursor=frozen_scores.next_cursor, limit=2)
        assert live.items[0].graph_status is TaskStatus.RECOVERY_REQUIRED
        assert frozen.items[0].graph_status is TaskStatus.SUCCEEDED and scores.items[0].status == "valid"
        for read, cursor in ((run.current_trials, frozen_trials.next_cursor), (run.trials, current.next_cursor)):
            with pytest.raises(AIError) as caught:
                await read(cursor=cursor, limit=1)
            assert caught.value.code is ErrorCode.CURSOR_INVALID
        assert await report_count(storage) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", ("evidence", "source", "owner"))
async def test_current_reads_enforce_selected_evidence_authorization(
    tmp_path: Path, boundary: str,
) -> None:
    async with completed(tmp_path) as (runtime, storage, run, _scorer, policy):
        await runtime.evaluations.close()
        record = await storage.evaluation.records.get(run.experiment_id, tenant_id=PRINCIPAL.tenant_id)
        first = next(item for item in record.evidence
                     if item.trial.trial_id == record.manifest.trials[0].trial_id)
        policy.reject = lambda action, resource: action is AuthorizationAction.EVALUATION_EVIDENCE_READ and (
            boundary == "evidence" and resource.id == first.evidence_ref.evidence_id
            or boundary == "source" and resource.id == run.experiment_id and resource.owner_principal_id is None
            or boundary == "owner" and resource.id == run.experiment_id and resource.owner_principal_id == PRINCIPAL.principal_id)
        for read in (lambda: run.current_trials(limit=1), lambda: run.read_trial(first.trial.trial_id)):
            with pytest.raises(AIError) as caught:
                await read()
            assert caught.value.code is ErrorCode.AUTHORIZATION_DENIED
        assert await report_count(storage) == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("field", ("metadata_expires_at", "content_expires_at"))
async def test_current_reads_reject_expired_evidence_sources(tmp_path: Path, field: str) -> None:
    async with completed(tmp_path) as (runtime, storage, run, _scorer, _policy):
        await runtime.evaluations.close()
        record = await storage.evaluation.records.update(run.experiment_id, lambda value: replace(
            value, **{field: datetime.now(timezone.utc) - timedelta(seconds=1)}))
        for read in (lambda: run.current_trials(limit=1),
                     lambda: run.read_trial(record.manifest.trials[0].trial_id)):
            with pytest.raises(AIError) as caught:
                await read()
            assert caught.value.code is ErrorCode.EVALUATION_EVIDENCE_UNAVAILABLE
        assert await report_count(storage) == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", ("digest", "attachment_missing", "attachment_corrupt"))
async def test_current_reads_validate_selected_evidence_and_attachment_integrity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str,
) -> None:
    async with completed(tmp_path) as (runtime, storage, run, _scorer, _policy):
        await runtime.evaluations.close()
        repository = storage.evaluation.records
        if fault == "digest":
            read = repository.read_evidence

            async def corrupt(ref: EvidenceRef) -> EvidenceBundle:
                bundle = await read(ref)
                return replace(bundle, usage_complete=not bundle.usage_complete)

            monkeypatch.setattr(repository, "read_evidence", corrupt)
        else:
            reference = await _add_attachment(storage, run.experiment_id, 0)
            objects = storage.object_store(RuntimeDomain.EVALUATION)
            stat = objects.stat

            async def changed(key: str) -> ObjectStat | None:
                result = await stat(key)
                if key != reference.key:
                    return result
                return None if fault == "attachment_missing" else replace(result, size=result.size + 1)

            monkeypatch.setattr(objects, "stat", changed)
        with pytest.raises(AIError) as caught:
            await run.current_trials(limit=1)
        assert caught.value.code is (ErrorCode.INPUT_CAPTURE_UNAVAILABLE if fault == "attachment_missing"
                                      else ErrorCode.STORAGE_INTEGRITY_ERROR)
        assert await report_count(storage) == 0


@pytest.mark.asyncio
async def test_unselected_bad_attachment_does_not_block_current_window_or_direct_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with completed(tmp_path) as (runtime, storage, run, _scorer, _policy):
        await runtime.evaluations.close()
        reference = await _add_attachment(storage, run.experiment_id, 1)
        objects = storage.object_store(RuntimeDomain.EVALUATION)
        stat = objects.stat
        inspected: list[str] = []

        async def missing(key: str) -> ObjectStat | None:
            inspected.append(key)
            return None if key == reference.key else await stat(key)

        monkeypatch.setattr(objects, "stat", missing)
        page = await run.current_trials(limit=1)
        assert await run.read_trial(page.items[0].trial.trial_id) == page.items[0]
        assert reference.key not in inspected
        with pytest.raises(AIError) as caught:
            await run.current_trials(cursor=page.next_cursor, limit=1)
        assert caught.value.code is ErrorCode.INPUT_CAPTURE_UNAVAILABLE
        assert reference.key in inspected and await report_count(storage) == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("mutation", ("purge", "revoke"))
async def test_current_window_rechecks_sources_before_returning_after_concurrent_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str,
) -> None:
    async with completed(tmp_path) as (runtime, storage, run, _scorer, policy):
        await runtime.evaluations.close()
        reference = await _add_attachment(storage, run.experiment_id, 0)
        objects = storage.object_store(RuntimeDomain.EVALUATION)
        stat = objects.stat
        changed = False

        async def mutate(key: str) -> ObjectStat | None:
            nonlocal changed
            result = await stat(key)
            if key == reference.key and not changed:
                changed = True
                if mutation == "purge":
                    await storage.evaluation.records.update(run.experiment_id, lambda value: replace(
                        value, content_deleted_at=datetime.now(timezone.utc)))
                else:
                    policy.reject = lambda action, resource: (
                        action is AuthorizationAction.EVALUATION_EVIDENCE_READ
                        and resource.id == run.experiment_id
                        and resource.owner_principal_id == PRINCIPAL.principal_id)
            return result

        monkeypatch.setattr(objects, "stat", mutate)
        with pytest.raises(AIError) as caught:
            await run.current_trials(limit=1)
        assert changed
        assert caught.value.code is (ErrorCode.EVALUATION_EVIDENCE_UNAVAILABLE if mutation == "purge"
                                      else ErrorCode.AUTHORIZATION_DENIED)
        assert await report_count(storage) == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", ("owner", "expired", "missing", "anonymous_input"))
async def test_get_case_preserves_owner_expiry_and_missing_boundaries(tmp_path: Path, fault: str) -> None:
    storage = RuntimeStorage.sqlite(tmp_path / "state.sqlite")
    policy = Policy()
    async with Runtime.open("case-read", storage=storage, models=ModelRegistry(),
                            context=CONTEXT, features=RuntimeFeatures("authorization", authorization=policy)) as runtime:
        ref = CaseRef("dataset", "case", 1)
        await runtime.evaluations.publish_dataset(DatasetSpec(DatasetRef("dataset", 1), (
            CaseSpec.task(ref, input={"answer": "yes"}, expected="yes"),)),
            principal=PRINCIPAL, idempotency_key="dataset")
        await runtime.evaluations.close()
        policy.calls.clear()
        repository = runtime._state.evaluation.records
        principal = PRINCIPAL
        if fault == "owner":
            principal = service_principal(PRINCIPAL.tenant_id, "another-owner")
        elif fault == "anonymous_input":
            principal = None
            policy.calls.clear()
            policy.reject = lambda action, resource: action is AuthorizationAction.EVALUATION_DATASET_READ
        elif fault == "missing":
            ref = CaseRef(ref.dataset_id, "missing", ref.revision)
        else:
            case = await repository.get_case(ref)
            identity = [ref.dataset_id, ref.case_id, ref.revision]
            key = repository._key("evaluation_case", identity)
            row = await repository.state_store.read(lambda tx: tx.get_record(key))
            value = EvaluationCaseRecord(case, PRINCIPAL.principal_id,
                content_expires_at=datetime.now(timezone.utc) - timedelta(seconds=1))
            expired = replace(repository._stored("evaluation_case", identity, value),
                              storage_revision=row.storage_revision + 1)
            assert await repository.state_store.mutate(lambda tx: tx.replace_record(
                expired, expected_storage_revision=row.storage_revision))
        with _sql_writes(runtime._state) as writes:
            with pytest.raises(AIError) as caught:
                await runtime.evaluations.get_case(ref, principal=principal)
        expected = {"owner": ErrorCode.AUTHORIZATION_DENIED, "anonymous_input": ErrorCode.AUTHORIZATION_DENIED,
                    "expired": ErrorCode.EVALUATION_EVIDENCE_UNAVAILABLE, "missing": ErrorCode.STORAGE_NOT_FOUND}
        assert caught.value.code is expected[fault]
        if fault == "anonymous_input":
            assert any(action is AuthorizationAction.EVALUATION_DATASET_READ for action, _ in policy.calls)
        assert not writes and await report_count(runtime._state) == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", ("dataset", "owner"))
async def test_current_reads_preserve_dataset_authorization(tmp_path: Path, boundary: str) -> None:
    async with completed(tmp_path) as (runtime, storage, run, _scorer, policy):
        await runtime.evaluations.close()
        record = await storage.evaluation.records.get(run.experiment_id, tenant_id=PRINCIPAL.tenant_id)
        policy.reject = lambda action, resource: action is AuthorizationAction.EVALUATION_DATASET_READ and (
            boundary == "dataset" and resource.owner_principal_id is None
            or boundary == "owner" and resource.owner_principal_id == PRINCIPAL.principal_id)
        for read in (lambda: run.current_trials(limit=1),
                     lambda: run.read_trial(record.manifest.trials[0].trial_id),
                     lambda: run.read_trial("unknown-trial")):
            with pytest.raises(AIError) as caught:
                await read()
            assert caught.value.code is ErrorCode.AUTHORIZATION_DENIED
        assert await report_count(storage) == 0


@pytest.mark.asyncio
async def test_reused_case_keeps_earlier_expiry_in_current_reads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage = RuntimeStorage.sqlite(tmp_path / "case-expiry.sqlite")
    async with Runtime.open("case-expiry", storage=storage, models=ModelRegistry(), context=CONTEXT) as runtime:
        owner = runtime.evaluations
        early = datetime.now(timezone.utc) + timedelta(hours=1)
        late = early + timedelta(hours=1)
        case = CaseSpec.task(CaseRef("dataset", "reused", 1), input={"answer": "yes"}, expected="yes")
        await owner.publish_dataset(DatasetSpec(DatasetRef("dataset", 1), (case,)),
            principal=PRINCIPAL, idempotency_key="original", content_expires_at=early)
        dataset = await owner.publish_dataset(DatasetSpec(DatasetRef("dataset", 2), (case.ref,)),
            principal=PRINCIPAL, idempotency_key="reused", content_expires_at=late)
        target = Task("case-expiry.target", echo, effect_policy="none")
        scorer = Task("case-expiry.score", exact, effect_policy="none")
        run = await owner.start(StartEvaluationRequest(EvaluationSpec(dataset,
            (CandidateSpec("candidate", task=target.ref),), (rule_scorer(scorer),)), PRINCIPAL, "run"),
            engine=runtime.tasks.bind(target, scorer))
        assert (await run.wait(timeout_seconds=EVALUATION_COMPLETION_TIMEOUT_SECONDS)).result.completion == "complete"
        await owner.close()
        record = await runtime._state.evaluation.records.get(run.experiment_id, tenant_id=PRINCIPAL.tenant_id)
        assert record.content_expires_at == early
        assert (await owner.get_case(case.ref, principal=PRINCIPAL)).ref == case.ref
        assert len((await run.current_trials()).items) == 1

        class ExpiredClock(datetime):
            @classmethod
            def now(cls, tz: timezone | None = None) -> datetime:
                return early if tz is not None else early.replace(tzinfo=None)

        monkeypatch.setattr(evaluation_module, "_now", lambda: early)
        monkeypatch.setattr(repository_module, "datetime", ExpiredClock)
        for read in (lambda: run.current_trials(),
                     lambda: run.read_trial(record.manifest.trials[0].trial_id),
                     lambda: owner.get_case(case.ref, principal=PRINCIPAL)):
            with pytest.raises(AIError) as caught:
                await read()
            assert caught.value.code is ErrorCode.EVALUATION_EVIDENCE_UNAVAILABLE
        assert await report_count(runtime._state) == 0


@pytest.mark.asyncio
async def test_current_state_survives_bounded_purge_of_shared_case_content(tmp_path: Path) -> None:
    storage = RuntimeStorage.sqlite(tmp_path / "bounded-purge.sqlite")
    async with Runtime.open("bounded-purge", storage=storage, models=ModelRegistry(), context=CONTEXT) as runtime:
        owner = runtime.evaluations
        early = datetime.now(timezone.utc) + timedelta(hours=1)
        late = early + timedelta(hours=1)
        ref = CaseRef("dataset", "reused", 1)
        await owner.publish_dataset(DatasetSpec(DatasetRef("dataset", 1), (
            CaseSpec.task(ref, input={"answer": "yes"}, expected="yes"),)),
            principal=PRINCIPAL, idempotency_key="early", content_expires_at=early)
        dataset = await owner.publish_dataset(DatasetSpec(DatasetRef("dataset", 2), (ref,)),
            principal=PRINCIPAL, idempotency_key="later", content_expires_at=late)
        target = Task("bounded-purge.target", echo, effect_policy="none")
        scorer = Task("bounded-purge.score", exact, effect_policy="none")
        runs = []
        for index in range(3):
            run = await owner.start(StartEvaluationRequest(EvaluationSpec(dataset,
                (CandidateSpec("candidate", task=target.ref),), (rule_scorer(scorer),)),
                PRINCIPAL, f"run-{index}"), engine=runtime.tasks.bind(target, scorer))
            assert (await run.wait(timeout_seconds=EVALUATION_COMPLETION_TIMEOUT_SECONDS)).result.completion == "complete"
            runs.append(run)
        await owner.close()
        purged = []
        for _ in range(2):
            result = await owner.purge_expired(principal=PRINCIPAL, now=early + timedelta(seconds=1),
                                             exclusive=Offline(), limit=1)
            purged.extend(result.evaluations)
        assert len(purged) == 2
        remaining = []
        for run in runs:
            record = await runtime._state.evaluation.records.get(run.experiment_id, tenant_id=PRINCIPAL.tenant_id)
            if record.content_deleted_at is None:
                remaining.append(run)
        assert len(remaining) == 1 and remaining[0].experiment_id not in purged
        run = remaining[0]
        assert await report_count(runtime._state) == 0
        with _sql_writes(runtime._state) as writes:
            page = await run.current_trials(limit=1)
            assert len(page.items) == 1 and page.next_cursor is None
            trial = page.items[0]
            assert trial.case_ref == ref and trial.execution_status is ExecutionStatus.SUCCEEDED
            assert await run.read_trial(trial.trial.trial_id) == trial
            for read in (lambda: owner.get_case(ref, principal=PRINCIPAL), run.preview_report):
                with pytest.raises(AIError) as caught:
                    await read()
                assert caught.value.code is ErrorCode.EVALUATION_EVIDENCE_UNAVAILABLE
        assert not writes and await report_count(runtime._state) == 0
