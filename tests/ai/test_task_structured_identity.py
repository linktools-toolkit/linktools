#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Task retries use the one durable canonical prompt identity."""

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import replace
from pathlib import Path

import pytest
from pydantic_ai.messages import BinaryContent, StructuredContent, TextContent

from linktools.ai.core import Principal, TaskStatus
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.inputs import AgentTaskInput
from linktools.ai.model import MaterializedUserContent, StoredUserInput
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.persistence.codec import decode_domain, encode_domain
from linktools.ai.persistence.state import RecordQuery, StateTransaction
from linktools.ai.storage import StoredPayload
from linktools.ai.task import (
    AgentTaskNode, DefaultTaskGraphService, TaskGraph, TaskGraphAdmission,
    TaskGraphRequest, TaskGraphSubmission, TaskNode, TaskRef, TaskSubmissionRef,
)
from .test_task_submission_fence import _Authorization, _Launcher


def _request(*, answer: int = 1, tag: str = "answer", text: str = "rendered",
             binary: bool = False, generic: bool = False) -> TaskGraphRequest:
    content = StructuredContent({"answer": answer}, content=text, tag=tag, metadata={"source": "native"})
    prompt = (content, BinaryContent(b"bytes", media_type="text/plain")) if binary else (content,)
    body = AgentTaskInput(prompt, planning=False, thinking=False)
    owner = TaskNode if generic else AgentTaskNode
    node = owner.from_resolved("worker", task=TaskRef("identity.agent", 1), input=body, original_input=body)
    return TaskGraphRequest(TaskGraph("structured", (node,)), Principal("submitter", "tenant"), "submit")


def _service(state: Persistence, *, preflight: object = None) -> DefaultTaskGraphService:
    return DefaultTaskGraphService(state.task, _Authorization(), _Launcher(), preflight=preflight)


@pytest.mark.parametrize("binary", (False, True))
def test_agent_prompt_identity_distinguishes_structured_content_from_text(binary: bool) -> None:
    current = _request(binary=binary)
    node = current.graph.nodes[0]
    text = TextContent("rendered", metadata={"source": "native"})
    prompt = (text, BinaryContent(b"bytes", media_type="text/plain")) if binary else (text,)
    body = AgentTaskInput(prompt, planning=False, thinking=False)
    legacy_node = AgentTaskNode.from_resolved("worker", task=node.task, input=body, original_input=body)
    assert node.to_mapping() != legacy_node.to_mapping()
    assert node.prompt_intent_digest() != legacy_node.prompt_intent_digest()


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ("prepared", "admitted", "cancelled"))
@pytest.mark.parametrize("backend", ("memory", "sqlite"))
async def test_submission_owner_survives_state_transitions_and_reopen(
    phase: str, backend: str, tmp_path: Path,
) -> None:
    def storage() -> PersistenceStorage:
        return PersistenceStorage.in_memory() if backend == "memory" else PersistenceStorage.sqlite(tmp_path / "state.db")

    state = Persistence(storage())
    await state.initialize(namespace="structured-identity", tenant_id="tenant")
    try:
        service = _service(state)
        prepared = await service.prepare_submission(_request())
        if phase == "admitted":
            await state.task.admissions.admit_prepared(prepared)
        elif phase == "cancelled":
            await service.cancel_submission(prepared.ref, principal=prepared.ref.principal, idempotency_key="cancel")
        if backend == "sqlite":
            await state.close()
            state = Persistence(storage())
            await state.initialize(namespace="structured-identity", tenant_id="tenant")
            service = _service(state)
        owner = await state.task.admissions.get_submission_ref("structured", tenant_id="tenant")
        assert owner == prepared.ref
        assert await state.task.admissions.get_submission_ref("structured", tenant_id="other") is None
        described = await service.describe_submission(_request())

        class UnavailableCapture:
            async def capture_admission(self, admission: TaskGraphAdmission, graph: TaskGraph) -> TaskGraph:
                pytest.fail("a retained submission must not recapture its original input")

        replay, created = await _service(state, preflight=UnavailableCapture()).prepare_described_with_disposition(described)
        assert replay.ref == prepared.ref and not created
        assert await state.task.admissions.submission_status(owner) == phase
        for changed in (_request(answer=2), _request(tag="other")):
            with pytest.raises(AIError) as caught:
                await service.prepare_submission(changed)
            assert caught.value.code is ErrorCode.IDEMPOTENCY_CONFLICT
        with pytest.raises(AIError) as caught:
            await service.prepare_submission(_request(text="changed rendering"))
        assert caught.value.code is ErrorCode.IDEMPOTENCY_CONFLICT
        if phase == "cancelled":
            assert await state.task.admissions.state_store.read(
                lambda transaction: transaction.list_records(RecordQuery(kind="task_submission_payload"))) == ()
            outcome = await service.start_prepared(replay)
            assert not outcome.admitted and outcome.result.status is TaskStatus.CANCELLED
        elif phase == "prepared":
            await state.task.admissions.admit_prepared(replay)
    finally:
        await state.close()


@pytest.mark.asyncio
async def test_generic_business_json_is_not_rewritten_by_identity() -> None:
    async with Persistence.open("structured-identity", storage=PersistenceStorage.in_memory(), tenant_id="tenant") as state:
        request = _request(generic=True)
        admission = TaskGraphAdmission.from_request(request)
        submission = TaskGraphSubmission(state.namespace, admission, request.graph)
        repository = state.task.admissions
        await repository.state_store.mutate(lambda transaction: transaction.insert_records((
            repository._stored("task_submission", ("structured",), submission.ref, state="prepared"),
            repository._stored("task_submission_payload", ("structured",), submission),
        )))
        assert (await _service(state).prepare_submission(request)).admission == admission
        with pytest.raises(AIError) as caught:
            await _service(state).prepare_submission(_request(answer=2, generic=True))
        assert caught.value.code is ErrorCode.IDEMPOTENCY_CONFLICT


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ("prepared", "admitted", "cancelled"))
async def test_direct_admission_uses_the_existing_submission_owner(phase: str) -> None:
    async with Persistence.open("structured-identity", storage=PersistenceStorage.in_memory(), tenant_id="tenant") as state:
        prepared = await _service(state).prepare_submission(_request())
        if phase == "admitted":
            await state.task.admissions.admit_prepared(prepared)
        elif phase == "cancelled":
            await _service(state).cancel_submission(prepared.ref, principal=prepared.ref.principal, idempotency_key="cancel")
        request = _request()
        view = await state.task.admissions.admit(TaskGraphAdmission.from_request(request), request.graph)
        assert view.status is (TaskStatus.CANCELLED if phase == "cancelled" else TaskStatus.PENDING)
        assert await state.task.admissions.get_submission_ref("structured", tenant_id="tenant") == prepared.ref


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ("prepared", "cancelled"))
@pytest.mark.parametrize("boundary", ("lookup", "capture"))
async def test_concurrent_winner_preserves_one_identity_without_recapturing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, phase: str, boundary: str,
) -> None:
    def storage() -> PersistenceStorage:
        return PersistenceStorage.sqlite(tmp_path / "state.db")
    async with (
        Persistence.open("structured-identity", storage=storage(), tenant_id="tenant") as first,
        Persistence.open("structured-identity", storage=storage(), tenant_id="tenant") as second,
    ):
        captures: list[str] = []

        async def win() -> None:
            winner = await _service(second).prepare_submission(_request())
            if phase == "cancelled":
                await _service(second).cancel_submission(winner.ref, principal=winner.ref.principal, idempotency_key="cancel")

        class Preflight:
            async def capture_admission(self, admission: TaskGraphAdmission, graph: TaskGraph) -> TaskGraph:
                captures.append(admission.initial_request_digest)
                await win()
                node = graph.nodes[0]
                body = dict(node.input)
                body["prompt"] = {
                    "kind": "stored-user-content-v1", "intent": "task-admission",
                    "source_intent_digest": node.prompt_intent_digest(),
                    "value": encode_domain(StoredUserInput("text", StoredPayload.inline_text("captured"))),
                }
                return TaskGraph(graph.graph_id, (AgentTaskNode.from_resolved(
                    node.node_id, task=node.task, input=body, original_input=node.original_input),))

        service = _service(first, preflight=Preflight() if boundary == "capture" else None)
        request = _request()
        candidate = TaskGraphSubmission(first.namespace, TaskGraphAdmission.from_request(request), request.graph)
        if boundary == "lookup":
            lookup = first.task.admissions.submission_status
            fired = False

            async def racing_lookup(reference: TaskSubmissionRef) -> str | None:
                nonlocal fired
                result = await lookup(reference)
                if not fired:
                    fired = True
                    assert result is None
                    await win()
                return result

            monkeypatch.setattr(first.task.admissions, "submission_status", racing_lookup)
        prepared, created = await service.prepare_described_with_disposition(candidate)
        assert not created and prepared.ref == candidate.ref
        assert captures == ([candidate.ref.request_digest] if boundary == "capture" else [])
        assert await first.task.admissions.submission_status(prepared.ref) == phase
        assert (await service.prepare_described(candidate)).ref == prepared.ref
        assert captures == ([candidate.ref.request_digest] if boundary == "capture" else [])


@pytest.mark.asyncio
async def test_concurrent_preparation_and_uncertain_commit_preserve_owner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    def storage() -> PersistenceStorage:
        return PersistenceStorage.sqlite(tmp_path / "state.db")
    async with (
        Persistence.open("structured-identity", storage=storage(), tenant_id="tenant") as first,
        Persistence.open("structured-identity", storage=storage(), tenant_id="tenant") as second,
    ):
        services = (_service(first), _service(second))
        candidate = await services[0].describe_submission(_request())
        results = await asyncio.gather(*(service.prepare_described_with_disposition(candidate) for service in services))
        assert sorted(created for _, created in results) == [False, True]
        assert all(prepared.ref == candidate.ref for prepared, _ in results)
        store = first.task.admissions.state_store
        mutate = store.mutate

        async def lose_response(operation: Callable[[StateTransaction], Awaitable[object]]) -> object:
            await mutate(operation)
            raise AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN)

        monkeypatch.setattr(store, "mutate", lose_response)
        assert await services[0].prepare_described_with_disposition(candidate) == (candidate, False)
        assert await first.task.admissions.get_submission_ref("structured", tenant_id="tenant") == candidate.ref


@pytest.mark.asyncio
async def test_concurrent_changed_structure_conflicts_against_current_winner(tmp_path: Path) -> None:
    async with (
        Persistence.open("structured-identity", storage=PersistenceStorage.sqlite(tmp_path / "state.db"), tenant_id="tenant") as first,
        Persistence.open("structured-identity", storage=PersistenceStorage.sqlite(tmp_path / "state.db"), tenant_id="tenant") as second,
    ):
        requests = (_request(), _request(answer=2))
        services = (_service(first), _service(second))
        described = await asyncio.gather(*(service.describe_submission(request)
                                          for service, request in zip(services, requests)))
        results = await asyncio.gather(*(service.prepare_described(submission)
                                        for service, submission in zip(services, described)), return_exceptions=True)
        successful = [item for item in results if isinstance(item, TaskGraphSubmission)]
        rejected = [item for item in results if isinstance(item, AIError)]
        assert len(successful) == len(rejected) == 1
        assert rejected[0].code is ErrorCode.IDEMPOTENCY_CONFLICT
        assert await first.task.admissions.get_submission_ref("structured", tenant_id="tenant") == successful[0].ref


@pytest.mark.asyncio
async def test_agent_materialization_retains_prompt_identity() -> None:
    from .test_agent_task_admission_input_owner import _setup

    executor, _, tasks, _ = _setup()
    request = _request(binary=True)
    original = request.graph.nodes[0]
    node = AgentTaskNode.from_resolved(original.node_id, task=TaskRef("agent", 1),
                                       input=original.input, original_input=original.original_input)
    request = replace(request, graph=TaskGraph(request.graph.graph_id, (node,)))
    admission = TaskGraphAdmission.from_request(request)
    captured = await executor._materialize_node_input(node, "tenant", tasks)
    assert captured.input != node.input
    admission.validate_graph(TaskGraph(request.graph.graph_id, (captured,)))
    restored = decode_domain(encode_domain(captured), TaskNode)
    admission.validate_graph(TaskGraph(request.graph.graph_id, (restored,)))


def test_materialized_agent_input_projects_only_owned_prompt_provenance() -> None:
    def node(structured: bool) -> AgentTaskNode:
        leaf = ({"kind": "structured-content", "value": {"answer": 1}, "tag": "answer", "description": None,
                 "content": "rendered", "metadata": None} if structured else
                {"kind": "text-content", "content": "rendered", "metadata": None})
        item = StructuredContent({"answer": 1}, content="rendered", tag="answer") if structured else TextContent("rendered")
        view = {"version": 1, "prompt": {"kind": "items", "items": [
            {"kind": "native", "codec": "user-content-v1", "value": {"items": [leaf]}},
        ]}, "files": [], "attachments": []}
        body = AgentTaskInput(MaterializedUserContent((item,), view), planning=False, thinking=False,
                              parameters={"kind": "structured-content", "value": {"business": 2}})
        return AgentTaskNode.from_resolved("worker", task=TaskRef("identity.agent", 1), input=body, original_input=body)

    current, old = node(True), node(False)
    assert current.to_mapping() != old.to_mapping()
    assert current.to_mapping()["input"]["parameters"] == current.input["parameters"]


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ("prepared", "admitted"))
async def test_submission_lookup_rejects_digest_disagreement_with_admission(phase: str) -> None:
    async with Persistence.open("structured-identity", storage=PersistenceStorage.in_memory(), tenant_id="tenant") as state:
        prepared = await _service(state).prepare_submission(_request())
        repository = state.task.admissions
        if phase == "admitted":
            await repository.admit_prepared(prepared)

        async def corrupt(transaction: StateTransaction) -> None:
            record = await transaction.get_record(repository._submission_key("structured"))
            changed = repository._stored("task_submission", ("structured",), replace(prepared.ref, request_digest="f" * 64))
            assert await transaction.replace_record(
                replace(record, data=changed.data, storage_revision=record.storage_revision + 1),
                expected_storage_revision=record.storage_revision,
            )

        await repository.state_store.mutate(corrupt)
        with pytest.raises(AIError) as caught:
            await repository.get_submission_ref("structured", tenant_id="tenant")
        assert caught.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
        with pytest.raises(AIError) as caught:
            await repository.prepare(prepared)
        assert caught.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


@pytest.mark.asyncio
async def test_cancel_before_prepare_retains_canonical_identity() -> None:
    async with Persistence.open("structured-identity", storage=PersistenceStorage.in_memory(), tenant_id="tenant") as state:
        request = _request()
        admission = TaskGraphAdmission.from_request(request)
        reference = TaskSubmissionRef.from_admission(state.namespace, admission)
        service = _service(state)
        cancelled = await service.cancel_submission(reference, principal=reference.principal, idempotency_key="cancel")
        assert not cancelled.admitted and cancelled.status is TaskStatus.CANCELLED
        assert (await service.prepare_submission(request)).ref == reference
        assert await state.task.admissions.submission_status(reference) == "cancelled"


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ("admitted", "cancelled"))
async def test_start_prepared_returns_the_durable_winners_reference(phase: str) -> None:
    async with Persistence.open("structured-identity", storage=PersistenceStorage.in_memory(), tenant_id="tenant") as state:
        request = _request()
        stale = TaskGraphSubmission(state.namespace, TaskGraphAdmission.from_request(request), request.graph)
        winner = await _service(state).prepare_submission(_request())
        if phase == "admitted":
            await state.task.admissions.admit_prepared(winner)
        else:
            await _service(state).cancel_submission(winner.ref, principal=winner.ref.principal, idempotency_key="cancel")
        result = await _service(state).start_prepared(stale)
        assert result.submission == winner.ref
        assert await state.task.admissions.submission_status(result.submission) == phase
