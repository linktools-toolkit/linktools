#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Conversation receipts project canonical terminal data from Execution."""
import asyncio
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

import pytest
from pydantic import BaseModel
from pydantic_ai.messages import BinaryContent, ModelResponse, TextPart, ToolCallPart

from linktools.ai.capability import ExecutionCapability, StepPersistenceCapability
from linktools.ai.runtime import RuntimeFeatures
from linktools.ai.core import AuthorizationAction, ExecutionStatus, Principal, PromptLimits, ResourceKind, TenantAuthorizationPolicy
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.history import History
from linktools.ai.model import StoredUserInput
from linktools.ai.runtime import Runtime
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.runtime._agent_executor import AgentExecutor
from linktools.ai.inputs import ExecutionInputMaterializer
from linktools.ai.persistence import PersistenceDomain, PersistencePlan
from linktools.ai.persistence.codec import (
    _encode_persisted_domain, iter_runtime_object_refs,
)
from linktools.ai.persistence.state import OperationQuery, StateTransaction
from linktools.ai.storage import FilesystemObjectStore, InMemoryCacheBackend, StorageRoute, StoredPayload, read_object

from ._runtime_test_helpers import _Models, _UsageFunctionModel, model_features
from .test_evaluation_retention import Offline

pytestmark = pytest.mark.asyncio
_NAMESPACE = "conversation-output-retention"


class _HostAuthorization(TenantAuthorizationPolicy):
    """Explicit host policy for native SQL Session fixtures."""


async def _send(runtime: Runtime, session: str, prompt: str):
    turn = await runtime.sessions.start(session, prompt, agent="default", principal=runtime.default_principal,
                                        idempotency_key=prompt)
    return await turn.wait_final(timeout_seconds=5)


async def _timeline(history: History, principal: Principal, session: str = "session", **pagination):
    return (await history.turn_session(session, principal=principal, **pagination))["timeline"]


def _features(*, conversation: bool = True) -> RuntimeFeatures:
    features = RuntimeFeatures(_NAMESPACE)
    features.agent("default", model="default", allow_tools=())
    if conversation:
        features.capability(ExecutionCapability(retain_history=True), scope="runtime")
    return features


def _models() -> _Models:
    async def respond(messages, info):
        return ModelResponse(parts=[TextPart("model transcript, not the public result")])
    return _Models(_UsageFunctionModel(respond))


def _storage(
    path: Path, backend: str = "sqlite", *, objects: FilesystemObjectStore | None = None,
) -> PersistenceStorage:
    objects = objects or FilesystemObjectStore(path / "objects")
    if backend == "local_sqlite":
        return PersistenceStorage.from_plan(PersistenceStorage.from_root(path / "state", exclusive_writer=True).plan, object_store=objects)
    return PersistenceStorage.sqlite(path / "state.db", object_store=objects)


async def _erase_domain(storage: Persistence, domain: PersistenceDomain) -> None:
    """Remove domain roots for corruption and object-reachability tests."""
    async def erase(transaction: StateTransaction) -> None:
        records = await transaction.scan_records()
        facts = await transaction.scan_facts()
        for owner in {fact.owner_key for fact in facts}:
            await transaction.delete_fact_streams(owner)
        await transaction.delete_records(tuple(record.key for record in records))
        await transaction.delete_operations(OperationQuery())
    await storage._stores[domain].mutate(erase)


def _canonical_output(monkeypatch: pytest.MonkeyPatch, output: object) -> None:
    original = AgentExecutor.execute

    async def execute(self, scope):
        result = await original(self, scope)
        return replace(result, output=output)
    monkeypatch.setattr(AgentExecutor, "execute", execute)


class _Answer(BaseModel):
    answer: list[str | int] | None


class _OwnerAuthorization(_HostAuthorization):
    async def authorize(self, principal, action, resource):
        if action is AuthorizationAction.SESSION_READ and resource.kind is ResourceKind.SESSION:
            if principal.principal_id != "owner":
                raise AIError(ErrorCode.AUTHORIZATION_DENIED)
        await super().authorize(principal, action, resource)


@pytest.mark.parametrize("backend", ("sqlite", "local_sqlite"))
@pytest.mark.parametrize("answer", (["canonical", 42], None))
async def test_native_output_uses_transcript_owner_after_reopen(tmp_path: Path, backend: str, answer) -> None:
    output = {"answer": answer}

    async def model(_messages, info):
        [tool] = info.output_tools
        return ModelResponse(parts=[ToolCallPart(tool.name, output, tool_call_id="canonical-output")])

    storage = _storage(tmp_path, backend)
    authorization = _OwnerAuthorization()
    features = model_features(_Models(_UsageFunctionModel(model)), _features())
    async with (
        Persistence.open(_NAMESPACE, storage=storage) as state,
        Runtime.open(persistence=state, features=features, authorization=authorization) as runtime,
    ):
        principal = Principal("owner", runtime.tenant_id)
        await runtime.sessions.create_turn_session("session", principal=principal)
        turn = await runtime.sessions.start("session", "original user input", agent="default", principal=principal,
                                            idempotency_key="input", output=_Answer)
        result = await turn.wait_final(timeout_seconds=5)
        assert result.status is ExecutionStatus.SUCCEEDED and result.view["output"] == output
        [live] = (await _timeline(runtime.history, principal))["items"]
        assert live["output"] == output
        assert live["user_input"]["prompt"] == "original user input"
        row = await state.session_turns.get_turn(turn.key)
        assert row["result_json"]["output_ref"]
        assert not {"stored_user_input", "output", "status"}.intersection(row["result_json"])
        assert await state.execution.executions.get(turn.key.turn_id, tenant_id=principal.tenant_id) is None

    async with (
        Persistence.open(_NAMESPACE, storage=storage, read_only=True) as state,
        History.open(persistence=state, authorization=authorization) as history,
    ):
        assert (await _timeline(history, principal))["items"] == [live]
        assert await history.turn_result("session", turn.key.turn_id, principal=principal) == live
        for denied in (Principal("other-owner", principal.tenant_id), Principal("owner", "foreign-tenant")):
            with pytest.raises(AIError) as error:
                await history.turn_result("session", turn.key.turn_id, principal=denied)
            assert error.value.code is ErrorCode.AUTHORIZATION_DENIED


async def test_missing_transcript_owner_never_falls_back_to_execution_payloads(tmp_path: Path) -> None:
    storage = _storage(tmp_path)
    authorization = _HostAuthorization()
    async with (
        Persistence.open(_NAMESPACE, storage=storage) as state,
        Runtime.open(persistence=state, features=model_features(_models(), _features()),
                     authorization=authorization) as runtime,
    ):
        actor = runtime.default_principal
        await runtime.sessions.create_turn_session("session", principal=actor)
        committed = await _send(runtime, "session", "canonical input")
        assert (await _timeline(runtime.history, actor))["items"] == [committed.view]
        turns = state.session_turns
        row = await turns.get_turn(committed.key)
        async with turns._context.engine.begin() as connection:
            mutation = turns._turns.update().filter_by(**turns._turn_values(committed.key)).values(
                result_json={**row["result_json"], "transcript": {}},
            )
            assert (await connection.execute(mutation)).rowcount == 1
    async with (
        Persistence.open(_NAMESPACE, storage=storage, read_only=True) as state,
        History.open(persistence=state, authorization=authorization) as history,
    ):
        with pytest.raises(AIError) as missing:
            await history.turn_result("session", committed.key.turn_id, principal=actor)
        assert missing.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


@pytest.mark.parametrize("mismatch", ("session", "principal", "parent", "nonterminal"))
async def test_committed_turn_validates_membership_independently_of_actor(tmp_path: Path, mismatch: str) -> None:
    storage = _storage(tmp_path)
    authorization = _HostAuthorization()
    async with (
        Persistence.open(_NAMESPACE, storage=storage) as state,
        Runtime.open(persistence=state, features=model_features(_models(), _features()),
                     authorization=authorization) as runtime,
    ):
        principal = runtime.default_principal
        await runtime.sessions.create_turn_session("session", principal=principal)
        committed = await _send(runtime, "session", "owned input")
        await runtime.sessions.create_turn_session("other", principal=principal)
        if mismatch == "principal":
            # Reader grants are issued by the host, not by rewriting the
            # admission's actor or requiring the same identity at read time.
            another_reader = Principal("other-owner", principal.tenant_id)
            assert await runtime.history.turn_result("session", committed.key.turn_id,
                                                     principal=another_reader) == committed.view
        elif mismatch == "nonterminal":
            turns = state.session_turns
            async with turns._context.engine.begin() as connection:
                mutation = turns._turns.update().filter_by(**turns._turn_values(committed.key)).values(
                    state="accepted", result_json=None,
                )
                assert (await connection.execute(mutation)).rowcount == 1
            with pytest.raises(AIError) as invalid:
                await _timeline(runtime.history, principal)
            assert invalid.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
        else:
            queried = committed.key.turn_id
            if mismatch == "parent":
                parent = runtime.agents.get().session("session")
                child = await parent.fork("child", expected_revision=(await parent.inspect())["revision"],
                                          idempotency_key="fork")
                assert child is not None
                [inherited] = (await child.timeline())["items"]
                assert inherited["turn_id"] == committed.key.turn_id
                assert await runtime.history.turn_result("child", queried, principal=principal) == inherited
                queried = (await _send(runtime, "session", "not inherited")).key.turn_id
            with pytest.raises(AIError) as invalid:
                await runtime.history.turn_result(
                    "child" if mismatch == "parent" else "other", queried, principal=principal,
                )
            assert invalid.value.code is ErrorCode.AUTHORIZATION_DENIED


@pytest.mark.parametrize("backend", ("sqlite", "local_sqlite"))
async def test_terminal_execution_facts_are_immutable_but_control_updates_preserve_result(
    tmp_path: Path, backend: str,
) -> None:
    async with Persistence.open(_NAMESPACE, storage=_storage(tmp_path, backend)) as persistence, Runtime.open(persistence=persistence, features=model_features(_models(), _features())) as runtime:
        principal = runtime.default_principal
        execution = (await runtime.agents.get().run("original input", timeout_seconds=5)).result
        repository = persistence.execution.executions
        owner = await repository.get(execution.execution_id, tenant_id=principal.tenant_id)
        previous = await runtime.executions.result(execution.execution_id, principal=principal)
        assert owner.completed_at is not None and owner.session_id is None
        changes = (
            {"session_id": "different-session"},
            {"principal_id": "different-owner"},
            {"status": ExecutionStatus.FAILED, "result": replace(owner.result, output=None)},
            {"stored_user_input": StoredUserInput("text", StoredPayload.inline_text("rewritten input"))},
            {"result": replace(owner.result, output=StoredPayload.inline_json("rewritten output"))},
            {"created_at": owner.created_at + timedelta(seconds=1)},
            {"completed_at": owner.completed_at + timedelta(seconds=1)},
            {"error_code": ErrorCode.MODEL_UNAVAILABLE.value},
            {"safe_error_details": {"rewritten": True}},
        )
        for changed in changes:
            with pytest.raises(AIError) as conflict:
                await repository.compare_and_swap(
                    owner.execution_id, tenant_id=principal.tenant_id,
                    expected_revision=owner.revision,
                    next_record=replace(owner, revision=owner.revision + 1, **changed),
                )
            assert conflict.value.code is ErrorCode.EXECUTION_RESULT_CONFLICT
        assert await repository.get(owner.execution_id, tenant_id=principal.tenant_id) == owner
        updated = await repository.compare_and_swap(
            owner.execution_id, tenant_id=principal.tenant_id,
            expected_revision=owner.revision,
            next_record=replace(owner, revision=owner.revision + 1,
                                updated_at=owner.updated_at + timedelta(seconds=5)),
        )
        assert updated.completed_at == owner.completed_at
        assert await runtime.executions.result(execution.execution_id, principal=principal) == previous
    async with Persistence.open(_NAMESPACE, storage=_storage(tmp_path, backend), read_only=True) as persistence, History.open(persistence=persistence) as history:
        assert await history.result(execution.execution_id, principal=principal) == previous


async def test_retained_payload_and_attachment_roots_survive_gc_then_release(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    objects = FilesystemObjectStore(tmp_path / "objects")
    attachment = b"ATTACHMENT CONTENT" * (8 * 1024)
    output = {"canonical": "x" * (80 * 1024)}
    _canonical_output(monkeypatch, output)
    storage = _storage(tmp_path, objects=objects)
    async with Persistence.open(_NAMESPACE, storage=storage) as persistence_132, Runtime.open(persistence=persistence_132, features=model_features(_models(), _features())) as runtime:
        principal = runtime.default_principal
        result = (await runtime.agents.get().run(("inspect original", BinaryContent(
            attachment, media_type="text/plain", identifier="retained-file",
        )), timeout_seconds=5)).result
        record = await runtime._state.execution.executions.get(result.execution_id, tenant_id=principal.tenant_id)
        assert record.result.output.kind == "inline"
        assert record.result.output.decode() == output
        assert record.result.output.object_refs() == ()
        assert record.stored_user_input.payload.kind == "inline"
        assert record.stored_user_input.payload.encoding == "json-attachments"
        references = tuple(set(iter_runtime_object_refs(
            _encode_persisted_domain(record), default_domain=PersistenceDomain.EXECUTION,
        )))
        assert record.session_id is None
        assert (await runtime.executions.result(result.execution_id, principal=principal)).output == output
        assert references == tuple(
            (PersistenceDomain.EXECUTION, ref)
            for ref in record.stored_user_input.payload.object_refs()
        )
        assert len(references) == 1, "binary input must remain a discoverable GC root"
        assert all(domain is PersistenceDomain.EXECUTION for domain, _ in references)
    for _, ref in references:
        assert await objects.stat(ref.key) is not None
    storage = _storage(tmp_path, objects=objects)
    storage = Persistence(storage)
    await storage.initialize(namespace=_NAMESPACE, tenant_id=principal.tenant_id)
    try:
        retained = await storage.purge_unreferenced_objects(references, exclusive=Offline())
        assert set(retained.retained) == set(references)
        for domain, ref in references:
            await read_object(storage.object_store(domain), ref.key,
                              expected_digest=ref.digest, expected_size=ref.size)
        restored = await ExecutionInputMaterializer(
            None, PromptLimits(), object_store=storage.object_store(PersistenceDomain.EXECUTION),
        ).restore(record.stored_user_input)
        binaries = [part for part in restored if isinstance(part, BinaryContent)]
        assert len(binaries) == 1 and binaries[0].data == attachment
    finally:
        await storage.close()
    async with Persistence.open(_NAMESPACE, storage=_storage(tmp_path, objects=objects), tenant_id="default", read_only=True) as history_persistence_175, History.open(persistence=history_persistence_175) as history:
        assert (await history.result(result.execution_id, principal=principal)).output == output
    storage = _storage(tmp_path, objects=objects)
    storage = Persistence(storage)
    await storage.initialize(namespace=_NAMESPACE, tenant_id=principal.tenant_id)
    try:
        await _erase_domain(storage, PersistenceDomain.CONVERSATION)
        retained = await storage.purge_unreferenced_objects(references, exclusive=Offline())
        assert set(retained.retained) == set(references)
        await _erase_domain(storage, PersistenceDomain.EXECUTION)
        released = await storage.purge_unreferenced_objects(references, exclusive=Offline())
        assert set((*released.deleted, *released.missing)) == set(references)
        assert not released.retained
    finally:
        await storage.close()


async def test_fork_retains_only_selected_history_and_independent_turn_outputs(tmp_path: Path) -> None:
    replies = iter(("first answer", "parent answer", "child answer"))

    async def model(_messages, _info):
        return ModelResponse(parts=[TextPart(next(replies))])

    storage = _storage(tmp_path)
    authorization = _HostAuthorization()
    features = model_features(_Models(_UsageFunctionModel(model)), _features())
    async with (
        Persistence.open(_NAMESPACE, storage=storage) as state,
        Runtime.open(persistence=state, features=features, authorization=authorization) as runtime,
    ):
        principal = runtime.default_principal
        agent = runtime.agents.get()
        parent = await agent.create_session("parent")
        await _send(runtime, "parent", "first")
        child = await parent.fork("child", expected_revision=(await parent.inspect())["revision"],
                                  idempotency_key="fork")
        assert child is not None
        await _send(runtime, "parent", "parent only")
        await _send(runtime, "child", "child only")
        live = (await _timeline(runtime.history, principal, "child"))["items"]
        assert [item["output"] for item in live] == [{"text": "first answer"}, {"text": "child answer"}]
        assert [item["user_input"]["prompt"] for item in live] == ["first", "child only"]
    async with (
        Persistence.open(_NAMESPACE, storage=storage, read_only=True) as state,
        History.open(persistence=state, authorization=authorization) as history,
    ):
        assert (await _timeline(history, principal, "child"))["items"] == live
        newest = await _timeline(history, principal, "child", limit=1)
        assert newest["items"] == live[-1:] and newest["next_cursor"] is not None
        oldest = await _timeline(history, principal, "child", limit=1, cursor=newest["next_cursor"])
        assert oldest["items"] == live[:1] and oldest["next_cursor"] is None


async def test_disabled_conversation_does_not_retain_new_chat_or_break_results(tmp_path: Path) -> None:
    storage = _storage(tmp_path)
    async with Persistence.open(_NAMESPACE, storage=storage) as persistence_226, Runtime.open(persistence=persistence_226, features=model_features(_models(), _features(conversation=False))) as runtime:
        principal = runtime.default_principal
        result = (await runtime.agents.get().run("not retained as chat", timeout_seconds=5)).result
        owner = await persistence_226.execution.executions.get(result.execution_id, tenant_id=principal.tenant_id)
        assert owner is not None and not owner.conversation and owner.session_id is None
        assert (await runtime.executions.result(result.execution_id, principal=principal)).output == result.output
    async with Persistence.open(_NAMESPACE, storage=_storage(tmp_path)) as persistence_235, Runtime.open(persistence=persistence_235, features=model_features(_models(), _features(conversation=False))) as runtime:
        assert (await runtime.executions.result(result.execution_id, principal=principal)).output == result.output


async def test_zero_message_failure_keeps_input_and_never_reuses_prior_output(tmp_path: Path) -> None:
    async def model(messages, _info):
        from pydantic_ai.messages import ModelRequest, UserPromptPart

        prompt = next(part.content for part in reversed(messages[-1].parts) if isinstance(part, UserPromptPart))
        assert isinstance(messages[-1], ModelRequest)
        if prompt == "failed original input":
            raise AIError(ErrorCode.MODEL_UNAVAILABLE, safe_details={"phase": "before_model"})
        return ModelResponse(parts=[TextPart("reply to " + prompt)])

    storage = _storage(tmp_path)
    authorization = _HostAuthorization()
    features = model_features(_Models(_UsageFunctionModel(model)), _features())
    async with (
        Persistence.open(_NAMESPACE, storage=storage) as state,
        Runtime.open(persistence=state, features=features, authorization=authorization) as runtime,
    ):
        principal = runtime.default_principal
        await runtime.sessions.create_turn_session("session", principal=principal)
        await _send(runtime, "session", "first success")
        failed = await _send(runtime, "session", "failed original input")
        assert failed.status is ExecutionStatus.FAILED
        await _send(runtime, "session", "last success")
        live = (await _timeline(runtime.history, principal))["items"]
        assert len(live) == 3
        assert [turn.get("output") for turn in live] == [
            {"text": "reply to first success"}, None, {"text": "reply to last success"},
        ]
        assert live[1]["turn_id"] == failed.key.turn_id
        assert live[1]["user_input"]["prompt"] == "failed original input"
        assert live[1]["status"] == "FAILED" and live[1]["items"] == []
        assert live[1]["error_code"] == ErrorCode.MODEL_UNAVAILABLE.value
        assert await state.execution.executions.get(failed.key.turn_id, tenant_id=principal.tenant_id) is None
    async with (
        Persistence.open(_NAMESPACE, storage=storage, read_only=True) as state,
        History.open(persistence=state, authorization=authorization) as history,
    ):
        assert (await _timeline(history, principal))["items"] == live


async def test_cancelled_turn_retains_original_input_without_completed_output(tmp_path: Path) -> None:
    entered = asyncio.Event()

    async def respond(_messages, _info):
        entered.set()
        await asyncio.Event().wait()

    storage = _storage(tmp_path)
    authorization = _HostAuthorization()
    features = model_features(_Models(_UsageFunctionModel(respond)), _features())
    async with (
        Persistence.open(_NAMESPACE, storage=storage) as state,
        Runtime.open(persistence=state, features=features, authorization=authorization) as runtime,
    ):
        principal = runtime.default_principal
        await runtime.sessions.create_turn_session("session", principal=principal)
        turn = await runtime.sessions.start("session", "cancelled original input", agent="default",
                                            principal=principal, idempotency_key="cancelled")
        await asyncio.wait_for(entered.wait(), 5)
        active = await runtime.history.turn_result("session", turn.key.turn_id, principal=principal)
        assert active.get("output") is None and not active["conversation_committed"]
        assert await runtime.sessions.cancel_turn("session", turn.key.turn_id, principal=principal)
        result = await turn.wait_final(timeout_seconds=5)
        assert result.status is ExecutionStatus.CANCELLED
        live = (await _timeline(runtime.history, principal))["items"]
        assert len(live) == 1 and live[0]["status"] == "CANCELLED"
        assert live[0].get("output") is None
        assert live[0]["user_input"]["prompt"] == "cancelled original input"
    async with (
        Persistence.open(_NAMESPACE, storage=storage, read_only=True) as state,
        History.open(persistence=state, authorization=authorization) as history,
    ):
        assert (await _timeline(history, principal))["items"] == live


async def test_unknown_terminal_commit_retains_exactly_one_canonical_turn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    async def model(_messages, _info):
        calls.append(1)
        return ModelResponse(parts=[TextPart("durably committed despite lost acknowledgement")])

    storage = _storage(tmp_path)
    authorization = _HostAuthorization()
    features = model_features(_Models(_UsageFunctionModel(model)), _features())
    lost = asyncio.Event()
    async with (
        Persistence.open(_NAMESPACE, storage=storage) as state,
        Runtime.open(persistence=state, features=features, authorization=authorization) as runtime,
    ):
        principal = runtime.default_principal
        await runtime.sessions.create_turn_session("session", principal=principal)
        turns = state.session_turns
        original = turns.finish

        async def finish(key, **kwargs):
            assert kwargs["state"] == "succeeded"
            assert await original(key, **kwargs)
            lost.set()
            raise AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN)

        monkeypatch.setattr(turns, "finish", finish)
        turn = await runtime.sessions.start("session", "original unknown-commit input", agent="default",
                                            principal=principal, idempotency_key="request")
        await asyncio.wait_for(lost.wait(), 5)
        expected = await runtime.history.turn_result("session", turn.key.turn_id, principal=principal)
        assert expected["output"] == {"text": "durably committed despite lost acknowledgement"}
        assert expected["user_input"]["prompt"] == "original unknown-commit input"
        assert (await _timeline(runtime.history, principal))["items"] == [expected]
    async with (
        Persistence.open(_NAMESPACE, storage=storage) as state,
        Runtime.open(persistence=state, features=features, authorization=authorization) as runtime,
    ):
        replay = await runtime.sessions.start("session", "original unknown-commit input", agent="default",
                                              principal=principal, idempotency_key="request")
        assert replay.key == turn.key
        assert (await replay.wait_final(timeout_seconds=5)).view == expected
        assert (await _timeline(runtime.history, principal))["items"] == [expected]
    assert calls == [1]


async def test_completed_native_display_does_not_expand_execution(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    storage = _storage(tmp_path)
    authorization = _HostAuthorization()
    async with (
        Persistence.open(_NAMESPACE, storage=storage) as state,
        Runtime.open(persistence=state, features=model_features(_models(), _features()),
                     authorization=authorization) as runtime,
    ):
        actor = runtime.default_principal
        await runtime.sessions.create_turn_session("session", principal=actor)
        committed = await _send(runtime, "session", "retained display input")

        async def execution_read(*args, **kwargs):
            pytest.fail("native display must not expand an Execution owner or binding")

        for method in ("get", "get_many", "get_many_fresh"):
            monkeypatch.setattr(state.execution.executions, method, execution_read)
        for _ in range(2):
            assert (await _timeline(runtime.history, actor))["items"] == [committed.view]


@pytest.mark.parametrize("publication", ("confirmed", "lost-ack", "missing"))
async def test_split_route_never_publishes_completed_output_before_terminal_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, publication: str,
) -> None:
    from linktools.ai.runtime._canonical_split_step import _CanonicalSplitStep

    source = StorageRoute.sqlite(tmp_path / "source.db")
    storage = PersistenceStorage.from_plan(PersistencePlan(
        execution=source, checkpoint=source, conversation=StorageRoute.sqlite(tmp_path / "conversation.db"),
    ))
    entered, release, finished = asyncio.Event(), asyncio.Event(), asyncio.Event()
    authorization = _HostAuthorization()
    calls: list[str] = []

    async def model(_messages, _info):
        calls.append("model")
        return ModelResponse(parts=[TextPart("split route result")])

    features = _features(conversation=False)
    features.capability(StepPersistenceCapability(), scope="runtime")
    model_features(_Models(_UsageFunctionModel(model)), features)
    receipts = []
    original_publish = _CanonicalSplitStep.publish_terminal

    async def publish(self, key, **kwargs):
        try:
            return await original_publish(self, key, **kwargs)
        finally:
            finished.set()

    async with (
        Persistence.open(_NAMESPACE, storage=storage) as state,
        Runtime.open(persistence=state, features=features, authorization=authorization) as runtime,
    ):
        principal = runtime.default_principal
        await runtime.sessions.create_turn_session("session", principal=principal)
        turns = state.session_turns
        original = turns.publish_split_terminal

        async def target(key, **kwargs):
            assert kwargs["source"].phase == "sealed" and kwargs["state"] == "succeeded"
            receipts.append(kwargs["source"].pin.receipt_id)
            entered.set()
            await release.wait()
            if publication != "missing":
                committed = await original(key, **kwargs)
                if publication == "confirmed":
                    return committed
                assert committed
            raise AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN)

        with monkeypatch.context() as patch:
            patch.setattr(_CanonicalSplitStep, "publish_terminal", publish)
            patch.setattr(turns, "publish_split_terminal", target)
            turn = await runtime.sessions.start("session", "original split input", agent="default",
                                                principal=principal, idempotency_key="request")
            try:
                await asyncio.wait_for(entered.wait(), 5)
                row = await turns.get_turn(turn.key)
                assert row["split_source_receipt"] is None and row["state"] == "accepted"
                active = await runtime.history.turn_result("session", turn.key.turn_id, principal=principal)
                assert active.get("output") is None and not active["conversation_committed"]
                assert active["user_input"]["prompt"] == "original split input"
            finally:
                release.set()
            await asyncio.wait_for(finished.wait(), 5)
            if publication != "missing":
                assert (await turn.wait_final(timeout_seconds=5)).status is ExecutionStatus.SUCCEEDED
            else:
                assert (await turns.get_turn(turn.key))["split_source_receipt"] is None
    async with (
        Persistence.open(_NAMESPACE, storage=storage) as state,
        Runtime.open(persistence=state, features=features, authorization=authorization) as runtime,
    ):
        turns = state.session_turns
        if publication == "missing":
            await turns.expire_stopped_lease(turn.key, generation=1)
            assert await runtime.sessions.recover_turn("session", turn.key.turn_id, principal=principal)
        row = await turns.get_turn(turn.key)
        assert (row["state"], row["generation"], row["split_source_receipt"]) == (
            "succeeded", 1, receipts[0].encode(),
        )
        [retained] = (await _timeline(runtime.history, principal))["items"]
        assert retained["output"] == {"text": "split route result"}
        assert retained["user_input"]["prompt"] == "original split input"
        replay = await runtime.sessions.start("session", "original split input", agent="default",
                                              principal=principal, idempotency_key="request")
        assert replay.key == turn.key
        assert (await replay.wait_final(timeout_seconds=5)).view == retained
    assert len(receipts) == 1 and calls == ["model"]


async def test_native_display_five_reads_keep_bounded_selects(tmp_path: Path) -> None:
    from sqlalchemy import event

    storage = PersistenceStorage.sqlite(
        tmp_path / "state.db", cache=InMemoryCacheBackend(max_bytes=4 * 1024 * 1024),
    )
    async with Persistence.open(_NAMESPACE, storage=storage) as persistence_494, Runtime.open(persistence=persistence_494, features=model_features(_models(), _features()), authorization=_HostAuthorization()) as runtime:
        actor = runtime.default_principal
        await runtime.sessions.create_turn_session("session", principal=actor)
        committed = await _send(runtime, "session", "measure retained display")
        engine = persistence_494.session_turns._context.engine
        statements = []

        def capture(_connection, _cursor, statement, *_args):
            if statement.lstrip().upper().startswith("SELECT"):
                statements.append(statement)

        event.listen(engine.sync_engine, "before_cursor_execute", capture)
        try:
            first = await _timeline(runtime.history, actor)
            assert first["items"] == [committed.view]
            cold = len(statements)
            counts = []
            for _ in range(5):
                statements.clear()
                assert await _timeline(runtime.history, actor) == first
                counts.append(len(statements))
            assert 0 < cold <= 5
            assert all(0 < count <= cold for count in counts)
        finally:
            event.remove(engine.sync_engine, "before_cursor_execute", capture)


async def test_split_object_stores_read_canonical_result_and_binary_input_after_reopen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = {"canonical": "y" * (80 * 1024)}
    attachment = b"separate physical attachment" * (4 * 1024)
    _canonical_output(monkeypatch, output)
    plan = PersistencePlan(
        execution=StorageRoute.sqlite(tmp_path / "execution.db"),
        conversation=StorageRoute.sqlite(tmp_path / "conversation.db"),
    )
    async with Persistence.open(_NAMESPACE, storage=PersistenceStorage.from_plan(plan)) as persistence, Runtime.open(persistence=persistence, features=model_features(_models(), _features())) as runtime:
        principal = runtime.default_principal
        source = persistence.object_store(PersistenceDomain.EXECUTION)
        conversation_objects = persistence.object_store(PersistenceDomain.CONVERSATION)
        assert source is not conversation_objects
        result = (await runtime.agents.get().run(("keep binary", BinaryContent(
            attachment, media_type="text/plain", identifier="separate-store-file",
        )), timeout_seconds=5)).result
        assert result.status is ExecutionStatus.SUCCEEDED
        owner = await persistence.execution.executions.get(result.execution_id, tenant_id=principal.tenant_id)
        assert owner.result.output.kind == "inline"
        assert owner.result.output.decode() == output
        assert owner.result.output.object_refs() == ()
        assert owner.stored_user_input.payload.encoding == "json-attachments"
        source_refs = owner.stored_user_input.payload.object_refs()
        assert len(source_refs) == 1
        assert owner.session_id is None
        for ref in source_refs:
            assert await source.stat(ref.key) is not None
            assert await conversation_objects.stat(ref.key) is None
        assert (await runtime.executions.result(result.execution_id, principal=principal)).output == output
    async with Persistence.open(_NAMESPACE, storage=PersistenceStorage.from_plan(plan), tenant_id="default", read_only=True) as persistence, History.open(persistence=persistence) as history:
        assert (await history.result(result.execution_id, principal=principal)).output == output
        owner = await persistence.execution.executions.get(result.execution_id, tenant_id=principal.tenant_id)
        restored = await ExecutionInputMaterializer(
            None, PromptLimits(), object_store=persistence.object_store(PersistenceDomain.EXECUTION),
        ).restore(owner.stored_user_input)
        assert [item.data for item in restored if isinstance(item, BinaryContent)] == [attachment]


@pytest.mark.parametrize("status", (ExecutionStatus.SUCCEEDED, ExecutionStatus.FAILED, ExecutionStatus.CANCELLED))
@pytest.mark.parametrize("conversation", (False, True))
async def test_missing_terminal_receipt_never_fabricates_retained_history(
    tmp_path: Path, status: ExecutionStatus, conversation: bool,
) -> None:
    from linktools.ai.core import HmacCursorSigner, TenantAuthorizationPolicy
    from linktools.ai.history._session_timeline import project_session_timeline
    from .test_session_timeline import _execution, _session

    storage = Persistence(PersistenceStorage.in_memory())
    await storage.initialize(namespace=_NAMESPACE, tenant_id="tenant")
    try:
        await storage.conversation.sessions.create(_session())
        execution = replace(_execution("execution", status=status, prompt="unconfirmed original input"), conversation=conversation)
        await storage.execution.executions.create(execution)
        await storage.conversation.sessions.admit_execution(
            "session", tenant_id="tenant", execution_id="execution", expected=None,
        )
        async def read():
            return await project_session_timeline(
                storage.conversation, storage.execution.executions, TenantAuthorizationPolicy(),
                HmacCursorSigner("retention", b"retained-terminal-read"),
                storage.run_store.conversation_persistence, "session", principal=Principal("owner", "tenant"),
            )
        if conversation or status is ExecutionStatus.SUCCEEDED:
            with pytest.raises(AIError) as missing:
                await read()
            assert missing.value.code is (
                ErrorCode.STORAGE_RECOVERY_REQUIRED if conversation else ErrorCode.SESSION_HISTORY_UNAVAILABLE
            )
        else:
            page = await read()
            assert page.items[0].status is status and page.items[0].output is None
            assert not page.items[0].conversation_committed
            assert page.items[0].user_input["prompt"]["text"] == "unconfirmed original input"
    finally:
        await storage.close()


@pytest.mark.parametrize("split", (False, True))
async def test_cancelled_unconfirmed_model_response_does_not_enter_following_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, split: bool,
) -> None:
    seen = []
    entered, release = asyncio.Event(), asyncio.Event()
    original = AgentExecutor.execute
    runs = 0

    async def respond(messages, info):
        seen.append(tuple(messages))
        text = "CONFIRMED_FIRST_RESPONSE" if len(seen) == 1 else "UNCONFIRMED_CANCELLED_RESPONSE" if len(seen) == 2 else "FINAL_RESPONSE"
        return ModelResponse(parts=[TextPart(text)])

    async def hold_second(self, scope):
        nonlocal runs
        runs += 1
        current = runs
        result = await original(self, scope)
        if current == 2:
            entered.set()
            await release.wait()
        return result

    monkeypatch.setattr(AgentExecutor, "execute", hold_second)
    storage = PersistenceStorage.from_plan(PersistencePlan(
        execution=StorageRoute.sqlite(tmp_path / "execution.db"),
        conversation=StorageRoute.sqlite(tmp_path / "conversation.db"),
    )) if split else _storage(tmp_path)
    async with Persistence.open(_NAMESPACE, storage=storage) as persistence_677, Runtime.open(persistence=persistence_677, features=model_features(_Models(_UsageFunctionModel(respond)), _features()), authorization=_HostAuthorization()) as runtime:
        actor = runtime.default_principal
        await runtime.sessions.create_turn_session("session", principal=actor)
        assert (await _send(runtime, "session", "confirmed first input")).status is ExecutionStatus.SUCCEEDED
        second = await runtime.sessions.start("session", "unconfirmed cancelled input", agent="default",
                                              principal=actor, idempotency_key="cancel")
        try:
            await asyncio.wait_for(entered.wait(), 5)
            assert await runtime.sessions.cancel_turn("session", second.key.turn_id, principal=actor)
        finally:
            release.set()
        assert (await second.wait_final(timeout_seconds=5)).status is ExecutionStatus.CANCELLED
        assert (await _send(runtime, "session", "follow-up input")).status is ExecutionStatus.SUCCEEDED
        assert len(seen) == 3
        following_context = repr(seen[2])
        assert "CONFIRMED_FIRST_RESPONSE" in following_context
        assert "UNCONFIRMED_CANCELLED_RESPONSE" not in following_context
        assert "unconfirmed cancelled input" in following_context
        assert "This turn was interrupted before a final answer." in following_context
        turns = (await _timeline(runtime.history, actor))["items"]
        assert len(turns) == 3 and turns[1]["status"] == "CANCELLED"
        assert turns[1]["user_input"]["prompt"] == "unconfirmed cancelled input"
        assert turns[1].get("output") is None and turns[1]["items"] == []
