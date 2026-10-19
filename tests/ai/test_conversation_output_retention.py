#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Canonical conversation turns survive independent execution retention."""

from linktools.ai.runtime.state import RuntimeState
import asyncio
from dataclasses import replace
from pathlib import Path

import pytest
from pydantic_ai.messages import BinaryContent, ModelResponse, TextPart

from linktools.ai.capability import ConversationCapability, ExecutionCapability, RuntimeFeatures
from linktools.ai.core import CommitPolicy, ExecutionStatus, Principal, PromptLimits
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime import Runtime, RuntimeHistory, RuntimeStorage
from linktools.ai.runtime._agent_executor import AgentExecutor
from linktools.ai.runtime._input import ExecutionInputMaterializer
from linktools.ai.runtime.state import RuntimeDomain, RuntimeStoragePlan
from linktools.ai.runtime.state._codec import _encode_persisted_domain, iter_runtime_object_refs
from linktools.ai.runtime.state._store import OperationQuery, StateTransaction
from linktools.ai.storage import FilesystemObjectStore, InMemoryCacheBackend, StorageRoute, read_object

from ._runtime_test_helpers import _Models, _UsageFunctionModel
from .test_evaluation_retention import Offline

pytestmark = pytest.mark.asyncio
_NAMESPACE = "conversation-output-retention"


def _features(*, conversation: bool = True) -> RuntimeFeatures:
    features = RuntimeFeatures(_NAMESPACE)
    features.agent("default", model="default", allow_tools=())
    if conversation:
        features.capability(ConversationCapability(), scope="runtime")
    return features


def _models() -> _Models:
    async def respond(messages, info):
        return ModelResponse(parts=[TextPart("model transcript, not the public result")])
    return _Models(_UsageFunctionModel(respond))


def _storage(
    path: Path, backend: str = "sqlite", *, objects: FilesystemObjectStore | None = None,
) -> RuntimeStorage:
    objects = objects or FilesystemObjectStore(path / "objects")
    if backend == "filesystem":
        return RuntimeStorage.filesystem(path / "state", object_store=objects)
    return RuntimeStorage.sqlite(path / "state.db", object_store=objects)


async def _erase_domain(storage: RuntimeState, domain: RuntimeDomain) -> None:
    """Physically prune records and facts, rather than fake a missing repository read."""
    async def erase(transaction: StateTransaction) -> None:
        records = await transaction.scan_records()
        facts = await transaction.scan_facts()
        for owner in {fact.owner_key_digest for fact in facts}:
            await transaction.delete_fact_streams(owner)
        await transaction.delete_records(tuple(record.key_digest for record in records))
        await transaction.delete_operations(OperationQuery())
    await storage._stores[domain].mutate(erase)


async def _erase_execution_owners(storage: RuntimeState) -> None:
    for domain in RuntimeDomain:
        if domain is not RuntimeDomain.CONVERSATION:
            await _erase_domain(storage, domain)


def _canonical_output(monkeypatch: pytest.MonkeyPatch, output: object) -> None:
    original = AgentExecutor.execute

    async def execute(self, scope):
        result = await original(self, scope)
        return replace(result, output=output)
    monkeypatch.setattr(AgentExecutor, "execute", execute)


@pytest.mark.parametrize("backend", ("sqlite", "filesystem"))
@pytest.mark.parametrize("output", ({"answer": ["canonical", 42]}, None))
async def test_completed_output_is_canonical_after_actual_prune_and_reopen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend: str, output: object,
) -> None:
    _canonical_output(monkeypatch, output)
    storage = _storage(tmp_path, backend)
    async with Runtime.open(_NAMESPACE, storage=storage, features=_features(), models=_models()) as runtime:
        principal = runtime.default_principal
        session = await runtime.agents.get().create_session("session")
        result = (await session.run("original user input", timeout_seconds=5)).result
        assert result.status is ExecutionStatus.SUCCEEDED and result.output == output
        live = (await session.timeline()).items
        assert len(live) == 1 and live[0].output == output
        assert live[0].user_input["prompt"]["text"] == "original user input"
        assert live[0].items and live[0].items[0].content != output
        commits = await runtime._state.conversation.sessions.list_timeline_commits(
            "session", tenant_id=principal.tenant_id, start_turn_seq=1, end_turn_seq=2,
        )
        assert commits[0].output is not None
        assert commits[0].output.decode() == output
    storage = _storage(tmp_path, backend)
    storage = RuntimeState(storage)
    await storage.initialize(namespace=_NAMESPACE, tenant_id=principal.tenant_id)
    try:
        await _erase_execution_owners(storage)
        assert await storage.execution.executions.get(result.execution_id, tenant_id=principal.tenant_id) is None
    finally:
        await storage.close()
    async with RuntimeHistory.open(_NAMESPACE, storage=_storage(tmp_path, backend)) as history:
        page = await history.session_timeline("session", principal=principal)
        assert page.items == live
        assert page.items[0].output == output
        with pytest.raises(AIError) as denied:
            await history.session_timeline("session", principal=Principal("other-owner", principal.tenant_id))
        assert denied.value.code is ErrorCode.AUTHORIZATION_DENIED
        with pytest.raises(AIError) as foreign:
            await history.session_timeline("session", principal=Principal(principal.principal_id, "foreign-tenant"))
        assert foreign.value.code is ErrorCode.AUTHORIZATION_DENIED


async def test_retained_payload_and_attachment_roots_survive_gc_then_release(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    objects = FilesystemObjectStore(tmp_path / "objects")
    attachment = b"ATTACHMENT CONTENT" * (8 * 1024)
    output = {"canonical": "x" * (80 * 1024)}
    _canonical_output(monkeypatch, output)
    storage = _storage(tmp_path, objects=objects)
    async with Runtime.open(_NAMESPACE, storage=storage, features=_features(), models=_models()) as runtime:
        principal = runtime.default_principal
        session = await runtime.agents.get().create_session("session")
        result = (await session.run(("inspect original", BinaryContent(
            attachment, media_type="text/plain", identifier="retained-file",
        )), timeout_seconds=5)).result
        record = await runtime._state.execution.executions.get(result.execution_id, tenant_id=principal.tenant_id)
        assert record.result.output.kind == "object"
        assert record.stored_user_input.payload.kind == "object"
        originals = tuple(set(iter_runtime_object_refs(
            _encode_persisted_domain(record), default_domain=RuntimeDomain.EXECUTION,
        )))
        live = (await session.timeline()).items
        commits = await runtime._state.conversation.sessions.list_timeline_commits(
            "session", tenant_id=principal.tenant_id, start_turn_seq=1, end_turn_seq=2,
        )
        references = tuple(set(iter_runtime_object_refs(
            _encode_persisted_domain(commits[0]), default_domain=RuntimeDomain.CONVERSATION,
        )))
        assert len(references) >= 2, "retained output and binary input must be discoverable GC roots"
    for _, ref in references:
        assert await objects.stat(ref.key) is not None
    storage = _storage(tmp_path, objects=objects)
    storage = RuntimeState(storage)
    await storage.initialize(namespace=_NAMESPACE, tenant_id=principal.tenant_id)
    try:
        await _erase_execution_owners(storage)
        await storage.purge_unreferenced_objects(originals, exclusive=Offline())
        retained = await storage.purge_unreferenced_objects(references, exclusive=Offline())
        assert set(retained.retained) == set(references)
        for domain, ref in references:
            await read_object(storage.object_store(domain), ref.key,
                              expected_digest=ref.digest, expected_size=ref.size)
        restored = await ExecutionInputMaterializer(
            None, PromptLimits(), object_store=storage.object_store(RuntimeDomain.CONVERSATION),
        ).restore(commits[0].stored_user_input)
        binaries = [part for part in restored if isinstance(part, BinaryContent)]
        assert len(binaries) == 1 and binaries[0].data == attachment
    finally:
        await storage.close()
    async with RuntimeHistory.open(_NAMESPACE, storage=_storage(tmp_path, objects=objects)) as history:
        assert (await history.session_timeline("session", principal=principal)).items == live
    storage = _storage(tmp_path, objects=objects)
    storage = RuntimeState(storage)
    await storage.initialize(namespace=_NAMESPACE, tenant_id=principal.tenant_id)
    try:
        await _erase_domain(storage, RuntimeDomain.CONVERSATION)
        all_references = tuple(set((*references, *originals)))
        released = await storage.purge_unreferenced_objects(all_references, exclusive=Offline())
        assert set((*released.deleted, *released.missing)) == set(all_references)
        assert not released.retained
    finally:
        await storage.close()


async def test_fork_retains_only_selected_history_and_independent_turn_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = AgentExecutor.execute
    count = 0

    async def execute(self, scope):
        nonlocal count
        result = await original(self, scope)
        count += 1
        return replace(result, output={"turn": count})
    monkeypatch.setattr(AgentExecutor, "execute", execute)
    async with Runtime.open(_NAMESPACE, storage=_storage(tmp_path), features=_features(), models=_models()) as runtime:
        principal = runtime.default_principal
        parent = await runtime.agents.get().create_session("parent")
        await parent.run("first", timeout_seconds=5)
        child = await parent.fork("child")
        await parent.run("parent only", timeout_seconds=5)
        await child.run("child only", timeout_seconds=5)
        live = (await child.timeline()).items
        assert [item.output for item in live] == [{"turn": 1}, {"turn": 3}]
        assert [item.user_input["prompt"]["text"] for item in live] == ["first", "child only"]
    storage = _storage(tmp_path)
    storage = RuntimeState(storage)
    await storage.initialize(namespace=_NAMESPACE, tenant_id=principal.tenant_id)
    try:
        await _erase_execution_owners(storage)
    finally:
        await storage.close()
    async with RuntimeHistory.open(_NAMESPACE, storage=_storage(tmp_path)) as history:
        assert (await history.session_timeline("child", principal=principal)).items == live
        newest = await history.session_timeline("child", principal=principal, limit=1)
        assert newest.items == live[-1:] and newest.next_cursor is not None
        oldest = await history.session_timeline("child", principal=principal, limit=1, cursor=newest.next_cursor)
        assert oldest.items == live[:1] and oldest.next_cursor is None


async def test_disabled_conversation_does_not_retain_new_chat_or_break_results(tmp_path: Path) -> None:
    storage = _storage(tmp_path)
    async with Runtime.open(_NAMESPACE, storage=storage, features=_features(conversation=False), models=_models()) as runtime:
        principal = runtime.default_principal
        session = await runtime.agents.get().create_session("session")
        result = (await session.run("not retained as chat", timeout_seconds=5)).result
        commits = await runtime._state.conversation.sessions.list_timeline_commits(
            "session", tenant_id=principal.tenant_id, start_turn_seq=1, end_turn_seq=2,
        )
        assert commits == ()
        assert (await runtime.executions.result(result.execution_id, principal=principal)).output == result.output
    async with Runtime.open(_NAMESPACE, storage=_storage(tmp_path), features=_features(conversation=False), models=_models()) as runtime:
        assert (await runtime.executions.result(result.execution_id, principal=principal)).output == result.output


async def test_zero_message_failure_keeps_input_and_never_reuses_prior_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = AgentExecutor.execute
    count = 0

    async def execute(self, scope):
        nonlocal count
        count += 1
        if count == 2:
            raise AIError(ErrorCode.MODEL_UNAVAILABLE, safe_details={"phase": "before_model"})
        result = await original(self, scope)
        return replace(result, output={"completed_turn": count})
    monkeypatch.setattr(AgentExecutor, "execute", execute)
    storage = _storage(tmp_path)
    async with Runtime.open(_NAMESPACE, storage=storage, features=_features(), models=_models()) as runtime:
        principal = runtime.default_principal
        session = await runtime.agents.get().create_session("session")
        await session.run("first success", timeout_seconds=5)
        failed = (await session.run("failed original input", timeout_seconds=5)).result
        assert failed.status is ExecutionStatus.FAILED
        await session.run("last success", timeout_seconds=5)
        live = (await session.timeline()).items
        assert len(live) == 3
        assert [turn.output for turn in live] == [{"completed_turn": 1}, None, {"completed_turn": 3}]
        assert live[1].user_input["prompt"]["text"] == "failed original input"
        assert live[1].status is ExecutionStatus.FAILED and live[1].items == ()
        assert live[1].error_code == ErrorCode.MODEL_UNAVAILABLE.value
        commits = await runtime._state.conversation.sessions.list_timeline_commits(
            "session", tenant_id=principal.tenant_id, start_turn_seq=1, end_turn_seq=4,
        )
        assert len(commits) == 3 and commits[1].output is None
        assert commits[1].start_message_index == commits[1].end_message_index
    storage = _storage(tmp_path)
    storage = RuntimeState(storage)
    await storage.initialize(namespace=_NAMESPACE, tenant_id=principal.tenant_id)
    try:
        await _erase_execution_owners(storage)
    finally:
        await storage.close()
    async with RuntimeHistory.open(_NAMESPACE, storage=_storage(tmp_path)) as history:
        assert (await history.session_timeline("session", principal=principal)).items == live


async def test_cancelled_turn_retains_original_input_without_completed_output(tmp_path: Path) -> None:
    entered = asyncio.Event()

    async def respond(messages, info):
        entered.set()
        await asyncio.Event().wait()
    async with Runtime.open(
        _NAMESPACE, storage=_storage(tmp_path), features=_features(),
        models=_Models(_UsageFunctionModel(respond)),
    ) as runtime:
        principal = runtime.default_principal
        session = await runtime.agents.get().create_session("session")
        execution = await session.start("cancelled original input")
        await asyncio.wait_for(entered.wait(), 5)
        active = (await session.timeline()).items
        assert len(active) == 1 and active[0].output is None
        assert active[0].status is ExecutionStatus.STARTED
        await execution.cancel()
        result = (await execution.wait(timeout_seconds=5)).result
        assert result.status is ExecutionStatus.CANCELLED
        live = (await session.timeline()).items
        assert len(live) == 1 and live[0].status is ExecutionStatus.CANCELLED
        assert live[0].output is None
        assert live[0].user_input["prompt"]["text"] == "cancelled original input"
    storage = _storage(tmp_path)
    storage = RuntimeState(storage)
    await storage.initialize(namespace=_NAMESPACE, tenant_id=principal.tenant_id)
    try:
        await _erase_execution_owners(storage)
    finally:
        await storage.close()
    async with RuntimeHistory.open(_NAMESPACE, storage=_storage(tmp_path)) as history:
        assert (await history.session_timeline("session", principal=principal)).items == live


async def test_unknown_terminal_commit_retains_exactly_one_canonical_turn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = {"canonical": "durably committed despite lost acknowledgement"}
    _canonical_output(monkeypatch, output)
    features = _features()
    features.capability(ExecutionCapability(commit=CommitPolicy.terminal()), scope="runtime")
    storage = RuntimeStorage.sqlite(
        tmp_path / "state.db", cache=InMemoryCacheBackend(max_bytes=4 * 1024 * 1024),
        object_store=FilesystemObjectStore(tmp_path / "objects"),
    )
    async with Runtime.open(_NAMESPACE, storage=storage, features=features, models=_models()) as runtime:
        principal = runtime.default_principal
        session = await runtime.agents.get().create_session("session")
        group = runtime._state.execution.executions.state_store.storage_group
        original_mutate = type(group).mutate
        original_read = type(group).read_many
        committed = asyncio.Event()
        unavailable = True
        writes = 0

        async def lose_ack(owner, stores, callback):
            nonlocal writes
            result = await original_mutate(owner, stores, callback)
            if owner is group:
                writes += 1
                if writes == 1:
                    committed.set()
                    raise AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN)
            return result

        async def fail_confirmation(owner, stores, callback):
            if owner is group and committed.is_set() and unavailable:
                raise OSError("source confirmation temporarily unavailable")
            return await original_read(owner, stores, callback)

        monkeypatch.setattr(type(group), "mutate", lose_ack)
        monkeypatch.setattr(type(group), "read_many", fail_confirmation)
        execution = await session.start("original unknown-commit input", idempotency_key="request")
        await asyncio.wait_for(committed.wait(), 5)
        with pytest.raises(AIError) as unknown:
            await execution.wait(timeout_seconds=5)
        assert unknown.value.code is ErrorCode.STORAGE_COMMIT_UNKNOWN
        unavailable = False
        assert (await execution.wait(timeout_seconds=5)).result.output == output
        live = (await session.timeline()).items
        assert len(live) == 1 and live[0].output == output
        assert writes == 1, "readback must not replay an already committed turn"
        replay = await session.start("original unknown-commit input", idempotency_key="request")
        assert replay.execution_id == execution.execution_id
        assert (await replay.wait(timeout_seconds=5)).result.output == output
        assert (await session.timeline()).items == live
        monkeypatch.setattr(type(group), "mutate", original_mutate)
        monkeypatch.setattr(type(group), "read_many", original_read)
    storage = _storage(tmp_path)
    storage = RuntimeState(storage)
    await storage.initialize(namespace=_NAMESPACE, tenant_id=principal.tenant_id)
    try:
        await _erase_execution_owners(storage)
    finally:
        await storage.close()
    async with RuntimeHistory.open(_NAMESPACE, storage=_storage(tmp_path)) as history:
        assert (await history.session_timeline("session", principal=principal)).items == live


async def test_completed_display_cache_uses_retained_envelope_after_execution_prune(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = {"canonical": "cached retained result"}
    _canonical_output(monkeypatch, output)
    storage = RuntimeStorage.sqlite(
        tmp_path / "state.db", cache=InMemoryCacheBackend(max_bytes=4 * 1024 * 1024),
    )
    async with Runtime.open(_NAMESPACE, storage=storage, features=_features(), models=_models()) as runtime:
        principal = runtime.default_principal
        session = await runtime.agents.get().create_session("session")
        await session.run("retained cached input", timeout_seconds=5)
        cold = await runtime.sessions.refresh("session", principal=principal, max_age=60)
        assert cold.timeline.items[0].output == output
        await _erase_domain(runtime._state, RuntimeDomain.EXECUTION)

        async def execution_read(*args, **kwargs):
            pytest.fail("completed display retention must not depend on an Execution row")

        async def transcript_read(*args, **kwargs):
            pytest.fail("a valid retained display cache must not reload transcript bodies")
            yield

        with monkeypatch.context() as scoped:
            scoped.setattr(runtime._state.execution.executions, "get_many", execution_read)
            scoped.setattr(runtime._state.execution.executions, "get_many_fresh", execution_read)
            scoped.setattr(runtime._state.run_store.conversation_persistence, "iter_conversation_message_range", transcript_read)
            warm = await runtime.sessions.refresh("session", principal=principal, max_age=60)
            assert warm == cold


@pytest.mark.parametrize("lost_release_ack", (False, True))
async def test_split_route_never_publishes_completed_output_before_execution_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, lost_release_ack: bool,
) -> None:
    output = {"canonical": "split route result"}
    _canonical_output(monkeypatch, output)
    plan = RuntimeStoragePlan(
        execution=StorageRoute.sqlite(tmp_path / "execution.db"),
        conversation=StorageRoute.sqlite(tmp_path / "conversation.db"),
    )
    storage = RuntimeStorage.from_plan(plan)
    entered, release = asyncio.Event(), asyncio.Event()
    async with Runtime.open(_NAMESPACE, storage=storage, features=_features(), models=_models()) as runtime:
        principal = runtime.default_principal
        session = await runtime.agents.get().create_session("session")
        executions = runtime._state.execution.executions
        original_terminal = executions.commit_terminal_in_transaction
        sessions = runtime._state.conversation.sessions
        original_mutate = sessions.state_store.mutate
        lost = 0

        async def pause_terminal(*args, **kwargs):
            entered.set()
            await release.wait()
            return await original_terminal(*args, **kwargs)

        async def lose_ack(callback, *args, **kwargs):
            nonlocal lost
            result = await original_mutate(callback, *args, **kwargs)
            if lost_release_ack and callback.__name__ == "release_conversation" and lost == 0:
                lost += 1
                raise AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN)
            return result

        monkeypatch.setattr(executions, "commit_terminal_in_transaction", pause_terminal)
        monkeypatch.setattr(sessions.state_store, "mutate", lose_ack)
        execution = await session.start("original split input", idempotency_key="request")
        try:
            await asyncio.wait_for(entered.wait(), 5)
            header = await sessions.get("session", tenant_id=principal.tenant_id)
            assert header.continuation is not None, "exercise the pre-terminal transcript publication window"
            commits = await sessions.list_timeline_commits(
                "session", tenant_id=principal.tenant_id, start_turn_seq=1, end_turn_seq=2,
            )
            assert commits == ()
            active = await asyncio.wait_for(session.timeline(), 5)
            assert len(active.items) == 1
            assert active.items[0].status is ExecutionStatus.FINALIZING
            assert active.items[0].output is None and not active.items[0].conversation_committed
            assert active.items[0].user_input["prompt"]["text"] == "original split input"
        finally:
            release.set()
        result = (await execution.wait(timeout_seconds=5)).result
        assert result.status is ExecutionStatus.SUCCEEDED and result.output == output
        assert lost == int(lost_release_ack)
        live = (await session.timeline()).items
        assert len(live) == 1 and live[0].output == output
        replay = await session.start("original split input", idempotency_key="request")
        assert replay.execution_id == execution.execution_id
        assert (await replay.wait(timeout_seconds=5)).result.output == output
        assert (await session.timeline()).items == live
        monkeypatch.setattr(executions, "commit_terminal_in_transaction", original_terminal)
        monkeypatch.setattr(sessions.state_store, "mutate", original_mutate)
    storage = RuntimeStorage.from_plan(plan)
    storage = RuntimeState(storage)
    await storage.initialize(namespace=_NAMESPACE, tenant_id=principal.tenant_id)
    try:
        await _erase_execution_owners(storage)
    finally:
        await storage.close()
    async with RuntimeHistory.open(_NAMESPACE, storage=RuntimeStorage.from_plan(plan)) as history:
        assert (await history.session_timeline("session", principal=principal)).items == live


async def test_display_cache_five_refresh_measurement(tmp_path: Path) -> None:
    from sqlalchemy import event

    storage = RuntimeStorage.sqlite(
        tmp_path / "state.db", cache=InMemoryCacheBackend(max_bytes=4 * 1024 * 1024),
    )
    async with Runtime.open(_NAMESPACE, storage=storage, features=_features(), models=_models()) as runtime:
        session = await runtime.agents.get().create_session("session")
        await session.run("measure retained display", timeout_seconds=5)
        engine = runtime._state.conversation.sessions.state_store.storage_group.context.engine
        statements = []

        def capture(_connection, _cursor, statement, *_args):
            if statement.lstrip().upper().startswith("SELECT"):
                statements.append(statement)

        event.listen(engine.sync_engine, "before_cursor_execute", capture)
        try:
            first = await runtime.sessions.refresh("session", principal=runtime.default_principal, max_age=60)
            cold = len(statements)
            counts = []
            for _ in range(5):
                statements.clear()
                assert await runtime.sessions.refresh("session", principal=runtime.default_principal, max_age=60) == first
                counts.append(len(statements))
            print(f"DISPLAY_CACHE_SQL cold={cold} warm_five={counts}", flush=True)
        finally:
            event.remove(engine.sync_engine, "before_cursor_execute", capture)


async def test_split_object_stores_copy_result_and_binary_input_before_source_deletion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = {"canonical": "y" * (80 * 1024)}
    attachment = b"separate physical attachment" * (4 * 1024)
    _canonical_output(monkeypatch, output)
    plan = RuntimeStoragePlan(
        execution=StorageRoute.sqlite(tmp_path / "execution.db"),
        conversation=StorageRoute.sqlite(tmp_path / "conversation.db"),
    )
    storage = RuntimeStorage.from_plan(plan)
    async with Runtime.open(_NAMESPACE, storage=storage, features=_features(), models=_models()) as runtime:
        principal = runtime.default_principal
        assert runtime._state.object_store(RuntimeDomain.EXECUTION) is not runtime._state.object_store(RuntimeDomain.CONVERSATION)
        session = await runtime.agents.get().create_session("session")
        result = (await session.run(("keep binary", BinaryContent(
            attachment, media_type="text/plain", identifier="separate-store-file",
        )), timeout_seconds=5)).result
        assert result.status is ExecutionStatus.SUCCEEDED
        record = await runtime._state.execution.executions.get(result.execution_id, tenant_id=principal.tenant_id)
        source_refs = (record.result.output.ref, record.stored_user_input.payload.ref)
        assert all(ref is not None for ref in source_refs)
        live = (await session.timeline()).items
    storage = RuntimeStorage.from_plan(plan)
    storage = RuntimeState(storage)
    await storage.initialize(namespace=_NAMESPACE, tenant_id=principal.tenant_id)
    try:
        await _erase_execution_owners(storage)
        source = storage.object_store(RuntimeDomain.EXECUTION)
        destination = storage.object_store(RuntimeDomain.CONVERSATION)
        assert source is not destination
        for ref in source_refs:
            assert await source.delete_object(ref.key, expected_digest=ref.digest)
            assert await source.stat(ref.key) is None
        commits = await storage.conversation.sessions.list_timeline_commits(
            "session", tenant_id=principal.tenant_id, start_turn_seq=1, end_turn_seq=2,
        )
        restored = await ExecutionInputMaterializer(
            None, PromptLimits(), object_store=destination,
        ).restore(commits[0].stored_user_input)
        assert [item.data for item in restored if isinstance(item, BinaryContent)] == [attachment]
    finally:
        await storage.close()
    async with RuntimeHistory.open(_NAMESPACE, storage=RuntimeStorage.from_plan(plan)) as history:
        page = await history.session_timeline("session", principal=principal)
        assert page.items == live and page.items[0].output == output


async def test_split_terminal_without_conversation_receipt_requires_recovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = RuntimeStoragePlan(
        execution=StorageRoute.sqlite(tmp_path / "execution.db"),
        conversation=StorageRoute.sqlite(tmp_path / "conversation.db"),
    )
    storage = RuntimeStorage.from_plan(plan)
    entered, release = asyncio.Event(), asyncio.Event()
    async with Runtime.open(_NAMESPACE, storage=storage, features=_features(), models=_models()) as runtime:
        principal = runtime.default_principal
        session = await runtime.agents.get().create_session("session")
        sessions = runtime._state.conversation.sessions
        original_mutate = sessions.state_store.mutate

        async def pause_receipt(callback, *args, **kwargs):
            if callback.__name__ == "release_conversation":
                entered.set()
                await release.wait()
            return await original_mutate(callback, *args, **kwargs)

        monkeypatch.setattr(sessions.state_store, "mutate", pause_receipt)
        execution = await session.start("input awaiting retained receipt")
        try:
            await asyncio.wait_for(entered.wait(), 5)
            terminal = await runtime._state.execution.executions.get(execution.execution_id, tenant_id=principal.tenant_id)
            assert terminal.status is ExecutionStatus.SUCCEEDED
            assert await sessions.list_timeline_commits(
                "session", tenant_id=principal.tenant_id, start_turn_seq=1, end_turn_seq=2,
            ) == ()
            with pytest.raises(AIError) as unconfirmed:
                await session.timeline()
            assert unconfirmed.value.code is ErrorCode.STORAGE_RECOVERY_REQUIRED
        finally:
            release.set()
        result = (await execution.wait(timeout_seconds=5)).result
        retained = (await session.timeline()).items
        assert len(retained) == 1 and retained[0].output == result.output
        assert retained[0].user_input["prompt"]["text"] == "input awaiting retained receipt"
        monkeypatch.setattr(sessions.state_store, "mutate", original_mutate)


@pytest.mark.parametrize("status", (ExecutionStatus.SUCCEEDED, ExecutionStatus.FAILED, ExecutionStatus.CANCELLED))
@pytest.mark.parametrize("conversation", (False, True))
async def test_missing_terminal_receipt_never_fabricates_retained_history(
    tmp_path: Path, status: ExecutionStatus, conversation: bool,
) -> None:
    from linktools.ai.core import HmacCursorSigner, TenantAuthorizationPolicy
    from linktools.ai.runtime._session_timeline import project_session_timeline
    from .test_session_timeline import _execution, _session

    storage = RuntimeStorage.sqlite(tmp_path / "state.db")
    storage = RuntimeState(storage)
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
    storage = RuntimeStorage.from_plan(RuntimeStoragePlan(
        execution=StorageRoute.sqlite(tmp_path / "execution.db"),
        conversation=StorageRoute.sqlite(tmp_path / "conversation.db"),
    )) if split else _storage(tmp_path)
    async with Runtime.open(_NAMESPACE, storage=storage, features=_features(), models=_Models(_UsageFunctionModel(respond))) as runtime:
        session = await runtime.agents.get().create_session("session")
        assert (await session.run("confirmed first input", timeout_seconds=5)).result.status is ExecutionStatus.SUCCEEDED
        second = await session.start("unconfirmed cancelled input")
        try:
            await asyncio.wait_for(entered.wait(), 5)
            await second.cancel()
        finally:
            release.set()
        assert (await second.wait(timeout_seconds=5)).result.status is ExecutionStatus.CANCELLED
        assert (await session.run("follow-up input", timeout_seconds=5)).result.status is ExecutionStatus.SUCCEEDED
        assert len(seen) == 3
        following_context = repr(seen[2])
        assert "CONFIRMED_FIRST_RESPONSE" in following_context
        assert "UNCONFIRMED_CANCELLED_RESPONSE" not in following_context
        assert "unconfirmed cancelled input" not in following_context
        turns = (await session.timeline()).items
        assert len(turns) == 3 and turns[1].status is ExecutionStatus.CANCELLED
        assert turns[1].user_input["prompt"]["text"] == "unconfirmed cancelled input"
        assert turns[1].output is None and turns[1].items == ()
