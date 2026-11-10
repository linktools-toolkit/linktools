#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Terminal archive sharing preserves owner facts and immutable caller plans."""

import hashlib
from contextvars import ContextVar
from dataclasses import replace
from pathlib import Path

from pydantic_ai.messages import BinaryContent, ModelRequest, ModelResponse, TextPart, UserPromptPart

import pytest
from sqlalchemy import event, select

from linktools.ai.capability import TraceCapability
from pydantic_ai_harness.step_persistence import ContinuableSnapshot, InMemoryStepStore, RunRecord
from linktools.ai.core import ExecutionEventType, ExecutionStatus, StopReason, UsageMetrics
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime import Runtime
from linktools.ai.storage import StoredPayload
from linktools.ai.persistence import Persistence, PersistenceDomain, SqlStepStore
from linktools.ai.persistence._step_schema import build_step_sql_metadata
from linktools.ai.persistence.runs import OfficialStepReader
from linktools.ai.persistence._archive import ContentArchive, ContentRecord, PreparedArchive
from linktools.ai.persistence._checkpoint_persistence import CheckpointPersistence
from linktools.ai.persistence.contracts import (
    ConversationCursor, ExecutionTerminalCommit, ExecutionTerminalCommitResult, ExecutionRecoveryDetail,
    ExecutionRecoveryState, ResultRecord,
)
from linktools.ai.persistence._runtime_commands import StateCommands
from linktools.ai.persistence._model_interaction_store import ModelInteractionStateStepArchive
from linktools.ai.persistence._step_contracts import AgentRunCheckpoint, AgentRunRecord
from linktools.ai.persistence._model_interaction import (
    StagedContextProjection, StagedContextSpan, StagedModelInteraction,
)

from ._runtime_test_helpers import model_features, native_checkpoint
from ._session_tool_test_helpers import _application, _relevant_kinds, _split_sqlite_storage, _ToolModels
from .test_checkpoint_metadata_snapshot_io import _checkpoint
from .test_incremental_projection_boundaries import _storage
from .test_step_projection import _execution
from .test_session_admission import _session

pytestmark = pytest.mark.asyncio


async def test_terminal_shared_preparation_bounds_owner_reads_and_deduplicates_content(tmp_path: Path) -> None:
    state = Persistence(_storage("sqlite", tmp_path))
    await state.initialize(namespace="terminal-preparation", tenant_id="tenant")
    conversation = state.run_store.read_store(PersistenceDomain.CONVERSATION)
    trace = state.run_store.read_store(PersistenceDomain.TRACE)
    run, checkpoint = AgentRunRecord("run"), _checkpoint(2)
    engine = conversation.state_store.storage_group.context.engine.sync_engine
    selected: list[str] = []

    def capture(_connection, _cursor, statement, _parameters, _context, _many):
        if statement.lstrip().upper().startswith(("SELECT", "WITH")):
            selected.append(statement)

    try:
        await state.execution.executions.create_execution(_execution())
        await conversation.transcript_repository.content_archive(create=True)
        await trace.transcript_repository.content_archive(create=True)
        batches, owner_reads = [], []
        for shared in (False, True):
            bundle = PreparedArchive() if shared else None
            selected.clear()
            event.listen(engine, "before_cursor_execute", capture)
            try:
                _, first = await conversation.prepare_conversation_checkpoint(run, checkpoint, archive=bundle)
                conversation_reads = len(selected)
                second = await trace.prepare_checkpoints(run, (checkpoint,), archive=bundle)
                owner_reads.append((conversation_reads, len(selected) - conversation_reads))
                batches.append((first, second))
            finally:
                event.remove(engine, "before_cursor_execute", capture)
            assert all("ai_step_messages" not in sql and "ai_step_sequences" not in sql for sql in selected)
        # Each owner captures its mutable head once; native content preparation
        # uses frozen bytes rather than querying prior message bodies.
        assert owner_reads == [(1, 1), (1, 1)]
        for separate, shared in zip(batches[0], batches[1], strict=True):
            assert replace(separate, checkpoints=tuple(
                replace(item, archive=PreparedArchive()) for item in separate
            )) == replace(shared, checkpoints=tuple(
                replace(item, archive=PreparedArchive()) for item in shared
            ))
        assert bundle is not None
        separate_roots = [
            root for batch in batches[0] for checkpoint in batch
            for root in checkpoint.archive.native_messages
        ]
        assert set(bundle.native_messages) == set(separate_roots)
        assert len(bundle.native_messages) < len(separate_roots)
        group = conversation.state_store.storage_group

        async def commit(transaction):
            await conversation.transcript_repository.publish_archive(bundle)
            for owner, batch in zip((conversation, trace), batches[1], strict=True):
                await owner.materialize_checkpoint_in_transaction(
                    transaction.transaction(owner.state_store), run,
                    replace(batch[0], archive=PreparedArchive()),
                    execution_id="execution" if owner is trace else None,
                )
            await conversation.transcript_repository.publish_native_archive(
                transaction.transaction(conversation.state_store), bundle, owner_written=True,
            )

        await group.mutate((state.execution.executions.state_store, conversation.state_store, trace.state_store), commit)
        for owner in (conversation, trace):
            latest = await owner.latest_checkpoint(agent_run_id="run")
            assert latest is not None and latest.messages == checkpoint.messages
            assert await owner.checkpoint_count(agent_run_id="run") == 1
    finally:
        await state.close()


@pytest.mark.parametrize(("domain", "override"), ((PersistenceDomain.CONVERSATION, "relocate"),
    (PersistenceDomain.CONVERSATION, "prepare"), (PersistenceDomain.TRACE, "prepare")))
async def test_terminal_shared_preparation_keeps_custom_owner_hooks(
    tmp_path: Path, domain: PersistenceDomain, override: str,
) -> None:
    state = Persistence(_storage("memory", tmp_path))
    await state.initialize(namespace="terminal-hooks", tenant_id="tenant")
    original = state.run_store.read_store(domain)
    calls: list[str] = []

    class CustomArchive(ModelInteractionStateStepArchive):
        async def relocate_conversation_checkpoint(self, run, checkpoint):
            calls.append("relocate")
            result = await super().relocate_conversation_checkpoint(run, checkpoint)
            return replace(result, step_index=19) if override == "relocate" else result

        async def prepare_checkpoints(self, run, checkpoints):
            calls.append("prepare")
            result = await super().prepare_checkpoints(run, checkpoints)
            return replace(result, target_transcript_message_count=19) if override == "prepare" else result

    custom = CustomArchive(
        original.state_store, object_store=None, namespace="terminal-hooks",
        tenant_id="tenant", runtime_domain=domain,
    )
    try:
        if domain is PersistenceDomain.CONVERSATION:
            _, prepared = await custom.prepare_conversation_checkpoint(
                AgentRunRecord("run"), _checkpoint(1), archive=PreparedArchive(),
            )
            assert calls == ["relocate", "prepare"]
        else:
            prepared = await custom.prepare_checkpoints(AgentRunRecord("run"), (_checkpoint(1),))
            assert calls == ["prepare"]
        assert (prepared[0].stored.step_index if override == "relocate"
                else prepared.target_transcript_message_count) == 19
    finally:
        await state.close()


@pytest.mark.parametrize("backend", ("memory", "sqlite", "split"))
@pytest.mark.parametrize("lost_ack", (False, True))
async def test_terminal_archive_batch_preserves_tool_history_and_caller_proofs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend: str, lost_ack: bool,
) -> None:
    storage = (_split_sqlite_storage(tmp_path / "split.db") if backend == "split"
               else _storage(backend, tmp_path))
    calls: list[str] = []
    application = _application(calls)
    native = native_checkpoint()
    application.capability(native, scope="runtime")
    application.capability(TraceCapability(), scope="runtime")
    terminal = ContextVar("terminal_archive_test", default=False)
    publications: list[list[tuple[str, str]]] = []
    run_publications: list[tuple[int, str]] = []
    verified_snapshots: list[ContinuableSnapshot] = []
    original_recovery = CheckpointPersistence.materialize_recovery_checkpoint
    original_terminal = StateCommands.commit_terminal_checkpoint
    original_publish = ContentArchive.publish
    original_native_publish = ContentArchive.publish_native
    checked_plans = 0

    async def publish(self, transaction, prepared, *, native=True):
        run_publications.append((id(self.store.storage_group), self.scope_id))
        if terminal.get():
            publications[-1].append(("archive", self.scope_id))
        return await original_publish(self, transaction, prepared, native=native)

    async def publish_native(self, transaction, prepared):
        if prepared.native_messages:
            run_publications.append((id(self.store.storage_group), self.scope_id))
            if terminal.get():
                publications[-1].append(("native", self.scope_id))
        return await original_native_publish(self, transaction, prepared)

    async def materialize_recovery(self, *, agent_run_id, require_complete, captured=None):
        before = await self.reader.latest_snapshot(run_id=agent_run_id, include_interrupted=True)
        raw_before = await native.store.latest_snapshot(run_id=agent_run_id, include_interrupted=True)
        await original_recovery(self, agent_run_id=agent_run_id, require_complete=require_complete, captured=captured)
        assert await self.reader.latest_snapshot(run_id=agent_run_id, include_interrupted=True) == before
        assert await native.store.latest_snapshot(run_id=agent_run_id, include_interrupted=True) == raw_before
        if require_complete:
            assert before is not None and before.state == "complete"
            verified_snapshots.append(before)

    async def commit(self, value, **kwargs):
        nonlocal checked_plans
        projections = tuple(kwargs.get("execution_projections", ()))
        bundles = {id(bundle): bundle for projection in projections for bundle in (
            projection.archive,
            *(item.archive for item in projection.checkpoints),
            *((projection.observation.archive,) if projection.observation is not None else ()),
        )}
        snapshots = []
        for bundle in bundles.values():
            for record in bundle.records.values():
                if record.kind == "archive_content":
                    bundle.verify_content(record)
            snapshots.append((bundle, bundle.owner, dict(bundle.records), dict(bundle.nodes),
                              tuple(bundle.content_verified(record) for record in bundle.records.values())))
        publications.append([])
        token = terminal.set(True)
        try:
            result = await original_terminal(self, value, **kwargs)
        finally:
            terminal.reset(token)
        if backend != "split":
            for bundle, owner, records, nodes, receipts in snapshots:
                assert bundle.owner == owner and bundle.records == records and bundle.nodes == nodes
                assert tuple(bundle.content_verified(record) for record in bundle.records.values()) == receipts
            checked_plans += bool(projections)
        return result

    monkeypatch.setattr(ContentArchive, "publish", publish)
    monkeypatch.setattr(ContentArchive, "publish_native", publish_native)
    monkeypatch.setattr(CheckpointPersistence, "materialize_recovery_checkpoint", materialize_recovery)
    monkeypatch.setattr(StateCommands, "commit_terminal_checkpoint", commit)
    async with Persistence.open('terminal-archive-tool', storage=storage) as persistence_211, Runtime.attach(persistence=persistence_211, features=model_features(_ToolModels(), application)) as runtime:
        await runtime.agents.get("default").create_session("session")
        session = runtime.agents.get("default").session("session")
        group = runtime._state.execution.executions.state_store.storage_group
        mutate = group.mutate
        lost = False

        async def lose_response(stores, callback):
            nonlocal lost
            result = await mutate(stores, callback)
            if lost_ack and not lost and isinstance(result, ExecutionTerminalCommitResult):
                lost = True
                raise AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN)
            return result

        monkeypatch.setattr(group, "mutate", lose_response)
        run_publications.clear()
        execution = await session.start("inspect", idempotency_key="turn")
        result = (await execution.wait(timeout_seconds=10)).result
        assert result.status is ExecutionStatus.SUCCEEDED
        assert calls == ["lookup"] and lost == lost_ack
        expected = ["user", "tool_call", "tool_result", "assistant"]
        assert _relevant_kinds((await session.history()).items) == expected
        history = await runtime.history.history(
            execution.execution_id, principal=runtime.default_principal, include_content=True,
        )
        assert _relevant_kinds(history.items) == expected
        if backend == "split":
            retained_scopes = set()
            for domain in (PersistenceDomain.TRACE, PersistenceDomain.CONVERSATION):
                owner = runtime._run_store.read_store(domain)
                content = await owner.transcript_repository.content_archive()
                retained_scopes.add((id(owner.state_store.storage_group), content.scope_id))
                table = build_step_sql_metadata().tables["ai_step_messages"]
                async with owner.state_store.context.engine.connect() as connection:
                    assert (await connection.execute(select(table.c.id).where(
                        table.c.store_digest == SqlStepStore.scope_digest(
                            namespace=runtime.namespace, tenant_id=runtime.tenant_id,
                        ),
                    ).limit(1))).first() is not None
                assert await owner.list_agent_runs()
            assert len(retained_scopes) > 1
            assert retained_scopes.issubset(set(run_publications))
        else:
            assert checked_plans == 1
            assert len(publications) == 1
            assert len({scope for _kind, scope in publications[0]}) == 1
            assert len(publications[0]) == len(set(publications[0]))
        assert verified_snapshots
        for snapshot in verified_snapshots:
            assert await OfficialStepReader(native.store).latest_snapshot(run_id=snapshot.run_id) == snapshot


@pytest.mark.parametrize("foreign_scope", (False, True))
@pytest.mark.parametrize("owner_conflict", (False, True))
async def test_terminal_archive_batch_keeps_distinct_content_scopes_independent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, foreign_scope: bool, owner_conflict: bool,
) -> None:
    state = Persistence(_storage("sqlite", tmp_path))
    await state.initialize(namespace="terminal-scopes", tenant_id="tenant")
    trace = state.run_store.read_store(PersistenceDomain.TRACE)
    original = state.run_store.read_store(PersistenceDomain.CONVERSATION)
    conversation = ModelInteractionStateStepArchive(
        original.state_store, object_store=None,
        namespace="foreign" if foreign_scope else "terminal-scopes",
        tenant_id="tenant", runtime_domain=PersistenceDomain.CONVERSATION,
    )
    current = replace(_execution(), session_id="session", conversation=True,
                      conversation_agent_run_id="run", restart_checkpoints=True, recovery=ExecutionRecoveryDetail(agent_run_id="run"))
    run = AgentRunRecord("run", metadata={"history_id": "history"})
    checkpoint = _checkpoint(2)
    result = ResultRecord(StoredPayload.inline_text("done"), StopReason.END_TURN, UsageMetrics(), current.updated_at)
    commit = ExecutionTerminalCommit(
        current.revision, current.event_seq,
        replace(current, status=ExecutionStatus.SUCCEEDED),
        result, ExecutionEventType.EXECUTION_SUCCEEDED, {"agent_run_id": "run"},
    )
    native = InMemoryStepStore()
    state.run_store.bind_step_store(native)
    await native.register_run(RunRecord(run_id="run"))
    snapshot = ContinuableSnapshot(run_id="run", step_index=2, messages=checkpoint.messages)
    await native.save_snapshot(snapshot)
    published_scopes: list[tuple[str, str]] = []
    publish = ContentArchive.publish
    publish_native = ContentArchive.publish_native

    async def capture(self, transaction, prepared, *, native=True):
        published_scopes.append(("archive", self.content_scope))
        return await publish(self, transaction, prepared, native=native)

    async def capture_native(self, transaction, prepared):
        if prepared.native_messages:
            published_scopes.append(("native", self.content_scope))
        return await publish_native(self, transaction, prepared)

    try:
        await state.execution.executions.create_execution(current)
        await state.conversation.sessions.create(replace(_session(), history_id="history"))
        await state.conversation.sessions.admit_execution(
            "session", tenant_id="tenant", execution_id="execution", expected=None,
        )
        commands = StateCommands(
            state.execution.executions, namespace="terminal-scopes", events=state.execution.events,
            checkpoint=state.checkpoint, execution_run_store=trace,
            conversation=state.conversation.sessions, conversation_history=state.conversation.histories,
            conversation_run_store=conversation, background_tasks=set(),
        )
        monkeypatch.setattr(ContentArchive, "publish", capture)
        monkeypatch.setattr(ContentArchive, "publish_native", capture_native)

        async def commit_terminal():
            return await commands.commit_terminal_checkpoint(
                commit, expected_execution=current, session_id="session",
                next_cursor=ConversationCursor("run", "history"),
                conversation_agent_run=run, conversation_checkpoint=checkpoint,
                execution_run=run, execution_checkpoints=(checkpoint,),
            )

        if owner_conflict:
            archive_baselines = []
            for owner in (trace, conversation):
                content = await owner.transcript_repository.content_archive()
                archive_baselines.append(await content.store.read(lambda tx: tx.scan_records()))

            async def reject_owner(*args, **kwargs):
                raise AIError(ErrorCode.STORAGE_CONFLICT)

            monkeypatch.setattr(conversation, "materialize_checkpoint_in_transaction", reject_owner)
            with pytest.raises(AIError) as raised:
                await commit_terminal()
            assert raised.value.code is ErrorCode.STORAGE_CONFLICT
            for owner, baseline in zip((trace, conversation), archive_baselines, strict=True):
                content = await owner.transcript_repository.content_archive()
                assert await content.store.read(lambda tx: tx.scan_records()) == baseline
                assert await owner.get_agent_run(agent_run_id="run") is None
            assert await state.execution.executions.get("execution", tenant_id="tenant") == current
            assert (await state.execution.executions.get("execution", tenant_id="tenant")).recovery == current.recovery
        else:
            committed = await commit_terminal()
            assert committed.execution.status is ExecutionStatus.SUCCEEDED
            assert len({scope for _kind, scope in published_scopes}) == (2 if foreign_scope else 1)
            assert len(published_scopes) == len(set(published_scopes))
            for owner in (trace, conversation):
                latest = await owner.latest_checkpoint(agent_run_id="run")
                assert latest is not None and latest.messages == checkpoint.messages
                assert await owner.checkpoint_count(agent_run_id="run") == 1
            assert committed.execution.recovery_state is ExecutionRecoveryState.COMPLETED
        assert await state.run_store.checkpoint_persistence.reader.latest_snapshot(run_id="run") == snapshot
    finally:
        await state.close()


async def test_terminal_archive_batch_does_not_open_scope_without_preparation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = Persistence(_storage("memory", tmp_path))
    await state.initialize(namespace="terminal-no-archive", tenant_id="tenant")
    current = _execution()
    result = ResultRecord(None, StopReason.ERROR, UsageMetrics(), current.updated_at)
    commit = ExecutionTerminalCommit(
        current.revision, current.event_seq,
        replace(current, status=ExecutionStatus.FAILED, error_code=ErrorCode.EXECUTION_FAILED.value),
        result, ExecutionEventType.EXECUTION_FAILED,
        {"error_code": ErrorCode.EXECUTION_FAILED.value, "safe_error_details": {}},
    )
    try:
        await state.execution.executions.create_execution(current)
        group = state.execution.executions.state_store.storage_group
        scope_before = await group.open_archive()
        open_archive = group.open_archive

        async def existing_archive_only(*, create=False):
            assert not create, "unused terminal participant must not prepare its archive"
            return await open_archive(create=create)

        monkeypatch.setattr(group, "open_archive", existing_archive_only)
        commands = StateCommands(
            state.execution.executions, namespace="terminal-no-archive", events=state.execution.events,
            execution_run_store=state.run_store.read_store(PersistenceDomain.TRACE), background_tasks=set(),
        )
        committed = await commands.commit_terminal_checkpoint(
            commit, expected_execution=current, execution_checkpoints=(_checkpoint(1),),
        )
        assert committed.execution.status is ExecutionStatus.FAILED
        assert await group.open_archive() == scope_before
    finally:
        await state.close()


@pytest.mark.parametrize("fault", ("model", "usage", "request", "missing_body"))
@pytest.mark.parametrize("with_baseline", (False, True))
async def test_terminal_cold_replay_rejects_changed_facts_and_missing_context(
    tmp_path: Path, fault: str, with_baseline: bool,
) -> None:
    database = tmp_path / "cold-split.db"
    state = Persistence(_split_sqlite_storage(database))
    await state.initialize(namespace="terminal-cold-replay", tenant_id="tenant")
    run = AgentRunRecord("run")
    timestamp = run.started_at
    messages = (
        ModelRequest(parts=[UserPromptPart(["retained request " * 2000,
                                            BinaryContent(b"request image", media_type="image/png")], timestamp=timestamp)]),
        ModelResponse(parts=[TextPart("response")], timestamp=timestamp),
        ModelRequest(parts=[UserPromptPart("next request", timestamp=timestamp)]),
    )
    envelope = b"{}"
    staged = StagedModelInteraction(
        "run", 1, 1, "agent", None, {"route_id": "default"},
        StagedContextProjection((StagedContextSpan(0, 1),)), hashlib.sha256(envelope).hexdigest(),
        StagedContextProjection((StagedContextSpan(1, 2),)), "SUCCEEDED", None, 1,
        UsageMetrics(model_requests=1, input_tokens=2, output_tokens=1), timestamp, timestamp,
    )
    archive = state.run_store.read_store(PersistenceDomain.TRACE)
    prepared = PreparedArchive()
    try:
        await state.execution.executions.create_execution(_execution())
        interactions = await archive.prepare_interactions(
            run, (staged,), lambda _: envelope,
            local_message_count=2, local_messages=messages[:2], prepared=prepared,
        )
        request_object = next(
            reference for native in prepared.native_messages.values()
            for _identity, content in native.messages
            for reference in content.object_refs()
        )
        assert request_object is not None
        await archive.sync_projection(
            run, events=(), checkpoints=(AgentRunCheckpoint(
                "run", 1, list(messages[:2]), timestamp=timestamp, transcript_message_count_before=0,
            ),), interactions=interactions, archive=prepared, execution_id="execution",
        )
        await archive.materialize_checkpoint(run, AgentRunCheckpoint(
            "run", 2, list(messages), timestamp=timestamp, transcript_message_count_before=2,
        ), execution_id="execution")
        original = await archive.list_model_interactions(agent_run_id="run")
        assert original[0].request_context.items[0].root.count == 1
    finally:
        await state.close()

    cold = Persistence(_split_sqlite_storage(database))
    await cold.initialize(namespace="terminal-cold-replay", tenant_id="tenant")
    archive = cold.run_store.read_store(PersistenceDomain.TRACE)
    try:
        current_messages = messages
        if fault == "model":
            staged = replace(staged, model={"route_id": "changed"})
        elif fault == "usage":
            staged = replace(staged, usage=UsageMetrics(model_requests=1, input_tokens=99, output_tokens=1))
        elif fault == "request":
            current_messages = (ModelRequest(parts=[UserPromptPart(
                "changed request", timestamp=timestamp,
            )]), *messages[1:])
        else:
            objects = cold.archive_object_store(PersistenceDomain.TRACE)
            assert await objects.delete_object(request_object.key, expected_digest=request_object.digest)
            assert await objects.stat(request_object.key) is None
        content_write = PreparedArchive()
        with pytest.raises(AIError) as raised:
            replayed = await archive.prepare_interactions(
                run, (staged,), lambda _: envelope, local_message_count=3,
                local_messages=current_messages, prepared=content_write, replay=True,
                baseline=original[0].request_context if with_baseline else None,
            )
            await archive.sync_projection(
                run, events=(), checkpoints=(), interactions=replayed, archive=content_write,
                execution_id="execution",
            )
        assert raised.value.code is (
            ErrorCode.STORAGE_NOT_FOUND if fault == "missing_body" else ErrorCode.STORAGE_INTEGRITY_ERROR
        )
        assert await archive.list_model_interactions(agent_run_id="run") == original
    finally:
        await cold.close()
