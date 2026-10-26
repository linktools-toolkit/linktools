#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from linktools.ai.persistence import Persistence
"""Terminal archive sharing preserves owner facts and immutable caller plans."""
from ._runtime_test_helpers import model_features

import hashlib
from contextvars import ContextVar
from dataclasses import replace
from pathlib import Path

from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart

import pytest
from sqlalchemy import event

from linktools.ai.capability import CheckpointCapability, TraceCapability
from linktools.ai.core import ExecutionEventType, ExecutionStatus, StopReason, UsageMetrics
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime import Runtime
from linktools.ai.persistence import PersistenceDomain
from linktools.ai.persistence import Persistence
from linktools.ai.persistence._archive import ContentArchive, ContentRecord, PreparedArchive
from linktools.ai.persistence._checkpoint_persistence import CheckpointPersistence
from linktools.ai.persistence.contracts import (
    ExecutionTerminalCommit, ExecutionTerminalCommitResult, RecoveryCheckpoint,
    RecoveryCheckpointState, ResultRecord,
)
from linktools.ai.persistence._runtime_commands import StateCommands
from linktools.ai.persistence._model_interaction_store import ModelInteractionStateStepArchive
from linktools.ai.persistence._step_contracts import AgentRunCheckpoint, AgentRunRecord
from linktools.ai.persistence._model_interaction import (
    StagedContextProjection, StagedContextSpan, StagedModelInteraction,
)

from ._session_tool_test_helpers import _application, _relevant_kinds, _split_sqlite_storage, _ToolModels
from .test_checkpoint_metadata_snapshot_io import _checkpoint
from .test_incremental_projection_boundaries import _storage
from .test_step_projection import _execution

pytestmark = pytest.mark.asyncio


async def test_terminal_shared_preparation_reduces_reads_and_preserves_owner_values(tmp_path: Path) -> None:
    state = Persistence(_storage("sqlite", tmp_path))
    await state.initialize(namespace="terminal-preparation", tenant_id="tenant")
    conversation = state.run_store.read_store(PersistenceDomain.CONVERSATION)
    recovery = state.run_store.read_store(PersistenceDomain.CHECKPOINT)
    run, checkpoint = AgentRunRecord("run"), _checkpoint(2)
    engine = conversation.state_store.storage_group.context.engine.sync_engine
    selected: list[str] = []

    def capture(_connection, _cursor, statement, _parameters, _context, _many):
        if statement.startswith("SELECT"):
            selected.append(statement)

    try:
        await conversation.transcript_repository.content_archive(create=True)
        await recovery.transcript_repository.content_archive(create=True)
        batches, counts = [], []
        for shared in (False, True):
            bundle = PreparedArchive() if shared else None
            selected.clear()
            event.listen(engine, "before_cursor_execute", capture)
            try:
                _, first = await conversation.prepare_conversation_checkpoint(run, checkpoint, archive=bundle)
                _, second = await recovery.prepare_run_checkpoint(run, checkpoint, archive=bundle)
                batches.append((first, second))
            finally:
                event.remove(engine, "before_cursor_execute", capture)
            counts.append(len(selected))
        assert counts[1] < counts[0]
        for separate, shared in zip(batches[0], batches[1], strict=True):
            assert replace(separate, checkpoints=tuple(
                replace(item, archive=PreparedArchive()) for item in separate
            )) == replace(shared, checkpoints=tuple(
                replace(item, archive=PreparedArchive()) for item in shared
            ))
        assert bundle is not None
        group = conversation.state_store.storage_group

        async def commit(transaction):
            await conversation.transcript_repository.publish_archive(bundle)
            for owner, batch in zip((conversation, recovery), batches[1], strict=True):
                await owner.materialize_checkpoint_in_transaction(
                    transaction.transaction(owner.state_store), run,
                    replace(batch[0], archive=PreparedArchive()),
                )

        await group.mutate((conversation.state_store, recovery.state_store), commit)
        for owner in (conversation, recovery):
            latest = await owner.latest_checkpoint(agent_run_id="run")
            assert latest is not None and latest.messages == checkpoint.messages
            assert await owner.checkpoint_count(agent_run_id="run") == 1
    finally:
        await state.close()


@pytest.mark.parametrize("domain", (PersistenceDomain.CONVERSATION, PersistenceDomain.CHECKPOINT))
@pytest.mark.parametrize("override", ("relocate", "prepare"))
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

        async def relocate_run_checkpoint(self, run, checkpoint):
            calls.append("relocate")
            result = await super().relocate_run_checkpoint(run, checkpoint)
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
        method = (custom.prepare_conversation_checkpoint if domain is PersistenceDomain.CONVERSATION
                  else custom.prepare_run_checkpoint)
        _, prepared = await method(AgentRunRecord("run"), _checkpoint(1), archive=PreparedArchive())
        assert calls == ["relocate", "prepare"]
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
    application.capability(CheckpointCapability(), scope="runtime")
    application.capability(TraceCapability(), scope="runtime")
    terminal = ContextVar("terminal_archive_test", default=False)
    publications: list[list[str]] = []
    run_publications: list[tuple[int, str]] = []
    replayed_contexts: list[tuple[object, object]] = []
    original_recovery = CheckpointPersistence.materialize_recovery_checkpoint
    original_terminal = StateCommands.commit_terminal_checkpoint
    original_publish = ContentArchive.publish
    checked_plans = 0

    async def publish(self, transaction, prepared):
        run_publications.append((id(self.store.storage_group), self.scope_id))
        if terminal.get():
            publications[-1].append(self.scope_id)
        return await original_publish(self, transaction, prepared)

    async def materialize_recovery(self, *, agent_run_id, require_complete):
        archive = self.archive
        before = (await archive.list_model_interactions(agent_run_id=agent_run_id)
                  if isinstance(archive, ModelInteractionStateStepArchive) else ())
        await original_recovery(self, agent_run_id=agent_run_id, require_complete=require_complete)
        if before:
            after = {item.model_request_seq: item for item in await archive.list_model_interactions(
                agent_run_id=agent_run_id,
            )}
            for previous in before:
                if previous.status != "RUNNING":
                    observed = after[previous.model_request_seq]
                    assert observed.request_context == previous.request_context
                    assert observed.response_context == previous.response_context
                    replayed_contexts.append((previous.request_context, previous.response_context))

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
    monkeypatch.setattr(CheckpointPersistence, "materialize_recovery_checkpoint", materialize_recovery)
    monkeypatch.setattr(StateCommands, "commit_terminal_checkpoint", commit)
    async with Persistence.open('terminal-archive-tool', storage=storage) as persistence_211, Runtime.open(persistence=persistence_211, features=model_features(_ToolModels(), application)) as runtime:
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
            for domain in (PersistenceDomain.TRACE, PersistenceDomain.CONVERSATION, PersistenceDomain.CHECKPOINT):
                owner = runtime._run_store.read_store(domain)
                content = await owner.transcript_repository.content_archive()
                retained_scopes.add((id(owner.state_store.storage_group), content.scope_id))
                assert any(record.kind == "archive_content" for record in
                           await content.store.read(lambda transaction: transaction.scan_records()))
                assert await owner.list_agent_runs()
            assert len(retained_scopes) > 1
            assert retained_scopes.issubset(set(run_publications))
            assert replayed_contexts
        else:
            assert checked_plans == 1
            assert [len(scopes) for scopes in publications] == [1]


@pytest.mark.parametrize("foreign_scope", (False, True))
@pytest.mark.parametrize("owner_conflict", (False, True))
async def test_terminal_archive_batch_keeps_distinct_content_scopes_independent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, foreign_scope: bool, owner_conflict: bool,
) -> None:
    state = Persistence(_storage("sqlite", tmp_path))
    await state.initialize(namespace="terminal-scopes", tenant_id="tenant")
    trace = state.run_store.read_store(PersistenceDomain.TRACE)
    original = state.run_store.read_store(PersistenceDomain.CHECKPOINT)
    recovery = ModelInteractionStateStepArchive(
        original.state_store, object_store=None,
        namespace="foreign" if foreign_scope else "terminal-scopes",
        tenant_id="tenant", runtime_domain=PersistenceDomain.CHECKPOINT,
    )
    current = _execution()
    run, checkpoint = AgentRunRecord("run"), _checkpoint(2)
    source = RecoveryCheckpoint(
        "execution", "run", RecoveryCheckpointState.ACTIVE, 0,
        current.created_at, current.updated_at,
    )
    target = replace(source, state=RecoveryCheckpointState.COMPLETED, revision=1)
    result = ResultRecord(None, StopReason.ERROR, UsageMetrics(), current.updated_at)
    commit = ExecutionTerminalCommit(
        current.revision, current.event_seq,
        replace(current, status=ExecutionStatus.FAILED, error_code=ErrorCode.EXECUTION_FAILED.value),
        result, ExecutionEventType.EXECUTION_FAILED,
        {"error_code": ErrorCode.EXECUTION_FAILED.value, "safe_error_details": {}},
    )
    published_scopes: list[str] = []
    publish = ContentArchive.publish

    async def capture(self, transaction, prepared):
        published_scopes.append(self.content_scope)
        return await publish(self, transaction, prepared)

    try:
        await state.execution.executions.create_with_history_head(current)
        await state.recovery.checkpoints.create(source)
        commands = StateCommands(
            state.execution.executions, namespace="terminal-scopes", events=state.execution.events,
            recovery=state.recovery.checkpoints, execution_run_store=trace,
            recovery_run_store=recovery, background_tasks=set(),
        )
        monkeypatch.setattr(ContentArchive, "publish", capture)
        async def commit_terminal():
            return await commands.commit_terminal_checkpoint(
                commit, expected_execution=current,
                recovery_checkpoint=target, recovery_run=run, recovery_agent_run_checkpoint=checkpoint,
                execution_run=run, execution_checkpoints=(checkpoint,),
            )

        if owner_conflict:
            async def reject_owner(*args, **kwargs):
                raise AIError(ErrorCode.STORAGE_CONFLICT)

            monkeypatch.setattr(recovery, "materialize_checkpoint_in_transaction", reject_owner)
            with pytest.raises(AIError) as raised:
                await commit_terminal()
            assert raised.value.code is ErrorCode.STORAGE_CONFLICT
            for owner in (trace, recovery):
                content = await owner.transcript_repository.content_archive()
                assert not await content.store.read(lambda tx: tx.scan_records())
                assert await owner.get_agent_run(agent_run_id="run") is None
            assert await state.execution.executions.get("execution", tenant_id="tenant") == current
            assert await state.recovery.checkpoints.get("execution", tenant_id="tenant") == source
            return
        committed = await commit_terminal()
        assert committed.execution.status is ExecutionStatus.FAILED
        assert len(published_scopes) == (2 if foreign_scope else 1)
        assert len(set(published_scopes)) == (2 if foreign_scope else 1)
        for owner in (trace, recovery):
            latest = await owner.latest_checkpoint(agent_run_id="run")
            assert latest is not None and latest.messages == checkpoint.messages
            assert await owner.checkpoint_count(agent_run_id="run") == 1
        assert await state.recovery.checkpoints.get("execution", tenant_id="tenant") == target
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
        await state.execution.executions.create_with_history_head(current)
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
async def test_terminal_cold_replay_rejects_changed_facts_and_missing_context(
    tmp_path: Path, fault: str,
) -> None:
    database = tmp_path / "cold-split.db"
    state = Persistence(_split_sqlite_storage(database))
    await state.initialize(namespace="terminal-cold-replay", tenant_id="tenant")
    run = AgentRunRecord("run")
    timestamp = run.started_at
    messages = (
        ModelRequest(parts=[UserPromptPart("retained request " * 2000, timestamp=timestamp)]),
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
    archive = state.run_store.read_store(PersistenceDomain.CHECKPOINT)
    prepared = PreparedArchive()
    try:
        interactions = await archive.prepare_interactions(
            run, (staged,), lambda _: envelope,
            local_message_count=2, local_messages=messages[:2], prepared=prepared,
        )
        request_object = next(
            content.payload.ref for record in prepared.records.values()
            if record.kind == "archive_content"
            for content in (ContentRecord.from_json(record.data),)
            if content.codec == "model-message" and content.raw_size > 16 * 1024
        )
        assert request_object is not None
        await archive.sync_projection(
            run, events=(), checkpoints=(AgentRunCheckpoint(
                "run", 1, list(messages[:2]), timestamp=timestamp, transcript_message_count_before=0,
            ),), interactions=interactions, archive=prepared,
        )
        await archive.materialize_checkpoint(run, AgentRunCheckpoint(
            "run", 2, list(messages), timestamp=timestamp, transcript_message_count_before=2,
        ))
        original = await archive.list_model_interactions(agent_run_id="run")
        assert original[0].request_context.items[0].root.count == 1
    finally:
        await state.close()

    cold = Persistence(_split_sqlite_storage(database))
    await cold.initialize(namespace="terminal-cold-replay", tenant_id="tenant")
    archive = cold.run_store.read_store(PersistenceDomain.CHECKPOINT)
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
            objects = cold.archive_object_store(PersistenceDomain.CHECKPOINT)
            assert await objects.delete_object(request_object.key, expected_digest=request_object.digest)
            assert await objects.stat(request_object.key) is None
        content_write = PreparedArchive()
        with pytest.raises(AIError) as raised:
            replayed = await archive.prepare_interactions(
                run, (staged,), lambda _: envelope, local_message_count=3,
                local_messages=current_messages, prepared=content_write, replay=True,
                baseline=original[0].request_context,
            )
            await archive.sync_projection(
                run, events=(), checkpoints=(), interactions=replayed, archive=content_write,
            )
        assert raised.value.code is (
            ErrorCode.STORAGE_NOT_FOUND if fault == "missing_body" else ErrorCode.STORAGE_INTEGRITY_ERROR
        )
        assert await archive.list_model_interactions(agent_run_id="run") == original
    finally:
        await cold.close()
