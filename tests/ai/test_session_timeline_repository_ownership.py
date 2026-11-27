#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Injected timeline repositories retain their independent task and read ownership."""

from linktools.ai.persistence import Persistence
import asyncio
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path

import pytest
from linktools.ai.core import (
    AuthorizationAction,
    ResourceRef,
    ExecutionStatus,
    HmacCursorSigner,
    Principal,
    TenantAuthorizationPolicy,
)
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.history import History
from linktools.ai.persistence import PersistenceStorage, PersistencePlan, PersistenceDomain
from linktools.ai.storage import StorageRoute
from linktools.ai.persistence.contracts import (
    ConversationCursor,
    ExecutionRecord,
    ExecutionRepository,
    SessionRecord,
    SessionRepository,
    SessionTurnCommit,
)
from linktools.ai.persistence._conversation_repositories import SessionRepositoryImpl
from linktools.ai.persistence._execution_repositories import ExecutionRepositoryImpl
from sqlalchemy import event

from .test_session_timeline import _commit_empty_turn, _execution, _materialize_conversation, _session
from .test_session_actor_identity import SharedSessionPolicy
from .test_session_timeline_window import _window_host

pytestmark = pytest.mark.asyncio


class ParallelExecutions:
    def __init__(self, repository: ExecutionRepository) -> None:
        self.state_store = repository.state_store
        self.tenant_id = repository.tenant_id
        self.namespace = repository.namespace
        self.get = repository.get
        self.get_header = repository.get_header
        self.get_result = repository.get_result

    async def get_timeline_many(
        self, execution_ids: Sequence[str], *, tenant_id: str,
    ) -> Mapping[str, ExecutionRecord]:
        records = await asyncio.gather(*(
            self.get(identity, tenant_id=tenant_id) for identity in execution_ids
        ))
        return {record.execution_id: record for record in records if record is not None}


class ParallelExecutionSubclass(ExecutionRepositoryImpl):
    async def get_timeline_many(
        self, execution_ids: Sequence[str], *, tenant_id: str,
    ) -> Mapping[str, ExecutionRecord]:
        records = await asyncio.gather(*(
            self.get(identity, tenant_id=tenant_id) for identity in execution_ids
        ))
        return {record.execution_id: record for record in records if record is not None}


class ParallelSessions:
    def __init__(self, repository: SessionRepository, phase: str) -> None:
        self.state_store = repository.state_store
        self.tenant_id = repository.tenant_id
        self.namespace = repository.namespace
        self.get_header = repository.get_header
        self.timeline_head = repository.timeline_head
        self.list_timeline_turns = repository.list_timeline_turns
        self.repository = repository
        self.phase = phase

    async def get(self, session_id: str, *, tenant_id: str) -> SessionRecord | None:
        read = self.repository.get(session_id, tenant_id=tenant_id)
        return (await asyncio.gather(read))[0] if self.phase == "session" else await read

    async def list_timeline_commits(
        self, session_id: str, *, tenant_id: str, start_turn_seq: int, end_turn_seq: int,
    ) -> tuple[SessionTurnCommit, ...]:
        read = self.repository.list_timeline_commits(
            session_id, tenant_id=tenant_id, start_turn_seq=start_turn_seq, end_turn_seq=end_turn_seq,
        )
        return (await asyncio.gather(read))[0] if self.phase == "commits" else await read


class ParallelSessionSubclass(SessionRepositoryImpl):
    phase = "session"

    async def get(self, session_id: str, *, tenant_id: str) -> SessionRecord | None:
        read = super().get(session_id, tenant_id=tenant_id)
        return (await asyncio.gather(read))[0] if self.phase == "session" else await read

    async def list_timeline_commits(
        self, session_id: str, *, tenant_id: str, start_turn_seq: int, end_turn_seq: int,
    ) -> tuple[SessionTurnCommit, ...]:
        read = super().list_timeline_commits(
            session_id, tenant_id=tenant_id, start_turn_seq=start_turn_seq, end_turn_seq=end_turn_seq,
        )
        return (await asyncio.gather(read))[0] if self.phase == "commits" else await read


def _history(
    state: Persistence, *, executions: ExecutionRepository | None = None,
    sessions: SessionRepository | None = None,
) -> History:
    return History(
        object(), tenant_id="tenant",
        executions=state.execution.executions if executions is None else executions,
        authorization=TenantAuthorizationPolicy(),
        cursor_signer=HmacCursorSigner("timeline", b"timeline-owner-test"),
        conversation=state.conversation if sessions is None else replace(state.conversation, sessions=sessions),
        session_transcript_store=state.run_store.conversation_persistence,
    )


async def _populate(state: Persistence) -> None:
    await state.initialize(namespace="timeline-owner", tenant_id="tenant")
    await state.conversation.sessions.create(_session())
    await state.execution.executions.create(
        _execution("failed", status=ExecutionStatus.FAILED, prompt="original"),
    )
    await state.conversation.sessions.admit_execution(
        "session", tenant_id="tenant", execution_id="failed", expected=None,
    )
    await _commit_empty_turn(state, await state.execution.executions.get("failed", tenant_id="tenant"))
    await state.conversation.sessions.release_execution(
        "session", tenant_id="tenant", execution_id="failed",
    )
    await state.execution.executions.create(
        _execution("active", status=ExecutionStatus.PENDING_START, prompt="pending prompt"),
    )
    await state.conversation.sessions.admit_execution(
        "session", tenant_id="tenant", execution_id="active", expected=None,
    )


@pytest.mark.parametrize("split", (False, True))
@pytest.mark.parametrize("subclass", (False, True))
@pytest.mark.parametrize("phase", ("executions", "session", "commits"))
async def test_public_history_accepts_parallel_injected_repository_reads(
    tmp_path: Path, split: bool, subclass: bool, phase: str,
) -> None:
    state = PersistenceStorage.from_plan(PersistencePlan(
        execution=StorageRoute.sqlite(tmp_path / "execution.db"),
    )) if split else PersistenceStorage.in_memory()
    try:
        state = Persistence(state)
        await _populate(state)
        conversation_group = state.conversation.sessions.state_store.storage_group
        execution_group = state.execution.executions.state_store.storage_group
        assert (conversation_group is not execution_group) is split
        principal = Principal("owner", "tenant")
        expected = (await _history(state).session_detail("session", principal=principal)).timeline
        injected = {}
        if phase == "executions":
            repository = state.execution.executions
            injected["executions"] = ParallelExecutionSubclass(
                repository.state_store, namespace=repository.namespace, tenant_id=repository.tenant_id,
            ) if subclass else ParallelExecutions(repository)
        else:
            repository = state.conversation.sessions
            adapter = ParallelSessionSubclass(
                repository.state_store, namespace=repository.namespace, tenant_id=repository.tenant_id,
            ) if subclass else ParallelSessions(repository, phase)
            adapter.phase = phase
            injected["sessions"] = adapter
        actual = (await _history(state, **injected).session_detail("session", principal=principal)).timeline
        assert actual == expected
        assert [turn.execution_id for turn in actual.items] == ["failed", "active"]
        assert [turn.conversation_committed for turn in actual.items] == [True, False]
    finally:
        await state.close()


async def test_native_session_metadata_reads_one_snapshot_without_history_bodies(
    tmp_path: Path,
) -> None:
    async with _window_host(tmp_path) as (state, runtime, principal, waiting, proceed):
        finished = await runtime.sessions.start(
            "session", "finished", agent="default", principal=principal, idempotency_key="finished",
        )
        assert (await finished.wait_final(timeout_seconds=5)).status is ExecutionStatus.SUCCEEDED
        active = await runtime.sessions.start(
            "session", "pending", agent="default", principal=principal, idempotency_key="pending",
        )
        await asyncio.wait_for(waiting.wait(), 5)

        engine = state.session_turns._context.engine.sync_engine
        statements, begins = [], []

        def statement(_connection, _cursor, sql, _parameters, _context, _many):
            if sql.lstrip().upper().startswith(("SELECT", "WITH")):
                statements.append(sql)

        on_begin = begins.append
        event.listen(engine, "before_cursor_execute", statement)
        event.listen(engine, "begin", on_begin)
        try:
            result = await runtime.history.turn_session_state("session", principal=principal)
        finally:
            event.remove(engine, "begin", on_begin)
            event.remove(engine, "before_cursor_execute", statement)
            proceed.set()
        assert result["last_turn_no"] == 1
        assert result["active_turn_id"] == active.key.turn_id
        assert len(begins) == 1
        assert not any("ai_messages" in sql or "ai_object_chunks" in sql for sql in statements)
        assert (await active.wait_final(timeout_seconds=5)).status is ExecutionStatus.SUCCEEDED


async def test_exact_turn_distinguishes_active_pending_from_recovery() -> None:
    state = Persistence(PersistenceStorage.in_memory())
    try:
        await _populate(state)
        history = _history(state)
        owner = Principal("owner", "tenant")
        with pytest.raises(AIError) as pending:
            await history.session_turn("session", "active", principal=owner)
        assert pending.value.code is ErrorCode.SESSION_TURN_PENDING

        with pytest.raises(AIError) as denied:
            await history.session_turn("session", "active", principal=Principal("other", "tenant"))
        assert denied.value.code is ErrorCode.AUTHORIZATION_DENIED

        sessions = state.conversation.sessions
        no_fact = replace(
            _execution("no-fact", status=ExecutionStatus.STARTED, prompt="missing admission fact"),
            session_id="no-fact-session",
        )
        await sessions.create(replace(_session(), session_id="no-fact-session", active_execution_id="no-fact"))
        await state.execution.executions.create(no_fact)
        with pytest.raises(AIError) as unavailable:
            await history.session_turn("no-fact-session", "no-fact", principal=owner)
        assert unavailable.value.code is ErrorCode.SESSION_HISTORY_UNAVAILABLE

        await sessions.release_execution("session", tenant_id="tenant", execution_id="active")
        for status in (ExecutionStatus.RECOVERY_REQUIRED, ExecutionStatus.FAILED):
            broken = _execution(f"broken-{status}", status=status, prompt="recover")
            await state.execution.executions.create(broken)
            await sessions.admit_execution("session", tenant_id="tenant", execution_id=broken.execution_id, expected=None)
            with pytest.raises(AIError) as recovery:
                await history.session_turn("session", broken.execution_id, principal=owner)
            assert recovery.value.code is ErrorCode.STORAGE_RECOVERY_REQUIRED
            await sessions.release_execution("session", tenant_id="tenant", execution_id=broken.execution_id)
    finally:
        await state.close()


async def test_releasing_published_failed_turn_advances_the_session_once() -> None:
    state = Persistence(PersistenceStorage.in_memory())
    try:
        await state.initialize(namespace="timeline-owner", tenant_id="tenant")
        sessions = state.conversation.sessions
        await sessions.create(_session())
        execution = _execution("failed", status=ExecutionStatus.FAILED, prompt="failed")
        await state.execution.executions.create(execution)
        admitted = await sessions.admit_execution(
            "session", tenant_id="tenant", execution_id=execution.execution_id, expected=None,
        )
        with pytest.raises(AIError) as missing_commit:
            await sessions.release_execution(
                "session", tenant_id="tenant", execution_id=execution.execution_id, require_turn_commit=True,
            )
        assert missing_commit.value.code is ErrorCode.STORAGE_RECOVERY_REQUIRED
        assert (await sessions.get("session", tenant_id="tenant")).active_execution_id == execution.execution_id

        async def publish(transaction):
            return await sessions.publish_and_release_turn_in_transaction(
                transaction, "session", tenant_id="tenant", execution_id=execution.execution_id,
                start_message_index=0, end_message_index=0, history_unavailable=False,
            )

        released, receipt = await sessions.state_store.mutate(publish)
        assert receipt is not None and receipt.execution_id == execution.execution_id
        assert released.active_execution_id is None
        assert released.revision == admitted.revision + 1
        assert (await _history(state).session_turn(
            "session", execution.execution_id, principal=Principal("owner", "tenant"),
        )).status is ExecutionStatus.FAILED

        repeated = await sessions.release_execution(
            "session", tenant_id="tenant", execution_id=execution.execution_id, require_turn_commit=True,
        )
        assert repeated == released
    finally:
        await state.close()


async def test_exact_committed_turn_remains_stable_after_a_newer_turn() -> None:
    state = Persistence(PersistenceStorage.in_memory())
    try:
        await _populate(state)
        sessions = state.conversation.sessions
        await sessions.release_execution("session", tenant_id="tenant", execution_id="active")
        newer = _execution("newer", status=ExecutionStatus.SUCCEEDED, prompt="newer prompt")
        await state.execution.executions.create(newer)
        await sessions.admit_execution(
            "session", tenant_id="tenant", execution_id="newer", expected=None,
        )
        await _commit_empty_turn(state, newer)
        await sessions.release_execution("session", tenant_id="tenant", execution_id="newer")
        await sessions.create(replace(_session(), session_id="other-session"))
        principal = Principal("owner", "tenant")
        history = _history(state)
        timeline = (await history.session_detail("session", principal=principal)).timeline
        older = next(turn for turn in timeline.items if turn.execution_id == "failed")
        latest = (await history.session_detail("session", principal=principal, limit=1)).timeline
        assert older.conversation_committed is True
        assert latest.items[0].conversation_committed is True
        assert latest.items[0].execution_id == "newer"
        assert latest.items[0] != older
        assert await history.session_turn("session", "failed", principal=principal) == older
        assert await history.session_turn("session", "newer", principal=principal) == latest.items[0]

        for session_id, execution_id, error_code in (
            ("session", "active", ErrorCode.STORAGE_RECOVERY_REQUIRED),
            ("other-session", "failed", ErrorCode.AUTHORIZATION_DENIED),
        ):
            with pytest.raises(AIError) as captured:
                await history.session_turn(session_id, execution_id, principal=principal)
            assert captured.value.code is error_code
    finally:
        await state.close()


class SessionOnlyReaderPolicy(SharedSessionPolicy):
    async def authorize(
        self, principal: Principal, action: AuthorizationAction, resource: ResourceRef,
    ) -> None:
        if principal.principal_id == "peer" and action is AuthorizationAction.EXECUTION_READ:
            raise AIError(ErrorCode.AUTHORIZATION_DENIED)
        await super().authorize(principal, action, resource)


async def test_exact_turn_uses_session_visibility_for_committed_tool_content() -> None:
    state = Persistence(PersistenceStorage.in_memory())
    await state.initialize(namespace="session-timeline", tenant_id="tenant")
    try:
        sessions = state.conversation.sessions
        created = await sessions.create(replace(_session(), session_id="shared"))
        execution = replace(_execution(
            "success", status=ExecutionStatus.SUCCEEDED, prompt="original prompt",
        ), session_id="shared")
        await state.execution.executions.create(execution)
        await sessions.admit_execution(
            "shared", tenant_id="tenant", execution_id="success", expected=None,
        )
        assert created.history_id is not None
        agent_run_id, message_count = await _materialize_conversation(state, created.history_id)
        archive = state.run_store.read_store(PersistenceDomain.CONVERSATION)
        checkpoint_sequence = await archive.latest_checkpoint_sequence(agent_run_id=agent_run_id)
        checkpoint = await archive.checkpoint_at_sequence(
            agent_run_id=agent_run_id, checkpoint_sequence=checkpoint_sequence,
        )

        async def commit(transaction) -> None:
            await sessions.commit_timeline_turn_in_transaction(
                transaction, "shared", tenant_id="tenant", execution_id="success",
                start_message_index=0, end_message_index=message_count,
            )
            await sessions.advance_continuation_in_transaction(
                transaction, "shared", tenant_id="tenant", execution_id="success", expected=None,
                next_cursor=ConversationCursor(
                    agent_run_id, history_id=created.history_id, message_count=message_count,
                    checkpoint_sequence=checkpoint_sequence, projection_digest=checkpoint.projection_digest,
                ), release_execution=True, history_quality="complete",
            )

        await sessions.state_store.mutate(commit)
        peer = Principal("peer", "tenant")
        async with History.open(persistence=state, authorization=SessionOnlyReaderPolicy()) as history:
            with pytest.raises(AIError) as denied:
                await history.execution_detail("success", session_id="shared", principal=peer)
            assert denied.value.code is ErrorCode.AUTHORIZATION_DENIED
            canonical = (await history.session_detail("shared", principal=peer)).timeline.items[0]
            assert canonical.conversation_committed is True
            assert [item.item_kind for item in canonical.items] == ["tool_result", "assistant"]
            assert canonical.items[-1].content == "visible answer"
            assert await history.session_turn("shared", "success", principal=peer) == canonical
    finally:
        await state.close()
