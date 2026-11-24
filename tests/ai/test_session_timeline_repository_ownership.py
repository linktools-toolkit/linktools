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
        conversation=StorageRoute.sqlite(tmp_path / "conversation.db"),
    )) if split else PersistenceStorage.sqlite(tmp_path / "shared.db")
    try:
        state = Persistence(state)
        await _populate(state)
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


async def test_builtin_public_timeline_keeps_one_metadata_transaction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = PersistenceStorage.sqlite(tmp_path / "builtin.db")
    try:
        state = Persistence(state)
        await _populate(state)
        from linktools.ai.persistence._step_message_content import NativeMessageContent
        from linktools.ai.storage import SqlObjectStore

        def forbid_body_read(*args, **kwargs):
            pytest.fail("timeline metadata must not hydrate message or object bodies")

        monkeypatch.setattr(NativeMessageContent, "hydrate_ranges", forbid_body_read)
        monkeypatch.setattr(SqlObjectStore, "open", forbid_body_read)
        from linktools.ai.persistence._sql import _SqlGroupTransaction

        capture_content = _SqlGroupTransaction.read_content_references
        native_checks = []

        async def metadata_content(group: _SqlGroupTransaction, **kwargs: object) -> object:
            values = await capture_content(group, **kwargs)
            # The fused query can contain a conditional Native branch. Metadata
            # owners here have no Native roots, so it must return no Native rows.
            assert group._native_message_rows == {}
            native_checks.append(True)
            return values

        monkeypatch.setattr(_SqlGroupTransaction, "read_content_references", metadata_content)
        engine = state.conversation.sessions.state_store.storage_group.context.engine.sync_engine
        statements, begins = [], []

        def statement(_connection, _cursor, sql, _parameters, _context, _many):
            if "ai_step_messages" in sql:
                from linktools.ai.persistence._step_payload_owner import payload_store_key
                assert payload_store_key("timeline-owner", "tenant") in _parameters
                assert "ai_step_messages.store_key =" in sql
            statements.append(sql)

        def begin(connection):
            begins.append(connection)

        event.listen(engine, "before_cursor_execute", statement)
        event.listen(engine, "begin", begin)
        try:
            result = (await _history(state).session_detail("session", principal=Principal("owner", "tenant"))).timeline
        finally:
            event.remove(engine, "before_cursor_execute", statement)
            event.remove(engine, "begin", begin)
        assert [turn.execution_id for turn in result.items] == ["failed", "active"]
        assert len(begins) == 2  # Fresh authorization header, then one metadata snapshot.
        assert native_checks
        assert [turn.conversation_committed for turn in result.items] == [True, False]
        selects = [sql for sql in statements if sql.lstrip().upper().startswith(("SELECT", "WITH"))]
        # Execution and conversation records may share one metadata batch.
        assert len(selects) <= 6
        assert not any("ai_object_chunks" in sql for sql in selects)
        # ai_step_messages also owns typed execution templates/input. The SQL
        # hook above requires their exact partition; Native hydration stays forbidden.
    finally:
        await state.close()


async def test_exact_turn_distinguishes_active_pending_from_recovery(tmp_path: Path) -> None:
    state = Persistence(PersistenceStorage.sqlite(tmp_path / "pending-exact-turn.db"))
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


async def test_releasing_published_failed_turn_advances_the_session_once(tmp_path: Path) -> None:
    state = Persistence(PersistenceStorage.sqlite(tmp_path / "failed-turn-release.db"))
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

        engine = sessions.state_store.storage_group.context.engine.sync_engine
        fact_reads = []

        def statement(_connection, _cursor, sql, _parameters, _context, _many):
            if sql.lstrip().upper().startswith("SELECT") and "FROM ai_state_facts" in sql:
                fact_reads.append(sql)

        event.listen(engine, "before_cursor_execute", statement)
        try:
            released, receipt = await sessions.state_store.mutate(publish)
        finally:
            event.remove(engine, "before_cursor_execute", statement)
        assert receipt is not None and receipt.execution_id == execution.execution_id
        assert len(fact_reads) == 1
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


async def test_exact_committed_turn_remains_stable_after_a_newer_turn(tmp_path: Path) -> None:
    state = Persistence(PersistenceStorage.sqlite(tmp_path / "exact-turn.db"))
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


async def test_exact_turn_uses_session_visibility_for_committed_tool_content(tmp_path: Path) -> None:
    state = Persistence(PersistenceStorage.sqlite(tmp_path / "shared-exact-turn.db"))
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
