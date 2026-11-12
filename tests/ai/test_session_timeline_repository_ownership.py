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
    ExecutionStatus,
    HmacCursorSigner,
    Principal,
    TenantAuthorizationPolicy,
)
from linktools.ai.history import History
from linktools.ai.persistence import PersistenceStorage, PersistencePlan
from linktools.ai.storage import StorageRoute
from linktools.ai.persistence.contracts import (
    ExecutionRecord,
    ExecutionRepository,
    SessionRecord,
    SessionRepository,
    SessionTurnCommit,
)
from linktools.ai.persistence._conversation_repositories import SessionRepositoryImpl
from linktools.ai.persistence._execution_repositories import ExecutionRepositoryImpl
from sqlalchemy import event

from .test_session_timeline import _commit_empty_turn, _execution, _session

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
        expected = await _history(state).session_timeline("session", principal=principal)
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
        actual = await _history(state, **injected).session_timeline("session", principal=principal)
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
        engine = state.conversation.sessions.state_store.storage_group.context.engine.sync_engine
        statements, begins = [], []

        def statement(_connection, _cursor, sql, _parameters, _context, _many):
            if "ai_step_messages" in sql:
                from linktools.ai.persistence._step_payload_owner import payload_store_key
                from linktools.ai.steps import native_store_key
                assert payload_store_key("timeline-owner", "tenant") in _parameters
                assert native_store_key("timeline-owner", "tenant") not in _parameters
                assert "ai_step_messages.store_key =" in sql
            statements.append(sql)

        def begin(connection):
            begins.append(connection)

        event.listen(engine, "before_cursor_execute", statement)
        event.listen(engine, "begin", begin)
        try:
            result = await _history(state).session_timeline("session", principal=Principal("owner", "tenant"))
        finally:
            event.remove(engine, "before_cursor_execute", statement)
            event.remove(engine, "begin", begin)
        assert [turn.execution_id for turn in result.items] == ["failed", "active"]
        assert len(begins) == 2  # Fresh authorization header, then one metadata snapshot.
        assert [turn.conversation_committed for turn in result.items] == [True, False]
        selects = [sql for sql in statements if sql.lstrip().upper().startswith("SELECT")]
        # Execution and conversation records may share one metadata batch.
        assert len(selects) <= 6
        assert not any("ai_object_chunks" in sql for sql in selects)
        # ai_step_messages also owns typed execution templates/input. The SQL
        # hook above requires their exact partition; Native hydration stays forbidden.
    finally:
        await state.close()
