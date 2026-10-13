#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Injected timeline repositories retain their independent task and read ownership."""

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
from linktools.ai.runtime import (
    RuntimeHistory,
    RuntimeStorage,
    RuntimeStoragePlan,
    RuntimeStorageRoute,
)
from linktools.ai.runtime.state._contracts import (
    ExecutionRecord,
    ExecutionRepository,
    SessionRecord,
    SessionRepository,
    SessionTurnCommitRef,
)
from linktools.ai.runtime.state._conversation_repositories import SessionRepositoryImpl
from linktools.ai.runtime.state._execution_repositories import ExecutionRepositoryImpl
from sqlalchemy import event

from .test_session_timeline import _execution, _session

pytestmark = pytest.mark.asyncio


class ParallelExecutions:
    def __init__(self, repository: ExecutionRepository) -> None:
        self.state_store = repository.state_store
        self.tenant_id = repository.tenant_id
        self.namespace = repository.namespace
        self.get = repository.get
        self.get_header = repository.get_header
        self.get_result = repository.get_result

    async def get_many(
        self, execution_ids: Sequence[str], *, tenant_id: str,
    ) -> Mapping[str, ExecutionRecord]:
        records = await asyncio.gather(*(
            self.get(identity, tenant_id=tenant_id) for identity in execution_ids
        ))
        return {record.execution_id: record for record in records if record is not None}


class ParallelExecutionSubclass(ExecutionRepositoryImpl):
    async def get_many(
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
    ) -> tuple[SessionTurnCommitRef, ...]:
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
    ) -> tuple[SessionTurnCommitRef, ...]:
        read = super().list_timeline_commits(
            session_id, tenant_id=tenant_id, start_turn_seq=start_turn_seq, end_turn_seq=end_turn_seq,
        )
        return (await asyncio.gather(read))[0] if self.phase == "commits" else await read


def _history(
    state: RuntimeStorage, *, executions: ExecutionRepository | None = None,
    sessions: SessionRepository | None = None,
) -> RuntimeHistory:
    return RuntimeHistory(
        object(), tenant_id="tenant",
        executions=state.execution.executions if executions is None else executions,
        authorization=TenantAuthorizationPolicy(),
        cursor_signer=HmacCursorSigner("timeline", b"timeline-owner-test"),
        conversation=state.conversation if sessions is None else replace(state.conversation, sessions=sessions),
        session_transcript_store=state.run_store,
    )


async def _populate(state: RuntimeStorage) -> None:
    await state.initialize(namespace="timeline-owner", tenant_id="tenant")
    await state.conversation.sessions.create(_session())
    await state.execution.executions.create(replace(
        _execution("failed", status=ExecutionStatus.FAILED, prompt="original"), result=None,
    ))
    await state.conversation.sessions.admit_execution(
        "session", tenant_id="tenant", execution_id="failed", expected=None,
    )


@pytest.mark.parametrize("split", (False, True))
@pytest.mark.parametrize("subclass", (False, True))
@pytest.mark.parametrize("phase", ("executions", "session", "commits"))
async def test_public_history_accepts_parallel_injected_repository_reads(
    tmp_path: Path, split: bool, subclass: bool, phase: str,
) -> None:
    state = RuntimeStorage.from_plan(RuntimeStoragePlan(
        execution=RuntimeStorageRoute.sqlite(tmp_path / "execution.db"),
        conversation=RuntimeStorageRoute.sqlite(tmp_path / "conversation.db"),
    )) if split else RuntimeStorage.sqlite(tmp_path / "shared.db")
    try:
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
        assert [turn.execution_id for turn in actual.items] == ["failed"]
    finally:
        await state.close()


async def test_builtin_public_timeline_keeps_one_metadata_transaction(tmp_path: Path) -> None:
    state = RuntimeStorage.sqlite(tmp_path / "builtin.db")
    try:
        await _populate(state)
        engine = state.conversation.sessions.state_store.storage_group.context.engine.sync_engine
        statements, begins = [], []

        def statement(_connection, _cursor, sql, _parameters, _context, _many):
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
        assert [turn.execution_id for turn in result.items] == ["failed"]
        assert len(begins) == 2  # Fresh authorization header, then one metadata snapshot.
        assert sum(sql.lstrip().upper().startswith("SELECT") for sql in statements) == 6
    finally:
        await state.close()
