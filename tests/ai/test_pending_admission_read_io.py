#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""The post-claim identity and session recheck shares one fresh SQL transaction."""

from pathlib import Path

import pytest
from pydantic_ai.messages import ModelResponse, TextPart
from sqlalchemy import event
from sqlalchemy.engine import Engine

from linktools.ai.capability import ExecutionCapability
from linktools.ai.core import CommitPolicy, ExecutionStatus
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.runtime import Runtime, RuntimeFeatures
from linktools.ai.runtime._local import LocalExecutionBackend
from linktools.ai.storage import InMemoryCacheBackend

from ._runtime_test_helpers import _Models, _UsageFunctionModel, model_features


@pytest.mark.asyncio
async def test_pending_admission_shares_fresh_identity_and_session_transaction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    async def model(messages, info):
        nonlocal calls
        calls += 1
        return ModelResponse(parts=[TextPart("answer")])

    active = False
    reads: list[int] = []
    current = 0
    original = LocalExecutionBackend.start_pending

    def sql(connection, cursor, statement, parameters, context, many):
        nonlocal current
        if active and statement.lstrip().upper().startswith("BEGIN"):
            current += 1

    async def start_pending(self, *args, **kwargs):
        nonlocal active, current
        current = 0
        active = True
        try:
            result = await original(self, *args, **kwargs)
            assert result is not None
            reads.append(current)
            return result
        finally:
            active = False

    monkeypatch.setattr(LocalExecutionBackend, "start_pending", start_pending)
    features = RuntimeFeatures("pending-admission-reads")
    features.agent("default", model="default", allow_tools=())
    features.capability(ExecutionCapability(retain_history=True, commit=CommitPolicy.terminal()), scope="runtime")
    cache = InMemoryCacheBackend(max_bytes=4 * 1024 * 1024)
    storage = PersistenceStorage.sqlite(tmp_path / "state.db", cache=cache, coordination=cache)
    event.listen(Engine, "before_cursor_execute", sql)
    try:
        async with Persistence.open("pending-admission-reads", storage=storage) as state:
            async with Runtime.open(persistence=state,
                features=model_features(_Models(_UsageFunctionModel(model)), features)) as runtime:
                session = await runtime.agents.get().create_session("session")
                for index in range(2):
                    execution = await session.start("question", idempotency_key=f"request-{index}")
                    assert (await execution.wait(timeout_seconds=10)).result.status is ExecutionStatus.SUCCEEDED
                    replay = await session.start("question", idempotency_key=f"request-{index}")
                    assert replay.execution_id == execution.execution_id
                assert calls == 2
                assert reads == [1, 1]
    finally:
        event.remove(Engine, "before_cursor_execute", sql)


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "sqlite"))
async def test_pending_admission_read_refreshes_session_and_supports_no_session(
    tmp_path: Path, backend: str,
) -> None:
    from linktools.ai.persistence.commands import StateCommands
    from .test_session_admission_conformance import _session

    storage = (PersistenceStorage.in_memory() if backend == "memory"
               else PersistenceStorage.sqlite(tmp_path / "fresh.db"))
    async with Persistence.open("pending-fresh", tenant_id="tenant", storage=storage) as state:
        sessions, identities = state.conversation.sessions, state.execution.idempotency
        created = await sessions.create(_session())
        assert await StateCommands.read_pending_admission(
            identities, "execution.start", "request", tenant_id="tenant",
        ) == (None, None)
        assert await StateCommands.read_pending_admission(
            identities, "session.resume", "request", tenant_id="tenant",
            sessions=sessions, session_id="session",
        ) == (None, created)
        admitted = await sessions.admit_execution(
            "session", tenant_id="tenant", execution_id="other-execution", expected=None,
        )
        assert await StateCommands.read_pending_admission(
            identities, "session.resume", "request", tenant_id="tenant",
            sessions=sessions, session_id="session",
        ) == (None, admitted)


@pytest.mark.asyncio
async def test_pending_replay_precedes_corrupt_session(
    tmp_path: Path,
) -> None:
    from datetime import datetime, timezone
    from linktools.ai.core import IdempotencyStatus, ResourceKind
    from linktools.ai.persistence.commands import StateCommands
    from linktools.ai.persistence.contracts import IdempotencyRecord
    from .test_session_admission_conformance import _session

    storage = PersistenceStorage.sqlite(tmp_path / "replay.db")
    async with Persistence.open("pending-replay", tenant_id="tenant", storage=storage) as state:
        sessions, identities = state.conversation.sessions, state.execution.idempotency
        await sessions.create(_session())
        now = datetime.now(timezone.utc)
        identity = IdempotencyRecord(
            "session.resume", "request", "a" * 64, ResourceKind.EXECUTION,
            "execution", IdempotencyStatus.RESERVED, None, None, now, now,
        )
        await identities.reserve(identity)

        store = sessions.state_store
        async with store.storage_group.context.engine.begin() as connection:
            await connection.exec_driver_sql(
                "UPDATE ai_state_records SET payload_json = ? WHERE store_key = ? AND kind = ?",
                ("[]", store.store_key, "session"),
            )
        assert await StateCommands.read_pending_admission(
            identities, "session.resume", "request", tenant_id="tenant",
            sessions=sessions, session_id="session",
        ) == (identity, None)
        from linktools.ai.errors import AIError, ErrorCode
        with pytest.raises(AIError) as rejected:
            await sessions.state_store.read(
                lambda _transaction: sessions.get("session", tenant_id="tenant")
            )
        assert rejected.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


@pytest.mark.asyncio
async def test_pending_admission_reads_authority_across_separate_groups(tmp_path: Path) -> None:
    from linktools.ai.persistence.commands import StateCommands
    from .test_session_admission_conformance import _session

    async with (
        Persistence.open("pending-split", tenant_id="tenant",
            storage=PersistenceStorage.sqlite(tmp_path / "identity.db")) as identities_state,
        Persistence.open("pending-split", tenant_id="tenant",
            storage=PersistenceStorage.sqlite(tmp_path / "session.db")) as sessions_state,
    ):
        sessions = sessions_state.conversation.sessions
        identities = identities_state.execution.idempotency
        created = await sessions.create(_session())
        assert sessions.state_store.storage_group is not identities.state_store.storage_group
        assert await StateCommands.read_pending_admission(
            identities, "session.resume", "request", tenant_id="tenant",
            sessions=sessions, session_id="session",
        ) == (None, created)
        admitted = await sessions.admit_execution(
            "session", tenant_id="tenant", execution_id="other-execution", expected=None,
        )
        assert await StateCommands.read_pending_admission(
            identities, "session.resume", "request", tenant_id="tenant",
            sessions=sessions, session_id="session",
        ) == (None, admitted)
