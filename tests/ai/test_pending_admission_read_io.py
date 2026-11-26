#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""The post-claim identity and session recheck shares one fresh SQL transaction."""

from dataclasses import replace
from pathlib import Path

import pytest
from pydantic_ai.messages import ModelResponse, TextPart
from sqlalchemy import event

from linktools.ai.core import ExecutionStatus, TenantAuthorizationPolicy
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.runtime import Runtime, RuntimeFeatures
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

    class _HostAuthorization(TenantAuthorizationPolicy):
        """Authorize the native SQL session path exercised below."""

    features = RuntimeFeatures("pending-admission-reads")
    features.agent("default", model="default", allow_tools=())
    cache = InMemoryCacheBackend(max_bytes=4 * 1024 * 1024)
    storage = PersistenceStorage.sqlite(tmp_path / "state.db", cache=cache, coordination=cache)
    async with Persistence.open("pending-admission-reads", storage=storage) as state:
        turns = state.session_turns
        engine = turns._context.engine.sync_engine
        batches = []
        current = None
        accept = type(turns).accept

        def sql(connection, _cursor, statement, _parameters, _context, _many):
            if current is not None:
                statement = statement.lstrip().upper()
                if statement.startswith(("UPDATE AI_SESSIONS ", "INSERT INTO AI_SESSION_TURNS ")):
                    current.append((statement.split()[0], connection.get_transaction()))

        async def accept_turn(self, *args, **kwargs):
            nonlocal current
            if self is not turns:
                return await accept(self, *args, **kwargs)
            current = []
            try:
                return await accept(self, *args, **kwargs)
            finally:
                batches.append(current)
                current = None

        monkeypatch.setattr(type(turns), "accept", accept_turn)
        event.listen(engine, "before_cursor_execute", sql)
        try:
            async with Runtime.open(
                persistence=state,
                features=model_features(_Models(_UsageFunctionModel(model)), features),
                authorization=_HostAuthorization(),
            ) as runtime:
                principal = runtime.default_principal
                await runtime.sessions.create_turn_session("session", principal=principal)
                for index in range(2):
                    request = f"request-{index}"
                    turn = await runtime.sessions.start("session", "question", principal=principal,
                                                        agent="default", idempotency_key=request)
                    assert (await turn.wait_final(timeout_seconds=10)).status is ExecutionStatus.SUCCEEDED
                    replay = await runtime.sessions.start("session", "question", principal=principal,
                                                          agent="default", idempotency_key=request)
                    assert replay.key == turn.key
                assert calls == 2
                assert len(batches) == 2
                for batch in batches:
                    [update, insert] = batch
                    assert (update[0], insert[0]) == ("UPDATE", "INSERT")
                    assert update[1] is not None and update[1] is insert[1]
        finally:
            event.remove(engine, "before_cursor_execute", sql)


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
        assert await StateCommands.read_pending_admission(
            identities, "execution.start", "request", tenant_id="tenant",
        ) == (None, None)
        if backend == "sqlite":
            assert sessions is None  # SQL only owns the independent Execution branch here.
            return
        created = await sessions.create(_session())
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

    storage = PersistenceStorage.in_memory()
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
        async def corrupt(transaction):
            [old] = [record for record in await transaction.scan_records() if record.kind == "session"]
            assert await transaction.replace_record(
                replace(old, data={}), expected_storage_revision=old.storage_revision,
            )

        await store.mutate(corrupt)
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
            storage=PersistenceStorage.in_memory()) as sessions_state,
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
