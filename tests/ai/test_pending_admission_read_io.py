#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""The post-claim identity and session recheck shares one fresh SQL transaction."""

from dataclasses import replace
from pathlib import Path

import pytest
from pydantic_ai.messages import ModelResponse, TextPart
from sqlalchemy import event

from linktools.ai.core import ExecutionStatus, Principal, TenantAuthorizationPolicy
from linktools.ai.errors import AIError, ErrorCode
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
        session_reads = []
        current = None
        accept = type(turns).accept

        def sql(connection, _cursor, statement, _parameters, _context, _many):
            statement = statement.lstrip().upper()
            if statement.startswith("SELECT ") and "\nFROM AI_SESSIONS " in statement:
                session_reads.append(statement)
            if current is not None and statement.startswith(("UPDATE AI_SESSIONS ", "INSERT INTO AI_SESSION_TURNS ")):
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
                    session_reads.clear()
                    turn = await runtime.sessions.start("session", "question", principal=principal,
                                                        agent="default", idempotency_key=request)
                    assert len(session_reads) == 1 and "EXISTS (SELECT" in session_reads[0]
                    assert (await turn.wait_final(timeout_seconds=10)).status is ExecutionStatus.SUCCEEDED
                    replay = await runtime.sessions.start("session", "question", principal=principal,
                                                          agent="default", idempotency_key=request)
                    assert replay.key == turn.key
                    with pytest.raises(AIError) as conflict:
                        await runtime.sessions.start("session", "different", principal=principal,
                                                     agent="default", idempotency_key=request)
                    assert conflict.value.code is ErrorCode.IDEMPOTENCY_CONFLICT
                    with pytest.raises(AIError) as denied:
                        await runtime.sessions.start("session", "question", agent="default", idempotency_key=request,
                                                     principal=Principal("foreign", principal.tenant_id))
                    assert denied.value.code is ErrorCode.AUTHORIZATION_DENIED
                    assert len(batches) == index + 1
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
                replace(old, data={}, storage_revision=old.storage_revision + 1), expected_storage_revision=old.storage_revision,
            )

        await store.mutate(corrupt)
        assert await StateCommands.read_pending_admission(
            identities, "session.resume", "request", tenant_id="tenant",
            sessions=sessions, session_id="session",
        ) == (identity, None)
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


@pytest.mark.asyncio
async def test_closed_native_session_can_replay_exact_receipt_but_cannot_admit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    async def model(messages, info):
        nonlocal calls
        calls += 1
        return ModelResponse(parts=[TextPart("answer")])

    class HostAuthorization(TenantAuthorizationPolicy):
        """Use native host authorization and require the admitted execution owner."""

    namespace = "closed-admission-replay"
    storage = PersistenceStorage.sqlite(tmp_path / "state.db")
    features = RuntimeFeatures(namespace)
    features.agent("default", model="default", allow_tools=())
    features = model_features(_Models(_UsageFunctionModel(model)), features)

    async def start(runtime, *, prompt="question", key="request", principal=None):
        return await runtime.sessions.start(
            "session", prompt, principal=principal or runtime.default_principal,
            agent="default", idempotency_key=key,
        )

    async with Runtime.open(namespace, storage=storage, features=features, authorization=HostAuthorization()) as runtime:
        await runtime.sessions.create_turn_session("session", principal=runtime.default_principal)

        async def lose_response():
            admitted = await start(runtime)
            assert (await admitted.wait_final(timeout_seconds=5)).status is ExecutionStatus.SUCCEEDED
            raise TimeoutError("client did not receive the admitted receipt")

        with pytest.raises(TimeoutError, match="client did not receive"):
            await lose_response()
        view = await runtime.history.turn_session("session", principal=runtime.default_principal)
        first, = view["timeline"]["items"]
        assert await runtime.sessions.close_turn_session(
            "session", expected_revision=view["session"]["revision"], principal=runtime.default_principal,
        )

    async with Runtime.open(namespace, storage=storage, features=features, authorization=HostAuthorization()) as runtime:
        turns = runtime._state.session_turns

        async def no_admission(*_args, **_kwargs):
            pytest.fail("not even an optimistic admission is allowed after session close")

        monkeypatch.setattr(turns, "accept", no_admission)
        replay = await start(runtime)
        assert replay.key.turn_id == first["turn_id"]
        result = await replay.wait_final(timeout_seconds=5)
        assert result.status is ExecutionStatus.SUCCEEDED
        assert result.view["output"] == {"text": "answer"}
        with pytest.raises(AIError) as changed:
            await start(runtime, prompt="changed question")
        assert changed.value.code is ErrorCode.IDEMPOTENCY_CONFLICT
        with pytest.raises(AIError) as denied:
            await start(runtime, principal=Principal("foreign", runtime.tenant_id))
        assert denied.value.code is ErrorCode.AUTHORIZATION_DENIED
        with pytest.raises(AIError) as closed:
            await start(runtime, key="new-request")
        assert closed.value.code is ErrorCode.SESSION_CONFLICT
        assert await turns.find_request("session", "new-request") is None
        assert calls == 1
