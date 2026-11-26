#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Optional Session Trace lives off the Session/Step publication path."""

import asyncio
from dataclasses import replace
from pathlib import Path

import pytest
from pydantic_ai.messages import ModelResponse, TextPart
from sqlalchemy import event
from sqlalchemy.ext.asyncio import create_async_engine

from linktools.ai.capability import SessionCapability, TraceCapability
from linktools.ai.core import CommitPolicy, ExecutionStatus, TenantAuthorizationPolicy
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.history import History
from linktools.ai.migrate import provision_database
from linktools.ai.persistence import Persistence, PersistencePlan, PersistenceStorage
from linktools.ai.runtime import Runtime, RuntimeFeatures
from linktools.ai.storage import StorageRoute

from ._runtime_test_helpers import _Models, _UsageFunctionModel, _wait_for_committed, model_features


class _SessionPolicy(TenantAuthorizationPolicy):
    """Explicit host policy for canonical SQL Session and its Trace."""


def _writes(statement: str) -> bool:
    return statement.lstrip().upper().startswith(("INSERT ", "UPDATE ", "DELETE ", "REPLACE "))


@pytest.mark.asyncio
@pytest.mark.parametrize(("sink", "expected"), (
    ("off", ExecutionStatus.SUCCEEDED),
    ("healthy", ExecutionStatus.SUCCEEDED),
    ("slow", ExecutionStatus.SUCCEEDED),
    ("unavailable", ExecutionStatus.SUCCEEDED),
    ("unavailable", ExecutionStatus.FAILED),
    ("slow", ExecutionStatus.CANCELLED),
))
async def test_session_trace_sink_cannot_hold_or_fail_authoritative_turn(
    tmp_path: Path, sink: str, expected: ExecutionStatus,
) -> None:
    primary_engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'session.db'}")
    trace_engine = create_async_engine(
        f"sqlite+aiosqlite:///{tmp_path / 'trace.db'}", connect_args={"timeout": 20},
    )
    attempts = asyncio.Event()
    model_entered = asyncio.Event()
    routed_statements: list[str] = []
    rejected = 0
    calls = 0

    def observe_trace_write(_connection, _cursor, statement, _parameters, _context, _many):
        nonlocal rejected
        if _writes(statement):
            routed_statements.append(statement)
            attempts.set()
            if sink == "unavailable":
                rejected += 1
                raise RuntimeError("synthetic independent trace sink outage")

    async def model(_messages, _info):
        nonlocal calls
        calls += 1
        model_entered.set()
        if expected is ExecutionStatus.CANCELLED:
            await asyncio.Event().wait()
        if expected is ExecutionStatus.FAILED:
            raise AIError(ErrorCode.MODEL_UNAVAILABLE)
        return ModelResponse(parts=[TextPart("durable answer")])

    policy = _SessionPolicy()
    features = RuntimeFeatures("session-trace-isolation")
    features.capability(SessionCapability(checkpoints=True), scope="runtime")
    if sink != "off":
        features.capability(TraceCapability(commit=CommitPolicy.sync()), scope="runtime")
    features.agent("default", model="default", allow_tools=())
    features = model_features(_Models(_UsageFunctionModel(model)), features)
    primary_route = StorageRoute.sql(primary_engine)
    plan = replace(PersistencePlan.from_default(primary_route), trace=StorageRoute.sql(trace_engine))
    storage = PersistenceStorage.from_plan(plan)
    try:
        await provision_database(primary_engine)
        await provision_database(trace_engine)
        async with (
            Persistence.open("session-trace-isolation", storage=storage) as state,
            Runtime.open(persistence=state, features=features, authorization=policy) as runtime,
        ):
            principal = runtime.default_principal
            session = await runtime.agents.get().create_session("chat", principal=principal)
            async with trace_engine.connect() as blocker:
                if sink == "slow":
                    await blocker.exec_driver_sql("BEGIN IMMEDIATE")
                event.listen(trace_engine.sync_engine, "before_cursor_execute", observe_trace_write)
                try:
                    turn = await runtime.sessions.start(
                        session.session_id, "question", agent=runtime.agents.get(), principal=principal,
                        idempotency_key="request",
                    )
                    if expected is ExecutionStatus.CANCELLED:
                        await asyncio.wait_for(model_entered.wait(), 5)
                        assert await asyncio.wait_for(turn.cancel(), 5)
                    if expected is ExecutionStatus.SUCCEEDED and sink in ("slow", "unavailable"):
                        # A physical write is attempted on the separate database
                        # before reading the result; the lock stays held until after it.
                        await asyncio.wait_for(attempts.wait(), 5)
                    terminal = await turn.wait_final(timeout_seconds=5)
                    assert terminal.status is expected
                    assert terminal.view["conversation_committed"] is True
                    if expected is ExecutionStatus.SUCCEEDED:
                        assert terminal.view["output"] == {"text": "durable answer"}
                    else:
                        assert terminal.view["output"] is None
                        if expected is ExecutionStatus.FAILED:
                            assert terminal.view["error_code"] == ErrorCode.MODEL_UNAVAILABLE.value
                    assert (await runtime.history.turn_session_state(
                        "chat", principal=principal,
                    ))["active_turn_id"] is None
                    page = await runtime.history.turn_session("chat", principal=principal)
                    assert page["timeline"]["items"] == [terminal.view]
                    if sink == "healthy":
                        traces = await _wait_for_committed(
                            lambda: runtime.history.turn_trace("chat", turn.key.turn_id, principal=principal),
                            lambda page: {item.payload["kind"] for item in page.items}
                            >= {"MODEL_REQUEST", "MODEL_RESPONSE"},
                        )
                        assert [(item.payload["kind"], item.payload["status"]) for item in traces.items] == [
                            ("MODEL_REQUEST", "STARTED"), ("MODEL_RESPONSE", "SUCCEEDED"),
                        ]
                        assert all(item.turn_id == turn.key.turn_id for item in traces.items)
                        assert routed_statements
                    if sink == "off":
                        assert not attempts.is_set()
                    if sink == "unavailable" and expected is ExecutionStatus.SUCCEEDED:
                        assert rejected > 0
                finally:
                    event.remove(trace_engine.sync_engine, "before_cursor_execute", observe_trace_write)
                    if sink == "slow":
                        await blocker.rollback()

            if expected is ExecutionStatus.SUCCEEDED:
                steps = await state.create_step_store()
                try:
                    snapshot = await steps.latest_snapshot(run_id=turn.key.turn_id)
                    assert snapshot is not None and snapshot.state == "complete"
                finally:
                    await steps.close()

        async with (
            Persistence.open("session-trace-isolation", storage=storage, read_only=True) as offline,
            History.open(persistence=offline, authorization=policy) as history,
        ):
            restored = await history.turn_result("chat", turn.key.turn_id, principal=principal)
            assert restored == terminal.view
            if sink == "healthy":
                evidence = await history.turn_trace("chat", turn.key.turn_id, principal=principal)
                by_sequence = {item.step_event_seq: item for item in evidence.items}
                assert all(by_sequence.get(item.step_event_seq) == item for item in traces.items)

            if sink in ("off", "healthy"):
                from linktools.ai.web import create_app
                from .test_web_console import client

                params = {"session_id": "chat", "turn_id": turn.key.turn_id, "models": "true"}
                async with client(create_app(history=history, local_workspace=True)) as http:
                    metadata = await http.get("/api/session", params=params)
                    assert metadata.status_code == 200, metadata.text
                    exact = await http.get("/api/session", params={
                        **params, "include_content": "true",
                    })
                    assert exact.status_code == 200, exact.text
                    wrong_session = await http.get("/api/session", params={**params, "session_id": "other"})
                    assert wrong_session.status_code == 404
                    if sink == "healthy":
                        item, = metadata.json()["items"]
                        assert item["turn_id"] == turn.key.turn_id and "execution_id" not in item
                        assert item["request"] == {} and item["response"] is None
                        model_item, = exact.json()["items"]
                        assert model_item["content_included"] is True
                        assert "question" in str(model_item["request"]["messages"])
                        assert "durable answer" in str(model_item["response"])
                    else:
                        assert exact.json() == {"items": [], "next_cursor": None}
                async with client(create_app(history=history, local_workspace=False)) as http:
                    denied = await http.get("/api/session", params={**params, "include_content": "true"})
                    assert denied.status_code == 404
        assert calls == 1
    finally:
        await trace_engine.dispose()
        await primary_engine.dispose()


@pytest.mark.asyncio
async def test_session_starts_with_unreachable_separate_trace(tmp_path: Path) -> None:
    primary = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'session.db'}")
    trace = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'trace.db'}")
    attempts = asyncio.Event()

    def unavailable(_dialect, _connection_record, _args, _kwargs):
        attempts.set()
        raise OSError("synthetic Trace connection outage")

    async def model(_messages, _info):
        return ModelResponse(parts=[TextPart("available")])

    features = RuntimeFeatures("trace-startup-outage")
    features.capability(SessionCapability(checkpoints=True), scope="runtime")
    features.capability(TraceCapability(), scope="runtime")
    features.agent("default", model="default", allow_tools=())
    features = model_features(_Models(_UsageFunctionModel(model)), features)
    plan = replace(
        PersistencePlan.from_default(StorageRoute.sql(primary)), trace=StorageRoute.sql(trace),
    )
    policy = _SessionPolicy()
    try:
        await provision_database(primary)
        await provision_database(trace)
        await trace.dispose()
        event.listen(trace.sync_engine, "do_connect", unavailable)
        try:
            async with (
                Persistence.open("trace-startup-outage", storage=PersistenceStorage.from_plan(plan)) as state,
                Runtime.open(persistence=state, features=features, authorization=policy) as runtime,
            ):
                assert not attempts.is_set()
                session = await runtime.agents.get().create_session("chat")
                turn = await runtime.sessions.start(
                    session.session_id, "question", agent=runtime.agents.get(), principal=runtime.default_principal,
                    idempotency_key="request",
                )
                terminal = await turn.wait_final(timeout_seconds=5)
                assert terminal.status is ExecutionStatus.SUCCEEDED
                assert terminal.view["output"] == {"text": "available"}
                await asyncio.wait_for(attempts.wait(), 5)
                assert (await runtime.history.turn_session_state(
                    "chat", principal=runtime.default_principal,
                ))["active_turn_id"] is None
        finally:
            event.remove(trace.sync_engine, "do_connect", unavailable)
    finally:
        await trace.dispose()
        await primary.dispose()
