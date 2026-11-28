#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""End-to-end Session timeline coverage through the public Runtime API."""

from collections.abc import Awaitable, Callable
from functools import partial
from pathlib import Path

import pytest

from linktools.ai.runtime import RuntimeFeatures
from linktools.ai.core import AuthorizationAction, ExecutionStatus, Principal, TenantAuthorizationPolicy
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.history import History
from linktools.ai.core import Page
from linktools.ai.session import SessionTurn
from linktools.ai.runtime import Runtime
from linktools.ai.persistence import Persistence, PersistenceStorage

from ._runtime_test_helpers import RuntimeUsageModels, model_features


@pytest.mark.asyncio
async def test_in_memory_session_run_restores_timeline() -> None:
    application = RuntimeFeatures("application", defaults={"execution": {"retain_history": True}})
    application.agent("default", model="default", allow_tools=())

    async with Persistence.open('default', storage=PersistenceStorage.in_memory()) as persistence_27, Runtime.open(persistence=persistence_27, features=model_features(RuntimeUsageModels(), application)) as runtime:
        session = await runtime.agents.get("default").create_session("session")
        first = (await session.run("hello", timeout_seconds=10)).result
        second = (await session.run("again", timeout_seconds=10)).result
        assert first.status is ExecutionStatus.SUCCEEDED
        assert second.status is ExecutionStatus.SUCCEEDED

        page = await session.timeline()
        assert [turn.execution_id for turn in page.items] == [
            first.execution_id,
            second.execution_id,
        ]
        assert [turn.status for turn in page.items] == [
            ExecutionStatus.SUCCEEDED,
            ExecutionStatus.SUCCEEDED,
        ]
        assert [turn.user_input["prompt"] for turn in page.items] == [
            {"kind": "text", "text": "hello"},
            {"kind": "text", "text": "again"},
        ]
        assert all(turn.conversation_committed for turn in page.items)
        assert [
            [item.item_kind for item in turn.items] for turn in page.items
        ] == [["assistant"], ["assistant"]]
        assert page.next_cursor is None


@pytest.mark.asyncio
async def test_in_memory_session_timeline_fork_matches_history() -> None:
    application = RuntimeFeatures("application", defaults={"execution": {"retain_history": True}})
    application.agent("default", model="default", allow_tools=())
    storage = PersistenceStorage.in_memory()

    async def check_pages(
        read: Callable[..., Awaitable[Page[SessionTurn]]],
        expected: tuple[SessionTurn, ...],
    ) -> None:
        full = await read()
        assert full.items == expected
        assert full.next_cursor is None
        newest = await read(limit=1)
        assert newest.items == expected[-1:]
        assert newest.next_cursor is not None
        oldest = await read(limit=1, cursor=newest.next_cursor)
        assert oldest.items == expected[:1]
        assert oldest.next_cursor is None

    async def read_history_page(history, principal, **paging):
        detail = await history.session_detail("child", principal=principal, **paging)
        return detail.timeline

    async with Persistence.open('default', storage=storage) as persistence_86, Runtime.open(persistence=persistence_86, features=model_features(RuntimeUsageModels(), application)) as runtime:
        principal = runtime.default_principal
        parent = await runtime.agents.get("default").create_session("parent")
        assert (await parent.timeline()).items == ()
        assert (await runtime.history.session_detail(
            "parent", principal=principal,
        )).timeline.items == ()
        parent_turn = (await parent.run("before fork", timeout_seconds=10)).result
        assert parent_turn.status is ExecutionStatus.SUCCEEDED

        child = await parent.fork("child")
        await parent.close()
        child_turn = (await child.run("after fork", timeout_seconds=10)).result
        assert child_turn.status is ExecutionStatus.SUCCEEDED

        page = await child.timeline()
        assert [turn.execution_id for turn in page.items] == [
            parent_turn.execution_id,
            child_turn.execution_id,
        ]
        assert [turn.user_input["prompt"] for turn in page.items] == [
            {"kind": "text", "text": "before fork"},
            {"kind": "text", "text": "after fork"},
        ]
        assert all(turn.conversation_committed for turn in page.items)
        assert all(turn.items for turn in page.items)
        await check_pages(child.timeline, page.items)
        await check_pages(
            partial(read_history_page, runtime.history, principal),
            page.items,
        )

class _TimelinePolicy(TenantAuthorizationPolicy):
    async def authorize(self, principal, action, resource):
        await super().authorize(principal, action, resource)
        if principal.principal_id == "restricted" and action is AuthorizationAction.SESSION_READ and resource.id == "parent":
            raise AIError(ErrorCode.AUTHORIZATION_DENIED)


def _native_timeline_features():
    features = RuntimeFeatures("application")
    features.agent("default", model="default", allow_tools=())
    return model_features(RuntimeUsageModels(), features)


async def _native_timeline_run(runtime, session_id, text, request):
    turn = await runtime.agents.get().start(text, session_id=session_id, idempotency_key=request)
    result = await turn.wait_final(timeout_seconds=10)
    assert result.status is ExecutionStatus.SUCCEEDED
    return result.view


@pytest.mark.asyncio
async def test_session_timeline_fork_reads_match_live_and_reopened_history(tmp_path: Path) -> None:
    storage = PersistenceStorage.from_root(tmp_path, exclusive_writer=True)
    policy = _TimelinePolicy()

    async def check_pages(read, expected):
        assert await read() == {"items": expected, "next_cursor": None}
        newest = await read(limit=1)
        assert newest["items"] == expected[-1:] and newest["next_cursor"] is not None
        assert await read(limit=1, cursor=newest["next_cursor"]) == {
            "items": expected[:1], "next_cursor": None,
        }

    async def read_history(history, principal, **paging):
        return (await history.turn_session("child", principal=principal, **paging))["timeline"]

    async with (
        Persistence.open("default", storage=storage) as state,
        Runtime.open(persistence=state, features=_native_timeline_features(), authorization=policy) as runtime,
    ):
        principal = runtime.default_principal
        await runtime.sessions.create_turn_session("parent", principal=principal)
        empty = await runtime.history.turn_session("parent", principal=principal)
        assert empty["timeline"] == {"items": [], "next_cursor": None}
        parent_turn = await _native_timeline_run(runtime, "parent", "before fork", "first")
        parent = await runtime.history.turn_session_state("parent", principal=principal)
        assert await runtime.sessions.fork_turn_session(
            "parent", "child", principal=principal, expected_revision=parent["revision"], idempotency_key="fork",
        )
        parent = await runtime.history.turn_session_state("parent", principal=principal)
        assert await runtime.sessions.close_turn_session("parent", principal=principal, expected_revision=parent["revision"])
        child_turn = await _native_timeline_run(runtime, "child", "after fork", "second")
        assert [turn["user_input"]["prompt"] for turn in (parent_turn, child_turn)] == ["before fork", "after fork"]
        assert all(turn["conversation_committed"] and turn["output"] for turn in (parent_turn, child_turn))
        expected = [parent_turn, child_turn]
        await check_pages(partial(read_history, runtime.history, principal), expected)
        with pytest.raises(AIError) as denied:
            await read_history(runtime.history, Principal("restricted", principal.tenant_id))
        assert denied.value.code is ErrorCode.AUTHORIZATION_DENIED

    async with (
        Persistence.open("default", storage=storage, read_only=True) as state,
        History.open(persistence=state, authorization=policy) as history,
    ):
        await check_pages(partial(read_history, history, principal), expected)


@pytest.mark.asyncio
@pytest.mark.parametrize(("turn_count", "limit"), ((2, 1), (6, 2)))
async def test_session_timeline_reads_only_captured_tail_during_append(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, turn_count: int, limit: int,
) -> None:
    from linktools.ai.persistence._session_turn_sql import SqlSessionTurnReader

    storage = PersistenceStorage.from_root(tmp_path, exclusive_writer=True)
    async with (
        Persistence.open("default", storage=storage) as state,
        Runtime.open(persistence=state, features=_native_timeline_features(), authorization=_TimelinePolicy()) as runtime,
    ):
        principal = runtime.default_principal

        async def read(**paging):
            return (await runtime.history.turn_session("session", principal=principal, **paging))["timeline"]

        await runtime.sessions.create_turn_session("session", principal=principal)
        results = [await _native_timeline_run(runtime, "session", f"repeat {n % 2}", f"turn-{n}") for n in range(turn_count)]
        assert (await read())["items"] == results
        original_page = SqlSessionTurnReader.turn_page
        captures = []

        async def append_after_capture(reader, session_id, **kwargs):
            assert kwargs["through"] == turn_count and kwargs["limit"] == limit + 1
            assert kwargs["body_limit"] == limit
            _, _, original_capture = reader._transcript_source

            async def capture(roots, **options):
                captures.append(tuple(roots))
                return await original_capture(roots, **options)

            turn_store, archive, _ = reader._transcript_source
            newest.append(await _native_timeline_run(runtime, "session", "newest", "newest"))
            with monkeypatch.context() as fault:
                fault.setattr(reader, "_transcript_source", (turn_store, archive, capture))
                return await original_page(reader, session_id, **kwargs)

        newest = []
        with monkeypatch.context() as fault:
            fault.setattr(SqlSessionTurnReader, "turn_page", append_after_capture)
            page = await read(limit=limit)
        assert page["items"] == results[-limit:]
        [roots] = captures
        assert len(roots) == limit  # The extra cursor row did not hydrate its body.
        older = await read(cursor=page["next_cursor"])
        assert older == {"items": results[:-limit], "next_cursor": None}
        assert (await read())["items"] == [*results, *newest]


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", ("missing", "malformed", "coordinate_gap", "head"))
@pytest.mark.parametrize("turn_count", (2, 3))
async def test_session_timeline_rejects_invalid_captured_tail(
    tmp_path: Path, fault: str, turn_count: int,
) -> None:
    from linktools.ai.persistence.turns import SessionTurnKey

    storage = PersistenceStorage.from_root(tmp_path, exclusive_writer=True)
    async with (
        Persistence.open("default", storage=storage) as state,
        Runtime.open(persistence=state, features=_native_timeline_features(), authorization=_TimelinePolicy()) as runtime,
    ):
        principal = runtime.default_principal
        await runtime.sessions.create_turn_session("session", principal=principal)
        completed = [
            await _native_timeline_run(runtime, "session", "repeat", f"turn-{n}") for n in range(turn_count)
        ]
        turns = state.session_turns
        assert (await runtime.history.turn_session("session", principal=principal, limit=1))["timeline"]["items"] == completed[-1:]
        async with turns._context.engine.begin() as connection:
            if fault == "head":
                query = turns._sessions.update().filter_by(**turns._session_values("session")).values(
                    last_turn_no=turn_count + 1,
                )
            elif fault == "coordinate_gap":
                key = SessionTurnKey("session", completed[-2]["turn_id"])
                query = turns._turns.update().filter_by(**turns._turn_values(key)).values(turn_no=turn_count + 1)
            else:
                key = SessionTurnKey("session", completed[-1]["turn_id"])
                query = turns._turns.delete() if fault == "missing" else turns._turns.update().values(result_json={})
                query = query.filter_by(**turns._turn_values(key))
            assert (await connection.execute(query)).rowcount == 1
        with pytest.raises(AIError) as error:
            await runtime.history.turn_session("session", principal=principal, limit=1)
        assert error.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
