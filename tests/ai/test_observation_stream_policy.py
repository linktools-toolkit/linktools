#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Opt-in source failures preserve authority, acknowledgement and owned cleanup."""

import asyncio
from dataclasses import dataclass

import pytest

from linktools.ai.errors import AIError, ErrorCode, ObservationError
from linktools.ai.observe.stream import report_observation_error, wait_for_result


@dataclass(frozen=True)
class _Event:
    cursor: str


async def _empty_final(result, cursor):
    async def events():
        if False:
            yield
    return events()


@pytest.mark.asyncio
@pytest.mark.parametrize("finalized", (False, True))
@pytest.mark.parametrize("deliver", (False, True))
async def test_source_fault_returns_last_ack_after_closing_single_observer(finalized, deliver) -> None:
    sessions, seen = set(), []
    closed = asyncio.Event()
    failure = ObservationError("stream", cursor="delivered-but-not-acknowledged")

    async def watch(ready):
        ready.set()
        try:
            if deliver:
                yield _Event("ack")
            raise failure
        finally:
            closed.set()

    async def waiter():
        await asyncio.Event().wait()

    async def observe(event):
        seen.append(event.cursor)

    with pytest.raises(ObservationError) as caught:
        await asyncio.wait_for(wait_for_result(
            scope="test", resource_id="one", waiter=waiter, watch=watch, on_event=observe,
            cursor="before", timeout_seconds=None, close_timeout_seconds=0.2,
            register=sessions.add, release=sessions.discard,
            finalize=_empty_final if finalized else None, raise_on_stream_error=True,
        ), 1)
    assert caught.value is failure and caught.value.cursor == ("ack" if deliver else "before")
    assert seen == (["ack"] if deliver else [])
    assert closed.is_set() and not sessions


@pytest.mark.asyncio
@pytest.mark.parametrize("authoritative_failure", (False, True))
async def test_available_authority_outranks_simultaneous_source_fault(authoritative_failure) -> None:
    sessions = set()
    acknowledged, settled = asyncio.Event(), asyncio.Event()
    result = object()
    business_error = AIError(ErrorCode.MODEL_UNAVAILABLE)

    async def watch(ready):
        ready.set()
        yield _Event("ack")
        await settled.wait()
        raise ObservationError("stream", cursor="source-position")

    async def waiter():
        await acknowledged.wait()
        settled.set()
        if authoritative_failure:
            raise business_error
        return result

    async def observe(event):
        acknowledged.set()

    waiting = wait_for_result(
        scope="test", resource_id="one", waiter=waiter, watch=watch, on_event=observe,
        cursor=None, timeout_seconds=None, close_timeout_seconds=0.2,
        register=sessions.add, release=sessions.discard,
        finalize=_empty_final, drain_live=True, raise_on_stream_error=True,
    )
    if authoritative_failure:
        with pytest.raises(AIError) as caught:
            await waiting
        assert caught.value is business_error
    else:
        outcome = await waiting
        assert outcome.result is result
        assert outcome.cursor == "ack"
        assert outcome.observation_error.origin == "stream"
        assert outcome.observation_error.cursor == "ack"
    assert not sessions


@pytest.mark.asyncio
@pytest.mark.parametrize("caller_cancel", (False, True))
async def test_source_fault_cannot_hide_pending_cleanup_or_authorize_overlap(caller_cancel) -> None:
    sessions = set()
    release = asyncio.Event()
    source_failed = asyncio.Event()
    failure = ObservationError("stream", cursor="unacknowledged")

    async def watch(ready):
        ready.set()
        yield _Event("ack")
        try:
            raise failure
        finally:
            report_observation_error(failure)
            source_failed.set()
            while not release.is_set():
                try:
                    await release.wait()
                except asyncio.CancelledError:
                    continue

    async def waiter():
        await asyncio.Event().wait()

    async def observe(event):
        pass

    waiting = asyncio.create_task(wait_for_result(
                scope="test", resource_id="one", waiter=waiter, watch=watch, on_event=observe,
                cursor=None, timeout_seconds=None, close_timeout_seconds=0.02,
                register=sessions.add, release=sessions.discard,
                finalize=_empty_final, raise_on_stream_error=True,
            ))
    try:
        await asyncio.wait_for(source_failed.wait(), 1)
        if caller_cancel:
            waiting.cancel("caller stopped observing")
            with pytest.raises(asyncio.CancelledError):
                await waiting
        else:
            with pytest.raises(ObservationError) as caught:
                await asyncio.wait_for(waiting, 1)
            assert caught.value.safe_details["phase"] == "cleanup"
            assert caught.value.safe_details["cleanup_pending"] is True
            assert caught.value.cursor == "ack"
        assert sessions
    finally:
        release.set()
        for session in tuple(sessions):
            await asyncio.wait_for(session.close(), 1)
    assert not sessions


@pytest.mark.asyncio
@pytest.mark.parametrize("callback_failure", (False, True))
async def test_late_optional_source_fault_cannot_replace_callback_error_or_deadline(callback_failure) -> None:
    sessions = set()
    cause = ValueError("callback failed while closing")

    async def watch(ready):
        ready.set()
        try:
            yield _Event("not-acknowledged")
        finally:
            report_observation_error(ObservationError("stream", cursor="source-position"))

    async def waiter():
        await asyncio.Event().wait()

    async def observe(event):
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            if callback_failure:
                raise cause
            raise

    with pytest.raises(ObservationError if callback_failure else AIError) as caught:
        await wait_for_result(
            scope="test", resource_id="one", waiter=waiter, watch=watch, on_event=observe,
            cursor="previous-ack", timeout_seconds=0.01, close_timeout_seconds=0.2,
            register=sessions.add, release=sessions.discard,
            finalize=_empty_final, raise_on_stream_error=True,
        )
    if callback_failure:
        assert caught.value.origin == "callback"
        assert caught.value.cursor == "previous-ack"
        assert caught.value.__cause__ is cause
    else:
        assert caught.value.code is ErrorCode.WAIT_TIMEOUT
        assert caught.value.safe_details["cursor"] == "previous-ack"
    assert not sessions


@pytest.mark.asyncio
@pytest.mark.parametrize("owner", ("execution", "graph", "evaluation", "facade", "execution_models"))
async def test_public_wait_owners_share_opt_in_source_failure_semantics(owner, monkeypatch) -> None:
    from dataclasses import replace
    from linktools.ai.execution import Execution
    from linktools.ai.task.engine import TaskGraphRun
    from linktools.ai.evaluation.engine._service import EvaluationRun
    from .test_unified_wait_contract import _Bundle

    bundle = await _Bundle.create()
    bundle.execution.release.clear()
    bundle.graph.release.clear()
    bundle.evaluations.result = replace(bundle.evaluations.result, completion="running")
    closed = asyncio.Event()

    async def broken(self, cursor, include_content, ready, *args):
        ready.set()
        try:
            yield _Event("ack")
            raise ObservationError("stream", cursor="source-position")
        finally:
            closed.set()

    monkeypatch.setattr(Execution, "_watch_prepared", broken)
    monkeypatch.setattr(Execution, "_watch_models", broken)
    monkeypatch.setattr(TaskGraphRun, "_watch_prepared", broken)
    monkeypatch.setattr(EvaluationRun, "_watch_prepared", broken)

    async def observe(event):
        pass

    try:
        options = {"include_model_interactions": True} if owner == "execution_models" else {}
        with pytest.raises(ObservationError) as caught:
            await asyncio.wait_for(bundle.wait(
                "execution" if owner == "execution_models" else owner,
                on_event=observe, raise_on_stream_error=True, **options,
            ), 1)
        assert caught.value.origin == "stream" and caught.value.cursor == "ack"
        assert closed.is_set()
    finally:
        bundle.execution.release.set()
        bundle.graph.release.set()
        await bundle.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("acknowledge_during_cleanup", (False, True))
async def test_source_failure_uses_callback_ack_after_owned_cleanup(acknowledge_during_cleanup) -> None:
    sessions = set()
    callback_entered = asyncio.Event()
    seen = []
    failure = ObservationError("stream", cursor="producer-position")

    async def watch(ready):
        ready.set()

        async def source_failure():
            await callback_entered.wait()
            report_observation_error(failure)

        producer = asyncio.create_task(source_failure())
        try:
            yield _Event("ack")
        finally:
            await producer

    async def waiter():
        await asyncio.Event().wait()

    async def observe(event):
        callback_entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            if not acknowledge_during_cleanup:
                raise
            seen.append(event.cursor)

    with pytest.raises(ObservationError) as caught:
        await asyncio.wait_for(wait_for_result(
            scope="test", resource_id="one", waiter=waiter, watch=watch, on_event=observe,
            cursor="previous", timeout_seconds=None, close_timeout_seconds=0.2,
            register=sessions.add, release=sessions.discard,
            finalize=_empty_final, raise_on_stream_error=True,
        ), 1)
    assert caught.value is failure
    assert failure.cursor == ("ack" if acknowledge_during_cleanup else "previous")
    assert seen == (["ack"] if acknowledge_during_cleanup else [])
    assert not sessions
