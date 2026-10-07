#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Successful callback cleanup acknowledges delivery without reopening a stream."""

import asyncio
from dataclasses import replace
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from pydantic_ai.messages import ModelResponse, TextPart

from linktools.ai.capability import CapabilityGroup
from linktools.ai.core import ExecutionEventType, TaskStatus
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime import Execution, Runtime, RuntimeStorage
from linktools.ai.task import TaskEvent, TaskEventType

from ._runtime_test_helpers import _UsageFunctionModel
from .test_live_history_readback_integration import _Models
from .test_unified_wait_contract import _Bundle, _PRINCIPAL, _execution_event


@pytest.mark.asyncio
@pytest.mark.parametrize("contract", ["cursor", "close"])
async def test_live_wait_acknowledges_callback_cleanup_without_waiting_for_model(contract) -> None:
    model_started = asyncio.Event()
    release_model = asyncio.Event()
    acknowledged = []

    async def model(messages, info):
        model_started.set()
        await release_model.wait()
        return ModelResponse(parts=[TextPart("done")])

    group = CapabilityGroup("wait-cleanup-ack")
    group.agent("default", model="default", allow_tools=())
    async with Runtime.open(
        "wait-cleanup-ack", models=_Models(_UsageFunctionModel(model)),
        storage=RuntimeStorage.in_memory(), capabilities=(group,),
    ) as runtime:
        execution = await runtime.agents.get().start("hello")

        async def callback(event):
            if contract == "close" and event.event.event_type != ExecutionEventType.MODEL_REQUEST_STARTED:
                return
            await model_started.wait()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                acknowledged.append(event.cursor)

        closed = False
        try:
            with pytest.raises(AIError) as raised:
                await execution.wait(on_event=callback, timeout_seconds=0.2, close_timeout_seconds=0.05)
            assert raised.value.code is ErrorCode.WAIT_TIMEOUT
            assert len(acknowledged) == 1 and acknowledged[0] is not None
            if contract == "cursor":
                assert raised.value.safe_details["cursor"] == acknowledged[0]
            else:
                await runtime.close()
                closed = True
                assert not release_model.is_set()
        finally:
            release_model.set()
            if not closed:
                await execution.wait(timeout_seconds=2)
            await runtime.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("owner", ["execution", "graph", "evaluation"])
@pytest.mark.parametrize("exit_kind", ["deadline", "caller_cancel"])
async def test_wait_callback_can_finish_during_cleanup_without_reopening_observation(owner, exit_kind) -> None:
    bundle = _Bundle()
    bundle.execution.release.clear()
    bundle.graph.release.clear()
    entered = asyncio.Event()
    release_stream = asyncio.Event()
    acknowledged = []

    async def tree(*args, ready=None, **kwargs):
        if ready is not None:
            ready.set()
        yield _execution_event(1)
        await release_stream.wait()

    async def graph_events(*args, **kwargs):
        yield TaskEvent(1, "graph", 1, TaskEventType.GRAPH_CHANGED,
                        datetime.now(timezone.utc), TaskStatus.RUNNING, TaskStatus.PENDING)
        await release_stream.wait()

    async def inspect_evaluation(*args):
        return replace(bundle.evaluations.result, completion="running")

    async def record(*args):
        return SimpleNamespace(intents=(SimpleNamespace(
            confirmed=True, submission=SimpleNamespace(graph=SimpleNamespace(graph_id="graph")),
        ),))

    async def callback(event):
        entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            acknowledged.append(event.cursor)

    bundle.execution_run = Execution(bundle.runtime, "execution", _PRINCIPAL, tree)
    bundle.graph.stream_events = graph_events
    bundle.evaluations._inspect = inspect_evaluation
    bundle.evaluations._record = record
    waiting = asyncio.create_task(bundle.wait(
        owner, on_event=callback, timeout_seconds=0.05 if exit_kind == "deadline" else None,
        close_timeout_seconds=0.05,
    ))
    try:
        await asyncio.wait_for(entered.wait(), 1)
        if exit_kind == "caller_cancel":
            waiting.cancel("caller requested cancellation")
            with pytest.raises(asyncio.CancelledError):
                await waiting
        else:
            with pytest.raises(AIError) as raised:
                await waiting
            assert raised.value.code is ErrorCode.WAIT_TIMEOUT
            assert raised.value.safe_details["cursor"] == acknowledged[-1]
        assert len(acknowledged) == 1
        await bundle.runtime.close()
        assert bundle.storage_closed
        assert not release_stream.is_set()
    finally:
        release_stream.set()
        if not waiting.done():
            waiting.cancel()
        await asyncio.gather(waiting, return_exceptions=True)
        await bundle.runtime.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("observe", [False, True])
async def test_authoritative_wait_timeout_keeps_its_own_details(observe) -> None:
    bundle = _Bundle()
    cause = AIError(ErrorCode.WAIT_TIMEOUT, safe_details={"cursor": "service-owned", "scope": "authority"})
    bundle.execution.error = cause
    bundle.tree.emit = True

    async def callback(event):
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            return

    try:
        with pytest.raises(AIError) as raised:
            await bundle.wait("execution", on_event=callback if observe else None, timeout_seconds=1)
        assert raised.value is cause
        assert raised.value.safe_details == {"cursor": "service-owned", "scope": "authority"}
    finally:
        await bundle.runtime.close()
