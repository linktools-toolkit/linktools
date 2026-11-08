#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Every independently owned watch stream closes after model cleanup fails."""

import asyncio
from typing import NoReturn

import pytest

from linktools.ai.core import Principal
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.execution.projection import ModelProjection
from linktools.ai.task.engine import _run

from .test_runtime_watch_api import _Runtime, _TaskGraphService, _task_graph_run, _watch_tree


@pytest.mark.asyncio
@pytest.mark.parametrize("primary", ("close", "authority", "cancel"))
@pytest.mark.parametrize("cleanup_cancel", (False, True))
async def test_watch_closes_independent_streams_after_model_cleanup_failure(
    monkeypatch: pytest.MonkeyPatch, primary: str, cleanup_cancel: bool,
) -> None:
    cleanup_error = (asyncio.CancelledError("subscription cancelled") if cleanup_cancel
                     else RuntimeError("subscription close failed"))
    primary_error = (asyncio.CancelledError("reader cancelled") if primary == "cancel"
                     else AIError(ErrorCode.STORAGE_INTEGRITY_ERROR))
    fail_read = asyncio.Event()
    closed = []
    reported = []

    class Stream:
        def __init__(self, name: str) -> None:
            self.name = name

        def __aiter__(self) -> "Stream":
            return self

        async def __anext__(self) -> NoReturn:
            if self.name == "graph":
                await fail_read.wait()
                raise primary_error
            await asyncio.Event().wait()

        async def aclose(self) -> None:
            closed.append(self.name)

    class Graph(_TaskGraphService):
        def stream_events(
            self, graph_id: str, *, principal: Principal, after_event_seq: int = 0,
        ) -> Stream:
            return Stream("graph")

    def watch_tree(*args, ready=None, **kwargs):
        if ready is not None:
            ready.set()
        return Stream("execution")

    async def close_models(self) -> None:
        raise cleanup_error

    monkeypatch.setattr(ModelProjection, "close", close_models)
    monkeypatch.setattr(_run, "report_observation_error", reported.append)
    runtime = _Runtime()
    runtime.graph = Graph()
    run = _task_graph_run(runtime, "graph", Principal("owner", "tenant"), watch_tree)
    stream = run.watch(ready=asyncio.Event())
    await anext(stream)
    expected = cleanup_error if primary == "close" else primary_error
    with pytest.raises(type(expected)) as raised:
        if primary == "close":
            await stream.aclose()
        else:
            fail_read.set()
            async for _ in stream:
                pass
    if primary != "close" or not cleanup_cancel:
        assert raised.value is expected
    assert set(closed) == {"graph", "execution"}
    assert cleanup_error in reported


@pytest.mark.asyncio
async def test_watch_closes_final_replay_after_model_cleanup_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    closed = asyncio.Event()
    failure = RuntimeError("model close failed")

    class FinalStream:
        def __aiter__(self) -> "FinalStream":
            return self

        async def __anext__(self) -> NoReturn:
            raise StopAsyncIteration

        async def aclose(self) -> None:
            closed.set()

    async def final_projection(*args, **kwargs):
        return FinalStream()

    async def close_models(self) -> None:
        raise failure

    run = _task_graph_run(_Runtime(), "graph", Principal("owner", "tenant"), _watch_tree)
    monkeypatch.setattr(type(run), "_final_projection", final_projection)
    monkeypatch.setattr(ModelProjection, "close", close_models)
    with pytest.raises(RuntimeError) as raised:
        async for _ in run.watch():
            pass
    assert raised.value is failure
    assert closed.is_set()
