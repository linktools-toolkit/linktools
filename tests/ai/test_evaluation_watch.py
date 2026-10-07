#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from linktools.ai.core import Principal, TaskStatus
from linktools.ai.errors import AIError, ErrorCode, ObservationError
from linktools.ai.runtime import EvaluationRun
from linktools.ai.runtime.service_api import TaskGraphRunEvent
from linktools.ai.runtime._watch_cursor import (
    decode_evaluation_watch_cursor, decode_graph_watch_cursor, encode_graph_watch_cursor,
)
from linktools.ai.task import TaskEvent, TaskEventType

PRINCIPAL = Principal("owner", "tenant")


def intent(graph_id, *, confirmed=True, released=False):
    return SimpleNamespace(confirmed=confirmed, released=released,
                           submission=SimpleNamespace(graph=SimpleNamespace(graph_id=graph_id)))


def event(graph_id, sequence, content=False):
    cursor = encode_graph_watch_cursor("eval-watch", "tenant", graph_id,
                                      include_content=content, graph_sequence=sequence,
                                      execution_sequences={})
    return TaskGraphRunEvent(graph_id, None, TaskEvent(
        1, graph_id, sequence, TaskEventType.GRAPH_CHANGED,
        datetime.now(timezone.utc), TaskStatus.RUNNING, TaskStatus.PENDING,
    ), cursor)


class Evaluations:
    _namespace = "eval-watch"

    def __init__(self, members=(), completion="running"):
        self.members = list(members)
        self.completion = completion
        self.streams = []
        self.replays = []
        self.sessions = set()
        self.record_error = None
        self.cleanup_error = None
        self.stream_release = asyncio.Event()
        self.sequences = {}
        self._graph = SimpleNamespace(state=self.graph_state)

    async def graph_state(self, graph_id, *, principal):
        return SimpleNamespace(event_sequence=self.sequences.get(graph_id, 1))

    async def _record(self, experiment_id, principal):
        if self.record_error:
            raise self.record_error
        return SimpleNamespace(intents=tuple(self.members))

    async def _inspect(self, experiment_id, principal):
        return SimpleNamespace(completion=self.completion)

    def _register_observation(self, session):
        self.sessions.add(session)

    def _release_observation(self, session):
        self.sessions.discard(session)

    def _after(self, graph_id, cursor, content):
        if cursor is None:
            return 0
        return decode_graph_watch_cursor(self._namespace, "tenant", graph_id, cursor,
                                         include_content=content)[0]

    async def _watch_graph(self, graph_id, principal, cursor, content, ready):
        self.streams.append(graph_id)
        sequence = self._after(graph_id, cursor, content)
        if ready is not None:
            ready.set()
        try:
            for number in range(sequence + 1, self.sequences.get(graph_id, 1) + 1):
                yield event(graph_id, number, content)
            await self.stream_release.wait()
        finally:
            if self.cleanup_error:
                raise self.cleanup_error

    async def _replay_graph(self, graph_id, principal, cursor, content):
        self.replays.append(graph_id)
        sequence = self._after(graph_id, cursor, content)
        async def values():
            for number in range(sequence + 1, 3):
                yield event(graph_id, number, content)
        return values()


def run(owner, experiment="experiment"):
    return EvaluationRun(owner, experiment, PRINCIPAL)


@pytest.mark.asyncio
@pytest.mark.parametrize("completion", ["complete", "cancelled", "needs_attention"])
async def test_evaluation_finite_drain_includes_confirmed_released_members(completion):
    owner = Evaluations([intent("target", released=True), intent("score"),
                         intent("unsettled", confirmed=False)], completion)
    items = [item async for item in run(owner).watch(include_content=True)]
    assert {item.graph_id for item in items} == {"target", "score"}
    assert owner.streams == []
    assert len(items) == 4
    cursors = decode_evaluation_watch_cursor("eval-watch", "tenant", "experiment", items[-1].cursor,
                                             include_content=True)
    assert set(cursors) == {"target", "score"}
    assert [item async for item in run(owner).watch(cursor=items[-1].cursor, include_content=True)] == []


@pytest.mark.asyncio
async def test_evaluation_observation_spans_empty_target_scorer_gap():
    owner = Evaluations([intent("target", released=True)])
    owner.stream_release.set()
    stream = run(owner).watch()
    first = await anext(stream)
    assert first.graph_id == "target"
    pending = asyncio.create_task(anext(stream))
    await asyncio.sleep(0.08)
    assert not pending.done()
    owner.members.append(intent("score"))
    second = await asyncio.wait_for(pending, 1)
    assert second.graph_id == "score"
    owner.completion = "needs_attention"
    rest = [item async for item in stream]
    assert [(item.graph_id, item.event.sequence) for item in rest] == [("score", 2), ("target", 2)]


@pytest.mark.asyncio
async def test_evaluation_terminal_transition_captures_latest_confirmed_members():
    owner = Evaluations()
    async def inspect(experiment_id, principal):
        owner.members = [intent("late-score", released=True)]
        return SimpleNamespace(completion="complete")
    owner._inspect = inspect
    items = [item async for item in run(owner).watch()]
    assert {item.graph_id for item in items} == {"late-score"}


@pytest.mark.asyncio
async def test_evaluation_cursor_is_scoped_and_unknown_graphs_fail_before_wait_success():
    owner = Evaluations([intent("target")], "complete")
    first = await anext(run(owner).watch())
    async def ignore(item):
        pass
    with pytest.raises(AIError) as error:
        await run(owner, "another-experiment").wait(on_event=ignore, cursor=first.cursor)
    assert error.value.code is ErrorCode.CURSOR_INVALID
    owner.members = []
    with pytest.raises(AIError) as error:
        await run(owner).wait(on_event=ignore, cursor=first.cursor)
    assert error.value.code is ErrorCode.CURSOR_INVALID
    assert not owner.sessions


@pytest.mark.asyncio
async def test_evaluation_cleanup_error_preserves_evaluation_cursor_and_primary_read_error():
    owner = Evaluations([intent("target")])
    stream = run(owner).watch()
    first = await anext(stream)
    owner.cleanup_error = ObservationError("stream", cursor="graph-cursor", safe_details={"phase": "cleanup"})
    with pytest.raises(ObservationError) as error:
        await stream.aclose()
    assert error.value.cursor == first.cursor

    owner = Evaluations([intent("target")])
    stream = run(owner).watch()
    await anext(stream)
    cause = AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    owner.record_error = cause
    owner.cleanup_error = ObservationError("stream", cursor="graph-cursor", safe_details={"phase": "cleanup"})
    with pytest.raises(AIError) as error:
        await anext(stream)
    assert error.value is cause


@pytest.mark.asyncio
async def test_evaluation_wait_returns_fixed_result_and_acknowledges_callback_only():
    owner = Evaluations(completion="needs_attention")
    result = await run(owner).wait()
    assert result.result.completion == "needs_attention"
    assert result.cursor is None and result.observation_error is None
    assert owner.streams == owner.replays == []
    owner.members = [intent("target")]
    seen = []
    async def consume(item):
        seen.append(item.cursor)
    outcome = await run(owner).wait(on_event=consume)
    assert outcome.result.completion == "needs_attention"
    assert outcome.cursor == (seen[-1] if seen else None)
    assert not owner.sessions


@pytest.mark.asyncio
async def test_evaluation_reopens_a_known_graph_after_waiting_stage_resumes():
    owner = Evaluations([intent("human-score")])
    owner.stream_release.set()
    stream = run(owner).watch()
    assert (await anext(stream)).event.sequence == 1
    pending = asyncio.create_task(anext(stream))
    await asyncio.sleep(0.08)
    assert not pending.done()
    assert owner.streams == ["human-score"]
    owner.sequences["human-score"] = 2
    assert (await asyncio.wait_for(pending, 1)).event.sequence == 2
    assert owner.streams == ["human-score", "human-score"]
    await stream.aclose()
