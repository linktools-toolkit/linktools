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


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["memory", "filesystem"])
async def test_evaluation_real_task_graphs_watch_and_wait_share_resumable_events(tmp_path, backend):
    from linktools.ai.evaluation import (
        CandidateSpec, CaseRef, CaseSpec, DatasetRef, DatasetSpec, EvaluationSpec, EvaluationPolicy, StartEvaluationRequest,
    )
    from linktools.ai.runtime import Runtime, RuntimeStorage
    from linktools.ai.task import Task
    from .test_evaluation_consumers import FixtureModels, echo, exact, rule_scorer, CONTEXT, PRINCIPAL as principal

    storage = RuntimeStorage.in_memory() if backend == "memory" else RuntimeStorage.filesystem(tmp_path)
    target = Task("watch.echo", echo, effect_policy="none")
    scorer = Task("watch.exact", exact, effect_policy="none")
    async with Runtime.open("evaluation-watch-real", models=FixtureModels(), storage=storage, context=CONTEXT) as runtime:
        dataset = await runtime.evaluations.publish_dataset(DatasetSpec(
            DatasetRef("watch-cases", 1), cases=(CaseSpec.task(
                CaseRef("watch-cases", "one", 1), input={"answer": "yes"}, expected="yes",
            ),),
        ), principal=principal, idempotency_key="publish-watch-cases")
        handle = await runtime.evaluations.start(StartEvaluationRequest(EvaluationSpec(
            dataset, (CandidateSpec("current", task=target.ref),), (rule_scorer(scorer),),
            policy=EvaluationPolicy(allow_volatile=backend == "memory"),
        ), principal, "start-watch-cases"), engine=runtime.tasks.bind(target, scorer))
        seen = []
        async def consume(item):
            seen.append(item)
        outcome = await handle.wait(on_event=consume, timeout_seconds=15)
        assert outcome.result.completion == "complete"
        assert outcome.observation_error is None
        remaining = [item async for item in handle.watch(cursor=outcome.cursor)]
        all_items = seen + remaining
        identities = [(item.graph_id, item.event.sequence) for item in all_items]
        assert len(identities) == len(set(identities))
        assert len({item.graph_id for item in all_items}) == 2
        assert [item async for item in handle.watch(cursor=all_items[-1].cursor)] == []


@pytest.mark.asyncio
@pytest.mark.parametrize("cleanup_fails", [False, True], ids=["owned-cancellation", "contract-error"])
async def test_evaluation_wait_preserves_outcome_when_live_stream_cleanup_is_interrupted(
    cleanup_fails: bool, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from collections.abc import AsyncIterator

    from linktools.ai.runtime._observation import _ObservationSession

    owner = Evaluations([intent("first"), intent("second")])
    cancelled = {graph_id: asyncio.Event() for graph_id in ("first", "second")}
    release = asyncio.Event()
    pending = asyncio.Event()
    completed = SimpleNamespace(completion="complete")
    contract_error = AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    observer = None

    async def inspect(experiment_id: str, principal: Principal) -> SimpleNamespace:
        nonlocal observer
        current = asyncio.current_task()
        if observer is None:
            observer = current
            return SimpleNamespace(completion="running")
        if current is not observer:
            await cancelled["first"].wait()
            await cancelled["second"].wait()
        return completed

    async def watch_graph(
        graph_id: str, principal: Principal, cursor: str | None,
        content: bool, ready: asyncio.Event | None,
    ) -> AsyncIterator[TaskGraphRunEvent]:
        assert ready is not None
        ready.set()
        try:
            await pending.wait()
        except asyncio.CancelledError:
            cancelled[graph_id].set()
            if graph_id == "second":
                await release.wait()
                if cleanup_fails:
                    raise contract_error
            # Python 3.10 can discard the cancellation message after a result read.
            raise asyncio.CancelledError from None
        yield event(graph_id, 1, content)

    register = owner._register_observation

    def register_observation(session: _ObservationSession) -> None:
        register(session)
        stop = session.stop

        def stop_and_release() -> None:
            stop()
            # Let observer cancellation re-enter cleanup before this child resumes.
            asyncio.get_running_loop().call_soon(release.set)

        monkeypatch.setattr(session, "stop", stop_and_release)

    monkeypatch.setattr(owner, "_inspect", inspect)
    monkeypatch.setattr(owner, "_watch_graph", watch_graph)
    monkeypatch.setattr(owner, "_register_observation", register_observation)
    seen = []

    async def consume(item: TaskGraphRunEvent) -> None:
        seen.append(item)

    if cleanup_fails:
        with pytest.raises(AIError) as error:
            await run(owner).wait(on_event=consume, timeout_seconds=1, close_timeout_seconds=1)
        assert error.value is contract_error
    else:
        outcome = await run(owner).wait(on_event=consume, timeout_seconds=1, close_timeout_seconds=1)
        assert outcome.result is completed
        assert outcome.cursor is None
        assert outcome.observation_error is None
    assert all(barrier.is_set() for barrier in cancelled.values())
    assert seen == []
    assert not owner.sessions
