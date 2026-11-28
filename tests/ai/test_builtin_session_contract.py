#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Built-in sessions keep authority independent of optional execution retention."""

import asyncio
from contextlib import asynccontextmanager
from copy import deepcopy

import pytest
from pydantic_ai.messages import ModelResponse, TextPart

from linktools.ai.core import CommitPolicy, ExecutionStatus, Principal, TenantAuthorizationPolicy
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence import Persistence, PersistencePlan, PersistenceStorage
from linktools.ai.runtime import Runtime, RuntimeFeatures
from linktools.ai.storage import InMemoryCacheBackend, StorageRoute

from ._runtime_test_helpers import configure_execution
from ._runtime_test_helpers import _Models, _UsageFunctionModel, model_features


_OWNER = Principal("owner", "tenant")
_NAMESPACE = "builtin-session"


class _Authority(TenantAuthorizationPolicy):
    """Explicit host policy for this isolated native Session test."""


def _storage(path, backend="sqlite", cache=None):
    if backend == "split_sqlite":
        return PersistenceStorage.from_plan(PersistencePlan(
            conversation=StorageRoute.sqlite(path / "conversation.db", cache=cache),
            execution=StorageRoute.sqlite(path / "execution.db", cache=cache),
        ), coordination=cache)
    if backend == "local_sqlite":
        return PersistenceStorage.from_root(path / "state", exclusive_writer=True, cache=cache, coordination=cache)
    return PersistenceStorage.sqlite(path / "state.db", cache=cache, coordination=cache)


@asynccontextmanager
async def _open(path, model, *, retain=False, backend="sqlite", cache=None, commit=None):
    features = RuntimeFeatures(_NAMESPACE)
    features.agent("default", model="default", allow_tools=())
    if retain:
        configure_execution(features, retain_history=True, commit=commit or CommitPolicy.sync())
    async with Runtime.open(
        _NAMESPACE, storage=_storage(path, backend, cache), tenant_id="tenant",
        features=model_features(_Models(_UsageFunctionModel(model)), features),
        authorization=_Authority(),
    ) as runtime:
        yield runtime


async def _start(runtime, prompt, key, *, session_id="session"):
    return await runtime.sessions.start(
        session_id, prompt, principal=_OWNER, agent=runtime.agents.get(), idempotency_key=key,
    )


async def _wait(turn, status=ExecutionStatus.SUCCEEDED):
    result = await turn.wait_final(timeout_seconds=5)
    assert result.status is status
    return result


async def _create(runtime, session_id="session", **kwargs):
    return await runtime.agents.get().create_session(session_id, principal=_OWNER, **kwargs)


async def _view(runtime, session_id="session"):
    return await (await runtime.sessions.get(session_id, principal=_OWNER)).inspect()


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["sqlite", "local_sqlite"])
async def test_builtin_metadata_lifecycle_survives_feature_and_cache_changes(tmp_path, backend):
    async def model(messages, info):
        pytest.fail("metadata operations must not execute a model")

    async with _open(tmp_path, model, backend=backend) as runtime:
        session = await _create(runtime, metadata={"title": "first"}, idempotency_key="create")
        replay = await _create(runtime, metadata={"title": "first"}, idempotency_key="create")
        assert replay.session_id == session.session_id
        view = await _view(runtime)
        assert await session.update(expected_revision=view["revision"], metadata={"title": "second"})
        assert not await session.update(expected_revision=view["revision"], metadata={"title": "stale"})
        assert (await _view(runtime))["metadata"]["title"] == "second"
        with pytest.raises(AIError) as foreign:
            await runtime.sessions.get("session", principal=Principal("owner", "other-tenant"))
        assert foreign.value.code is ErrorCode.AUTHORIZATION_DENIED

    async with _open(tmp_path, model, backend=backend, retain=True,
                     cache=InMemoryCacheBackend(max_bytes=1_000_000)) as runtime:
        page = await runtime.history.turn_sessions(principal=_OWNER, session_ids=["session"])
        assert [(item["session_id"], item["metadata"]["title"]) for item in page["items"]] == [("session", "second")]
        session = await runtime.sessions.get("session", principal=_OWNER)
        assert await session.close(expected_revision=(await _view(runtime))["revision"])

    async with _open(tmp_path, model, backend=backend) as runtime:
        assert (await _view(runtime))["status"] == "CLOSED"


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["sqlite", "local_sqlite", "split_sqlite"])
async def test_default_turn_retains_session_context_without_execution_retention(tmp_path, backend):
    calls = []

    async def model(messages, info):
        calls.append(deepcopy(messages))
        return ModelResponse(parts=[TextPart("answer")])

    async with _open(tmp_path, model, backend=backend) as runtime:
        await _create(runtime)
        turn = await _start(runtime, "first", "first")
        await _wait(turn)
        assert (await _view(runtime))["active_turn_id"] is None
        assert await runtime._state.execution.executions.get(turn.key.turn_id, tenant_id="tenant") is None

    async with _open(tmp_path, model, backend=backend, retain=True) as runtime:
        assert (await _view(runtime))["active_turn_id"] is None
        await _wait(await _start(runtime, "second", "second"))
        assert len(calls) == 2
        assert "first" in repr(calls[-1])
        session = await runtime.sessions.get("session", principal=_OWNER)
        assert await session.close(expected_revision=(await _view(runtime))["revision"])


@pytest.mark.asyncio
async def test_turn_advances_retained_context_through_feature_toggles(tmp_path):
    calls = []

    async def model(messages, info):
        calls.append(deepcopy(messages))
        return ModelResponse(parts=[TextPart("answer")])

    for index, retain in enumerate((True, False, True)):
        async with _open(tmp_path, model, retain=retain) as runtime:
            if index == 0:
                await _create(runtime)
            await _wait(await _start(runtime, f"question-{index}", f"request-{index}"))
            session = await runtime.sessions.get("session", principal=_OWNER)
            assert len((await session.history())["timeline"]["items"]) == index + 1
    assert "question-0" in repr(calls[2]) and "question-1" in repr(calls[2])


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", [ExecutionStatus.FAILED, ExecutionStatus.CANCELLED])
async def test_interrupted_turn_preserves_prior_context_without_execution_retention(tmp_path, outcome):
    calls = []
    entered = asyncio.Event()
    phase = "seed"

    async def model(messages, info):
        calls.append((deepcopy(messages), info.instructions))
        if phase == "interrupt":
            entered.set()
            if outcome is ExecutionStatus.FAILED:
                raise RuntimeError("model failed before response")
            await asyncio.Event().wait()
        return ModelResponse(parts=[TextPart("retained answer")])

    async with _open(tmp_path, model, retain=True) as runtime:
        await _create(runtime)
        await _wait(await _start(runtime, "retained question", "seed"))

    phase = "interrupt"
    async with _open(tmp_path, model) as runtime:
        turn = await _start(runtime, "aborted question", "interrupt")
        await asyncio.wait_for(entered.wait(), 5)
        if outcome is ExecutionStatus.CANCELLED:
            assert await turn.cancel()
        await _wait(turn, outcome)
        assert (await _view(runtime))["active_turn_id"] is None

    phase = "continue"
    async with _open(tmp_path, model, retain=True) as runtime:
        await _wait(await _start(runtime, "continue question", "continue"))
    messages, instructions = calls[-1]
    assert "retained question" in repr(messages)
    assert "aborted question" in repr(messages)
    assert "The previous turn was interrupted. Its results were not confirmed." in instructions


@pytest.mark.asyncio
async def test_history_and_fork_do_not_depend_on_standalone_execution_headers(tmp_path, monkeypatch):
    calls = []

    async def model(messages, info):
        calls.append(deepcopy(messages))
        return ModelResponse(parts=[TextPart("answer")])

    async with _open(tmp_path, model, retain=True) as runtime:
        await _create(runtime, "source")
        await _wait(await _start(runtime, "retained source question", "seed", session_id="source"))

    async with _open(tmp_path, model) as runtime:
        async def no_execution_evidence(*args, **kwargs):
            pytest.fail("canonical Session authority must not require standalone Execution evidence")

        monkeypatch.setattr(runtime._state.execution.executions, "get", no_execution_evidence)
        source = await runtime.sessions.get("source", principal=_OWNER)
        fork = await source.fork("fork", expected_revision=(await _view(runtime, "source"))["revision"],
                                 idempotency_key="fork")
        assert fork is not None
        assert len((await fork.history())["timeline"]["items"]) == 1
        await _wait(await _start(runtime, "fork question", "child", session_id="fork"))
        assert len((await fork.history())["timeline"]["items"]) == 2
        assert len((await source.history())["timeline"]["items"]) == 1
    assert "retained source question" in repr(calls[-1])


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["running", "before_publish"])
async def test_pending_turn_keeps_durable_owner_and_publishes_once_across_execution_policies(
    tmp_path, monkeypatch, phase,
):
    held, finish = asyncio.Event(), asyncio.Event()
    calls = []

    async def model(messages, info):
        calls.append(messages)
        if phase == "running":
            held.set()
            await finish.wait()
        return ModelResponse(parts=[TextPart("answer")])

    cache = InMemoryCacheBackend(max_bytes=4_000_000)
    async with (
        _open(tmp_path, model, retain=True, commit=CommitPolicy.terminal(), cache=cache,
              backend="split_sqlite") as owner,
        _open(tmp_path, model, retain=True, cache=cache, backend="split_sqlite") as peer,
    ):
        session = await _create(owner)
        store = owner._state.session_turns
        write_terminal = store.finish

        async def hold_publication(*args, **kwargs):
            if phase == "before_publish":
                held.set()
                await finish.wait()
            return await write_terminal(*args, **kwargs)

        monkeypatch.setattr(store, "finish", hold_publication)
        turn = await _start(owner, "question", "request")
        try:
            await asyncio.wait_for(held.wait(), 5)
            assert (await _view(peer))["active_turn_id"] == turn.key.turn_id
            assert (await session.history())["timeline"]["items"] == []
            assert await owner._state.execution.executions.get(turn.key.turn_id, tenant_id="tenant") is None
            same = await _start(peer, "question", "request")
            assert same.key == turn.key
            with pytest.raises(AIError) as conflict:
                await _start(peer, "different", "request")
            assert conflict.value.code is ErrorCode.IDEMPOTENCY_CONFLICT
            with pytest.raises(AIError) as busy:
                await _start(peer, "next", "next")
            assert busy.value.code is ErrorCode.SESSION_BUSY
            assert len(calls) == 1
        finally:
            finish.set()
        assert (await _wait(turn)).view["output"] == {"text": "answer"}
        assert (await _wait(same)).view["output"] == {"text": "answer"}
        assert (await _view(peer))["active_turn_id"] is None
        assert len((await session.history())["timeline"]["items"]) == 1
        assert len(calls) == 1


@pytest.mark.asyncio
async def test_memory_unretained_context_unavailable_even_after_execution_header_pruning(monkeypatch):
    calls = []

    async def model(messages, info):
        calls.append(messages)
        return ModelResponse(parts=[TextPart("unretained answer")])

    features = RuntimeFeatures(_NAMESPACE)
    features.agent("default", model="default", allow_tools=())
    models = _Models(_UsageFunctionModel(model))
    storage = PersistenceStorage.in_memory()
    async with Persistence.open(_NAMESPACE, storage=storage, tenant_id="tenant") as persistence:
        async with Runtime.open(persistence=persistence, features=model_features(models, features)) as runtime:
            session = await _create(runtime)
            result = await (await session.start("first", idempotency_key="first")).wait(timeout_seconds=5)
            assert result.result.status is ExecutionStatus.SUCCEEDED
            idle = await runtime.sessions.wait_inactive("session", principal=_OWNER, timeout_seconds=5)
            assert idle.active_execution_id is None and idle.history_quality == "unavailable"

        retained = RuntimeFeatures(_NAMESPACE, defaults={"execution": {"retain_history": True}})
        retained.agent("default", model="default", allow_tools=())
        async with Runtime.open(persistence=persistence, features=model_features(models, retained)) as runtime:
            session = await runtime.sessions.get("session", principal=_OWNER)
            assert (await runtime.sessions.reconcile("session", principal=_OWNER)).history_quality == "unavailable"

            async def no_execution_evidence(*args, **kwargs):
                pytest.fail("missing-context authority must survive execution pruning")

            monkeypatch.setattr(runtime._state.execution.executions, "get", no_execution_evidence)
            for operation in (lambda: session.start("must not forget", idempotency_key="second"),
                              lambda: session.fork("fork", idempotency_key="fork")):
                with pytest.raises(AIError) as missing:
                    await operation()
                assert missing.value.code is ErrorCode.SESSION_HISTORY_UNAVAILABLE
            assert len(calls) == 1


@pytest.mark.asyncio
async def test_memory_coordination_does_not_opt_default_execution_into_terminal_admission():
    class ClaimCache(InMemoryCacheBackend):
        claims = 0

        async def claim_turn(self, *args, **kwargs):
            self.claims += 1
            return await super().claim_turn(*args, **kwargs)

    cache = ClaimCache(max_bytes=4_000_000)
    entered, finish = asyncio.Event(), asyncio.Event()

    async def model(messages, info):
        entered.set()
        await finish.wait()
        return ModelResponse(parts=[TextPart("answer")])

    features = RuntimeFeatures(_NAMESPACE, defaults={"execution": {"retain_history": True}})
    features.agent("default", model="default", allow_tools=())
    async with Runtime.open(
        _NAMESPACE, storage=PersistenceStorage.in_memory(cache=cache, coordination=cache), tenant_id="tenant",
        features=model_features(_Models(_UsageFunctionModel(model)), features),
    ) as runtime:
        session = await _create(runtime)
        execution = await session.start("question")
        try:
            await asyncio.wait_for(entered.wait(), 5)
            record = await runtime._state.execution.executions.get(execution.execution_id, tenant_id="tenant")
            assert record is not None and record.status is ExecutionStatus.STARTED
        finally:
            finish.set()
        assert (await execution.wait(timeout_seconds=5)).result.status is ExecutionStatus.SUCCEEDED
        assert (await session.history()).items
    assert cache.claims >= 1
