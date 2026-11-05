#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from linktools.ai.persistence import Persistence
"""Preparation uses captured control values without claiming producer authority."""
from ._runtime_test_helpers import model_features

import asyncio
from pathlib import Path

import pytest
from linktools.ai.persistence import PersistenceDomain
from linktools.ai.runtime import RuntimeFeatures
from linktools.ai.core import ExecutionStatus, SessionStatus
from linktools.ai.capability import ExecutionCapability
from pydantic_ai_harness.step_persistence import StepPersistence
from tests.ai._runtime_test_helpers import native_checkpoint
from linktools.ai.runtime import Runtime
from linktools.ai.persistence import PersistenceStorage
from linktools.ai.execution import ExecutionInputContext
from linktools.ai.runtime._local import LocalExecutionBackend
from pydantic_ai.messages import ModelResponse, TextPart

from ._runtime_test_helpers import _Models, _UsageFunctionModel
from ._session_tool_test_helpers import _split_sqlite_storage


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ("cancel", "close", "metadata"))
@pytest.mark.parametrize("restart", (False, True))
async def test_admission_capture_is_stable_while_cancellation_stops_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: str, restart: bool,
) -> None:
    launched, release = asyncio.Event(), asyncio.Event()
    calls = []

    async def model(messages, info):
        calls.append(messages)
        return ModelResponse(parts=[TextPart("answer")])

    group = RuntimeFeatures("launch-change")
    group.capability(ExecutionCapability(retain_history=True), scope="runtime")
    if restart:
        group.capability(native_checkpoint(), scope="runtime")
    group.agent("default", model="default", allow_tools=())
    models = _Models(_UsageFunctionModel(model))
    first_state = PersistenceStorage.sqlite(tmp_path / "state.db")
    second_state = PersistenceStorage.sqlite(tmp_path / "state.db")
    original = LocalExecutionBackend._run

    async def delayed(worker, *args, **kwargs):
        launched.set()
        await release.wait()
        return await original(worker, *args, **kwargs)

    monkeypatch.setattr(LocalExecutionBackend, "_run", delayed)
    async with Persistence.open('launch-change', storage=first_state) as persistence_48, Runtime.attach(persistence=persistence_48, features=model_features(models, group)) as first, Persistence.open('launch-change', storage=second_state) as persistence_48, Runtime.attach(persistence=persistence_48, features=model_features(models, group)) as second:
        session = await first.agents.get().create_session("session", metadata={"marker": "before"})
        execution = await session.start("question", idempotency_key="same")
        await asyncio.wait_for(launched.wait(), 5)
        try:
            remote = second.agents.get().session("session")
            if change == "cancel":
                await (await second.executions.get(execution.execution_id)).cancel(idempotency_key="remote-cancel")
            elif change == "close":
                await remote.close(force=True, idempotency_key="remote-close", wait_timeout_seconds=1)
            else:
                observed = await second._state.conversation.sessions.get("session", tenant_id=second.tenant_id)
                await remote.update(expected_revision=observed.revision, metadata={"marker": "after"},
                                    idempotency_key="remote-update")
        finally:
            release.set()
        result = (await asyncio.wait_for(execution.wait(), 5)).result
        if change == "metadata":
            assert result.status is ExecutionStatus.SUCCEEDED
            assert len(calls) == 1
            current = await first._state.execution.executions.get(execution.execution_id, tenant_id=first.tenant_id)
            from linktools.ai.inputs import ExecutionContextArchive
            context = await ExecutionContextArchive(
                first._state.execution.executions.state_store,
                first._state.archive_object_store(PersistenceDomain.EXECUTION),
                namespace=first.namespace, tenant_id=first.tenant_id,
                object_reader=lambda reference: first._state.read_archive_object(
                    PersistenceDomain.EXECUTION, reference,
                ),
            ).load(current.input_context)
            assert context.session_metadata == {"marker": "before"}
            following = await session.start("next question", idempotency_key="next")
            assert (await following.wait(timeout_seconds=5)).result.status is ExecutionStatus.SUCCEEDED
            following_record = await first._state.execution.executions.get(
                following.execution_id, tenant_id=first.tenant_id,
            )
            following_context = await ExecutionContextArchive(
                first._state.execution.executions.state_store,
                first._state.archive_object_store(PersistenceDomain.EXECUTION),
                namespace=first.namespace, tenant_id=first.tenant_id,
                object_reader=lambda reference: first._state.read_archive_object(
                    PersistenceDomain.EXECUTION, reference,
                ),
            ).load(following_record.input_context)
            assert following_context.session_metadata == {"marker": "after"}
        else:
            assert result.status is ExecutionStatus.CANCELLED
            assert calls == []
        session_record = await second._state.conversation.sessions.get("session", tenant_id=second.tenant_id)
        assert session_record.active_execution_id is None
        if change == "close":
            assert session_record.status is SessionStatus.CLOSED


@pytest.mark.asyncio
@pytest.mark.parametrize("restart", (False, True))
async def test_other_runtime_claims_before_delayed_worker_without_duplicate_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, restart: bool,
) -> None:
    launched, release_first, provider_entered, release_provider = (asyncio.Event() for _ in range(4))
    calls, workers = [], []

    async def model(messages, info):
        calls.append(messages)
        provider_entered.set()
        await release_provider.wait()
        return ModelResponse(parts=[TextPart("answer")])

    group = RuntimeFeatures("launch-winner")
    group.capability(ExecutionCapability(retain_history=True), scope="runtime")
    if restart:
        group.capability(native_checkpoint(), scope="runtime")
    group.agent("default", model="default", allow_tools=())
    models = _Models(_UsageFunctionModel(model))
    first_state = PersistenceStorage.sqlite(tmp_path / "state.db")
    second_state = PersistenceStorage.sqlite(tmp_path / "state.db")
    original = LocalExecutionBackend._run

    async def delayed(worker, *args, **kwargs):
        workers.append(worker)
        if len(workers) == 1:
            launched.set()
            await release_first.wait()
        return await original(worker, *args, **kwargs)

    monkeypatch.setattr(LocalExecutionBackend, "_run", delayed)
    async with Persistence.open('launch-winner', storage=first_state) as persistence_119, Runtime.attach(persistence=persistence_119, features=model_features(models, group)) as first, Persistence.open('launch-winner', storage=second_state) as persistence_119, Runtime.attach(persistence=persistence_119, features=model_features(models, group)) as second:
        await first.agents.get().create_session("session")
        one = await first.agents.get().session("session").start("question", idempotency_key="same")
        await asyncio.wait_for(launched.wait(), 5)
        try:
            two = await second.agents.get().session("session").start("question", idempotency_key="same")
            assert two.execution_id == one.execution_id
            await asyncio.wait_for(provider_entered.wait(), 5)
        finally:
            release_first.set()
            release_provider.set()
        results = await asyncio.wait_for(asyncio.gather(one.wait(), two.wait()), 5)
        assert all(value.result.status is ExecutionStatus.SUCCEEDED for value in results)
        assert len(calls) == 1
        current = await first._state.execution.executions.get(one.execution_id, tenant_id=first.tenant_id)
        assert current.agent_run_seq == 1
        assert (await first._state.conversation.sessions.get("session", tenant_id=first.tenant_id)).active_execution_id is None


@pytest.mark.asyncio
@pytest.mark.parametrize("split_storage", (False, True))
async def test_restartable_launch_reuses_reads_but_requires_attempt_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, split_storage: bool,
) -> None:
    calls = []
    claimed = False

    async def model(messages, info):
        assert claimed
        calls.append("provider")
        return ModelResponse(parts=[TextPart("answer")])

    features = RuntimeFeatures("application")
    features.capability(ExecutionCapability(retain_history=True), scope="runtime")
    features.capability(native_checkpoint(), scope="runtime")
    features.agent("default", model="default", allow_tools=())
    storage = (_split_sqlite_storage if split_storage else PersistenceStorage.sqlite)(tmp_path / "state.db")
    async with Persistence.open("initial-capture", storage=storage) as persistence, Runtime.attach(
        persistence=persistence, features=model_features(_Models(_UsageFunctionModel(model)), features),
    ) as runtime:
        await runtime.agents.get().create_session("session")
        backend = runtime._execution_service.runtime_backend()
        run = backend._run
        before_claim = False
        load = backend.load_execution
        checkpoint_get = persistence.execution.executions.get

        async def read_execution(*args, **kwargs):
            assert not before_claim, "launch already captured the execution before the attempt CAS"
            return await load(*args, **kwargs)

        async def read_checkpoint(*args, **kwargs):
            assert not before_claim, "launch already captured the checkpoint before the attempt CAS"
            return await checkpoint_get(*args, **kwargs)

        def observe_claim(claim):
            async def commit(*args, **kwargs):
                nonlocal before_claim, claimed
                before_claim = False
                result = await claim(*args, **kwargs)
                claimed = True
                return result
            return commit

        async def observed_run(*args, **kwargs):
            nonlocal before_claim
            before_claim = True
            try:
                return await run(*args, **kwargs)
            finally:
                before_claim = False

        monkeypatch.setattr(backend, "_run", observed_run)
        monkeypatch.setattr(backend, "load_execution", read_execution)
        monkeypatch.setattr(persistence.execution.executions, "get", read_checkpoint)
        for method in ("commit_agent_attempt_checkpoint", "commit_prepared_agent_attempt_checkpoint"):
            monkeypatch.setattr(
                backend._runtime_commands, method,
                observe_claim(getattr(backend._runtime_commands, method)),
            )
        execution = await runtime.agents.get().session("session").start("question", idempotency_key="start")
        result = (await execution.wait(timeout_seconds=10)).result
        assert result.status is ExecutionStatus.SUCCEEDED
        assert result.output == {"text": "answer"}
        assert calls == ["provider"]
