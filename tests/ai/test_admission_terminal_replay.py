#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Canonical admission receipts have the same replay semantics after a race."""

import asyncio
from dataclasses import replace
from pathlib import Path

import pytest
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo

from linktools.ai.capability import ExecutionCapability
from linktools.ai.core import CommitPolicy, ExecutionStatus, IdempotencyStatus, ResourceKind, RunBudget
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.execution import Execution, ExecutionStartReservation, ExecutionStartReservationResult
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.persistence.coordination import PendingTurnCoordination
from linktools.ai.runtime import Runtime, RuntimeFeatures
from linktools.ai.storage import InMemoryCacheBackend, Lease

from ._runtime_test_helpers import _Models, _UsageFunctionModel, model_features


async def _terminal_race(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    backend: str,
    mode: str,
    outcome: str,
    mismatch: bool = False,
    hold: bool = False,
    receipt_fault: str | None = None,
    session: bool = False,
) -> None:
    absent, release, provider_entered = (asyncio.Event() for _ in range(3))
    provider_calls = 0

    async def provider(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal provider_calls
        del messages, info
        provider_calls += 1
        provider_entered.set()
        if outcome == "cancelled":
            await asyncio.Event().wait()
        if outcome == "failed":
            raise ValueError("model failed")
        return ModelResponse(parts=[TextPart("canonical result")])

    coordination = None if mode == "sync" else InMemoryCacheBackend(max_bytes=4 * 1024 * 1024)
    turn_released = asyncio.Event()
    if coordination is not None:
        original_release = coordination.release_turn

        async def release_turn(lease: Lease) -> bool:
            released = await original_release(lease)
            if released:
                turn_released.set()
            return released

        monkeypatch.setattr(coordination, "release_turn", release_turn)
    storage = (PersistenceStorage.in_memory(coordination=coordination) if backend == "memory" else
               PersistenceStorage.sqlite(tmp_path / "state.db", coordination=coordination))
    features = RuntimeFeatures("terminal-replay")
    features.capability(ExecutionCapability(retain_history=True, commit=(
        CommitPolicy.terminal() if mode == "pending" else CommitPolicy.sync()
    )), scope="runtime")
    features.agent("default", model="default", allow_tools=())
    model_features(_Models(_UsageFunctionModel(provider)), features)

    async def start(runtime: Runtime[object], *, prompt: str, hold_id: str | None) -> Execution[object]:
        if not hold:
            agent = runtime.agents.get()
            target = agent.session("session") if session else agent
            return await target.start(prompt, idempotency_key="same-key")
        return await runtime.agents.get().execution_port.start(
            prompt, files=(), output=None, principal=None, session_id=None,
            idempotency_key="same-key", mode="run", planning=None, thinking=None,
            dependency_hold_id=hold_id, budget=RunBudget(model_requests=2),
        )

    async with Persistence.open("terminal-replay", storage=storage) as state:
        repository = state.execution.idempotency
        receipt_faults: list[str] = []
        if receipt_fault is not None:
            original_reserve = state.execution.executions.reserve_start

            async def reserve(
                reservation: ExecutionStartReservation, *, input_context_write: object = None,
            ) -> ExecutionStartReservationResult:
                assert input_context_write is None
                result = await original_reserve(reservation)
                if not result.created:
                    # Fault only the returned receipt; keep the durable winner
                    # unchanged so replay rejection cannot mask a second run.
                    receipt_faults.append(receipt_fault)
                    if receipt_fault == "not_started":
                        return replace(result, execution=replace(result.execution, started_at=None))
                    if receipt_fault == "missing_result":
                        # Inject a corrupt repository receipt without weakening constructor validation.
                        corrupt = replace(result.execution)
                        object.__setattr__(corrupt, "result", None)
                        return replace(result, execution=corrupt)
                    if receipt_fault == "digest":
                        return replace(result, idempotency=replace(result.idempotency, result_digest="f" * 64))
                    return replace(result, idempotency=replace(result.idempotency, status=IdempotencyStatus.START_UNKNOWN))
                return result

            monkeypatch.setattr(state.execution.executions, "reserve_start", reserve)
        async with Runtime.attach(persistence=state, features=features) as delayed_runtime:
            if session:
                await delayed_runtime.agents.get().create_session("session")
            original_admit = delayed_runtime._execution_service._admit_execution_start

            async def delayed_admit(*args, **kwargs):
                absent.set()
                await release.wait()
                return await original_admit(*args, **kwargs)

            monkeypatch.setattr(delayed_runtime._execution_service, "_admit_execution_start", delayed_admit)
            delayed = asyncio.create_task(start(
                delayed_runtime, prompt="different" if mismatch else "same request", hold_id="late-hold",
            ), name="late-replay")
            try:
                await asyncio.wait_for(absent.wait(), 5)
                async with Runtime.attach(persistence=state, features=features) as winner_runtime:
                    assert winner_runtime.default_principal == delayed_runtime.default_principal
                    winner = await start(winner_runtime, prompt="same request", hold_id="original-hold")
                    if outcome == "cancelled":
                        await asyncio.wait_for(provider_entered.wait(), 5)
                        await winner.cancel(idempotency_key="stop-winner")
                    result = (await winner.wait(timeout_seconds=5)).result
                    assert result.status is ExecutionStatus(outcome.upper())
                    identities = await repository.list_by_resource(
                        ResourceKind.EXECUTION, winner.execution_id, tenant_id=state.tenant_id,
                    )
                    assert len(identities) == 1
                    identity = identities[0]
                    source = await state.execution.executions.get(winner.execution_id, tenant_id=state.tenant_id)
                    assert source is not None and source.result is not None
                    assert source.status is result.status
                    budget_before = await winner.budget_usage() if hold else None
                # Only the winner is closed. The delayed Runtime remains open
                # while its real absent observation becomes stale.
                turn_released.clear()
                release.set()
                if receipt_fault is not None:
                    with pytest.raises(AIError) as rejected:
                        await asyncio.wait_for(delayed, 5)
                    assert receipt_faults == [receipt_fault]
                    if outcome != "succeeded" and (
                        receipt_fault == "not_started" or receipt_fault == "missing_result" and not session
                    ):
                        assert rejected.value.code.value == identity.error_code
                    else:
                        assert rejected.value.code is (
                            ErrorCode.EXECUTION_START_UNKNOWN if receipt_fault == "unknown" else ErrorCode.STORAGE_INTEGRITY_ERROR
                        )
                elif mismatch:
                    with pytest.raises(AIError) as rejected:
                        await asyncio.wait_for(delayed, 5)
                    assert rejected.value.code is ErrorCode.IDEMPOTENCY_CONFLICT
                elif outcome == "succeeded" or session:
                    replay = await asyncio.wait_for(delayed, 5)
                    assert replay.execution_id == winner.execution_id
                    assert (await replay.wait(timeout_seconds=5)).result.output == result.output
                    if hold:
                        held = await state.execution.executions.get(winner.execution_id, tenant_id=state.tenant_id)
                        assert held is not None
                        assert set(held.dependency_hold_ids) == {"original-hold", "late-hold"}
                        assert await replay.budget_usage() == budget_before
                else:
                    with pytest.raises(AIError) as rejected:
                        await asyncio.wait_for(delayed, 5)
                    assert rejected.value.code.value == identity.error_code
                if coordination is not None:
                    # Admission errors are reported before lifetime drain so
                    # joined callers cannot deadlock their owner's cleanup.
                    await asyncio.wait_for(turn_released.wait(), 5)
                    observer = PendingTurnCoordination(
                        coordination, coordination, namespace="terminal-replay", tenant_id=state.tenant_id,
                    )
                    assert await observer.read_owner(winner.execution_id, tenant_id=state.tenant_id) is None
                    assert await observer.read_request_execution(
                        f"{identity.scope}:{identity.idempotency_key_digest}", tenant_id=state.tenant_id,
                    ) is None
                if outcome == "succeeded" or session:
                    sequential = await start(delayed_runtime, prompt="same request", hold_id="late-hold")
                    assert sequential.execution_id == winner.execution_id
                else:
                    with pytest.raises(AIError) as sequential_error:
                        await start(delayed_runtime, prompt="same request", hold_id="late-hold")
                    assert sequential_error.value.code.value == identity.error_code
                retained = await state.execution.executions.get_result(winner.execution_id, tenant_id=state.tenant_id)
                assert retained is not None and retained.output == source.result.output
                assert provider_calls == 1
                if hold:
                    for hold_id in ("original-hold", "late-hold"):
                        await state.execution.executions.release_dependency_hold(
                            winner.execution_id, tenant_id=state.tenant_id, hold_id=hold_id,
                        )
            finally:
                release.set()
                if not delayed.done():
                    delayed.cancel()
                await asyncio.gather(delayed, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("backend,mode", (
    ("memory", "sync"), ("sqlite", "sync"), ("sqlite", "coordinated"), ("sqlite", "pending"),
))
@pytest.mark.parametrize("outcome", ("succeeded", "failed", "cancelled"))
async def test_admission_terminal_replay_matches_sequential_outcome(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend: str, mode: str, outcome: str,
) -> None:
    await _terminal_race(tmp_path, monkeypatch, backend=backend, mode=mode, outcome=outcome)


@pytest.mark.asyncio
async def test_admission_terminal_replay_rejects_changed_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    await _terminal_race(tmp_path, monkeypatch, backend="sqlite", mode="coordinated",
                         outcome="succeeded", mismatch=True)


@pytest.mark.asyncio
async def test_admission_terminal_replay_retains_dependency_and_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    await _terminal_race(tmp_path, monkeypatch, backend="sqlite", mode="coordinated",
                         outcome="succeeded", hold=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome,receipt_fault", (
    ("succeeded", "missing_result"), ("succeeded", "digest"), ("succeeded", "unknown"),
    ("failed", "missing_result"), ("cancelled", "missing_result"),
))
async def test_admission_terminal_replay_preserves_receipt_guards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, outcome: str, receipt_fault: str,
) -> None:
    await _terminal_race(tmp_path, monkeypatch, backend="memory", mode="sync",
                         outcome=outcome, receipt_fault=receipt_fault)


@pytest.mark.asyncio
@pytest.mark.parametrize("coordinated", (False, True))
async def test_admission_terminal_replay_does_not_restart_live_winner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, coordinated: bool,
) -> None:
    absent, release, provider_entered, finish = (asyncio.Event() for _ in range(4))
    provider_calls = 0

    async def provider(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal provider_calls
        del messages, info
        provider_calls += 1
        provider_entered.set()
        await finish.wait()
        return ModelResponse(parts=[TextPart("one result")])

    features = RuntimeFeatures("live-replay")
    features.capability(ExecutionCapability(retain_history=True), scope="runtime")
    features.agent("default", model="default", allow_tools=())
    model_features(_Models(_UsageFunctionModel(provider)), features)
    storage = PersistenceStorage.sqlite(
        tmp_path / "state.db", coordination=(
            InMemoryCacheBackend(max_bytes=4 * 1024 * 1024) if coordinated else None
        ),
    )
    async with Persistence.open("live-replay", storage=storage) as state:
        async with Runtime.attach(persistence=state, features=features) as delayed_runtime, Runtime.attach(
            persistence=state, features=features,
        ) as winner_runtime:
            original_admit = delayed_runtime._execution_service._admit_execution_start

            async def delayed_admit(*args, **kwargs):
                absent.set()
                await release.wait()
                return await original_admit(*args, **kwargs)

            monkeypatch.setattr(delayed_runtime._execution_service, "_admit_execution_start", delayed_admit)
            delayed = asyncio.create_task(
                delayed_runtime.agents.get().start("same", idempotency_key="same"), name="late-live",
            )
            try:
                await asyncio.wait_for(absent.wait(), 5)
                winner = await winner_runtime.agents.get().start("same", idempotency_key="same")
                await asyncio.wait_for(provider_entered.wait(), 5)
                release.set()
                replay = await asyncio.wait_for(delayed, 5)
                assert replay.execution_id == winner.execution_id
                assert provider_calls == 1
                assert (await winner_runtime.executions.inspect(
                    winner.execution_id, principal=winner_runtime.default_principal,
                )).status is ExecutionStatus.STARTED
                finish.set()
                result = (await winner.wait(timeout_seconds=5)).result
                assert result.status is ExecutionStatus.SUCCEEDED
                assert (await replay.wait(timeout_seconds=5)).result.output == result.output
                assert provider_calls == 1
            finally:
                release.set()
                finish.set()
                if not delayed.done():
                    delayed.cancel()
                await asyncio.gather(delayed, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome,receipt_fault", (
    ("failed", None), ("cancelled", None),
    ("failed", "not_started"), ("cancelled", "not_started"),
    ("failed", "missing_result"), ("cancelled", "digest"),
))
async def test_session_admission_terminal_replay_preserves_started_failure_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, outcome: str, receipt_fault: str | None,
) -> None:
    await _terminal_race(tmp_path, monkeypatch, backend="memory", mode="sync",
                         outcome=outcome, receipt_fault=receipt_fault, session=True)
