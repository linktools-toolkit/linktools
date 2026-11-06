#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Cross-Runtime cancellation reaches an entered source-backed turn."""

import asyncio
from pathlib import Path

import pytest
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart

from linktools.ai.capability import AgentContext, ExecutionCapability
from linktools.ai.core import CommitPolicy, ExecutionStatus, ToolOperationStatus
from linktools.ai.execution import CancelExecutionRequest
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.runtime import Runtime, RuntimeFeatures
from linktools.ai.storage import InMemoryCacheBackend, Lease, RenewingTurnScope, TurnCoordination, TurnScope

from ._runtime_test_helpers import _Models, _UsageFunctionModel, model_features, native_checkpoint


class _DelayedRenewal(InMemoryCacheBackend):
    def __init__(self, *, lease_seconds: float = 30) -> None:
        super().__init__(max_bytes=1024 * 1024)
        self.deliver = asyncio.Event()
        self.renewing = asyncio.Event()
        self.released = asyncio.Event()
        self.lease_seconds = lease_seconds
        self.cancelled_executions: list[str] = []

    def enter(
        self, scope: str, *, session_key: str, request_key: str, request_digest: str,
        execution_id: str, owner_id: str,
    ) -> TurnScope:
        return RenewingTurnScope(
            self, scope, session_key=session_key, request_key=request_key,
            request_digest=request_digest, execution_id=execution_id, owner_id=owner_id,
            lease_seconds=self.lease_seconds, renew_seconds=.01,
        )

    async def renew_turn(self, lease: Lease, *, lease_seconds: float) -> TurnCoordination | None:
        self.renewing.set()
        await self.deliver.wait()
        return await super().renew_turn(lease, lease_seconds=lease_seconds)

    async def release_turn(self, lease: Lease) -> bool:
        result = await super().release_turn(lease)
        self.released.set()
        return result

    async def cancel_turn(self, scope: str, execution_id: str) -> bool:
        self.cancelled_executions.append(execution_id)
        return await super().cancel_turn(scope, execution_id)


@pytest.mark.asyncio
@pytest.mark.parametrize("force", (False, True))
async def test_source_cancel_reaches_owner_after_remote_terminal_settlement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, force: bool,
) -> None:
    entered, finish, stopped = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def model(messages, info):
        entered.set()
        try:
            await finish.wait()
            return ModelResponse(parts=[TextPart("late answer")])
        finally:
            stopped.set()

    features = RuntimeFeatures("source-cancel-signal")
    features.capability(ExecutionCapability(retain_history=True, commit=CommitPolicy.sync()), scope="runtime")
    features.agent("default", model="default", allow_tools=())
    features = model_features(_Models(_UsageFunctionModel(model)), features)
    coordination = _DelayedRenewal()
    database = tmp_path / "state.db"
    async with Persistence.open("source-cancel-signal", storage=PersistenceStorage.sqlite(
        database, coordination=coordination,
    )) as owner_state, Runtime.attach(persistence=owner_state, features=features) as owner, Persistence.open(
        "source-cancel-signal", storage=PersistenceStorage.sqlite(database, coordination=coordination),
    ) as remote_state, Runtime.attach(persistence=remote_state, features=features) as remote:
        execution = await owner.agents.get().start("wait", idempotency_key="request")
        await asyncio.wait_for(entered.wait(), 5)
        await asyncio.wait_for(coordination.renewing.wait(), 5)
        statuses = []
        cancel = coordination.cancel_turn

        async def observed_cancel(scope: str, execution_id: str) -> bool:
            repository = remote_state.execution.executions
            record = await repository.state_store.read(lambda _tx: repository.get(
                execution_id, tenant_id=remote.tenant_id,
            ))
            statuses.append(record.status)
            return await cancel(scope, execution_id)

        monkeypatch.setattr(coordination, "cancel_turn", observed_cancel)
        try:
            result = await remote.executions.cancel(execution.execution_id, CancelExecutionRequest(
                remote.default_principal, "remote-cancel", force=force,
            ))
            assert result.cancelled is force
            assert statuses and set(statuses) == {ExecutionStatus.CANCELLING}
            assert set(coordination.cancelled_executions) == {execution.execution_id}
            assert not stopped.is_set()
            coordination.deliver.set()
            await asyncio.wait_for(stopped.wait(), 5)
            assert (await execution.wait(timeout_seconds=5)).result.status is ExecutionStatus.CANCELLED
            await asyncio.wait_for(coordination.released.wait(), 5)
        finally:
            coordination.deliver.set()
            finish.set()


@pytest.mark.asyncio
async def test_expired_scope_stops_worker_without_business_cancel(tmp_path: Path) -> None:
    entered, finish, stopped = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def model(messages, info):
        entered.set()
        try:
            await finish.wait()
            return ModelResponse(parts=[TextPart("late answer")])
        finally:
            stopped.set()

    features = RuntimeFeatures("expired-turn-signal")
    features.capability(ExecutionCapability(retain_history=True), scope="runtime")
    features.agent("default", model="default", allow_tools=())
    features = model_features(_Models(_UsageFunctionModel(model)), features)
    coordination = _DelayedRenewal(lease_seconds=.5)
    database = tmp_path / "state.db"
    async with Persistence.open("expired-turn-signal", storage=PersistenceStorage.sqlite(
        database, coordination=coordination,
    )) as owner_state, Runtime.attach(persistence=owner_state, features=features) as owner, Persistence.open(
        "expired-turn-signal", storage=PersistenceStorage.sqlite(database, coordination=coordination),
    ) as remote_state, Runtime.attach(persistence=remote_state, features=features) as remote:
        execution = await owner.agents.get().start("wait", idempotency_key="request")
        try:
            await asyncio.wait_for(entered.wait(), 5)
            await asyncio.wait_for(coordination.renewing.wait(), 5)
            await asyncio.wait_for(stopped.wait(), 5)
            await asyncio.wait_for(coordination.released.wait(), 5)
            observed = await remote.executions.inspect(execution.execution_id, principal=remote.default_principal)
            assert observed.status is ExecutionStatus.FAILED
            assert coordination.cancelled_executions == []
        finally:
            coordination.deliver.set()
            finish.set()


@pytest.mark.asyncio
async def test_remote_force_signal_preserves_unsettled_tool_effect(tmp_path: Path) -> None:
    entered, finish, stopped = asyncio.Event(), asyncio.Event(), asyncio.Event()
    requests = []

    async def change(_ctx: AgentContext[None]) -> str:
        entered.set()
        try:
            await finish.wait()
            return "effect completed"
        finally:
            stopped.set()

    async def model(messages, info):
        requests.append(1)
        return ModelResponse(parts=[ToolCallPart("change", {}, tool_call_id="effect")])

    features = RuntimeFeatures("tool-cancel-signal")
    features.capability(native_checkpoint(), scope="runtime")
    features.tool(change, effect_policy="non_replay_safe")
    features.agent("default", model="default", allow_tools=("change",))
    features = model_features(_Models(_UsageFunctionModel(model)), features)
    coordination = _DelayedRenewal()
    database = tmp_path / "state.db"
    async with Persistence.open("tool-cancel-signal", storage=PersistenceStorage.sqlite(
        database, coordination=coordination,
    )) as owner_state, Runtime.attach(persistence=owner_state, features=features) as owner, Persistence.open(
        "tool-cancel-signal", storage=PersistenceStorage.sqlite(database, coordination=coordination),
    ) as remote_state, Runtime.attach(persistence=remote_state, features=features) as remote:
        execution = await owner.agents.get().start("change", idempotency_key="request")
        try:
            await asyncio.wait_for(entered.wait(), 5)
            result = await remote.executions.cancel(execution.execution_id, CancelExecutionRequest(
                remote.default_principal, "remote-force", force=True,
            ))
            assert not result.cancelled
            assert not stopped.is_set()
            coordination.deliver.set()
            await asyncio.wait_for(stopped.wait(), 5)
            await asyncio.wait_for(coordination.released.wait(), 5)
            observed = await remote.executions.inspect(execution.execution_id, principal=remote.default_principal)
            assert observed.status is ExecutionStatus.RECOVERY_REQUIRED
            operations = await owner_state.checkpoint.tools.list_by_execution(
                execution.execution_id, tenant_id=owner.tenant_id,
            )
            assert len(operations) == 1 and operations[0].status is ToolOperationStatus.EFFECT_UNKNOWN
            assert requests == [1]
        finally:
            coordination.deliver.set()
            finish.set()
