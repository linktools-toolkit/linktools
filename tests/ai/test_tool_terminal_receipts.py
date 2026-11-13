#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Only a complete fresh turn can replace terminal effect scans with receipts."""

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
from types import SimpleNamespace

import pytest
import pytest_asyncio
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.tools import ToolDefinition
from pydantic_ai_harness.step_persistence import StepPersistence
from sqlalchemy.ext.asyncio import create_async_engine

from linktools.ai.capability import AgentContext, ExecutionCapability, ToolCallFailed
from linktools.ai.core import ExecutionStatus, ToolOperationStatus
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.execution import CancelExecutionRequest
from linktools.ai.migrate import provision_database
from linktools.ai.persistence import Persistence, PersistenceStorage, SqlStepStore
from linktools.ai.runtime import Runtime, RuntimeFeatures
from linktools.ai.runtime._local import LocalExecutionBackend
from linktools.ai.runtime._tool import RuntimeToolOperationBridge
from linktools.ai.storage import InMemoryObjectStore

from ._runtime_test_helpers import (
    _Models, _UsageFunctionModel, model_features, native_checkpoint, tool_run_context,
)


@pytest_asyncio.fixture(params=("memory", "sqlite"))
async def state(request: pytest.FixtureRequest, tmp_path: Path) -> AsyncIterator[Persistence]:
    storage = (PersistenceStorage.in_memory() if request.param == "memory"
               else PersistenceStorage.sqlite(tmp_path / "tools.db"))
    async with Persistence.open("terminal-receipts", storage=storage) as persistence:
        yield persistence


def _bridge(
    state: Persistence, *, fresh: bool = True, recovery_run: str | None = None,
    operation_ids: dict[str, str] | None = None,
) -> RuntimeToolOperationBridge:
    return RuntimeToolOperationBridge(
        state.checkpoint.tools, InMemoryObjectStore(), namespace="terminal-receipts",
        tenant_id="default", execution_id="execution", agent_run_id="run",
        binding_digest="b" * 64, owner="owner", background_tasks=set(),
         fresh_execution=fresh,
        recovery_agent_run_id=recovery_run, operation_ids=operation_ids,
    )


async def _admit(bridge: RuntimeToolOperationBridge):
    return await bridge.begin(
        tool_run_context(), ToolCallPart("work", {}, tool_call_id="call"),
        ToolDefinition(name="work"), {}, False,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("failed", (False, True))
async def test_complete_fresh_receipts_replace_only_terminal_scan(
    state: Persistence, monkeypatch: pytest.MonkeyPatch, failed: bool,
) -> None:
    bridge = _bridge(state)
    scans = []
    read = state.checkpoint.tools.get_operations

    async def source(*args, **kwargs):
        scans.append(args)
        return await read(*args, **kwargs)

    monkeypatch.setattr(state.checkpoint.tools, "get_operations", source)
    assert bridge.confirmed_terminal_operations() == ()
    assert await bridge.list_operations() == ()
    decision = await _admit(bridge)
    assert bridge.confirmed_terminal_operations() is None
    assert (await bridge.list_operations())[0].status is ToolOperationStatus.CLAIMED
    assert len(scans) == 1
    if failed:
        await bridge.fail(decision, ToolCallFailed("known failure"))
    else:
        await bridge.complete(decision, "known result")
    records = bridge.confirmed_terminal_operations()
    assert records is not None and len(records) == 1
    assert records[0].status is (ToolOperationStatus.FAILED if failed else ToolOperationStatus.COMPLETED)
    assert await bridge.list_operations() == records
    assert len(scans) == 1
    cold = _bridge(state, fresh=False, operation_ids={"call": decision.operation_id})
    assert cold.confirmed_terminal_operations() is None
    assert await cold.list_operations() == records
    assert len(scans) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ("unknown", "deferred", "lost_admission"))
async def test_missing_or_nonterminal_receipts_require_source_facts(
    state: Persistence, monkeypatch: pytest.MonkeyPatch, outcome: str,
) -> None:
    bridge = _bridge(state)
    if outcome == "lost_admission":
        admit = state.checkpoint.tools.admit

        async def lose_receipt(request):
            await admit(request)
            raise AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN)

        monkeypatch.setattr(state.checkpoint.tools, "admit", lose_receipt)
        with pytest.raises(AIError) as failure:
            await _admit(bridge)
        assert failure.value.code is ErrorCode.STORAGE_COMMIT_UNKNOWN
        expected = ToolOperationStatus.CLAIMED
    else:
        decision = await _admit(bridge)
        if outcome == "unknown":
            with pytest.raises(AIError) as failure:
                await bridge.unknown(decision, RuntimeError("external effect lost"))
            assert failure.value.code is ErrorCode.TOOL_EFFECT_UNKNOWN
            expected = ToolOperationStatus.EFFECT_UNKNOWN
        else:
            await bridge.defer(decision)
            expected = ToolOperationStatus.PENDING
    assert bridge.confirmed_terminal_operations() is None
    records = await bridge.list_operations()
    assert len(records) == 1 and records[0].status is expected
    recovery = _bridge(state, recovery_run="run", operation_ids={"call": records[0].tool_operation_id})
    assert recovery.confirmed_terminal_operations() is None
    assert await recovery.list_operations() == records


@pytest.mark.asyncio
@pytest.mark.parametrize("change", (None, "owner", "generation", "attempt", "run", "cancelling"))
async def test_terminal_receipts_are_bound_to_the_exact_execution_owner(
    state: Persistence, change: str | None,
) -> None:
    bridge = _bridge(state)
    execution = SimpleNamespace(
        execution_id="execution", producer_generation=None, producer_claim_id="owner",
        agent_run_seq=1, recovery=SimpleNamespace(agent_run_id="run"), status=ExecutionStatus.STARTED,
    )
    if change == "owner":
        execution.producer_claim_id = "successor"
    elif change == "generation":
        execution.producer_generation = 2
    elif change == "attempt":
        execution.agent_run_seq = 2
    elif change == "run":
        execution.recovery.agent_run_id = "other-run"
    elif change == "cancelling":
        execution.status = ExecutionStatus.CANCELLING
    scans = []

    async def reconcile(execution_id, *, tenant_id):
        scans.append((execution_id, tenant_id))
        return (), 0

    backend = SimpleNamespace(_tenant_id="default", _reconcile_tool_effects=reconcile)
    assert await LocalExecutionBackend._check_reconcile_tool_effects(
        backend, execution, producer_generation=None, tool_operations=bridge,
    ) is None
    assert len(scans) == int(change is not None)


@pytest.mark.asyncio
async def test_fresh_runtime_terminal_uses_receipts_without_effect_scans(
    state: Persistence, monkeypatch: pytest.MonkeyPatch,
) -> None:
    scans, effects = [], []
    for name in ("list_by_agent_run", "list_by_execution"):
        original = getattr(state.checkpoint.tools, name)

        def observe(method, label):
            async def read(*args, **kwargs):
                scans.append(label)
                return await method(*args, **kwargs)
            return read

        monkeypatch.setattr(state.checkpoint.tools, name, observe(original, name))

    async def work(ctx: AgentContext[None]) -> str:
        effects.append(ctx.execution_id)
        return "known result"

    async def model(messages, info):
        returns = [part for message in messages for part in message.parts if isinstance(part, ToolReturnPart)]
        return (ModelResponse(parts=[TextPart("done")]) if returns else
                ModelResponse(parts=[ToolCallPart("work", {}, tool_call_id="call")]))

    features = RuntimeFeatures("terminal-receipts")
    features.capability(ExecutionCapability(retain_history=True), scope="runtime")
    features.capability(native_checkpoint(), scope="runtime")
    features.tool(work, effect_policy="non_replay_safe")
    features.agent("default", model="default", allow_tools=("work",))
    async with Runtime.attach(
        persistence=state, features=model_features(_Models(_UsageFunctionModel(model)), features),
    ) as runtime:
        execution = await runtime.agents.get().start("work", idempotency_key="work")
        result = await execution.wait(timeout_seconds=10)
        assert result.result.status is ExecutionStatus.SUCCEEDED, result.result
        backend = runtime._execution_service.runtime_backend()
        worker = backend._tasks.get(execution.execution_id)
        if worker is not None:
            await asyncio.wait_for(asyncio.shield(worker), 5)
        assert effects == [execution.execution_id]
        assert scans == []


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", (False, True))
async def test_native_success_reuses_control_receipt_but_source_cancel_wins(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cancel: bool,
) -> None:
    entered, release = asyncio.Event(), asyncio.Event()
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'success.db'}")
    await provision_database(engine)
    native = SqlStepStore(engine, namespace="success-receipts", tenant_id="default")
    features = RuntimeFeatures("success-receipts")
    features.capability(StepPersistence(store=native, capture_frontier=True), scope="runtime")
    features.agent("default", model="default", allow_tools=())
    calls, control_reads, committed_receipts = [], [], []
    success_task = None

    async def provider(messages, info):
        calls.append("model")
        return ModelResponse(parts=[TextPart("done")])

    success = LocalExecutionBackend._commit_success
    load = LocalExecutionBackend.load_execution

    async def pause_success(backend, *args, **kwargs):
        nonlocal success_task
        owner = backend._turn_owners[args[0].execution_id]
        assert owner.tool_admission is not None and owner.observed is not None
        committed_receipts.append(owner.observed)
        success_task = asyncio.current_task()
        entered.set()
        await release.wait()
        return await success(backend, *args, **kwargs)

    async def observe_load(backend, *args, **kwargs):
        if asyncio.current_task() is success_task:
            control_reads.append(args[0])
        return await load(backend, *args, **kwargs)

    monkeypatch.setattr(LocalExecutionBackend, "_commit_success", pause_success)
    monkeypatch.setattr(LocalExecutionBackend, "load_execution", observe_load)
    binding = model_features(_Models(_UsageFunctionModel(provider)), features)
    try:
        async with Persistence.open("success-receipts", storage=PersistenceStorage.sql(engine)) as first:
            async with Runtime.attach(persistence=first, features=binding) as runtime:
                async with Persistence.open("success-receipts", storage=PersistenceStorage.sql(engine)) as second:
                    async with Runtime.attach(persistence=second, features=binding) as remote:
                        execution = await runtime.agents.get().start("work", idempotency_key="work")
                        await asyncio.wait_for(entered.wait(), 5)
                        try:
                            if cancel:
                                result = await remote.executions.cancel(
                                    execution.execution_id,
                                    CancelExecutionRequest(runtime.default_principal, "stop"),
                                )
                                assert result.cancelled is False
                                source = await second.execution.executions.get(
                                    execution.execution_id, tenant_id="default",
                                )
                                assert source.status is ExecutionStatus.CANCELLING
                                assert committed_receipts[0].value.status is ExecutionStatus.STARTED
                        finally:
                            release.set()
                        result = await execution.wait(timeout_seconds=10)
                        assert result.result.status is (
                            ExecutionStatus.CANCELLED if cancel else ExecutionStatus.SUCCEEDED
                        ), result.result
                        backend = runtime._execution_service.runtime_backend()
                        worker = backend._tasks.get(execution.execution_id)
                        if worker is not None:
                            await asyncio.wait_for(asyncio.shield(worker), 5)
                        assert calls == ["model"]
                        if not cancel:
                            assert control_reads == []
    finally:
        release.set()
        await native.close()
        await engine.dispose()
