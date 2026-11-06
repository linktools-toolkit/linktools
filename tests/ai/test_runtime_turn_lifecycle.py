#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Turn ownership follows workers and their durable work, not caller handles."""
import asyncio
from pathlib import Path
from contextlib import nullcontext

import pytest
from pydantic_ai.messages import ModelResponse, TextPart

from linktools.ai.capability import ExecutionCapability
from pydantic_ai_harness.step_persistence import StepPersistence
from tests.ai._runtime_test_helpers import native_checkpoint
from linktools.ai.core import CommitPolicy, ExecutionStatus
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.runtime import Runtime, RuntimeFeatures
from linktools.ai.storage import InMemoryCacheBackend, Lease, TurnCoordination, TurnScope, RenewingTurnScope
from tests.ai._runtime_test_helpers import _Models, _UsageFunctionModel, model_features


class _ObservedCache(InMemoryCacheBackend):
    def __init__(self) -> None:
        super().__init__(max_bytes=4 * 1024 * 1024)
        self.releases = 0
        self.released = asyncio.Event()
        self.renewed = asyncio.Event()

    def enter(self, scope: str, *, session_key: str, request_key: str, request_digest: str,
              execution_id: str, owner_id: str) -> TurnScope:
        return RenewingTurnScope(self, scope, session_key=session_key, request_key=request_key,
                                 request_digest=request_digest, execution_id=execution_id,
                                 owner_id=owner_id, lease_seconds=30, renew_seconds=.05)

    async def release_turn(self, lease: Lease) -> bool:
        self.releases += 1
        result = await super().release_turn(lease)
        self.released.set()
        return result

    async def renew_turn(self, lease: Lease, *, lease_seconds: float) -> TurnCoordination | None:
        result = await super().renew_turn(lease, lease_seconds=lease_seconds)
        self.renewed.set()
        return result


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ("sync", "checkpoint", "pending", "promotion"))
async def test_turn_owner_survives_handle_return_timeout_and_promotion(tmp_path: Path, mode: str) -> None:
    entered, finish = asyncio.Event(), asyncio.Event()

    async def model(messages, info):
        entered.set()
        await finish.wait()
        return ModelResponse(parts=[TextPart("answer")])

    features = RuntimeFeatures("lifecycle-probe")
    features.capability(ExecutionCapability(retain_history=True,
        commit=CommitPolicy.sync() if mode in {"sync", "checkpoint"} else CommitPolicy.terminal()), scope="runtime")
    if mode == "checkpoint":
        features.capability(native_checkpoint(), scope="runtime")
    features.agent("default", model="default", allow_tools=())
    cache = _ObservedCache()
    storage = PersistenceStorage.sqlite(tmp_path / "state.db", cache=cache, coordination=cache)
    async with Persistence.open("lifecycle-probe", storage=storage) as persistence, Runtime.attach(
        persistence=persistence, features=model_features(_Models(_UsageFunctionModel(model)), features),
    ) as runtime:
        session = await runtime.agents.get().create_session("session")
        execution = await session.start("question", idempotency_key="request")
        await asyncio.wait_for(entered.wait(), 5)
        backend = runtime._execution_service.runtime_backend()
        coordinator = backend._pending_turns.coordination
        async def owner():
            return await coordinator.read_owner(execution.execution_id, tenant_id=runtime.tenant_id) is not None
        assert await owner()
        assert not cache.released.is_set()
        try:
            if mode == "promotion":
                await backend.promote_pending_execution(execution.execution_id)
                assert await owner()
                assert not cache.released.is_set()
            with pytest.raises(TimeoutError):
                await asyncio.wait_for(execution.wait(), .02)
            assert await owner()
            if mode == "promotion":
                await asyncio.wait_for(cache.renewed.wait(), 5)
                assert await owner()
        finally:
            finish.set()
        result = await execution.wait(timeout_seconds=10)
        assert result.result.status is ExecutionStatus.SUCCEEDED
        await asyncio.wait_for(cache.released.wait(), 5)
        assert not await owner()


@pytest.mark.asyncio
@pytest.mark.parametrize("terminal", (False, True))
async def test_parent_and_parallel_children_keep_independent_turn_owners(tmp_path: Path, terminal: bool) -> None:
    from pydantic_ai.messages import ToolCallPart, ToolReturnPart
    entered = 0
    both_entered, finish = asyncio.Event(), asyncio.Event()
    parent_requests = 0
    child_requests = 0
    child_tools = []

    async def parent(messages, info):
        nonlocal parent_requests
        parent_requests += 1
        if any(isinstance(part, ToolReturnPart) for message in messages for part in message.parts):
            return ModelResponse(parts=[TextPart("complete")])
        return ModelResponse(parts=[ToolCallPart("delegate_task", {"subagent_id": "child", "task": f"work-{number}"}, tool_call_id=f"child-{number}") for number in (1, 2)])

    async def child(messages, info):
        nonlocal entered, child_requests
        child_requests += 1
        child_tools.extend(tool.name for tool in info.function_tools)
        entered += 1
        if entered == 2:
            both_entered.set()
        await finish.wait()
        return ModelResponse(parts=[TextPart("child complete")])

    features = RuntimeFeatures("subagent-owner-probe")
    features.model("parent", _Models(_UsageFunctionModel(parent)))
    features.model("child", _Models(_UsageFunctionModel(child)))
    features.capability(ExecutionCapability(retain_history=True, commit=CommitPolicy.terminal() if terminal else CommitPolicy.sync()), scope="runtime")
    features.agent("default", model="parent", allow_tools=(), allow_subagents=("child",))
    features.agent("child", model="child", allow_tools=(), allow_subagents=("default",))
    cache = _ObservedCache()
    storage = PersistenceStorage.sqlite(tmp_path / "state.db", cache=cache, coordination=cache)
    async with Persistence.open("subagent-owner-probe", storage=storage) as persistence, Runtime.attach(persistence=persistence, features=features) as runtime:
        session = await runtime.agents.get().create_session("session")
        execution = await session.start("delegate twice", idempotency_key="parent-request")
        try:
            await asyncio.wait_for(both_entered.wait(), 5)
            children = await runtime.executions.list_children(execution.execution_id, principal=runtime.default_principal)
            assert len(children) == 2 and len({value.execution_id for value in children}) == 2
            backend = runtime._execution_service.runtime_backend()
            coordinator = backend._pending_turns.coordination
            owner_ids = set()
            for value in children:
                record = await runtime._state.execution.executions.get(value.execution_id, tenant_id=runtime.tenant_id)
                assert record.session_id is None and record.parent_execution_id == execution.execution_id
                assert record.root_execution_id == execution.execution_id
                assert backend.worker_installed(record.execution_id)
                owner = await coordinator.read_owner(value.execution_id, tenant_id=runtime.tenant_id)
                assert owner is not None
                owner_ids.add(owner.owner_id)
            parent_owner = await coordinator.read_owner(execution.execution_id, tenant_id=runtime.tenant_id)
            assert parent_owner is not None
            owner_ids.add(parent_owner.owner_id)
            assert len(owner_ids) == 3
            assert "delegate_task" not in child_tools
        finally:
            finish.set()
        assert (await execution.wait(timeout_seconds=10)).result.status is ExecutionStatus.SUCCEEDED
        assert child_requests == 2 and parent_requests == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("background_kind", ("durable", "reconciliation"))
async def test_turn_owner_outlives_worker_until_owned_work_settles(tmp_path: Path, background_kind: str) -> None:
    entered, finish, settled = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def model(messages, info):
        entered.set()
        await finish.wait()
        return ModelResponse(parts=[TextPart("answer")])

    features = RuntimeFeatures("owned-work")
    features.capability(ExecutionCapability(retain_history=True), scope="runtime")
    features.agent("default", model="default", allow_tools=())
    cache = _ObservedCache()
    storage = PersistenceStorage.sqlite(tmp_path / "state.db", cache=cache, coordination=cache)
    async with Persistence.open("owned-work", storage=storage) as persistence, Runtime.attach(
        persistence=persistence, features=model_features(_Models(_UsageFunctionModel(model)), features),
    ) as runtime:
        session = await runtime.agents.get().create_session("session")
        execution = await session.start("question", idempotency_key="request")
        await asyncio.wait_for(entered.wait(), 5)
        backend = runtime._execution_service.runtime_backend()
        coordinator = backend._pending_turns.coordination
        worker = backend._tasks[execution.execution_id]
        background = asyncio.create_task(settled.wait())
        if background_kind == "durable":
            backend._execution_task_set(execution.execution_id).add(background)
        else:
            backend._recovery_reconcile_tasks[execution.execution_id] = background
        try:
            finish.set()
            await asyncio.wait_for(asyncio.shield(worker), 5)
            assert not cache.released.is_set()
            assert await coordinator.read_owner(execution.execution_id, tenant_id=runtime.tenant_id) is not None
        finally:
            settled.set()
            await background
        await asyncio.wait_for(cache.released.wait(), 5)
        assert await coordinator.read_owner(execution.execution_id, tenant_id=runtime.tenant_id) is None
        assert (await execution.wait(timeout_seconds=5)).result.status is ExecutionStatus.SUCCEEDED


@pytest.mark.asyncio
async def test_lease_renewal_failure_stops_worker_and_releases_ownership(tmp_path: Path) -> None:
    from linktools.ai.errors import AIError, ErrorCode

    entered, stopped = asyncio.Event(), asyncio.Event()

    async def model(messages, info):
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            stopped.set()

    class LostLeaseCache(_ObservedCache):
        lost = False

        async def renew_turn(self, lease: Lease, *, lease_seconds: float) -> TurnCoordination | None:
            if self.lost:
                return None
            return await super().renew_turn(lease, lease_seconds=lease_seconds)

    features = RuntimeFeatures("lost-owner")
    features.capability(ExecutionCapability(retain_history=True), scope="runtime")
    features.agent("default", model="default", allow_tools=())
    cache = LostLeaseCache()
    storage = PersistenceStorage.sqlite(tmp_path / "state.db", cache=cache, coordination=cache)
    async with Persistence.open("lost-owner", storage=storage) as persistence, Runtime.attach(
        persistence=persistence, features=model_features(_Models(_UsageFunctionModel(model)), features),
    ) as runtime:
        session = await runtime.agents.get().create_session("session")
        execution = await session.start("question", idempotency_key="request")
        await asyncio.wait_for(entered.wait(), 5)
        cache.lost = True
        await asyncio.wait_for(stopped.wait(), 5)
        await asyncio.wait_for(cache.released.wait(), 5)
        backend = runtime._execution_service.runtime_backend()
        assert not backend.worker_installed(execution.execution_id)
        coordinator = backend._pending_turns.coordination
        assert await coordinator.read_owner(execution.execution_id, tenant_id=runtime.tenant_id) is None
        record = await persistence.execution.executions.get(execution.execution_id, tenant_id=persistence.tenant_id)
        assert record.status is ExecutionStatus.FAILED
        assert record.error_code == ErrorCode.STORAGE_CONFLICT.value
        assert record.safe_error_details["phase"] == "turn_owner_stopped"
        with pytest.raises(AIError) as local_wait:
            await execution.wait(timeout_seconds=2)
        assert local_wait.value.safe_details["phase"] == "turn_owner_lost"
        async with Runtime.attach(persistence=persistence, features=model_features(
            _Models(_UsageFunctionModel(model)), features,
        )) as observer:
            observed = await observer.executions.inspect(execution.execution_id, principal=observer.default_principal)
            assert observed.status is ExecutionStatus.FAILED
            result = await observer.executions.wait(execution.execution_id, principal=observer.default_principal,
                                                   timeout_seconds=2)
            assert result.result.status is ExecutionStatus.FAILED
            assert result.result.error_code == ErrorCode.STORAGE_CONFLICT.value

@pytest.mark.asyncio
@pytest.mark.parametrize("coordinated", (False, True))
async def test_runtime_handoff_stops_child_wait_without_cancelling_logical_child(
    tmp_path: Path, coordinated: bool,
) -> None:
    from pydantic_ai.messages import ToolCallPart

    entered, stopped = asyncio.Event(), asyncio.Event()

    async def parent(messages, info):
        return ModelResponse(parts=[ToolCallPart(
            "delegate_task", {"subagent_id": "child", "task": "work"}, tool_call_id="child-call",
        )])

    async def child(messages, info):
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            stopped.set()

    features = RuntimeFeatures("subagent-handoff")
    features.model("parent", _Models(_UsageFunctionModel(parent)))
    features.model("child", _Models(_UsageFunctionModel(child)))
    features.capability(ExecutionCapability(retain_history=True), scope="runtime")
    features.capability(native_checkpoint(), scope="runtime")
    features.agent("default", model="parent", allow_tools=(), allow_subagents=("child",))
    features.agent("child", model="child", allow_tools=())
    cache = _ObservedCache() if coordinated else None
    storage = PersistenceStorage.sqlite(tmp_path / "state.db", cache=cache, coordination=cache)
    async with Persistence.open("subagent-handoff", storage=storage) as persistence:
        runtime = Runtime.attach(persistence=persistence, features=features)
        await runtime.__aenter__()
        backend = runtime._execution_service.runtime_backend()
        try:
            session = await runtime.agents.get().create_session("session")
            execution = await session.start("delegate", idempotency_key="parent-request")
            await asyncio.wait_for(entered.wait(), 5)
            children = await runtime.executions.list_children(
                execution.execution_id, principal=runtime.default_principal,
            )
            assert len(children) == 1
            async with asyncio.timeout(5):
                while not backend._handoff_scopes[execution.execution_id]._child_waits:
                    await asyncio.sleep(0)
            await asyncio.wait_for(runtime.handoff(), 5)
            assert stopped.is_set()
            for execution_id in (execution.execution_id, children[0].execution_id):
                record = await persistence.execution.executions.get(execution_id, tenant_id=persistence.tenant_id)
                assert record.status is ExecutionStatus.STARTED
                assert not backend.worker_installed(execution_id)
                if coordinated:
                    assert await backend._pending_turns.coordination.read_owner(
                        execution_id, tenant_id=persistence.tenant_id,
                    ) is None
        finally:
            # A failed assertion must not strand the parent's observer during close.
            for scope in tuple(backend._handoff_scopes.values()):
                scope.stop()
            await asyncio.wait_for(runtime.close(), 5)


@pytest.mark.asyncio
@pytest.mark.parametrize("checkpoint", (False, True))
async def test_parent_lease_loss_stops_only_old_workers_without_logical_child_cancel(
    tmp_path: Path, checkpoint: bool,
) -> None:
    from pydantic_ai.messages import ToolCallPart
    from linktools.ai.errors import AIError

    entered, child_stopped = asyncio.Event(), asyncio.Event()

    async def parent(messages, info):
        return ModelResponse(parts=[ToolCallPart(
            "delegate_task", {"subagent_id": "child", "task": "work"}, tool_call_id="child-call",
        )])

    async def child(messages, info):
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            child_stopped.set()

    class LostParent(_ObservedCache):
        parent_lease: Lease | None = None

        async def renew_turn(self, lease: Lease, *, lease_seconds: float) -> TurnCoordination | None:
            if lease == self.parent_lease:
                return None
            return await super().renew_turn(lease, lease_seconds=lease_seconds)

    features = RuntimeFeatures("parent-owner-loss")
    features.model("parent", _Models(_UsageFunctionModel(parent)))
    features.model("child", _Models(_UsageFunctionModel(child)))
    features.capability(ExecutionCapability(retain_history=True), scope="runtime")
    if checkpoint:
        features.capability(native_checkpoint(), scope="runtime")
    features.agent("default", model="parent", allow_tools=(), allow_subagents=("child",))
    features.agent("child", model="child", allow_tools=())
    cache = LostParent()
    storage = PersistenceStorage.sqlite(tmp_path / "state.db", cache=cache, coordination=cache)
    with pytest.raises(AIError) if checkpoint else nullcontext():
        async with Persistence.open("parent-owner-loss", storage=storage) as persistence, Runtime.attach(
            persistence=persistence, features=features,
        ) as runtime:
            session = await runtime.agents.get().create_session("session")
            execution = await session.start("delegate", idempotency_key="parent-request")
            await asyncio.wait_for(entered.wait(), 5)
            backend = runtime._execution_service.runtime_backend()
            owner = backend._turn_owners[execution.execution_id]
            children = await runtime.executions.list_children(
                execution.execution_id, principal=runtime.default_principal,
            )
            assert len(children) == 1
            cache.parent_lease = owner.coordination.lease
            await asyncio.wait_for(owner.closed.wait(), 5)
            assert child_stopped.is_set()
            for execution_id in (execution.execution_id, children[0].execution_id):
                record = await persistence.execution.executions.get(execution_id, tenant_id=persistence.tenant_id)
                assert record.status is (ExecutionStatus.STARTED if checkpoint else ExecutionStatus.FAILED)
                assert not backend.worker_installed(execution_id)
                assert execution_id not in backend._turn_owners


@pytest.mark.asyncio
async def test_child_owner_loss_does_not_force_cancel_logical_child(tmp_path: Path) -> None:
    from pydantic_ai.messages import ToolCallPart
    from linktools.ai.errors import AIError

    entered = asyncio.Event()

    async def parent(messages, info):
        return ModelResponse(parts=[ToolCallPart(
            "delegate_task", {"subagent_id": "child", "task": "work"}, tool_call_id="child-call",
        )])

    async def child(messages, info):
        entered.set()
        await asyncio.Event().wait()

    class LostChild(_ObservedCache):
        lost: Lease | None = None

        async def renew_turn(self, lease: Lease, *, lease_seconds: float) -> TurnCoordination | None:
            if lease == self.lost:
                return None
            return await super().renew_turn(lease, lease_seconds=lease_seconds)

    features = RuntimeFeatures("child-owner-loss")
    features.model("parent", _Models(_UsageFunctionModel(parent)))
    features.model("child", _Models(_UsageFunctionModel(child)))
    features.capability(ExecutionCapability(retain_history=True), scope="runtime")
    features.capability(native_checkpoint(), scope="runtime")
    features.agent("default", model="parent", allow_tools=(), allow_subagents=("child",))
    features.agent("child", model="child", allow_tools=())
    cache = LostChild()
    storage = PersistenceStorage.sqlite(tmp_path / "state.db", cache=cache, coordination=cache)
    with pytest.raises(AIError):
        async with Persistence.open("child-owner-loss", storage=storage) as persistence, Runtime.attach(
            persistence=persistence, features=features,
        ) as runtime:
            session = await runtime.agents.get().create_session("session")
            execution = await session.start("delegate", idempotency_key="parent-request")
            await asyncio.wait_for(entered.wait(), 5)
            backend = runtime._execution_service.runtime_backend()
            parent_owner = backend._turn_owners[execution.execution_id]
            children = await runtime.executions.list_children(execution.execution_id,
                                                             principal=runtime.default_principal)
            child_id = children[0].execution_id
            child_owner = backend._turn_owners[child_id]
            cache.lost = child_owner.coordination.lease
            await asyncio.wait_for(child_owner.closed.wait(), 5)
            await asyncio.wait_for(parent_owner.closed.wait(), 5)
            record = await persistence.execution.executions.get(child_id, tenant_id=persistence.tenant_id)
            assert record.status is ExecutionStatus.STARTED
            assert record.error_code is None
