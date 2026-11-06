#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Execution admission delegates lifetime ownership and settles cancellation."""

import asyncio
from contextlib import asynccontextmanager

import pytest

from linktools.ai.core import ExecutionStatus, Principal
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.storage import InMemoryCacheBackend
from linktools.ai.persistence.coordination import PendingTurnCoordination

from .persistence.test_execution_start_service import _binding, _Launcher, _request, _service


@asynccontextmanager
async def _fixture():
    async with Persistence.open(
        "admission-bridge", tenant_id="tenant", storage=PersistenceStorage.in_memory(),
    ) as state:
        launcher = _Launcher(state.execution.executions)
        service = _service(state, backend=launcher)
        try:
            yield service, launcher
        finally:
            await service.preflight_close()


async def _start(service):
    return await service.start(
        _binding().binding_digest, _request("question", Principal("owner", "tenant"), "request"),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("existing", (False, True))
async def test_coordination_without_lifetime_owner_rejects_before_launch(existing) -> None:
    async with _fixture() as (service, launcher):
        if existing:
            await _start(service)
        cache = InMemoryCacheBackend(max_bytes=1024 * 1024)
        service._runtime_bridge.pending_turns.bind_coordination(PendingTurnCoordination(
            cache, cache, namespace="admission-bridge", tenant_id="tenant",
        ))
        with pytest.raises(RuntimeError, match="requires a turn lifecycle owner"):
            await _start(service)
        assert launcher.calls == (1 if existing else 0)


@pytest.mark.asyncio
async def test_uncoordinated_admission_uses_configured_lifetime_owner() -> None:
    async with _fixture() as (service, launcher):
        admitted = []

        async def run_turn(reservation, principal, admit):
            assert principal.principal_id == "owner"
            result = await admit(None)
            admitted.append(reservation.execution.execution_id)
            return result

        service._runtime_bridge.run_turn = run_turn
        result = await _start(service)
        assert admitted == [result.execution_id]
        assert launcher.calls == 1


@pytest.mark.asyncio
async def test_caller_cancellation_and_close_wait_for_lifetime_admission() -> None:
    async with _fixture() as (service, launcher):
        entered, release = asyncio.Event(), asyncio.Event()
        admitted = []

        async def run_turn(reservation, principal, admit):
            result = await admit(None)
            entered.set()
            await release.wait()
            admitted.append(result.execution_id)
            return result

        service._runtime_bridge.run_turn = run_turn
        starting = asyncio.create_task(_start(service))
        closing = None
        try:
            await asyncio.wait_for(entered.wait(), 2)
            starting.cancel()
            closing = asyncio.create_task(service.preflight_close())
            await asyncio.sleep(0)
            assert not starting.done()
            assert not closing.done()
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(starting, 2)
            await asyncio.wait_for(closing, 2)
            assert len(admitted) == 1
            assert launcher.calls == 1
        finally:
            release.set()
            await asyncio.gather(*(t for t in (starting, closing) if t is not None), return_exceptions=True)


@pytest.mark.asyncio
async def test_lifetime_admission_error_propagates_without_launch() -> None:
    async with _fixture() as (service, launcher):
        async def run_turn(reservation, principal, admit):
            raise OSError("owner entry failed")

        service._runtime_bridge.run_turn = run_turn
        with pytest.raises(OSError, match="owner entry failed"):
            await _start(service)
        assert launcher.calls == 0


@pytest.mark.asyncio
async def test_initial_and_replayed_installation_run_inside_owner_boundary(monkeypatch) -> None:
    async with _fixture() as (service, launcher):
        from contextvars import ContextVar

        owned = ContextVar("admission_owned", default=False)
        prepare = launcher.prepare_start
        launch = launcher.launch
        prepared, launched = [], []

        async def run_existing_turn(execution, parent_generation, action):
            token = owned.set(True)
            try:
                await action()
            finally:
                owned.reset(token)

        async def prepare_owned(request, execution, identity):
            assert owned.get()
            prepared.append(execution.execution_id)
            return await prepare(request, execution, identity)

        async def launch_owned(request, execution):
            assert owned.get()
            launched.append(execution.execution_id)
            await launch(request, execution)

        service._runtime_bridge.run_existing_turn = run_existing_turn
        monkeypatch.setattr(launcher, "prepare_start", prepare_owned)
        monkeypatch.setattr(launcher, "launch", launch_owned)
        first = await _start(service)
        replay = await _start(service)
        assert replay.execution_id == first.execution_id
        assert prepared == [first.execution_id]
        assert launched == [first.execution_id, first.execution_id]
        assert launcher.calls == 1


@pytest.mark.asyncio
async def test_active_replay_observes_without_entering_installation_owner() -> None:
    from dataclasses import replace

    async with _fixture() as (service, launcher):
        first = await _start(service)
        execution = await service.load_execution(first.execution_id, tenant_id="tenant")
        await service._state.executions.compare_and_swap(
            execution.execution_id, tenant_id="tenant", expected_revision=execution.revision,
            next_record=replace(execution, agent_run_seq=1, revision=execution.revision + 1),
        )

        async def forbid_owner(execution, parent_generation, action):
            raise AssertionError("active replay must not acquire an installation owner")

        service._runtime_bridge.run_existing_turn = forbid_owner
        replay = await _start(service)
        assert replay.execution_id == first.execution_id
        assert launcher.calls == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("matching_request", (False, True))
async def test_subagent_recovery_follows_successful_request_validation(matching_request) -> None:
    from linktools.ai.errors import AIError, ErrorCode
    from .test_subagent_repository_instructions import (
        _binding as child_binding, _create_parent, _request as child_request, _service as child_service,
    )

    async with Persistence.open(
        "child-replay-bridge", tenant_id="tenant", storage=PersistenceStorage.in_memory(),
    ) as state:
        await _create_parent(state, None)
        service = child_service(state, resolver=None, ids=("child", "unused"))

        class BindingRestorer:
            def restore(self, contract):
                binding = child_binding()
                assert binding.binding_contract == contract
                return binding

        service._compiler = BindingRestorer()
        request = child_request("child-key")
        recovered = []

        async def resume_subagent(execution, parent_generation):
            recovered.append((execution.execution_id, parent_generation))

        async def run_existing_turn(execution, parent_generation, action):
            assert execution.parent_execution_id == "parent"
            assert parent_generation == 1
            await action()

        service._runtime_bridge.run_existing_turn = run_existing_turn
        service._runtime_bridge.resume_subagent = resume_subagent
        child = await service.start_subagent(
            child_binding().binding_digest, request,
            parent_execution_id="parent", parent_producer_generation=1,
            root_execution_id="parent", parent_invocation_id="delegate-call",
        )
        assert recovered == []
        replay = service.replay_subagent(
            agent_id="agent", user_prompt="work" if matching_request else "changed",
            principal=request.principal, idempotency_key="child-key",
            memory_scope=request.memory_scope, mode=request.mode,
            parent_execution_id="parent", parent_producer_generation=1,
            root_execution_id="parent", parent_invocation_id="delegate-call",
        )
        if matching_request:
            handle = await replay
            assert handle.execution_id == child.execution_id
            assert recovered == [(child.execution_id, 1)]
        else:
            with pytest.raises(AIError) as raised:
                await replay
            assert raised.value.code is ErrorCode.IDEMPOTENCY_CONFLICT
            assert recovered == []
        await service.preflight_close()


@pytest.mark.asyncio
async def test_reserved_replay_prepares_and_installs_inside_owner_boundary(monkeypatch) -> None:
    async with _fixture() as (service, launcher):
        prepare = launcher.prepare_start

        async def interrupted_prepare(request, execution, identity):
            raise OSError("preparation interrupted")

        monkeypatch.setattr(launcher, "prepare_start", interrupted_prepare)
        with pytest.raises(OSError, match="preparation interrupted"):
            await _start(service)
        assert launcher.calls == 0
        entered = []

        async def run_existing_turn(execution, parent_generation, action):
            entered.append(execution.status)
            await action()

        service._runtime_bridge.run_existing_turn = run_existing_turn
        monkeypatch.setattr(launcher, "prepare_start", prepare)
        replay = await _start(service)
        assert replay.execution_id
        assert ExecutionStatus.PENDING_START in entered
        assert launcher.calls == 1
