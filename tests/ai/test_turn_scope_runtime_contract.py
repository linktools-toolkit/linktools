#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime honors backend-owned context lifetime and admission authority."""

import asyncio
from contextvars import ContextVar
from pathlib import Path
from types import TracebackType
from typing import Literal

import pytest
from pydantic_ai.messages import ModelResponse, TextPart

from linktools.ai.capability import ExecutionCapability
from linktools.ai.core import CommitPolicy, ExecutionStatus
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.runtime import Runtime, RuntimeFeatures
from linktools.ai.storage import (
    CacheBackend, InMemoryCacheBackend, Lease, RenewingTurnScope,
    TurnCoordination, TurnCoordinationBackend, TurnScope,
)
from ._runtime_test_helpers import _Models, _UsageFunctionModel, model_features


_context: ContextVar[str | None] = ContextVar("turn_scope_runtime_contract", default=None)


class _ContextScope:
    def __init__(self, inner: TurnScope, owner_id: str) -> None:
        self.inner = inner
        self.owner_id = owner_id
        self.enter_task = None
        self.exit_task = None
        self.token = None
        self.group = asyncio.TaskGroup()
        self.group_finish = asyncio.Event()
        self.group_settled = asyncio.Event()
        self.entered = asyncio.Event()
        self.exited = asyncio.Event()
        self.lost = asyncio.Event()

    async def __aenter__(self) -> TurnCoordination | None:
        self.enter_task = asyncio.current_task()
        result = await self.inner.__aenter__()
        self.token = _context.set(self.owner_id)
        await self.group.__aenter__()

        async def child() -> None:
            await self.group_finish.wait()
            assert _context.get() == self.owner_id
            self.group_settled.set()

        self.group.create_task(child())
        self.entered.set()
        return result

    def require_active(self) -> None:
        self.inner.require_active()
        if self.lost.is_set():
            raise AIError(ErrorCode.STORAGE_CONFLICT, "test owner lost")

    async def wait_signal(self) -> Literal["owner_lost", "cancel_requested"]:
        source = asyncio.create_task(self.inner.wait_signal())
        forced = asyncio.create_task(self.lost.wait())
        try:
            await asyncio.wait((source, forced), return_when=asyncio.FIRST_COMPLETED)
            return "owner_lost" if self.lost.is_set() else source.result()
        finally:
            for task in (source, forced):
                task.cancel()
            await asyncio.gather(source, forced, return_exceptions=True)

    async def __aexit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.exit_task = asyncio.current_task()
        assert self.exit_task is self.enter_task
        assert _context.get() == self.owner_id
        self.group_finish.set()
        await self.group.__aexit__(None, None, None)
        assert self.group_settled.is_set()
        _context.reset(self.token)
        await self.inner.__aexit__(exc_type, exc, traceback)
        self.exited.set()


class _TurnAuthority:
    """Full turn authority with deliberately no CacheBackend operations."""

    def __init__(self) -> None:
        self.backend = InMemoryCacheBackend(max_bytes=4 * 1024 * 1024)
        self.scopes: list[_ContextScope] = []
        self.claimed: list[TurnCoordination] = []

    def enter(
        self, scope: str, *, session_key: str, request_key: str, request_digest: str,
        execution_id: str, owner_id: str,
    ) -> TurnScope:
        result = _ContextScope(RenewingTurnScope(
            self, scope, session_key=session_key, request_key=request_key,
            request_digest=request_digest, execution_id=execution_id, owner_id=owner_id,
        ), owner_id)
        self.scopes.append(result)
        return result

    async def claim_turn(
        self, scope: str, *, session_key: str, request_key: str, request_digest: str,
        execution_id: str, owner_id: str, lease_seconds: float,
    ) -> TurnCoordination | None:
        result = await self.backend.claim_turn(
            scope, session_key=session_key, request_key=request_key, request_digest=request_digest,
            execution_id=execution_id, owner_id=owner_id, lease_seconds=lease_seconds,
        )
        if result is not None and result.lease is not None:
            self.claimed.append(result)
        return result

    async def read_turn(self, scope: str, execution_id: str) -> TurnCoordination | None:
        return await self.backend.read_turn(scope, execution_id)

    async def renew_turn(self, lease: Lease, *, lease_seconds: float) -> TurnCoordination | None:
        return await self.backend.renew_turn(lease, lease_seconds=lease_seconds)

    async def release_turn(self, lease: Lease) -> bool:
        return await self.backend.release_turn(lease)

    async def cancel_turn(self, scope: str, execution_id: str) -> bool:
        return await self.backend.cancel_turn(scope, execution_id)


class _ProjectionAuthority(_TurnAuthority):
    async def get(self, key: str) -> bytes | None:
        return await self.backend.get(key)

    async def put(self, key: str, content: bytes) -> None:
        await self.backend.put(key, content)

    async def delete(self, key: str) -> None:
        await self.backend.delete(key)


@pytest.mark.asyncio
@pytest.mark.parametrize("caller_end", ("timeout", "cancel"))
@pytest.mark.parametrize("terminal", (False, True))
async def test_authority_only_context_lives_in_one_task_beyond_handle_and_waiter(
    tmp_path: Path, caller_end: str, terminal: bool,
) -> None:
    started, finish = asyncio.Event(), asyncio.Event()
    observed_context = []

    async def model(messages, info):
        observed_context.append(_context.get())
        started.set()
        await finish.wait()
        return ModelResponse(parts=[TextPart("answer")])

    authority = _TurnAuthority()
    assert isinstance(authority, TurnCoordinationBackend)
    assert not isinstance(authority, CacheBackend)
    features = RuntimeFeatures("scope-contract")
    features.capability(ExecutionCapability(
        retain_history=True, commit=CommitPolicy.terminal() if terminal else CommitPolicy.sync(),
    ), scope="runtime")
    features.agent("default", model="default", allow_tools=())
    storage = PersistenceStorage.sqlite(tmp_path / "state.db", coordination=authority)
    async with Persistence.open("scope-contract", storage=storage) as persistence, Runtime.attach(
        persistence=persistence, features=model_features(_Models(_UsageFunctionModel(model)), features),
    ) as runtime:
        session = await runtime.agents.get().create_session("session")
        execution = await session.start("question", idempotency_key="request")
        await asyncio.wait_for(started.wait(), 5)
        assert len(authority.scopes) == len(authority.claimed) == 1
        scope = authority.scopes[0]
        assert scope.entered.is_set() and not scope.exited.is_set()
        assert scope.enter_task is not asyncio.current_task()
        assert _context.get() is None
        assert observed_context == [scope.owner_id]
        try:
            if caller_end == "timeout":
                with pytest.raises(TimeoutError):
                    await asyncio.wait_for(execution.wait(), .02)
            else:
                waiter = asyncio.create_task(execution.wait())
                await asyncio.sleep(0)
                waiter.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await waiter
            scope.require_active()
            assert not scope.exited.is_set()
            assert not scope.group_settled.is_set()
        finally:
            finish.set()
        assert (await execution.wait(timeout_seconds=5)).result.status is ExecutionStatus.SUCCEEDED
        await asyncio.wait_for(scope.exited.wait(), 5)
        assert scope.exit_task is scope.enter_task
        assert scope.group_settled.is_set()
        assert _context.get() is None
        assert not await authority.release_turn(authority.claimed[0].lease)


@pytest.mark.asyncio
async def test_pending_publish_cannot_install_worker_after_captured_owner_is_lost(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    publishing, finish = asyncio.Event(), asyncio.Event()
    provider_calls = []
    captured_ids = []

    async def model(messages, info):
        provider_calls.append(1)
        return ModelResponse(parts=[TextPart("must not execute")])

    authority = _ProjectionAuthority()
    features = RuntimeFeatures("publish-owner-loss")
    features.capability(ExecutionCapability(retain_history=True, commit=CommitPolicy.terminal()), scope="runtime")
    features.agent("default", model="default", allow_tools=())
    storage = PersistenceStorage.sqlite(tmp_path / "state.db", cache=authority, coordination=authority)
    async with Persistence.open("publish-owner-loss", storage=storage) as persistence, Runtime.attach(
        persistence=persistence, features=model_features(_Models(_UsageFunctionModel(model)), features),
    ) as runtime:
        backend = runtime._execution_service.runtime_backend()
        coordinator = backend._pending_turns.coordination
        publish = coordinator.publish

        async def paused_publish(record, coordination):
            captured_ids.append(record.execution_id)
            publishing.set()
            await finish.wait()
            await publish(record, coordination)

        monkeypatch.setattr(coordinator, "publish", paused_publish)
        session = await runtime.agents.get().create_session("session")
        caller = asyncio.create_task(session.start("question", idempotency_key="request"))
        try:
            await asyncio.wait_for(publishing.wait(), 5)
            scope = authority.scopes[0]
            owner = backend._turn_owners[captured_ids[0]]
            scope.lost.set()
            async with asyncio.timeout(5):
                while not owner.stopping:
                    await asyncio.sleep(0)
            assert not scope.exited.is_set()
            assert not backend.worker_installed(captured_ids[0])
        finally:
            finish.set()
        with pytest.raises(AIError) as error:
            await asyncio.wait_for(caller, 5)
        assert error.value.code is ErrorCode.STORAGE_COMMIT_UNKNOWN
        assert error.value.safe_details["phase"] == "execution_admission"
        await asyncio.wait_for(scope.exited.wait(), 5)
        assert not backend.worker_installed(captured_ids[0])
        assert provider_calls == []
