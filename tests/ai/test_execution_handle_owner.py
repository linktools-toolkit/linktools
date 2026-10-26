#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Execution handles borrow execution-domain operations and observation resources."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Mapping

import pytest

from linktools.ai.core import BudgetUsage, ExecutionStatus, Principal, UsageMetrics
from linktools.ai.execution import Execution
from linktools.ai.execution._observation import _ExecutionModelObservation
from linktools.ai.execution.handle import ExecutionBinding
from linktools.ai.execution import CancelExecutionRequest, CancelExecutionResult, ExecutionResult, ExecutionTreeEvent, ExecutionView


class _Owner:
    namespace = "independent"
    _history = None

    def __init__(self) -> None:
        self.sessions: set[object] = set()
        self.requests: dict[str, tuple[str, Principal]] = {}
        self._execution_service = self

    def binding(self) -> ExecutionBinding:
        return ExecutionBinding(
            self.namespace, self, self._ensure_open, self._register_observation,
            self._release_observation, self._capture_execution_tree, self._replay_execution_tree,
        )

    def _ensure_open(self) -> None:
        pass

    def _register_observation(self, session: object) -> None:
        self.sessions.add(session)

    def _release_observation(self, session: object) -> None:
        self.sessions.discard(session)

    async def wait(self, execution_id: str, *, principal: Principal) -> ExecutionResult:
        self.requests["wait"] = execution_id, principal
        return ExecutionResult(execution_id, ExecutionStatus.SUCCEEDED, 7, UsageMetrics())

    async def cancel(self, execution_id: str, request: CancelExecutionRequest) -> CancelExecutionResult:
        self.requests["cancel"] = execution_id, request.principal
        return CancelExecutionResult(execution_id, True)

    async def budget_usage(self, execution_id: str, *, principal: Principal) -> BudgetUsage | None:
        self.requests["budget"] = execution_id, principal
        return None

    async def _capture_execution_tree(
        self, execution_id: str, *, principal: Principal, after_event_seqs: Mapping[str, int] | None = None,
    ) -> tuple[tuple[ExecutionView, int, int], ...]:
        self.requests["capture"] = execution_id, principal
        return ()

    async def _replay_execution_tree(
        self, captured: tuple[tuple[ExecutionView, int, int], ...], *, principal: Principal,
        after_event_seqs: Mapping[str, int] | None = None, include_content: bool = False,
    ) -> AsyncIterator[ExecutionTreeEvent]:
        if False:
            yield


async def _tree(
    execution_id: str, *, principal: Principal, after_event_seqs: Mapping[str, int] | None = None,
    include_content: bool = False, ready: asyncio.Event | None = None,
) -> AsyncIterator[ExecutionTreeEvent]:
    if ready is not None:
        ready.set()
    if False:
        yield


@pytest.mark.asyncio
async def test_handle_controls_use_execution_owner_and_principal() -> None:
    owner = _Owner()
    principal = Principal("worker", "tenant")
    handle = Execution(owner.binding(), "execution", principal, _tree)
    assert (await handle.wait()).result.output == 7
    assert (await handle.cancel(idempotency_key="cancel")).cancelled
    assert await handle.budget_usage() is None
    assert owner.requests == {
        operation: ("execution", principal) for operation in ("wait", "cancel", "budget")
    }


@pytest.mark.asyncio
async def test_handle_watch_releases_observation_scope() -> None:
    owner = _Owner()
    handle = Execution(owner.binding(), "execution", Principal("worker", "tenant"), _tree)
    stream = handle.watch()
    assert [event async for event in stream] == []
    assert not owner.sessions


@pytest.mark.asyncio
async def test_model_observation_replays_empty_execution_snapshot_without_history() -> None:
    owner = _Owner()
    principal = Principal("worker", "tenant")
    observation = _ExecutionModelObservation(owner.binding(), "execution", principal, _tree, None, False)
    stream = await observation.final()
    assert [event async for event in stream] == []
    assert owner.requests["capture"] == ("execution", principal)
    assert not owner.sessions
