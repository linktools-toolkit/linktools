#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Call-local model metadata reconciliation for the graph watch composition."""

import asyncio
from collections.abc import AsyncIterator
from dataclasses import replace
from datetime import datetime, timezone

from ..core import Principal
from ..errors import AIError, ErrorCode
from ._runtime_history import RuntimeHistory
from .service_api import (
    ExecutionView, ModelInteractionItem, ModelInteractionReadBoundary,
    ModelInteractionSubscription, TaskModelProjection, UsageReadCutoff,
)


class _GraphModelProjection:
    def __init__(self, history: RuntimeHistory | None, principal: Principal) -> None:
        self.history = history
        self.principal = principal
        self.views: dict[str, tuple[str, ExecutionView, int]] = {}
        self.boundaries: dict[str, ModelInteractionReadBoundary] = {}
        self.subscriptions: dict[str, ModelInteractionSubscription] = {}
        self.generations: dict[str, int] = {}
        self.waits: dict[str, asyncio.Task[int]] = {}
        self.high_waters: dict[tuple[str, int], int] = {}
        self.active: set[tuple[str, int, int]] = set()
        self.rows: dict[tuple[str, int, int], TaskModelProjection] = {}

    async def add(self, node_id: str, view: ExecutionView, depth: int) -> None:
        existing = self.views.get(view.execution_id)
        if existing is not None:
            if existing[0] != node_id or existing[1].root_execution_id != view.root_execution_id:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            return
        self.views[view.execution_id] = (node_id, view, depth)
        if self.history is not None:
            subscription = await self.history.subscribe_model_interactions(
                view.execution_id, principal=self.principal,
            )
            if subscription is not None:
                self.subscriptions[view.execution_id] = subscription
                self.generations[view.execution_id] = subscription.generation
                self.arm(view.execution_id)

    def arm(self, execution_id: str) -> None:
        subscription = self.subscriptions.get(execution_id)
        if subscription is not None and execution_id not in self.waits:
            self.waits[execution_id] = asyncio.create_task(
                subscription.wait(self.generations[execution_id]),
                name=f"graph-model-change-{execution_id}",
            )

    def changed(self, done: set[asyncio.Task[object]]) -> set[str]:
        changed: set[str] = set()
        for execution_id, task in tuple(self.waits.items()):
            if task in done:
                self.generations[execution_id] = task.result()
                del self.waits[execution_id]
                changed.add(execution_id)
        return changed

    async def capture(self, execution_id: str) -> ModelInteractionReadBoundary:
        if self.history is None:
            boundary = ModelInteractionReadBoundary((), (), False, False)
        else:
            boundary = await self.history.capture_model_interaction_cutoffs(
                execution_id, principal=self.principal,
            )
        self.boundaries[execution_id] = boundary
        return boundary

    async def refresh(
        self, execution_id: str, *, boundary: ModelInteractionReadBoundary | None = None,
    ) -> AsyncIterator[tuple[str, TaskModelProjection]]:
        boundary = await self.capture(execution_id) if boundary is None else boundary
        self.boundaries[execution_id] = boundary
        if self.history is None:
            return
        node_id, view, depth = self.views[execution_id]
        visible_runs = {value.agent_run_seq for value in boundary.cutoffs}
        if any(key[0] == execution_id and key[1] not in visible_runs for key in self.high_waters):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        durable = {value.agent_run_seq: value.model_request_seq for value in boundary.durable_cutoffs}
        for cutoff in boundary.cutoffs:
            if cutoff.execution_id != execution_id:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            run_key = (execution_id, cutoff.agent_run_seq)
            previous = self.high_waters.get(run_key, 0)
            if cutoff.model_request_seq < previous:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            # Previously active requests may finish without increasing the high water.
            ranges = [
                (sequence - 1, sequence)
                for identity_execution, run, sequence in sorted(self.active)
                if identity_execution == execution_id and run == cutoff.agent_run_seq
                and sequence <= previous
            ]
            if previous < cutoff.model_request_seq:
                ranges.append((previous, cutoff.model_request_seq))
            for after, through in ranges:
                while after < through:
                    rows = await self.history.read_model_interaction_metadata(
                        execution_id, principal=self.principal,
                        agent_run_seq=cutoff.agent_run_seq,
                        after_model_request_seq=after, through_model_request_seq=through,
                        limit=min(200, through - after),
                    )
                    if not rows:
                        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                    for item in rows:
                        if (item.execution_id != execution_id or item.agent_run_seq != cutoff.agent_run_seq
                                or item.model_request_seq != after + 1 or item.model_request_seq > through):
                            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                        after = item.model_request_seq
                        item = replace(item, depth=depth)
                        identity = (execution_id, item.agent_run_seq, item.model_request_seq)
                        old = self.rows.get(identity)
                        if old is not None and old.item.status != "RUNNING":
                            if item.status == "RUNNING":
                                continue
                            if old.item != item:
                                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                        is_durable = item.model_request_seq <= durable.get(item.agent_run_seq, 0)
                        if item.status == "RUNNING" or not is_durable and boundary.durable_history_available:
                            self.active.add(identity)
                        else:
                            self.active.discard(identity)
                        projection = TaskModelProjection(
                            item, view.root_execution_id, view.parent_execution_id,
                            view.parent_invocation_id,
                            "durable_history" if is_durable
                            else "local_staging", datetime.now(timezone.utc),
                        )
                        if old is None or old.item != item or old.visibility != projection.visibility:
                            self.rows[identity] = projection
                            yield node_id, projection
            self.high_waters[run_key] = cutoff.model_request_seq
        self.arm(execution_id)

    def cutoffs(self) -> tuple[UsageReadCutoff, ...]:
        return tuple(value for key in sorted(self.boundaries) for value in self.boundaries[key].cutoffs)

    async def close(self) -> None:
        tasks = tuple(self.waits.values())
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        for subscription in self.subscriptions.values():
            await subscription.close()
