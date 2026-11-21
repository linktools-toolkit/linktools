#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Call-local model metadata reconciliation for the graph watch composition."""

import asyncio
from collections.abc import AsyncIterator
from dataclasses import replace
from datetime import datetime, timezone

from ..core import JsonValue, Principal
from ..errors import AIError, ErrorCode
from ..observe.stream import drain_stream_tasks
from .ranges import model_interaction_update_ranges
from .model_progress import decode_model_progress
from .stream import OBSERVATION_TIMEOUT, is_fresh_observation_coverage
from ._service_api import ExecutionHistoryService, ExecutionView, ModelInteractionItem, ModelInteractionReadBoundary, ModelInteractionSubscription, TaskModelProjection, UsageReadCutoff


class ModelProjection:
    def __init__(self, history: ExecutionHistoryService | None, principal: Principal) -> None:
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

    async def add(
        self, node_id: str, view: ExecutionView, depth: int, *, subscribe: bool = True,
    ) -> None:
        existing = self.views.get(view.execution_id)
        if existing is not None:
            if existing[0] != node_id or existing[1].root_execution_id != view.root_execution_id:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            return
        self.views[view.execution_id] = (node_id, view, depth)
        if subscribe and self.history is not None:
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
        prefetched: dict[tuple[int, int], ModelInteractionItem] = {}
        local_runs = {identity[1] for identity, row in self.rows.items()
                      if identity[0] == execution_id and row.visibility == "local_staging"}
        allow_unavailable = (
            all(run in local_runs for identity, run in self.high_waters if identity == execution_id)
            and not any(identity[0] == execution_id and row.visibility == "durable_history"
                        for identity, row in self.rows.items())
        )
        if boundary is None and self.history is not None:
            afters = {run: value for (identity, run), value in self.high_waters.items() if identity == execution_id}
            contiguous = True
            for run, previous in tuple(afters.items()):
                active = sorted(sequence for identity, active_run, sequence in self.active
                                if identity == execution_id and active_run == run and sequence <= previous)
                if active:
                    if len(active) != previous - active[0] + 1:
                        contiguous = False
                        break
                    afters[run] = active[0] - 1
            read_updates = getattr(self.history, "read_model_interaction_updates", None)
            if contiguous and callable(read_updates):
                after_cutoffs = tuple(UsageReadCutoff(execution_id, run, after) for run, after in sorted(afters.items()))
                boundary, items = await read_updates(
                    execution_id, principal=self.principal, after_cutoffs=after_cutoffs, limit=200,
                )
                ranges = model_interaction_update_ranges(
                    execution_id, boundary, after_cutoffs, 200, allow_unavailable=allow_unavailable,
                )
                if ranges and not items:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                expected = []
                for run, after, through in ranges:
                    count = sum(item.agent_run_seq == run for item in items)
                    if count > through - after:
                        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                    expected.extend((run, sequence) for sequence in range(after + 1, after + count + 1))
                if [(item.agent_run_seq, item.model_request_seq) for item in items] != expected:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                for item in items:
                    key = (item.agent_run_seq, item.model_request_seq)
                    if item.execution_id != execution_id or key in prefetched:
                        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                    prefetched[key] = item
        boundary = await self.capture(execution_id) if boundary is None else boundary
        self.boundaries[execution_id] = boundary
        if self.history is None:
            return
        if (not boundary.cutoffs and not boundary.local_staging_available
                and not boundary.durable_history_available):
            if not allow_unavailable:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            self.arm(execution_id)
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
            contiguous: list[tuple[int, int]] = []
            for after, through in ranges:
                if contiguous and contiguous[-1][1] == after:
                    contiguous[-1] = (contiguous[-1][0], through)
                else:
                    contiguous.append((after, through))
            for after, through in contiguous:
                while after < through:
                    page = []
                    for sequence in range(after + 1, min(through, after + 200) + 1):
                        item = prefetched.pop((cutoff.agent_run_seq, sequence), None)
                        if item is None:
                            break
                        page.append(item)
                    rows = tuple(page) if page else await self.history.read_model_interaction_metadata(
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
        if prefetched:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        self.arm(execution_id)

    def cutoffs(self) -> tuple[UsageReadCutoff, ...]:
        return tuple(value for key in sorted(self.boundaries) for value in self.boundaries[key].cutoffs)

    async def close(self) -> None:
        tasks = tuple(self.waits.values())
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        results = await asyncio.gather(
            *(subscription.close() for subscription in self.subscriptions.values()),
            return_exceptions=True,
        )
        for result in results:
            if isinstance(result, BaseException):
                raise result


class _LiveModelSample:
    def __init__(self, projection: ModelProjection, view: ExecutionView) -> None:
        self.projection = projection
        self.generation = 0
        self.rows: dict[tuple[str, int, int], TaskModelProjection] = {}
        self.updates: tuple[tuple[tuple[str, int, int], TaskModelProjection], ...] = ()
        self.changed = asyncio.Event()
        self.references = 0
        self._remote_wakeup = asyncio.Event()
        self._remote_generation = 0
        self._remote_position = ("", 0)
        self._remote_deadline = 0.0
        self._remote_baseline = True
        self._remote_items: dict[tuple[int, int], ModelInteractionItem] = {}
        self._tree_sample: object | None = None
        self.task = asyncio.create_task(self.run(view))

    def reconcile_after_sample(self, sample: object) -> None:
        if self._tree_sample is not sample:
            self._tree_sample = sample
            self._remote_wakeup.set()

    def observe(self, execution_id: str, position: tuple[str, int], record: dict[str, JsonValue]) -> None:
        kind = record.get("kind")
        generation = record.get("producer_generation")
        if kind not in ("model", "coverage", "gap") or type(generation) is not int:
            return
        if generation < self._remote_generation:
            return
        if generation > self._remote_generation:
            self._remote_generation = generation
            self._remote_position = (position[0], 0)
            self._remote_deadline = 0
            self._remote_baseline = True
            self._remote_items.clear()
        if position[0] != self._remote_position[0] or position[1] < self._remote_position[1]:
            return
        if position == self._remote_position and kind != "gap":
            return
        self._remote_position = position
        now = asyncio.get_running_loop().time()
        if kind == "coverage":
            authority = record.get("authority_deadline")
            if (record.get("model") is True and type(authority) in (float, int) and authority > now
                    and is_fresh_observation_coverage(record)):
                if self._remote_deadline <= now:
                    self._remote_baseline = True
                self._remote_deadline = min(authority, now + OBSERVATION_TIMEOUT)
            elif record.get("model") is not True:
                self._remote_deadline = 0
                self._remote_items.clear()
        elif kind == "gap":
            self._remote_deadline = 0
            self._remote_items.clear()
        else:
            try:
                item = decode_model_progress(execution_id, record.get("item"))
            except (TypeError, ValueError):
                self._remote_deadline = 0
                self._remote_items.clear()
            else:
                if len(self._remote_items) < 200:
                    self._remote_items[(item.agent_run_seq, item.model_request_seq)] = item
                else:
                    self._remote_deadline = 0
                    self._remote_items.clear()
        self._remote_wakeup.set()

    async def run(self, view: ExecutionView) -> None:
        execution_id = view.execution_id
        try:
            await self.projection.add(execution_id, view, 0)
            while True:
                updates = {}
                self._remote_wakeup.clear()
                if self._remote_baseline or asyncio.get_running_loop().time() >= self._remote_deadline:
                    self._remote_baseline = False
                    async for _, update in self.projection.refresh(execution_id):
                        item = update.item
                        updates[(item.execution_id, item.agent_run_seq, item.model_request_seq)] = update
                items, self._remote_items = self._remote_items, {}
                for (run, sequence), item in items.items():
                    identity = (execution_id, run, sequence)
                    canonical = self.projection.rows.get(identity)
                    if canonical is not None and canonical.item.status != "RUNNING":
                        continue
                    existing = updates.get(identity) or self.rows.get(identity)
                    if existing is not None and (
                        existing.item == item and existing.visibility == "durable_history"
                        or existing.item.status != "RUNNING"
                    ):
                        continue
                    updates[identity] = TaskModelProjection(
                        item, view.root_execution_id, view.parent_execution_id, view.parent_invocation_id,
                        "durable_history", datetime.now(timezone.utc),
                    )
                for identity, update in tuple(updates.items()):
                    previous = self.rows.get(identity)
                    if previous is not None and (
                        previous.item == update.item and previous.visibility == update.visibility
                        or previous.item.status != "RUNNING" and update.item.status == "RUNNING"
                    ):
                        del updates[identity]
                if updates or not self.generation:
                    self.rows.update(updates)
                    self.updates = tuple(updates.items())
                    self.generation += 1
                    previous, self.changed = self.changed, asyncio.Event()
                    previous.set()
                remote = asyncio.create_task(self._remote_wakeup.wait())
                try:
                    done, _ = await asyncio.wait(
                        (remote, *self.projection.waits.values()), timeout=1.0,
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    self.projection.changed(done)
                    for changed_execution in tuple(self.projection.subscriptions):
                        self.projection.arm(changed_execution)
                finally:
                    await drain_stream_tasks((remote,))
        finally:
            try:
                await self.projection.close()
            finally:
                self.changed.set()

    async def wait(self, after_generation: int) -> int:
        while self.generation <= after_generation and not self.task.done():
            wake = asyncio.create_task(self.changed.wait())
            try:
                await asyncio.wait((wake, self.task), return_when=asyncio.FIRST_COMPLETED)
            finally:
                await drain_stream_tasks((wake,))
        if self.task.done():
            self.task.result()
        return self.generation


class ModelProgress:
    """One live metadata projection for equivalent readers on the same execution binding."""

    def __init__(self) -> None:
        self._samples: dict[tuple[int, Principal, str], _LiveModelSample] = {}

    def attach(
        self, history: ExecutionHistoryService, principal: Principal, view: ExecutionView,
    ) -> _LiveModelSample:
        key = (id(history), principal, view.execution_id)
        sample = self._samples.get(key)
        if sample is None:
            sample = self._samples[key] = _LiveModelSample(ModelProjection(history, principal), view)
        sample.references += 1
        return sample

    async def release(
        self, history: ExecutionHistoryService, principal: Principal, execution_id: str, sample: _LiveModelSample,
    ) -> None:
        sample.references -= 1
        if sample.references:
            return
        del self._samples[(id(history), principal, execution_id)]
        errors = await drain_stream_tasks((sample.task,))
        if errors:
            raise errors[0]


class SharedModelProjection(ModelProjection):
    def __init__(
        self, history: ExecutionHistoryService, principal: Principal, progress: ModelProgress,
    ) -> None:
        super().__init__(history, principal)
        self._progress = progress
        self._samples: dict[str, _LiveModelSample] = {}
        self._delivered: dict[str, int] = {}

    def observe_remote(self, execution_id: str, position: tuple[str, int], record: dict[str, JsonValue]) -> None:
        sample = self._samples.get(execution_id)
        if sample is not None:
            sample.observe(execution_id, position, record)

    def reconcile_after_sample(self, views: tuple[tuple[ExecutionView, int, int], ...]) -> None:
        for view, _, _ in views:
            sample = self._samples.get(view.execution_id)
            if sample is not None:
                sample.reconcile_after_sample(views)

    async def add(
        self, node_id: str, view: ExecutionView, depth: int, *, subscribe: bool = True,
    ) -> None:
        if view.execution_id in self.views:
            await super().add(node_id, view, depth, subscribe=False)
            return
        assert self.history is not None
        # Joining is still an authorized history operation even if a sample is cached.
        authorization = await self.history.subscribe_model_interactions(
            view.execution_id, principal=self.principal,
        )
        if authorization is not None:
            await authorization.close()
        await super().add(node_id, view, depth, subscribe=False)
        self._samples[view.execution_id] = self._progress.attach(self.history, self.principal, view)
        self.generations[view.execution_id] = self._delivered[view.execution_id] = 0

    def arm(self, execution_id: str) -> None:
        if execution_id not in self.waits:
            self.waits[execution_id] = asyncio.create_task(
                self._samples[execution_id].wait(self.generations[execution_id]),
                name=f"execution-model-change-{execution_id}",
            )

    async def refresh(
        self, execution_id: str, *, boundary: ModelInteractionReadBoundary | None = None,
    ) -> AsyncIterator[tuple[str, TaskModelProjection]]:
        sample = self._samples[execution_id]
        revision = await sample.wait(0)
        delivered = self._delivered[execution_id]
        if delivered < revision:
            node_id, _, depth = self.views[execution_id]
            self._delivered[execution_id] = revision
            # A slow reader can reconstruct the latest metadata without holding a
            # queue open or delaying refresh for healthy readers.
            rows = sample.updates if delivered == revision - 1 else tuple(sample.rows.items())
            for identity, projection in rows:
                if self.rows.get(identity) != projection:
                    self.rows[identity] = projection
                    yield node_id, replace(projection, item=replace(projection.item, depth=depth))
        self.generations[execution_id] = max(revision, self.generations[execution_id])
        self.arm(execution_id)

    async def close(self) -> None:
        try:
            await super().close()
        finally:
            assert self.history is not None
            results = await asyncio.gather(
                *(self._progress.release(self.history, self.principal, execution_id, sample)
                  for execution_id, sample in self._samples.items()),
                return_exceptions=True,
            )
            for result in results:
                if isinstance(result, BaseException):
                    raise result


__all__ = ["ModelProjection", "ModelProgress", "SharedModelProjection"]
