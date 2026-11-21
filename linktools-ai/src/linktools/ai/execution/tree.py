#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Derived live event projection over one execution and all its descendants."""

import asyncio
import sys
from collections import deque
from collections.abc import AsyncIterator, Mapping
from dataclasses import replace
from typing import Protocol

from ..core import ExecutionLineageKind, JsonValue, Page, Principal
from ..errors import AIError, ErrorCode
from .events import project_event_payload
from ._tree_ready import ExecutionTreeReady
from .stream import (
    OBSERVATION_TIMEOUT, is_fresh_observation_coverage, shared_stream_observation, shared_stream_positions,
)
from ..observe.cursors import ExecutionWatchPositions, get_preview_positions
from ..observe.stream import is_observation_cleanup, drain_stream_tasks, await_stream_cleanup, report_observation_error
from . import ExecutionStreamEvent, ExecutionEvent, ExecutionTreeEvent, ExecutionView
from ..observe.stream import ExecutionStreamFailure

_DISCOVERY_BACKOFF_INITIAL = 1.0
_DISCOVERY_BACKOFF_MAX = 30.0
_MAX_REMOTE_CHILD_HINTS = 256


class _ExecutionTreeReader(Protocol):
    async def inspect(
        self,
        execution_id: str,
        *,
        principal: Principal,
    ) -> ExecutionView: ...

    async def list_children(
        self,
        execution_id: str,
        *,
        principal: Principal,
    ) -> tuple[ExecutionView, ...]: ...


class _ExecutionEventStreamer(Protocol):
    async def list(
        self,
        execution_id: str,
        *,
        principal: Principal,
        after_event_seq: int = 0,
        limit: int = 100,
    ) -> Page[ExecutionEvent]: ...

    def stream(
        self,
        execution_id: str,
        *,
        principal: Principal,
        after_event_seq: int = 0,
    ) -> AsyncIterator[ExecutionStreamEvent]: ...


class _ExecutionTreeSubscription:
    def __init__(
        self,
        owner: "ExecutionTreeBroker",
        parent_execution_id: str,
    ) -> None:
        self._owner = owner
        self._parent_execution_id = parent_execution_id
        self._pending: set[str] = set()
        self._event = asyncio.Event()
        self._closed = False

    def publish(self, child_execution_id: str) -> None:
        if self._closed:
            return
        self._pending.add(child_execution_id)
        self._event.set()

    def drain(self) -> tuple[str, ...]:
        values = tuple(sorted(self._pending))
        self._pending.clear()
        self._event.clear()
        return values

    async def wait(self) -> tuple[str, ...]:
        while not self._pending and not self._closed:
            self._event.clear()
            await self._event.wait()
        return self.drain()

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._owner._remove(self._parent_execution_id, self)
        self._event.set()


class ExecutionTreeBroker:
    """Notify active tree readers about newly established direct children."""

    def __init__(self) -> None:
        self._subscriptions: dict[str, set[_ExecutionTreeSubscription]] = {}

    def subscribe(self, parent_execution_id: str) -> _ExecutionTreeSubscription:
        subscription = _ExecutionTreeSubscription(self, parent_execution_id)
        self._subscriptions.setdefault(parent_execution_id, set()).add(subscription)
        return subscription

    def publish(self, parent_execution_id: str, child_execution_id: str) -> None:
        for subscription in tuple(
            self._subscriptions.get(parent_execution_id, ())
        ):
            subscription.publish(child_execution_id)

    def _remove(
        self,
        parent_execution_id: str,
        subscription: _ExecutionTreeSubscription,
    ) -> None:
        values = self._subscriptions.get(parent_execution_id)
        if values is None:
            return
        values.discard(subscription)
        if not values:
            self._subscriptions.pop(parent_execution_id, None)


def _capture_views(
    views: dict[str, ExecutionView], depths: dict[str, int], sequences: Mapping[str, int],
) -> tuple[tuple[ExecutionView, int, int], ...]:
    captured = []
    for view in views.values():
        cutoff = view.event_seq
        if (
            isinstance(cutoff, bool)
            or not isinstance(cutoff, int)
            or cutoff < sequences.get(view.execution_id, 0)
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        captured.append((view, depths[view.execution_id], cutoff))
    return tuple(captured)


class _LiveTreeSample:
    def __init__(self) -> None:
        self.references = 0
        self.task: asyncio.Task[tuple[tuple[ExecutionView, int, int], ...]] | None = None
        self.expires_at = 0.0


class ExecutionTreeStreamer:
    """Merge an execution subtree with relative depths and per-stream ordering."""

    def __init__(
        self,
        executions: _ExecutionTreeReader,
        events: _ExecutionEventStreamer,
        broker: ExecutionTreeBroker,
    ) -> None:
        self._executions = executions
        self._events = events
        self._broker = broker
        self._live_samples: dict[tuple[str, Principal], _LiveTreeSample] = {}

    async def capture(
        self,
        execution_id: str,
        *,
        principal: Principal,
        after_event_seqs: Mapping[str, int] | None = None,
        include_descendants: bool = True,
    ) -> tuple[tuple[ExecutionView, int, int], ...]:
        """Freeze one finite visible subtree and its per-execution durable cutoffs."""
        if not isinstance(include_descendants, bool):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        sequences = _normalize_after_event_seqs(after_event_seqs)
        if not include_descendants and (set(sequences) | set(get_preview_positions(sequences))) - {execution_id}:
            raise AIError(ErrorCode.CURSOR_INVALID)
        root = await self._executions.inspect(execution_id, principal=principal)
        _validate_root(root, execution_id)
        views = {execution_id: root}
        depths = {execution_id: 0}
        scanned: set[str] = set()

        def add_child(parent_id: str, child: ExecutionView) -> bool:
            _validate_child(views[parent_id], child)
            existing = views.get(child.execution_id)
            if existing is not None:
                if (
                    existing.parent_execution_id != child.parent_execution_id
                    or existing.root_execution_id != child.root_execution_id
                    or existing.lineage_kind != child.lineage_kind
                    or existing.parent_invocation_id != child.parent_invocation_id
                ):
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                return False
            views[child.execution_id] = child
            depths[child.execution_id] = depths[parent_id] + 1
            return True

        async def discover(parent_ids: tuple[str, ...]) -> None:
            if not include_descendants:
                return
            remaining = deque(parent_ids)
            while remaining:
                parent_id = remaining.popleft()
                if parent_id in scanned:
                    continue
                scanned.add(parent_id)
                for child in await self._executions.list_children(parent_id, principal=principal):
                    if add_child(parent_id, child):
                        remaining.append(child.execution_id)

        await discover((execution_id,))
        for member_id in sequences:
            chain: list[ExecutionView] = []
            seen: set[str] = set()
            current_id = member_id
            while current_id not in views:
                if current_id in seen:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                seen.add(current_id)
                view = await self._executions.inspect(current_id, principal=principal)
                if view.execution_id != current_id:
                    raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
                if (
                    view.lineage_kind is not ExecutionLineageKind.SUBAGENT
                    or not view.parent_execution_id
                    or view.root_execution_id != root.root_execution_id
                ):
                    raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
                chain.append(view)
                current_id = view.parent_execution_id
            for view in reversed(chain):
                add_child(current_id, view)
                current_id = view.execution_id
            await discover(tuple(view.execution_id for view in reversed(chain)))
        return _capture_views(views, depths, sequences)

    async def replay(
        self,
        captured: tuple[tuple[ExecutionView, int, int], ...],
        *,
        principal: Principal,
        after_event_seqs: Mapping[str, int] | None = None,
        include_content: bool = False,
    ) -> AsyncIterator[ExecutionTreeEvent]:
        """Read only the captured durable suffix, failing closed on any gap."""
        if not isinstance(include_content, bool):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        sequences = _normalize_after_event_seqs(after_event_seqs)
        if set(sequences) - {view.execution_id for view, _, _ in captured}:
            raise AIError(ErrorCode.CURSOR_INVALID)
        for view, depth, cutoff in captured:
            cursor = sequences.get(view.execution_id, 0)
            if cursor > cutoff:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            while cursor < cutoff:
                page = await self._events.list(
                    view.execution_id, principal=principal, after_event_seq=cursor,
                    limit=min(200, cutoff - cursor),
                )
                items = tuple(event for event in page.items if event.event_seq <= cutoff)
                if not items:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                for event in items:
                    if event.execution_id != view.execution_id or event.event_seq != cursor + 1:
                        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                    cursor = event.event_seq
                    yield ExecutionTreeEvent(
                        view.execution_id, view.agent_id, view.lineage_kind,
                        view.parent_execution_id, view.root_execution_id,
                        view.parent_invocation_id, depth,
                        _project_stream_event(ExecutionStreamEvent(
                            event.execution_id, event.event_seq, event.event_type, event.payload,
                        ), include_content),
                    )

    def stream(
        self,
        execution_id: str,
        *,
        principal: Principal,
        after_event_seqs: Mapping[str, int] | None = None,
        include_content: bool = False,
        include_output_preview: bool = False,
        include_descendants: bool = True,
        ready: asyncio.Event | None = None,
    ) -> AsyncIterator[ExecutionTreeEvent]:
        if (not isinstance(include_content, bool) or not isinstance(include_output_preview, bool)
                or not isinstance(include_descendants, bool)):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        return self._stream(
            execution_id,
            principal=principal,
            after_event_seqs=_normalize_after_event_seqs(after_event_seqs),
            include_content=include_content,
            include_output_preview=include_output_preview,
            include_descendants=include_descendants,
            ready=ready,
        )

    async def _stream(
        self,
        execution_id: str,
        *,
        principal: Principal,
        after_event_seqs: Mapping[str, int],
        include_content: bool,
        include_output_preview: bool,
        include_descendants: bool,
        ready: asyncio.Event | None,
    ) -> AsyncIterator[ExecutionTreeEvent]:
        if not include_descendants and (set(after_event_seqs) | set(get_preview_positions(after_event_seqs))) - {execution_id}:
            raise AIError(ErrorCode.CURSOR_INVALID)
        root = await self._executions.inspect(execution_id, principal=principal)
        _validate_root(root, execution_id)

        views: dict[str, ExecutionView] = {}
        depths: dict[str, int] = {}
        subscriptions: dict[str, _ExecutionTreeSubscription] = {}
        child_waits: dict[str, asyncio.Task[tuple[str, ...]]] = {}
        streams: dict[str, AsyncIterator[ExecutionStreamEvent]] = {}
        pending: dict[str, asyncio.Task[ExecutionStreamEvent]] = {}
        discovery_wait: asyncio.Task[None] | None = None
        discovery: asyncio.Task[None] | None = None
        observing = False
        shared = ready if isinstance(ready, ExecutionTreeReady) and ready.positions is not None else None
        preparation_failure: ExecutionStreamFailure | None = None
        sample_key = (execution_id, principal)
        live_sample: _LiveTreeSample | None = None
        remote_coverage: dict[str, tuple[str, float]] = {}
        remote_baselines: set[str] = set()
        remote_children: dict[tuple[str, str], str] = {}
        remote_changed = asyncio.Event()
        remote_wait: asyncio.Task[bool] | None = None

        def invalidate_topology(parent_id: str) -> None:
            remote_coverage.pop(parent_id, None)
            remote_baselines.add(parent_id)
            remote_changed.set()

        def observe_topology(parent_id: str, position: tuple[str, int], record: dict[str, JsonValue]) -> None:
            if parent_id not in subscriptions:
                return
            identity, _ = position
            kind = record.get("kind")
            now = asyncio.get_running_loop().time()
            coverage = remote_coverage.get(parent_id)
            if kind == "coverage":
                authority = record.get("authority_deadline")
                if (record.get("tree") is True and type(authority) in (int, float)
                        and authority > now and is_fresh_observation_coverage(record)):
                    remote_coverage[parent_id] = (identity, min(authority, now + OBSERVATION_TIMEOUT))
                    if coverage is None or coverage[0] != identity or coverage[1] <= now:
                        remote_baselines.add(parent_id)
                        remote_changed.set()
                elif coverage is not None:
                    invalidate_topology(parent_id)
            elif kind == "gap":
                if coverage is not None:
                    invalidate_topology(parent_id)
            elif kind == "child" and coverage is not None and coverage[0] == identity:
                child_id = record.get("execution_id")
                if (not isinstance(child_id, str) or not child_id or child_id == parent_id
                        or len(remote_children) >= _MAX_REMOTE_CHILD_HINTS):
                    invalidate_topology(parent_id)
                else:
                    remote_children[(parent_id, child_id)] = identity
                    remote_changed.set()

        def next_event(stream: AsyncIterator[ExecutionStreamEvent], member_id: str) -> asyncio.Task[ExecutionStreamEvent]:
            if include_descendants:
                with shared_stream_observation(observe_topology):
                    return _next_event_task(stream, member_id, after_event_seqs)
            return _next_event_task(stream, member_id, after_event_seqs)

        def start(view: ExecutionView) -> None:
            stream = self._events.stream(
                view.execution_id,
                principal=principal,
                after_event_seq=after_event_seqs.get(view.execution_id, 0),
            )
            streams[view.execution_id] = stream
            pending[view.execution_id] = next_event(stream, view.execution_id)

        def wait_for_children(parent_id: str) -> None:
            if not include_descendants:
                return
            child_waits[parent_id] = asyncio.create_task(
                subscriptions[parent_id].wait(),
                name=f"execution-tree-children-{parent_id}",
            )

        def add_view(view: ExecutionView, depth: int) -> bool:
            nonlocal preparation_failure
            existing = views.get(view.execution_id)
            if existing is not None:
                if (
                    existing.parent_execution_id != view.parent_execution_id
                    or existing.root_execution_id != view.root_execution_id
                    or existing.lineage_kind != view.lineage_kind
                    or existing.parent_invocation_id != view.parent_invocation_id
                ):
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                return False
            try:
                subscription = self._broker.subscribe(view.execution_id) if include_descendants else None
            except AIError:
                raise
            except Exception as error:
                if observing:
                    raise ExecutionStreamFailure(error) from error
                if preparation_failure is None:
                    preparation_failure = ExecutionStreamFailure(error)
                subscription = None
            # A missing optional broker must not bypass durable cursor validation.
            if subscription is not None:
                subscriptions[view.execution_id] = subscription
            views[view.execution_id] = view
            depths[view.execution_id] = depth
            if observing:
                start(view)
                wait_for_children(view.execution_id)
            return True

        def add_child(parent_id: str, child: ExecutionView) -> bool:
            _validate_child(views[parent_id], child)
            return add_view(child, depths[parent_id] + 1)

        async def discover_persisted_children(parent_ids: tuple[str, ...]) -> bool:
            if not include_descendants:
                return False
            added = False
            remaining = deque(parent_ids)
            scanned: set[str] = set()
            while remaining:
                parent_id = remaining.popleft()
                if parent_id in scanned:
                    continue
                scanned.add(parent_id)
                for child in await self._executions.list_children(
                    parent_id, principal=principal,
                ):
                    if add_child(parent_id, child):
                        added = True
                        remaining.append(child.execution_id)
            return added

        async def discover_child(parent_id: str, child_id: str) -> None:
            if child_id in views:
                _validate_child(views[parent_id], views[child_id])
                return
            child = await self._executions.inspect(child_id, principal=principal)
            if child.execution_id != child_id:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if add_child(parent_id, child):
                await discover_persisted_children((child_id,))

        async def validate_cursor_member(member_id: str) -> None:
            # A durable cursor may name a member whose index is not visible yet.
            # Read back its parent chain instead of rejecting an authorized resume.
            chain: list[ExecutionView] = []
            seen: set[str] = set()
            current_id = member_id
            while current_id not in views:
                if current_id in seen:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                seen.add(current_id)
                view = await self._executions.inspect(current_id, principal=principal)
                if view.execution_id != current_id:
                    raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
                if (
                    view.lineage_kind is not ExecutionLineageKind.SUBAGENT
                    or not view.parent_execution_id
                    or view.root_execution_id != root.root_execution_id
                ):
                    raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
                chain.append(view)
                current_id = view.parent_execution_id
            for view in reversed(chain):
                add_child(current_id, view)
                current_id = view.execution_id
            if chain:
                await discover_persisted_children(tuple(view.execution_id for view in chain))

        def broker_result(task: asyncio.Task[tuple[str, ...]]) -> tuple[str, ...]:
            try:
                return task.result()
            except AIError:
                raise
            except Exception as error:
                raise ExecutionStreamFailure(error) from error

        async def discover_remote() -> None:
            baselines, hints = tuple(remote_baselines), tuple(remote_children.items())
            remote_baselines.clear()
            remote_children.clear()
            remote_changed.clear()
            if baselines:
                await discover_persisted_children(baselines)
            for (parent_id, child_id), identity in hints:
                coverage = remote_coverage.get(parent_id)
                if coverage is None or coverage[0] != identity:
                    continue
                try:
                    child = views.get(child_id)
                    if child is None:
                        child = await self._executions.inspect(child_id, principal=principal)
                    if child.execution_id != child_id:
                        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                    _validate_child(views[parent_id], child)
                except Exception:
                    # A transport hint is not proof of a child. Re-read the
                    # authenticated index, leaving true storage errors visible.
                    invalidate_topology(parent_id)
                    continue
                if add_child(parent_id, child):
                    await discover_persisted_children((child_id,))

        async def discover_ready(done: set[asyncio.Task]) -> None:
            nonlocal discovery_wait, discovery_backoff, remote_wait
            observed = len(views)
            members = tuple(pending)
            if remote_wait in done:
                await discover_remote()
                remote_wait = asyncio.create_task(remote_changed.wait())
            for parent_id, task in tuple(child_waits.items()):
                if task in done:
                    for child_id in broker_result(task):
                        await discover_child(parent_id, child_id)
                    wait_for_children(parent_id)
            if shared is not None and len(views) != observed:
                shared.snapshot = _capture_views(views, depths, {})
                shared.sampled.set()

            if discovery_wait in done:
                now = asyncio.get_running_loop().time()
                # A member attached in this round just read its child index.
                covered = members and all(
                    member not in remote_baselines and member in remote_coverage
                    and remote_coverage[member][1] > now for member in members
                )
                if covered:
                    # The baseline and terminal pass still use the authoritative
                    # index; covered facts only replace steady reconciliation.
                    discovery_backoff = _DISCOVERY_BACKOFF_INITIAL
                elif shared is not None:
                    assert shared.positions is not None
                    assert live_sample is not None
                    positions = shared.positions()
                    now = asyncio.get_running_loop().time()
                    sample_task = live_sample.task
                    if sample_task is None or (
                        sample_task.done() and (
                            now >= live_sample.expires_at or sample_task.cancelled()
                            or sample_task.exception() is not None
                        )
                    ):
                        sample_task = asyncio.create_task(
                            self.capture(execution_id, principal=principal, include_descendants=include_descendants),
                        )
                        live_sample.task = sample_task
                        live_sample.expires_at = now + _DISCOVERY_BACKOFF_INITIAL
                    snapshot = await asyncio.shield(sample_task)
                    cutoffs = {view.execution_id: cutoff for view, _, cutoff in snapshot}
                    if ((set(positions) | set(get_preview_positions(positions))) - cutoffs.keys()
                            or any(sequence > cutoffs[member] for member, sequence in positions.items())):
                        # A cached sample need not cover a newer live cursor.
                        # Recheck its membership and durable watermark from
                        # authority using the position captured at read start.
                        snapshot = await self.capture(
                            execution_id, principal=principal, after_event_seqs=positions,
                            include_descendants=include_descendants,
                        )
                    for view, depth, _ in snapshot:
                        add_view(view, depth)
                    shared.snapshot = _capture_views(views, depths, {})
                    shared.sampled.set()
                else:
                    added = await discover_persisted_children(tuple(views))
                    discovery_backoff = (
                        _DISCOVERY_BACKOFF_INITIAL if added
                        else min(_DISCOVERY_BACKOFF_MAX, discovery_backoff * 2)
                    )
                discovery_wait = asyncio.create_task(
                    asyncio.sleep(discovery_backoff),
                    name=f"execution-tree-discovery-{execution_id}",
                )

        try:
            add_view(root, 0)
            await discover_persisted_children((execution_id,))
            for member_id in after_event_seqs:
                await validate_cursor_member(member_id)

            if preparation_failure is not None:
                if ready is not None:
                    ready.set()
                raise preparation_failure
            if isinstance(ready, ExecutionTreeReady):
                ready.views = _capture_views(views, depths, after_event_seqs)
            for view in tuple(views.values()):
                start(view)
                wait_for_children(view.execution_id)
            observing = True
            if ready is not None:
                ready.set()

            discovery_backoff = _DISCOVERY_BACKOFF_INITIAL
            if shared is not None:
                live_sample = self._live_samples.setdefault(sample_key, _LiveTreeSample())
                live_sample.references += 1
            if include_descendants:
                remote_wait = asyncio.create_task(remote_changed.wait())
                discovery_wait = asyncio.create_task(
                    asyncio.sleep(discovery_backoff),
                    name=f"execution-tree-discovery-{execution_id}",
                )
            final_discovery = False
            while True:
                if not pending:
                    if discovery is not None:
                        # Cleanup owns cancellation of this read, including adapters
                        # that discard the cancellation message while unwinding.
                        await asyncio.shield(discovery)
                        discovery = None
                        if pending:
                            continue
                    if final_discovery:
                        return
                    # This is a finite observation phase, not permanent subtree
                    # closure. Later admissions belong to the next watch call.
                    final_discovery = True
                    for parent_id, subscription in tuple(subscriptions.items()):
                        task = child_waits.get(parent_id)
                        child_ids = broker_result(task) if task is not None and task.done() else ()
                        try:
                            child_ids += subscription.drain()
                        except AIError:
                            raise
                        except Exception as error:
                            raise ExecutionStreamFailure(error) from error
                        for child_id in child_ids:
                            await discover_child(parent_id, child_id)
                    await discover_persisted_children(tuple(views))
                    if not pending:
                        return

                wait_tasks = tuple(pending.values())
                if discovery is not None:
                    wait_tasks += (discovery,)
                elif not final_discovery:
                    wait_tasks += tuple(child_waits.values())
                    if discovery_wait is not None:
                        wait_tasks += (discovery_wait,)
                    if remote_wait is not None:
                        wait_tasks += (remote_wait,)
                done, _ = await asyncio.wait(wait_tasks, return_when=asyncio.FIRST_COMPLETED)

                # Contract failures and cancellation outrank optional transport
                # failures from any parent notification in the same round.
                for task in done:
                    if task.cancelled():
                        task.result()
                broker_tasks = set(child_waits.values())
                for task in (*pending.values(), *child_waits.values(), *((discovery,) if discovery is not None else ())):
                    if task not in done:
                        continue
                    error = task.exception()
                    if error is not None and not isinstance(
                        error, (StopAsyncIteration, ExecutionStreamFailure)
                    ) and (task not in broker_tasks or isinstance(error, AIError)):
                        raise error
                if discovery is not None and discovery in done:
                    discovery.result()
                    discovery = None
                if not final_discovery and discovery is None and (
                    discovery_wait in done or remote_wait in done or any(task in done for task in child_waits.values())
                ):
                    discovery = asyncio.create_task(discover_ready(done))

                completed = sorted(
                    (key, task) for key, task in pending.items() if task in done
                )
                for key, task in completed:
                    pending.pop(key, None)
                    try:
                        event = task.result()
                    except StopAsyncIteration:
                        streams.pop(key, None)
                        remote_coverage.pop(key, None)
                        continue
                    view = views[key]
                    yield ExecutionTreeEvent(
                        view.execution_id,
                        view.agent_id,
                        view.lineage_kind,
                        view.parent_execution_id,
                        view.root_execution_id,
                        view.parent_invocation_id,
                        depths[key],
                        _project_stream_event(event, include_content, include_output_preview=include_output_preview),
                    )
                    pending[key] = next_event(streams[key], key)
        finally:
            active_error = sys.exc_info()[1]

            async def cleanup() -> None:
                errors = await drain_stream_tasks((discovery,)) if discovery is not None else []
                if live_sample is not None:
                    live_sample.references -= 1
                    if not live_sample.references:
                        self._live_samples.pop(sample_key)
                        if live_sample.task is not None:
                            errors.extend(await drain_stream_tasks((live_sample.task,)))
                wait_tasks = tuple(child_waits.values()) + tuple(pending.values())
                if discovery_wait is not None:
                    wait_tasks += (discovery_wait,)
                if remote_wait is not None:
                    wait_tasks += (remote_wait,)
                broker_tasks = set(child_waits.values())

                def map_error(task: asyncio.Task, error: BaseException) -> BaseException:
                    if task in broker_tasks and isinstance(error, Exception) and not isinstance(
                        error, (AIError, ExecutionStreamFailure)
                    ):
                        return ExecutionStreamFailure(error)
                    return error

                errors.extend(await drain_stream_tasks(wait_tasks, map_error=map_error))

                async def close_stream(stream: AsyncIterator[ExecutionStreamEvent]) -> None:
                    close = getattr(stream, "aclose", None)
                    if close is not None:
                        try:
                            await close()
                        except BaseException as error:
                            report_observation_error(error)
                            errors.append(error)

                async def close_subscription(subscription: _ExecutionTreeSubscription) -> None:
                    try:
                        await subscription.close()
                    except AIError as error:
                        report_observation_error(error)
                        errors.append(error)
                    except Exception as error:
                        errors.append(ExecutionStreamFailure(error))

                await asyncio.gather(
                    *(close_stream(stream) for stream in tuple(streams.values())),
                    *(close_subscription(subscription) for subscription in tuple(subscriptions.values())),
                )
                failure = _cleanup_failure(active_error, errors)
                if failure is not None:
                    raise failure

            await await_stream_cleanup(cleanup(), active_error)


def _cleanup_failure(
    active_error: BaseException | None,
    errors: list[BaseException],
) -> BaseException | None:
    if (active_error is not None
            and not isinstance(active_error, (ExecutionStreamFailure, GeneratorExit))
            and not is_observation_cleanup(active_error)):
        return None
    return next(
        (error for error in errors if isinstance(error, asyncio.CancelledError)),
        next(
            (error for error in errors if not isinstance(error, ExecutionStreamFailure)),
            errors[0] if errors and (active_error is None or isinstance(active_error, GeneratorExit)
                                     or is_observation_cleanup(active_error)) else None,
        ),
    )


def _next_event_task(
    stream: AsyncIterator[ExecutionStreamEvent], execution_id: str,
    positions: Mapping[str, int],
) -> asyncio.Task[ExecutionStreamEvent]:
    if isinstance(positions, ExecutionWatchPositions):
        # The new task inherits this context; the generator never owns a cross-yield token.
        with shared_stream_positions(get_preview_positions(positions)):
            return asyncio.create_task(stream.__anext__(), name=f"execution-events-{execution_id}")
    return asyncio.create_task(stream.__anext__(), name=f"execution-events-{execution_id}")


def _validate_root(view: ExecutionView, execution_id: str) -> None:
    if view.execution_id != execution_id:
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
    if view.lineage_kind is ExecutionLineageKind.SUBAGENT:
        if (
            not view.parent_execution_id
            or view.parent_execution_id == execution_id
            or not view.parent_invocation_id
            or view.root_execution_id == execution_id
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    elif view.parent_execution_id is not None or view.parent_invocation_id is not None:
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID)


def _validate_child(root: ExecutionView, child: ExecutionView) -> None:
    if (
        child.execution_id == root.execution_id
        or child.lineage_kind is not ExecutionLineageKind.SUBAGENT
        or child.parent_execution_id != root.execution_id
        or child.root_execution_id != root.root_execution_id
        or not child.parent_invocation_id
    ):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)


def _project_stream_event(
    event: ExecutionStreamEvent,
    include_content: bool, *, include_output_preview: bool = False,
) -> ExecutionStreamEvent:
    return replace(event, payload=project_event_payload(
        event.event_type, event.payload, include_content=include_content, include_output_preview=include_output_preview,
    ))


def _normalize_after_event_seqs(
    value: Mapping[str, int] | None,
) -> Mapping[str, int]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
    result: dict[str, int] = {}
    for execution_id, sequence in value.items():
        if (
            not isinstance(execution_id, str)
            or not execution_id
            or isinstance(sequence, bool)
            or not isinstance(sequence, int)
            or sequence < 0
        ):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        result[execution_id] = sequence
    return (ExecutionWatchPositions(result, get_preview_positions(value))
            if isinstance(value, ExecutionWatchPositions) else result)


__all__ = ["ExecutionTreeBroker", "ExecutionTreeStreamer"]
