#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Execution-owned composition of tree events and model request metadata."""

import asyncio
import sys
from collections.abc import AsyncIterator, Iterable
from dataclasses import replace
from typing import TYPE_CHECKING

from ..core import Principal
from ..errors import AIError, ErrorCode, ObservationError
from .projection import ModelProjection
from ..observe.stream import await_stream_cleanup, drain_stream_tasks, is_observation_cleanup, report_observation_error
from .stream import SharedStreamEvent
from ..observe.cursors import get_preview_positions, ExecutionWatchPositions, decode_execution_observation_cursor, encode_execution_observation_cursor, encode_execution_watch_cursor
from ._service_api import ExecutionObservationEvent, ExecutionTreeEvent, ExecutionView, TaskModelProjection
from ..observe.stream import ExecutionStreamFailure

if TYPE_CHECKING:
    from .handle import ExecutionTreeWatcher, ExecutionBinding


class _ExecutionModelObservation:
    def __init__(
        self, owner: "ExecutionBinding", execution_id: str, principal: Principal,
        watch_tree: "ExecutionTreeWatcher", cursor: str | None, include_content: bool,
        finalizing: asyncio.Event | None = None,
    ) -> None:
        self.owner = owner
        self.execution_id = execution_id
        self.principal = principal
        self.watch_tree = watch_tree
        self.include_content = include_content
        self.finalizing = finalizing
        positions = {} if cursor is None else decode_execution_observation_cursor(
            owner.namespace, principal.tenant_id, execution_id, cursor,
            include_content=include_content,
        )
        self.previews = get_preview_positions(positions)
        self.sequences = ExecutionWatchPositions(positions, self.previews)
        self.cursor = cursor
        self.models = ModelProjection(owner.history, principal)

    def checkpoint(self) -> str:
        return encode_execution_observation_cursor(
            self.owner.namespace, self.principal.tenant_id, self.execution_id,
            include_content=self.include_content, event_seqs=self.sequences, preview_positions=self.previews,
        )

    def tree_event(self, event: ExecutionTreeEvent) -> ExecutionObservationEvent:
        if isinstance(event.event, SharedStreamEvent):
            if event.event.position[0]:
                self.previews[event.execution_id] = event.event.position
            else:
                self.previews.pop(event.execution_id, None)
        sequence = event.event.durable_seq
        if sequence is not None:
            if sequence <= self.sequences.get(event.execution_id, 0):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            self.sequences[event.execution_id] = sequence
        inner = replace(event, cursor=encode_execution_watch_cursor(
            self.owner.namespace, self.principal.tenant_id, self.execution_id,
            include_content=self.include_content, event_seqs=self.sequences, preview_positions=self.previews,
        ))
        self.cursor = self.checkpoint()
        return ExecutionObservationEvent(inner, self.cursor)

    async def capture(self, *, subscribe: bool = True) -> tuple[tuple[ExecutionView, int, int], ...]:
        captured = await self.owner.capture_tree(
            self.execution_id, principal=self.principal, after_event_seqs=ExecutionWatchPositions(self.sequences, self.previews),
        )
        for view, depth, _ in captured:
            await self.models.add(self.execution_id, view, depth, subscribe=subscribe)
        return captured

    async def refresh(self, execution_ids: Iterable[str], *, finite: bool = False) -> AsyncIterator[ExecutionObservationEvent]:
        for execution_id in sorted(execution_ids):
            boundary = self.models.boundaries[execution_id] if finite else None
            async for _, projection in self.models.refresh(execution_id, boundary=boundary):
                self.cursor = self.checkpoint()
                yield ExecutionObservationEvent(projection, self.cursor)

    async def reconcile(self, discovered: asyncio.Event) -> AsyncIterator[TaskModelProjection]:
        tick = asyncio.create_task(asyncio.sleep(1.0))
        discovery = asyncio.create_task(discovered.wait())
        try:
            while True:
                done, _ = await asyncio.wait(
                    {tick, discovery, *self.models.waits.values()},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                dirty = self.models.changed(done)
                if tick in done or discovery in done:
                    discovered.clear()
                    await self.capture()
                    dirty.update(self.models.views)
                    if tick in done:
                        tick.result()
                        tick = asyncio.create_task(asyncio.sleep(1.0))
                    if discovery in done:
                        discovery.result()
                        discovery = asyncio.create_task(discovered.wait())
                for execution_id in sorted(dirty):
                    async for _, projection in self.models.refresh(execution_id):
                        yield projection
        finally:
            await drain_stream_tasks((tick, discovery))

    async def live(self, ready: asyncio.Event | None) -> AsyncIterator[ExecutionObservationEvent]:
        tree_ready = asyncio.Event()
        stream = self.watch_tree(
            self.execution_id, principal=self.principal, after_event_seqs=self.sequences,
            include_content=self.include_content, ready=tree_ready,
        )
        pending = asyncio.create_task(stream.__anext__())
        prepared = asyncio.create_task(tree_ready.wait())
        discovered = asyncio.Event()
        reconciliation = self.reconcile(discovered)
        metadata: asyncio.Task[TaskModelProjection] | None = None
        try:
            done, _ = await asyncio.wait({pending, prepared}, return_when=asyncio.FIRST_COMPLETED)
            if pending in done:
                error = None if pending.cancelled() else pending.exception()
                if error is not None and not isinstance(error, StopAsyncIteration):
                    raise error
            await self.capture()
            if ready is not None:
                ready.set()
            async for event in self.refresh(self.models.views):
                yield event
            metadata = asyncio.create_task(reconciliation.__anext__())
            while pending is not None:
                done, _ = await asyncio.wait(
                    {pending, metadata}, return_when=asyncio.FIRST_COMPLETED,
                )
                for task in done:
                    if task.cancelled():
                        task.result()
                    error = task.exception()
                    if error is not None and not isinstance(error, (StopAsyncIteration, ExecutionStreamFailure)):
                        raise error
                if pending in done:
                    task = pending
                    pending = None
                    try:
                        event = task.result()
                    except StopAsyncIteration:
                        if self.finalizing is not None:
                            self.finalizing.set()
                    else:
                        if event.execution_id not in self.models.views:
                            discovered.set()
                        yield self.tree_event(event)
                        pending = asyncio.create_task(stream.__anext__())
                if metadata in done:
                    projection = metadata.result()
                    self.cursor = self.checkpoint()
                    yield ExecutionObservationEvent(projection, self.cursor)
                    metadata = asyncio.create_task(reconciliation.__anext__())
        except ExecutionStreamFailure as failure:
            cause = failure.cause
            raise ObservationError(
                "stream", cursor=self.cursor,
                cause_code=cause.code.value if isinstance(cause, AIError) else None,
                safe_details=cause.safe_details if isinstance(cause, AIError) else None,
            ) from cause
        finally:
            active_error = sys.exc_info()[1]

            async def cleanup() -> None:
                tasks = [prepared, *self.models.waits.values()]
                if pending is not None:
                    tasks.append(pending)
                if metadata is not None:
                    tasks.append(metadata)
                errors = await drain_stream_tasks(tasks)
                for close in (reconciliation.aclose, self.models.close, stream.aclose):
                    try:
                        await close()
                    except BaseException as error:
                        report_observation_error(error)
                        errors.append(error)
                if errors and (active_error is None or isinstance(active_error, GeneratorExit)
                               or is_observation_cleanup(active_error)
                               or isinstance(active_error, ObservationError) and active_error.origin == "stream"):
                    error = next((value for value in errors if not isinstance(value, ExecutionStreamFailure)), errors[0])
                    if isinstance(error, ExecutionStreamFailure):
                        raise ObservationError("stream", cursor=self.cursor, safe_details={"phase": "cleanup"}) from error.cause
                    raise error

            await await_stream_cleanup(cleanup(), active_error)

    async def final(self) -> AsyncIterator[ExecutionObservationEvent]:
        try:
            captured = await self.capture(subscribe=False)
            for execution_id in self.models.views:
                await self.models.capture(execution_id)
        except BaseException as error:
            try:
                await await_stream_cleanup(self.models.close(), error)
            except BaseException as cleanup_error:
                report_observation_error(cleanup_error)
                if isinstance(cleanup_error, asyncio.CancelledError) and not is_observation_cleanup(cleanup_error):
                    raise
            raise

        async def deliver() -> AsyncIterator[ExecutionObservationEvent]:
            stream = self.owner.replay_tree(
                captured, principal=self.principal, after_event_seqs=self.sequences,
                include_content=self.include_content,
            )
            try:
                async for event in stream:
                    yield self.tree_event(event)
                async for event in self.refresh(self.models.views, finite=True):
                    yield event
                if any(
                    not self.models.boundaries[view.execution_id].durable_history_available
                    for view, _, _ in captured
                ):
                    raise ObservationError(
                        "stream", cursor=self.cursor,
                        safe_details={"phase": "drain", "reason": "model_state_coverage_unavailable"},
                    )
            finally:
                active_error = sys.exc_info()[1]

                async def cleanup() -> None:
                    errors = await drain_stream_tasks(self.models.waits.values())
                    for close in (stream.aclose, self.models.close):
                        try:
                            await close()
                        except BaseException as error:
                            report_observation_error(error)
                            errors.append(error)
                    if errors and (active_error is None or isinstance(active_error, GeneratorExit)
                                   or is_observation_cleanup(active_error)
                                   or isinstance(active_error, ObservationError) and active_error.origin == "stream"):
                        raise errors[0]

                await await_stream_cleanup(cleanup(), active_error)
        return deliver()

    async def watch(self, ready: asyncio.Event | None) -> AsyncIterator[ExecutionObservationEvent]:
        live = self.live(ready)
        final: AsyncIterator[ExecutionObservationEvent] | None = None
        try:
            async for event in live:
                yield event
            if ready is None:
                if self.finalizing is not None:
                    self.finalizing.set()
                # A new compensation pass also covers identities preceding the cursor.
                final_owner = _ExecutionModelObservation(
                    self.owner, self.execution_id, self.principal, self.watch_tree,
                    self.cursor, self.include_content,
                )
                final = await final_owner.final()
                async for event in final:
                    self.cursor = event.cursor
                    yield event
        finally:
            active_error = sys.exc_info()[1]

            async def cleanup() -> None:
                await live.aclose()
                if final is not None:
                    await final.aclose()

            await await_stream_cleanup(cleanup(), active_error)
