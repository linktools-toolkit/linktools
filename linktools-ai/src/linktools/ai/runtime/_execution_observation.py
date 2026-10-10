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
from ._graph_projection import _GraphModelProjection
from ._observation import (
    _await_stream_cleanup, _drain_stream_tasks, _is_observation_cleanup,
    _report_observation_error,
)
from ._watch_cursor import (
    decode_execution_observation_cursor, encode_execution_observation_cursor,
    encode_execution_watch_cursor,
)
from .service_api import ExecutionObservationEvent, ExecutionTreeEvent, ExecutionView, _ExecutionStreamFailure

if TYPE_CHECKING:
    from ._agent import _ExecutionTreeWatcher
    from ._runtime_service import Runtime


class _ExecutionModelObservation:
    def __init__(
        self, runtime: "Runtime", execution_id: str, principal: Principal,
        watch_tree: "_ExecutionTreeWatcher", cursor: str | None, include_content: bool,
        finalizing: asyncio.Event | None = None,
    ) -> None:
        self.runtime = runtime
        self.execution_id = execution_id
        self.principal = principal
        self.watch_tree = watch_tree
        self.include_content = include_content
        self.finalizing = finalizing
        self.sequences = dict({} if cursor is None else decode_execution_observation_cursor(
            runtime.namespace, principal.tenant_id, execution_id, cursor,
            include_content=include_content,
        ))
        self.cursor = cursor
        self.models = _GraphModelProjection(runtime.history, principal)

    def checkpoint(self) -> str:
        return encode_execution_observation_cursor(
            self.runtime.namespace, self.principal.tenant_id, self.execution_id,
            include_content=self.include_content, event_seqs=self.sequences,
        )

    def tree_event(self, event: ExecutionTreeEvent) -> ExecutionObservationEvent:
        sequence = event.event.durable_seq
        if sequence is not None:
            if sequence <= self.sequences.get(event.execution_id, 0):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            self.sequences[event.execution_id] = sequence
        inner = replace(event, cursor=encode_execution_watch_cursor(
            self.runtime.namespace, self.principal.tenant_id, self.execution_id,
            include_content=self.include_content, event_seqs=self.sequences,
        ))
        self.cursor = self.checkpoint()
        return ExecutionObservationEvent(inner, self.cursor)

    async def capture(self) -> tuple[tuple[ExecutionView, int, int], ...]:
        captured = await self.runtime._capture_execution_tree(
            self.execution_id, principal=self.principal, after_event_seqs=self.sequences,
        )
        for view, depth, _ in captured:
            await self.models.add(self.execution_id, view, depth)
        return captured

    async def refresh(self, execution_ids: Iterable[str], *, finite: bool = False) -> AsyncIterator[ExecutionObservationEvent]:
        for execution_id in sorted(execution_ids):
            boundary = self.models.boundaries[execution_id] if finite else None
            async for _, projection in self.models.refresh(execution_id, boundary=boundary):
                self.cursor = self.checkpoint()
                yield ExecutionObservationEvent(projection, self.cursor)

    async def live(self, ready: asyncio.Event | None) -> AsyncIterator[ExecutionObservationEvent]:
        tree_ready = asyncio.Event()
        stream = self.watch_tree(
            self.execution_id, principal=self.principal, after_event_seqs=self.sequences,
            include_content=self.include_content, ready=tree_ready,
        )
        pending = asyncio.create_task(stream.__anext__())
        prepared = asyncio.create_task(tree_ready.wait())
        tick: asyncio.Task[None] | None = None
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
            tick = asyncio.create_task(asyncio.sleep(1.0))
            while pending is not None:
                tasks = {pending, tick, *self.models.waits.values()}
                done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    if task.cancelled():
                        task.result()
                    error = task.exception()
                    if error is not None and not isinstance(error, (StopAsyncIteration, _ExecutionStreamFailure)):
                        raise error
                dirty = self.models.changed(done)
                if tick in done:
                    await self.capture()
                    dirty.update(self.models.views)
                    tick = asyncio.create_task(asyncio.sleep(1.0))
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
                            await self.capture()
                            if event.execution_id not in self.models.views:
                                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                            dirty.add(event.execution_id)
                        yield self.tree_event(event)
                        pending = asyncio.create_task(stream.__anext__())
                async for event in self.refresh(dirty):
                    yield event
        except _ExecutionStreamFailure as failure:
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
                if tick is not None:
                    tasks.append(tick)
                errors = await _drain_stream_tasks(tasks)
                for close in (self.models.close, stream.aclose):
                    try:
                        await close()
                    except BaseException as error:
                        _report_observation_error(error)
                        errors.append(error)
                if errors and (active_error is None or isinstance(active_error, GeneratorExit)
                               or _is_observation_cleanup(active_error)
                               or isinstance(active_error, ObservationError) and active_error.origin == "stream"):
                    error = next((value for value in errors if not isinstance(value, _ExecutionStreamFailure)), errors[0])
                    if isinstance(error, _ExecutionStreamFailure):
                        raise ObservationError("stream", cursor=self.cursor, safe_details={"phase": "cleanup"}) from error.cause
                    raise error

            await _await_stream_cleanup(cleanup(), active_error)

    async def final(self) -> AsyncIterator[ExecutionObservationEvent]:
        try:
            captured = await self.capture()
            for execution_id in self.models.views:
                await self.models.capture(execution_id)
        except BaseException as error:
            try:
                await _await_stream_cleanup(self.models.close(), error)
            except BaseException as cleanup_error:
                _report_observation_error(cleanup_error)
                if isinstance(cleanup_error, asyncio.CancelledError) and not _is_observation_cleanup(cleanup_error):
                    raise
            raise

        async def deliver() -> AsyncIterator[ExecutionObservationEvent]:
            stream = self.runtime._replay_execution_tree(
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
                    errors = await _drain_stream_tasks(self.models.waits.values())
                    for close in (stream.aclose, self.models.close):
                        try:
                            await close()
                        except BaseException as error:
                            _report_observation_error(error)
                            errors.append(error)
                    if errors and (active_error is None or isinstance(active_error, GeneratorExit)
                                   or _is_observation_cleanup(active_error)
                                   or isinstance(active_error, ObservationError) and active_error.origin == "stream"):
                        raise errors[0]

                await _await_stream_cleanup(cleanup(), active_error)
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
                    self.runtime, self.execution_id, self.principal, self.watch_tree,
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

            await _await_stream_cleanup(cleanup(), active_error)
