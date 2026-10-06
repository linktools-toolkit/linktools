#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Call-local observation lifetimes retained by the owning Runtime."""

import asyncio
import math
from collections.abc import Callable, Coroutine
from typing import Any, TypeVar

from linktools.core import environ

from ..errors import AIError, ErrorCode, TaskObservationError

_CLEANUP_CANCEL = object()

def _is_observation_cleanup(error: BaseException | None) -> bool:
    return (
        isinstance(error, asyncio.CancelledError)
        and len(error.args) == 1
        and error.args[0] is _CLEANUP_CANCEL
    )

def _cancel_stream_task(task: asyncio.Task[Any]) -> None:
    task.cancel(_CLEANUP_CANCEL)


T = TypeVar("T")
_logger = environ.get_logger("ai.runtime.task.observation")


async def _await_stream_cleanup(
    cleanup: Coroutine[Any, Any, None], active_error: BaseException | None,
) -> None:
    """Finish owned cleanup even if cancellation arrives while it is unwinding."""
    task = asyncio.create_task(cleanup, name="task-stream-cleanup")
    cancellation: asyncio.CancelledError | None = None
    while True:
        try:
            await asyncio.shield(task)
            break
        except asyncio.CancelledError as error:
            if task.cancelled():
                raise
            if cancellation is None or not _is_observation_cleanup(error):
                cancellation = error
        except BaseException:
            if cancellation is not None and not _is_observation_cleanup(cancellation):
                raise cancellation
            raise
    if cancellation is not None and (
        not _is_observation_cleanup(cancellation)
        or active_error is None or isinstance(active_error, GeneratorExit)
    ):
        raise cancellation


def _validate_timeout(value: float | None, *, positive: bool = False) -> None:
    if value is None and not positive:
        return
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or (value <= 0 if positive else value < 0)
    ):
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID)


class _ObservationSession:
    def __init__(
        self, graph_id: str, cursor: str | None, close_timeout: float,
        release: Callable[["_ObservationSession"], None],
    ) -> None:
        self.graph_id = graph_id
        self.cursor = cursor
        self.close_timeout = close_timeout
        self.closing = False
        self.observer_cancellation: asyncio.CancelledError | None = None
        self.tasks: set[asyncio.Task[Any]] = set()
        self.cancelled_by_owner: set[asyncio.Task[Any]] = set()
        self._release = release

    def track(self, task: asyncio.Task[T]) -> None:
        self.tasks.add(task)
        task.add_done_callback(self._done)
        if self.closing:
            self._cancel(task)

    def start(self, coroutine: Coroutine[Any, Any, T], *, name: str) -> asyncio.Task[T]:
        task = asyncio.create_task(coroutine, name=name)
        self.track(task)
        return task

    def _done(self, task: asyncio.Task[Any]) -> None:
        if not task.cancelled():
            task.exception()
        if self.closing and all(task.done() for task in self.tasks):
            self._release(self)

    def _cancel(self, task: asyncio.Task[Any]) -> None:
        if not task.done() and task not in self.cancelled_by_owner:
            self.cancelled_by_owner.add(task)
            task.cancel(_CLEANUP_CANCEL)

    def seal(self) -> None:
        self.closing = True

    async def close(self, *, deadline: float | None = None) -> None:
        self.seal()
        loop = asyncio.get_running_loop()
        if deadline is None:
            deadline = loop.time() + self.close_timeout
        while True:
            pending = {task for task in self.tasks if not task.done()}
            if not pending:
                self._release(self)
                return
            for task in pending:
                self._cancel(task)
            remaining = max(0.0, deadline - loop.time())
            if remaining == 0:
                raise self.cleanup_error()
            await asyncio.wait(pending, timeout=remaining, return_when=asyncio.FIRST_COMPLETED)

    def cleanup_error(self) -> TaskObservationError:
        return TaskObservationError(
            "stream", cursor=self.cursor,
            safe_details={"phase": "cleanup", "cleanup_pending": True, "graph_id": self.graph_id},
        )

    async def wait(
        self,
        observe: Coroutine[Any, Any, None],
        wait: Coroutine[Any, Any, T],
        timeout_seconds: float | None,
    ) -> tuple[T, TaskObservationError | None]:
        loop = asyncio.get_running_loop()
        deadline = None if timeout_seconds is None else loop.time() + timeout_seconds
        observer = self.start(observe, name=f"task-observe-{self.graph_id}")
        waiter = self.start(wait, name=f"task-wait-{self.graph_id}")
        observation_error: TaskObservationError | None = None
        primary: BaseException | None = None
        cleanup_error: BaseException | None = None
        try:
            active = {observer, waiter}
            while True:
                remaining = None if deadline is None else max(0.0, deadline - loop.time())
                done, _ = await asyncio.wait(active, timeout=remaining, return_when=asyncio.FIRST_COMPLETED)
                if observer in done:
                    active.discard(observer)
                    try:
                        observer.result()
                    except TaskObservationError as error:
                        if error.origin != "stream" or error.safe_details.get("phase") == "cleanup":
                            raise
                        observation_error = error
                if waiter in done:
                    break
                if not done or (deadline is not None and loop.time() >= deadline):
                    raise AIError(ErrorCode.TASK_WAIT_TIMEOUT, safe_details={"graph_id": self.graph_id})
        except BaseException as error:
            primary = error
        finally:
            self.seal()
            try:
                await self.close()
            except BaseException as error:
                cleanup_error = error

        # Inspect both tasks after bounded cleanup so a simultaneous authoritative
        # failure cannot be hidden by observer completion or a successful wait.
        errors: list[tuple[int, BaseException]] = []
        if self.observer_cancellation is not None:
            errors.append((0, self.observer_cancellation))
        if primary is not None:
            errors.append((_error_priority(primary, authoritative=False), primary))
        for task, authoritative in ((waiter, True), (observer, False)):
            if not task.done() or (task.cancelled() and task in self.cancelled_by_owner):
                continue
            try:
                task.result()
            except TaskObservationError as error:
                if not authoritative and error.origin == "stream" and error.safe_details.get("phase") != "cleanup":
                    observation_error = error
                else:
                    errors.append((_error_priority(error, authoritative=authoritative), error))
            except BaseException as error:
                errors.append((_error_priority(error, authoritative=authoritative), error))
        if cleanup_error is not None:
            errors.append((0 if isinstance(cleanup_error, asyncio.CancelledError) else 4, cleanup_error))
        if any(priority == 4 for priority, _ in errors) and any(priority < 4 for priority, _ in errors):
            _logger.warning("task observation cleanup pending: graph_id=%s", self.graph_id)
        if errors:
            errors.sort(key=lambda item: item[0])
            raise errors[0][1]
        return waiter.result(), observation_error

def _error_priority(error: BaseException, *, authoritative: bool) -> int:
    if isinstance(error, asyncio.CancelledError):
        return 0
    if authoritative:
        return 1
    if isinstance(error, TaskObservationError):
        if error.origin == "callback":
            return 2
        if error.safe_details.get("phase") == "cleanup":
            return 4
    if isinstance(error, AIError) and error.code is ErrorCode.TASK_WAIT_TIMEOUT:
        return 3
    return 1
