#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Cancellation-safe coordination for one durable commit boundary."""

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import Enum
from typing import Generic, TypeVar, cast

from ...errors import AIError, ErrorCode

ValueT = TypeVar("ValueT")


class DurableCommitState(str, Enum):
    __str__ = str.__str__
    __format__ = str.__format__
    COMMITTED = "committed"
    NOT_COMMITTED = "not_committed"
    PARTIAL_INTEGRITY_ERROR = "partial_integrity_error"
    UNRESOLVED = "unresolved"


@dataclass(frozen=True, slots=True)
class CommitObservation(Generic[ValueT]):
    state: DurableCommitState
    value: ValueT | None = None
    error: BaseException | None = None


@dataclass(frozen=True, slots=True)
class DurableCommitResult(Generic[ValueT]):
    state: DurableCommitState
    value: ValueT | None = None
    error: BaseException | None = None
    cancelled: bool = False

    @property
    def committed(self) -> bool:
        return self.state is DurableCommitState.COMMITTED


async def run_durable_commit(
    operation: Callable[[], Awaitable[ValueT]],
    readback: Callable[[], Awaitable[CommitObservation[ValueT]]],
    *,
    background_tasks: "set[asyncio.Task[object]]",
) -> DurableCommitResult[ValueT]:
    """Resolve durable truth before propagating caller cancellation."""
    task = asyncio.create_task(operation())
    value, operation_error, operation_cancelled = await _await_owned_task(
        task,
        background_tasks,
    )
    cancellation_requested = operation_cancelled

    if operation_error is None:
        return DurableCommitResult(
            DurableCommitState.COMMITTED,
            value=value,
            cancelled=cancellation_requested,
        )

    readback_task = asyncio.create_task(readback())
    observation, readback_error, readback_cancelled = await _await_owned_task(
        readback_task,
        background_tasks,
    )
    cancellation_requested = cancellation_requested or readback_cancelled
    if readback_error is not None:
        if isinstance(readback_error, AIError):
            return _readback_error_result(
                readback_error,
                cancelled=cancellation_requested,
            )
        return DurableCommitResult(
            DurableCommitState.UNRESOLVED,
            error=readback_error,
            cancelled=cancellation_requested,
        )
    if not isinstance(observation, CommitObservation):
        raise TypeError("durable commit readback returned an invalid observation")
    return DurableCommitResult(
        observation.state,
        value=observation.value,
        error=observation.error or operation_error,
        cancelled=cancellation_requested,
    )


async def _await_owned_task(
    task: "asyncio.Task[ValueT]",
    background_tasks: "set[asyncio.Task[object]]",
) -> "tuple[ValueT | None, BaseException | None, bool]":
    """Keep a shielded durable task owned until its outcome is observable."""
    cancelled = False
    owned = False
    current = asyncio.current_task()
    tracked = cast("asyncio.Task[object]", task)
    try:
        while True:
            try:
                return await asyncio.shield(task), None, cancelled
            except asyncio.CancelledError as error:
                caller_cancelled = current is not None and current.cancelling() > 0
                if task.done():
                    try:
                        return task.result(), None, cancelled or caller_cancelled
                    except BaseException as task_error:  # noqa: BLE001
                        return None, task_error, cancelled or caller_cancelled
                if not caller_cancelled:
                    return None, error, cancelled
                cancelled = True
                if not owned:
                    background_tasks.add(tracked)
                    owned = True
            except BaseException as error:  # noqa: BLE001
                return None, error, cancelled
    finally:
        if owned:
            background_tasks.discard(tracked)


def _readback_error_result(
    error: AIError,
    *,
    cancelled: bool,
) -> "DurableCommitResult[ValueT]":
    return DurableCommitResult(
        DurableCommitState.PARTIAL_INTEGRITY_ERROR
        if error.code is ErrorCode.STORAGE_INTEGRITY_ERROR
        else DurableCommitState.UNRESOLVED,
        error=error,
        cancelled=cancelled,
    )


__all__ = [
    "CommitObservation",
    "DurableCommitResult",
    "DurableCommitState",
    "run_durable_commit",
]
