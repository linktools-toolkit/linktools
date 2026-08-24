#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Cancellation-safe coordination for one durable commit boundary."""

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import Enum
from typing import Generic, TypeVar

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
) -> DurableCommitResult[ValueT]:
    """Resolve a possibly interrupted commit without inferring rollback.

    The commit task is shielded from caller cancellation.  If its outcome is
    not directly observable, the supplied readback is the only authority used
    to classify the boundary.
    """
    task = asyncio.create_task(operation())
    cancellation_requested = False
    operation_error: BaseException | None = None
    value: ValueT | None = None
    try:
        value = await asyncio.shield(task)
    except asyncio.CancelledError as error:
        cancellation_requested = True
        operation_error = error
        value, task_error = await _settle_task(task)
        operation_error = task_error
    except BaseException as error:  # noqa: BLE001
        operation_error = error

    if operation_error is None:
        return DurableCommitResult(
            DurableCommitState.COMMITTED,
            value=value,
            cancelled=cancellation_requested,
        )

    readback_task = asyncio.create_task(readback())
    readback_error: BaseException | None = None
    try:
        observation = await asyncio.shield(readback_task)
    except asyncio.CancelledError:
        cancellation_requested = True
        observation, readback_error = await _settle_task(readback_task)
        if readback_error is not None:
            return DurableCommitResult(
                DurableCommitState.UNRESOLVED,
                error=readback_error,
                cancelled=cancellation_requested,
            )
    except AIError as error:
        if error.code is ErrorCode.STORAGE_INTEGRITY_ERROR:
            return DurableCommitResult(
                DurableCommitState.PARTIAL_INTEGRITY_ERROR,
                error=error,
                cancelled=cancellation_requested,
            )
        return DurableCommitResult(
            DurableCommitState.UNRESOLVED,
            error=error,
            cancelled=cancellation_requested,
        )
    except BaseException as error:  # noqa: BLE001
        return DurableCommitResult(
            DurableCommitState.UNRESOLVED,
            error=error,
            cancelled=cancellation_requested,
        )

    if readback_error is not None:
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


async def _settle_task(
    task: "asyncio.Task[ValueT]",
) -> tuple[ValueT | None, BaseException | None]:
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            continue
    try:
        return task.result(), None
    except BaseException as error:  # noqa: BLE001
        return None, error


__all__ = [
    "CommitObservation",
    "DurableCommitResult",
    "DurableCommitState",
    "run_durable_commit",
]
