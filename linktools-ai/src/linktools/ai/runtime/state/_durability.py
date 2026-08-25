#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Cancellation-safe coordination for one durable commit boundary."""

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Generic, TypeVar, cast

from linktools.core import environ

from ...errors import AIError, ErrorCode

ValueT = TypeVar("ValueT")
_logger = environ.get_logger("ai.runtime.state.durability")
_DETACHED_TASKS: set[asyncio.Task[object]] = set()


class DurableCommitState(StrEnum):
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
    """Resolve a durable commit without waiting indefinitely after cancellation."""
    task = asyncio.create_task(operation())
    cancellation_requested = False
    operation_error: BaseException | None = None
    value: ValueT | None = None
    try:
        value = await asyncio.shield(task)
    except asyncio.CancelledError as error:
        cancellation_requested = True
        if not task.done():
            _detach_task(cast("asyncio.Task[object]", task), "durable commit")
            return DurableCommitResult(
                DurableCommitState.UNRESOLVED,
                error=error,
                cancelled=True,
            )
        try:
            value = task.result()
        except BaseException as task_error:  # noqa: BLE001
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
    try:
        observation = await asyncio.shield(readback_task)
    except asyncio.CancelledError as error:
        cancellation_requested = True
        if not readback_task.done():
            _detach_task(
                cast("asyncio.Task[object]", readback_task),
                "durable commit readback",
            )
            return DurableCommitResult(
                DurableCommitState.UNRESOLVED,
                error=error,
                cancelled=True,
            )
        try:
            observation = readback_task.result()
        except AIError as readback_error:
            return _readback_error_result(
                readback_error,
                cancelled=True,
            )
        except BaseException as readback_error:  # noqa: BLE001
            return DurableCommitResult(
                DurableCommitState.UNRESOLVED,
                error=readback_error,
                cancelled=True,
            )
    except AIError as error:
        return _readback_error_result(
            error,
            cancelled=cancellation_requested,
        )
    except BaseException as error:  # noqa: BLE001
        return DurableCommitResult(
            DurableCommitState.UNRESOLVED,
            error=error,
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


def _detach_task(task: "asyncio.Task[object]", label: str) -> None:
    _DETACHED_TASKS.add(task)

    def consume(done: "asyncio.Task[object]") -> None:
        try:
            done.result()
        except asyncio.CancelledError:
            pass
        except BaseException:  # noqa: BLE001
            _logger.exception("detached %s failed", label)
        finally:
            _DETACHED_TASKS.discard(done)

    task.add_done_callback(consume)


__all__ = [
    "CommitObservation",
    "DurableCommitResult",
    "DurableCommitState",
    "run_durable_commit",
]
