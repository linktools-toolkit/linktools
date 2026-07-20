#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CallableTaskHandler: wrap a plain async callable as a TaskHandler."""

from collections.abc import Awaitable, Callable

from ..models import TaskFailureKind
from ..protocols import TaskContext, TaskFailure, TaskRequest, TaskSuccess


class CallableTaskHandler:
    """Adapt ``async def fn(request, context) -> Any`` to the TaskHandler
    contract. A return value becomes a TaskSuccess; a raised exception becomes
    a typed TaskFailure."""

    def __init__(
        self,
        fn: "Callable[[TaskRequest, TaskContext], Awaitable[object]]",
        *,
        permanent_exceptions: "tuple[type[Exception], ...]" = (ValueError, TypeError),
    ) -> None:
        self._fn = fn
        self._permanent = permanent_exceptions

    async def execute(
        self, request: TaskRequest, context: TaskContext
    ) -> "TaskSuccess | TaskFailure":
        try:
            result = await self._fn(request, context)
        except self._permanent as exc:
            return TaskFailure(
                kind=TaskFailureKind.PERMANENT,
                error_type=type(exc).__name__,
                message=str(exc),
            )
        except Exception as exc:  # noqa: BLE001
            return TaskFailure(
                kind=TaskFailureKind.TRANSIENT,
                error_type=type(exc).__name__,
                message=str(exc),
            )
        if isinstance(result, TaskSuccess):
            return result
        if isinstance(result, TaskFailure):
            return result
        return TaskSuccess(metadata={"result": result} if result is not None else {})


__all__: "list[str]" = ["CallableTaskHandler"]
