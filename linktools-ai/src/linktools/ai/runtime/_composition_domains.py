#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime-owned composition of independent Task and Evaluation domains."""

from collections.abc import Awaitable, Callable, Sequence
from typing import TYPE_CHECKING, Generic, TypeVar

if TYPE_CHECKING:
    from ..asset import AssetStoreReader
    from ..evaluation_engine import Evaluations
    from ..task import Task, TaskExpander
    from ..task_engine import TaskEngine

AppT = TypeVar("AppT")


class RuntimeTasks(Generic[AppT]):
    def __init__(self, bind: Callable[..., Awaitable["TaskEngine[AppT]"]]) -> None:
        self._bind = bind

    async def bind(self, *definitions: "Task[AppT, object] | TaskExpander") -> "TaskEngine[AppT]":
        """Create one owned engine; reuse it or its definition views for later starts."""
        return await self._bind(*definitions)


class RuntimeEvaluations(Generic[AppT]):
    def __init__(self, bind: Callable[..., Awaitable["Evaluations"]]) -> None:
        self._bind = bind

    async def bind(
        self, engine: "TaskEngine[AppT]", *, assets: Sequence["AssetStoreReader"] = (),
    ) -> "Evaluations":
        """Own evaluation coordination, borrowing the given engine and its scope."""
        return await self._bind(engine, assets=assets)


__all__ = ["RuntimeTasks", "RuntimeEvaluations"]
