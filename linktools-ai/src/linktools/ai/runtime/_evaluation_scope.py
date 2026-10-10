#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Call-local resources for one durable evaluation submission."""

import asyncio
from collections.abc import Iterator
from contextlib import AbstractAsyncContextManager, contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, TypeVar

from ..core import Principal
from ..evaluation import TargetTrialRef
from ..task import TaskGraphSubmission

if TYPE_CHECKING:
    from ._tasks import TaskEngine

AppT = TypeVar("AppT")


@dataclass(frozen=True, slots=True)
class EvaluationTrialScope:
    """Stable logical identity passed to a trial resource context manager."""

    experiment_id: str
    trial: TargetTrialRef
    slot_id: str
    principal: Principal
    submission: TaskGraphSubmission
    scorer_slot_id: str | None = None


class EvaluationTrialScopeCallback(Protocol[AppT]):
    def __call__(self, scope: EvaluationTrialScope) -> AbstractAsyncContextManager["TaskEngine[AppT]"]: ...


class _EnteredTrialScope:
    """Keep context entry and exit in the same owned task."""

    def __init__(self, manager: AbstractAsyncContextManager["TaskEngine"]) -> None:
        self._manager = manager
        self._entered: asyncio.Future["TaskEngine"] = asyncio.get_running_loop().create_future()
        self._exit_requested = asyncio.Event()
        self._entry_error_observed = False
        self._borrowers = 0
        self._idle = asyncio.Event()
        self._idle.set()
        self._lifetime = asyncio.create_task(self._run(), name="evaluation-scope")

    async def _run(self) -> None:
        try:
            async with self._manager as engine:
                self._entered.set_result(engine)
                await self._exit_requested.wait()
                await self._idle.wait()
        except BaseException as error:
            if not self._entered.done():
                self._entered.set_exception(error)
            raise

    async def engine(self) -> "TaskEngine":
        try:
            if not self._entered.done():
                await asyncio.wait((self._entered,))
            return self._entered.result()
        except Exception:
            if not self._exit_requested.is_set():
                self._entry_error_observed = True
            raise

    @property
    def closing(self) -> bool:
        return self._exit_requested.is_set()

    @contextmanager
    def borrow_engine(self) -> Iterator["TaskEngine | None"]:
        if self._exit_requested.is_set() or not self._entered.done() or self._entered.exception() is not None:
            yield None
            return
        self._borrowers += 1
        self._idle.clear()
        try:
            yield self._entered.result()
        finally:
            self._borrowers -= 1
            if not self._borrowers:
                self._idle.set()

    def request_close(self) -> None:
        if not self._exit_requested.is_set():
            self._exit_requested.set()
            if not self._entered.done():
                self._lifetime.cancel()

    async def close(self) -> None:
        self.request_close()
        try:
            if not self._lifetime.done():
                await asyncio.wait((self._lifetime,))
            self._lifetime.result()
        except asyncio.CancelledError:
            if not self._lifetime.cancelled():
                raise
            if not self._entered.done():
                self._entered.set_exception(asyncio.CancelledError())
            elif self._entered.exception() is None:
                raise
        except Exception:
            if not self._entry_error_observed:
                raise
        finally:
            if self._entered.done():
                self._entered.exception()


__all__ = ["EvaluationTrialScope", "EvaluationTrialScopeCallback"]
