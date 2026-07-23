#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""LateBoundRunDispatcher: the runtime's dispatch binding seam.

The subagent executor is built before the AgentEngine it eventually delegates
to exists -- the runner depends on the capability resolver, which depends on
the subagent executor: a genuine self-reference, not an accidental cycle. This
handle confines that one-time forward reference to a single bind-once seam
instead of a bare closure; every caller only ever sees the narrow
``RunDispatcher`` Protocol, never the runner or build-kernel internals."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..run.dispatch import RunDispatchRequest, RunDispatcher
    from ..run.models import RunResult


class LateBoundRunDispatcher:
    """A RunDispatcher bound to its real target after construction. Built by the
    build kernel, handed to the SubagentExecutor, and bound to the real runner
    once the runner exists."""

    def __init__(self) -> None:
        self._target: "RunDispatcher | None" = None

    def bind(self, target: "RunDispatcher") -> None:
        self._target = target

    async def dispatch(self, request: "RunDispatchRequest") -> "RunResult":
        if self._target is None:
            raise RuntimeError(
                "LateBoundRunDispatcher.dispatch() called before bind() -- "
                "the build kernel must bind the real dispatcher before any "
                "subagent execution can occur"
            )
        return await self._target.dispatch(request)


__all__: "list[str]" = ["LateBoundRunDispatcher"]
