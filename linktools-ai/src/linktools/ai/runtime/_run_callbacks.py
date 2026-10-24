#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Borrow SDK-bound callbacks for one Runtime agent run."""

from __future__ import annotations

from contextlib import contextmanager
from copy import copy
from typing import Any, TYPE_CHECKING

from pydantic_ai.capabilities import AbstractCapability, CombinedCapability, WrapperCapability, CapabilityOrdering

from ..capability import ReferencedInputsCapability, ToolResultsCapability
from ..errors import AIError, ErrorCode

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from ..spec import CapabilityRef
    from pydantic_ai.tools import RunContext, ToolDefinition
    from pydantic_ai.agent import AbstractAgent
    from pydantic_ai.run import AgentRunResult
    from pydantic_ai.messages import ToolCallPart, ModelResponse
    from pydantic_ai.models import ModelRequestContext
    from pydantic_ai.capabilities import WrapRunHandler, WrapModelRequestHandler, WrapToolExecuteHandler


class RunCallbacks:
    """Run-local borrowed references, never a lifecycle or persistence owner."""

    def __init__(self) -> None:
        self._tools: dict[str, list[ToolResultsCapability[Any]]] = {}
        self._inputs: dict[CapabilityRef, tuple[AbstractCapability[Any], ReferencedInputsCapability[Any]]] = {}
        self.context: RunContext[Any] | None = None

    @contextmanager
    def tool_scope(self, call_id: str, processor: ToolResultsCapability[Any]) -> Iterator[None]:
        processors = self._tools.setdefault(call_id, [])
        processors.append(processor)
        try:
            yield
        finally:
            processors.pop()
            if not processors:
                del self._tools[call_id]

    def tool_results(self, call_id: str) -> tuple[ToolResultsCapability[Any], ...]:
        return tuple(self._tools.get(call_id, ()))

    def bind_input(self, source: CapabilityRef, proxy: AbstractCapability[Any],
                   loader: ReferencedInputsCapability[Any]) -> None:
        if source in self._inputs:
            raise AIError(ErrorCode.CAPABILITY_CONFLICT)
        self._inputs[source] = proxy, loader

    def discard_input(self, source: CapabilityRef, proxy: AbstractCapability[Any]) -> None:
        current = self._inputs.get(source)
        if current is not None and current[0] is proxy:
            del self._inputs[source]

    def referenced_inputs(self) -> dict[CapabilityRef, ReferencedInputsCapability[Any]]:
        if self.context is None:
            return {}
        active = self.context.active_capability_ids
        proxies = {id(value) for key, value in self.context.capabilities.items() if key in active}
        return {source: loader for source, (proxy, loader) in self._inputs.items() if id(proxy) in proxies}


class RunCallbackScope(AbstractCapability[Any]):
    """Expose the current native request context without retaining its messages."""

    def __init__(self, callbacks: RunCallbacks) -> None:
        self.callbacks = callbacks

    def get_ordering(self) -> CapabilityOrdering:
        return CapabilityOrdering(position="outermost")

    async def wrap_run(self, ctx: RunContext[Any], *, handler: WrapRunHandler) -> AgentRunResult[Any]:
        previous, self.callbacks.context = self.callbacks.context, ctx
        try:
            return await handler()
        finally:
            self.callbacks.context = previous

    async def wrap_model_request(self, ctx: RunContext[Any], *, request_context: ModelRequestContext,
                                 handler: WrapModelRequestHandler) -> ModelResponse:
        previous, self.callbacks.context = self.callbacks.context, ctx
        try:
            return await handler(request_context)
        finally:
            self.callbacks.context = previous


class _CallbackAdapter(WrapperCapability[Any]):
    def __init__(self, wrapped: AbstractCapability[Any], source: CapabilityRef, callbacks: RunCallbacks,
                 *, bound: bool = False) -> None:
        super().__init__(wrapped)
        self.bound = bound
        self.source = source
        self.callbacks = callbacks

    def apply(self, visitor: Callable[[AbstractCapability[Any]], None]) -> None:
        self.wrapped.apply(visitor)

    def visit_and_replace(self, visitor: Callable[[AbstractCapability[Any]], AbstractCapability[Any] | None]
                          ) -> AbstractCapability[Any] | None:
        replacement = self.wrapped.visit_and_replace(visitor)
        if replacement is None:
            if self.bound:
                self.callbacks.discard_input(self.source, self.wrapped)
            return None
        if replacement is self.wrapped:
            return self
        if self.bound:
            self.callbacks.discard_input(self.source, self.wrapped)
        return bind_run_callbacks(replacement, self.source, self.callbacks, bound=self.bound)

    def get_ordering(self) -> CapabilityOrdering | None:
        return self.wrapped.get_ordering()

    def for_agent(self, agent: AbstractAgent[Any, Any]) -> AbstractCapability[Any]:
        rebound = super().for_agent(agent)
        return bind_run_callbacks(rebound.wrapped, self.source, self.callbacks)

    async def for_run(self, ctx: RunContext[Any]) -> AbstractCapability[Any]:
        rebound = await super().for_run(ctx)
        return bind_run_callbacks(rebound.wrapped, self.source, self.callbacks, bound=True)

    async def wrap_tool_execute(
        self, ctx: RunContext[Any], *, call: ToolCallPart, tool_def: ToolDefinition,
        args: dict[str, Any], handler: WrapToolExecuteHandler,
    ) -> Any:
        leaf = self.wrapped
        while isinstance(leaf, WrapperCapability):
            leaf = leaf.wrapped
        if isinstance(leaf, ToolResultsCapability):
            with self.callbacks.tool_scope(call.tool_call_id, leaf):
                return await super().wrap_tool_execute(ctx, call=call, tool_def=tool_def, args=args, handler=handler)
        return await super().wrap_tool_execute(ctx, call=call, tool_def=tool_def, args=args, handler=handler)


def bind_run_callbacks(capability: AbstractCapability[Any], source: CapabilityRef, callbacks: RunCallbacks,
                       *, bound: bool = False) -> AbstractCapability[Any]:
    """Adapt only callback proxies while preserving native container structure."""
    if isinstance(capability, CombinedCapability):
        return capability.visit_and_replace(lambda child: bind_run_callbacks(child, source, callbacks, bound=bound))
    leaf = capability
    while isinstance(leaf, WrapperCapability):
        leaf = leaf.wrapped
    if isinstance(leaf, CombinedCapability):
        rebound = copy(capability)
        rebound.wrapped = bind_run_callbacks(capability.wrapped, source, callbacks, bound=bound)
        return rebound
    if isinstance(leaf, (ToolResultsCapability, ReferencedInputsCapability)):
        if bound and isinstance(leaf, ReferencedInputsCapability):
            callbacks.bind_input(source, capability, leaf)
        return _CallbackAdapter(capability, source, callbacks, bound=bound)
    return capability


__all__ = ["RunCallbacks", "RunCallbackScope", "bind_run_callbacks"]
