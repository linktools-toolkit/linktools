#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Borrow SDK-bound callbacks without owning their lifecycle."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, replace
from copy import copy
from typing import Any, TYPE_CHECKING, cast

from pydantic_ai.capabilities import AbstractCapability, CombinedCapability, WrapperCapability, CapabilityOrdering
from pydantic_ai.toolsets import AbstractToolset, ToolsetTool, WrapperToolset

from ._referenced_inputs import ReferencedInputsCapability
from ._tool_results import ToolResultsCapability
from ..errors import AIError, ErrorCode
from ..model import ReferencedContent, ReferencedInputModel

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator, Sequence
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

    def replace_inputs(
        self, bindings: Iterable[tuple[CapabilityRef, AbstractCapability[Any], ReferencedInputsCapability[Any]]],
    ) -> None:
        inputs = {}
        for source, proxy, loader in bindings:
            if source in inputs:
                raise AIError(ErrorCode.CAPABILITY_CONFLICT)
            inputs[source] = proxy, loader
        self._inputs = inputs

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
    def __init__(self, wrapped: AbstractCapability[Any], source: CapabilityRef, callbacks: RunCallbacks | None,
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
            if self.bound and self.callbacks is not None:
                self.callbacks.discard_input(self.source, self.wrapped)
            return None
        if replacement is self.wrapped:
            return self
        if self.bound and self.callbacks is not None:
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
            callbacks = self.callbacks or _NativeExecutionScope.callbacks_for(ctx)
            with callbacks.tool_scope(call.tool_call_id, leaf):
                return await super().wrap_tool_execute(ctx, call=call, tool_def=tool_def, args=args, handler=handler)
        return await super().wrap_tool_execute(ctx, call=call, tool_def=tool_def, args=args, handler=handler)


class _DeferredCallbacks(WrapperCapability[Any]):
    """Keep native deferred dispatch separate from transparent registration."""

    wrapped: CombinedCapability[Any]

    def __init__(self, adapter: _CallbackAdapter) -> None:
        super().__init__(CombinedCapability([adapter]))

    @property
    def adapter(self) -> _CallbackAdapter:
        return cast(_CallbackAdapter, self.wrapped.capabilities[0])

    def apply(self, visitor: Callable[[AbstractCapability[Any]], None]) -> None:
        self.wrapped.apply(visitor)

    def visit_and_replace(self, visitor: Callable[[AbstractCapability[Any]], AbstractCapability[Any] | None]
                          ) -> AbstractCapability[Any] | None:
        replacement = self.adapter.visit_and_replace(visitor)
        return self if replacement is self.adapter else replacement

    def for_agent(self, agent: AbstractAgent[Any, Any]) -> AbstractCapability[Any]:
        return self.adapter.for_agent(agent)

    async def for_run(self, ctx: RunContext[Any]) -> AbstractCapability[Any]:
        return await self.adapter.for_run(ctx)



def bind_run_callbacks(capability: AbstractCapability[Any], source: CapabilityRef, callbacks: RunCallbacks | None,
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
        if bound and callbacks is not None and isinstance(leaf, ReferencedInputsCapability):
            callbacks.bind_input(source, capability, leaf)
        adapter = _CallbackAdapter(capability, source, callbacks, bound=bound)
        return _DeferredCallbacks(adapter) if adapter.defer_loading is True else adapter
    return capability


@dataclass
class _NativeToolset(WrapperToolset[Any]):
    callbacks: RunCallbacks

    async def call_tool(
        self, name: str, tool_args: dict[str, Any], ctx: RunContext[Any], tool: ToolsetTool[Any],
    ) -> Any:
        processors = self.callbacks.tool_results(ctx.tool_call_id)
        result = await super().call_tool(name, tool_args, ctx, tool)
        return await ToolResultsCapability.process_results(
            processors, ctx.deps, tool.tool_def, tool_args, result,
        )


class _NativeExecutionScope(RunCallbackScope):
    """One SDK-composed callback scope shared by independent declarations."""

    id = "linktools.ai.native-capability-scope"

    def __init__(self, *, bound: bool = False) -> None:
        super().__init__(RunCallbacks())
        self.bound = bound
        self._provider_view: ReferencedInputModel | None = None

    @classmethod
    def combine(cls, capabilities: Sequence[AbstractCapability[Any]]) -> AbstractCapability[Any]:
        return cls(bound=any(capability.bound for capability in capabilities))

    async def for_run(self, ctx: RunContext[Any]) -> AbstractCapability[Any]:
        return type(self)(bound=True)

    def get_ordering(self) -> CapabilityOrdering:
        return CapabilityOrdering(position="outermost", wraps=(AbstractCapability,) if self.bound else ())

    @classmethod
    def callbacks_for(cls, ctx: RunContext[Any]) -> RunCallbacks:
        scope = ctx.capabilities.get(cls.id)
        if not isinstance(scope, cls):
            raise AIError(ErrorCode.CAPABILITY_REQUIRED_MISSING)
        return scope.callbacks

    def get_wrapper_toolset(self, toolset: AbstractToolset[Any]) -> AbstractToolset[Any]:
        return _NativeToolset(toolset, self.callbacks)

    def prepare_inputs(self, ctx: RunContext[Any]) -> None:
        def bindings(capability: AbstractCapability[Any]) -> Iterator[
            tuple[CapabilityRef, AbstractCapability[Any], ReferencedInputsCapability[Any]]
        ]:
            if isinstance(capability, _CallbackAdapter):
                leaf = capability.wrapped
                while isinstance(leaf, WrapperCapability):
                    leaf = leaf.wrapped
                if capability.callbacks is None and isinstance(leaf, ReferencedInputsCapability):
                    yield capability.source, capability.wrapped, leaf
                return
            if isinstance(capability, CombinedCapability):
                for child in capability.capabilities:
                    yield from bindings(child)
            elif isinstance(capability, WrapperCapability):
                yield from bindings(capability.wrapped)

        self.callbacks.replace_inputs(() if ctx.root_capability is None else bindings(ctx.root_capability))
        self.callbacks.context = ctx

    async def before_model_request(
        self, ctx: RunContext[Any], request_context: ModelRequestContext,
    ) -> ModelRequestContext:
        self.prepare_inputs(ctx)
        return self.prepare_model(ctx, request_context)

    def prepare_model(self, ctx: RunContext[Any], request_context: ModelRequestContext) -> ModelRequestContext:
        if request_context.model is self._provider_view:
            return request_context

        async def load(content: ReferencedContent) -> bytes:
            sources = self.callbacks.referenced_inputs()
            source = sources.get(content.source)
            if source is None:
                raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID,
                              safe_details={"reason": "referenced_input_source_mismatch"})
            return await source.load(ctx.deps, content)

        self._provider_view = ReferencedInputModel(request_context.model, load=load)
        return replace(request_context, model=self._provider_view)

    async def wrap_model_request(
        self, ctx: RunContext[Any], *, request_context: ModelRequestContext,
        handler: WrapModelRequestHandler,
    ) -> ModelResponse:
        previous = self._provider_view
        self.prepare_inputs(ctx)
        try:
            return await super().wrap_model_request(
                ctx, request_context=self.prepare_model(ctx, request_context), handler=handler,
            )
        finally:
            self._provider_view = previous


class _NativeInputGuard(AbstractCapability[Any]):
    id = "linktools.ai.native-input-guard"

    def __init__(self, *, bound: bool = False) -> None:
        self.bound = bound

    @classmethod
    def combine(cls, capabilities: Sequence[AbstractCapability[Any]]) -> AbstractCapability[Any]:
        return cls(bound=any(capability.bound for capability in capabilities))

    async def for_run(self, ctx: RunContext[Any]) -> AbstractCapability[Any]:
        return type(self)(bound=True)

    def get_ordering(self) -> CapabilityOrdering:
        return CapabilityOrdering(position="innermost", wrapped_by=(AbstractCapability,) if self.bound else ())

    async def before_model_request(
        self, ctx: RunContext[Any], request_context: ModelRequestContext,
    ) -> ModelRequestContext:
        scope = ctx.capabilities.get(_NativeExecutionScope.id)
        if not isinstance(scope, _NativeExecutionScope):
            raise AIError(ErrorCode.CAPABILITY_REQUIRED_MISSING)
        return scope.prepare_model(ctx, request_context)

    async def wrap_model_request(
        self, ctx: RunContext[Any], *, request_context: ModelRequestContext,
        handler: WrapModelRequestHandler,
    ) -> ModelResponse:
        return await handler(await self.before_model_request(ctx, request_context))


def materialize_native_capability(
    capability: AbstractCapability[Any], source: CapabilityRef,
) -> AbstractCapability[Any]:
    return CombinedCapability([
        _NativeExecutionScope(), _NativeInputGuard(), bind_run_callbacks(capability, source, None),
    ])


__all__ = ["RunCallbacks", "RunCallbackScope", "bind_run_callbacks"]
