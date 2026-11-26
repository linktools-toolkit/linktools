"""Agent-selected projection of provider-bound messages."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from copy import deepcopy
from dataclasses import replace
from typing import Generic, TypeVar

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models import ModelRequestContext
from pydantic_ai.tools import RunContext

from ._context import AgentContext


AppT = TypeVar("AppT")


class ProviderMessagesCapability(AbstractCapability[AgentContext[AppT]], Generic[AppT]):
    """Project provider messages without rewriting the admitted transcript.

    Register this as a revisioned Agent-scoped capability. The trusted host
    callback receives the admitted context, a detached copy of the request
    messages, and the names of the available function tools. It may filter or
    rewrite the provider view, but must not perform effects: it can run again
    when the runtime issues another model request. It does not change tool
    registration, model settings, or the original inputs seen by tools/history.
    """

    def __init__(
        self,
        project: Callable[
            [AgentContext[AppT], list[ModelMessage], tuple[str, ...]], Sequence[ModelMessage],
        ],
    ) -> None:
        self._project = project

    async def before_model_request(
        self, ctx: RunContext[AgentContext[AppT]], request_context: ModelRequestContext,
    ) -> ModelRequestContext:
        messages = self._project(
            ctx.deps, deepcopy(list(request_context.messages)),
            tuple(tool.name for tool in request_context.model_request_parameters.function_tools),
        )
        return replace(request_context, messages=list(messages))
