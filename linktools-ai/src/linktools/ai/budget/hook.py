#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""BudgetCapability: enforce a BudgetTracker's cap via pydantic-ai's native
model-request hooks. Fallback-model retry (`fallback_models`) is out of scope
here — see the plan header this module was built from for why."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pydantic_ai.capabilities import AbstractCapability

from .tracker import BudgetTracker

if TYPE_CHECKING:
    from pydantic_ai import RunContext
    from pydantic_ai.messages import ModelResponse
    from pydantic_ai.models import ModelRequestContext


@dataclass
class BudgetCapability(AbstractCapability[None]):
    tracker: BudgetTracker

    async def before_model_request(
        self,
        ctx: "RunContext[Any]",
        request_context: "ModelRequestContext",
    ) -> "ModelRequestContext":
        self.tracker.check()
        return request_context

    async def after_model_request(
        self,
        ctx: "RunContext[Any]",
        *,
        request_context: "ModelRequestContext",
        response: "ModelResponse",
    ) -> "ModelResponse":
        self.tracker.record(response.usage)
        return response
