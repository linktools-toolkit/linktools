#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""PeriodicReminderCapability: injects a one-time system reminder once the
conversation's message count crosses a configurable fraction of a soft budget."""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import ModelRequest, SystemPromptPart

if TYPE_CHECKING:
    from pydantic_ai import RunContext
    from pydantic_ai.models import ModelRequestContext


@dataclass
class PeriodicReminderCapability(AbstractCapability[None]):
    max_messages: int = 80
    threshold_ratio: float = 0.7
    _already_reminded: bool = field(default=False, repr=False, compare=False)

    async def before_model_request(
        self,
        ctx: "RunContext[Any]",
        request_context: "ModelRequestContext",
    ) -> "ModelRequestContext":
        if not self._already_reminded and self.max_messages > 0:
            occupancy = len(request_context.messages) / self.max_messages
            if occupancy >= self.threshold_ratio:
                self._already_reminded = True
                request_context.messages.append(
                    ModelRequest(
                        parts=[
                            SystemPromptPart(
                                content=(
                                    "Reminder: this conversation is approaching its context "
                                    "budget. Wrap up outstanding work and summarize where "
                                    "things stand if you have not already."
                                )
                            )
                        ]
                    )
                )
        return request_context
