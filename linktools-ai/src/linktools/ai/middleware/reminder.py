#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ReminderMiddleware: appends a one-shot reminder message once the running
message count crosses threshold_ratio * max_messages. Ports
periodic_reminder/capability.py's PeriodicReminderCapability logic to the
Middleware Protocol (that pre-vNext module is untouched by this plan)."""

from typing import Any

from .base import Middleware


class ReminderMiddleware(Middleware):
    def __init__(
        self,
        *,
        max_messages: int = 80,
        threshold_ratio: float = 0.7,
        reminder_text: str,
    ) -> None:
        self._max_messages = max_messages
        self._threshold_ratio = threshold_ratio
        self._reminder_text = reminder_text
        self._already_reminded = False

    async def before_model(self, context: Any, request: Any) -> Any:
        if (
            not self._already_reminded
            and len(request.messages) / self._max_messages >= self._threshold_ratio
        ):
            request.messages.append(self._reminder_text)
            self._already_reminded = True
        return request
