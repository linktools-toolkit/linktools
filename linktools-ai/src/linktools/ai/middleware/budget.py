#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""BudgetMiddleware: records model usage against a BudgetTracker after each
model call, then checks the cap. Adapts the existing pre-vNext BudgetTracker
(budget/tracker.py, untouched by this plan) to the new Middleware Protocol."""

from typing import Any

from ..budget.tracker import BudgetTracker
from .base import Middleware


class BudgetMiddleware(Middleware):
    def __init__(self, *, tracker: BudgetTracker) -> None:
        self._tracker = tracker

    async def after_model(self, context: Any, response: Any) -> Any:
        self._tracker.record(response.usage)
        self._tracker.check()
        return response
