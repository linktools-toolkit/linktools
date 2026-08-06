#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluation gateway adapter boundary."""

from typing import Protocol

from ..core import Principal
from ..runtime.services import EvaluationView


class EvaluationGateway(Protocol):
    async def inspect_evaluation(self, evaluation_id: str, *, principal: Principal) -> EvaluationView: ...


__all__ = ["EvaluationGateway"]
