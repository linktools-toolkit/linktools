#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluation contracts."""

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from .models import EvalCase, EvalExecution, EvalScore, EvalTarget

if TYPE_CHECKING:
    from .snapshot import RunSnapshot


@runtime_checkable
class EvalExecutor(Protocol):
    async def execute(self, target: EvalTarget, case: EvalCase) -> EvalExecution: ...


@runtime_checkable
class Evaluator(Protocol):
    """Scores an execution. The optional ``snapshot`` carries the captured
    RunSnapshot (run record / definition / events) so a trajectory/usage
    evaluator can read the run's actual behavior, not just the final output."""

    @property
    def name(self) -> str: ...

    async def evaluate(
        self,
        case: EvalCase,
        execution: EvalExecution,
        snapshot: "RunSnapshot | None" = None,
    ) -> EvalScore: ...


__all__: "list[str]" = ["EvalExecutor", "Evaluator"]
