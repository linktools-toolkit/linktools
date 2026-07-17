#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DelegationEvaluator: a core hook for SubAgent delegation.

Scores whether a run that was expected to delegate actually produced sub-agent
activity, read from the captured RunSnapshot's event stream. This is the
business-neutral core hook: rich delegation scoring (was the RIGHT sub-agent
chosen, was context passed correctly, did permissions shrink, was output
reused) is a business-layer concern that builds on the snapshot this evaluator
reads."""

from ..models import EvalCase, EvalExecution, EvalScore


class DelegationEvaluator:
    @property
    def name(self) -> str:
        return "delegation"

    async def evaluate(
        self,
        case: EvalCase,
        execution: EvalExecution,
        snapshot: "object | None" = None,
    ) -> EvalScore:
        expected = case.metadata.get("expects_delegation")
        if expected is None:
            # No delegation expectation declared -- neutral.
            return EvalScore(evaluator_name=self.name, score=1.0)
        # Delegation produces events in the run's stream; the snapshot carries
        # them as event artifacts. Presence is the core signal; business layers
        # inspect event payloads for delegation correctness.
        event_count = len(snapshot.event_artifact_ids) if snapshot is not None else 0
        delegated = event_count > 0
        score = 1.0 if bool(expected) == delegated else 0.0
        return EvalScore(
            evaluator_name=self.name,
            score=score,
            details={"expected_delegation": bool(expected), "delegated": delegated},
        )


__all__: "list[str]" = ["DelegationEvaluator"]
