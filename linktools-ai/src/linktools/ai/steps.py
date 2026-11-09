#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Native step records with explicitly owned Runtime lifecycle semantics."""

from dataclasses import dataclass
from typing import Literal

from pydantic_ai_harness.step_persistence import StepEvent


LifecycleOutcome = Literal["succeeded", "interrupted", "failed", "cancelled"]


@dataclass(kw_only=True)
class NativeLifecycleEvent(StepEvent):
    """One native terminal body, without modifying shared run metadata."""

    outcome: LifecycleOutcome

    def __post_init__(self) -> None:
        kind = {"succeeded": "run_completed", "interrupted": "run_completed",
                "failed": "run_failed", "cancelled": "run_failed"}.get(self.outcome)
        if (kind is None or self.kind != kind or self.idempotency_key is None
                or self.tool_call_id is not None or self.tool_name is not None
                or (kind == "run_completed" and self.error is not None)):
            raise ValueError("native lifecycle outcome does not match its event")

    @classmethod
    def source_key(cls) -> str:
        """The first shared terminal observation of this single-shot native run."""
        return "terminal"

    @property
    def event_type(self) -> Literal["AGENT_RUN_SUCCEEDED", "AGENT_RUN_INTERRUPTED", "AGENT_RUN_FAILED"]:
        if self.outcome == "succeeded":
            return "AGENT_RUN_SUCCEEDED"
        if self.outcome == "interrupted":
            return "AGENT_RUN_INTERRUPTED"
        return "AGENT_RUN_FAILED"


__all__ = ["LifecycleOutcome", "NativeLifecycleEvent"]
