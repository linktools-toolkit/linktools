#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""The persisted, resumable execution snapshot and the engine-return snapshot data."""


from dataclasses import dataclass, field
from decimal import Decimal
from typing import Literal
from ..json import JsonValue

from .domain import MessageCaptureState, RunUsage

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datetime import datetime
    from .domain import RunStatus


@dataclass(slots=True)
class RunUsageCapture:
    """Per-run usage sink that preserves the latest confirmed model usage."""

    latest: "RunUsage" = field(default_factory=lambda: RunUsage())
    cost_known: bool = True

    def observe_absolute(self, usage: object) -> None:
        input_tokens = int(getattr(usage, "input_tokens", 0))
        output_tokens = int(getattr(usage, "output_tokens", 0))
        cache_write = int(getattr(usage, "cache_write_tokens", 0))
        cache_read = int(getattr(usage, "cache_read_tokens", 0))
        raw_cost = getattr(usage, "total_cost", None)
        if raw_cost is None:
            self.cost_known = False
        self.latest = RunUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            cache_write_tokens=cache_write,
            cache_read_tokens=cache_read,
            total_cost=Decimal(str(raw_cost)) if raw_cost is not None else None,
        )

    def snapshot(self) -> "RunUsage":
        if self.cost_known:
            return self.latest
        return RunUsage(
            input_tokens=self.latest.input_tokens,
            output_tokens=self.latest.output_tokens,
            total_tokens=self.latest.total_tokens,
            cache_write_tokens=self.latest.cache_write_tokens,
            cache_read_tokens=self.latest.cache_read_tokens,
            total_cost=None,
        )

@dataclass(frozen=True, slots=True)
class AgentSnapshotData:
    # TURN_DELTA material: this run's new_messages(), window-policy immune.
    # Persisted on the turn row at every terminal state (audit record).
    delta_messages: "tuple[JsonValue, ...]"
    final_output: "JsonValue | None"
    usage: "RunUsage"
    trace_end_sequence: int
    # Honesty marker for the turn's delta (see MessageCaptureState).
    capture_state: MessageCaptureState = MessageCaptureState.COMPLETE
    # RESUME_CHECKPOINT material: all_messages() -- the exact post-window-policy
    # pause context. Non-empty ONLY for PAUSED; the store clears it on every
    # terminal state so no cumulative history is retained (O(N^2) eliminated).
    checkpoint_messages: "tuple[JsonValue, ...]" = ()


@dataclass(frozen=True, slots=True)
class RunSnapshot:
    schema: "Literal['run-snapshot.v1']"
    run_id: str
    revision: int
    # RESUME_CHECKPOINT (column name kept): non-empty only for PAUSED; the store
    # clears it on every terminal state so steady-state snapshots hold no
    # cumulative history.
    resume_messages: "tuple[JsonValue, ...]"
    final_output: "JsonValue | None"
    status: "RunStatus"
    usage: "RunUsage"
    trace_end_sequence: int
    created_at: "datetime"


__all__: "list[str]" = ["AgentSnapshotData", "RunSnapshot", "RunUsageCapture"]
