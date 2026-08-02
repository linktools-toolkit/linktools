#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""The persisted, resumable execution snapshot and the engine-return snapshot data."""


from dataclasses import dataclass
from typing import Literal
from ..json import JsonValue

from .domain import MessageCaptureState

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datetime import datetime
    from .domain import RunStatus, RunUsage

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


__all__: "list[str]" = ["AgentSnapshotData", "RunSnapshot"]
