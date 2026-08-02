#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Structural execution-store contract implemented directly by backends."""


from typing import Protocol
from .domain import Page
from .session import SessionTurn
from .snapshots import RunSnapshot
from .trace_models import NewRunTraceStep, RunEvent, RunTraceStep

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..json import JsonValue
    from ..storage.database import CoordinationScope
    from .commands import AbortExecution, AcknowledgeCancellation, ClaimExecution, CompleteExecution, DecideApproval, FailExecution, HeartbeatExecution, PauseExecution, RequestCancellation, ResumeExecution, StartExecution
    from .domain import RunRecord
    from ..evaluation import RunEvaluation
    from .session import SessionRecord

class ExecutionStore(Protocol):
    coordination_scope: "CoordinationScope"

    async def create_session(
        self,
        *,
        session_id: str,
        user_id: "str | None",
        tenant_id: "str | None",
    ) -> "SessionRecord": ...

    async def get_session(self, session_id: str) -> "SessionRecord | None": ...

    async def start_run(self, command: "StartExecution") -> "RunRecord": ...

    async def get_run(self, run_id: str) -> "RunRecord | None": ...

    async def claim_run(self, command: "ClaimExecution") -> "RunRecord": ...

    async def heartbeat_run(self, command: "HeartbeatExecution") -> "RunRecord": ...

    async def request_cancel(
        self, command: "RequestCancellation"
    ) -> "RunRecord": ...

    async def pause_run(self, command: "PauseExecution") -> "RunRecord": ...

    async def resume_run(self, command: "ResumeExecution") -> "RunRecord": ...

    async def decide_approval(
        self, command: "DecideApproval"
    ) -> "RunRecord": ...

    async def complete_run(self, command: "CompleteExecution") -> "RunRecord": ...

    async def fail_run(self, command: "FailExecution") -> "RunRecord": ...

    async def acknowledge_cancel(
        self, command: "AcknowledgeCancellation"
    ) -> "RunRecord": ...

    async def abort_run(self, command: "AbortExecution") -> "RunRecord": ...

    async def append_trace_steps(
        self,
        run_id: str,
        *,
        expected_sequence: int,
        steps: "tuple[NewRunTraceStep, ...]",
    ) -> int: ...

    async def list_trace_steps(
        self,
        run_id: str,
        *,
        after_sequence: int = 0,
        through_sequence: "int | None" = None,
    ) -> "tuple[RunTraceStep, ...]": ...

    async def get_snapshot(self, run_id: str) -> "RunSnapshot | None": ...

    async def list_session_turns(
        self,
        session_id: str,
        *,
        before_sequence: "int | None" = None,
        limit: int = 50,
    ) -> "Page[SessionTurn]": ...

    async def load_session_context(
        self, session_id: str
    ) -> "tuple[JsonValue, ...]": ...
    # Session Context: COMPLETED turns' TURN_DELTA concatenated in sequence
    # order. NOT a resume source -- PAUSED/CANCELLED/FAILED deltas are excluded.

    async def load_resume_messages(
        self, execution_id: str
    ) -> "tuple[JsonValue, ...]": ...
    # Resume Context: the PAUSED run's RESUME_CHECKPOINT (all_messages() at the
    # pause point). Empty for a non-PAUSED run or one with no checkpoint.

    async def get_session_messages(
        self, session_id: str
    ) -> "tuple[SessionTurn, ...]": ...
    # Audit History: every turn's TURN_DELTA + status + capture_state, in
    # sequence order. NOT grouped/filtered -- the query layer shapes it.

    async def get_turn(
        self, session_id: str, sequence: int
    ) -> "SessionTurn | None": ...
    # O(1) single-turn read by (session_id, sequence).

    async def list_run_events(
        self, run_id: str, *, after_sequence: int = 0, limit: int = 100
    ) -> "Page[RunEvent]": ...

    async def save_evaluation(self, evaluation: "RunEvaluation") -> None: ...

    async def list_evaluations(
        self, run_id: str
    ) -> "tuple[RunEvaluation, ...]": ...


__all__ = ["ExecutionStore"]
