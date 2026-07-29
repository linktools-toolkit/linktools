"""Execution backend contract and composed Store."""

from typing import Protocol

from .commands import (
    AbortExecution,
    AcknowledgeCancellation,
    ClaimExecution,
    CompleteExecution,
    DecideApproval,
    FailExecution,
    HeartbeatExecution,
    PauseExecution,
    RequestCancellation,
    ResumeExecution,
    StartExecution,
)
from ..json import JsonValue
from .domain import Page, RunRecord
from .evaluation import RunEvaluation
from .session import SessionRecord, SessionTurn
from .snapshots import AgentSnapshotData, RunSnapshot
from .trace_models import NewRunTraceStep, RunEvent, RunTraceStep


class ExecutionBackend(Protocol):
    async def create_session(self, *, session_id: str, user_id: str | None, tenant_id: str | None) -> SessionRecord: ...
    async def get_session(self, session_id: str) -> SessionRecord | None: ...
    async def list_session_turns(self, session_id: str, *, before_sequence: int | None = None, limit: int = 50) -> Page: ...
    async def load_session_context(self, session_id: str) -> tuple[JsonValue, ...]: ...
    async def get_run(self, run_id: str) -> RunRecord | None: ...
    async def start_run(self, command: StartExecution) -> RunRecord: ...
    async def claim_run(self, command: ClaimExecution) -> RunRecord: ...
    async def heartbeat_run(self, command: HeartbeatExecution) -> RunRecord: ...
    async def pause_run(self, command: PauseExecution) -> RunRecord: ...
    async def decide_approval(self, command: DecideApproval) -> RunRecord: ...
    async def resume_run(self, command: ResumeExecution) -> RunRecord: ...
    async def request_cancel(self, command: RequestCancellation) -> RunRecord: ...
    async def complete_run(self, command: CompleteExecution) -> RunRecord: ...
    async def fail_run(self, command: FailExecution) -> RunRecord: ...
    async def acknowledge_cancel(self, command: AcknowledgeCancellation) -> RunRecord: ...
    async def abort_run(self, command: AbortExecution) -> RunRecord: ...
    async def append_trace_steps(self, run_id: str, *, expected_sequence: int, steps: tuple[NewRunTraceStep, ...]) -> int: ...
    async def list_trace_steps(self, run_id: str, *, after_sequence: int = 0, through_sequence: int | None = None) -> tuple[RunTraceStep, ...]: ...
    async def get_snapshot(self, run_id: str) -> RunSnapshot | None: ...
    async def list_run_events(self, run_id: str, *, after_sequence: int = 0, limit: int = 100) -> Page: ...
    async def save_evaluation(self, evaluation: RunEvaluation) -> None: ...
    async def list_evaluations(self, run_id: str) -> tuple[RunEvaluation, ...]: ...


class ExecutionStore:
    def __init__(self, backend: ExecutionBackend) -> None:
        self._backend = backend

    @property
    def backend(self) -> ExecutionBackend:
        return self._backend

    async def initialize_storage(self, *args: object) -> None:
        await self._backend.initialize_storage(*args)

    async def create_session(self, *, session_id: str, user_id: str | None, tenant_id: str | None) -> SessionRecord:
        return await self._backend.create_session(
            session_id=session_id,
            user_id=user_id,
            tenant_id=tenant_id,
        )

    async def get_session(self, session_id: str) -> SessionRecord | None:
        return await self._backend.get_session(session_id)

    async def list_session_turns(self, session_id: str, *, before_sequence: int | None = None, limit: int = 50) -> Page:
        return await self._backend.list_session_turns(
            session_id,
            before_sequence=before_sequence,
            limit=limit,
        )

    async def load_session_context(self, session_id: str) -> tuple[JsonValue, ...]:
        return await self._backend.load_session_context(session_id)

    async def get_run(self, run_id: str) -> RunRecord | None:
        return await self._backend.get_run(run_id)

    async def start_run(self, command: StartExecution) -> RunRecord:
        return await self._backend.start_run(command)

    async def claim_run(self, command: ClaimExecution) -> RunRecord:
        return await self._backend.claim_run(command)

    async def heartbeat_run(self, command: HeartbeatExecution) -> RunRecord:
        return await self._backend.heartbeat_run(command)

    async def pause_run(self, command: PauseExecution) -> RunRecord:
        return await self._backend.pause_run(command)

    async def decide_approval(self, command: DecideApproval) -> RunRecord:
        return await self._backend.decide_approval(command)

    async def resume_run(self, command: ResumeExecution) -> RunRecord:
        return await self._backend.resume_run(command)

    async def request_cancel(self, command: RequestCancellation) -> RunRecord:
        return await self._backend.request_cancel(command)

    async def complete_run(self, command: CompleteExecution) -> RunRecord:
        return await self._backend.complete_run(command)

    async def fail_run(self, command: FailExecution) -> RunRecord:
        return await self._backend.fail_run(command)

    async def acknowledge_cancel(self, command: AcknowledgeCancellation) -> RunRecord:
        return await self._backend.acknowledge_cancel(command)

    async def abort_run(self, command: AbortExecution) -> RunRecord:
        return await self._backend.abort_run(command)

    async def append_trace_steps(self, run_id: str, *, expected_sequence: int, steps: tuple[NewRunTraceStep, ...]) -> int:
        return await self._backend.append_trace_steps(
            run_id,
            expected_sequence=expected_sequence,
            steps=steps,
        )

    async def list_trace_steps(self, run_id: str, *, after_sequence: int = 0, through_sequence: int | None = None) -> tuple[RunTraceStep, ...]:
        return await self._backend.list_trace_steps(
            run_id,
            after_sequence=after_sequence,
            through_sequence=through_sequence,
        )

    async def get_snapshot(self, run_id: str) -> RunSnapshot | None:
        return await self._backend.get_snapshot(run_id)

    async def list_run_events(self, run_id: str, *, after_sequence: int = 0, limit: int = 100) -> Page:
        return await self._backend.list_run_events(
            run_id,
            after_sequence=after_sequence,
            limit=limit,
        )

    async def save_evaluation(self, evaluation: RunEvaluation) -> None:
        await self._backend.save_evaluation(evaluation)

    async def list_evaluations(self, run_id: str) -> tuple[RunEvaluation, ...]:
        return await self._backend.list_evaluations(run_id)


__all__ = ["ExecutionBackend", "ExecutionStore"]
