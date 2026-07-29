"""Execution backend contract and composed Store."""

from typing import Protocol

from .commands import (
    AcknowledgeRunCancel,
    ClaimRun,
    CompleteRun,
    DecideRunApproval,
    FailRun,
    HeartbeatRun,
    PauseRun,
    RequestRunCancel,
    ResumeRun,
    StartRun,
)
from ..storage.composition import StorageComposition
from .models import (
    JsonValue,
    NewRunTraceStep,
    Page,
    RunEvaluation,
    RunEvent,
    RunSnapshot,
    RunTraceStep,
    SessionRecord,
    SessionTurn,
)
from .run import RunRecord


class ExecutionBackend(Protocol):
    async def create_session(self, *, session_id: str, user_id: str | None, tenant_id: str | None) -> SessionRecord: ...
    async def get_session(self, session_id: str) -> SessionRecord | None: ...
    async def list_session_turns(self, session_id: str, *, before_sequence: int | None = None, limit: int = 50) -> Page: ...
    async def load_session_context(self, session_id: str) -> tuple[JsonValue, ...]: ...
    async def get_run(self, run_id: str) -> RunRecord | None: ...
    async def start_run(self, command: StartRun) -> RunRecord: ...
    async def claim_run(self, command: ClaimRun) -> RunRecord: ...
    async def heartbeat_run(self, command: HeartbeatRun) -> RunRecord: ...
    async def pause_run(self, command: PauseRun) -> RunRecord: ...
    async def decide_approval(self, command: DecideRunApproval) -> RunRecord: ...
    async def resume_run(self, command: ResumeRun) -> RunRecord: ...
    async def request_cancel(self, command: RequestRunCancel) -> RunRecord: ...
    async def complete_run(self, command: CompleteRun) -> RunRecord: ...
    async def fail_run(self, command: FailRun) -> RunRecord: ...
    async def acknowledge_cancel(self, command: AcknowledgeRunCancel) -> RunRecord: ...
    async def append_trace_steps(self, run_id: str, *, expected_sequence: int, steps: tuple[NewRunTraceStep, ...]) -> int: ...
    async def list_trace_steps(self, run_id: str, *, after_sequence: int = 0, through_sequence: int | None = None) -> tuple[RunTraceStep, ...]: ...
    async def get_snapshot(self, run_id: str) -> RunSnapshot | None: ...
    async def list_run_events(self, run_id: str, *, after_sequence: int = 0, limit: int = 100) -> Page: ...
    async def save_evaluation(self, evaluation: RunEvaluation) -> None: ...
    async def list_evaluations(self, run_id: str) -> tuple[RunEvaluation, ...]: ...


class ExecutionStore:
    def __init__(self, backend: ExecutionBackend) -> None:
        self._storage = StorageComposition(primary=backend)

    @property
    def backend(self) -> ExecutionBackend:
        return self._storage.primary

    async def initialize_storage(self, *args: object) -> None:
        await self._storage.initialize(*args)

    async def create_session(self, *, session_id: str, user_id: str | None, tenant_id: str | None) -> SessionRecord:
        return await self._storage.primary.create_session(
            session_id=session_id,
            user_id=user_id,
            tenant_id=tenant_id,
        )

    async def get_session(self, session_id: str) -> SessionRecord | None:
        return await self._storage.primary.get_session(session_id)

    async def list_session_turns(self, session_id: str, *, before_sequence: int | None = None, limit: int = 50) -> Page:
        return await self._storage.primary.list_session_turns(
            session_id,
            before_sequence=before_sequence,
            limit=limit,
        )

    async def load_session_context(self, session_id: str) -> tuple[JsonValue, ...]:
        return await self._storage.primary.load_session_context(session_id)

    async def get_run(self, run_id: str) -> RunRecord | None:
        return await self._storage.primary.get_run(run_id)

    async def start_run(self, command: StartRun) -> RunRecord:
        return await self._storage.primary.start_run(command)

    async def claim_run(self, command: ClaimRun) -> RunRecord:
        return await self._storage.primary.claim_run(command)

    async def heartbeat_run(self, command: HeartbeatRun) -> RunRecord:
        return await self._storage.primary.heartbeat_run(command)

    async def pause_run(self, command: PauseRun) -> RunRecord:
        return await self._storage.primary.pause_run(command)

    async def decide_approval(self, command: DecideRunApproval) -> RunRecord:
        return await self._storage.primary.decide_approval(command)

    async def resume_run(self, command: ResumeRun) -> RunRecord:
        return await self._storage.primary.resume_run(command)

    async def request_cancel(self, command: RequestRunCancel) -> RunRecord:
        return await self._storage.primary.request_cancel(command)

    async def complete_run(self, command: CompleteRun) -> RunRecord:
        return await self._storage.primary.complete_run(command)

    async def fail_run(self, command: FailRun) -> RunRecord:
        return await self._storage.primary.fail_run(command)

    async def acknowledge_cancel(self, command: AcknowledgeRunCancel) -> RunRecord:
        return await self._storage.primary.acknowledge_cancel(command)

    async def append_trace_steps(self, run_id: str, *, expected_sequence: int, steps: tuple[NewRunTraceStep, ...]) -> int:
        return await self._storage.primary.append_trace_steps(
            run_id,
            expected_sequence=expected_sequence,
            steps=steps,
        )

    async def list_trace_steps(self, run_id: str, *, after_sequence: int = 0, through_sequence: int | None = None) -> tuple[RunTraceStep, ...]:
        return await self._storage.primary.list_trace_steps(
            run_id,
            after_sequence=after_sequence,
            through_sequence=through_sequence,
        )

    async def get_snapshot(self, run_id: str) -> RunSnapshot | None:
        return await self._storage.primary.get_snapshot(run_id)

    async def list_run_events(self, run_id: str, *, after_sequence: int = 0, limit: int = 100) -> Page:
        return await self._storage.primary.list_run_events(
            run_id,
            after_sequence=after_sequence,
            limit=limit,
        )

    async def save_evaluation(self, evaluation: RunEvaluation) -> None:
        await self._storage.primary.save_evaluation(evaluation)

    async def list_evaluations(self, run_id: str) -> tuple[RunEvaluation, ...]:
        return await self._storage.primary.list_evaluations(run_id)


__all__ = ["ExecutionBackend", "ExecutionStore"]
