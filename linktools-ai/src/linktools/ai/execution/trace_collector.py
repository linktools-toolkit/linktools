"""Semantic trace collection with persisted sequence continuity."""

from dataclasses import dataclass, field
from datetime import datetime, timezone

from ..json import JsonValue
from .domain import RunStatus, RunUsage
from .snapshots import AgentSnapshotData
from .store import ExecutionStore
from .trace_models import NewRunTraceStep


@dataclass(slots=True)
class SemanticTraceCollector:
    run_id: str
    trace_port: ExecutionStore
    next_sequence: int = 0
    _pending: list[NewRunTraceStep] = field(default_factory=list)

    def _add(self, kind: str, payload: JsonValue) -> None:
        self._pending.append(NewRunTraceStep(kind=kind, payload=payload, created_at=datetime.now(timezone.utc)))

    async def model_request_succeeded(self, payload: JsonValue) -> None:
        self._add("model_interaction", payload)
        await self.flush()

    async def model_request_failed(self, payload: JsonValue) -> None:
        self._add("model_interaction", payload)
        await self.flush()

    async def tool_result(self, payload: JsonValue) -> None:
        self._add("tool_result", payload)
        await self.flush()

    async def flush(self) -> int:
        if not self._pending:
            return self.next_sequence
        steps = tuple(self._pending)
        self.next_sequence = await self.trace_port.append_trace_steps(
            self.run_id, expected_sequence=self.next_sequence, steps=steps
        )
        self._pending.clear()
        return self.next_sequence

    async def build_snapshot(self, *, resume_messages: tuple[JsonValue, ...], final_output: "JsonValue | str | None", status: RunStatus, usage: RunUsage) -> AgentSnapshotData:
        await self.flush()
        return AgentSnapshotData(resume_messages, final_output, usage, self.next_sequence)
