"""Authorized stable DTO queries over the execution port."""

from __future__ import annotations

from dataclasses import dataclass

from ..errors import PrincipalAccessDeniedError
from ..governance.identity import PrincipalContext
from ..json import JsonValue
from .domain import Page, RunRecord, RunStatus, RunUsage
from .store import ExecutionStore


@dataclass(frozen=True, slots=True)
class SessionTurnView:
    session_id: str
    sequence: int
    run_id: str
    input: JsonValue
    assistant_summary: JsonValue | None
    status: RunStatus


@dataclass(frozen=True, slots=True)
class ModelInteractionView:
    sequence: int
    request: JsonValue
    response: JsonValue | None
    status: str


@dataclass(frozen=True, slots=True)
class ToolCallView:
    call_id: str
    tool_name: str
    arguments: JsonValue
    result: JsonValue | None
    status: str


@dataclass(frozen=True, slots=True)
class ExecutionDetailView:
    run_id: str
    session_id: str
    status: RunStatus
    effective_input: JsonValue | None
    interactions: tuple[ModelInteractionView, ...]
    tool_calls: tuple[ToolCallView, ...]
    final_output: JsonValue | None
    usage: RunUsage | None


@dataclass(frozen=True, slots=True)
class ExecutionResultView:
    run_id: str
    output: JsonValue | None


class ExecutionQueryService:
    def __init__(self, store: ExecutionStore) -> None:
        self._store = store

    @staticmethod
    def _authorize(record: RunRecord, principal: PrincipalContext) -> None:
        if record.tenant_id != principal.tenant_id or (record.user_id is not None and record.user_id != principal.user_id):
            raise PrincipalAccessDeniedError("execution is not visible to this principal")

    async def list_session_turns(self, *, session_id: str, principal: PrincipalContext, before_sequence: int | None = None, limit: int = 50) -> Page[SessionTurnView]:
        session = await self._store.get_session(session_id)
        if session is None or session.tenant_id != principal.tenant_id or (session.user_id is not None and session.user_id != principal.user_id):
            raise PrincipalAccessDeniedError("session is not visible to this principal")
        page = await self._store.list_session_turns(session_id, before_sequence=before_sequence, limit=limit)
        return Page(tuple(SessionTurnView(item.session_id, item.sequence, item.run_id, item.input, item.assistant_summary, item.status) for item in page.items), page.has_more, page.next_cursor)

    async def get_run_detail(self, *, run_id: str, principal: PrincipalContext) -> ExecutionDetailView:
        run = await self._store.get_run(run_id)
        if run is None:
            raise PrincipalAccessDeniedError("run is not visible to this principal")
        self._authorize(run, principal)
        snapshot = await self._store.get_snapshot(run_id)
        steps = await self._store.list_trace_steps(run_id, through_sequence=snapshot.trace_end_sequence if snapshot else None)
        interactions: list[ModelInteractionView] = []
        calls: dict[str, ToolCallView] = {}
        for step in steps:
            payload = step.payload
            if step.kind == "model_interaction":
                interactions.append(ModelInteractionView(step.sequence, payload.get("request", {}), payload.get("response"), payload.get("status", "completed")))
                response = payload.get("response") or {}
                for part in response.get("parts", ()):
                    if part.get("type") == "tool_call":
                        calls[part["call_id"]] = ToolCallView(part["call_id"], part["tool_name"], part.get("arguments"), None, "pending")
            elif step.kind == "tool_result":
                call_id = payload.get("call_id")
                if call_id in calls:
                    previous = calls[call_id]
                    calls[call_id] = ToolCallView(previous.call_id, previous.tool_name, previous.arguments, payload.get("result"), payload.get("status", "completed"))
        return ExecutionDetailView(run.id, run.session_id, run.status, run.input, tuple(interactions), tuple(calls.values()), snapshot.final_output if snapshot else None, snapshot.usage if snapshot else None)


__all__ = ["ExecutionQueryService", "ModelInteractionView", "ExecutionDetailView", "ExecutionResultView", "SessionTurnView", "ToolCallView"]
