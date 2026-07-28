"""Authorized stable DTO queries over the trusted execution store."""

from dataclasses import dataclass
from typing import Any

from ..errors import PrincipalAccessDeniedError
from ..identity.principal import PrincipalContext
from .models import Page, RunRecord, RunStatus, SessionTurn
from .store import ExecutionStore


@dataclass(frozen=True, slots=True)
class SessionTurnView:
    session_id: str
    sequence: int
    run_id: str
    user_prompt: Any
    assistant_summary: Any
    status: RunStatus


@dataclass(frozen=True, slots=True)
class ModelInteractionView:
    sequence: int
    request: Any
    response: Any
    status: str | None


@dataclass(frozen=True, slots=True)
class ToolCallView:
    call_id: str
    tool_name: str
    arguments: Any
    result: Any | None


@dataclass(frozen=True, slots=True)
class RunDetailView:
    run_id: str
    session_id: str
    status: RunStatus
    effective_prompt: str
    interactions: tuple[ModelInteractionView, ...]
    tool_calls: tuple[ToolCallView, ...]
    final_output: Any
    usage: Any


class ExecutionQueryService:
    def __init__(self, store: ExecutionStore) -> None:
        self._store = store

    async def _authorize(self, record: RunRecord, principal: PrincipalContext) -> None:
        if record.tenant_id != principal.tenant_id or (record.user_id is not None and record.user_id != principal.user_id):
            raise PrincipalAccessDeniedError("execution is not visible to this principal")

    async def list_session_turns(self, *, session_id: str, principal: PrincipalContext, before_sequence: int | None = None, limit: int = 50) -> Page:
        session = await self._store.get_session(session_id)
        if session is None or session.tenant_id != principal.tenant_id or (session.user_id is not None and session.user_id != principal.user_id):
            raise PrincipalAccessDeniedError("session is not visible to this principal")
        page = await self._store.list_session_turns(session_id, before_sequence=before_sequence, limit=limit)
        return Page(tuple(SessionTurnView(t.session_id, t.sequence, t.run_id, t.user_prompt, t.assistant_summary, t.status) for t in page.items), page.has_more, page.next_cursor)

    async def get_run_detail(self, *, run_id: str, principal: PrincipalContext) -> RunDetailView:
        run = await self._store.get_run(run_id)
        if run is None:
            raise PrincipalAccessDeniedError("run is not visible to this principal")
        await self._authorize(run, principal)
        snapshot = await self._store.get_snapshot(run_id)
        steps = await self._store.list_trace_steps(run_id, through_sequence=snapshot.trace_end_sequence if snapshot else None)
        interactions = tuple(ModelInteractionView(s.sequence, s.payload.get("request"), s.payload.get("response"), s.payload.get("status")) for s in steps if s.kind == "model_interaction")
        calls: dict[str, ToolCallView] = {}
        for step in steps:
            payload = step.payload
            call_id = payload.get("call_id")
            if not call_id:
                continue
            if step.kind == "model_interaction" and payload.get("tool_calls"):
                for call in payload["tool_calls"]:
                    calls[call["call_id"]] = ToolCallView(call["call_id"], call["tool_name"], call.get("arguments"), None)
            elif step.kind == "tool_result" and call_id in calls:
                previous = calls[call_id]
                calls[call_id] = ToolCallView(previous.call_id, previous.tool_name, previous.arguments, payload.get("result"))
        prompt = ""
        if snapshot is not None:
            for message in snapshot.resume_messages:
                if isinstance(message, dict) and message.get("role") == "user":
                    prompt = str(message.get("content", ""))
                    break
        return RunDetailView(run.id, run.session_id, run.status, prompt, interactions, tuple(calls.values()), snapshot.final_output if snapshot else None, snapshot.usage if snapshot else None)
