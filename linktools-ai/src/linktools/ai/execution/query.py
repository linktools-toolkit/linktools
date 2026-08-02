#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Authorized stable DTO queries over the execution port."""


from dataclasses import dataclass
from ..errors import PrincipalAccessDeniedError, StorageCorruptionError
from ..governance.authorization import AuthorizationPolicy, ExecutionAction, OwnershipAuthorizationPolicy
from .domain import MessageCaptureState, Page, RunUsage

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..governance.identity import PrincipalContext
    from ..json import JsonValue
    from .domain import RunRecord, RunStatus
    from .store import ExecutionStore

@dataclass(frozen=True, slots=True)
class SessionTurnView:
    session_id: str
    sequence: int
    run_id: str
    input: "JsonValue"
    status: "RunStatus"
    capture_state: MessageCaptureState


@dataclass(frozen=True, slots=True)
class RunMessagesView:
    # Audit-history DTO: one turn's TURN_DELTA + status + capture_state. Not a
    # flatten target -- callers must respect capture_state (PARTIAL/UNAVAILABLE
    # deltas may be incomplete) and not treat every view as resumable context.
    run_id: str
    session_id: str
    turn_sequence: int
    status: "RunStatus"
    capture_state: MessageCaptureState
    messages: "tuple[JsonValue, ...]"


@dataclass(frozen=True, slots=True)
class ModelInteractionView:
    sequence: int
    request: "JsonValue"
    response: "JsonValue | None"
    status: str


@dataclass(frozen=True, slots=True)
class ToolCallView:
    call_id: str
    tool_name: str
    arguments: "JsonValue"
    result: "JsonValue | None"
    status: str


@dataclass(frozen=True, slots=True)
class ExecutionDetailView:
    run_id: str
    session_id: str
    status: "RunStatus"
    effective_input: "JsonValue | None"
    interactions: "tuple[ModelInteractionView, ...]"
    tool_calls: "tuple[ToolCallView, ...]"
    final_output: "JsonValue | None"
    usage: "RunUsage | None"


@dataclass(frozen=True, slots=True)
class ExecutionResultView:
    run_id: str
    output: "JsonValue | None"


class ExecutionQueryService:
    def __init__(
        self,
        store: "ExecutionStore",
        authorization: "AuthorizationPolicy | None" = None,
    ) -> None:
        self._store = store
        self._authorization = authorization or OwnershipAuthorizationPolicy()

    def _authorize(self, record: "RunRecord", principal: "PrincipalContext") -> None:
        self._authorization.assert_execution_access(
            principal=principal,
            tenant_id=record.tenant_id,
            user_id=record.user_id,
            action=ExecutionAction.INSPECT,
        )

    async def list_session_turns(self, *, session_id: str, principal: "PrincipalContext", before_sequence: "int | None" = None, limit: int = 50) -> "Page[SessionTurnView]":
        session = await self._store.get_session(session_id)
        if session is None:
            raise PrincipalAccessDeniedError("session is not visible to this principal")
        self._authorization.assert_execution_access(
            principal=principal,
            tenant_id=session.tenant_id,
            user_id=session.user_id,
            action=ExecutionAction.INSPECT,
        )
        page = await self._store.list_session_turns(session_id, before_sequence=before_sequence, limit=limit)
        return Page(tuple(SessionTurnView(item.session_id, item.sequence, item.run_id, item.input, item.status, item.capture_state) for item in page.items), page.has_more, page.next_cursor)

    async def get_run_detail(self, *, run_id: str, principal: "PrincipalContext") -> ExecutionDetailView:
        run = await self._store.get_run(run_id)
        if run is None:
            raise PrincipalAccessDeniedError("run is not visible to this principal")
        self._authorize(run, principal)
        snapshot = await self._store.get_snapshot(run_id)
        steps = await self._store.list_trace_steps(run_id, through_sequence=snapshot.trace_end_sequence if snapshot else None)
        interactions: "list[ModelInteractionView]" = []
        calls: "dict[str, ToolCallView]" = {}
        tool_results: "dict[str, JsonValue]" = {}
        for step in steps:
            payload = step.payload
            if step.kind == "model_interaction":
                status = payload.get("status")
                if not isinstance(status, str):
                    raise StorageCorruptionError(
                        "model interaction trace is missing status"
                    )
                interactions.append(ModelInteractionView(step.sequence, payload.get("request", {}), payload.get("response"), status))
                response = payload.get("response") or {}
                for part in response.get("parts", ()):
                    if part.get("type") == "tool_call":
                        call_id = part["call_id"]
                        candidate = ToolCallView(
                            call_id,
                            part["tool_name"],
                            part.get("arguments"),
                            None,
                            "pending",
                        )
                        existing = calls.get(call_id)
                        if existing is not None and existing != candidate:
                            raise StorageCorruptionError(
                                "conflicting duplicate model tool call"
                            )
                        calls[call_id] = candidate
            elif step.kind == "tool_result":
                call_id = payload.get("tool_call_id")
                if not isinstance(call_id, str) or call_id not in calls:
                    raise StorageCorruptionError(
                        "tool result has no matching tool call"
                    )
                status = payload.get("status")
                if not isinstance(status, str):
                    raise StorageCorruptionError(
                        "tool result trace is missing status"
                    )
                previous_payload = tool_results.get(call_id)
                if previous_payload is not None:
                    if previous_payload != payload:
                        raise StorageCorruptionError(
                            "conflicting duplicate tool result trace"
                        )
                    continue
                tool_results[call_id] = payload
                previous = calls[call_id]
                calls[call_id] = ToolCallView(previous.call_id, previous.tool_name, previous.arguments, payload.get("result"), status)
        return ExecutionDetailView(run.id, run.session_id, run.status, run.input, tuple(interactions), tuple(calls.values()), snapshot.final_output if snapshot else None, snapshot.usage if snapshot else None)

    async def get_run_messages(
        self, *, run_id: str, principal: "PrincipalContext"
    ) -> "tuple[JsonValue, ...]":
        # TURN_DELTA for the run's turn (new_messages(), window-policy immune).
        # A run with no session_turn_sequence (non-USER_TURN) has no turn delta.
        run = await self._store.get_run(run_id)
        if run is None:
            raise PrincipalAccessDeniedError("run is not visible to this principal")
        self._authorize(run, principal)
        if run.session_turn_sequence is None:
            return ()
        turn = await self._store.get_turn(run.session_id, run.session_turn_sequence)
        return () if turn is None else turn.delta_messages

    async def get_session_messages(
        self, *, session_id: str, principal: "PrincipalContext"
    ) -> "tuple[RunMessagesView, ...]":
        # Audit History: every turn (any status) grouped per turn, with status
        # + capture_state. NOT flattened -- a PARTIAL/UNAVAILABLE view must not
        # be mistaken for resumable context.
        session = await self._store.get_session(session_id)
        if session is None:
            raise PrincipalAccessDeniedError("session is not visible to this principal")
        self._authorization.assert_execution_access(
            principal=principal,
            tenant_id=session.tenant_id,
            user_id=session.user_id,
            action=ExecutionAction.INSPECT,
        )
        turns = await self._store.get_session_messages(session_id)
        return tuple(
            RunMessagesView(t.run_id, t.session_id, t.sequence, t.status, t.capture_state, t.delta_messages)
            for t in turns
        )


__all__ = ["ExecutionQueryService", "ModelInteractionView", "ExecutionDetailView", "ExecutionResultView", "RunMessagesView", "SessionTurnView", "ToolCallView"]
