#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Authorized history projections for the ordinary turn store."""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from typing import TYPE_CHECKING

from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ThinkingPart, ToolCallPart

from ..core import (
    AuthorizationAction, AuthorizationPolicy, CursorSigner, ExecutionStatus, JsonValue, Page, Principal, PrincipalKind, ResourceKind,
    ResourceRef, SessionListingAuthorization, TenantAuthorizationPolicy, UsageMetrics, canonical_sha256, decode_cursor, encode_cursor,
    principal_identity_payload, validate_execution_id, validate_page_limit, validate_session_id,
)
from ..errors import AIError, ErrorCode
from ..inputs import stored_input_attachment_views
from ..model import StoredUserInput
from ..model.messages import (
    CANONICAL_OUTPUT_TOOL_NAME, THINKING_ELAPSED_MS_METADATA_KEY, canonical_structured_output,
)
from ..persistence.codec import decode_domain
from ..persistence.contracts import ModelInteractionRecord
from ..persistence.runs import ModelInteractionStateStepArchive
from ..persistence.turns import SessionTurnKey, SessionTurnSegment, SqlSessionTurnReader, decode_session_lineage
from ._projection import StepExecutionHistoryReader, _trace_payload
from ..session import SessionTurnItem
from ._session_timeline import _project_turn_tool_process

if TYPE_CHECKING:
    from sqlalchemy import RowMapping

    from ..persistence.runs import StateStepArchive

_TURN_PAGE_MAX = 64

_STATUS = {
    "accepted": "PENDING_START",
    "waiting": "WAITING_DEFERRED",
    "ready": "WAITING_DEFERRED",
    "cancelling": "CANCELLING",
    "cancelled": "CANCELLED",
    "failed": "FAILED",
    "succeeded": "SUCCEEDED",
}


@dataclass(frozen=True, slots=True)
class _ObservedTurnResult:
    key: SessionTurnKey
    generation: int
    status: ExecutionStatus
    usage: UsageMetrics | None
    view: dict[str, JsonValue]


def _session_view(row: RowMapping) -> dict[str, JsonValue]:
    config = row["config_json"]
    active = row["active_turn_id"]
    return {
        "session_id": row["session_key"].decode("utf-8"),
        "status": row["status"].upper(),
        "revision": row["revision"],
        "cwd": config.get("cwd"),
        "active_turn_id": None if active is None else active.decode("utf-8"),
        "last_turn_no": row["last_turn_no"],
        "metadata": config.get("metadata", {}),
        "history_quality": "complete",
    }


def project_ordinary_multipart_prompt(value: StoredUserInput) -> dict[str, JsonValue]:
    """Project admitted text without opening attachments or exposing provider metadata."""
    payload = value.payload
    content = payload.value
    if payload.encoding == "json-attachments" and isinstance(content, Mapping):
        content = content.get("content")
    if (value.codec != "user-content-v1" or not isinstance(content, Mapping)
            or not isinstance(items := content.get("items"), list)):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    text_items: list[JsonValue] = []
    for item in items:
        if not isinstance(item, Mapping):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        kind = item.get("kind")
        if kind == "text":
            text = item.get("text")
        elif kind in ("text-content", "structured-content"):
            metadata = item.get("metadata")
            if isinstance(metadata, Mapping) and metadata.get("source") == "linktools.attachment":
                continue
            text = item.get("content")
        else:
            continue
        if not isinstance(text, str):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        text_items.append({"kind": "text", "text": text})
    return {"kind": "items", "items": text_items}


def project_ordinary_turn_activity(
    messages: tuple[ModelMessage, ...], *, output: JsonValue, output_ref: Sequence[int] | None,
    local_tool_feedback: bool = False,
) -> dict[str, list[JsonValue]]:
    """Project the retained public timeline identically for SSE and history."""
    items: list[JsonValue] = []
    output_position = None if output_ref is None else tuple(output_ref)
    for message_index, message in enumerate(messages):
        if not isinstance(message, ModelResponse):
            continue
        durations = (message.metadata or {}).get(THINKING_ELAPSED_MS_METADATA_KEY)
        for index, part in enumerate(message.parts):
            ordinal = len(items) + 1
            if isinstance(part, TextPart):
                item = SessionTurnItem(ordinal, "assistant", part.content)
            elif isinstance(part, ThinkingPart):
                elapsed = durations.get(str(index)) if isinstance(durations, Mapping) else None
                item = SessionTurnItem(
                    ordinal, "thinking", part.content,
                    elapsed_ms=elapsed if type(elapsed) is int and elapsed >= 0 else None,
                )
            elif isinstance(part, ToolCallPart):
                if part.tool_name != CANONICAL_OUTPUT_TOOL_NAME:
                    item = SessionTurnItem(ordinal, "tool_call", None, part.tool_name, part.tool_call_id)
                elif (message_index, index) == output_position:
                    item = SessionTurnItem(ordinal, "structured_output", output)
                else:
                    continue
            else:
                continue
            items.append(asdict(item))
    process = [
        asdict(replace(
            item, error_message=item.error_message if local_tool_feedback else None,
            error_code=(item.error_code or ErrorCode.TOOL_EXECUTION_FAILED)
            if item.status == "FAILED" else item.error_code,
        ))
        for item in _project_turn_tool_process(messages) if item.tool_name != CANONICAL_OUTPUT_TOOL_NAME
    ]
    return {"items": items, "process": process}


async def _turn_view(
    row: RowMapping, read_scalar_input: Callable[[StoredUserInput], Awaitable[str]], *, local_tool_feedback: bool = False,
) -> dict[str, JsonValue]:
    state = row["state"]
    result = row["result_json"]
    output = None
    activity: dict[str, list[JsonValue]] = {"items": [], "process": []}
    if state == "succeeded":
        messages = row["messages"]
        output_ref = result["output_ref"] if "output_ref" in result else None
        if "output_ref" in result:
            output = canonical_structured_output(messages, output_ref)
        else:
            if not messages or not isinstance(messages[-1], ModelResponse) or messages[-1].text is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            output = {"text": messages[-1].text}
        activity = project_ordinary_turn_activity(
            messages, output=output, output_ref=output_ref, local_tool_feedback=local_tool_feedback,
        )
    prompt = row["prompt"]
    attachments = []
    if row["stored_input"] is not None:
        stored_input = decode_domain(row["stored_input"], StoredUserInput)
        attachments = [
            {key: item[key] for key in ("attachment_id", "source", "media_type", "size", "position", "display_name")
             if key in item}
            for item in stored_input_attachment_views(stored_input)
        ]
        if prompt == "":
            prompt = (await read_scalar_input(stored_input) if stored_input.codec == "text"
                      else project_ordinary_multipart_prompt(stored_input))
    return {
        "turn_id": row["turn_id"].decode("utf-8"),
        "turn_no": row["turn_no"],
        "status": "RECOVERY_REQUIRED" if row.get("needs_recovery") else _STATUS[state],
        "can_recover": row.get("graph_execution") is None and (
            bool(row.get("needs_recovery")) or state == "ready" and row.get("split_step") is True
        ),
        "created_at": row["created_at"].isoformat(),
        "updated_at": row["updated_at"].isoformat(),
        "user_input": {"version": 1, "prompt": prompt, "files": [], "attachments": attachments},
        "conversation_committed": state in ("succeeded", "failed", "cancelled"),
        **({"abandoned": True} if isinstance(result, Mapping) and result.get("abandoned") is True else {}),
        **activity,
        "output": output,
        "error_code": result.get("error_code") if isinstance(result, Mapping) else None,
        "safe_error_details": {},
    }


@dataclass(frozen=True, slots=True)
class SessionTurnTraceItem:
    turn_id: str
    step_event_seq: int
    payload: JsonValue


class _OrdinarySessionHistory:
    def __init__(
        self, store: SqlSessionTurnReader, tenant_id: str, authorization: AuthorizationPolicy, signer: CursorSigner,
    ) -> None:
        self._store = store
        self._tenant_id = tenant_id
        self._authorization = authorization
        self._signer = signer

    def _require_access(self, principal: Principal, *, local_workspace: bool) -> None:
        if principal.tenant_id != self._tenant_id:
            raise AIError(ErrorCode.AUTHORIZATION_DENIED)
        if local_workspace and principal.kind != PrincipalKind.LOCAL_TRUSTED.value:
            raise AIError(ErrorCode.AUTHORIZATION_DENIED)
        if not local_workspace and type(self._authorization) is TenantAuthorizationPolicy:
            raise AIError(ErrorCode.AUTHORIZATION_DENIED, "Ordinary sessions require a session authorization policy")

    async def _authorize_session(self, session_id: str, principal: Principal) -> None:
        await self._authorization.authorize(
            principal, AuthorizationAction.SESSION_READ,
            ResourceRef(ResourceKind.SESSION, session_id, self._tenant_id),
        )

    async def _session(self, session_id: str, principal: Principal, *, local_workspace: bool) -> RowMapping:
        self._require_access(principal, local_workspace=local_workspace)
        validate_session_id(session_id)
        await self._authorize_session(session_id, principal)
        row = await self._store.get_session(session_id)
        if row is None:
            raise AIError(ErrorCode.AUTHORIZATION_DENIED)
        return row

    async def _authorize_lineage(
        self, row: RowMapping, principal: Principal,
    ) -> tuple[SessionTurnSegment, ...]:
        lineage = decode_session_lineage(row["lineage_json"])
        for part in lineage:
            await self._authorize_session(part.session_id, principal)
        return lineage

    def _cursor_position(self, cursor: str | None, kind: str, digest: str) -> int | None:
        if cursor is None:
            return None
        value = decode_cursor(
            cursor, self._signer, tenant_id=self._tenant_id, resource_kind=kind, filter_digest=digest,
        ).position
        if not value.isascii() or not value.isdecimal() or len(value) > 19 or int(value) < 1:
            raise AIError(ErrorCode.CURSOR_INVALID)
        return int(value)

    def _next_cursor(self, position: int, kind: str, digest: str) -> str:
        return encode_cursor(
            self._signer, tenant_id=self._tenant_id, resource_kind=kind, filter_digest=digest,
            position=str(position),
        )

    async def sessions(
        self, *, principal: Principal, cursor: str | None, limit: int,
        session_ids: Sequence[str] | None, local_workspace: bool,
    ) -> dict[str, JsonValue]:
        self._require_access(principal, local_workspace=local_workspace)
        limit = validate_page_limit(limit)
        prefix = None
        if session_ids is None:
            if not local_workspace:
                policy = self._authorization
                if not isinstance(policy, SessionListingAuthorization):
                    raise AIError(ErrorCode.AUTHORIZATION_DENIED, "Listing requires explicit session IDs")
                prefix = await policy.session_list_prefix(principal)
                if prefix is None:
                    raise AIError(ErrorCode.AUTHORIZATION_DENIED)
                validate_session_id(prefix)
            selected = None
        else:
            if isinstance(session_ids, (str, bytes)) or len(session_ids) > 1000:
                raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
            selected = tuple(sorted({validate_session_id(session_id) for session_id in session_ids}))
        kind = "ORDINARY_SESSIONS"
        digest = canonical_sha256({
            "principal": principal_identity_payload(principal), "local_workspace": local_workspace,
            "session_ids": None if selected is None else list(selected),
            **({} if prefix is None else {"session_prefix": prefix}),
        })
        before = self._cursor_position(cursor, kind, digest)
        rows = await self._store.session_page(
            session_ids=selected, prefix=prefix, before=before, limit=limit + 1,
        )
        items: list[JsonValue] = []
        for row in rows[:limit]:
            try:
                await self._authorize_session(row["session_key"].decode("utf-8"), principal)
            except AIError as error:
                if error.code is not ErrorCode.AUTHORIZATION_DENIED:
                    raise
            else:
                items.append(_session_view(row))
        next_cursor = self._next_cursor(rows[limit - 1]["id"], kind, digest) if len(rows) > limit else None
        return {"items": items, "next_cursor": next_cursor}

    async def session_state(
        self, session_id: str, *, principal: Principal, local_workspace: bool,
    ) -> dict[str, JsonValue]:
        return _session_view(await self._session(session_id, principal, local_workspace=local_workspace))

    async def session(
        self, session_id: str, *, principal: Principal, cursor: str | None, limit: int, local_workspace: bool,
    ) -> dict[str, JsonValue]:
        limit = min(validate_page_limit(limit), _TURN_PAGE_MAX)
        row = await self._session(session_id, principal, local_workspace=local_workspace)
        lineage = await self._authorize_lineage(row, principal)
        kind = "ORDINARY_TURNS"
        digest = canonical_sha256({
            "principal": principal_identity_payload(principal), "session": session_id, "local_workspace": local_workspace,
        })
        before = self._cursor_position(cursor, kind, digest)
        through = row["last_turn_no"] if before is None else min(row["last_turn_no"], before - 1)
        turns = () if through == 0 else await self._store.turn_page(session_id, through=through, limit=limit + 1, lineage=lineage, body_limit=limit)
        next_cursor = self._next_cursor(turns[limit - 1]["turn_no"], kind, digest) if len(turns) > limit else None
        return {
            "session": _session_view(row),
            "timeline": {"items": [
                await _turn_view(turn, self._store.read_scalar_input, local_tool_feedback=local_workspace)
                for turn in reversed(turns[:limit])
            ],
                         "next_cursor": next_cursor},
        }

    async def _authorize_turn_trace(
        self, session_id: str, turn_id: str, principal: Principal, *, local_workspace: bool,
    ) -> None:
        validate_execution_id(turn_id)
        await self._session(session_id, principal, local_workspace=local_workspace)
        key = SessionTurnKey(session_id, turn_id)
        owner = await self._store.get_turn(key)
        if owner is None:
            raise AIError(ErrorCode.AUTHORIZATION_DENIED)
        await self._authorization.authorize(
            principal, AuthorizationAction.EXECUTION_READ,
            ResourceRef(ResourceKind.EXECUTION, turn_id, self._tenant_id, owner["request_json"]["principal"]["principal_id"]),
        )

    async def turn_trace(
        self, session_id: str, turn_id: str, *, principal: Principal, local_workspace: bool,
        source: StateStepArchive | None, cursor: str | None, limit: int,
    ) -> Page[SessionTurnTraceItem]:
        await self._authorize_turn_trace(session_id, turn_id, principal, local_workspace=local_workspace)
        if source is None:
            raise AIError(ErrorCode.STORAGE_CAPABILITY_MISSING)
        limit = validate_page_limit(limit)
        digest = canonical_sha256({
            "session": session_id, "turn": turn_id, "principal": principal_identity_payload(principal),
        })
        kind = "SESSION_TURN_TRACE"
        head = timestamp = None
        sequence = 0
        if cursor is not None:
            encoded = decode_cursor(cursor, self._signer, tenant_id=self._tenant_id,
                                    resource_kind=kind, filter_digest=digest).position
            try:
                head, time, sequence = json.loads(encoded)
                timestamp = datetime.fromisoformat(time)
                if timestamp.tzinfo is None or type(head) is not int or type(sequence) is not int or not 0 < sequence <= head:
                    raise ValueError("invalid trace cursor")
            except (TypeError, ValueError) as error:
                raise AIError(ErrorCode.CURSOR_INVALID) from error
        head, events = await source.list_turn_trace_events(
            turn_id=turn_id, high_water=head, after_timestamp=timestamp, after_sequence=sequence, limit=limit + 1,
        )
        items = []
        for ordinal, event in events[:limit]:
            if event.agent_run_id != turn_id or event.agent_conversation_id != session_id:
                raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)
            payload = _trace_payload(event, agent_run_seq=1, depth=0)
            if payload is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            items.append(SessionTurnTraceItem(turn_id, ordinal, payload))
        next_cursor = None
        if len(events) > limit:
            last_sequence, last_event = events[limit - 1]
            next_cursor = encode_cursor(
                self._signer, tenant_id=self._tenant_id, resource_kind=kind, filter_digest=digest,
                position=json.dumps((head, last_event.timestamp.isoformat(), last_sequence), separators=(",", ":")),
            )
        return Page(tuple(items), next_cursor)

    async def turn_model_interactions(
        self, session_id: str, turn_id: str, *, principal: Principal, local_workspace: bool,
        source: StateStepArchive | None, cursor: str | None, limit: int,
        include_content: bool, model_request_seq: int | None,
    ) -> Page[dict[str, JsonValue]]:
        await self._authorize_turn_trace(session_id, turn_id, principal, local_workspace=local_workspace)
        if include_content and not local_workspace:
            raise AIError(ErrorCode.AUTHORIZATION_DENIED)
        limit = validate_page_limit(limit)
        kind = "SESSION_TURN_MODELS"
        digest = None
        if model_request_seq is None:
            digest = canonical_sha256({
                "session": session_id, "turn": turn_id, "principal": principal_identity_payload(principal),
                "include_content": include_content,
            })
            after = self._cursor_position(cursor, kind, digest) if cursor is not None else 0
            read_limit = limit + 1
        else:
            if type(model_request_seq) is not int or not 0 < model_request_seq <= (1 << 63) - 1 or cursor is not None:
                raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
            after = model_request_seq - 1
            read_limit = 1
        if not isinstance(source, ModelInteractionStateStepArchive):
            return Page((), None)
        run, rows = await source.read_model_interactions_with_run(
            agent_run_id=turn_id, after_model_request_seq=after,
            limit=read_limit,
        )
        if run is None:
            if rows:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            return Page((), None)
        if run.agent_run_id != turn_id or run.agent_conversation_id != session_id:
            raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)
        previous = after
        for value in rows:
            if (not isinstance(value, ModelInteractionRecord) or value.agent_run_id != turn_id
                    or value.model_request_seq <= previous
                    or model_request_seq is not None and value.model_request_seq != model_request_seq):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            previous = value.model_request_seq
        selected = rows[:limit]
        resolved = (await source.resolve_model_interactions(selected) if include_content else [None] * len(selected))
        items = []
        for value, body in zip(selected, resolved, strict=True):
            projected = asdict(StepExecutionHistoryReader._project_model_interaction(
                value, turn_id, 1, 0, body, include_content=include_content,
            ))
            del projected["execution_id"]
            items.append({"turn_id": turn_id, **projected})
        next_cursor = (self._next_cursor(selected[-1].model_request_seq, kind, digest)
                       if digest is not None and len(rows) > limit else None)
        return Page(tuple(items), next_cursor)

    async def turn_result(
        self, session_id: str, turn_id: str, *, principal: Principal, local_workspace: bool,
        producer_stopped_generation: int | None = None,
    ) -> dict[str, JsonValue]:
        result = await self._read_turn_result(session_id, turn_id, principal=principal, local_workspace=local_workspace)
        if (producer_stopped_generation is not None and result.generation == producer_stopped_generation
                and result.status in (ExecutionStatus.PENDING_START, ExecutionStatus.CANCELLING)):
            raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)
        return result.view

    async def _read_turn_result(
        self, session_id: str, turn_id: str, *, principal: Principal, local_workspace: bool,
        for_execution: bool = False,
    ) -> _ObservedTurnResult:
        validate_execution_id(turn_id)
        session = await self._session(session_id, principal, local_workspace=local_workspace)
        lineage = await self._authorize_lineage(session, principal)
        row = await self._store.get_history_turn(
            SessionTurnKey(session_id, turn_id), lineage=lineage, include_authority=for_execution,
        )
        if row is None:
            raise AIError(ErrorCode.AUTHORIZATION_DENIED)
        # A fork may read an ancestor's committed turn. Keep its actual owner ID.
        key = SessionTurnKey(row["session_key"].decode("utf-8"), row["turn_id"].decode("utf-8"))
        usage = None
        if for_execution:
            owner = row["owner_principal_id"]
            if not isinstance(owner, str) or not owner:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            await self._authorization.authorize(
                principal, AuthorizationAction.EXECUTION_READ,
                ResourceRef(ResourceKind.EXECUTION, key.turn_id, self._tenant_id, owner),
            )
            usage = row.get("usage")
        view = await _turn_view(row, self._store.read_scalar_input, local_tool_feedback=local_workspace)
        return _ObservedTurnResult(key, row["generation"], ExecutionStatus(view["status"]), usage, view)
