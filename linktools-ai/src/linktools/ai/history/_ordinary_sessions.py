#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Authorized history projections for the ordinary turn store."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
from typing import TYPE_CHECKING

from pydantic_ai.messages import ModelResponse

from ..core import (
    AuthorizationAction, AuthorizationPolicy, CursorSigner, JsonValue, Principal, ResourceKind,
    ResourceRef, canonical_json_bytes, canonical_sha256, decode_cursor, encode_cursor,
    principal_identity_payload, validate_page_limit, validate_resource_id,
)
from ..errors import AIError, ErrorCode
from ..model.messages import decode_model_messages
from ..persistence.turns import SessionTurnKey, SqlSessionTurnReader
from ._session_timeline import _timeline_items

if TYPE_CHECKING:
    from sqlalchemy import RowMapping

_STATUS = {
    "accepted": "PENDING_START",
    "cancelling": "CANCELLING",
    "cancelled": "CANCELLED",
    "failed": "FAILED",
    "succeeded": "SUCCEEDED",
}


def _session_view(row: RowMapping) -> dict[str, JsonValue]:
    config = row["config_json"]
    active = row["active_turn_id"]
    return {
        "session_id": row["session_key"].decode("utf-8"),
        "agent_id": row["agent_id"].decode("utf-8"),
        "status": row["status"].upper(),
        "revision": row["revision"],
        "cwd": config.get("cwd"),
        "active_turn_id": None if active is None else active.decode("utf-8"),
        "last_turn_no": row["last_turn_no"],
        "metadata": config.get("metadata", {}),
        "history_quality": "complete",
    }


def _turn_view(row: RowMapping) -> dict[str, JsonValue]:
    state = row["state"]
    result = row["result_json"]
    output = None
    items: list[JsonValue] = []
    if state == "succeeded":
        messages = decode_model_messages(canonical_json_bytes(result))
        if not messages or not isinstance(messages[-1], ModelResponse) or messages[-1].text is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        output = {"text": messages[-1].text}
        items = [asdict(item) for item in _timeline_items(messages)]
    return {
        "turn_id": row["turn_id"].decode("utf-8"),
        "turn_no": row["turn_no"],
        "status": _STATUS[state],
        "created_at": row["created_at"].isoformat(),
        "updated_at": row["updated_at"].isoformat(),
        "user_input": {"version": 1, "prompt": row["request_json"]["prompt"], "files": [], "attachments": []},
        "conversation_committed": state in ("succeeded", "failed", "cancelled"),
        "items": items,
        "process": [],
        "output": output,
        "error_code": result.get("error_code") if isinstance(result, Mapping) else None,
        "safe_error_details": {},
    }


class _OrdinarySessionHistory:
    def __init__(
        self, store: SqlSessionTurnReader, tenant_id: str, authorization: AuthorizationPolicy, signer: CursorSigner,
    ) -> None:
        self._store = store
        self._tenant_id = tenant_id
        self._authorization = authorization
        self._signer = signer

    def _require_tenant(self, principal: Principal) -> None:
        if principal.tenant_id != self._tenant_id:
            raise AIError(ErrorCode.AUTHORIZATION_DENIED)

    async def _authorize_row(self, row: RowMapping, principal: Principal) -> None:
        await self._authorization.authorize(
            principal, AuthorizationAction.SESSION_READ,
            ResourceRef(ResourceKind.SESSION, row["session_key"].decode("utf-8"), self._tenant_id,
                        row["owner_principal_id"].decode("utf-8")),
        )

    async def _session(self, session_id: str, principal: Principal) -> RowMapping:
        self._require_tenant(principal)
        validate_resource_id(session_id)
        row = await self._store.get_session(session_id)
        if row is None:
            raise AIError(ErrorCode.AUTHORIZATION_DENIED)
        await self._authorize_row(row, principal)
        return row

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

    async def sessions(self, *, principal: Principal, cursor: str | None, limit: int) -> dict[str, JsonValue]:
        self._require_tenant(principal)
        limit = validate_page_limit(limit)
        await self._authorization.authorize(
            principal, AuthorizationAction.SESSION_READ, ResourceRef(ResourceKind.SESSION, "list", self._tenant_id),
        )
        kind = "ORDINARY_SESSIONS"
        digest = canonical_sha256({"principal": principal_identity_payload(principal)})
        before = self._cursor_position(cursor, kind, digest)
        rows = await self._store.session_page(owner_principal_id=principal.principal_id, before=before, limit=limit + 1)
        items: list[JsonValue] = []
        for row in rows[:limit]:
            try:
                await self._authorize_row(row, principal)
            except AIError as error:
                if error.code is not ErrorCode.AUTHORIZATION_DENIED:
                    raise
            else:
                items.append(_session_view(row))
        next_cursor = self._next_cursor(rows[limit - 1]["id"], kind, digest) if len(rows) > limit else None
        return {"items": items, "next_cursor": next_cursor}

    async def session_state(self, session_id: str, *, principal: Principal) -> dict[str, JsonValue]:
        return _session_view(await self._session(session_id, principal))

    async def session(
        self, session_id: str, *, principal: Principal, cursor: str | None, limit: int,
    ) -> dict[str, JsonValue]:
        limit = validate_page_limit(limit)
        row = await self._session(session_id, principal)
        kind = "ORDINARY_TURNS"
        digest = canonical_sha256({"principal": principal_identity_payload(principal), "session": session_id})
        before = self._cursor_position(cursor, kind, digest)
        through = row["last_turn_no"] if before is None else min(row["last_turn_no"], before - 1)
        turns = () if through == 0 else await self._store.turn_page(session_id, through=through, limit=limit + 1)
        next_cursor = self._next_cursor(turns[limit - 1]["turn_no"], kind, digest) if len(turns) > limit else None
        return {
            "session": _session_view(row),
            "timeline": {"items": [_turn_view(turn) for turn in reversed(turns[:limit])], "next_cursor": next_cursor},
        }

    async def turn_result(
        self, session_id: str, turn_id: str, *, principal: Principal,
    ) -> dict[str, JsonValue]:
        await self._session(session_id, principal)
        validate_resource_id(turn_id)
        row = await self._store.get_turn(SessionTurnKey(session_id, turn_id))
        if row is None:
            raise AIError(ErrorCode.AUTHORIZATION_DENIED)
        return _turn_view(row)
