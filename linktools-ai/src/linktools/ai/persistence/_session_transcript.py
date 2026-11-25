#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared ownership references for persisted session transcript bodies."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

from ..errors import AIError, ErrorCode
from ..steps import NATIVE_STORE_KEY_MAX_BYTES
from ._message_root import NativeMessageRoot
from ._record_key import decode_key_components
from .state import state_store_key

if TYPE_CHECKING:
    from ._sql import SqlStateStorageGroup

SESSION_TURN_TABLE = "ai_session_turns"


def decode_session_transcript_root(result: object) -> NativeMessageRoot:
    if not isinstance(result, dict) or set(result) != {"transcript", "binding_refs"}:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR, "Invalid session transcript reference")
    try:
        root = NativeMessageRoot.from_json(result["transcript"])
    except (TypeError, ValueError, AIError) as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR, "Invalid session transcript reference") from error
    refs = result["binding_refs"]
    if not isinstance(refs, list) or any(
        not isinstance(ref, list) or len(ref) != 2
        or type(ref[0]) is not int or not 0 <= ref[0] < root.count
        or ref[1] is not None and (type(ref[1]) is not int or ref[1] < 0)
        for ref in refs
    ):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR, "Invalid session transcript reference")
    order = [(message, -1 if part is None else part) for message, part in refs]
    if not root.count or any(previous >= current for previous, current in zip(order, order[1:])):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR, "Invalid session transcript reference")
    return root


async def scan_ordinary_transcript_roots(group: SqlStateStorageGroup) -> AsyncIterator[NativeMessageRoot]:
    """Offline only; visit the exact owners of native history across all scopes."""
    from sqlalchemy import select

    table = group.metadata.tables.get(SESSION_TURN_TABLE)
    if table is None:
        return
    after = 0
    async with group.context.sessions() as session:
        await group.context.dialect.begin_consistent_read(session)
        while True:
            rows = (await group.context.run_statement(session.execute(select(
                table.c.id, table.c.scope_key, table.c.result_json,
            ).where(table.c.id > after, table.c.state == "succeeded").order_by(table.c.id).limit(128)))).mappings().all()
            for row in rows:
                root = decode_session_transcript_root(row["result_json"])
                _, namespace, tenant_id = decode_key_components(root.store_key, NATIVE_STORE_KEY_MAX_BYTES)
                if row["scope_key"] != state_store_key(namespace, tenant_id, "regenerable-session"):
                    raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)
                yield root
            if len(rows) < 128:
                return
            after = rows[-1]["id"]
