#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared ownership references for persisted session transcript bodies."""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from dataclasses import fields
from typing import TYPE_CHECKING

from ..core import UsageMetrics
from ..errors import AIError, ErrorCode
from ..steps import NATIVE_STORE_KEY_MAX_BYTES, native_store_key
from ._message_root import NativeMessageRoot, NativeUserInputReference
from ._record_key import decode_key_components
from .state import state_store_key

if TYPE_CHECKING:
    from ._sql import SqlStateStorageGroup

SESSION_TURN_TABLE = "ai_session_turns"
_USAGE_KEYS = frozenset(field.name for field in fields(UsageMetrics))


def decode_session_turn_usage(value: object) -> UsageMetrics:
    if not isinstance(value, dict) or value.keys() != _USAGE_KEYS:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR, "Invalid session turn usage")
    try:
        return UsageMetrics(**value)
    except (TypeError, ValueError) as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR, "Invalid session turn usage") from error


def decode_session_transcript_root(result: object) -> NativeMessageRoot:
    if not isinstance(result, Mapping):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR, "Invalid session transcript reference")
    fields = {"transcript", "binding_refs"}
    if "output_ref" in result:
        fields.add("output_ref")
    if "graph_trace_suffix" in result:
        fields.add("graph_trace_suffix")
        if result["graph_trace_suffix"] is not True or result.get("binding_refs") != []:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR, "Invalid raw Graph transcript reference")
    if not fields <= result.keys() <= fields | {"usage"}:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR, "Invalid session transcript reference")
    if "usage" in result:
        decode_session_turn_usage(result["usage"])
    try:
        root = NativeMessageRoot.from_json(result["transcript"])
    except (TypeError, ValueError, AIError) as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR, "Invalid session transcript reference") from error
    if "output_ref" in result:
        ref = result["output_ref"]
        if (not isinstance(ref, list) or len(ref) != 2
                or any(type(index) is not int or index < 0 for index in ref) or ref[0] >= root.count):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR, "Invalid session output reference")
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


def decode_session_input_reference(
    request: Mapping[str, object], turn_id: str,
) -> NativeUserInputReference | None:
    value = request.get("input")
    has_native = isinstance(value, Mapping) and "$native_user_input" in value
    if not has_native and request.get("graph_execution") is None:
        return None
    if not has_native or request.get("graph_execution") != turn_id or request.get("prompt") != "":
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR, "Invalid Graph session input reference")
    try:
        reference = NativeUserInputReference.from_json(value)
        if reference.part_index != 0:
            raise ValueError("Graph input must refer to the single native prompt")
        return reference
    except (TypeError, ValueError, AIError) as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR, "Invalid Graph session input reference") from error


async def scan_ordinary_message_roots(group: SqlStateStorageGroup) -> AsyncIterator[NativeMessageRoot]:
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
                table.c.id, table.c.scope_key, table.c.state, table.c.turn_id, table.c.result_json,
                table.c.request_json["graph_execution"].as_string().label("graph_execution"),
                table.c.request_json["prompt"].as_string().label("prompt"),
                table.c.request_json["input"].label("input"),
            ).where(table.c.id > after).order_by(table.c.id).limit(128)))).mappings().all()
            for row in rows:
                reference = decode_session_input_reference(row, row["turn_id"].decode("utf-8"))
                roots = () if reference is None else (reference.root,)
                if row["state"] == "succeeded":
                    roots = (*roots, decode_session_transcript_root(row["result_json"]))
                for root in roots:
                    _, namespace, tenant_id = decode_key_components(root.store_key, NATIVE_STORE_KEY_MAX_BYTES)
                    if (root.store_key != native_store_key(namespace, tenant_id)
                            or row["scope_key"] != state_store_key(namespace, tenant_id, "regenerable-session")):
                        raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)
                    yield root
            if len(rows) < 128:
                return
            after = rows[-1]["id"]
