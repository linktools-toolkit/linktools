#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Portable, tenant-scoped ordinary conversation snapshots."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from ..core import Principal, canonical_json_bytes, validate_resource_id
from ..errors import AIError, ErrorCode
from ._session_turn_sql import SESSION_TABLE, SESSION_TURN_TABLE
from ._turn_sql_schema import TURN_EXECUTION_MAX_BYTES, TURN_REQUEST_MAX_BYTES, TURN_SESSION_MAX_BYTES
from .state import SqlStateRowReader, SqlStateTransaction, StateTransaction, state_store_key

if TYPE_CHECKING:
    from ._sql import SqlStateStore

_AUDIT = ("created_at", "updated_at")
_SESSION_BINARY = {"session_key": TURN_SESSION_MAX_BYTES, "active_turn_id": TURN_EXECUTION_MAX_BYTES}
_TURN_BINARY = {"session_key": TURN_SESSION_MAX_BYTES, "turn_id": TURN_EXECUTION_MAX_BYTES,
                "request_key": TURN_REQUEST_MAX_BYTES}
_SESSION_FIELDS = (*_SESSION_BINARY, "status", "revision", "last_turn_no", "config_json", *_AUDIT)
_TURN_FIELDS = (*_TURN_BINARY, "turn_no", "context_from", "context_through", "state", "generation",
                "request_json", "result_json", *_AUDIT)
_TABLES = (("sessions", SESSION_TABLE, _SESSION_FIELDS, _SESSION_BINARY),
           ("turns", SESSION_TURN_TABLE, _TURN_FIELDS, _TURN_BINARY))
_ACTIVE = {"accepted", "cancelling"}
_TERMINAL = {"succeeded", "failed", "cancelled"}


def _invalid() -> AIError:
    return AIError(ErrorCode.STORAGE_INTEGRITY_ERROR, "Invalid ordinary session snapshot")


def _timestamp(value: datetime) -> datetime:
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)


def _encode_row(row: Mapping[str, object], binary: Mapping[str, int]) -> dict[str, object]:
    result = dict(row)
    for key in binary:
        if result[key] is not None:
            result[key] = result[key].decode("utf-8")
    for key in _AUDIT:
        result[key] = _timestamp(result[key]).isoformat()
    return result


def _validate_text(value: object, width: int) -> str:
    if not isinstance(value, str) or not 1 <= len(value.encode("utf-8")) <= width:
        raise _invalid()
    return value


def decode_ordinary_snapshot(value: object, *, tenant_id: str) -> dict[str, list[dict[str, object]]]:
    """Validate and decode before opening the destination; exclude live lease values."""
    if not isinstance(value, dict) or set(value) != {"sessions", "turns"}:
        raise _invalid()
    decoded: dict[str, list[dict[str, object]]] = {"sessions": [], "turns": []}
    session_ids: dict[bytes, dict[str, object]] = {}
    turn_ids: set[bytes] = set()
    request_keys: set[tuple[bytes, bytes]] = set()
    positions: dict[bytes, set[int]] = {}
    active: dict[bytes, bytes] = {}
    for section, _, fields, binary in _TABLES:
        rows = value[section]
        if not isinstance(rows, list):
            raise _invalid()
        for wire in rows:
            if not isinstance(wire, dict) or wire.keys() != set(fields):
                raise _invalid()
            row = dict(wire)
            try:
                for key, width in binary.items():
                    if key == "active_turn_id" and row[key] is None:
                        continue
                    text = _validate_text(row[key], width)
                    if key in {"session_key", "turn_id", "active_turn_id"}:
                        validate_resource_id(text)
                    row[key] = text.encode("utf-8")
                for key in _AUDIT:
                    timestamp = datetime.fromisoformat(row[key])
                    if _timestamp(timestamp).isoformat() != row[key]:
                        raise _invalid()
                    row[key] = timestamp
            except (TypeError, ValueError, UnicodeError, AIError) as error:
                raise _invalid() from error
            session_id = row["session_key"]
            if section == "sessions":
                if (session_id in session_ids or row["status"] not in ("open", "closed")
                        or not isinstance(row["config_json"], dict)
                        or any(type(row[key]) is not int or not 0 <= row[key] < 2**63 for key in ("revision", "last_turn_no"))
                        or row["status"] == "closed" and row["active_turn_id"] is not None):
                    raise _invalid()
                session_ids[session_id] = row
            else:
                request = row["request_json"]
                result = row["result_json"]
                number = row["turn_no"]
                state = row["state"]
                turn_id, request_key = row["turn_id"], row["request_key"]
                if (session_id not in session_ids or turn_id in turn_ids or (session_id, request_key) in request_keys
                        or state not in (*_ACTIVE, *_TERMINAL)
                        or any(type(row[key]) is not int or not 1 <= row[key] < 2**63
                               for key in ("generation", "turn_no", "context_from"))
                        or type(row["context_through"]) is not int or row["context_through"] != number - 1
                        or not 0 <= number - row["context_from"] <= 128
                        or not isinstance(request, dict) or not isinstance(request.get("prompt"), str)
                        or not isinstance(request.get("binding"), dict)
                        or len(canonical_json_bytes(request)) + len(canonical_json_bytes(result)) > 64 * 1024):
                    raise _invalid()
                try:
                    principal = Principal(**request["principal"])
                except (TypeError, ValueError, KeyError) as error:
                    raise _invalid() from error
                if principal.tenant_id != tenant_id:
                    raise _invalid()
                if state in _ACTIVE:
                    if (session_id in active or number != session_ids[session_id]["last_turn_no"] + 1
                            or result is not None):
                        raise _invalid()
                    active[session_id] = turn_id
                    row["lease_until"] = 0
                else:
                    published = positions.setdefault(session_id, set())
                    if number in published or number > session_ids[session_id]["last_turn_no"]:
                        raise _invalid()
                    if state == "succeeded" and not isinstance(result, list):
                        raise _invalid()
                    if state != "succeeded" and not (isinstance(result, dict) and isinstance(result.get("error_code"), str)):
                        raise _invalid()
                    published.add(number)
                    row["lease_until"] = None
                turn_ids.add(turn_id)
                request_keys.add((session_id, request_key))
            decoded[section].append(row)
    if any(row["active_turn_id"] != active.get(key) or row["last_turn_no"] != len(positions.get(key, ()))
           for key, row in session_ids.items()):
        raise _invalid()
    return decoded


async def capture_ordinary_snapshot(
    store: SqlStateStore, *, namespace: str, tenant_id: str, accept: Callable[[object], None],
    page_size: Callable[[int], int],
) -> dict[str, list[dict[str, object]]]:
    """Collect the same read-transaction view used by the Conversation snapshot."""
    from sqlalchemy import select

    metadata = store.storage_group.metadata
    scope = state_store_key(namespace, tenant_id, "regenerable-session")

    async def capture(transaction: StateTransaction) -> dict[str, list[dict[str, object]]]:
        if not isinstance(transaction, SqlStateRowReader):
            raise AIError(ErrorCode.SNAPSHOT_UNSUPPORTED)
        result: dict[str, list[dict[str, object]]] = {}
        for section, table_name, fields, binary in _TABLES:
            table = metadata.tables[table_name]
            output: list[dict[str, object]] = []
            after = 0
            while True:
                limit = page_size(16)
                statement = select(table.c.id, *(table.c[name] for name in fields)).where(
                    table.c.scope_key == scope, table.c.id > after,
                ).order_by(table.c.id).limit(limit)
                rows = await transaction.read_sql_rows(store.context, statement)
                for row in rows:
                    wire = _encode_row({key: row[key] for key in fields}, binary)
                    accept(wire)
                    output.append(wire)
                if len(rows) < limit:
                    break
                after = rows[-1]["id"]
            result[section] = output
        return result

    wire = await store.read(capture)
    decode_ordinary_snapshot(wire, tenant_id=tenant_id)
    return wire


async def restore_ordinary_snapshot(
    store: SqlStateStore, transaction: StateTransaction, *, namespace: str, tenant_id: str,
    rows: Mapping[str, list[dict[str, object]]],
) -> None:
    """Insert the complete logical session graph in the active Conversation write boundary."""
    if not isinstance(transaction, SqlStateTransaction):
        raise AIError(ErrorCode.SNAPSHOT_UNSUPPORTED)
    scope = state_store_key(namespace, tenant_id, "regenerable-session")
    metadata = store.storage_group.metadata

    async def insert(connection) -> None:
        for section, table, _, _ in _TABLES:
            selected = rows[section]
            for start in range(0, len(selected), 16):
                await connection.execute(
                    metadata.tables[table].insert(),
                    [{**value, "scope_key": scope} for value in selected[start:start + 16]],
                )
    await transaction.run_sql(store.context, insert)
