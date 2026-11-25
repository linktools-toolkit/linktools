#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Portable, tenant-scoped ordinary conversation snapshots."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from ..core import Principal, RUNTIME_OBJECT_STORE_ID, canonical_json_bytes, semantic_json_sha256, validate_resource_id
from ..errors import AIError, ErrorCode
from ._ordinary_lineage import decode_session_lineage, fork_segments
from ._session_turn_sql import SESSION_BINDING_TABLE, SESSION_TABLE, SESSION_TURN_TABLE
from ._turn_sql_schema import TURN_EXECUTION_MAX_BYTES, TURN_REQUEST_MAX_BYTES, TURN_SESSION_MAX_BYTES
from ..model import StoredUserInput
from ..model import decode_ordinary_transcript
from ..storage import ObjectRef
from .codec import decode_domain
from .objects import ObjectKeyFactory
from ._plan import PersistenceDomain
from .state import SqlStateRowReader, SqlStateTransaction, StateTransaction, state_store_key

if TYPE_CHECKING:
    from ._sql import SqlStateStore

_AUDIT = ("created_at", "updated_at")
_SESSION_BINARY = {"session_key": TURN_SESSION_MAX_BYTES, "active_turn_id": TURN_EXECUTION_MAX_BYTES}
_TURN_BINARY = {"session_key": TURN_SESSION_MAX_BYTES, "turn_id": TURN_EXECUTION_MAX_BYTES,
                "request_key": TURN_REQUEST_MAX_BYTES}
_SESSION_FIELDS = (*_SESSION_BINARY, "status", "revision", "last_turn_no", "config_json", "lineage_json", *_AUDIT)
_BINDING_FIELDS = ("binding_key", "binding_json", *_AUDIT)
_TURN_FIELDS = (*_TURN_BINARY, "binding_key", "turn_no", "context_from", "context_through", "state", "generation",
                "request_json", "result_json", *_AUDIT)
_TABLES = (("bindings", SESSION_BINDING_TABLE, _BINDING_FIELDS, {}),
           ("sessions", SESSION_TABLE, _SESSION_FIELDS, _SESSION_BINARY),
           ("turns", SESSION_TURN_TABLE, _TURN_FIELDS, _TURN_BINARY))
_ACTIVE = {"accepted", "cancelling"}
_TERMINAL = {"succeeded", "failed", "cancelled"}


def _invalid() -> AIError:
    return AIError(ErrorCode.STORAGE_INTEGRITY_ERROR, "Invalid ordinary session snapshot")


def ordinary_input_object_refs(
    rows: Mapping[str, list[dict[str, object]]], *, namespace: str, tenant_id: str,
) -> Iterator[ObjectRef]:
    keys = ObjectKeyFactory(namespace)
    for row in rows["turns"]:
        manifest = row["request_json"].get("input")
        if manifest is None:
            continue
        for reference in decode_domain(manifest, StoredUserInput).payload.object_refs():
            if (reference.store_id != RUNTIME_OBJECT_STORE_ID
                    or reference.key != keys.key(PersistenceDomain.EXECUTION, tenant_id, reference.digest)):
                raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)
            yield reference


def _timestamp(value: datetime) -> datetime:
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)


def _encode_row(row: Mapping[str, object], binary: Mapping[str, int]) -> dict[str, object]:
    result = dict(row)
    for key in binary:
        if result[key] is not None:
            result[key] = result[key].decode("utf-8")
    if "binding_key" in result:
        result["binding_key"] = result["binding_key"].hex()
    for key in _AUDIT:
        if key in result:
            result[key] = _timestamp(result[key]).isoformat()
    return result


def _validate_text(value: object, width: int) -> str:
    if not isinstance(value, str) or not 1 <= len(value.encode("utf-8")) <= width:
        raise _invalid()
    return value


def decode_ordinary_snapshot(value: object, *, tenant_id: str) -> dict[str, list[dict[str, object]]]:
    """Validate and decode before opening the destination; exclude live lease values."""
    if not isinstance(value, dict) or set(value) != {"bindings", "sessions", "turns"}:
        raise _invalid()
    decoded: dict[str, list[dict[str, object]]] = {"bindings": [], "sessions": [], "turns": []}
    bindings: dict[bytes, dict[str, object]] = {}
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
                if "binding_key" in row:
                    text = row["binding_key"]
                    if not isinstance(text, str) or len(text) != 64:
                        raise _invalid()
                    key = bytes.fromhex(text)
                    if key.hex() != text:
                        raise _invalid()
                    row["binding_key"] = key
                for key, width in binary.items():
                    if key == "active_turn_id" and row[key] is None:
                        continue
                    text = _validate_text(row[key], width)
                    if key in {"session_key", "turn_id", "active_turn_id"}:
                        validate_resource_id(text)
                    row[key] = text.encode("utf-8")
                for key in _AUDIT:
                    if key not in row:
                        continue
                    timestamp = datetime.fromisoformat(row[key])
                    if _timestamp(timestamp).isoformat() != row[key]:
                        raise _invalid()
                    row[key] = timestamp
            except (TypeError, ValueError, UnicodeError, AIError) as error:
                raise _invalid() from error
            if section == "bindings":
                binding_key, payload = row["binding_key"], row["binding_json"]
                if (binding_key in bindings or not isinstance(payload, dict)
                        or semantic_json_sha256(payload) != binding_key.hex()):
                    raise _invalid()
                bindings[binding_key] = payload
                decoded[section].append(row)
                continue
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
                        or "binding" in request or row["binding_key"] not in bindings
                        or len(canonical_json_bytes({**request, "binding": bindings[row["binding_key"]]}))
                        + len(canonical_json_bytes(result)) > 64 * 1024):
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
                    if state == "succeeded":
                        if not isinstance(result, (dict, list)):
                            raise _invalid()
                        if isinstance(result, dict):
                            spec = bindings[row["binding_key"]].get("agent_spec")
                            if not isinstance(spec, dict):
                                raise _invalid()
                            frozen = {}
                            if result.get("instructions_from_binding") is True:
                                instructions = spec.get("instructions")
                                if not isinstance(instructions, list) or not all(isinstance(item, str) for item in instructions):
                                    raise _invalid()
                                frozen["instructions"] = instructions
                            if result.get("system_prompt_from_binding") is True:
                                system_prompt = spec.get("system_prompt")
                                if not isinstance(system_prompt, str):
                                    raise _invalid()
                                frozen["system_prompt"] = system_prompt
                            try:
                                decode_ordinary_transcript(result, **frozen)
                            except AIError as error:
                                raise _invalid() from error
                    elif not (isinstance(result, dict) and isinstance(result.get("error_code"), str)):
                        raise _invalid()
                    published.add(number)
                    row["lease_until"] = None
                turn_ids.add(turn_id)
                request_keys.add((session_id, request_key))
            decoded[section].append(row)
    lineages = {key: decode_session_lineage(row["lineage_json"]) for key, row in session_ids.items()}
    parents: dict[bytes, bytes] = {}
    for key, row in session_ids.items():
        lineage = lineages[key]
        inherited = lineage[-1].through if lineage else 0
        head = row["last_turn_no"]
        own = positions.get(key, ())
        if (row["active_turn_id"] != active.get(key) or inherited > head
                or head - inherited != len(own) or any(number <= inherited for number in own)):
            raise _invalid()
        if row["lineage_json"] is not None:
            parent_id = row["lineage_json"]["source"]
            parent_key = parent_id.encode("utf-8")
            parent = session_ids.get(parent_key)
            if parent is None or parent_key == key or inherited > parent["last_turn_no"]:
                raise _invalid()
            try:
                if fork_segments(parent_id, lineages[parent_key], inherited) != lineage:
                    raise _invalid()
            except AIError as error:
                raise _invalid() from error
            parents[key] = parent_key
    checked: set[bytes] = set()
    for start in parents:
        path: set[bytes] = set()
        key = start
        while key in parents and key not in checked:
            if key in path:
                raise _invalid()
            path.add(key)
            key = parents[key]
        checked.update(path)
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
            ordering = table.c.binding_key if section == "bindings" else table.c.id
            extra = () if section == "bindings" else (ordering,)
            after = None
            while True:
                limit = page_size(16)
                statement = select(*extra, *(table.c[name] for name in fields)).where(table.c.scope_key == scope)
                if after is not None:
                    statement = statement.where(ordering > after)
                statement = statement.order_by(ordering).limit(limit)
                rows = await transaction.read_sql_rows(store.context, statement)
                for row in rows:
                    wire = _encode_row({key: row[key] for key in fields}, binary)
                    accept(wire)
                    output.append(wire)
                if len(rows) < limit:
                    break
                after = rows[-1][ordering.key]
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
