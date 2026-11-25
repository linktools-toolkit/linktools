#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Portable, tenant-scoped ordinary conversation snapshots."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from dataclasses import replace
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from ..core import Principal, RUNTIME_OBJECT_STORE_ID, canonical_json_bytes, validate_resource_id
from ..errors import AIError, ErrorCode
from ._ordinary_lineage import decode_session_lineage, fork_segments
from ._session_turn_sql import (
    SESSION_TABLE, SESSION_TURN_TABLE, CanonicalDeferredFrontier, SessionInstructionFrontier, SessionTurnKey,
)
from ._turn_sql_schema import TURN_EXECUTION_MAX_BYTES, TURN_REQUEST_MAX_BYTES, TURN_SESSION_MAX_BYTES
from ..model import StoredUserInput
from ..steps import NATIVE_STORE_KEY_MAX_BYTES, encode_native_replay_key
from ._message_root import NativeMessageRoot
from ._session_transcript import decode_session_transcript_root
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
_TURN_FIELDS = (*_TURN_BINARY, "binding_key", "turn_no", "context_from", "context_through", "state", "generation",
                "request_json", "result_json", *_AUDIT)
_TABLES = (("sessions", SESSION_TABLE, _SESSION_FIELDS, _SESSION_BINARY),
           ("turns", SESSION_TURN_TABLE, _TURN_FIELDS, _TURN_BINARY))
_WAITING = {"waiting", "ready"}
_ACTIVE = {"accepted", "cancelling", *_WAITING}
_TERMINAL = {"succeeded", "failed", "cancelled"}


def _invalid() -> AIError:
    return AIError(ErrorCode.STORAGE_INTEGRITY_ERROR, "Invalid ordinary session snapshot")


def has_waiting_turns(rows: Mapping[str, list[dict[str, object]]]) -> bool:
    return any(row["state"] in _WAITING for row in rows["turns"])


def require_waiting_checkpoints(rows, backups) -> str | None:
    """Match each occupied paused turn to its exact retained Step occurrence."""
    expected = set()
    for row in rows["turns"]:
        if row["state"] not in _WAITING:
            continue
        session_id, turn_id = (
            value.decode("utf-8") if isinstance(value, bytes) else value
            for value in (row["session_key"], row["turn_id"])
        )
        frontier = CanonicalDeferredFrontier.from_json(
            SessionTurnKey(session_id, turn_id), row["generation"], row["result_json"],
        )
        expected.add((turn_id, session_id, encode_native_replay_key(frontier.checkpoint_id)))
    if not expected:
        return None
    groups = []
    for group_id, backup in backups.items():
        retained = {
            (row["run_id"], row["conversation_id"], row["idempotency_key"])
            for row in backup.snapshots if not row["pruned"]
        }
        matching = expected & retained
        if matching == expected:
            groups.append(group_id)
        elif matching:
            raise _invalid()
    if len(groups) != 1:
        raise _invalid()
    return groups[0]


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



def ordinary_transcript_roots(
    rows: Mapping[str, list[dict[str, object]]],
) -> Iterator[NativeMessageRoot]:
    for row in rows["turns"]:
        if row["state"] == "succeeded":
            yield decode_session_transcript_root(row["result_json"])


def rekey_ordinary_transcripts(
    rows: Mapping[str, list[dict[str, object]]], group_ids: Mapping[str, str],
) -> dict[str, list[dict[str, object]]]:
    def move(row: dict[str, object]) -> dict[str, object]:
        if row["state"] != "succeeded":
            return row
        result = row["result_json"]
        root = decode_session_transcript_root(result)
        if root.group_id not in group_ids:
            raise _invalid()
        return {**row, "result_json": {
            **result, "transcript": replace(root, group_id=group_ids[root.group_id]).to_json(),
        }}
    return {**rows, "turns": [move(row) for row in rows["turns"]]}



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


def decode_ordinary_snapshot(value: object, *, tenant_id: str, paired_split: bool = False) -> dict[str, list[dict[str, object]]]:
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
            has_split_receipt = section == "turns" and paired_split and isinstance(wire, dict) and "split_source_receipt" in wire
            expected = {*fields, "split_source_receipt"} if has_split_receipt else set(fields)
            if not isinstance(wire, dict) or wire.keys() != expected:
                raise _invalid()
            row = dict(wire)
            try:
                if has_split_receipt:
                    if row["state"] not in _TERMINAL:
                        raise _invalid()
                    receipt = _validate_text(row["split_source_receipt"], 64)
                    row["split_source_receipt"] = validate_resource_id(receipt).encode("utf-8")
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
                if paired_split and state in _ACTIVE:
                    # Before the first source registration an active split turn
                    # has no receipt. The paired profile never exports that gap.
                    raise AIError(ErrorCode.SNAPSHOT_UNSUPPORTED, "Paired split Step snapshots require terminal turns")
                turn_id, request_key = row["turn_id"], row["request_key"]
                if (session_id not in session_ids or turn_id in turn_ids or (session_id, request_key) in request_keys
                        or state not in (*_ACTIVE, *_TERMINAL)
                        or any(type(row[key]) is not int or not 1 <= row[key] < 2**63
                               for key in ("generation", "turn_no", "context_from"))
                        or type(row["context_through"]) is not int or row["context_through"] != number - 1
                        or not 0 <= number - row["context_from"] <= 128
                        or not isinstance(request, dict) or not isinstance(request.get("prompt"), str)
                        or "binding" in request):
                    raise _invalid()
                try:
                    request_core = dict(request)
                    frontier = request_core.pop("instruction_frontier", None)
                    if len(canonical_json_bytes(request_core)) + max(
                        len(canonical_json_bytes(result)),
                        0 if frontier is None else len(canonical_json_bytes(frontier)),
                    ) > 64 * 1024:
                        raise _invalid()
                    if frontier is not None:
                        SessionInstructionFrontier.from_json(frontier)
                    principal = Principal(**request["principal"])
                except (TypeError, ValueError, KeyError, AIError) as error:
                    raise _invalid() from error
                if principal.tenant_id != tenant_id:
                    raise _invalid()
                if state in _ACTIVE:
                    if session_id in active or number != session_ids[session_id]["last_turn_no"] + 1:
                        raise _invalid()
                    if state in _WAITING:
                        CanonicalDeferredFrontier.from_json(
                            SessionTurnKey(session_id.decode("utf-8"), turn_id.decode("utf-8")),
                            row["generation"], result,
                        )
                        row["lease_until"] = None
                    else:
                        if result is not None:
                            raise _invalid()
                        row["lease_until"] = 0
                    active[session_id] = turn_id
                else:
                    published = positions.setdefault(session_id, set())
                    if number in published or number > session_ids[session_id]["last_turn_no"]:
                        raise _invalid()
                    if state == "succeeded":
                        decode_session_transcript_root(result)
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
    page_size: Callable[[int], int], paired_split: bool = False,
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
            ordering = table.c.id
            extra = (ordering,)
            if section == "turns" and "split_source_receipt" in table.c:
                extra += (table.c.split_source_receipt,)
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
                    if section == "turns" and (receipt := row.get("split_source_receipt")) is not None:
                        if not paired_split or row["state"] not in _TERMINAL:
                            raise AIError(ErrorCode.SNAPSHOT_UNSUPPORTED, "Split Step receipts require a paired terminal snapshot")
                        wire["split_source_receipt"] = receipt.decode("utf-8")
                    accept(wire)
                    output.append(wire)
                if len(rows) < limit:
                    break
                after = rows[-1][ordering.key]
            result[section] = output
        return result

    wire = await store.read(capture)
    decode_ordinary_snapshot(wire, tenant_id=tenant_id, paired_split=paired_split)
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
