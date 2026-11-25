#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Portable execution-domain snapshots of source-owned SQL coordination."""

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone

from ..errors import AIError, ErrorCode
from ..storage import TurnScopeIdentity
from ._turn_sql import SESSION, STANDALONE, SqlTurnProof, turn_hex, turn_session_address, turn_session_identity
from ._turn_sql_schema import TURN_EXECUTION_MAX_BYTES, TURN_REQUEST_MAX_BYTES, RUNTIME_TURN_SCOPE, unoccupied_turn_route
from .state import StoredRecord, record_key, scope_key


TURN_SNAPSHOT_KINDS = frozenset({"turn_slot", "turn_admission"})
GENERIC_TURN_SNAPSHOT_KINDS = frozenset({"turn_session", "turn_request", "turn_execution"})


def _integrity() -> AIError:
    return AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)


def _bytes(value: object, limit: int) -> bytes:
    if not isinstance(value, str) or not value:
        raise _integrity()
    result = value.encode("utf-8")
    if len(result) > limit:
        raise _integrity()
    return result


def _address(scope: TurnScopeIdentity, kind: str, identity: str) -> tuple[bytes, bytes]:
    domain, purpose = "execution", scope.scope
    return (
        record_key(scope.namespace, scope.tenant_id, domain, kind, (purpose, identity)),
        scope_key(scope.namespace, scope.tenant_id, domain, "turn", "scope", (purpose,)),
    )


def _utc(value: object) -> datetime:
    if not isinstance(value, datetime):
        raise _integrity()
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)


def strict_session_snapshot_record(
    session_key: bytes, execution: bytes | None, fence: int, *, namespace: str, tenant_id: str,
) -> StoredRecord:
    """Encode a logical strict Conversation pointer and fencing high-water."""
    scope = TurnScopeIdentity(namespace, tenant_id, RUNTIME_TURN_SCOPE)
    kind = "turn_slot"
    identity = turn_session_identity(SESSION, session_key)
    key, grouped = _address(scope, kind, identity)
    return StoredRecord(key, grouped, None, kind, key, "ACTIVE" if execution is not None else "IDLE",
                        0, None, fence, None, {
        "scope": scope.scope, "session_key": identity, "session_record": True,
        "execution_id": execution.decode("utf-8") if execution is not None else None,
    })


def execution_receipt_snapshot_record(
    row: Mapping[str, object], proof: SqlTurnProof, *, namespace: str, tenant_id: str, split_session: bool = False,
) -> StoredRecord:
    """Encode the portable receipt without duplicating the owning Execution body."""
    scope = TurnScopeIdentity(namespace, tenant_id, RUNTIME_TURN_SCOPE)
    kind = "turn_admission"
    key, grouped = _address(scope, kind, proof.execution_id)
    retained = row["turn_occupied_route"] if split_session else row["turn_request_route"]
    state = "ACTIVE" if proof.active else "RETIRED" if retained == unoccupied_turn_route(row["key_locator"]) else "PARKED"
    token = row["turn_token"].decode("ascii") if proof.active else None
    return StoredRecord(
        key, grouped, None, kind, key, state, row["turn_revision"], token, proof.fence, _utc(row["turn_expiry"]),
        {"scope": scope.scope, "execution_id": proof.execution_id, "session_key": proof.session_key,
         "request_key": proof.request_key, "execution_receipt": True, "owner_kind": row["kind"],
         **({"split_session": True} if split_session else {}), **row["turn_receipt"]},
    )


def _decode_execution_receipt(record: StoredRecord, *, session_kind: int, session_key: bytes) -> dict[str, object]:
    data = record.data
    proof_fields = {"owner_id", "request_digest", "receipt_id", "receipt_kind", "cancel_requested"}
    split_session = "split_session" in data
    if (data["execution_receipt"] is not True or split_session and (session_kind != SESSION or data["split_session"] is not True)
            or set(data) - {"split_session"} != proof_fields | {"scope", "execution_id", "session_key", "request_key", "execution_receipt", "owner_kind"}
            or data["owner_kind"] not in ("execution", "execution_shell")
            or session_kind not in (SESSION, STANDALONE) or record.state not in ("ACTIVE", "PARKED", "RETIRED")
            or session_kind == SESSION and not split_session and record.state == "PARKED"
            or record.lease_fence < 1 or record.lease_expires_at is None):
        raise _integrity()
    request = turn_hex(data["request_key"], TURN_REQUEST_MAX_BYTES)
    if session_kind == STANDALONE and request != session_key:
        raise _integrity()
    active, retained = record.state == "ACTIVE", record.state != "RETIRED"
    token = turn_hex(record.lease_owner, 32) if active else None
    if not active and record.lease_owner is not None:
        raise _integrity()
    execution = _bytes(data["execution_id"], TURN_EXECUTION_MAX_BYTES)
    receipt = {field: data[field] for field in proof_fields}
    if session_kind == SESSION:
        receipt["request_key"] = data["request_key"]
    if (type(receipt["cancel_requested"]) is not bool
            or receipt["receipt_kind"] not in (("claim", "renew", "cancel") if active else ("release", "expire"))
            or receipt["receipt_kind"] == "claim" and receipt["cancel_requested"]
            or receipt["receipt_kind"] == "cancel" and not receipt["cancel_requested"]):
        raise _integrity()
    for field, width in (("owner_id", 32), ("request_digest", 64), ("receipt_id", 32)):
        turn_hex(receipt[field], width)
    if active and receipt["receipt_kind"] == "claim" and token != receipt["receipt_id"].encode("ascii"):
        raise _integrity()
    return {
        "owner_kind": data["owner_kind"], "split_session": split_session,
        "turn_occupied_route": session_key if split_session and retained else None,
        "session_kind": session_kind, "session_key": session_key,
        "execution_id": execution, "turn_request_route": request if active or session_kind == STANDALONE and retained else None,
        "turn_receipt": receipt, "turn_token": token, "turn_fence": record.lease_fence,
        "turn_revision": record.storage_revision, "turn_expiry": _utc(record.lease_expires_at),
    }


def decode_turn_snapshot(
    records: Sequence[StoredRecord], *, namespace: str, tenant_id: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Validate logical cross-links before returning slot/admission restore values."""
    slots, execution_receipts = {}, {}
    requests = set()
    for record in records:
        if record.kind not in TURN_SNAPSHOT_KINDS:
            continue
        try:
            data = record.data
            scope = TurnScopeIdentity(namespace, tenant_id, data["scope"])
            if scope.scope != RUNTIME_TURN_SCOPE:
                raise _integrity()
            is_slot = record.kind == "turn_slot"
            identity = data["session_key"] if is_slot else data["execution_id"]
            key, grouped = _address(scope, record.kind, identity)
            if ((record.key, record.scope_key, record.sort_key) != (key, grouped, key)
                    or record.parent_key is not None or record.subject_id is not None
                    or record.payload_bytes is not None):
                raise _integrity()
            session_kind, session_key = turn_session_address(data["session_key"])
            if is_slot:
                active = record.state == "ACTIVE"
                if (set(data) != {"scope", "session_key", "execution_id", "session_record"}
                        or data["session_record"] is not True or session_kind != SESSION
                        or record.state not in ("ACTIVE", "IDLE") or (data["execution_id"] is not None) != active
                        or record.lease_owner is not None or record.lease_expires_at is not None
                        or record.storage_revision != 0 or active and record.lease_fence < 1):
                    raise _integrity()
                address = session_kind, session_key
                if address in slots:
                    raise _integrity()
                slots[address] = {
                    "session_kind": session_kind, "session_key": session_key, "lease_fence": record.lease_fence,
                    "active_execution_id": _bytes(data["execution_id"], TURN_EXECUTION_MAX_BYTES) if active else None,
                }
            else:
                receipt = _decode_execution_receipt(record, session_kind=session_kind, session_key=session_key)
                execution = receipt["execution_id"]
                route = receipt["turn_request_route"]
                if execution in execution_receipts or route is not None and route in requests:
                    raise _integrity()
                execution_receipts[execution] = receipt
                if route is not None:
                    requests.add(route)
        except (KeyError, TypeError, ValueError) as error:
            raise _integrity() from error

    standalone = [row for row in execution_receipts.values() if row["session_kind"] == STANDALONE]
    retained = {row["session_key"]: row["turn_fence"] for row in standalone
                if row["turn_request_route"] is not None}
    if any(row["turn_request_route"] is None and retained.get(row["session_key"], 0) <= row["turn_fence"]
           for row in standalone):
        raise _integrity()
    split_rows = [row for row in execution_receipts.values() if row["split_session"]]
    split_current = {}
    for row in split_rows:
        route = row["turn_occupied_route"]
        if (SESSION, row["session_key"]) in slots or route is not None and route in split_current:
            raise _integrity()
        if route is not None:
            split_current[route] = row["turn_fence"]
    if any(row["turn_occupied_route"] is None and split_current.get(row["session_key"], 0) <= row["turn_fence"]
           for row in split_rows):
        raise _integrity()
    for execution, receipt in execution_receipts.items():
        if receipt["session_kind"] == SESSION and not receipt["split_session"]:
            owner = slots.get((SESSION, receipt["session_key"]))
            if (owner is None or owner["lease_fence"] < receipt["turn_fence"]
                    or receipt["turn_token"] is not None and (
                        owner["active_execution_id"] != execution or owner["lease_fence"] != receipt["turn_fence"]
                    )):
                raise _integrity()
    for address, row in slots.items():
        if row["active_execution_id"] is not None:
            receipt = execution_receipts.get(row["active_execution_id"])
            if (receipt is None or (receipt["session_kind"], receipt["session_key"]) != address
                    or receipt["turn_token"] is None or receipt["turn_fence"] != row["lease_fence"]):
                raise _integrity()
    return list(slots.values()), list(execution_receipts.values())
