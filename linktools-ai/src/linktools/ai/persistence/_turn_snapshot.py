#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Portable execution-domain snapshots of source-owned SQL coordination."""

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone

from ..errors import AIError, ErrorCode
from ..storage import TurnScopeIdentity
from ._turn_sql import ACTIVE_REQUEST, RETIRED_EXECUTION, SqlTurnProof, turn_hex, turn_session_address, turn_session_identity
from ._turn_sql_schema import (
    TURN_ADMISSION_TABLE, TURN_EXECUTION_MAX_BYTES, TURN_REQUEST_MAX_BYTES,
    RUNTIME_TURN_SCOPE, TURN_SLOT_TABLE,
)
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


def turn_snapshot_record(table: str, row: Mapping[str, object], *, namespace: str, tenant_id: str) -> StoredRecord:
    """Encode one source row without persisting or exporting SQL surrogate IDs."""
    try:
        scope = TurnScopeIdentity(namespace, tenant_id, RUNTIME_TURN_SCOPE)
        if table == TURN_SLOT_TABLE:
            kind = "turn_slot"
            identity = turn_session_identity(row["session_kind"], row["session_key"])
            execution = row["active_execution_id"]
            token, expiry = row["lease_token"], row["lease_expires_at"]
            active = execution is not None
            if expiry is not None:
                if not isinstance(expiry, datetime):
                    raise _integrity()
                if expiry.tzinfo is None:
                    expiry = expiry.replace(tzinfo=timezone.utc)
                else:
                    expiry = expiry.astimezone(timezone.utc)
            key, grouped = _address(scope, kind, identity)
            return StoredRecord(
                key, grouped, None, kind, key, "ACTIVE" if active else "IDLE", row["revision"],
                None if token is None else token.decode("utf-8"), row["lease_fence"], expiry,
                {"scope": scope.scope, "session_key": identity,
                 "execution_id": execution.decode("utf-8") if active else None},
            )
        if table != TURN_ADMISSION_TABLE:
            raise _integrity()
        proof = SqlTurnProof.from_row(row)
        kind = "turn_admission"
        key, grouped = _address(scope, kind, proof.execution_id)
        return StoredRecord(key, grouped, None, kind, key, "ACTIVE" if proof.active else "RETIRED", 0, None, 0, None, {
            "scope": scope.scope, "execution_id": proof.execution_id, "session_key": proof.session_key,
            "request_key": proof.request_key, "request_digest": proof.request_digest, "owner_id": proof.owner_id,
            "fence": proof.fence, "cancel_requested": proof.cancel_requested,
            "receipt_id": proof.receipt_id, "receipt_kind": proof.receipt_kind,
            **({"coowned": True} if proof.active and row.get("control_id") is not None else {}),
        })
    except (AttributeError, KeyError, TypeError, ValueError) as error:
        raise _integrity() from error


def decode_turn_snapshot(
    records: Sequence[StoredRecord], *, namespace: str, tenant_id: str, store_key: bytes = b"",
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Validate logical cross-links before returning slot/admission restore values."""
    slots, admissions = {}, {}
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
            row = {"store_key": store_key, "session_kind": session_kind, "session_key": session_key}
            if is_slot:
                if set(data) != {"scope", "session_key", "execution_id"} or record.state not in ("ACTIVE", "IDLE"):
                    raise _integrity()
                active = record.state == "ACTIVE"
                contents = data["execution_id"], record.lease_owner, record.lease_expires_at
                if (active and (any(item is None for item in contents) or record.lease_fence < 1)
                        or not active and any(item is not None for item in contents)):
                    raise _integrity()
                row.update(
                    active_execution_id=_bytes(data["execution_id"], TURN_EXECUTION_MAX_BYTES) if active else None,
                    lease_token=turn_hex(record.lease_owner, 32) if active else None,
                    lease_fence=record.lease_fence, lease_expires_at=record.lease_expires_at,
                    revision=record.storage_revision,
                )
                identity = session_kind, session_key
                if identity in slots:
                    raise _integrity()
                slots[identity] = row
            else:
                if (("coowned" in data and (data["coowned"] is not True or record.state != "ACTIVE"))
                        or set(data) - {"coowned"} != {"scope", "session_key", "execution_id", "request_key", "request_digest",
                                  "owner_id", "fence", "cancel_requested", "receipt_id", "receipt_kind"}
                        or record.state not in ("ACTIVE", "RETIRED")
                        or record.storage_revision != 0 or record.lease_owner is not None
                        or record.lease_fence != 0 or record.lease_expires_at is not None):
                    raise _integrity()
                request = turn_hex(data["request_key"], TURN_REQUEST_MAX_BYTES)
                execution = _bytes(data["execution_id"], TURN_EXECUTION_MAX_BYTES)
                active = record.state == "ACTIVE"
                row.update(execution_id=execution, retired_request_key=None if active else request,
                           route_kind=ACTIVE_REQUEST if active else RETIRED_EXECUTION, route_key=request if active else execution,
                           request_digest=data["request_digest"], owner_id=turn_hex(data["owner_id"], 32), lease_fence=data["fence"],
                           cancel_requested=data["cancel_requested"], receipt_id=turn_hex(data["receipt_id"], 32),
                           receipt_kind=data["receipt_kind"])
                proof = SqlTurnProof.from_row(row)
                if (type(proof.cancel_requested) is not bool
                        or proof.receipt_kind == "claim" and proof.cancel_requested
                        or proof.receipt_kind == "cancel" and not proof.cancel_requested):
                    raise _integrity()
                identity = row["execution_id"]
                if identity in admissions or proof.active and request in requests:
                    raise _integrity()
                if data.get("coowned") is True:
                    row["_coowned"] = True
                admissions[identity] = row
                if proof.active:
                    requests.add(request)
        except (KeyError, TypeError, ValueError) as error:
            raise _integrity() from error

    for address, row in slots.items():
        if row["active_execution_id"] is None:
            continue
        linked = admissions.get(row["active_execution_id"])
        if (linked is None or linked["route_kind"] != ACTIVE_REQUEST
                or (linked["session_kind"], linked["session_key"]) != address or linked["lease_fence"] != row["lease_fence"]
                or linked["receipt_kind"] == "claim" and linked["receipt_id"] != row["lease_token"]):
            raise _integrity()
    for execution, row in admissions.items():
        slot = slots.get((row["session_kind"], row["session_key"]))
        if (slot is None or slot["lease_fence"] < row["lease_fence"]
                or row["route_kind"] == ACTIVE_REQUEST and slot["active_execution_id"] != execution):
            raise _integrity()
    return list(slots.values()), list(admissions.values())
