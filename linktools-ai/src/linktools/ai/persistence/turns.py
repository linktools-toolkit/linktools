#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Source-owned live turn admissions, independent of disposable display content."""

from __future__ import annotations

import asyncio
import json
import math
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import timedelta
from typing import TypeVar
from uuid import uuid4

from ..core import JsonValue, canonical_json_bytes, canonical_sha256, validate_persistence_namespace, validate_tenant_id
from ..errors import AIError, ErrorCode
from ..storage import Lease, TurnCoordination
from .state import (
    RecordLeaseTransition, StateStore, StateTransaction, StateTransactionNestingError,
    StoredRecord, active_state_scope, record_key_digest, scope_digest,
)

_Value = TypeVar("_Value")
_KINDS = frozenset({"turn_session", "turn_request", "turn_execution"})
_FIELDS = {
    "turn_session": {"format", "scope", "session_key", "execution_id", "request_key"},
    "turn_request": {"format", "scope", "session_key", "request_key", "request_digest", "execution_id"},
    "turn_execution": {"format", "scope", "session_key", "request_key", "request_digest", "execution_id",
                       "owner_id", "token", "fence", "cancel_requested", "receipt_id", "receipt_kind"},
}
_IDENTITIES = {"turn_session": "session_key", "turn_request": "request_key", "turn_execution": "execution_id"}


def _integrity() -> AIError:
    return AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)


def _text(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("turn identities must be nonempty strings")
    return value


def _duration(seconds: float) -> float:
    if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("turn lease must be finite and positive")
    if seconds > (2 ** 52) / 1000:
        raise ValueError("turn lease exceeds timestamp precision")
    rounded = math.ceil(seconds * 1000) / 1000
    try:
        timedelta(seconds=rounded)
    except OverflowError as error:
        raise ValueError("turn lease duration is not representable") from error
    return rounded


class _TurnRecords:
    def __init__(self, namespace: str, tenant_id: str) -> None:
        self.namespace = validate_persistence_namespace(namespace)
        self.tenant_id = validate_tenant_id(tenant_id)

    def key(self, scope: str, kind: str, identity: str) -> bytes:
        return record_key_digest(self.namespace, self.tenant_id, "execution", kind, [scope, identity])

    def make(self, kind: str, data: Mapping[str, JsonValue], *, state: str) -> StoredRecord:
        scope, identity = _text(data["scope"]), _text(data[_IDENTITIES[kind]])
        return StoredRecord(
            self.key(scope, kind, identity),
            scope_digest(self.namespace, self.tenant_id, "execution", "turn", "scope", scope),
            None, kind, canonical_sha256([scope, identity]), state, 0, None, 0, None, dict(data),
        )

    def validate(self, record: StoredRecord) -> StoredRecord:
        try:
            data = record.data
            if record.kind not in _KINDS or set(data) != _FIELDS[record.kind] or type(data["format"]) is not int or data["format"] != 1:
                raise _integrity()
            for key in _FIELDS[record.kind] - {"format", "fence", "cancel_requested"}:
                if record.kind == "turn_session" and key in {"execution_id", "request_key"} and data[key] is None:
                    continue
                _text(data[key])
            expected = self.make(record.kind, data, state=record.state)
            if (record.key_digest != expected.key_digest or record.scope_digest != expected.scope_digest
                    or record.parent_digest is not None or record.sort_key != expected.sort_key):
                raise _integrity()
            if record.kind == "turn_session":
                if record.state == "IDLE":
                    if (data["execution_id"] is not None or data["request_key"] is not None
                            or record.lease_owner is not None or record.lease_expires_at is not None):
                        raise _integrity()
                elif (record.state != "ACTIVE" or data["execution_id"] is None or data["request_key"] is None
                      or not record.lease_owner or record.lease_fence < 1 or record.lease_expires_at is None):
                    raise _integrity()
            else:
                if record.state not in {"ACTIVE", "RETIRED"} or (
                    record.lease_owner is not None or record.lease_fence != 0 or record.lease_expires_at is not None
                ):
                    raise _integrity()
            if record.kind == "turn_execution":
                if (isinstance(data["fence"], bool) or not isinstance(data["fence"], int) or data["fence"] < 1
                        or not isinstance(data["cancel_requested"], bool)
                        or data["receipt_kind"] not in {"claim", "renew", "cancel", "release", "expire"}):
                    raise _integrity()
                receipt_kind = data["receipt_kind"]
                if (record.state == "ACTIVE" and receipt_kind not in {"claim", "renew", "cancel"}
                        or record.state == "RETIRED" and receipt_kind not in {"release", "expire"}
                        or receipt_kind == "cancel" and not data["cancel_requested"]
                        or receipt_kind == "claim" and (data["receipt_id"] != data["token"] or data["cancel_requested"])):
                    raise _integrity()
        except (KeyError, TypeError, ValueError) as error:
            raise _integrity() from error
        return record

    def graph(self, execution: StoredRecord, request: StoredRecord, session: StoredRecord) -> _Graph:
        for record in (execution, request, session):
            self.validate(record)
        e, r, s = execution.data, request.data, session.data
        if (execution.state != "ACTIVE" or request.state != "ACTIVE" or session.state != "ACTIVE"
                or any(e[key] != r[key] for key in ("scope", "session_key", "request_key", "request_digest", "execution_id"))
                or any(e[key] != s[key] for key in ("scope", "session_key", "request_key", "execution_id"))
                or session.lease_owner != e["token"] or session.lease_fence != e["fence"]):
            raise _integrity()
        return _Graph(execution, request, session)


@dataclass(frozen=True, slots=True)
class _Graph:
    execution: StoredRecord
    request: StoredRecord
    session: StoredRecord

    def signature(self) -> bytes:
        return canonical_json_bytes([
            [record.kind, record.state, dict(record.data), record.lease_owner, record.lease_fence,
             None if record.lease_expires_at is None else record.lease_expires_at.isoformat()]
            for record in (self.execution, self.request, self.session)
        ])


class StateTurnCoordination:
    """Atomic S/R/E admissions over one borrowed, initialized StateStore.

    Session anchors and execution receipts are retained; execution IDs and owner
    IDs must never be reused. Fences advance per session; only the complete
    opaque lease key, token and fence identify an owner. Authority reads use the
    source, never a display cache.
    Lease durations round upward to milliseconds. Stored authority must only be
    restored as a complete offline resource, never rolled back beneath a live
    producer or cloned into concurrent writers. This object supplies only turn
    authority. Runtime selection also requires a host-owned CacheBackend on the
    same configured object: explicitly bind that thin composition once, after
    Persistence is ready and before attaching any Runtime. Never fall back to
    the display cache's authority methods while the source is unbound.
    """

    def __init__(
        self, store: StateStore, *, namespace: str, tenant_id: str,
        read_only: bool = False, require_ready: Callable[[], None] | None = None,
    ) -> None:
        self._read_only = read_only
        self._require_ready = require_ready
        self._store = store
        self._records = _TurnRecords(namespace, tenant_id)

    def _outside_transaction(self) -> None:
        if self._require_ready is not None:
            self._require_ready()
        if active_state_scope() is not None:
            raise StateTransactionNestingError("turn authority requires independent source transaction scopes")

    def _writable(self) -> None:
        self._outside_transaction()
        if self._read_only:
            raise StateTransactionNestingError("read-only turn authority cannot mutate admissions")

    async def _load(self, tx: StateTransaction, scope: str, kind: str, identity: str) -> StoredRecord | None:
        record = await tx.get_record(self._records.key(scope, kind, identity))
        return None if record is None else self._records.validate(record)

    async def _graph(self, tx: StateTransaction, execution: StoredRecord) -> _Graph:
        scope = _text(execution.data["scope"])
        request = await self._load(tx, scope, "turn_request", _text(execution.data["request_key"]))
        session = await self._load(tx, scope, "turn_session", _text(execution.data["session_key"]))
        if request is None or session is None:
            raise _integrity()
        return self._records.graph(execution, request, session)

    async def _anchor(self, tx: StateTransaction, scope: str, kind: str, identity: str) -> _Graph | None:
        anchor = await self._load(tx, scope, kind, identity)
        if anchor is None or anchor.state == "IDLE":
            return None
        if kind == "turn_execution":
            execution = anchor
        else:
            execution = await self._load(tx, scope, "turn_execution", _text(anchor.data["execution_id"]))
            if execution is None:
                raise _integrity()
        if anchor.state == "RETIRED":
            if execution.state != "RETIRED" or any(
                anchor.data[key] != execution.data[key]
                for key in ({"scope", "execution_id"} if kind == "turn_execution"
                            else {"scope", "execution_id", "session_key", "request_key", "request_digest"})
            ):
                raise _integrity()
            return None
        return await self._graph(tx, execution)

    async def _observe(self, scope: str, kind: str, identity: str) -> _Graph | None:
        self._outside_transaction()
        _text(scope)
        _text(identity)
        for attempt in range(3):
            first = await self._store.read(lambda tx: self._anchor(tx, scope, kind, identity))
            if first is None:
                return None
            sampled = await self._store.read(lambda tx: tx.now())
            second = await self._store.read(lambda tx: self._anchor(tx, scope, kind, identity))
            if second is None:
                return None
            if first.signature() == second.signature():
                return None if second.session.lease_expires_at <= sampled else second
            if attempt < 2:
                await asyncio.sleep(0)
        raise AIError(ErrorCode.STORAGE_CONFLICT)

    def _coordination(self, graph: _Graph, *, owned: bool = False) -> TurnCoordination:
        data = graph.execution.data
        lease = None
        if owned:
            lease = Lease(canonical_json_bytes([
                "state-turn", 1, self._records.namespace, self._records.tenant_id,
                data["scope"], data["execution_id"],
            ]).decode("utf-8"), data["fence"], data["token"])
        return TurnCoordination(data["execution_id"], data["owner_id"], data["cancel_requested"], lease)

    async def read_turn(self, scope: str, execution_id: str) -> TurnCoordination | None:
        graph = await self._observe(scope, "turn_execution", execution_id)
        return None if graph is None else self._coordination(graph)

    async def read_turn_by_request(self, scope: str, request_key: str) -> TurnCoordination | None:
        graph = await self._observe(scope, "turn_request", request_key)
        return None if graph is None else self._coordination(graph)

    async def read_turn_by_session(self, scope: str, session_key: str) -> TurnCoordination | None:
        graph = await self._observe(scope, "turn_session", session_key)
        return None if graph is None else self._coordination(graph)

    async def _replace(self, tx: StateTransaction, record: StoredRecord, **changes: object) -> StoredRecord:
        candidate = replace(record, storage_revision=record.storage_revision + 1, **changes)
        if not await tx.replace_record(candidate, expected_storage_revision=record.storage_revision):
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        return candidate

    async def _retire(self, tx: StateTransaction, graph: _Graph, *, receipt: str, kind: str) -> StoredRecord:
        execution = await self._replace(tx, graph.execution, state="RETIRED", data={
            **graph.execution.data, "receipt_id": receipt, "receipt_kind": kind,
        })
        await self._replace(tx, graph.request, state="RETIRED")
        session = await self._replace(tx, graph.session, state="IDLE", lease_owner=None, lease_expires_at=None,
                                      data={**graph.session.data, "execution_id": None, "request_key": None})
        self._records.validate(execution)
        return self._records.validate(session)

    async def _mutation(
        self, callback: Callable[[StateTransaction], Awaitable[_Value]],
        readback: Callable[[], Awaitable[_Value | None]],
    ) -> _Value:
        self._writable()
        for attempt in range(3):
            try:
                return await self._store.mutate(callback)
            except AIError as error:
                if error.code is ErrorCode.STORAGE_CONFLICT and attempt < 2:
                    await asyncio.sleep(0)
                    continue
                if error.code is ErrorCode.STORAGE_COMMIT_UNKNOWN:
                    try:
                        confirmed = await readback()
                    except Exception as read_error:
                        raise error from read_error
                    if confirmed is not None:
                        return confirmed
                raise
        raise AssertionError("unreachable mutation attempt")

    async def _confirmed(
        self, scope: str, execution_id: str, *, receipt: str, kind: str,
    ) -> TurnCoordination | bool | None:
        execution = await self._store.read(lambda tx: self._load(tx, scope, "turn_execution", execution_id))
        if execution is None:
            return None
        if kind == "release":
            return (True if execution.state == "RETIRED" and execution.data["receipt_id"] == receipt
                    and execution.data["receipt_kind"] == kind else None)
        graph = await self._observe(scope, "turn_execution", execution_id)
        if graph is None:
            return None
        data = graph.execution.data
        if kind == "claim":
            if data["token"] != receipt:
                return None
        elif data["receipt_id"] != receipt or data["receipt_kind"] != kind:
            return None
        if kind == "cancel":
            return True if data["cancel_requested"] else None
        return self._coordination(graph, owned=True)

    async def claim_turn(
        self, scope: str, *, session_key: str, request_key: str, request_digest: str,
        execution_id: str, owner_id: str, lease_seconds: float,
    ) -> TurnCoordination | None:
        for value in (scope, session_key, request_key, request_digest, execution_id, owner_id):
            _text(value)
        duration = _duration(lease_seconds)
        token = uuid4().hex

        async def claim(tx: StateTransaction) -> TurnCoordination | None:
            session = await self._load(tx, scope, "turn_session", session_key)
            request = await self._load(tx, scope, "turn_request", request_key)
            proposed = await self._load(tx, scope, "turn_execution", execution_id)
            graphs: dict[bytes, _Graph] = {}
            for kind, anchor in (("turn_session", session), ("turn_request", request)):
                if anchor is not None and anchor.state == "ACTIVE":
                    graph = await self._anchor(tx, scope, kind, _text(anchor.data[_IDENTITIES[kind]]))
                    if graph is None:
                        raise _integrity()
                    graphs[graph.session.key_digest] = graph
                elif kind == "turn_request" and anchor is not None:
                    await self._anchor(tx, scope, kind, request_key)
            snapshot_time = await tx.now()
            protected: dict[bytes, tuple[StoredRecord, bool]] = {}
            sessions = {key: graph.session for key, graph in graphs.items()}
            if session is not None:
                sessions[session.key_digest] = session
            for key, current in sorted(sessions.items()):
                live = (current.lease_owner is not None and current.lease_expires_at is not None
                        and current.lease_expires_at > snapshot_time)
                guarded = await tx.transition_record_lease(
                    key, expected_storage_revision=current.storage_revision,
                    transition=RecordLeaseTransition("guard" if live else "guard_available",
                                                     current.lease_owner, current.lease_fence),
                )
                if guarded is None:
                    raise AIError(ErrorCode.STORAGE_CONFLICT)
                protected[key] = guarded, live
            if request is not None and request.state == "ACTIVE":
                graph = graphs[self._records.key(scope, "turn_session", _text(request.data["session_key"]))]
                guarded, live = protected[graph.session.key_digest]
                if live:
                    if request.data["session_key"] != session_key or request.data["request_digest"] != request_digest:
                        raise AIError(ErrorCode.IDEMPOTENCY_CONFLICT)
                    return self._coordination(replace(graph, session=guarded))
            if session is not None and protected[session.key_digest][1]:
                return None
            if proposed is not None:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            for key, graph in sorted(graphs.items()):
                guarded, live = protected[key]
                if live:
                    raise _integrity()
                retired = await self._retire(tx, replace(graph, session=guarded), receipt=token, kind="expire")
                if key == self._records.key(scope, "turn_session", session_key):
                    session = retired
            if session is None:
                session = self._records.make("turn_session", {
                    "format": 1, "scope": scope, "session_key": session_key,
                    "execution_id": None, "request_key": None,
                }, state="IDLE")
                await tx.insert_record(session)
            elif session.key_digest not in graphs:
                session = protected[session.key_digest][0]
            acquired = await tx.transition_record_lease(
                session.key_digest, expected_storage_revision=session.storage_revision,
                transition=RecordLeaseTransition("acquire", token, session.lease_fence, duration),
            )
            if acquired is None:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            common: dict[str, JsonValue] = {
                "format": 1, "scope": scope, "session_key": session_key, "request_key": request_key,
                "request_digest": request_digest, "execution_id": execution_id,
            }
            execution = self._records.make("turn_execution", {
                **common, "owner_id": owner_id, "token": token, "fence": acquired.lease_fence,
                "cancel_requested": False, "receipt_id": token, "receipt_kind": "claim",
            }, state="ACTIVE")
            await tx.insert_record(execution)
            request = await self._load(tx, scope, "turn_request", request_key)
            if request is None:
                request = self._records.make("turn_request", common, state="ACTIVE")
                await tx.insert_record(request)
            else:
                request = await self._replace(tx, request, data=common, state="ACTIVE")
            session = await self._replace(tx, acquired, state="ACTIVE", data={
                **acquired.data, "execution_id": execution_id, "request_key": request_key,
            })
            return self._coordination(self._records.graph(execution, request, session), owned=True)

        return await self._mutation(claim, lambda: self._confirmed(
            scope, execution_id, receipt=token, kind="claim",
        ))

    def _lease_identity(self, lease: Lease) -> tuple[str, str] | None:
        try:
            if (isinstance(lease.fence, bool) or not isinstance(lease.fence, int) or lease.fence < 1
                    or not isinstance(lease.token, str) or not lease.token):
                return None
            value = json.loads(lease.key)
            if (not isinstance(value, list) or len(value) != 6 or value[:2] != ["state-turn", 1]
                    or isinstance(value[1], bool)
                    or value[2:4] != [self._records.namespace, self._records.tenant_id]):
                return None
            return _text(value[4]), _text(value[5])
        except (ValueError, TypeError):
            return None

    async def _change(
        self, scope: str, execution_id: str, *, kind: str, lease: Lease | None = None,
        duration: float | None = None,
    ) -> TurnCoordination | bool | None:
        self._writable()
        _text(scope)
        _text(execution_id)
        receipt = uuid4().hex

        async def change(tx: StateTransaction) -> TurnCoordination | bool | None:
            execution = await self._load(tx, scope, "turn_execution", execution_id)
            if execution is None or execution.state == "RETIRED":
                return None if kind == "renew" else False
            if lease is not None and (execution.data["token"] != lease.token or execution.data["fence"] != lease.fence):
                return None if kind == "renew" else False
            graph = await self._graph(tx, execution)
            session = await tx.transition_record_lease(
                graph.session.key_digest, expected_storage_revision=graph.session.storage_revision,
                transition=RecordLeaseTransition("guard" if kind == "cancel" else kind,
                                                 graph.session.lease_owner, graph.session.lease_fence, duration),
            )
            if session is None:
                return None if kind == "renew" else False
            graph = replace(graph, session=session)
            if kind == "release":
                await self._retire(tx, graph, receipt=receipt, kind=kind)
                return True
            execution = await self._replace(tx, execution, data={
                **execution.data, "receipt_id": receipt, "receipt_kind": kind,
                "cancel_requested": kind == "cancel" or execution.data["cancel_requested"],
            })
            graph = self._records.graph(execution, graph.request, session)
            return True if kind == "cancel" else self._coordination(graph, owned=True)

        return await self._mutation(change, lambda: self._confirmed(
            scope, execution_id, receipt=receipt, kind=kind,
        ))

    async def renew_turn(self, lease: Lease, *, lease_seconds: float) -> TurnCoordination | None:
        self._writable()
        duration = _duration(lease_seconds)
        identity = self._lease_identity(lease)
        if identity is None:
            return None
        return await self._change(*identity, kind="renew", lease=lease, duration=duration)

    async def release_turn(self, lease: Lease) -> bool:
        self._writable()
        identity = self._lease_identity(lease)
        if identity is None:
            return False
        return await self._change(*identity, kind="release", lease=lease)

    async def cancel_turn(self, scope: str, execution_id: str) -> bool:
        return await self._change(scope, execution_id, kind="cancel")


def validate_turn_records(records: Sequence[StoredRecord], *, namespace: str, tenant_id: str) -> None:
    """Validate the complete source-owned admission graph in an offline snapshot."""
    codec = _TurnRecords(namespace, tenant_id)
    owned = tuple(record for record in records if record.kind in _KINDS)
    values = {record.key_digest: codec.validate(record) for record in owned}
    if len(values) != len(owned):
        raise _integrity()
    for record in values.values():
        if record.state != "ACTIVE":
            if record.kind == "turn_request":
                execution = values.get(codec.key(_text(record.data["scope"]), "turn_execution",
                                                _text(record.data["execution_id"])))
                if execution is None or execution.state != "RETIRED" or any(
                    execution.data[key] != record.data[key]
                    for key in ("scope", "session_key", "request_key", "request_digest", "execution_id")
                ):
                    raise _integrity()
            continue
        scope = _text(record.data["scope"])
        execution = (record if record.kind == "turn_execution" else
                     values.get(codec.key(scope, "turn_execution", _text(record.data["execution_id"])) ))
        if execution is None:
            raise _integrity()
        request = values.get(codec.key(scope, "turn_request", _text(execution.data["request_key"])))
        session = values.get(codec.key(scope, "turn_session", _text(execution.data["session_key"])))
        if request is None or session is None:
            raise _integrity()
        codec.graph(execution, request, session)


__all__ = ["StateTurnCoordination", "validate_turn_records"]
