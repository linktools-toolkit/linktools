#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Exact scoped access to SQL turn slots and admissions."""

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Literal
from collections.abc import Callable, Mapping

from ..core import canonical_json_bytes
from ..errors import AIError, ErrorCode
from ..storage import Lease, TurnCoordination, TurnScopeIdentity
from ._turn_sql_schema import (
    TURN_ADMISSION_TABLE, TURN_EXECUTION_MAX_BYTES, TURN_REQUEST_MAX_BYTES,
    RUNTIME_TURN_SCOPE, TURN_SESSION_MAX_BYTES, TURN_SLOT_TABLE,
)

if TYPE_CHECKING:
    from sqlalchemy import Insert, MetaData, Select, Update
    from sqlalchemy.sql.elements import ColumnElement


Route = Literal["session", "execution", "request"]


ACTIVE_REQUEST, RETIRED_EXECUTION = 1, 2
SESSION, STANDALONE = 1, 2


def turn_hex(value: str, width: int) -> bytes:
    if type(value) is not str or len(value) != width or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"runtime turn identity must be {width} lowercase hex characters")
    return value.encode("ascii")


def turn_session_address(value: str) -> tuple[int, bytes]:
    # The reserved runtime protocol has exactly two explicit families. An
    # arbitrary adapter scope is never passed to this parser.
    family, delimiter, identity = value.partition(":")
    if not delimiter or family not in ("session", "request"):
        raise ValueError("invalid runtime turn slot family")
    if family == "request":
        return STANDALONE, turn_hex(identity, TURN_REQUEST_MAX_BYTES)
    key = identity.encode("utf-8")
    if not 1 <= len(key) <= TURN_SESSION_MAX_BYTES:
        raise ValueError("turn session identity exceeds its byte limit")
    return SESSION, key


def turn_session_identity(kind: int, key: bytes) -> str:
    if kind not in (SESSION, STANDALONE) or type(key) is not bytes:
        raise ValueError("invalid stored runtime turn slot")
    value = ("session:" if kind == SESSION else "request:") + key.decode("utf-8")
    if turn_session_address(value) != (kind, key):
        raise ValueError("invalid stored runtime turn slot")
    return value


@dataclass(frozen=True, slots=True)
class SqlTurnProof:
    execution_id: str
    session_key: str
    request_key: str
    request_digest: str
    owner_id: str
    fence: int
    cancel_requested: bool
    receipt_id: str
    receipt_kind: str
    active: bool

    @classmethod
    def from_row(cls, row: Mapping[str, object]) -> "SqlTurnProof":
        try:
            active = row["route_kind"] == ACTIVE_REQUEST
            receipt_kind = row["receipt_kind"]
            if (row["route_kind"] not in (ACTIVE_REQUEST, RETIRED_EXECUTION)
                    or (active and row["retired_request_key"] is not None)
                    or (not active and row["route_key"] != row["execution_id"])
                    or type(row["cancel_requested"]) is not bool
                    or receipt_kind not in (("claim", "renew", "cancel") if active else ("release", "expire"))
                    or type(row["lease_fence"]) is not int or row["lease_fence"] <= 0):
                raise ValueError("invalid coordination proof")
            def hex_field(name: str, width: int) -> str:
                value = row[name].decode("ascii")
                turn_hex(value, width)
                return value
            request_digest = row["request_digest"]
            turn_hex(request_digest, 64)
            execution = row["execution_id"]
            if type(execution) is not bytes or not 1 <= len(execution) <= TURN_EXECUTION_MAX_BYTES:
                raise ValueError("invalid stored execution identity")
            return cls(
                execution.decode("utf-8"), turn_session_identity(row["session_kind"], row["session_key"]),
                hex_field("route_key" if active else "retired_request_key", TURN_REQUEST_MAX_BYTES), request_digest, hex_field("owner_id", 32),
                row["lease_fence"], row["cancel_requested"], hex_field("receipt_id", 32), receipt_kind, active,
            )
        except (KeyError, TypeError, AttributeError, ValueError) as error:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error

    def validate_live_token(self, token: bytes) -> None:
        try:
            turn_hex(token.decode("ascii"), 32)
        except (UnicodeError, ValueError) as error:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
        if (not self.active or self.receipt_kind == "claim" and self.receipt_id.encode("utf-8") != token
                or self.receipt_kind == "cancel" and not self.cancel_requested):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

    def coordination(self, scope: TurnScopeIdentity, *, token: str | None = None) -> TurnCoordination:
        lease = None if token is None else Lease(canonical_json_bytes([
            "state-turn", 1, *scope.as_tuple(), self.execution_id, self.session_key, self.request_key,
        ]).decode("utf-8"), self.fence, token)
        return TurnCoordination(self.execution_id, self.owner_id, self.cancel_requested, lease)


@dataclass(frozen=True, slots=True)
class SqlTurnObservation:
    proof: SqlTurnProof
    slot_revision: int
    expired: bool


class SqlTurnNotOwned(AIError):
    """Roll back conditional writes when the exact live ownership cannot be proven."""

    def __init__(self) -> None:
        super().__init__(ErrorCode.STORAGE_CONFLICT)


class SqlTurnRows:
    """Statement construction for one physical store and logical coordination scope."""

    def __init__(self, metadata: "MetaData", store_key: bytes, scope: TurnScopeIdentity) -> None:
        self._slot = metadata.tables[TURN_SLOT_TABLE]
        self._admission = metadata.tables[TURN_ADMISSION_TABLE]
        if type(store_key) is not bytes or len(store_key) != 32:
            raise ValueError("typed turn SQL requires a runtime partition")
        self._store_key = store_key
        if scope.scope != RUNTIME_TURN_SCOPE:
            raise ValueError("typed turn SQL accepts only the fixed runtime scope")

    def _owns(self, table):
        return table.c.store_key == self._store_key

    @staticmethod
    def _session(table, address: tuple[int, bytes]):
        from sqlalchemy import and_

        return and_(table.c.session_kind == address[0], table.c.session_key == address[1])

    @staticmethod
    def _key(value: str, limit: int) -> bytes:
        raw = value.encode("utf-8")
        if not 1 <= len(raw) <= limit:
            raise ValueError("turn identity exceeds its byte limit")
        return raw

    def select_proof(self, execution_id: str) -> "Select":
        """Retained latest execution proof, independent of the current session slot."""
        from sqlalchemy import select

        admission = self._admission
        return select(admission).where(
            self._owns(admission), admission.c.execution_id == self._key(execution_id, TURN_EXECUTION_MAX_BYTES),
        )

    def select_live(
        self, *, route: Route, identity: str, clock: "ColumnElement",
    ) -> "Select":
        """Read the anchor and match separately so a broken live link fails closed."""
        from sqlalchemy import and_, select

        s, a = self._slot, self._admission
        linked = and_(a.c.store_key == s.c.store_key, a.c.session_kind == s.c.session_kind,
                      a.c.session_key == s.c.session_key, a.c.execution_id == s.c.active_execution_id)
        if route == "session":
            joined = s.outerjoin(a, linked)
            anchor, predicate = s, self._session(s, turn_session_address(identity))
        else:
            joined = a.outerjoin(s, linked)
            anchor = a
            predicate = (a.c.execution_id == self._key(identity, TURN_EXECUTION_MAX_BYTES) if route == "execution"
                         else and_(a.c.route_kind == ACTIVE_REQUEST, a.c.route_key == turn_hex(identity, TURN_REQUEST_MAX_BYTES)))
        return select(
            a, s.c.session_key.label("slot_session_key"), s.c.session_kind.label("slot_session_kind"), s.c.active_execution_id,
            s.c.lease_fence.label("slot_fence"), s.c.revision.label("slot_revision"),
            s.c.lease_expires_at, s.c.lease_token, clock.label("observed_at"),
        ).select_from(joined).where(self._owns(anchor), predicate)

    @staticmethod
    def parse_observation(
        row: Mapping[str, object], *, decode_time: Callable[[object], datetime],
    ) -> SqlTurnObservation | None:
        if row["active_execution_id"] is None and row["execution_id"] is None:
            if row["lease_token"] is not None or row["lease_expires_at"] is not None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            return None  # A present idle slot.
        if row["execution_id"] is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)  # A live slot whose admission is missing.
        proof = SqlTurnProof.from_row(row)
        if not proof.active:
            if row["active_execution_id"] is not None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)  # Slot points to a retired admission.
            return None
        if (row["slot_session_kind"] != row["session_kind"] or row["slot_session_key"] != row["session_key"] or row["active_execution_id"] != row["execution_id"]
                or type(row["lease_token"]) is not bytes or not row["lease_token"]
                or row["slot_fence"] != proof.fence):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        proof.validate_live_token(row["lease_token"])
        revision = row["slot_revision"]
        if type(revision) is not int or revision < 0:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return SqlTurnObservation(proof, revision, decode_time(row["lease_expires_at"]) <= decode_time(row["observed_at"]))

    @staticmethod
    def live_proof(row: Mapping[str, object], *, decode_time: Callable[[object], datetime]) -> SqlTurnProof | None:
        observed = SqlTurnRows.parse_observation(row, decode_time=decode_time)
        return None if observed is None or observed.expired else observed.proof

    def select_claim(self, *, session_key: str, request_key: str, execution_id: str, clock: "ColumnElement") -> "Select":
        """Read at most two involved slots and three admission identities in one statement."""
        from sqlalchemy import and_, literal, select

        seed = select(literal(1).label("seed")).subquery()
        target = self._slot.alias("target")
        incumbent = self._slot.alias("incumbent")
        live_request = self._admission.alias("live_request")
        target_owner = self._admission.alias("target_owner")
        proposed = self._admission.alias("proposed")
        joined = seed.outerjoin(target, and_(self._owns(target), self._session(target, turn_session_address(session_key)))).outerjoin(
            live_request, and_(self._owns(live_request), live_request.c.route_kind == ACTIVE_REQUEST,
                live_request.c.route_key == turn_hex(request_key, TURN_REQUEST_MAX_BYTES)),
        ).outerjoin(incumbent, and_(self._owns(incumbent), incumbent.c.session_kind == live_request.c.session_kind, incumbent.c.session_key == live_request.c.session_key)
        ).outerjoin(target_owner, and_(self._owns(target_owner), target_owner.c.execution_id == target.c.active_execution_id)
        ).outerjoin(proposed, and_(self._owns(proposed), proposed.c.execution_id == self._key(execution_id, TURN_EXECUTION_MAX_BYTES)))
        return select(clock.label("observed_at"), *(
            column.label(table.name + "__" + column.name)
            for table in (target, incumbent, live_request, target_owner, proposed)
            for column in table.c if column.name not in {"created_at", "updated_at", "store_key"}
        )).select_from(joined)

    def create_slot(self, session: str) -> "Insert":
        from sqlalchemy import insert

        kind, key = turn_session_address(session)
        return insert(self._slot).values(store_key=self._store_key, session_kind=kind, session_key=key)

    def guard_slot_revision(self, session: tuple[int, bytes], revision: int) -> "Update":
        from sqlalchemy import update

        slot = self._slot
        return update(slot).where(self._owns(slot), self._session(slot, session), slot.c.revision == revision
                                  ).values(revision=revision + 1)

    def bind_slot(self, session: tuple[int, bytes], *, execution: str | None, token: str | None,
                  fence: int, expiry: datetime | None) -> "Update":
        from sqlalchemy import update

        slot = self._slot
        return update(slot).where(self._owns(slot), self._session(slot, session)).values(
            active_execution_id=None if execution is None else self._key(execution, TURN_EXECUTION_MAX_BYTES),
            lease_token=None if token is None else turn_hex(token, 32), lease_fence=fence, lease_expires_at=expiry,
        )

    def publish_claim(self, proof: SqlTurnProof, *, previous: SqlTurnProof | None) -> "Insert | Update":
        from sqlalchemy import insert, update

        a = self._admission
        request = turn_hex(proof.request_key, TURN_REQUEST_MAX_BYTES)
        execution = self._key(proof.execution_id, TURN_EXECUTION_MAX_BYTES)
        kind, key = turn_session_address(proof.session_key)
        values = dict(session_kind=kind, session_key=key, retired_request_key=None,
                      route_kind=ACTIVE_REQUEST, route_key=request, request_digest=proof.request_digest,
                      owner_id=turn_hex(proof.owner_id, 32), lease_fence=proof.fence, cancel_requested=False,
                      receipt_id=turn_hex(proof.receipt_id, 32), receipt_kind="claim")
        if previous is None:
            return insert(a).values(store_key=self._store_key, execution_id=execution, **values)
        return update(a).where(
            self._owns(a), a.c.execution_id == execution, a.c.route_kind == RETIRED_EXECUTION, a.c.route_key == execution,
            self._session(a, turn_session_address(previous.session_key)),
            a.c.retired_request_key == turn_hex(previous.request_key, TURN_REQUEST_MAX_BYTES),
            a.c.lease_fence == previous.fence, a.c.receipt_id == turn_hex(previous.receipt_id, 32),
            a.c.receipt_kind == previous.receipt_kind,
        ).values(**values)

    def refresh_admission(self, *, execution_id: str, session_key: str, request_key: str,
                          fence: int, receipt: str, kind: str) -> "Update":
        from sqlalchemy import update

        a = self._admission
        values = dict(receipt_id=turn_hex(receipt, 32), receipt_kind=kind)
        if kind == "cancel":
            values["cancel_requested"] = True
        return update(a).where(
            self._owns(a), a.c.execution_id == self._key(execution_id, TURN_EXECUTION_MAX_BYTES),
            self._session(a, turn_session_address(session_key)), a.c.lease_fence == fence,
            a.c.route_kind == ACTIVE_REQUEST, a.c.route_key == turn_hex(request_key, TURN_REQUEST_MAX_BYTES),
        ).values(**values)

    def clear_serialized_slot(self, *, session_key: str, clock: "ColumnElement", sqlite: bool) -> "Update":
        """Check clock in a statement issued after slot-write serialization."""
        from sqlalchemy import func, update

        slot = self._slot
        expiry = slot.c.lease_expires_at
        # SQLite DateTime serialization and the authoritative clock use distinct
        # ISO separators. Compare times, not their incompatible string forms.
        live = func.julianday(expiry) > func.julianday(clock) if sqlite else expiry > clock
        return update(slot).where(
            self._owns(slot), self._session(slot, turn_session_address(session_key)), live,
        ).values(active_execution_id=None, lease_token=None, lease_expires_at=None)

    def retire_admission(self, *, execution_id: str, session_key: str, request_key: str,
                         fence: int, receipt: str, token: str, kind: str = "release") -> "Update":
        from sqlalchemy import and_, or_, update

        admission = self._admission
        request = turn_hex(request_key, TURN_REQUEST_MAX_BYTES)
        return update(admission).where(
            self._owns(admission), admission.c.execution_id == self._key(execution_id, TURN_EXECUTION_MAX_BYTES),
            self._session(admission, turn_session_address(session_key)),
            admission.c.lease_fence == fence,
            admission.c.retired_request_key.is_(None), admission.c.route_kind == ACTIVE_REQUEST, admission.c.route_key == request,
            or_(and_(admission.c.receipt_kind == "claim", admission.c.receipt_id == turn_hex(token, 32)),
                admission.c.receipt_kind == "renew",
                and_(admission.c.receipt_kind == "cancel", admission.c.cancel_requested)),
        ).values(route_kind=RETIRED_EXECUTION, route_key=admission.c.execution_id, retired_request_key=request,
                 receipt_id=turn_hex(receipt, 32), receipt_kind=kind)

    def serialize_slot(self, *, session_key: str, execution_id: str, token: str, fence: int) -> "Update":
        """Acquire real write serialization; expiry must still be tested after waiting."""
        from sqlalchemy import update

        slot = self._slot
        return update(slot).where(
            self._owns(slot), self._session(slot, turn_session_address(session_key)),
            slot.c.active_execution_id == self._key(execution_id, TURN_EXECUTION_MAX_BYTES),
            slot.c.lease_token == turn_hex(token, 32), slot.c.lease_fence == fence,
        ).values(revision=slot.c.revision + 1)


__all__ = ["SqlTurnProof", "SqlTurnRows", "SqlTurnObservation", "SqlTurnNotOwned"]
