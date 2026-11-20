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
from ._record_key import encode_key_components
from ._turn_sql_schema import (
    TURN_ADMISSION_TABLE, TURN_EXECUTION_MAX_BYTES, TURN_REQUEST_MAX_BYTES,
    TURN_SCOPE_MAX_BYTES, TURN_SESSION_MAX_BYTES, TURN_SLOT_TABLE,
)
from .state import validate_store_key

if TYPE_CHECKING:
    from sqlalchemy import Insert, MetaData, Select, Update
    from sqlalchemy.sql.elements import ColumnElement


Route = Literal["session", "execution", "request"]


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
            active = row["live_request_key"] is not None
            receipt_kind = row["receipt_kind"]
            if ((active and row["live_request_key"] != row["request_key"])
                    or receipt_kind not in (("claim", "renew", "cancel") if active else ("release", "expire"))
                    or type(row["lease_fence"]) is not int or row["lease_fence"] <= 0
                    or any(not isinstance(row[name], str) or not row[name]
                           for name in ("request_digest", "owner_id", "receipt_id"))):
                raise ValueError("invalid coordination proof")
            return cls(
                row["execution_id"].decode("utf-8"), row["session_key"].decode("utf-8"),
                row["request_key"].decode("utf-8"), row["request_digest"], row["owner_id"],
                row["lease_fence"], row["cancel_requested"], row["receipt_id"], receipt_kind, active,
            )
        except (KeyError, TypeError, AttributeError, ValueError) as error:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error

    def validate_live_token(self, token: bytes) -> None:
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
        self._store_key = validate_store_key(store_key)
        self._scope_key = encode_key_components(scope.as_tuple(), TURN_SCOPE_MAX_BYTES)

    def _owns(self, table):
        from sqlalchemy import and_

        return and_(table.c.store_key == self._store_key, table.c.scope_key == self._scope_key)

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
        linked = and_(a.c.store_key == s.c.store_key, a.c.scope_key == s.c.scope_key,
                      a.c.session_key == s.c.session_key, a.c.execution_id == s.c.active_execution_id)
        if route == "session":
            joined = s.outerjoin(a, linked)
            anchor, route_key, limit = s, s.c.session_key, TURN_SESSION_MAX_BYTES
        else:
            joined = a.outerjoin(s, linked)
            anchor = a
            route_key, limit = {
                "execution": (a.c.execution_id, TURN_EXECUTION_MAX_BYTES),
                "request": (a.c.live_request_key, TURN_REQUEST_MAX_BYTES),
            }[route]
        return select(
            a, s.c.session_key.label("slot_session_key"), s.c.active_execution_id,
            s.c.lease_fence.label("slot_fence"), s.c.revision.label("slot_revision"),
            s.c.lease_expires_at, s.c.lease_token, clock.label("observed_at"),
        ).select_from(joined).where(self._owns(anchor), route_key == self._key(identity, limit))

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
        if (row["slot_session_key"] != row["session_key"] or row["active_execution_id"] != row["execution_id"]
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
        joined = seed.outerjoin(target, and_(self._owns(target), target.c.session_key ==
            self._key(session_key, TURN_SESSION_MAX_BYTES))).outerjoin(
            live_request, and_(self._owns(live_request), live_request.c.live_request_key ==
                self._key(request_key, TURN_REQUEST_MAX_BYTES)),
        ).outerjoin(incumbent, and_(self._owns(incumbent), incumbent.c.session_key == live_request.c.session_key)
        ).outerjoin(target_owner, and_(self._owns(target_owner), target_owner.c.execution_id == target.c.active_execution_id)
        ).outerjoin(proposed, and_(self._owns(proposed), proposed.c.execution_id == self._key(execution_id, TURN_EXECUTION_MAX_BYTES)))
        return select(clock.label("observed_at"), *(
            column.label(table.name + "__" + column.name)
            for table in (target, incumbent, live_request, target_owner, proposed)
            for column in table.c if column.name not in {"created_at", "updated_at", "store_key", "scope_key"}
        )).select_from(joined)

    def create_slot(self, session: str) -> "Insert":
        from sqlalchemy import insert

        return insert(self._slot).values(store_key=self._store_key, scope_key=self._scope_key,
                                         session_key=self._key(session, TURN_SESSION_MAX_BYTES))

    def guard_slot_revision(self, session: bytes, revision: int) -> "Update":
        from sqlalchemy import update

        slot = self._slot
        return update(slot).where(self._owns(slot), slot.c.session_key == session, slot.c.revision == revision
                                  ).values(revision=revision + 1)

    def bind_slot(self, session: bytes, *, execution: str | None, token: str | None,
                  fence: int, expiry: datetime | None) -> "Update":
        from sqlalchemy import update

        slot = self._slot
        return update(slot).where(self._owns(slot), slot.c.session_key == session).values(
            active_execution_id=None if execution is None else self._key(execution, TURN_EXECUTION_MAX_BYTES),
            lease_token=None if token is None else token.encode("utf-8"), lease_fence=fence, lease_expires_at=expiry,
        )

    def publish_claim(self, proof: SqlTurnProof, *, previous: SqlTurnProof | None) -> "Insert | Update":
        from sqlalchemy import insert, update

        a = self._admission
        request = self._key(proof.request_key, TURN_REQUEST_MAX_BYTES)
        execution = self._key(proof.execution_id, TURN_EXECUTION_MAX_BYTES)
        values = dict(session_key=self._key(proof.session_key, TURN_SESSION_MAX_BYTES),
                      request_key=request, live_request_key=request, request_digest=proof.request_digest,
                      owner_id=proof.owner_id, lease_fence=proof.fence, cancel_requested=False,
                      receipt_id=proof.receipt_id, receipt_kind="claim")
        if previous is None:
            return insert(a).values(store_key=self._store_key, scope_key=self._scope_key, execution_id=execution, **values)
        return update(a).where(
            self._owns(a), a.c.execution_id == execution, a.c.live_request_key.is_(None),
            a.c.session_key == self._key(previous.session_key, TURN_SESSION_MAX_BYTES),
            a.c.request_key == self._key(previous.request_key, TURN_REQUEST_MAX_BYTES),
            a.c.lease_fence == previous.fence, a.c.receipt_id == previous.receipt_id,
            a.c.receipt_kind == previous.receipt_kind,
        ).values(**values)

    def refresh_admission(self, *, execution_id: str, session_key: str, request_key: str,
                          fence: int, receipt: str, kind: str) -> "Update":
        from sqlalchemy import update

        a = self._admission
        values = dict(receipt_id=receipt, receipt_kind=kind)
        if kind == "cancel":
            values["cancel_requested"] = True
        return update(a).where(
            self._owns(a), a.c.execution_id == self._key(execution_id, TURN_EXECUTION_MAX_BYTES),
            a.c.session_key == self._key(session_key, TURN_SESSION_MAX_BYTES), a.c.lease_fence == fence,
            a.c.live_request_key == self._key(request_key, TURN_REQUEST_MAX_BYTES),
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
            self._owns(slot), slot.c.session_key == self._key(session_key, TURN_SESSION_MAX_BYTES), live,
        ).values(active_execution_id=None, lease_token=None, lease_expires_at=None)

    def retire_admission(self, *, execution_id: str, session_key: str, request_key: str,
                         fence: int, receipt: str, token: str, kind: str = "release") -> "Update":
        from sqlalchemy import and_, func, or_, update

        admission = self._admission
        request = self._key(request_key, TURN_REQUEST_MAX_BYTES)
        return update(admission).where(
            self._owns(admission), admission.c.execution_id == self._key(execution_id, TURN_EXECUTION_MAX_BYTES),
            admission.c.session_key == self._key(session_key, TURN_SESSION_MAX_BYTES),
            admission.c.lease_fence == fence,
            admission.c.request_key == request, admission.c.live_request_key == request,
            or_(and_(admission.c.receipt_kind == "claim", admission.c.receipt_id == token,
                    func.length(admission.c.receipt_id) == len(token)),
                admission.c.receipt_kind == "renew",
                and_(admission.c.receipt_kind == "cancel", admission.c.cancel_requested)),
        ).values(live_request_key=None, receipt_id=receipt, receipt_kind=kind)

    def serialize_slot(self, *, session_key: str, execution_id: str, token: str, fence: int) -> "Update":
        """Acquire real write serialization; expiry must still be tested after waiting."""
        from sqlalchemy import update

        slot = self._slot
        return update(slot).where(
            self._owns(slot), slot.c.session_key == self._key(session_key, TURN_SESSION_MAX_BYTES),
            slot.c.active_execution_id == self._key(execution_id, TURN_EXECUTION_MAX_BYTES),
            slot.c.lease_token == token.encode("utf-8"), slot.c.lease_fence == fence,
        ).values(revision=slot.c.revision + 1)


__all__ = ["SqlTurnProof", "SqlTurnRows", "SqlTurnObservation", "SqlTurnNotOwned"]
