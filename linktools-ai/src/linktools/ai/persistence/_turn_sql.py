#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SQL statements for exact source-owned Execution turn receipts."""

from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256
from typing import TYPE_CHECKING, Literal
from collections.abc import Callable, Mapping

from ..core import canonical_json_bytes
from ..errors import AIError, ErrorCode
from ..storage import Lease, TurnCoordination, TurnScopeIdentity
from .state import record_key
from ._turn_sql_schema import (
    TURN_EXECUTION_MAX_BYTES, TURN_REQUEST_MAX_BYTES,
    RUNTIME_TURN_SCOPE, TURN_SESSION_MAX_BYTES, unoccupied_turn_route,
)

if TYPE_CHECKING:
    from sqlalchemy import MetaData, Select, Update
    from sqlalchemy.sql.elements import ColumnElement
    from sqlalchemy.sql.selectable import FromClause


Route = Literal["session", "execution", "request"]


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

    def __init__(self, metadata: "MetaData", store_key: bytes, scope: TurnScopeIdentity, *, coowned: bool = False) -> None:
        self._scope = scope
        self._metadata = metadata
        self.coowned = coowned
        self._control = metadata.tables["ai_execution_records"]
        if type(store_key) is not bytes or len(store_key) != 32:
            raise ValueError("typed turn SQL requires a runtime partition")
        self._store_key = store_key
        if scope.scope != RUNTIME_TURN_SCOPE:
            raise ValueError("typed turn SQL accepts only the fixed runtime scope")

    def execution_key(self, execution_id: str) -> bytes:
        return record_key(self._scope.namespace, self._scope.tenant_id, "execution", "execution", (execution_id,))

    def _execution_address(self, control: "FromClause", execution_id: str) -> "ColumnElement[bool]":
        from sqlalchemy import and_

        key = self.execution_key(execution_id)
        return and_(
            control.c.store_key == self._store_key,
            control.c.key_locator == sha256(key).hexdigest(),
            control.c.key == key,
        )

    def _owns(self, table):
        return table.c.store_key == self._store_key

    @staticmethod
    def _key(value: str, limit: int) -> bytes:
        raw = value.encode("utf-8")
        if not 1 <= len(raw) <= limit:
            raise ValueError("turn identity exceeds its byte limit")
        return raw

    def select_standalone_execution_receipt(
        self, *, route: Literal["execution", "request"], identity: str, clock: "ColumnElement",
    ) -> "Select":
        """Select the standalone owner directly from its indexed Execution row."""
        from sqlalchemy import select

        if not self.coowned:
            raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)
        control = self._control
        if route == "execution":
            exact = self._execution_address(control, identity)
        elif route == "request":
            exact = control.c.turn_request_route == turn_hex(identity, TURN_REQUEST_MAX_BYTES)
        else:
            raise ValueError("invalid standalone receipt route")
        return select(
            control.c.key, control.c.key_locator, control.c.turn_slot_kind, control.c.turn_slot_key,
            control.c.turn_request_route, control.c.turn_receipt, control.c.turn_token,
            control.c.turn_fence, control.c.turn_expiry, control.c.turn_revision, control.c.kind,
            clock.label("observed_at"),
        ).where(self._owns(control), control.c.turn_slot_kind == STANDALONE, control.c.turn_receipt.is_not(None), exact)

    def parse_native_session_receipt(
        self, row: Mapping[str, object], *, decode_time: Callable[[object], datetime],
    ) -> tuple[SqlTurnProof, SqlTurnObservation | None]:
        from ._record_key import RECORD_KEY_MAX_BYTES, decode_key_components

        try:
            key = row["key"]
            *_, execution_id = decode_key_components(key, RECORD_KEY_MAX_BYTES)
            logical = row["turn_slot_key"].decode("utf-8")
            session_key = f"session:{logical}"
            turn_session_address(session_key)
            owner_key = record_key(self._scope.namespace, self._scope.tenant_id, "conversation", "session", (logical,))
            owner_id = row["turn_session_owner_id"]
            receipt = row["turn_receipt"]
            token = row["turn_token"]
            fence, revision = row["turn_fence"], row["turn_revision"]
            active = token is not None
            if (key != self.execution_key(execution_id) or row["key_locator"] != sha256(key).hexdigest()
                    or row["kind"] not in ("execution", "execution_shell")
                    or type(owner_id) is not int or owner_id < 1
                    or row["session_owner_key"] != owner_key
                    or row["session_owner_locator"] != sha256(owner_key).hexdigest()
                    or not isinstance(receipt, Mapping)
                    or receipt.keys() != {"request_key", "owner_id", "request_digest", "receipt_id", "receipt_kind", "cancel_requested"}
                    or type(fence) is not int or fence < 1 or type(revision) is not int or revision < 0
                    or type(receipt["cancel_requested"]) is not bool or row["turn_expiry"] is None
                    or receipt["receipt_kind"] not in (("claim", "renew", "cancel") if active else ("release", "expire"))):
                raise ValueError("invalid native session receipt")
            request = receipt["request_key"]
            route = turn_hex(request, TURN_REQUEST_MAX_BYTES)
            if row["turn_request_route"] != (route if active else unoccupied_turn_route(row["key_locator"])):
                raise ValueError("invalid native session request route")
            for field, width in (("owner_id", 32), ("request_digest", 64), ("receipt_id", 32)):
                turn_hex(receipt[field], width)
            proof = SqlTurnProof(execution_id, session_key, request, receipt["request_digest"], receipt["owner_id"],
                                 fence, receipt["cancel_requested"], receipt["receipt_id"], receipt["receipt_kind"], active)
            if not active:
                return proof, None
            if (type(token) is not bytes or row["session_owner_fence"] != fence
                    or row["session_owner_execution_id"] != self._key(execution_id, TURN_EXECUTION_MAX_BYTES)):
                raise ValueError("invalid native session lease owner")
            proof.validate_live_token(token)
            return proof, SqlTurnObservation(
                proof, revision, decode_time(row["turn_expiry"]) <= decode_time(row["observed_at"]),
            )
        except (KeyError, AttributeError, TypeError, ValueError) as error:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error

    def parse_standalone_control_receipt(
        self, row: Mapping[str, object], *, decode_time: Callable[[object], datetime],
    ) -> tuple[SqlTurnProof, SqlTurnObservation | None]:
        """A parked/archived Execution proves a receipt, never live ownership."""
        from ._record_key import RECORD_KEY_MAX_BYTES, decode_key_components

        try:
            key = row["key"]
            *_, execution_id = decode_key_components(key, RECORD_KEY_MAX_BYTES)
            request = row["turn_slot_key"]
            route = row["turn_request_route"]
            receipt = row["turn_receipt"]
            token = row["turn_token"]
            fence, revision = row["turn_fence"], row["turn_revision"]
            active = token is not None
            if (key != self.execution_key(execution_id) or row["key_locator"] != sha256(key).hexdigest()
                    or row["kind"] not in ("execution", "execution_shell")
                    or route not in (unoccupied_turn_route(row["key_locator"]), request) or active and route != request
                    or not isinstance(receipt, Mapping)
                    or receipt.keys() != {"owner_id", "request_digest", "receipt_id", "receipt_kind", "cancel_requested"}
                    or type(fence) is not int or fence < 1 or type(revision) is not int or revision < 0
                    or type(receipt["cancel_requested"]) is not bool or row["turn_expiry"] is None
                    or receipt["receipt_kind"] not in (("claim", "renew", "cancel") if active else ("release", "expire"))):
                raise ValueError("invalid standalone control receipt")
            request_key = request.decode("ascii")
            turn_hex(request_key, TURN_REQUEST_MAX_BYTES)
            for field, width in (("owner_id", 32), ("request_digest", 64), ("receipt_id", 32)):
                turn_hex(receipt[field], width)
            proof = SqlTurnProof(
                execution_id, f"request:{request_key}", request_key, receipt["request_digest"], receipt["owner_id"],
                fence, receipt["cancel_requested"], receipt["receipt_id"], receipt["receipt_kind"], active,
            )
            if not active:
                return proof, None
            if type(token) is not bytes:
                raise ValueError("invalid standalone control token")
            proof.validate_live_token(token)
            return proof, SqlTurnObservation(
                proof, revision, decode_time(row["turn_expiry"]) <= decode_time(row["observed_at"]),
            )
        except (KeyError, AttributeError, TypeError, ValueError) as error:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error

    def standalone_control_claim_values(self, proof: SqlTurnProof, *, expiry: datetime) -> dict[str, object]:
        """Supply the single authority/receipt to an Execution or hidden-shell INSERT."""
        request = turn_hex(proof.request_key, TURN_REQUEST_MAX_BYTES)
        if (not self.coowned or turn_session_address(proof.session_key) != (STANDALONE, request)
                or not proof.active or proof.receipt_kind != "claim" or proof.cancel_requested
                or type(proof.fence) is not int or proof.fence < 1 or expiry.tzinfo is None):
            raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)
        token = turn_hex(proof.receipt_id, 32)
        turn_hex(proof.owner_id, 32)
        turn_hex(proof.request_digest, 64)
        return {
            "turn_slot_kind": STANDALONE, "turn_slot_key": request, "turn_request_route": request,
            "turn_token": token, "turn_fence": proof.fence, "turn_expiry": expiry,
            "turn_receipt": {
                "owner_id": proof.owner_id, "request_digest": proof.request_digest,
                "receipt_id": proof.receipt_id, "receipt_kind": "claim", "cancel_requested": False,
            },
        }

    def claim_standalone_control(
        self, proof: SqlTurnProof, *, expiry: datetime, previous_revision: int,
    ) -> "Update":
        """Claim a new generation on its exact idle Execution without another receipt row."""
        from sqlalchemy import and_, or_, update

        control = self._control
        values = self.standalone_control_claim_values(proof, expiry=expiry)
        request = values["turn_request_route"]
        return update(control).where(
            self._execution_address(control, proof.execution_id), control.c.turn_revision == previous_revision,
            control.c.turn_token.is_(None),
            or_(control.c.turn_slot_kind.is_(None), and_(
                control.c.turn_slot_kind == STANDALONE, control.c.turn_slot_key == request,
                control.c.turn_fence < proof.fence,
                control.c.turn_request_route.in_((
                    unoccupied_turn_route(sha256(self.execution_key(proof.execution_id)).hexdigest()), request,
                )),
                control.c.turn_receipt["request_digest"].as_string() == proof.request_digest,
            )),
        ).values(**values, turn_revision=control.c.turn_revision + 1)

    def park_standalone_control(
        self, proof: SqlTurnProof, *, token: str, revision: int, receipt: str,
        kind: "Literal['release', 'expire']", clock: "ColumnElement", sqlite: bool,
    ) -> "Update":
        """Close the exact token after same-transaction authority serialization.

        Pass the post-guard revision and a fresh statement-clock expression;
        a clock from before waiting on the exact owner cannot decide expiry.
        """
        from sqlalchemy import update

        request = turn_hex(proof.request_key, TURN_REQUEST_MAX_BYTES)
        if (not self.coowned or not proof.active or kind not in ("release", "expire")
                or turn_session_address(proof.session_key) != (STANDALONE, request)):
            raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)
        token_key = turn_hex(token, 32)
        proof.validate_live_token(token_key)
        turn_hex(receipt, 32)
        control = self._control
        live = self._deadline_is_live(clock, sqlite=sqlite)
        return update(control).where(
            self._execution_address(control, proof.execution_id), control.c.turn_slot_kind == STANDALONE,
            control.c.turn_slot_key == request, control.c.turn_request_route == request,
            control.c.turn_token == token_key, control.c.turn_fence == proof.fence,
            control.c.turn_revision == revision, live if kind == "release" else ~live,
        ).values(
            turn_token=None, turn_revision=control.c.turn_revision + 1,
            turn_receipt={
                "owner_id": proof.owner_id, "request_digest": proof.request_digest, "receipt_id": receipt,
                "receipt_kind": kind, "cancel_requested": proof.cancel_requested,
            },
        )

    def archive_parked_standalone_control(self, proof: SqlTurnProof, *, revision: int) -> "Update":
        """Free the request route for its successor, leaving the exact prior receipt addressable."""
        from sqlalchemy import update

        request = turn_hex(proof.request_key, TURN_REQUEST_MAX_BYTES)
        if (not self.coowned or proof.active or proof.receipt_kind not in ("release", "expire")
                or turn_session_address(proof.session_key) != (STANDALONE, request)):
            raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)
        control = self._control
        return update(control).where(
            self._execution_address(control, proof.execution_id), control.c.turn_slot_kind == STANDALONE,
            control.c.turn_slot_key == request, control.c.turn_request_route == request,
            control.c.turn_fence == proof.fence, control.c.turn_revision == revision,
            control.c.turn_token.is_(None), control.c.turn_receipt["receipt_id"].as_string() == proof.receipt_id,
            control.c.turn_receipt["receipt_kind"].as_string() == proof.receipt_kind,
        ).values(
            turn_request_route=unoccupied_turn_route(sha256(self.execution_key(proof.execution_id)).hexdigest()),
            turn_revision=control.c.turn_revision + 1,
        )

    def control_guard(
        self, *, session_key: str, execution_id: str, token: str, fence: int,
        revision: int | None = None, published_only: bool = False, revoke: bool = False,
    ) -> "Update":
        from sqlalchemy import update
        control = self._control
        slot_kind, slot = turn_session_address(session_key)
        exact = [self._execution_address(control, execution_id),
                 control.c.turn_slot_kind == slot_kind, control.c.turn_slot_key == slot,
                 control.c.turn_token == turn_hex(token, 32), control.c.turn_fence == fence]
        if published_only:
            exact.append(control.c.kind == "execution")
        if revision is not None:
            exact.append(control.c.turn_revision == revision)
        return update(control).where(*exact).values(
            turn_revision=control.c.turn_revision + 1, **({"turn_token": None} if revoke else {}),
        )

    def control_set_expiry(
        self, *, execution_id: str, token: str, fence: int, expiry: datetime | None,
        clock: "ColumnElement | None" = None, sqlite: bool = False,
    ) -> "Update":
        from sqlalchemy import func, update
        control = self._control
        values = {"turn_expiry": expiry}
        if expiry is None:
            values.update(turn_token=None)
        expires = control.c.turn_expiry
        live = None if clock is None else (func.julianday(expires) > func.julianday(clock) if sqlite else expires > clock)
        exact = [] if live is None else [live]
        return update(control).where(
            self._execution_address(control, execution_id),
            control.c.turn_token == turn_hex(token, 32), control.c.turn_fence == fence, *exact,
        ).values(**values)

    def _deadline_is_live(self, clock: "ColumnElement", *, sqlite: bool) -> "ColumnElement[bool]":
        from sqlalchemy import func

        expiry = self._control.c.turn_expiry
        return func.julianday(expiry) > func.julianday(clock) if sqlite else expiry > clock


__all__ = ["SqlTurnProof", "SqlTurnRows", "SqlTurnObservation", "SqlTurnNotOwned"]
