#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SQL authority for sessions whose turns can be regenerated."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from ..core import canonical_json_bytes, canonical_sha256, validate_idempotency_key
from ..errors import AIError, ErrorCode
from ..storage import (
    sql_audit_columns,
    sql_audit_indexes,
    sql_binary_key,
    sql_epoch_seconds,
    sql_id_column,
    sql_table_options,
    sql_unique,
)
from ._ordinary_lineage import SessionTurnSegment, decode_session_lineage, fork_segments, lineage_predicate
from ._turn_sql_schema import (
    TURN_EXECUTION_MAX_BYTES,
    TURN_REQUEST_MAX_BYTES,
    TURN_SESSION_MAX_BYTES,
)
from .state import state_store_key

if TYPE_CHECKING:
    from sqlalchemy import MetaData, RowMapping
    from sqlalchemy.ext.asyncio import AsyncSession

    from ..core import JsonValue
    from ..storage import SqlStorageContext


SESSION_TABLE = "ai_sessions"
SESSION_TURN_TABLE = "ai_session_turns"
TerminalState = Literal["succeeded", "failed", "cancelled"]
_CONTEXT_POSITIONS = 128
_HISTORY_CACHE_BYTES = 16 * 1024 * 1024
_HISTORY_CACHE_SESSIONS = 32


@dataclass(frozen=True, slots=True)
class SessionContextTurn:
    turn_no: int
    request: JsonValue
    result: JsonValue
    size: int


@dataclass(frozen=True, slots=True)
class _ContextWindow:
    after: int
    through: int
    turns: tuple[SessionContextTurn, ...]
    size: int


def build_session_turn_sql_metadata(metadata: MetaData) -> None:
    from sqlalchemy import JSON, BigInteger, Column, String, Table

    def key(name: str, size: int, description: str, *, nullable: bool = False) -> Column:
        return Column(name, sql_binary_key(size), nullable=nullable, comment=description)

    session = Table(
        SESSION_TABLE, metadata, sql_id_column(),
        key("scope_key", 32, "Partition of one tenant in the configured session store."),
        key("session_key", TURN_SESSION_MAX_BYTES, "Exact session identity."),
        Column("status", String(8), nullable=False, comment="Session lifecycle status."),
        Column("revision", BigInteger, nullable=False, comment="Compare-and-swap revision of the session."),
        key("active_turn_id", TURN_EXECUTION_MAX_BYTES, "Accepted turn occupying the session.", nullable=True),
        Column("last_turn_no", BigInteger, nullable=False, comment="Last finalized turn number."),
        Column("config_json", JSON(), nullable=False, comment="Session metadata, workspace path and instruction scopes."),
        Column("lineage_json", JSON(), nullable=True, comment="Immutable referenced turn ranges for a fork."),
        *sql_audit_columns(),
        comment="Identity, configuration and publication head of a regenerable conversation.",
        **sql_table_options(),
    )
    sql_unique(session, "scope_key", "session_key")
    sql_audit_indexes(session)

    turn = Table(
        SESSION_TURN_TABLE, metadata, sql_id_column(),
        key("scope_key", 32, "Partition of one tenant in the configured session store."),
        key("session_key", TURN_SESSION_MAX_BYTES, "Session owning the turn."),
        key("turn_id", TURN_EXECUTION_MAX_BYTES, "Exact public turn identity."),
        key("request_key", TURN_REQUEST_MAX_BYTES, "Idempotency key scoped to the session."),
        Column("turn_no", BigInteger, nullable=False, comment="Sequential turn position within the session."),
        Column("context_from", BigInteger, nullable=False, comment="First selected context position, inclusive."),
        Column("context_through", BigInteger, nullable=False, comment="Finalized context boundary at admission."),
        Column("state", String(10), nullable=False, comment="Admission, cancellation or terminal state."),
        Column("generation", BigInteger, nullable=False, comment="Fencing generation of the accepted producer."),
        Column("lease_until", BigInteger, nullable=True, comment="Expiry in database-clock seconds since epoch."),
        Column("request_json", JSON(), nullable=False, comment="Accepted principal, canonical request and frozen binding."),
        Column("result_json", JSON(none_as_null=True), nullable=True,
               comment="Canonical terminal output and continuation material."),
        *sql_audit_columns(),
        comment="Authoritative admission and publication of one regenerable conversation turn.",
        **sql_table_options(),
    )
    sql_unique(turn, "scope_key", "turn_id")
    sql_unique(turn, "scope_key", "session_key", "request_key")
    sql_unique(turn, "scope_key", "session_key", "turn_no")
    sql_audit_indexes(turn)


@dataclass(frozen=True, slots=True)
class SessionTurnKey:
    session_id: str
    turn_id: str


class SqlSessionTurnReader:
    """Read one tenant’s committed session and turn history."""

    def __init__(self, context: SqlStorageContext, metadata: MetaData, *, namespace: str, tenant_id: str) -> None:
        self._context = context
        self._scope = state_store_key(namespace, tenant_id, "regenerable-session")
        self._sessions = metadata.tables[SESSION_TABLE]
        self._turns = metadata.tables[SESSION_TURN_TABLE]
        self._recent: OrderedDict[str, _ContextWindow] = OrderedDict()
        self._recent_bytes = 0

    @staticmethod
    def _key(value: str, size: int) -> bytes:
        encoded = value.encode("utf-8")
        if not 1 <= len(encoded) <= size:
            raise ValueError(f"session turn key must contain 1 to {size} UTF-8 bytes")
        return encoded

    def _session_values(self, session_id: str) -> dict[str, bytes]:
        return {"scope_key": self._scope, "session_key": self._key(session_id, TURN_SESSION_MAX_BYTES)}

    def _turn_values(self, key: SessionTurnKey) -> dict[str, bytes]:
        return {**self._session_values(key.session_id), "turn_id": self._key(key.turn_id, TURN_EXECUTION_MAX_BYTES)}

    async def get_session(self, session_id: str) -> RowMapping | None:
        async with self._context.engine.connect() as connection:
            return (await connection.execute(
                self._sessions.select().filter_by(**self._session_values(session_id)),
            )).mappings().one_or_none()

    async def get_turn(self, key: SessionTurnKey) -> RowMapping | None:
        async with self._context.engine.connect() as connection:
            return (await connection.execute(self._turns.select().filter_by(**self._turn_values(key)))).mappings().one_or_none()

    async def get_history_turn(
        self, key: SessionTurnKey, *, lineage: tuple[SessionTurnSegment, ...],
    ) -> RowMapping | None:
        if not lineage:
            return await self.get_turn(key)
        from sqlalchemy import and_, or_

        columns = self._turns.c
        async with self._context.engine.connect() as connection:
            return (await connection.execute(self._turns.select().where(
                columns.scope_key == self._scope,
                columns.turn_id == self._key(key.turn_id, TURN_EXECUTION_MAX_BYTES),
                or_(
                    columns.session_key == self._key(key.session_id, TURN_SESSION_MAX_BYTES),
                    and_(
                        columns.state.in_(("succeeded", "failed", "cancelled")),
                        lineage_predicate(columns, b"", lineage, after=0, through=lineage[-1].through),
                    ),
                ),
            ))).mappings().one_or_none()

    async def find_request(self, session_id: str, request_key: str) -> RowMapping | None:
        async with self._context.engine.connect() as connection:
            return (await connection.execute(self._turns.select().filter_by(
                **self._session_values(session_id), request_key=self._key(request_key, TURN_REQUEST_MAX_BYTES),
            ))).mappings().one_or_none()

    async def history(
        self, session_id: str, *, through: int, limit: int, after: int = 0, only_success: bool = False,
        lineage: tuple[SessionTurnSegment, ...] = (),
    ) -> tuple[RowMapping, ...]:
        from sqlalchemy import select

        if not 1 <= limit or not 0 <= after <= through:
            raise ValueError("invalid session turn history range")
        columns = self._turns.c
        statement = select(self._turns).where(
            columns.scope_key == self._scope,
            lineage_predicate(columns, self._key(session_id, TURN_SESSION_MAX_BYTES), lineage,
                              after=after, through=through),
            columns.state.in_(("succeeded",) if only_success else ("succeeded", "failed", "cancelled")),
        ).order_by(columns.turn_no).limit(limit)
        async with self._context.engine.connect() as connection:
            return tuple((await connection.execute(statement)).mappings())

    async def session_page(
        self, *, session_ids: Sequence[str] | None, before: int | None, limit: int,
    ) -> tuple[RowMapping, ...]:
        columns = self._sessions.c
        statement = self._sessions.select().where(columns.scope_key == self._scope)
        if session_ids is not None:
            if not session_ids:
                return ()
            keys = tuple(self._key(identity, TURN_SESSION_MAX_BYTES) for identity in session_ids)
            statement = statement.where(columns.session_key.in_(keys))
        if before is not None:
            statement = statement.where(columns.id < before)
        async with self._context.engine.connect() as connection:
            return tuple((await connection.execute(statement.order_by(columns.id.desc()).limit(limit))).mappings())

    async def turn_page(
        self, session_id: str, *, through: int, limit: int,
        lineage: tuple[SessionTurnSegment, ...] = (),
    ) -> tuple[RowMapping, ...]:
        columns = self._turns.c
        statement = self._turns.select().where(
            columns.scope_key == self._scope,
            lineage_predicate(columns, self._key(session_id, TURN_SESSION_MAX_BYTES), lineage,
                              after=0, through=through),
            columns.state.in_(("succeeded", "failed", "cancelled")),
        ).order_by(columns.turn_no.desc()).limit(limit)
        async with self._context.engine.connect() as connection:
            return tuple((await connection.execute(statement)).mappings())

    async def recent_context(
        self, session_id: str, *, through: int, lineage: tuple[SessionTurnSegment, ...] = (),
    ) -> tuple[SessionContextTurn, ...]:
        """Read bounded candidates after the caller authorizes the current session head."""
        from sqlalchemy import case, select, type_coerce

        start = max(0, through - _CONTEXT_POSITIONS)
        cached = self._recent.get(session_id)
        if cached is not None and cached.after <= start <= cached.through <= through:
            if cached.through == through:
                self._recent.move_to_end(session_id)
                return cached.turns
            after = cached.through
            turns = tuple(turn for turn in cached.turns if turn.turn_no > start)
        else:
            after = start
            turns = ()
        if after == through:
            return turns

        columns = self._turns.c
        succeeded = columns.state == "succeeded"
        prompt = case((succeeded, columns.request_json["prompt"].as_string())).label("prompt")
        result = type_coerce(case((succeeded, columns.result_json)), columns.result_json.type).label("result_json")
        stored_input = type_coerce(
            case((succeeded, columns.request_json["input"])), columns.request_json.type,
        ).label("stored_input")
        statement = select(columns.turn_no, columns.state, prompt, stored_input, result).where(
            columns.scope_key == self._scope,
            lineage_predicate(columns, self._key(session_id, TURN_SESSION_MAX_BYTES), lineage,
                              after=after, through=through),
            columns.state.in_(("succeeded", "failed", "cancelled")),
        ).order_by(columns.turn_no).limit(_CONTEXT_POSITIONS)
        async with self._context.engine.connect() as connection:
            rows = (await connection.execute(statement)).mappings().all()
        if len(rows) != through - after:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR, "The finalized session context range is incomplete")
        turns += tuple(
            self._context_turn(row["turn_no"], row["prompt"], row["stored_input"], row["result_json"])
            for row in rows if row["state"] == "succeeded"
        )
        self._remember_context(session_id, after=start, through=through, turns=turns)
        return turns

    @staticmethod
    def _context_turn(turn_no: int, prompt: str, stored_input: JsonValue, result: JsonValue) -> SessionContextTurn:
        request = {"prompt": prompt, **({"input": stored_input} if stored_input is not None else {})}
        size = max(1024, len(canonical_json_bytes(request)) + len(canonical_json_bytes(result)))
        return SessionContextTurn(turn_no, request, result, size)

    def _remember_context(
        self, session_id: str, *, after: int, through: int, turns: tuple[SessionContextTurn, ...],
    ) -> None:
        previous = self._recent.get(session_id)
        if previous is not None and previous.through > through:
            return
        if previous is not None:
            self._recent_bytes -= self._recent.pop(session_id).size
        size = sum(turn.size for turn in turns)
        if size > _HISTORY_CACHE_BYTES:
            return
        while self._recent and (
            self._recent_bytes + size > _HISTORY_CACHE_BYTES or len(self._recent) >= _HISTORY_CACHE_SESSIONS
        ):
            self._recent_bytes -= self._recent.popitem(last=False)[1].size
        self._recent[session_id] = _ContextWindow(after, through, turns, size)
        self._recent_bytes += size


class SqlSessionTurns(SqlSessionTurnReader):
    """Write authority for one tenant’s ordinary SQL sessions."""

    def __init__(self, context: SqlStorageContext, metadata: MetaData, *, namespace: str, tenant_id: str) -> None:
        super().__init__(context, metadata, namespace=namespace, tenant_id=tenant_id)
        self._now = sql_epoch_seconds()

    def note_terminal_context(
        self, key: SessionTurnKey, *, turn_no: int, prompt: str | None, stored_input: JsonValue,
        result: JsonValue, succeeded: bool,
    ) -> None:
        """Advance coverage only over a known committed adjacent position."""
        cached = self._recent.get(key.session_id)
        after = max(0, turn_no - _CONTEXT_POSITIONS)
        if cached is None:
            if turn_no != 1:
                return
            turns = ()
        elif cached.through != turn_no - 1 or cached.after > after:
            return
        else:
            turns = tuple(turn for turn in cached.turns if turn.turn_no > after)
        if succeeded:
            if prompt is None:
                return
            turns += (self._context_turn(turn_no, prompt, stored_input, result),)
        self._remember_context(key.session_id, after=after, through=turn_no, turns=turns)

    async def create(self, session_id: str, *, config: JsonValue) -> None:
        async def write(connection: AsyncSession) -> None:
            await connection.execute(self._sessions.insert().values(
                **self._session_values(session_id), status="open", revision=0, last_turn_no=0, config_json=config,
            ))

        await self._context.run_mutation(write, domain="session.create")

    async def fork_session(
        self, source_id: str, target_id: str, *, expected_revision: int, config: JsonValue, request_id: str,
    ) -> bool:
        """Freeze the source head with a CAS and insert one child session; never copy turns."""
        validate_idempotency_key(request_id)
        if source_id == target_id:
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        source = self._session_values(source_id)
        target = self._session_values(target_id)

        async def write(connection: AsyncSession) -> bool:
            existing = (await connection.execute(self._sessions.select().filter_by(**target))).mappings().one_or_none()
            if existing is not None:
                lineage = existing["lineage_json"]
                if (lineage is not None and lineage["source"] == source_id and lineage["request_id"] == request_id
                        and lineage["config_fingerprint"] == canonical_sha256(config)):
                    return True
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            row = (await connection.execute(self._sessions.select().filter_by(
                **source, revision=expected_revision,
            ))).mappings().one_or_none()
            if row is None:
                return False
            through = row["last_turn_no"]
            segments = fork_segments(source_id, decode_session_lineage(row["lineage_json"]), through)
            result = await connection.execute(self._sessions.update().filter_by(
                **source, revision=expected_revision, last_turn_no=through,
            ).values(revision=expected_revision + 1))
            if result.rowcount != 1:
                return False
            await connection.execute(self._sessions.insert().values(
                **target, status="open", revision=0, last_turn_no=through, config_json=config,
                lineage_json={"source": source_id, "request_id": request_id, "config_fingerprint": canonical_sha256(config),
                              "segments": [[part.session_id, part.after, part.through] for part in segments]},
            ))
            return True

        return await self._context.run_mutation(write, domain="session.fork")

    async def update_session(
        self, session_id: str, *, expected_revision: int, config: JsonValue,
    ) -> bool:
        """Replace inactive session configuration at the caller’s authorized revision."""
        async def write(connection: AsyncSession) -> bool:
            result = await connection.execute(self._sessions.update().filter_by(
                **self._session_values(session_id),
                status="open", revision=expected_revision, active_turn_id=None,
            ).values(config_json=config, revision=expected_revision + 1))
            return result.rowcount == 1

        return await self._context.run_mutation(write, domain="session.update")

    async def close_session(
        self, session_id: str, *, expected_revision: int,
    ) -> bool:
        """Close an unoccupied session atomically so a concurrent admission cannot win as well."""
        async def write(connection: AsyncSession) -> bool:
            result = await connection.execute(self._sessions.update().filter_by(
                **self._session_values(session_id),
                status="open", revision=expected_revision, active_turn_id=None,
            ).values(status="closed", revision=expected_revision + 1))
            return result.rowcount == 1

        return await self._context.run_mutation(write, domain="session.close")

    async def accept(
        self, key: SessionTurnKey, *, expected_revision: int, context_from: int, context_through: int,
        request_key: str, request: JsonValue, lease_seconds: int,
    ) -> bool:
        """Claim and record a turn atomically; a uniqueness failure rolls back the claim."""
        if lease_seconds < 1:
            raise ValueError("the turn lease must be positive")
        if not 1 <= context_from <= context_through + 1:
            raise ValueError("invalid admitted session context range")
        session_key = self._session_values(key.session_id)
        turn_key = self._turn_values(key)
        encoded_request_key = self._key(request_key, TURN_REQUEST_MAX_BYTES)

        async def write(connection: AsyncSession) -> bool:
            claim = await connection.execute(self._sessions.update().filter_by(
                **session_key, status="open", revision=expected_revision,
                last_turn_no=context_through, active_turn_id=None,
            ).values(active_turn_id=turn_key["turn_id"], revision=expected_revision + 1))
            if claim.rowcount != 1:
                return False
            await connection.execute(self._turns.insert().values(
                **turn_key, request_key=encoded_request_key, turn_no=context_through + 1,
                context_from=context_from, context_through=context_through, state="accepted", generation=1,
                lease_until=self._now + lease_seconds, request_json=request,
            ))
            return True

        return await self._context.run_mutation(write, domain="session.accept")

    async def finish(
        self, key: SessionTurnKey, *, generation: int, turn_no: int, state: TerminalState, result: JsonValue,
    ) -> bool:
        """Finalize and release using the same lock order as acceptance."""
        class FenceLost(Exception):
            pass

        if state not in ("succeeded", "failed", "cancelled"):
            raise ValueError("invalid terminal state")
        turn_key = self._turn_values(key)

        async def write(connection: AsyncSession) -> bool:
            released = await connection.execute(self._sessions.update().filter_by(
                **self._session_values(key.session_id), active_turn_id=turn_key["turn_id"], last_turn_no=turn_no - 1,
            ).values(active_turn_id=None, last_turn_no=turn_no, revision=self._sessions.c.revision + 1))
            if released.rowcount != 1:
                return False
            finished = await connection.execute(self._turns.update().filter_by(
                **turn_key, generation=generation, turn_no=turn_no,
                state="cancelling" if state == "cancelled" else "accepted",
            ).where(self._turns.c.lease_until > self._now).values(
                state=state, lease_until=None, result_json=result,
            ))
            if finished.rowcount != 1:
                raise FenceLost
            return True

        try:
            return await self._context.run_mutation(write, domain="session.finish")
        except FenceLost:
            return False

    async def request_cancel(self, key: SessionTurnKey) -> bool:
        async def write(connection: AsyncSession) -> bool:
            result = await connection.execute(self._turns.update().filter_by(
                **self._turn_values(key), state="accepted",
            ).values(state="cancelling"))
            return result.rowcount == 1

        return await self._context.run_mutation(write, domain="session.cancel")

    async def renew(self, key: SessionTurnKey, *, generation: int, lease_seconds: int) -> bool:
        from sqlalchemy import case

        if lease_seconds < 1:
            raise ValueError("the turn lease must be positive")

        async def write(connection: AsyncSession) -> bool:
            columns = self._turns.c
            result = await connection.execute(self._turns.update().filter_by(
                **self._turn_values(key), generation=generation, state="accepted",
            ).where(columns.lease_until > self._now).values(
                lease_until=case((columns.lease_until < self._now + lease_seconds, self._now + lease_seconds),
                                 else_=columns.lease_until),
            ))
            return result.rowcount == 1

        return await self._context.run_mutation(write, domain="session.renew")

    async def take_over(
        self, key: SessionTurnKey, *, generation: int, lease_seconds: int, cancelling: bool = False,
    ) -> bool:
        if lease_seconds < 1:
            raise ValueError("the turn lease must be positive")

        async def write(connection: AsyncSession) -> bool:
            result = await connection.execute(self._turns.update().filter_by(
                **self._turn_values(key), generation=generation, state="cancelling" if cancelling else "accepted",
            ).where(self._turns.c.lease_until <= self._now).values(
                generation=generation + 1, lease_until=self._now + lease_seconds,
            ))
            return result.rowcount == 1

        return await self._context.run_mutation(write, domain="session.recover")
