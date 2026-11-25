#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SQL authority for sessions whose turns can be regenerated."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from ..storage import (
    sql_audit_columns,
    sql_audit_indexes,
    sql_binary_key,
    sql_epoch_seconds,
    sql_id_column,
    sql_table_options,
    sql_unique,
)
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


def build_session_turn_sql_metadata(metadata: MetaData) -> None:
    from sqlalchemy import JSON, BigInteger, Column, String, Table

    def key(name: str, size: int, description: str, *, nullable: bool = False) -> Column:
        return Column(name, sql_binary_key(size), nullable=nullable, comment=description)

    session = Table(
        SESSION_TABLE, metadata, sql_id_column(),
        key("scope_key", 32, "Partition of one tenant in the configured session store."),
        key("session_key", TURN_SESSION_MAX_BYTES, "Exact session identity."),
        key("owner_principal_id", 128, "Principal that owns the session."),
        key("agent_id", 128, "Agent named by the session."),
        Column("status", String(8), nullable=False, comment="Session lifecycle status."),
        Column("revision", BigInteger, nullable=False, comment="Compare-and-swap revision of the session."),
        key("active_turn_id", TURN_EXECUTION_MAX_BYTES, "Accepted turn occupying the session.", nullable=True),
        Column("last_turn_no", BigInteger, nullable=False, comment="Last finalized turn number."),
        Column("config_json", JSON(), nullable=False, comment="Session metadata, workspace path and instruction scopes."),
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


class SqlSessionTurns:
    """A tenant-scoped store. The session service owns authorization and request codecs."""

    def __init__(self, context: SqlStorageContext, metadata: MetaData, *, namespace: str, tenant_id: str) -> None:
        self._context = context
        self._scope = state_store_key(namespace, tenant_id, "regenerable-session")
        self._sessions = metadata.tables[SESSION_TABLE]
        self._turns = metadata.tables[SESSION_TURN_TABLE]
        self._now = sql_epoch_seconds()

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

    async def create(self, session_id: str, *, owner_principal_id: str, agent_id: str, config: JsonValue) -> None:
        async def write(connection: AsyncSession) -> None:
            await connection.execute(self._sessions.insert().values(
                **self._session_values(session_id), owner_principal_id=self._key(owner_principal_id, 128),
                agent_id=self._key(agent_id, 128), status="open", revision=0, last_turn_no=0, config_json=config,
            ))

        await self._context.run_mutation(write, domain="session.create")

    async def get_session(self, session_id: str) -> RowMapping | None:
        async with self._context.engine.connect() as connection:
            return (await connection.execute(
                self._sessions.select().filter_by(**self._session_values(session_id)),
            )).mappings().one_or_none()

    async def get_turn(self, key: SessionTurnKey) -> RowMapping | None:
        async with self._context.engine.connect() as connection:
            return (await connection.execute(self._turns.select().filter_by(**self._turn_values(key)))).mappings().one_or_none()

    async def find_request(self, session_id: str, request_key: str) -> RowMapping | None:
        async with self._context.engine.connect() as connection:
            return (await connection.execute(self._turns.select().filter_by(
                **self._session_values(session_id), request_key=self._key(request_key, TURN_REQUEST_MAX_BYTES),
            ))).mappings().one_or_none()

    async def history(
        self, session_id: str, *, through: int, limit: int, after: int = 0, only_success: bool = False,
    ) -> tuple[RowMapping, ...]:
        from sqlalchemy import select

        if not 1 <= limit or not 0 <= after <= through:
            raise ValueError("invalid session turn history range")
        columns = self._turns.c
        statement = select(self._turns).filter_by(**self._session_values(session_id)).where(
            columns.turn_no > after, columns.turn_no <= through,
            columns.state.in_(("succeeded",) if only_success else ("succeeded", "failed", "cancelled")),
        ).order_by(columns.turn_no).limit(limit)
        async with self._context.engine.connect() as connection:
            return tuple((await connection.execute(statement)).mappings())

    async def accept(
        self, key: SessionTurnKey, *, expected_revision: int, context_through: int,
        request_key: str, request: JsonValue, lease_seconds: int,
    ) -> bool:
        """Claim and record a turn atomically; a uniqueness failure rolls back the claim."""
        if lease_seconds < 1:
            raise ValueError("the turn lease must be positive")
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
                context_through=context_through, state="accepted", generation=1,
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
