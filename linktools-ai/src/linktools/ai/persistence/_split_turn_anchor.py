#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Execution-local SESSION slot routing for split Runtime storage."""

from ._turn_sql import SESSION, STANDALONE, turn_session_address
from ._turn_sql_schema import unoccupied_turn_route


class SplitSessionTurnTransactions:
    """Execution-local split SESSION authority (never checks a remote row)."""

    def __init__(self, tx, scope) -> None:
        from .state import record_key

        self._tx = tx
        self._scope = scope
        self._table = tx._table("ai_execution_records")
        self._store = tx._store_key
        self._dialect = tx._context.dialect
        self._record_key = record_key

    def _execution_key(self, execution_id):
        return self._record_key(
            self._scope.namespace, self._scope.tenant_id, "execution", "execution", (execution_id,),
        )

    def _match_execution(self, table, execution_id):
        from hashlib import sha256
        from sqlalchemy import and_

        key = self._execution_key(execution_id)
        return and_(table.c.store_key == self._store, table.c.key_locator == sha256(key).hexdigest(),
                    table.c.key == key)

    async def _source_clock(self):
        from sqlalchemy import literal
        from ..storage import MySQLDialect, PostgreSQLDialect, SQLiteDialect

        if type(self._dialect) in (SQLiteDialect, MySQLDialect, PostgreSQLDialect):
            return self._dialect.read_snapshot_time_expression()
        return literal(await self._tx._context.run_statement(self._dialect.database_now(self._tx._session)))

    def _route(self, route, identity, *, clock):
        from sqlalchemy import select
        from ._turn_sql import turn_hex
        from ._turn_sql_schema import TURN_REQUEST_MAX_BYTES

        table = self._table
        family = table.c.turn_slot_kind.in_((SESSION, STANDALONE))
        if route == "execution":
            exact = self._match_execution(table, identity)
        elif route == "request":
            exact = table.c.turn_request_route == turn_hex(identity, TURN_REQUEST_MAX_BYTES)
        elif route == "session":
            kind, slot = turn_session_address(identity)
            family = table.c.turn_slot_kind == kind
            exact = (table.c.turn_occupied_route == slot if kind == SESSION
                     else table.c.turn_request_route == slot)
        else:
            raise ValueError("invalid split session route")
        return select(
            table.c.id, table.c.key, table.c.key_locator, table.c.kind, table.c.scan_kind,
            table.c.turn_slot_kind, table.c.turn_slot_key, table.c.turn_occupied_route, table.c.turn_request_route,
            table.c.turn_receipt, table.c.turn_token, table.c.turn_fence, table.c.turn_expiry,
            table.c.turn_revision, clock.label("observed_at"),
        ).where(
            table.c.store_key == self._store, family,
            table.c.turn_receipt.is_not(None), exact,
        )

    @staticmethod
    def parse_receipt(row, *, scope, decode_time):
        from hashlib import sha256
        from collections.abc import Mapping
        from ..errors import AIError, ErrorCode
        from ._record_key import RECORD_KEY_MAX_BYTES, decode_key_components
        from ._record_owner import EXECUTION_SHELL_KIND
        from ._turn_sql import SqlTurnObservation, SqlTurnProof, turn_hex, turn_session_identity
        from ._turn_sql_schema import TURN_REQUEST_MAX_BYTES
        from .state import record_key

        try:
            key = row["key"]
            *_, execution_id = decode_key_components(key, RECORD_KEY_MAX_BYTES)
            slot = row["turn_slot_key"]
            session_key = turn_session_identity(SESSION, slot)
            receipt, token = row["turn_receipt"], row["turn_token"]
            fence, revision = row["turn_fence"], row["turn_revision"]
            active = token is not None
            expected_key = record_key(scope.namespace, scope.tenant_id, "execution", "execution", (execution_id,))
            if (key != expected_key or row["key_locator"] != sha256(key).hexdigest()
                    or row["kind"] not in ("execution", EXECUTION_SHELL_KIND)
                    or row["turn_occupied_route"] not in (unoccupied_turn_route(row["key_locator"]), slot)
                    or active and (type(token) is not bytes or row["turn_occupied_route"] != slot)
                    or not isinstance(receipt, Mapping)
                    or receipt.keys() != {"request_key", "owner_id", "request_digest", "receipt_id", "receipt_kind", "cancel_requested"}
                    or type(fence) is not int or fence < 1 or type(revision) is not int or revision < 0
                    or type(receipt["cancel_requested"]) is not bool or row["turn_expiry"] is None
                    or receipt["receipt_kind"] not in (("claim", "renew", "cancel") if active else ("release", "expire"))):
                raise ValueError("invalid split session receipt")
            request = receipt["request_key"]
            inactive = unoccupied_turn_route(row["key_locator"])
            if row["turn_request_route"] != (turn_hex(request, TURN_REQUEST_MAX_BYTES) if active else inactive):
                raise ValueError("invalid split request route")
            for name, width in (("owner_id", 32), ("request_digest", 64), ("receipt_id", 32)):
                turn_hex(receipt[name], width)
            proof = SqlTurnProof(execution_id, session_key, request, receipt["request_digest"],
                                 receipt["owner_id"], fence, receipt["cancel_requested"],
                                 receipt["receipt_id"], receipt["receipt_kind"], active)
            if not active:
                return proof, None
            proof.validate_live_token(token)
            return proof, SqlTurnObservation(
                proof, revision, decode_time(row["turn_expiry"])
                <= decode_time(row["observed_at"]),
            )
        except (KeyError, AttributeError, TypeError, ValueError) as error:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error

    async def read(self, *, route, identity, lease=None):
        clock = await self._source_clock()
        row = (await self._tx._execute(self._route(route, identity, clock=clock))).mappings().one_or_none()
        if row is None:
            return None
        if lease is not None:
            # A wrong or retired lease proves no mutation authority, even if
            # the unrelated receipt payload has been corrupted.
            from ._turn_sql import turn_hex
            from ._turn_sql_schema import TURN_REQUEST_MAX_BYTES

            session, request, token, fence = lease
            try:
                slot = turn_session_address(session)
                expected = (slot[0], slot[1], turn_hex(request, TURN_REQUEST_MAX_BYTES), turn_hex(token, 32), fence)
            except (TypeError, ValueError):
                return None
            if expected != (row["turn_slot_kind"], row["turn_slot_key"], row["turn_request_route"],
                            row["turn_token"], row["turn_fence"]):
                return None
        if row["turn_slot_kind"] == STANDALONE:
            from ._turn_sql import SqlTurnRows

            source = SqlTurnRows(self._tx._metadata, self._store, self._scope, coowned=True)
            return row, *source.parse_standalone_control_receipt(row, decode_time=self._dialect.decode_database_time)
        return row, *self.parse_receipt(row, scope=self._scope, decode_time=self._dialect.decode_database_time)

    async def _deadline(self, *, live):
        from sqlalchemy import func

        expires, clock = self._table.c.turn_expiry, await self._source_clock()
        if self._dialect.single_writer:
            expires, clock = func.julianday(expires), func.julianday(clock)
        return expires > clock if live else expires <= clock

    @staticmethod
    def _receipt(proof, receipt, kind):
        return {"request_key": proof.request_key, "request_digest": proof.request_digest,
                "owner_id": proof.owner_id, "receipt_id": receipt, "receipt_kind": kind,
                "cancel_requested": kind == "cancel" or proof.cancel_requested}

    async def _advance(self, statement):
        from ._turn_sql import SqlTurnNotOwned

        if (await self._tx._execute(statement)).rowcount != 1:
            raise SqlTurnNotOwned()

    def _fence(self, row, proof, *, revision):
        table = self._table
        return table.update().where(
            self._match_execution(table, proof.execution_id), table.c.id == row["id"],
            table.c.turn_slot_kind == SESSION,
            table.c.turn_occupied_route == row["turn_slot_key"],
            table.c.turn_slot_key == row["turn_slot_key"], table.c.turn_fence == proof.fence,
            table.c.turn_token == row["turn_token"], table.c.turn_revision == revision,
        )

    async def claim(self, *, session_key, request_key, request_digest, execution_id, owner_id, token, duration):
        from datetime import timedelta
        from sqlalchemy import or_, select, true
        from ..errors import AIError, ErrorCode
        from ._execution_authority import ExecutionAuthority
        from ._record_owner import EXECUTION_SHELL_KIND
        from ._turn_sql import SqlTurnProof, turn_hex
        from ._turn_sql_schema import TURN_REQUEST_MAX_BYTES
        from .state import StateTransactionNestingError, active_state_scope, is_active_state_transaction

        active = active_state_scope()
        if active is None or not active.writable or not is_active_state_transaction(self._tx):
            raise StateTransactionNestingError("split session claim requires its active SQL write boundary")
        table = self._table
        kind, slot = turn_session_address(session_key)
        if kind != SESSION:
            raise ValueError("split session claims require a SESSION slot")
        request_route = turn_hex(request_key, TURN_REQUEST_MAX_BYTES)
        for value, width in ((owner_id, 32), (token, 32), (request_digest, 64)):
            turn_hex(value, width)
        clock = await self._source_clock()
        seed = select(clock.label("clock")).subquery()
        current = self._route("session", session_key, clock=clock).subquery()
        replay = self._route("request", request_key, clock=clock).subquery()
        candidate = table.alias()
        row = (await self._tx._execute(select(
            seed.c.clock, current,
            *(column.label("replay_" + column.name) for column in replay.c),
            candidate.c.id.label("candidate_id"),
        ).select_from(seed.outerjoin(current, true()).outerjoin(replay, true()).outerjoin(
            candidate, self._match_execution(candidate, execution_id),
        )))).mappings().one()
        if row["replay_id"] is not None:
            if row["replay_turn_slot_kind"] != SESSION:
                raise AIError(ErrorCode.IDEMPOTENCY_CONFLICT)
            replayed, observation = self.parse_receipt(
                {col.name: row["replay_" + col.name] for col in replay.c},
                scope=self._scope, decode_time=self._dialect.decode_database_time,
            )
            if (replayed.session_key, replayed.request_digest) != (session_key, request_digest):
                raise AIError(ErrorCode.IDEMPOTENCY_CONFLICT)
            if observation is not None and not observation.expired:
                return replayed.coordination(self._scope)
        previous, observation = (None, None) if row["id"] is None else self.parse_receipt(row, scope=self._scope, decode_time=self._dialect.decode_database_time)
        if observation is not None and not observation.expired:
            return None
        if previous is not None and previous.request_key == request_key and previous.request_digest != request_digest:
            raise AIError(ErrorCode.IDEMPOTENCY_CONFLICT)
        if (previous is not None and previous.execution_id != execution_id
                and row["kind"] == "execution" and row["scan_kind"] != "e:0"):
            # Keep the local anchor recoverable until the published execution
            # has finished its separate Conversation handoff. Otherwise a new
            # request can archive it, then fail to acquire Conversation.
            raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)
        if row["candidate_id"] is not None and (
            previous is None or previous.execution_id != execution_id
            or row["kind"] != EXECUTION_SHELL_KIND or previous.request_key != request_key
        ):
            raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)
        fence = 1 if previous is None else previous.fence + 1
        if not 1 <= fence <= 2**63 - 1:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if previous is not None and previous.owner_id == owner_id:
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        proof = SqlTurnProof(execution_id, session_key, request_key, request_digest, owner_id,
                             fence, False, token, "claim", True)
        authority = ExecutionAuthority(self._scope, execution_id, session_key, token, fence)
        expiry = self._dialect.decode_database_time(row["clock"]) + timedelta(seconds=duration)
        values = {**authority.authority_values(), "turn_occupied_route": slot,
                  "turn_request_route": request_route, "turn_receipt": self._receipt(proof, token, "claim"),
                  "turn_expiry": expiry}
        try:
            revision = None if previous is None else row["turn_revision"]
            if observation is not None:
                await self._advance(self._fence(row, previous, revision=revision).values(turn_revision=revision + 1))
                revision += 1
            if previous is not None and previous.execution_id != execution_id:
                retire = self._fence(row, previous, revision=revision).where(or_(
                    table.c.kind == EXECUTION_SHELL_KIND,
                    (table.c.kind == "execution") & (table.c.scan_kind == "e:0"),
                ))
                extras = {}
                unoccupied = unoccupied_turn_route(row["key_locator"])
                if observation is not None:
                    retire = retire.where(await self._deadline(live=False))
                    extras.update(turn_token=None, turn_request_route=unoccupied,
                                  turn_receipt=self._receipt(previous, token, "expire"))
                await self._advance(retire.values(turn_occupied_route=unoccupied, turn_revision=revision + 1, **extras))
            if row["candidate_id"] is None:
                await self._advance(table.insert().values(
                    authority.shell_values(store_key=self._store, expiry=expiry) | values,
                ))
                self._tx._record_cache[authority.key] = None
                granted_revision = 0
            else:
                reclaim = self._fence(row, previous, revision=revision)
                if observation is not None:
                    reclaim = reclaim.where(await self._deadline(live=False))
                granted_revision = revision + 1
                await self._advance(reclaim.values(turn_revision=granted_revision, **values))
            now = await self._tx._context.run_statement(self._dialect.database_current_time(self._tx._session))
            await self._advance(table.update().where(
                authority.owner_matches(table, self._store), table.c.turn_occupied_route == slot,
                table.c.turn_revision == granted_revision,
            ).values(turn_expiry=now + timedelta(seconds=duration)))
        except BaseException as error:
            self._tx._abort_mutation(error)
            raise
        return proof.coordination(self._scope, token=token)

    async def change(self, *, execution_id, receipt, kind, lease, duration=None, loaded=None, inline_deadline=None):
        from dataclasses import replace
        from datetime import timedelta
        from ..storage import MySQLDialect, PostgreSQLDialect, SQLiteDialect
        from ._turn_sql import turn_hex
        from .state import StateTransactionNestingError, active_state_scope, is_active_state_transaction

        active = active_state_scope()
        if active is None or not active.writable or not is_active_state_transaction(self._tx):
            raise StateTransactionNestingError("split session mutation requires its active SQL write boundary")
        if kind not in {"renew", "cancel", "release"} or kind == "renew" and duration is None:
            raise ValueError("invalid split session change")
        result = loaded if loaded is not None else await self.read(route="execution", identity=execution_id, lease=lease)
        if result is None:
            return None
        row, proof, observed = result
        if row["turn_slot_kind"] != SESSION or proof.execution_id != execution_id:
            raise ValueError("split session change requires its exact SESSION receipt")
        if observed is None or observed.expired:
            return None
        token = row["turn_token"].decode("ascii")
        if (lease is None and kind != "cancel") or (
            lease is not None and lease != (proof.session_key, proof.request_key, token, proof.fence)
        ):
            return None
        turn_hex(receipt, 32)
        revision = observed.slot_revision
        values = {"turn_receipt": self._receipt(proof, receipt, kind)}
        try:
            await self._advance(self._fence(row, proof, revision=revision).values(turn_revision=revision + 1))
            if kind == "release":
                # Park, retaining the slot anchor and this exact receipt.
                values.update(turn_token=None, turn_request_route=unoccupied_turn_route(row["key_locator"]))
            elif kind == "renew":
                interval = timedelta(seconds=duration)
                if inline_deadline is not None and type(self._dialect) in (SQLiteDialect, MySQLDialect, PostgreSQLDialect):
                    values["turn_expiry"] = inline_deadline(self._dialect, interval)
                else:
                    now = await self._tx._context.run_statement(self._dialect.database_current_time(self._tx._session))
                    values["turn_expiry"] = now + interval
            await self._advance(self._fence(row, proof, revision=revision + 1).where(
                await self._deadline(live=True),
            ).values(**values))
        except BaseException as error:
            self._tx._abort_mutation(error)
            raise
        return replace(proof, receipt_id=receipt, receipt_kind=kind).coordination(self._scope, token=token) if kind == "renew" else True

    async def regrant(self, *, session_key, request_key, request_digest, execution_id, owner_id, token, duration):
        """Resume only the exact published execution holding this split slot."""
        from dataclasses import replace
        from datetime import timedelta
        from ..errors import AIError, ErrorCode
        from ._turn_sql import turn_hex
        from ._turn_sql_schema import TURN_REQUEST_MAX_BYTES
        from .state import StateTransactionNestingError, active_state_scope, is_active_state_transaction

        active = active_state_scope()
        if active is None or not active.writable or not is_active_state_transaction(self._tx):
            raise StateTransactionNestingError("split session recovery requires its active SQL write boundary")
        result = await self.read(route="execution", identity=execution_id)
        if result is None:
            raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)
        row, proof, observed = result
        if (proof.session_key, proof.request_key, proof.request_digest) != (session_key, request_key, request_digest):
            raise AIError(ErrorCode.IDEMPOTENCY_CONFLICT)
        if row["turn_slot_kind"] != SESSION or row["kind"] != "execution":
            raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)
        if row["turn_occupied_route"] != row["turn_slot_key"] or proof.owner_id == owner_id:
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        if observed is not None and not observed.expired:
            return proof.coordination(self._scope)
        request_route = turn_hex(request_key, TURN_REQUEST_MAX_BYTES)
        turn_hex(owner_id, 32)
        token_key = turn_hex(token, 32)
        fence, revision = proof.fence + 1, row["turn_revision"]
        if fence > 2**63 - 1:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        replacement = replace(proof, owner_id=owner_id, fence=fence, receipt_id=token,
                              receipt_kind="claim", active=True)
        try:
            await self._advance(self._fence(row, proof, revision=revision).values(turn_revision=revision + 1))
            now = await self._tx._context.run_statement(self._dialect.database_current_time(self._tx._session))
            resume = self._fence(row, proof, revision=revision + 1)
            if proof.active:
                resume = resume.where(await self._deadline(live=False))
            await self._advance(resume.values(
                turn_token=token_key, turn_fence=fence, turn_request_route=request_route,
                turn_expiry=now + timedelta(seconds=duration),
                turn_receipt=self._receipt(replacement, token, "claim"),
            ))
        except BaseException as error:
            self._tx._abort_mutation(error)
            raise
        return replacement.coordination(self._scope, token=token)
