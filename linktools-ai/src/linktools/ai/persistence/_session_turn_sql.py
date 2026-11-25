#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SQL authority for sessions whose turns can be regenerated."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from threading import RLock
from typing import TYPE_CHECKING, Literal
from weakref import finalize

from ..core import canonical_json_bytes, canonical_sha256, semantic_json_sha256, validate_idempotency_key
from ..errors import AIError, ErrorCode
from pydantic_ai.messages import ModelMessage, ModelRequest, SystemPromptPart

from ..model.messages import decode_model_messages, encode_model_messages
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
from ._message_root import NativeMessageRoot
from .state import SqlStateRowReader, state_store_key

if TYPE_CHECKING:
    from sqlalchemy import MetaData, RowMapping
    from sqlalchemy.ext.asyncio import AsyncSession

    from ..core import JsonValue
    from ..storage import SqlStorageContext


SESSION_TABLE = "ai_sessions"
SESSION_TURN_TABLE = "ai_session_turns"
SESSION_BINDING_TABLE = "ai_session_bindings"
TerminalState = Literal["succeeded", "failed", "cancelled"]
_CONTEXT_POSITIONS = 128
_HISTORY_CACHE_BYTES = 16 * 1024 * 1024
_HISTORY_CACHE_SESSIONS = 32
_HISTORY_CACHE_LOCK = RLock()
_history_cache_total = 0


@dataclass(slots=True)
class _ContextBudget:
    used: int = 0


def _release_context_budget(budget: _ContextBudget) -> None:
    global _history_cache_total
    with _HISTORY_CACHE_LOCK:
        _history_cache_total -= budget.used
        budget.used = 0



@dataclass(frozen=True, slots=True)
class SessionContextTurn:
    turn_no: int
    request: JsonValue
    messages: tuple[ModelMessage, ...]
    size: int
    transcript_bytes: int


@dataclass(frozen=True, slots=True)
class _ContextWindow:
    after: int
    through: int
    turns: tuple[SessionContextTurn, ...]
    size: int


def build_session_turn_sql_metadata(metadata: MetaData) -> None:
    from sqlalchemy import JSON, BigInteger, Column, PrimaryKeyConstraint, String, Table

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

    bindings = Table(
        SESSION_BINDING_TABLE, metadata,
        key("scope_key", 32, "Partition of one tenant in the configured session store."),
        key("binding_key", 32, "Content identity of the frozen compiled Agent binding."),
        Column("binding_json", JSON(), nullable=False, comment="Immutable admitted Agent binding."),
        *sql_audit_columns(),
        PrimaryKeyConstraint("scope_key", "binding_key"),
        comment="Shared frozen Agent bindings for regenerable conversation turns.",
        **sql_table_options(),
    )

    sql_audit_indexes(bindings)

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
        key("binding_key", 32, "Frozen Agent binding in the same scope."),
        Column("request_json", JSON(), nullable=False, comment="Accepted principal and canonical request."),
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
        self._bindings = metadata.tables[SESSION_BINDING_TABLE]
        self._transcript_source = None
        self._recent: OrderedDict[str, _ContextWindow] = OrderedDict()
        self._recent_budget = _ContextBudget()
        self._recent_finalizer = finalize(self, _release_context_budget, self._recent_budget)

    def bind_transcript_reader(self, conversation_store, archive_provider, capture) -> None:
        """Bind native transcript reads to the same SQL snapshot as turn metadata."""
        self._transcript_source = conversation_store, archive_provider, capture

    async def _read_transcripts(self, statement, *, display: bool, body_limit: int | None = None) -> tuple[Mapping[str, object], ...]:
        if self._transcript_source is None:
            raise AIError(ErrorCode.STORAGE_DEPENDENCY_NOT_READY)
        turn_store, archive_provider, capture = self._transcript_source
        group = turn_store.storage_group
        opened_archive = group.opened_archive or await group.open_archive(create=False)
        archive_store = None if opened_archive is None else (await archive_provider()).store
        if archive_store is not None and archive_store.storage_group is not group:
            raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)

        async def read(group):
            owner = group.transaction(turn_store)
            if not isinstance(owner, SqlStateRowReader):
                raise AIError(ErrorCode.STORAGE_DEPENDENCY_NOT_READY)
            rows = await owner.read_sql_rows(self._context, statement)
            manifests = [
                self._transcript_manifest(row["result_json"])
                for row in rows[:body_limit] if row["state"] == "succeeded"
            ]
            if not manifests:
                return rows, manifests, None
            if archive_store is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            archive = group.transaction(archive_store)
            if not isinstance(archive, SqlStateRowReader):
                raise AIError(ErrorCode.STORAGE_DEPENDENCY_NOT_READY)
            load = await capture(tuple(root for root, _ in manifests), transaction=archive)
            return rows, manifests, load

        members = (turn_store, archive_store) if archive_store not in (None, turn_store) else (turn_store,)
        rows, manifests, load = await group.read_many(members, read)
        loaded = iter(zip(manifests, () if load is None else await load(), strict=True))
        projected = []
        for position, row in enumerate(rows):
            if row["state"] != "succeeded" or (body_limit is not None and position >= body_limit):
                projected.append(row)
                continue
            try:
                (root, refs), encoded = next(loaded)
            except (StopIteration, ValueError) as error:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
            if len(encoded) != root.count:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            messages = []
            for body in encoded:
                parts = decode_model_messages(body)
                if len(parts) != 1:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                messages.append(parts[0])
            if not display:
                self._restore_binding_refs(messages, refs, row["instructions"], row["system_prompt"])
            projected.append({**row, "messages": tuple(messages)})
        try:
            next(loaded, None)
        except ValueError as error:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
        return tuple(projected)

    @staticmethod
    def _transcript_manifest(result) -> tuple[NativeMessageRoot, tuple[tuple[int, int | None], ...]]:
        try:
            if not isinstance(result, Mapping) or result.keys() != {"transcript", "binding_refs"}:
                raise ValueError("invalid session transcript manifest")
            root = NativeMessageRoot.from_json(result["transcript"])
            refs = result["binding_refs"]
            if not root.count or not isinstance(refs, list):
                raise ValueError("invalid session transcript range")
            seen = set()
            parsed = []
            for ref in refs:
                if (not isinstance(ref, list) or len(ref) != 2 or type(ref[0]) is not int
                        or not 0 <= ref[0] < root.count
                        or (ref[1] is not None and (type(ref[1]) is not int or ref[1] < 0))):
                    raise ValueError("invalid session transcript binding reference")
                item = tuple(ref)
                if item in seen:
                    raise ValueError("duplicate session transcript binding reference")
                seen.add(item)
                parsed.append(item)
            return root, tuple(parsed)
        except (TypeError, ValueError) as error:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error

    @staticmethod
    def _restore_binding_refs(messages, refs, instructions, system_prompt) -> None:
        static_instructions = None
        for position, part_index in refs:
            current = messages[position]
            if not isinstance(current, ModelRequest):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if part_index is None:
                if current.instructions is not None:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                if static_instructions is None:
                    if not isinstance(instructions, list) or not all(isinstance(line, str) for line in instructions):
                        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                    static_instructions = "\n".join(instructions)
                    if not static_instructions:
                        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                messages[position] = replace(current, instructions=static_instructions)
                continue
            if part_index >= len(current.parts) or not isinstance(system_prompt, str) or not system_prompt:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            part = current.parts[part_index]
            if not isinstance(part, SystemPromptPart) or part.content != "" or part.dynamic_ref is not None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            parts = list(current.parts)
            parts[part_index] = replace(part, content=system_prompt)
            messages[position] = replace(current, parts=parts)

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

    def _bound_turns(self, *, extra=None):
        turns, bindings = self._turns.c, self._bindings.c
        same_binding = (turns.scope_key == bindings.scope_key) & (turns.binding_key == bindings.binding_key)
        return self._turns.outerjoin(self._bindings, same_binding if extra is None else same_binding & extra)

    async def get_turn_for_recovery(self, key: SessionTurnKey) -> RowMapping | None:
        from sqlalchemy import select

        statement = select(self._turns, self._bindings.c.binding_json).select_from(self._bound_turns()).where(
            *(self._turns.c[name] == value for name, value in self._turn_values(key).items()),
        )
        async with self._context.engine.connect() as connection:
            return (await connection.execute(statement)).mappings().one_or_none()

    def _history_view_select(self):
        from sqlalchemy import select

        columns = self._turns.c
        return select(
            columns.turn_id, columns.turn_no, columns.state, columns.created_at, columns.updated_at,
            columns.result_json, columns.request_json["prompt"].as_string().label("prompt"),
            columns.request_json["input"].label("stored_input"),
        )

    async def get_history_turn(
        self, key: SessionTurnKey, *, lineage: tuple[SessionTurnSegment, ...],
    ) -> Mapping[str, object] | None:
        from sqlalchemy import and_, or_

        columns = self._turns.c
        own = columns.session_key == self._key(key.session_id, TURN_SESSION_MAX_BYTES)
        scope = own if not lineage else or_(
            own,
            and_(
                columns.state.in_(("succeeded", "failed", "cancelled")),
                lineage_predicate(columns, b"", lineage, after=0, through=lineage[-1].through),
            ),
        )
        statement = self._history_view_select().where(
            columns.scope_key == self._scope,
            columns.turn_id == self._key(key.turn_id, TURN_EXECUTION_MAX_BYTES),
            scope,
        )
        rows = await self._read_transcripts(statement, display=True)
        return rows[0] if rows else None

    async def find_request(self, session_id: str, request_key: str) -> RowMapping | None:
        from sqlalchemy import select

        columns = self._turns.c
        agent_id = self._bindings.c.binding_json[("agent_spec", "id")].as_string().label("agent_id")
        statement = select(columns.turn_id, columns.request_json, agent_id).select_from(self._bound_turns()).where(
            columns.scope_key == self._scope,
            columns.session_key == self._key(session_id, TURN_SESSION_MAX_BYTES),
            columns.request_key == self._key(request_key, TURN_REQUEST_MAX_BYTES),
        )
        async with self._context.engine.connect() as connection:
            return (await connection.execute(statement)).mappings().one_or_none()

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
        body_limit: int | None = None,
    ) -> tuple[Mapping[str, object], ...]:
        columns = self._turns.c
        statement = self._history_view_select().where(
            columns.scope_key == self._scope,
            lineage_predicate(columns, self._key(session_id, TURN_SESSION_MAX_BYTES), lineage,
                              after=0, through=through),
            columns.state.in_(("succeeded", "failed", "cancelled")),
        ).order_by(columns.turn_no.desc()).limit(limit)
        return await self._read_transcripts(statement, display=True, body_limit=body_limit)

    async def recent_context(
        self, session_id: str, *, through: int, lineage: tuple[SessionTurnSegment, ...] = (),
    ) -> tuple[SessionContextTurn, ...]:
        """Read bounded candidates after the caller authorizes the current session head."""
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

        turns += await self.read_context_range(session_id, after=after, through=through, lineage=lineage)
        self._remember_context(session_id, after=start, through=through, turns=turns)
        return turns

    async def read_context_range(
        self, session_id: str, *, after: int, through: int, lineage: tuple[SessionTurnSegment, ...] = (),
        missing: ErrorCode = ErrorCode.STORAGE_INTEGRITY_ERROR,
    ) -> tuple[SessionContextTurn, ...]:
        """Read an exact finalized interval, omitting non-success bodies."""
        from sqlalchemy import case, select, type_coerce

        if not 0 <= through - after <= _CONTEXT_POSITIONS or after < 0:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR, "The admitted context range is invalid")
        if after == through:
            return ()

        columns = self._turns.c
        succeeded = columns.state == "succeeded"
        prompt = case((succeeded, columns.request_json["prompt"].as_string())).label("prompt")
        result = type_coerce(case((succeeded, columns.result_json)), columns.result_json.type).label("result_json")
        stored_input = type_coerce(
            case((succeeded, columns.request_json["input"])), columns.request_json.type,
        ).label("stored_input")
        has_refs = columns.result_json["binding_refs"][0].as_string().is_not(None)
        binding_json = self._bindings.c.binding_json
        instructions = type_coerce(case((
            has_refs, binding_json[("agent_spec", "instructions")],
        )), binding_json.type).label("instructions")
        system_prompt = case((
            has_refs, binding_json[("agent_spec", "system_prompt")].as_string(),
        )).label("system_prompt")
        statement = select(
            columns.turn_no, columns.state, prompt, stored_input, result, instructions, system_prompt,
        ).select_from(self._bound_turns(extra=succeeded & has_refs)).where(
            columns.scope_key == self._scope,
            lineage_predicate(columns, self._key(session_id, TURN_SESSION_MAX_BYTES), lineage,
                              after=after, through=through),
            columns.state.in_(("succeeded", "failed", "cancelled")),
        ).order_by(columns.turn_no).limit(_CONTEXT_POSITIONS)
        rows = await self._read_transcripts(statement, display=False)
        if len(rows) != through - after:
            raise AIError(missing, "The finalized session context range is incomplete")
        return tuple(
            self._context_turn(
                row["turn_no"], row["prompt"], row["stored_input"], row["messages"],
            )
            for row in rows if row["state"] == "succeeded"
        )

    @staticmethod
    def _context_turn(
        turn_no: int, prompt: str, stored_input: JsonValue, messages: Sequence[ModelMessage],
    ) -> SessionContextTurn:
        request = {"prompt": prompt, **({"input": stored_input} if stored_input is not None else {})}
        transcript_bytes = len(encode_model_messages(messages))
        size = max(1024, len(canonical_json_bytes(request)) + transcript_bytes)
        return SessionContextTurn(turn_no, request, tuple(messages), size, transcript_bytes)

    def _remember_context(
        self, session_id: str, *, after: int, through: int, turns: tuple[SessionContextTurn, ...],
    ) -> None:
        global _history_cache_total
        # Charge an entry even for windows containing only failed/cancelled turns.
        size = 1024 + sum(turn.size for turn in turns)
        with _HISTORY_CACHE_LOCK:
            if not self._recent_finalizer.alive:
                return
            previous = self._recent.get(session_id)
            if previous is not None and previous.through > through:
                return
            owner = self._recent_budget
            if previous is not None:
                owner.used -= self._recent.pop(session_id).size
                _history_cache_total -= previous.size
            if size > _HISTORY_CACHE_BYTES:
                return
            while self._recent and (
                _history_cache_total + size > _HISTORY_CACHE_BYTES or len(self._recent) >= _HISTORY_CACHE_SESSIONS
            ):
                released = self._recent.popitem(last=False)[1].size
                owner.used -= released
                _history_cache_total -= released
            if _history_cache_total + size > _HISTORY_CACHE_BYTES:
                return
            self._recent[session_id] = _ContextWindow(after, through, turns, size)
            owner.used += size
            _history_cache_total += size

    def clear_recent_context(self) -> None:
        """Drop retained history and disable caching for this closed Persistence."""
        with _HISTORY_CACHE_LOCK:
            self._recent.clear()
            self._recent_finalizer()


class SqlSessionTurns(SqlSessionTurnReader):
    """Write authority for one tenant’s ordinary SQL sessions."""

    def __init__(self, context: SqlStorageContext, metadata: MetaData, *, namespace: str, tenant_id: str) -> None:
        super().__init__(context, metadata, namespace=namespace, tenant_id=tenant_id)
        self._now = sql_epoch_seconds()
        self._committed_bindings: OrderedDict[bytes, None] = OrderedDict()
        self._transcript_write_binding = None

    def bind_transcript_writer(self, store, archive_provider, native) -> None:
        """Bind the canonical turn to its existing same-group Native body writer."""
        self._transcript_write_binding = (store, archive_provider, native)

    def note_terminal_context(
        self, key: SessionTurnKey, *, turn_no: int, prompt: str | None, stored_input: JsonValue,
        messages: Sequence[ModelMessage], succeeded: bool,
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
            turns += (self._context_turn(
                turn_no, prompt, stored_input, messages,
            ),)
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
        request_key: str, request: JsonValue, binding: JsonValue, lease_seconds: int,
    ) -> bool:
        """Claim and record a turn atomically; a uniqueness failure rolls back the claim."""
        if lease_seconds < 1:
            raise ValueError("the turn lease must be positive")
        if not 1 <= context_from <= context_through + 1:
            raise ValueError("invalid admitted session context range")
        session_key = self._session_values(key.session_id)
        turn_key = self._turn_values(key)
        encoded_request_key = self._key(request_key, TURN_REQUEST_MAX_BYTES)
        binding_key = bytes.fromhex(semantic_json_sha256(binding))

        async def write(connection: AsyncSession) -> bool:
            claim = await connection.execute(self._sessions.update().filter_by(
                **session_key, status="open", revision=expected_revision,
                last_turn_no=context_through, active_turn_id=None,
            ).values(active_turn_id=turn_key["turn_id"], revision=expected_revision + 1))
            if claim.rowcount != 1:
                return False
            if binding_key not in self._committed_bindings:
                await self._context.dialect.upsert(
                    connection, table=self._bindings,
                    values={"scope_key": self._scope, "binding_key": binding_key, "binding_json": binding},
                    set_values={"binding_key": self._bindings.c.binding_key},
                    index_elements=("scope_key", "binding_key"),
                )
            await connection.execute(self._turns.insert().values(
                **turn_key, request_key=encoded_request_key, binding_key=binding_key, turn_no=context_through + 1,
                context_from=context_from, context_through=context_through, state="accepted", generation=1,
                lease_until=self._now + lease_seconds, request_json=request,
            ))
            return True

        accepted = await self._context.run_mutation(write, domain="session.accept")
        if accepted:
            self._committed_bindings[binding_key] = None
            self._committed_bindings.move_to_end(binding_key)
            if len(self._committed_bindings) > 128:
                self._committed_bindings.popitem(last=False)
        return accepted

    async def finish(
        self, key: SessionTurnKey, *, generation: int, turn_no: int, state: TerminalState,
        result: JsonValue = None, transcript: "Sequence[ModelMessage] | None" = None,
        binding_refs: tuple[tuple[int, int | None], ...] = (),
    ) -> bool:
        """Finalize and release using the same lock order as acceptance."""
        class FenceLost(Exception):
            pass

        if state not in ("succeeded", "failed", "cancelled"):
            raise ValueError("invalid terminal state")
        turn_key = self._turn_values(key)
        if state == "succeeded":
            from ._archive import PreparedArchive

            if result is not None or not transcript:
                raise ValueError("a successful session turn requires only its typed transcript")
            if self._transcript_write_binding is None:
                raise AIError(ErrorCode.STORAGE_DEPENDENCY_NOT_READY)
            store, archive_provider, native = self._transcript_write_binding
            archive = await archive_provider(create=True)
            group = store.storage_group
            if archive.store.storage_group is not group:
                raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)
            prepared = PreparedArchive()
            root = await native.prepare_archived_messages(transcript, prepared)
            result = {"transcript": root.to_json(), "binding_refs": [list(ref) for ref in binding_refs]}
        elif transcript is not None or binding_refs:
            raise ValueError("an unsuccessful session turn cannot publish a transcript")

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
            if state != "succeeded":
                return await self._context.run_mutation(write, domain="session.finish")

            async def publish(transaction) -> bool:
                from ._native_content_index import stage_owner

                async def write_owner(session: AsyncSession) -> bool:
                    if not await write(session):
                        return False
                    stage_owner(
                        session, SESSION_TURN_TABLE, self._scope,
                        (turn_key["session_key"], turn_key["turn_id"]), root,
                    )
                    return True

                if not await transaction.transaction(store).run_sql(self._context, write_owner):
                    return False
                await archive.publish(transaction.transaction(archive.store), prepared)
                return True

            return await group.mutate(tuple(dict.fromkeys((store, archive.store))), publish)
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
