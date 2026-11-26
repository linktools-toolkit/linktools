#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SQL authority for sessions whose turns can be regenerated."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from threading import RLock
from typing import TYPE_CHECKING, Literal, TypeVar
from weakref import finalize

from ..agent import AgentBindingContract
from ..core import (
    AuthorizationAction, ResourceKind, ResourceRef, TenantAuthorizationPolicy,
    RUNTIME_OBJECT_STORE_ID, SESSION_REQUEST_KEY_MAX_BYTES, ImmutableJsonMapping, canonical_json_bytes, canonical_sha256,
    deterministic_id, validate_session_request_key,
)
from ..errors import AIError, ErrorCode
from pydantic_ai.messages import ModelMessage, ModelRequest, SystemPromptPart, UserPromptPart

from ..model.messages import (
    binary_content_usage, decode_model_messages, encode_model_messages, project_transient_binary_content,
    session_compaction_summaries,
)
from ..spec import RepositoryInstructions
from ..workspace import validate_workspace_path
from ..storage import (
    sql_audit_columns,
    sql_audit_indexes,
    sql_binary_key,
    sql_epoch_seconds,
    sql_id_column,
    sql_table_options,
    sql_unique,
)
from ._archive import ArchiveContentInput, PreparedArchive
from ._object_key import ObjectKeyFactory
from ._plan import PersistenceDomain
from .codec import decode_agent_binding_content, encode_agent_binding_content, encode_domain, restore_native_user_input
from ._ordinary_lineage import SessionTurnSegment, decode_session_lineage, fork_segments, lineage_predicate
from ._turn_sql_schema import (
    NATIVE_TURN_ID_MAX_BYTES,
    TURN_SESSION_MAX_BYTES,
)
from ._message_root import NativeMessageRoot
from ._session_transcript import (
    SESSION_TURN_TABLE,
    decode_session_compacted_context,
    decode_session_input_reference,
    decode_session_turn_usage,
    decode_session_transcript_root,
)
from .state import SqlStateRowReader, state_store_key

if TYPE_CHECKING:
    from sqlalchemy import MetaData, RowMapping
    from sqlalchemy.ext.asyncio import AsyncSession

    from ..core import JsonValue, RunBudget, UsageMetrics
    from ..model import StoredUserInput
    from ..storage import ObjectRef
    from .contracts import BudgetRepository
    from ._checkpoint_repositories import ToolRepositoryImpl
    from ..core import AuthorizationPolicy, Principal
    from ._archive import ContentArchive
    from ..execution import PendingToolContinuation
    from .state import StateGroupTransaction, StateStore, StateTransaction
    from ..storage import SqlStorageContext
    from ._split_step_source import SplitStepSourceObservation, SplitStepWaiting
    from ._step_store import SqlStepStore
    from pydantic_ai_harness.step_persistence import ContinuableSnapshot


SESSION_TABLE = "ai_sessions"
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
    turn_id: str
    turn_no: int
    request: JsonValue
    messages: tuple[ModelMessage, ...]
    size: int
    transcript_bytes: int
    reset_context: bool = False
    interrupted: bool = False
    seed_checkpoint_prefix: Callable[[str, Sequence[ModelMessage]], bool] | None = None
    compacted_context: tuple[ModelMessage, ...] = ()


@dataclass(frozen=True, slots=True)
class _ContextWindow:
    after: int
    through: int
    max_context_bytes: int
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
        key("active_turn_id", NATIVE_TURN_ID_MAX_BYTES, "Accepted turn occupying the session.", nullable=True),
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
        key("turn_id", NATIVE_TURN_ID_MAX_BYTES, "Exact public turn identity."),
        key("request_key", SESSION_REQUEST_KEY_MAX_BYTES, "Idempotency key scoped to the session."),
        Column("turn_no", BigInteger, nullable=False, comment="Sequential turn position within the session."),
        Column("context_from", BigInteger, nullable=False, comment="First selected context position, inclusive."),
        Column("context_through", BigInteger, nullable=False, comment="Finalized context boundary at admission."),
        Column("state", String(10), nullable=False, comment="Admission, cancellation or terminal state."),
        Column("generation", BigInteger, nullable=False, comment="Fencing generation of the accepted producer."),
        Column("lease_until", BigInteger, nullable=True, comment="Expiry in database-clock seconds since epoch."),
        key("binding_key", 32, "Frozen Agent binding in the same scope."),
        Column("request_json", JSON(), nullable=False, comment="Accepted principal and canonical request."),
        Column("split_source_receipt", sql_binary_key(64), nullable=True,
               comment="Exact separately committed Step receipt; absent for same-group publication."),
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


@dataclass(frozen=True, slots=True)
class SessionInstructionFrontier:
    """Frozen discovery evidence outside the immutable request and public history."""

    overlay: RepositoryInstructions

    def to_json(self) -> dict[str, JsonValue]:
        return {"overlay": self.overlay.to_payload()}

    @classmethod
    def from_json(cls, value: object) -> SessionInstructionFrontier | None:
        if value is None:
            return None
        try:
            if not isinstance(value, Mapping) or value.keys() != {"overlay"}:
                raise ValueError("invalid instruction frontier")
            overlay = RepositoryInstructions.from_payload(value["overlay"])
            if not overlay.documents:
                raise ValueError("empty instruction frontier")
        except (TypeError, ValueError, AIError) as error:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
        return cls(overlay)


_Receipt = TypeVar("_Receipt")


@dataclass(frozen=True, slots=True)
class CanonicalDeferredFrontier:
    key: SessionTurnKey
    generation: int
    checkpoint_id: str
    approval_operation_ids: tuple[str, ...]
    external_operation_ids: tuple[str, ...]
    operation_metadata: ImmutableJsonMapping

    def to_json(self) -> dict[str, JsonValue]:
        return {
            "checkpoint_id": self.checkpoint_id,
            "approval_operation_ids": list(self.approval_operation_ids),
            "external_operation_ids": list(self.external_operation_ids),
            "operation_metadata": dict(self.operation_metadata),
        }

    @classmethod
    def from_json(cls, key: SessionTurnKey, generation: int, value: object) -> CanonicalDeferredFrontier:
        from ..core import validate_resource_id

        try:
            if not isinstance(value, Mapping) or value.keys() != {
                "checkpoint_id", "approval_operation_ids", "external_operation_ids", "operation_metadata",
            } or not all(isinstance(value[name], list) for name in ("approval_operation_ids", "external_operation_ids")):
                raise ValueError("invalid deferred frontier")
            checkpoint = validate_resource_id(value["checkpoint_id"])
            approvals = tuple(validate_resource_id(item) for item in value["approval_operation_ids"])
            calls = tuple(validate_resource_id(item) for item in value["external_operation_ids"])
            if not (approvals or calls) or len(set(approvals + calls)) != len(approvals + calls):
                raise ValueError("invalid deferred operations")
            metadata = value["operation_metadata"]
            if (not isinstance(metadata, Mapping) or metadata.keys() != set(approvals + calls)
                    or not all(isinstance(item, Mapping) for item in metadata.values())):
                raise ValueError("invalid deferred operation metadata")
            metadata = ImmutableJsonMapping(metadata)
        except (KeyError, ValueError, TypeError, AIError) as error:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
        return cls(key, generation, checkpoint, approvals, calls, metadata)

    @classmethod
    def from_split_waiting(
        cls, key: SessionTurnKey, *, checkpoint_id: str, waiting: SplitStepWaiting,
    ) -> CanonicalDeferredFrontier:
        """Rebuild the metadata-free frontier solely from its retained source proof."""
        operation = waiting.operation_id
        return cls.from_json(key, waiting.original_generation, {
            "checkpoint_id": checkpoint_id,
            "approval_operation_ids": [operation] if waiting.kind == "approval" else [],
            "external_operation_ids": [operation] if waiting.kind == "external" else [],
            "operation_metadata": {operation: {}},
        })

    @property
    def split_receipt_id(self) -> str:
        """Bind the source receipt to the exact target publication payload."""
        return deterministic_id(
            "canonical-split-waiting", self.key.session_id, self.key.turn_id, self.generation,
            canonical_sha256(self.to_json()),
        )

    def require_continuation(self, continuation: PendingToolContinuation) -> None:
        if (
            continuation.source_agent_run_id != self.key.turn_id
            or tuple(item.operation_id for item in continuation.approvals) != self.approval_operation_ids
            or tuple(item.operation_id for item in continuation.calls) != self.external_operation_ids
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)


class SqlSessionTurnReader:
    """Read one tenant’s committed session and turn history."""

    def __init__(self, context: SqlStorageContext, metadata: MetaData, *, namespace: str, tenant_id: str) -> None:
        self._context = context
        self._scope = state_store_key(namespace, tenant_id, "regenerable-session")
        self._sessions = metadata.tables[SESSION_TABLE]
        self._turns = metadata.tables[SESSION_TURN_TABLE]
        self._transcript_source = None
        self._input_reader: Callable[[ObjectRef], Awaitable[bytes]] | None = None
        self._input_object_key = ObjectKeyFactory(namespace)
        self._tenant_id = tenant_id
        self._split_transcript_source = None
        self._recent: OrderedDict[str, _ContextWindow] = OrderedDict()
        self._recent_budget = _ContextBudget()
        self._recent_finalizer = finalize(self, _release_context_budget, self._recent_budget)

    def bind_transcript_reader(self, conversation_store, archive_provider, capture) -> None:
        """Bind native transcript reads to the same SQL snapshot as turn metadata."""
        self._transcript_source = conversation_store, archive_provider, capture

    def bind_input_reader(self, read: Callable[[ObjectRef], Awaitable[bytes]]) -> None:
        self._input_reader = read

    async def read_scalar_input(self, value: StoredUserInput) -> str:
        """Resolve the authorized turn's external scalar, never an attachment body."""
        ref = value.payload.ref
        if value.codec != "text" or ref is None or value.view is not None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if ref.store_id != RUNTIME_OBJECT_STORE_ID or ref.key != self._input_object_key.key(
            PersistenceDomain.EXECUTION, self._tenant_id, ref.digest,
        ):
            raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)
        if self._input_reader is None:
            raise AIError(ErrorCode.STORAGE_DEPENDENCY_NOT_READY)
        try:
            return (await self._input_reader(ref)).decode("utf-8")
        except UnicodeDecodeError as error:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error

    def _bind_split_transcript_reader(self, resolve) -> None:
        self._split_transcript_source = resolve

    async def _read_transcripts(self, statement, *, display: bool, body_limit: int | None = None,
                                compacted_context: bool | str = False,
                                reset_window: tuple[int, int, ErrorCode] | None = None,
                                expected_turns: range | None = None,
                                ) -> tuple[Mapping[str, object], ...]:
        if self._transcript_source is None:
            raise AIError(ErrorCode.STORAGE_DEPENDENCY_NOT_READY)
        turn_store, archive_provider, capture = self._transcript_source
        group = turn_store.storage_group
        opened_archive = group.opened_archive or await group.open_archive(create=False)
        content_archive = None if opened_archive is None else await archive_provider()
        archive_store = None if content_archive is None else content_archive.store
        if archive_store is not None and archive_store.storage_group is not group:
            raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)

        async def read(group):
            owner = group.transaction(turn_store)
            if not isinstance(owner, SqlStateRowReader):
                raise AIError(ErrorCode.STORAGE_DEPENDENCY_NOT_READY)
            rows = await owner.read_sql_rows(self._context, statement)
            if expected_turns is not None and (
                    len(rows) != len(expected_turns)
                    or any(row["turn_no"] != number for row, number in zip(rows, expected_turns))):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR, "The finalized session history page is incomplete")
            if reset_window is not None:
                after, through, missing = reset_window
                if (len(rows) != through - after
                        or any(row["turn_no"] != number for number, row in enumerate(rows, after + 1))):
                    raise AIError(missing, "The finalized session context range is incomplete")
                for index in range(len(rows) - 1, -1, -1):
                    if rows[index]["state"] == "succeeded" and rows[index]["reset_context"] is True:
                        rows = rows[index:]
                        break
            selected = rows[:body_limit]
            prompts = {}
            for position, row in enumerate(selected):
                value = row["stored_input"]
                if not isinstance(value, Mapping) or "$native_user_input" not in value:
                    continue
                if row["split_source_receipt"] is not None:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                reference = decode_session_input_reference(
                    {"input": value, "prompt": row["prompt"], "graph_execution": row["graph_execution"]},
                    row["turn_id"].decode("utf-8"),
                )
                if reference is not None:
                    prompts[position] = reference
            successes = [row for row in selected if row["state"] == "succeeded"]
            manifests = [self._transcript_manifest(row["result_json"]) for row in successes]
            local_positions = [
                index for index, row in enumerate(successes) if row["split_source_receipt"] is None
            ]
            foreign = [
                (index, row, manifests[index][0]) for index, row in enumerate(successes)
                if row["split_source_receipt"] is not None
            ]
            latest_context = None
            if compacted_context:
                for index, row in reversed(tuple(enumerate(selected))):
                    if row["reset_context"] and row["state"] == "succeeded" and "compacted_context" not in row["result_json"]:
                        break
                    if row["state"] != "succeeded" or (isinstance(compacted_context, str)
                            and row["turn_id"].decode("utf-8") != compacted_context):
                        continue
                    context_root = decode_session_compacted_context(row["result_json"])
                    if context_root is not None:
                        if row["split_source_receipt"] is not None or row["graph_execution"] is not None:
                            raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)
                        latest_context = index, context_root
                        break
            load = None
            roots = tuple(dict.fromkeys((
                *(manifests[index][0] for index in local_positions),
                *(reference.root for reference in prompts.values()),
                *(() if latest_context is None else (latest_context[1],)),
            )))
            if roots:
                if archive_store is None:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                archive = group.transaction(archive_store)
                if not isinstance(archive, SqlStateRowReader):
                    raise AIError(ErrorCode.STORAGE_DEPENDENCY_NOT_READY)
                load = await capture(roots, transaction=archive)
            binding_keys = tuple(dict.fromkeys(
                row["binding_key"] for row in rows[:body_limit]
                if not display and row["state"] == "succeeded" and row["result_json"]["binding_refs"]
            ))
            if binding_keys and content_archive is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            bindings = await self._read_binding_payloads(content_archive, binding_keys)
            return rows, manifests, local_positions, foreign, load, bindings, roots, prompts, latest_context

        members = (turn_store, archive_store) if archive_store not in (None, turn_store) else (turn_store,)
        rows, manifests, local_positions, foreign, load, bindings, roots, prompts, latest_context = await group.read_many(members, read)
        bodies = dict(zip(roots, () if load is None else await load(), strict=True))
        encoded_by_position = {index: bodies[manifests[index][0]] for index in local_positions}
        if foreign:
            if self._split_transcript_source is None:
                raise AIError(ErrorCode.STORAGE_DEPENDENCY_NOT_READY)
            from ._split_session_transcript import SplitTranscriptReference

            # The earlier authorized Conversation read cannot be held open while
            # reading a different database. Exact retained receipts prove the link.
            source = await self._split_transcript_source()
            contents = await source.read(tuple(
                SplitTranscriptReference(
                    row["turn_id"].decode("utf-8"), row["session_key"].decode("utf-8"),
                    row["generation"], row["split_source_receipt"].decode("utf-8"), root,
                ) for _, row, root in foreign
            ))
            encoded_by_position.update(zip((index for index, _, _ in foreign), contents, strict=True))
        loaded = iter(
            (manifest, encoded_by_position[index]) for index, manifest in enumerate(manifests)
        )
        projected = []
        for position, row in enumerate(rows):
            admitted_prompt = None
            if position in prompts:
                from ..model import native_graph_input_content

                reference = prompts[position]
                parts = bodies[reference.root]
                if len(parts) != 1:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                stored = restore_native_user_input(reference, parts[0])
                admitted_prompt = native_graph_input_content(stored)
                if admitted_prompt is None:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                row = ({**row, "prompt": admitted_prompt, "stored_input": None} if isinstance(admitted_prompt, str)
                       else {**row, "prompt": "", "stored_input": encode_domain(stored)})
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
            if row["result_json"].get("graph_trace_suffix") is True:
                if position not in prompts:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR, "Raw Graph transcript lacks its admitted input")
                first = messages[0]
                if not isinstance(first, ModelRequest):
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR, "Raw Graph transcript has no request")
                parts = [part for part in first.parts if isinstance(part, UserPromptPart)]
                raw_prompt = parts[0].content if len(parts) == 1 else None
                if isinstance(raw_prompt, tuple):
                    raw_prompt = list(raw_prompt)
                matches = admitted_prompt is not None and raw_prompt == admitted_prompt
                if (not matches and len(parts) == 1 and admitted_prompt is not None
                        and row["result_json"].get("graph_step_source") is True):
                    # The native Step writer can retain the original terminal
                    # snapshot; the context recorder can instead retire consumed
                    # media. Prove either representation against the same input.
                    supplied = replace(first, parts=[
                        replace(part, content=admitted_prompt) if part is parts[0] else part
                        for part in first.parts
                    ])
                    projected_first = project_transient_binary_content((supplied, messages[-1]))[0]
                    expected_parts = [part for part in projected_first.parts if isinstance(part, UserPromptPart)]
                    if len(expected_parts) == 1:
                        expected_prompt = expected_parts[0].content
                        if isinstance(expected_prompt, tuple):
                            expected_prompt = list(expected_prompt)
                        matches = raw_prompt == expected_prompt
                if not matches:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR, "Raw Graph transcript differs from its admitted input")
                messages[0] = replace(first, parts=[
                    replace(part, content="") if part is parts[0] else part for part in first.parts
                ])
            if not display and refs:
                spec = bindings[row["binding_key"]]["agent_spec"]
                self._restore_binding_refs(messages, refs, spec["instructions"], spec["system_prompt"])
            context = ()
            if latest_context is not None and latest_context[0] == position:
                context_root = latest_context[1]
                context_messages = [decode_model_messages(body) for body in bodies[context_root]]
                if len(context_messages) != context_root.count or any(len(part) != 1 for part in context_messages):
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR, "Invalid Session compaction messages")
                context = tuple(singleton[0] for singleton in context_messages)
                if (not session_compaction_summaries(context) or context[-1] != messages[-1]
                        or binary_content_usage(context)[0]):
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR, "Session compaction does not match its turn")
            projected.append({**row, "messages": tuple(messages), "compacted_context": context})
        try:
            next(loaded, None)
        except ValueError as error:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
        return tuple(projected)

    @staticmethod
    def _transcript_manifest(result) -> tuple[NativeMessageRoot, tuple[tuple[int, int | None], ...]]:
        root = decode_session_transcript_root(result)
        return root, tuple((position, part) for position, part in result["binding_refs"])

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
        return {**self._session_values(key.session_id), "turn_id": self._key(key.turn_id, NATIVE_TURN_ID_MAX_BYTES)}

    async def get_session(self, session_id: str) -> RowMapping | None:
        async with self._context.engine.connect() as connection:
            return (await connection.execute(
                self._sessions.select().filter_by(**self._session_values(session_id)),
            )).mappings().one_or_none()

    async def get_session_for_admission(self, session_id: str, request_key: str) -> RowMapping | None:
        """Read the admission head and exact request existence without loading its body."""
        from sqlalchemy import select

        sessions, turns = self._sessions.c, self._turns.c
        exists = select(turns.turn_id).where(
            turns.scope_key == sessions.scope_key,
            turns.session_key == sessions.session_key,
            turns.request_key == self._key(request_key, SESSION_REQUEST_KEY_MAX_BYTES),
        ).exists()
        statement = self._sessions.select().add_columns(exists.label("has_request")).filter_by(
            **self._session_values(session_id),
        )
        async with self._context.engine.connect() as connection:
            return (await connection.execute(statement)).mappings().one_or_none()

    async def get_turn(self, key: SessionTurnKey) -> RowMapping | None:
        async with self._context.engine.connect() as connection:
            return (await connection.execute(self._turns.select().filter_by(**self._turn_values(key)))).mappings().one_or_none()

    async def _linked_graph_turn(
        self, execution_id: str, *, transaction: StateTransaction | None = None,
    ) -> tuple[SessionTurnKey, int] | None:
        from sqlalchemy import select

        if not 1 <= len(execution_id.encode("utf-8")) <= NATIVE_TURN_ID_MAX_BYTES:
            return None
        rows = self._turns.c
        statement = select(
            rows.session_key, rows.generation, rows.request_json["graph_execution"].as_string(),
        ).where(rows.scope_key == self._scope, rows.turn_id == self._key(execution_id, NATIVE_TURN_ID_MAX_BYTES))

        async def read(connection):
            return (await connection.execute(statement)).one_or_none()

        if transaction is None:
            async with self._context.engine.connect() as connection:
                linked = await read(connection)
        else:
            linked = await transaction.run_sql(self._context, read)
        if linked is None or linked[2] is None:
            return None
        session_key, generation, graph_execution = linked
        if graph_execution != execution_id:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return SessionTurnKey(session_key.decode("utf-8"), execution_id), generation

    async def read_linked_graph_context(
        self, execution_id: str, *, admitted: Principal, principal: Principal,
        authorization: AuthorizationPolicy,
    ) -> tuple[tuple[SessionContextTurn, ...], Mapping[str, JsonValue], Mapping[str, JsonValue] | None] | None:
        """Read the original Graph admission range, not the session's current context."""
        from sqlalchemy import select

        if principal.tenant_id != self._tenant_id or admitted.tenant_id != self._tenant_id:
            raise AIError(ErrorCode.AUTHORIZATION_DENIED)
        rows, sessions = self._turns.c, self._sessions.c
        statement = select(
            rows.session_key, rows.turn_no, rows.context_from, rows.context_through, rows.request_json,
            sessions.lineage_json, sessions.session_key.label("owner_key"),
        ).select_from(self._turns.outerjoin(self._sessions, (
            (rows.scope_key == sessions.scope_key) & (rows.session_key == sessions.session_key)
        ))).where(
            rows.scope_key == self._scope, rows.turn_id == self._key(execution_id, NATIVE_TURN_ID_MAX_BYTES),
            rows.request_json["graph_execution"].as_string().is_not(None),
        )
        async with self._context.engine.connect() as connection:
            row = (await connection.execute(statement)).mappings().one_or_none()
        if row is None:
            return None
        if type(authorization) is TenantAuthorizationPolicy:
            raise AIError(ErrorCode.AUTHORIZATION_DENIED)
        session_id = row["session_key"].decode("utf-8")
        await authorization.authorize(
            principal, AuthorizationAction.SESSION_READ,
            ResourceRef(ResourceKind.SESSION, session_id, self._tenant_id),
        )
        request = row["request_json"]
        if request.get("graph_execution") != execution_id or request.get("principal") != asdict(admitted):
            raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)
        if row["owner_key"] is None:
            raise AIError(ErrorCode.INPUT_CONTEXT_UNAVAILABLE)
        if row["context_through"] != row["turn_no"] - 1:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        lineage = decode_session_lineage(row["lineage_json"])
        for segment in lineage:
            await authorization.authorize(
                principal, AuthorizationAction.SESSION_READ,
                ResourceRef(ResourceKind.SESSION, segment.session_id, self._tenant_id),
            )
        context = await self.read_context_range(
            session_id, after=row["context_from"] - 1, through=row["context_through"], lineage=lineage,
            missing=ErrorCode.INPUT_CONTEXT_UNAVAILABLE,
        )
        return context, request["session_metadata"], request.get("repository_instructions")

    @staticmethod
    async def _read_binding_payloads(archive, keys: tuple[bytes, ...]) -> dict[bytes, dict]:
        if not keys:
            return {}
        raws = await archive.read_contents(
            tuple("execution-binding:" + key.hex() for key in keys), codec="execution-binding",
        )
        return {
            key: decode_agent_binding_content(raw).to_payload()
            for key, raw in zip(keys, raws, strict=True)
        }

    async def _read_bound_turn(self, statement) -> Mapping[str, object] | None:
        if self._transcript_source is None:
            raise AIError(ErrorCode.STORAGE_DEPENDENCY_NOT_READY)
        turn_store, archive_provider, _ = self._transcript_source
        group = turn_store.storage_group
        opened_archive = group.opened_archive or await group.open_archive(create=False)
        archive = None if opened_archive is None else await archive_provider()
        members = (turn_store,) if archive is None else tuple(dict.fromkeys((turn_store, archive.store)))

        async def read(snapshot):
            owner = snapshot.transaction(turn_store)
            if not isinstance(owner, SqlStateRowReader):
                raise AIError(ErrorCode.STORAGE_DEPENDENCY_NOT_READY)
            rows = await owner.read_sql_rows(self._context, statement)
            if not rows:
                return None
            if archive is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            row, = rows
            key = row["binding_key"]
            return {**row, "binding_json": (await self._read_binding_payloads(archive, (key,)))[key]}

        return await group.read_many(members, read)

    async def get_turn_for_recovery(self, key: SessionTurnKey) -> Mapping[str, object] | None:
        row = await self._read_bound_turn(self._turns.select().filter_by(**self._turn_values(key)))
        if row is None:
            return None
        # Recovery consumes discovery evidence only after the generation regrant.
        request = {name: value for name, value in row["request_json"].items() if name != "instruction_frontier"}
        return {**row, "request_json": request}

    def _history_view_select(self, *, observe_lease: bool = False, include_authority: bool = False):
        from sqlalchemy import select

        columns = self._turns.c
        statement = select(
            columns.turn_id, columns.turn_no, columns.state, columns.created_at, columns.updated_at,
            columns.session_key, columns.generation, columns.split_source_receipt,
            columns.result_json, columns.request_json["prompt"].as_string().label("prompt"),
            columns.request_json["input"].label("stored_input"),
            columns.request_json["graph_execution"].as_string().label("graph_execution"),
        )
        if observe_lease:
            statement = statement.add_columns(
                (columns.state.in_(("accepted", "cancelling")) & (columns.lease_until <= sql_epoch_seconds()))
                .label("needs_recovery"),
                columns.request_json["split_step"].as_boolean().label("split_step"),
            )
        if include_authority:
            statement = statement.add_columns(
                columns.request_json["principal"]["principal_id"].as_string().label("owner_principal_id"),
            )
        return statement

    async def get_history_turn(
        self, key: SessionTurnKey, *, lineage: tuple[SessionTurnSegment, ...], include_authority: bool = False,
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
        statement = self._history_view_select(observe_lease=True, include_authority=include_authority).where(
            columns.scope_key == self._scope,
            columns.turn_id == self._key(key.turn_id, NATIVE_TURN_ID_MAX_BYTES),
            scope,
        )
        rows = await self._read_transcripts(statement, display=True)
        if not rows:
            return None
        row = rows[0]
        result = row["result_json"]
        if row["state"] != "succeeded" or "usage" not in result:
            return row
        return {**row, "usage": decode_session_turn_usage(result["usage"])}

    async def read_admission_input(self, session_id: str, request_key: str) -> RowMapping | None:
        """Read the exact input descriptor without opening binding or transcript bodies."""
        from sqlalchemy import select

        columns = self._turns.c
        request = columns.request_json
        statement = select(
            columns.turn_id, request["prompt"].as_string().label("prompt"),
            request["input"].label("input"),
            request["graph_execution"].as_string().label("graph_execution"),
            request["principal"]["principal_id"].as_string().label("principal_id"),
        ).where(
            columns.scope_key == self._scope,
            columns.session_key == self._key(session_id, TURN_SESSION_MAX_BYTES),
            columns.request_key == self._key(request_key, SESSION_REQUEST_KEY_MAX_BYTES),
        )
        async with self._context.engine.connect() as connection:
            return (await connection.execute(statement)).mappings().one_or_none()

    async def read_native_admission_input(self, row: RowMapping, turn_id: str) -> StoredUserInput | None:
        """Read only the immutable input message owned by a Graph admission."""
        reference = decode_session_input_reference(row, turn_id)
        if reference is None:
            return None
        if self._transcript_source is None:
            raise AIError(ErrorCode.STORAGE_DEPENDENCY_NOT_READY)
        store, archive_provider, capture = self._transcript_source
        group = store.storage_group
        archive = await archive_provider(create=False)
        if archive.store.storage_group is not group:
            raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)

        async def read(transaction):
            return await capture((reference.root,), transaction=transaction.transaction(archive.store))

        load = await group.read_many((archive.store,), read)
        (parts,) = await load()
        if len(parts) != 1:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return restore_native_user_input(reference, parts[0])

    async def find_request(self, session_id: str, request_key: str) -> Mapping[str, object] | None:
        from sqlalchemy import select

        columns = self._turns.c
        request = columns.request_json
        required = ("prompt", "principal", "thinking")
        optional = ("input", "input_intent", "budgeted", "session_context", "correlation", "mode", "planning")
        statement = select(
            columns.turn_id, columns.binding_key, request["explicit_agent_binding"].as_boolean().label("explicit_agent_binding"),
            *(request[name].label(name) for name in (*required, *optional)),
        )
        row = await self._read_bound_turn(statement.where(
            columns.scope_key == self._scope,
            columns.session_key == self._key(session_id, TURN_SESSION_MAX_BYTES),
            columns.request_key == self._key(request_key, SESSION_REQUEST_KEY_MAX_BYTES),
        ))
        if row is None:
            return None
        binding = row["binding_json"]
        spec = binding["agent_spec"]
        return {
            "turn_id": row["turn_id"], "request_json": {
                name: row[name] for name in (*required, *optional)
                if name in required or row[name] is not None
            },
            "agent_id": spec["id"], "output_mode": binding["output_mode"], "output_schema": binding["output_schema"],
            "default_output_schema": spec.get("output_schema"),
            "default_planning": spec["planning"], "default_thinking": spec["thinking"],
            **({"binding_digest": AgentBindingContract.from_payload(binding).binding_digest}
               if row["explicit_agent_binding"] is True else {}),
        }

    async def session_page(
        self, *, session_ids: Sequence[str] | None, before: int | None, limit: int, prefix: str | None = None,
    ) -> tuple[RowMapping, ...]:
        columns = self._sessions.c
        statement = self._sessions.select().where(columns.scope_key == self._scope)
        if session_ids is not None:
            if not session_ids:
                return ()
            keys = tuple(self._key(identity, TURN_SESSION_MAX_BYTES) for identity in session_ids)
            statement = statement.where(columns.session_key.in_(keys))
        if prefix is not None:
            lower = self._key(prefix, TURN_SESSION_MAX_BYTES)
            # Valid UTF-8 never ends in 0xff. Compare indexed binary keys without
            # casting them, so a tenant's unrelated sessions stay out of range.
            upper = lower[:-1] + bytes((lower[-1] + 1,))
            statement = statement.where(columns.session_key >= lower, columns.session_key < upper)
        if before is not None:
            statement = statement.where(columns.id < before)
        async with self._context.engine.connect() as connection:
            return tuple((await connection.execute(statement.order_by(columns.id.desc()).limit(limit))).mappings())

    async def turn_page(
        self, session_id: str, *, through: int, limit: int,
        lineage: tuple[SessionTurnSegment, ...] = (),
        body_limit: int | None = None, include_authority: bool = False,
    ) -> tuple[Mapping[str, object], ...]:
        columns = self._turns.c
        statement = self._history_view_select(include_authority=include_authority).where(
            columns.scope_key == self._scope,
            lineage_predicate(columns, self._key(session_id, TURN_SESSION_MAX_BYTES), lineage,
                              after=0, through=through),
            columns.state.in_(("succeeded", "failed", "cancelled")),
        ).order_by(columns.turn_no.desc()).limit(limit)
        return await self._read_transcripts(
            statement, display=True, body_limit=body_limit,
            expected_turns=range(through, max(0, through - limit), -1),
        )

    async def recent_context(
        self, session_id: str, *, through: int, max_context_bytes: int,
        lineage: tuple[SessionTurnSegment, ...] = (), checkpoint: SqlStepStore | None = None,
    ) -> tuple[SessionContextTurn, ...]:
        """Read bounded candidates after the caller authorizes the current session head."""
        start = max(0, through - _CONTEXT_POSITIONS)
        cached = self._recent.get(session_id)
        if (cached is not None and cached.max_context_bytes == max_context_bytes
                and cached.after <= start <= cached.through <= through):
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

        turns += await self.read_context_range(
            session_id, after=after, through=through, lineage=lineage,
            checkpoint=checkpoint if cached is None and not lineage else None,
            compacted_context=not lineage, _trim_reset=True,
        )
        for index in range(len(turns) - 1, -1, -1):
            if turns[index].reset_context:
                turns = turns[index:]
                break
        self._remember_context(
            session_id, after=start, through=through, max_context_bytes=max_context_bytes,
            turns=tuple(
                turn if turn.seed_checkpoint_prefix is None else replace(turn, seed_checkpoint_prefix=None)
                for turn in turns
            ),
        )
        return turns

    async def read_context_range(
        self, session_id: str, *, after: int, through: int, lineage: tuple[SessionTurnSegment, ...] = (),
        missing: ErrorCode = ErrorCode.STORAGE_INTEGRITY_ERROR, checkpoint: SqlStepStore | None = None,
        compacted_context: bool | str = False, _trim_reset: bool = False,
    ) -> tuple[SessionContextTurn, ...]:
        """Read finalized input and only the confirmed conversation transcripts."""
        from sqlalchemy import case, select, type_coerce

        if not 0 <= through - after <= _CONTEXT_POSITIONS or after < 0:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR, "The admitted context range is invalid")
        if after == through:
            return ()

        columns = self._turns.c
        succeeded = columns.state == "succeeded"
        prompt = columns.request_json["prompt"].as_string().label("prompt")
        result = type_coerce(case((succeeded, columns.result_json)), columns.result_json.type).label("result_json")
        stored_input = columns.request_json["input"].label("stored_input")
        reset = (columns.request_json["session_context"].as_string() == "current_turn").label("reset_context")
        statement = select(
            columns.turn_no, columns.state, prompt, stored_input, result, columns.binding_key, reset,
            columns.turn_id, columns.session_key, columns.generation, columns.split_source_receipt,
            columns.request_json["graph_execution"].as_string().label("graph_execution"),
        ).where(
            columns.scope_key == self._scope,
            lineage_predicate(columns, self._key(session_id, TURN_SESSION_MAX_BYTES), lineage,
                              after=after, through=through),
            columns.state.in_(("succeeded", "failed", "cancelled")),
        ).order_by(columns.turn_no).limit(_CONTEXT_POSITIONS)
        if checkpoint is not None:
            if checkpoint.storage_context.engine is not self._context.engine:
                raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)
            if checkpoint._can_project_checkpoint_prefix():
                statement = checkpoint._project_checkpoint_prefix(
                    statement, self._turns, session_id=session_id, through=through,
                )
            else:
                checkpoint = None
        rows = await self._read_transcripts(
            statement, display=False, compacted_context=compacted_context,
            reset_window=(after, through, missing) if _trim_reset else None,
        )
        if not _trim_reset and len(rows) != through - after:
            raise AIError(missing, "The finalized session context range is incomplete")
        return tuple(
            self._context_turn(
                row["turn_id"].decode("utf-8"), row["turn_no"], row["prompt"], row["stored_input"],
                row["messages"] if row["state"] == "succeeded" else (),
                reset_context=row["state"] == "succeeded" and row["reset_context"] is True,
                interrupted=row["state"] != "succeeded",
                seed_checkpoint_prefix=None if checkpoint is None else checkpoint._retained_checkpoint_seed(row),
                compacted_context=row.get("compacted_context", ()),
            )
            for row in rows
        )

    @staticmethod
    def _context_turn(
        turn_id: str, turn_no: int, prompt: str, stored_input: JsonValue, messages: Sequence[ModelMessage], *,
        reset_context: bool = False, interrupted: bool = False,
        seed_checkpoint_prefix: Callable[[str, Sequence[ModelMessage]], bool] | None = None,
        compacted_context: tuple[ModelMessage, ...] = (),
    ) -> SessionContextTurn:
        request = {"prompt": prompt, **({"input": stored_input} if stored_input is not None else {})}
        transcript_bytes = len(encode_model_messages(messages))
        size = max(1024, len(canonical_json_bytes(request)) + transcript_bytes)
        if compacted_context:
            size += len(encode_model_messages(compacted_context))
        return SessionContextTurn(turn_id, turn_no, request, tuple(messages), size, transcript_bytes, reset_context, interrupted,
                                  seed_checkpoint_prefix, compacted_context)

    def _remember_context(
        self, session_id: str, *, after: int, through: int, turns: tuple[SessionContextTurn, ...],
        max_context_bytes: int,
    ) -> None:
        global _history_cache_total
        remaining = max_context_bytes
        for index in range(len(turns) - 1, -1, -1):
            turn = turns[index]
            # This is a lower bound on the admitted charge: stored input
            # and media may consume more. Keep the first non-fitting turn,
            # since its compacted context can replace the normal transcript.
            remaining -= turn.transcript_bytes + len(turn.request["prompt"].encode("utf-8"))
            if remaining < 0:
                turns = turns[index:]
                break
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
            self._recent[session_id] = _ContextWindow(after, through, max_context_bytes, turns, size)
            owner.used += size
            _history_cache_total += size

    def clear_recent_context(self) -> None:
        """Drop retained history and disable caching for this closed Persistence."""
        with _HISTORY_CACHE_LOCK:
            self._recent.clear()
            self._recent_finalizer()


class _TerminalFenceLost(AIError):
    pass


@dataclass(frozen=True, slots=True)
class _SplitTargetObservation:
    generation: int
    state: str
    receipt_id: str | None
    admitted_at: datetime | None = None

    def confirms(self, *, generation: int, state: str, receipt_id: str) -> bool:
        return self.generation == generation and self.state == state and self.receipt_id == receipt_id


@dataclass(frozen=True, slots=True)
class _PreparedSessionAdmission:
    key: SessionTurnKey
    binding_key: bytes
    write: Callable[[AsyncSession], Awaitable[bool]]
    store: StateStore | None
    archive: tuple[ContentArchive, PreparedArchive] | None
    budget: tuple[BudgetRepository, RunBudget] | None

    @property
    def stores(self) -> tuple[StateStore, ...]:
        if self.store is None:
            raise AIError(ErrorCode.STORAGE_DEPENDENCY_NOT_READY)
        members = (self.store,)
        if self.archive is not None:
            members = (*members, self.archive[0].store)
        if self.budget is not None:
            members = (*members, self.budget[0].state_store)
        return tuple(dict.fromkeys(members))


@dataclass(frozen=True, slots=True)
class _PreparedSessionTerminal:
    write: Callable[[AsyncSession], Awaitable[bool]]
    store: StateStore | None
    transcript: tuple[ContentArchive, PreparedArchive, NativeMessageRoot] | None
    result: JsonValue
    publish_transcript: bool = True

    @property
    def stores(self) -> tuple[StateStore, ...]:
        if self.store is None:
            raise AIError(ErrorCode.STORAGE_DEPENDENCY_NOT_READY)
        if self.transcript is None:
            return (self.store,)
        return tuple(dict.fromkeys((self.store, self.transcript[0].store)))



MAX_REMEMBERED_SCOPES = 128
MAX_REMEMBERED_SCOPE_BYTES = 16 * 1024


def decode_instruction_scopes(config: Mapping[str, JsonValue]) -> tuple[str, ...]:
    raw = config.get("instruction_scopes", [])
    try:
        if (not isinstance(raw, list) or len(raw) > MAX_REMEMBERED_SCOPES
                or len(canonical_json_bytes(raw)) > MAX_REMEMBERED_SCOPE_BYTES
                or any(not isinstance(scope, str) or validate_workspace_path(scope) != scope for scope in raw)
                or raw != sorted(set(raw))):
            raise ValueError("invalid remembered scope list")
    except (TypeError, ValueError) as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
    return tuple(raw)


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
        messages: Sequence[ModelMessage], succeeded: bool, max_context_bytes: int,
        reset_context: bool = False, compacted_context: tuple[ModelMessage, ...] = (),
    ) -> None:
        """Advance coverage only over a known committed adjacent position."""
        cached = self._recent.get(key.session_id)
        after = max(0, turn_no - _CONTEXT_POSITIONS)
        if succeeded and reset_context:
            turns = ()
        elif cached is None:
            if turn_no != 1:
                return
            turns = ()
        elif (cached.max_context_bytes != max_context_bytes or cached.through != turn_no - 1
              or cached.after > after):
            return
        else:
            turns = tuple(turn for turn in cached.turns if turn.turn_no > after)
        if prompt is None:
            return
        turns += (self._context_turn(
            key.turn_id, turn_no, prompt, stored_input, messages if succeeded else (),
            reset_context=succeeded and reset_context, interrupted=not succeeded, compacted_context=compacted_context,
        ),)
        self._remember_context(
            key.session_id, after=after, through=turn_no, turns=turns, max_context_bytes=max_context_bytes,
        )

    async def create(self, session_id: str, *, config: JsonValue) -> None:
        async def write(connection: AsyncSession) -> None:
            await connection.execute(self._sessions.insert().values(
                **self._session_values(session_id), status="open", revision=0, last_turn_no=0, config_json=config,
            ))

        await self._context.run_mutation(write, domain="session.create")

    async def fork_session(
        self, source_id: str, target_id: str, *, expected_revision: int, authorized_revision: int, request_id: str,
    ) -> bool:
        """Freeze the authorized source head or replay the exact fork request."""
        validate_session_request_key(request_id)
        if source_id == target_id:
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        source = self._session_values(source_id)
        target = self._session_values(target_id)

        async def write(connection: AsyncSession) -> bool:
            existing = (await connection.execute(self._sessions.select().filter_by(**target))).mappings().one_or_none()
            if existing is not None:
                lineage = existing["lineage_json"]
                if (lineage is not None and lineage["source"] == source_id and lineage["request_id"] == request_id
                        and lineage["source_revision"] == expected_revision):
                    return True
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            if authorized_revision != expected_revision:
                return False
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
            source_config = row["config_json"]
            config = {"metadata": source_config.get("metadata", {}), "cwd": source_config.get("cwd")}
            await connection.execute(self._sessions.insert().values(
                **target, status="open", revision=0, last_turn_no=through, config_json=config,
                lineage_json={"source": source_id, "request_id": request_id, "source_revision": expected_revision,
                              "segments": [[part.session_id, part.after, part.through] for part in segments]},
            ))
            return True

        return await self._context.run_mutation(write, domain="session.fork")

    async def update_session(
        self, session_id: str, *, expected_revision: int, config: JsonValue, metadata_only: bool = False,
    ) -> bool:
        """Update at the authorized revision; only metadata may change during a parked turn."""
        from sqlalchemy import or_, select

        sessions, turns = self._sessions.c, self._turns.c
        inactive = sessions.active_turn_id.is_(None)
        allowed = inactive
        if metadata_only:
            parked = select(turns.id).where(
                turns.scope_key == sessions.scope_key, turns.session_key == sessions.session_key,
                turns.turn_id == sessions.active_turn_id, turns.state.in_(("waiting", "ready")),
            ).exists()
            allowed = or_(inactive, parked)

        async def write(connection: AsyncSession) -> bool:
            result = await connection.execute(self._sessions.update().filter_by(
                **self._session_values(session_id), status="open", revision=expected_revision,
            ).where(allowed).values(config_json=config, revision=expected_revision + 1))
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
        _budget: tuple[BudgetRepository, RunBudget] | None = None,
        _instruction_scopes: tuple[str, ...] | None = None,
    ) -> bool:
        """Claim and record a turn atomically; a uniqueness failure rolls back the claim."""
        prepared = await self._prepare_accept(
            key, expected_revision=expected_revision, context_from=context_from, context_through=context_through,
            request_key=request_key, request=request, binding=binding, lease_seconds=lease_seconds, budget=_budget,
            instruction_scopes=_instruction_scopes,
        )
        if prepared.archive is None and prepared.budget is None:
            accepted = await self._context.run_mutation(prepared.write, domain="session.accept")
        else:
            stores = prepared.stores
            accepted = await stores[0].storage_group.mutate(
                stores, lambda transaction: self._accept_in_group(transaction, prepared),
            )
        if accepted:
            self._note_accepted(prepared)
        return accepted

    async def _prepare_accept(
        self, key: SessionTurnKey, *, expected_revision: int, context_from: int, context_through: int,
        request_key: str, request: JsonValue, binding: JsonValue, lease_seconds: int,
        budget: tuple[BudgetRepository, RunBudget] | None,
        instruction_scopes: tuple[str, ...] | None = None,
    ) -> _PreparedSessionAdmission:
        if lease_seconds < 1:
            raise ValueError("the turn lease must be positive")
        if not 1 <= context_from <= context_through + 1:
            raise ValueError("invalid admitted session context range")
        session_key = self._session_values(key.session_id)
        turn_key = self._turn_values(key)
        encoded_request_key = self._key(request_key, SESSION_REQUEST_KEY_MAX_BYTES)
        content = ArchiveContentInput(
            "execution-binding", encode_agent_binding_content(AgentBindingContract.from_payload(binding)),
        )
        binding_key = bytes.fromhex(content.digest)
        archive_write = None
        binding_store = self._transcript_write_binding
        store = None if binding_store is None else binding_store[0]
        if budget is not None:
            if store is None:
                raise AIError(ErrorCode.STORAGE_DEPENDENCY_NOT_READY)
            if budget[0].state_store.storage_group is not store.storage_group:
                raise AIError(ErrorCode.STORAGE_ATOMIC_BATCH_UNSUPPORTED)
        if binding_key not in self._committed_bindings:
            if binding_store is None:
                raise AIError(ErrorCode.STORAGE_DEPENDENCY_NOT_READY)
            store, archive_provider, _ = binding_store
            archive = await archive_provider(create=True)
            if archive.store.storage_group is not store.storage_group:
                raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)
            prepared = PreparedArchive()
            await archive.prepare_agent_binding(content, prepared)
            archive_write = archive, prepared

        values = {"active_turn_id": turn_key["turn_id"], "revision": expected_revision + 1}
        if instruction_scopes is not None:
            values["config_json"] = self._context.dialect.set_json_member(
                self._sessions.c.config_json, "instruction_scopes", list(instruction_scopes),
            )

        async def write(connection: AsyncSession) -> bool:
            claim = await connection.execute(self._sessions.update().filter_by(
                **session_key, status="open", revision=expected_revision,
                last_turn_no=context_through, active_turn_id=None,
            ).values(**values))
            if claim.rowcount != 1:
                return False
            await connection.execute(self._turns.insert().values(
                **turn_key, request_key=encoded_request_key, binding_key=binding_key, turn_no=context_through + 1,
                context_from=context_from, context_through=context_through, state="accepted", generation=1,
                lease_until=self._now + lease_seconds, request_json=request,
            ))
            return True

        return _PreparedSessionAdmission(key, binding_key, write, store, archive_write, budget)

    async def _accept_in_group(
        self, transaction: StateGroupTransaction, prepared: _PreparedSessionAdmission,
    ) -> bool:
        """Publish the prepared admission through the already enlisted group."""
        store = prepared.store
        if store is None:
            raise AIError(ErrorCode.STORAGE_DEPENDENCY_NOT_READY)
        if not await transaction.transaction(store).run_sql(self._context, prepared.write):
            return False
        if prepared.archive is not None:
            archive, content = prepared.archive
            await archive.publish(transaction.transaction(archive.store), content)
        if prepared.budget is not None:
            repository, limits = prepared.budget
            await repository.ensure_in_transaction(
                transaction.transaction(repository.state_store), "turn:" + prepared.key.turn_id, limits,
            )
        return True

    def _note_accepted(self, prepared: _PreparedSessionAdmission) -> None:
        """Cache only the binding of a confirmed admission."""
        key = prepared.binding_key
        self._committed_bindings[key] = None
        self._committed_bindings.move_to_end(key)
        if len(self._committed_bindings) > 128:
            self._committed_bindings.popitem(last=False)

    async def finish(
        self, key: SessionTurnKey, *, generation: int, turn_no: int, state: TerminalState,
        result: JsonValue = None, transcript: "Sequence[ModelMessage] | None" = None,
        binding_refs: tuple[tuple[int, int | None], ...] = (),
        output_ref: tuple[int, int] | None = None,
        usage: UsageMetrics | None = None,
        compacted_context: Sequence[ModelMessage] | None = None,
        _abandon_expired: bool = False,
        _guard_abandon: tuple[StateStore, Callable[[StateTransaction], Awaitable[None]]] | None = None,
    ) -> bool:
        """Finalize and release using the same lock order as acceptance."""
        if _abandon_expired and state != "failed":
            raise ValueError("an abandoned turn must be marked failed")
        if _guard_abandon is not None and self._context.engine.dialect.name != "sqlite":
            # SQLite either owns the writer snapshot after the CAS or cannot
            # upgrade a stale read transaction. A repeatable-read server
            # transaction can instead observe an older child set after a write.
            raise AIError(
                ErrorCode.STORAGE_RECOVERY_REQUIRED,
                safe_details={"reason": "abandon_child_snapshot_unsupported"},
            )
        prepared = await self._prepare_finish(
            key, generation=generation, turn_no=turn_no, state=state, result=result,
            transcript=transcript, binding_refs=binding_refs, output_ref=output_ref, usage=usage,
            compacted_context=compacted_context,
            _expired=_abandon_expired,
        )
        try:
            if prepared.transcript is None and _guard_abandon is None:
                return await self._context.run_mutation(prepared.write, domain="session.finish")

            guard_stores = () if _guard_abandon is None else (_guard_abandon[0],)
            stores = prepared.stores
            group = stores[0].storage_group
            if any(member.storage_group is not group for member in guard_stores):
                raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)

            async def publish(transaction: StateGroupTransaction) -> bool:
                if not await self._finish_in_group(transaction, prepared):
                    return False
                if _guard_abandon is not None:
                    member, guard = _guard_abandon
                    await guard(transaction.transaction(member))
                return True

            return await group.mutate(tuple(dict.fromkeys((*stores, *guard_stores))), publish)
        except _TerminalFenceLost:
            return False

    async def _prepare_finish(
        self, key: SessionTurnKey, *, generation: int, turn_no: int, state: TerminalState,
        result: JsonValue, transcript: Sequence[ModelMessage] | None,
        binding_refs: tuple[tuple[int, int | None], ...], output_ref: tuple[int, int] | None,
        usage: UsageMetrics | None = None,
        compacted_context: Sequence[ModelMessage] | None = None,
        _graph_trace_suffix: tuple[ContentArchive, PreparedArchive, NativeMessageRoot] | None = None,
        _graph_step_source: bool = False,
        _expired: bool = False,
    ) -> _PreparedSessionTerminal:
        """Prepare the caller's normalized suffix, without starting another mutation."""
        if state not in ("succeeded", "failed", "cancelled"):
            raise ValueError("invalid terminal state")
        turn_key = self._turn_values(key)
        binding = self._transcript_write_binding
        store = None if binding is None else binding[0]
        body = None
        context_root = None
        if state == "succeeded":
            if result is not None or bool(transcript) == (_graph_trace_suffix is not None):
                raise ValueError("a successful session turn requires one transcript source")
            if binding is None:
                raise AIError(ErrorCode.STORAGE_DEPENDENCY_NOT_READY)
            store, archive_provider, native = binding
            archive = await archive_provider(create=True)
            if archive.store.storage_group is not store.storage_group:
                raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)
            if _graph_trace_suffix is None:
                content = PreparedArchive()
                root = await native.prepare_archived_messages(transcript, content)
            else:
                source, content, root = _graph_trace_suffix
                if (source.store is not archive.store or source.content_scope != archive.content_scope
                        or binding_refs or root.count < 1):
                    raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)
                native._message_content.validate_root(root)
            body = (archive, content, root)
            result = {"transcript": root.to_json(), "binding_refs": [list(ref) for ref in binding_refs]}
            if compacted_context is not None:
                if _graph_trace_suffix is not None or not compacted_context:
                    raise ValueError("compaction requires an ordinary local terminal")
                context_root = await native.prepare_archived_messages(compacted_context, content)
                result["compacted_context"] = context_root.to_json()
            if _graph_trace_suffix is not None:
                result["graph_trace_suffix"] = True
                if _graph_step_source:
                    result["graph_step_source"] = True
            if output_ref is not None:
                result["output_ref"] = list(output_ref)
            if usage is not None:
                result["usage"] = asdict(usage)
        elif transcript is not None or binding_refs or output_ref is not None or usage is not None or _graph_trace_suffix is not None or compacted_context is not None:
            raise ValueError("an unsuccessful session turn cannot publish a transcript")

        async def write(connection: AsyncSession) -> bool:
            if not await self._finish_turn(
                connection, key, generation=generation, turn_no=turn_no, state=state, result=result, _expired=_expired,
            ):
                return False
            if body is not None:
                from ._native_content_index import stage_owner

                roots = body[2]
                if context_root is not None:
                    from .contracts import LoadedModelContext
                    roots = (LoadedModelContext((), archive_root=body[2]),
                             LoadedModelContext((), archive_root=context_root))
                stage_owner(
                    connection, SESSION_TURN_TABLE, self._scope,
                    (turn_key["session_key"], turn_key["turn_id"]), roots,
                )
            return True

        return _PreparedSessionTerminal(write, store, body, result, publish_transcript=_graph_trace_suffix is None)

    async def _finish_in_group(
        self, transaction: StateGroupTransaction, prepared: _PreparedSessionTerminal,
    ) -> bool:
        """Publish the exact terminal generation in the already enlisted group."""
        store = prepared.store
        if store is None:
            raise AIError(ErrorCode.STORAGE_DEPENDENCY_NOT_READY)
        if not await transaction.transaction(store).run_sql(self._context, prepared.write):
            return False
        if prepared.transcript is not None and prepared.publish_transcript:
            archive, content, _ = prepared.transcript
            await archive.publish(transaction.transaction(archive.store), content)
        return True

    async def _finish_turn(
        self, connection: AsyncSession, key: SessionTurnKey, *, generation: int, turn_no: int,
        state: TerminalState, result: JsonValue, source_receipt: bytes | None = None,
        _expired: bool = False,
    ) -> bool:
        turn_key = self._turn_values(key)
        source = {} if source_receipt is None else {"split_source_receipt": source_receipt}
        released = await connection.execute(self._sessions.update().filter_by(
            **self._session_values(key.session_id), active_turn_id=turn_key["turn_id"], last_turn_no=turn_no - 1,
        ).values(active_turn_id=None, last_turn_no=turn_no, revision=self._sessions.c.revision + 1))
        if released.rowcount != 1:
            return False
        lease = self._turns.c.lease_until
        deadline = lease <= self._now if _expired else lease > self._now
        finished = await connection.execute(self._turns.update().filter_by(
            **turn_key, generation=generation, turn_no=turn_no,
            state="cancelling" if state == "cancelled" else "accepted",
        ).where(deadline).values(
            state=state, lease_until=None, result_json=result, **source,
        ))
        if finished.rowcount != 1:
            raise _TerminalFenceLost(ErrorCode.STORAGE_CONFLICT)
        return True


    async def publish_split_terminal(
        self, key: SessionTurnKey, *, generation: int, turn_no: int, state: TerminalState,
        source: SplitStepSourceObservation, result: JsonValue = None,
        binding_refs: tuple[tuple[int, int | None], ...] = (),
        output_ref: tuple[int, int] | None = None, _expired: bool = False,
    ) -> bool:
        """Project an already sealed source receipt; no source writes or body copies here."""
        if state not in ("succeeded", "failed", "cancelled"):
            raise ValueError("invalid terminal state")
        if (source.turn_id != key.turn_id or source.session_id != key.session_id
                or source.generation != generation or source.pin.terminal_state != state
                or source.phase != "sealed"):
            raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)
        source_pin = source.pin
        receipt = source_pin.receipt_id.encode("utf-8")
        if state == "succeeded":
            root = source_pin.transcript
            if result is not None or root is None or not root.count:
                raise ValueError("successful split publication requires its pinned native transcript")
            if self._transcript_write_binding is None:
                raise AIError(ErrorCode.STORAGE_DEPENDENCY_NOT_READY)
            _store, _archive, native = self._transcript_write_binding
            if root.store_key != native.store_key:
                raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)
            result = {"transcript": root.to_json(), "binding_refs": [list(ref) for ref in binding_refs]}
            if output_ref is not None:
                result["output_ref"] = list(output_ref)
            self._transcript_manifest(result)
        elif source_pin.transcript is not None or binding_refs or output_ref is not None:
            raise ValueError("an unsuccessful turn cannot install a native transcript")

        async def write(connection: AsyncSession) -> bool:
            return await self._finish_turn(
                connection, key, generation=generation, turn_no=turn_no, state=state,
                result=result, source_receipt=receipt, _expired=_expired,
            )

        try:
            return await self._context.run_mutation(write, domain="session.finish")
        except _TerminalFenceLost:
            return False

    async def publish_split_discard(
        self, key: SessionTurnKey, *, turn_no: int, source: SplitStepSourceObservation,
        result: JsonValue = None, _expired: bool = False,
    ) -> bool:
        """Let a confirmed Stop win without rewriting the already sealed source."""
        if ((source.turn_id, source.session_id) != (key.turn_id, key.session_id)
                or source.pin.terminal_state not in ("succeeded", "failed") or source.phase != "sealed"):
            raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)

        async def write(connection: AsyncSession) -> bool:
            return await self._finish_turn(
                connection, key, generation=source.generation, turn_no=turn_no,
                state="cancelled", result=result, source_receipt=source.pin.receipt_id.encode("utf-8"),
                _expired=_expired,
            )

        try:
            return await self._context.run_mutation(write, domain="session.finish")
        except _TerminalFenceLost:
            return False

    async def observe_split_target(self, key: SessionTurnKey) -> _SplitTargetObservation | None:
        """Reconcile the exact scoped target after an uncertain Conversation commit."""
        from sqlalchemy import select

        columns = self._turns.c
        statement = select(columns.generation, columns.state, columns.split_source_receipt).where(
            *(columns[name] == value for name, value in self._turn_values(key).items()),
        )
        async with self._context.engine.connect() as connection:
            row = (await connection.execute(statement)).one_or_none()
        if row is None:
            return None
        generation, state, receipt = row
        return _SplitTargetObservation(generation, state, None if receipt is None else receipt.decode("utf-8"))

    async def get_waiting_turn(self, key: SessionTurnKey) -> CanonicalDeferredFrontier | None:
        from sqlalchemy import and_, select

        turns, sessions = self._turns.c, self._sessions.c
        statement = select(turns.generation, turns.result_json).select_from(self._turns.join(
            self._sessions, and_(
                sessions.scope_key == turns.scope_key, sessions.session_key == turns.session_key,
                sessions.active_turn_id == turns.turn_id,
            ),
        )).where(turns.state.in_(("waiting", "ready")), *(
            turns[name] == value for name, value in self._turn_values(key).items()
        ))
        async with self._context.engine.connect() as connection:
            row = (await connection.execute(statement)).mappings().one_or_none()
        return None if row is None else CanonicalDeferredFrontier.from_json(key, row["generation"], row["result_json"])

    async def publish_waiting_turn(
        self, key: SessionTurnKey, *, generation: int, checkpoint_id: str,
        continuation: PendingToolContinuation, receipt_store: StateStore,
        publish_receipts: Callable[[], Awaitable[None]], max_bytes: int,
    ) -> bool:
        frontier = CanonicalDeferredFrontier.from_json(key, generation, {
            "checkpoint_id": checkpoint_id,
            "approval_operation_ids": [item.operation_id for item in continuation.approvals],
            "external_operation_ids": [item.operation_id for item in continuation.calls],
            "operation_metadata": {
                item.operation_id: dict(item.metadata) for item in (*continuation.approvals, *continuation.calls)
            },
        })
        frontier.require_continuation(continuation)
        payload = frontier.to_json()
        if len(canonical_json_bytes(payload)) > max_bytes:
            raise AIError(ErrorCode.CONTEXT_PAYLOAD_TOO_LARGE)
        if self._transcript_write_binding is None:
            raise AIError(ErrorCode.STORAGE_DEPENDENCY_NOT_READY)
        turn_store = self._transcript_write_binding[0]
        group = turn_store.storage_group
        if receipt_store.storage_group is not group:
            raise AIError(ErrorCode.STORAGE_ATOMIC_BATCH_UNSUPPORTED)

        async def mutate(group_transaction) -> bool:
            async def wait(connection: AsyncSession) -> bool:
                changed = await connection.execute(self._turns.update().filter_by(
                    **self._turn_values(key), generation=generation, state="accepted",
                ).where(self._turns.c.lease_until > self._now).values(
                    state="waiting", lease_until=None, result_json=payload,
                ))
                return changed.rowcount == 1

            if not await group_transaction.transaction(turn_store).run_sql(self._context, wait):
                return False
            await publish_receipts()
            return True

        members = tuple(dict.fromkeys((turn_store, receipt_store)))
        return await group.mutate(members, mutate)

    async def publish_split_waiting(
        self, frontier: CanonicalDeferredFrontier, *, source: SplitStepSourceObservation,
        max_bytes: int, _expired: bool = False,
    ) -> bool:
        """Publish a committed one-leaf source receipt without writing the source again."""
        from sqlalchemy import select

        receipt = self._require_split_deferred_source(frontier, source, phase="waiting")
        key = frontier.key
        payload = frontier.to_json()
        if len(canonical_json_bytes(payload)) > max_bytes:
            raise AIError(ErrorCode.CONTEXT_PAYLOAD_TOO_LARGE)
        turn = self._turn_values(key)
        occupied = select(self._sessions.c.id).filter_by(
            **self._session_values(key.session_id), active_turn_id=turn["turn_id"],
        ).exists()
        lease = self._turns.c.lease_until

        async def write(connection: AsyncSession) -> bool:
            changed = await connection.execute(self._turns.update().filter_by(
                **turn, generation=frontier.generation, state="accepted",
            ).where(occupied, lease <= self._now if _expired else lease > self._now).values(
                state="waiting", lease_until=None, result_json=payload,
                split_source_receipt=receipt,
            ))
            return changed.rowcount == 1

        return await self._context.run_mutation(write, domain="session.wait")

    @staticmethod
    def _require_split_deferred_source(
        frontier: CanonicalDeferredFrontier, source: SplitStepSourceObservation, *, phase: str,
    ) -> bytes:
        key, pin = frontier.key, source.pin
        waiting = pin.waiting
        if (source.phase != phase or waiting is None
                or (source.turn_id, source.session_id, source.generation) != (
                    key.turn_id, key.session_id, frontier.generation,
                )
                or pin.checkpoint_id != frontier.checkpoint_id
                or waiting.original_generation != frontier.generation
                or pin.receipt_id != frontier.split_receipt_id
                or frontier.approval_operation_ids != ((waiting.operation_id,) if waiting.kind == "approval" else ())
                or frontier.external_operation_ids != ((waiting.operation_id,) if waiting.kind == "external" else ())):
            raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)
        return pin.receipt_id.encode("utf-8")

    async def publish_split_ready(
        self, frontier: CanonicalDeferredFrontier, *, source: SplitStepSourceObservation,
    ) -> bool:
        """Publish only the same admitted waiting frontier now settled at its source."""
        from sqlalchemy import select

        receipt = self._require_split_deferred_source(frontier, source, phase="ready")
        owner = self._turn_values(frontier.key)
        occupied = select(self._sessions.c.id).filter_by(
            **self._session_values(frontier.key.session_id), active_turn_id=owner["turn_id"],
        ).exists()
        columns = self._turns.c
        proved = self._split_deferred_projection(frontier, receipt=receipt, state="ready")

        async def matches(connection: AsyncSession) -> bool:
            row = (await connection.execute(proved)).one_or_none()
            return row is not None and CanonicalDeferredFrontier.from_json(
                frontier.key, frontier.generation, row[0],
            ) == frontier

        async def write(connection: AsyncSession) -> bool:
            changed = await connection.execute(self._turns.update().filter_by(
                **owner, generation=frontier.generation, state="waiting", split_source_receipt=receipt,
            ).where(occupied, columns.lease_until.is_(None)).values(state="ready"))
            if changed.rowcount == 1:
                # Compare the typed frontier while the same CAS owns the row.
                # A mismatching frontier rolls back READY rather than replacing it.
                if not await matches(connection):
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                return True
            return False

        try:
            changed = await self._context.run_mutation(write, domain="session.ready")
        except AIError as error:
            if error.code is not ErrorCode.STORAGE_COMMIT_UNKNOWN:
                raise
            async with self._context.engine.connect() as connection:
                if await matches(connection):
                    return True
            raise
        if changed:
            return True
        async with self._context.engine.connect() as connection:
            return await matches(connection)

    async def confirms_split_waiting(self, frontier: CanonicalDeferredFrontier, *, receipt_id: str) -> bool:
        """Read back only the expected exact waiting publication after an unknown commit."""
        statement = self._split_deferred_projection(
            frontier, receipt=receipt_id.encode("utf-8"), state="waiting",
        )
        async with self._context.engine.connect() as connection:
            row = (await connection.execute(statement)).one_or_none()
        return row is not None and CanonicalDeferredFrontier.from_json(
            frontier.key, frontier.generation, row[0],
        ) == frontier

    def _split_deferred_projection(self, frontier: CanonicalDeferredFrontier, *, receipt: bytes, state: str):
        from sqlalchemy import and_, select

        turns, sessions = self._turns.c, self._sessions.c
        return select(turns.result_json).select_from(self._turns.join(
            self._sessions, and_(
                sessions.scope_key == turns.scope_key, sessions.session_key == turns.session_key,
                sessions.active_turn_id == turns.turn_id,
            ),
        )).where(
            *(turns[name] == value for name, value in self._turn_values(frontier.key).items()),
            turns.generation == frontier.generation, turns.state == state,
            turns.split_source_receipt == receipt,
        )

    async def settle_waiting_turn(
        self, frontier: CanonicalDeferredFrontier, receipt_store: StateStore,
        settle: Callable[[], Awaitable[_Receipt]], *,
        is_ready: Callable[[], Awaitable[bool]] | None = None,
    ) -> _Receipt:
        """Serialize a decision/result with withdrawal and same-turn continuation."""
        if self._transcript_write_binding is None:
            raise AIError(ErrorCode.STORAGE_DEPENDENCY_NOT_READY)
        turn_store = self._transcript_write_binding[0]
        group = turn_store.storage_group
        if receipt_store.storage_group is not group:
            raise AIError(ErrorCode.STORAGE_ATOMIC_BATCH_UNSUPPORTED)

        async def mutate(group_transaction) -> _Receipt:
            async def guard(connection: AsyncSession) -> bool:
                columns = self._turns.c
                held = await connection.execute(self._turns.update().filter_by(
                    **self._turn_values(frontier.key), generation=frontier.generation,
                ).where(columns.state.in_(("waiting", "ready"))).values(generation=columns.generation))
                return held.rowcount == 1

            owner = group_transaction.transaction(turn_store)
            if not await owner.run_sql(self._context, guard):
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            result = await settle()
            if is_ready is not None and await is_ready():
                async def mark_ready(connection: AsyncSession) -> None:
                    await connection.execute(self._turns.update().filter_by(
                        **self._turn_values(frontier.key), generation=frontier.generation, state="waiting",
                    ).values(state="ready"))

                await owner.run_sql(self._context, mark_ready)
            return result

        members = (turn_store,) if receipt_store is turn_store else (turn_store, receipt_store)
        return await group.mutate(members, mutate)

    async def guard_step_in_transaction(
        self, transaction, key: SessionTurnKey, *, generation: int,
        principal: Principal | None = None, _ordinary_only: bool = False,
    ) -> None:
        """Fence Step publication in its existing same-group transaction.

        Like terminal publication, this uses the database statement clock.
        The exact generation and phase are locked by the guarded write until
        commit. This does not grant permission to start an external effect.
        """
        from .state import SqlStateTransaction, active_state_scope

        if self._transcript_write_binding is None:
            raise AIError(ErrorCode.STORAGE_DEPENDENCY_NOT_READY)
        store, _, _ = self._transcript_write_binding
        scope = active_state_scope()
        if (scope is None or scope.group is not store.storage_group or not scope.writable
                or not isinstance(transaction, SqlStateTransaction)):
            raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)

        async def guard(session: AsyncSession) -> None:
            statement = self._turns.update().filter_by(
                **self._turn_values(key), generation=generation, state="accepted",
            ).where(self._turns.c.lease_until > self._now)
            if _ordinary_only:
                from sqlalchemy import func

                columns = self._turns.c
                statement = statement.where(
                    columns.split_source_receipt.is_(None),
                    columns.request_json["graph_execution"].as_string().is_(None),
                    func.coalesce(columns.request_json["split_step"].as_boolean(), False).is_(False),
                )
            if principal is not None:
                actor = self._turns.c.request_json["principal"]
                statement = statement.where(
                    actor["principal_id"].as_string() == principal.principal_id,
                    actor["kind"].as_string() == principal.kind,
                )
            changed = await session.execute(statement.values(generation=self._turns.c.generation))
            if changed.rowcount != 1:
                raise AIError(ErrorCode.STORAGE_CONFLICT)

        await transaction.run_sql(self._context, guard)

    @property
    def transcript_state_store(self) -> StateStore | None:
        return None if self._transcript_write_binding is None else self._transcript_write_binding[0]

    async def with_claimed_subagent_parent(
        self, key: SessionTurnKey, *, generation: int, tools: ToolRepositoryImpl,
        operation_id: str, owner: str, fence: int, invocation_id: str,
        child_store: StateStore, principal: Principal,
        write: Callable[[StateTransaction], Awaitable[_Receipt]],
    ) -> _Receipt:
        """Run one child transition under the exact current parent and delegated tool claim."""
        from ..core import ToolOperationStatus

        if self._transcript_write_binding is None:
            raise AIError(ErrorCode.STORAGE_DEPENDENCY_NOT_READY)
        store, _, _ = self._transcript_write_binding
        tool_store = tools.state_store
        group = store.storage_group
        if tool_store.storage_group is not group or child_store.storage_group is not group:
            raise AIError(ErrorCode.STORAGE_ATOMIC_BATCH_UNSUPPORTED)
        if principal.tenant_id != self._tenant_id:
            raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)

        async def guarded(group_transaction: StateGroupTransaction) -> _Receipt:
            await self.guard_step_in_transaction(
                group_transaction.transaction(store), key, generation=generation, principal=principal,
            )
            receipt = await tools.get_operation_in_transaction(
                group_transaction.transaction(tool_store), operation_id, tenant_id=self._tenant_id,
            )
            if (receipt is None or receipt.status is not ToolOperationStatus.CLAIMED
                    or (receipt.execution_id, receipt.agent_run_id, receipt.producer_generation,
                        receipt.tool_name, receipt.tool_call_id, receipt.owner, receipt.fence)
                    != (key.turn_id, key.turn_id, generation, "delegate_task", invocation_id, owner, fence)):
                raise AIError(ErrorCode.TOOL_OPERATION_CONFLICT)
            return await write(group_transaction.transaction(child_store))

        return await group.mutate(tuple(dict.fromkeys((store, tool_store, child_store))), guarded)

    async def append_instruction_frontier(
        self, key: SessionTurnKey, *, generation: int, frontier: SessionInstructionFrontier, max_bytes: int,
        instruction_scopes: tuple[str, ...] | None = None,
    ) -> None:
        """Store a discovery from the live serialized producer before exposing it."""
        payload = frontier.to_json()
        if len(canonical_json_bytes(payload)) > max_bytes:
            raise AIError(ErrorCode.CONTEXT_PAYLOAD_TOO_LARGE)

        async def write(connection: AsyncSession) -> None:
            if instruction_scopes is not None:
                # Same transaction and Session→Turn ordering as accept/finish.
                session = await connection.execute(self._sessions.update().filter_by(
                    **self._session_values(key.session_id), status="open",
                    active_turn_id=self._turn_values(key)["turn_id"],
                ).values(config_json=self._context.dialect.set_json_member(
                    self._sessions.c.config_json, "instruction_scopes", list(instruction_scopes),
                )))
                if session.rowcount != 1:
                    raise AIError(ErrorCode.STORAGE_CONFLICT)
            changed = await connection.execute(self._turns.update().filter_by(
                **self._turn_values(key), generation=generation, state="accepted",
            ).where(self._turns.c.lease_until > self._now).values(
                request_json=self._context.dialect.set_json_member(
                    self._turns.c.request_json, "instruction_frontier", payload,
                ),
            ))
            if changed.rowcount != 1:
                raise AIError(ErrorCode.STORAGE_CONFLICT)

        await self._context.run_mutation(write, domain="session.instructions")

    async def request_parked_cancel(
        self, frontier: CanonicalDeferredFrontier, *, receipt_store: StateStore,
        cancel_receipts: Callable[[StateGroupTransaction], Awaitable[bool]],
    ) -> Literal["lost", "requested", "ready"]:
        """Stop an exact same-group frontier before any approval can resume it."""
        binding = self._transcript_write_binding
        if binding is None:
            raise AIError(ErrorCode.STORAGE_DEPENDENCY_NOT_READY)
        turn_store = binding[0]
        group = turn_store.storage_group
        if receipt_store.storage_group is not group:
            raise AIError(ErrorCode.STORAGE_ATOMIC_BATCH_UNSUPPORTED)

        async def mutate(scope: StateGroupTransaction):
            async def request(connection: AsyncSession) -> bool:
                from sqlalchemy import and_, or_

                columns = self._turns.c
                matched = await connection.execute(self._turns.update().filter_by(
                    **self._turn_values(frontier.key), generation=frontier.generation,
                ).where(
                    columns.split_source_receipt.is_(None),
                    columns.result_json["checkpoint_id"].as_string() == frontier.checkpoint_id,
                    or_(
                        and_(columns.state.in_(("waiting", "ready")), columns.lease_until.is_(None)),
                        and_(columns.state == "cancelling", columns.lease_until <= self._now,
                             columns.result_json["cancel_effects_unresolved"].as_boolean().is_(True)),
                    ),
                ).values(state="cancelling", lease_until=self._now))
                return matched.rowcount == 1

            owner = scope.transaction(turn_store)
            if not await owner.run_sql(self._context, request):
                return "lost"
            resolved = await cancel_receipts(scope)

            async def persist_outcome(connection: AsyncSession) -> None:
                await connection.execute(self._turns.update().filter_by(
                    **self._turn_values(frontier.key), generation=frontier.generation, state="cancelling",
                ).values(result_json=None if resolved else {
                    **frontier.to_json(), "cancel_effects_unresolved": True,
                }))

            await owner.run_sql(self._context, persist_outcome)
            return "ready" if resolved else "requested"

        return await group.mutate(tuple(dict.fromkeys((turn_store, receipt_store))), mutate)

    async def request_split_wait_cancel(self, key: SessionTurnKey, *, generation: int) -> bool:
        """Record Stop for a parked split owner; there is no live lease to wait for."""
        columns = self._turns.c

        async def write(connection: AsyncSession) -> bool:
            changed = await connection.execute(self._turns.update().filter_by(
                **self._turn_values(key), generation=generation,
            ).where(
                columns.state.in_(("waiting", "ready")), columns.split_source_receipt.is_not(None),
                columns.lease_until.is_(None),
            ).values(state="cancelling", lease_until=self._now))
            return changed.rowcount == 1

        return await self._context.run_mutation(write, domain="session.cancel")

    async def request_cancel(self, key: SessionTurnKey) -> bool:
        async def write(connection: AsyncSession) -> bool:
            result = await connection.execute(self._turns.update().filter_by(
                **self._turn_values(key), state="accepted",
            ).values(state="cancelling"))
            return result.rowcount == 1

        return await self._context.run_mutation(write, domain="session.cancel")

    async def can_dispatch(self, key: SessionTurnKey, *, generation: int) -> bool:
        """Check the live generation without reading the turn body."""
        from sqlalchemy import select

        turns = self._turns
        statement = select(turns.c.id).filter_by(
            **self._turn_values(key), generation=generation, state="accepted",
        ).where(turns.c.lease_until > self._now)
        async with self._context.engine.connect() as connection:
            return (await connection.execute(statement)).scalar_one_or_none() is not None

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

    async def expire_stopped_lease(self, key: SessionTurnKey, *, generation: int) -> bool:
        """Relinquish the exact accepted generation after its producer has stopped."""
        async def write(connection: AsyncSession) -> bool:
            result = await connection.execute(self._turns.update().filter_by(
                **self._turn_values(key), generation=generation, state="accepted",
            ).where(self._turns.c.lease_until > self._now).values(lease_until=self._now))
            return result.rowcount == 1

        return await self._context.run_mutation(write, domain="session.expire_stopped_lease")

    async def _take_over(
        self, session: AsyncSession, key: SessionTurnKey, *, generation: int,
        lease_seconds: int, cancelling: bool, source_receipt: bytes | None = None,
    ) -> bool:
        if lease_seconds < 1:
            raise ValueError("the turn lease must be positive")
        source = {} if source_receipt is None else {"split_source_receipt": source_receipt}
        result = await session.execute(self._turns.update().filter_by(
            **self._turn_values(key), generation=generation, state="cancelling" if cancelling else "accepted",
        ).where(self._turns.c.lease_until <= self._now).values(
            generation=generation + 1, lease_until=self._now + lease_seconds, **source,
        ))
        return result.rowcount == 1

    async def take_over(
        self, key: SessionTurnKey, *, generation: int, lease_seconds: int, cancelling: bool = False,
    ) -> bool:
        async def write(session: AsyncSession) -> bool:
            return await self._take_over(session, key, generation=generation,
                                         lease_seconds=lease_seconds, cancelling=cancelling)

        return await self._context.run_mutation(write, domain="session.recover")

    async def split_target_lease(
        self, key: SessionTurnKey, *, generation: int, cancelling: bool, expired: bool,
    ) -> _SplitTargetObservation | None:
        """Check the exact target phase and authoritative lease before fencing its source."""
        from sqlalchemy import select

        cols = self._turns.c
        query = select(cols.state, cols.split_source_receipt, cols.created_at).where(
            *(cols[name] == value for name, value in self._turn_values(key).items()),
            cols.generation == generation, cols.state == ("cancelling" if cancelling else "accepted"),
            cols.lease_until <= self._now if expired else cols.lease_until > self._now,
        )
        if cancelling and expired:
            # Prove occupancy before a cold Stop can seal a different database.
            owner = self._sessions.c
            query = query.where(select(owner.active_turn_id).where(
                owner.scope_key == cols.scope_key, owner.session_key == cols.session_key,
                owner.active_turn_id == cols.turn_id,
            ).exists())
        async with self._context.engine.connect() as connection:
            row = (await connection.execute(query)).one_or_none()
        if row is None:
            return None
        return _SplitTargetObservation(
            generation, row.state,
            None if row.split_source_receipt is None else row.split_source_receipt.decode("utf-8"),
            row.created_at,
        )

    async def take_over_split_generation(
        self, key: SessionTurnKey, *, generation: int, lease_seconds: int,
        source: SplitStepSourceObservation,
    ) -> tuple[bool, SessionInstructionFrontier | None]:
        """Adopt the exact source seal only after the previous lease expired."""
        if (source.turn_id != key.turn_id or source.session_id != key.session_id
                or source.generation != generation or source.phase != "sealed"
                or source.pin.terminal_state is not None or source.pin.checkpoint_id is None):
            raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)

        async def write(session: AsyncSession) -> tuple[bool, SessionInstructionFrontier | None]:
            acquired = await self._take_over(
                session, key, generation=generation, lease_seconds=lease_seconds, cancelling=False,
                source_receipt=source.pin.receipt_id.encode("utf-8"),
            )
            return acquired, (await self._capture_instruction_frontier(session, key)) if acquired else None

        return await self._context.run_mutation(write, domain="session.recover")

    async def take_over_split_ready(
        self, frontier: CanonicalDeferredFrontier, *, lease_seconds: int, source: SplitStepSourceObservation,
    ) -> tuple[bool, SessionInstructionFrontier | None]:
        """Grant a settled frontier once, without a second session or tool receipt."""
        from sqlalchemy import select

        if lease_seconds < 1:
            raise ValueError("the turn lease must be positive")
        key = frontier.key
        receipt = self._require_split_deferred_source(frontier, source, phase="ready")
        owner = self._turn_values(key)
        occupied = select(self._sessions.c.id).filter_by(
            **self._session_values(key.session_id), active_turn_id=owner["turn_id"],
        ).exists()
        proved = self._split_deferred_projection(frontier, receipt=receipt, state="ready")

        async def write(session: AsyncSession):
            row = (await session.execute(proved)).one_or_none()
            if row is None:
                return False, None
            if CanonicalDeferredFrontier.from_json(key, frontier.generation, row[0]) != frontier:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            # The existing READY payload is immutable at this generation. The
            # CAS fences concurrent regrant/Stop. Retain the exact original
            # frontier until the source durably checkpoints the resumed state.
            changed = await session.execute(self._turns.update().filter_by(
                **owner, generation=frontier.generation, state="ready", lease_until=None,
                split_source_receipt=receipt,
            ).where(occupied).values(
                state="accepted", generation=frontier.generation + 1, lease_until=self._now + lease_seconds,
            ))
            if changed.rowcount != 1:
                return False, None
            return True, await self._capture_instruction_frontier(session, key)

        return await self._context.run_mutation(write, domain="session.recover")

    async def read_split_ready_frontier(
        self, key: SessionTurnKey, *, source: SplitStepSourceObservation,
    ) -> CanonicalDeferredFrontier | None:
        """Read the settled target while its exact occupied READY owner survives."""
        waiting, checkpoint_id = source.pin.waiting, source.pin.checkpoint_id
        if waiting is None or checkpoint_id is None:
            raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)
        expected = CanonicalDeferredFrontier.from_split_waiting(key, checkpoint_id=checkpoint_id, waiting=waiting)
        receipt = self._require_split_deferred_source(expected, source, phase="ready")
        statement = self._split_deferred_projection(expected, receipt=receipt, state="ready").where(
            self._turns.c.lease_until.is_(None),
        )
        async with self._context.engine.connect() as connection:
            row = (await connection.execute(statement)).one_or_none()
        if row is None:
            return None
        frontier = CanonicalDeferredFrontier.from_json(key, expected.generation, row[0])
        if frontier != expected:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return frontier

    async def read_split_resuming_frontier(
        self, key: SessionTurnKey, *, generation: int, original_generation: int, receipt_id: str,
    ) -> CanonicalDeferredFrontier | None:
        """Read the original frontier on the exact expired split regrant."""
        from sqlalchemy import and_, select

        if not 1 <= original_generation < generation:
            return None
        turns, sessions = self._turns.c, self._sessions.c
        statement = select(turns.result_json).select_from(self._turns.join(
            self._sessions, and_(
                sessions.scope_key == turns.scope_key, sessions.session_key == turns.session_key,
                sessions.active_turn_id == turns.turn_id,
            ),
        )).where(
            *(turns[name] == value for name, value in self._turn_values(key).items()),
            turns.generation == generation, turns.state == "accepted", turns.lease_until <= self._now,
            turns.split_source_receipt == receipt_id.encode("utf-8"),
        )
        async with self._context.engine.connect() as connection:
            row = (await connection.execute(statement)).one_or_none()
        if row is None or row[0] is None:
            return None
        return CanonicalDeferredFrontier.from_json(key, original_generation, row[0])

    async def take_over_split_resuming(
        self, frontier: CanonicalDeferredFrontier, *, source: SplitStepSourceObservation, lease_seconds: int,
    ) -> tuple[bool, SessionInstructionFrontier | None]:
        """Grant a replacement lease without releasing the frozen waiting proof."""
        from sqlalchemy import select

        key, generation = frontier.key, source.generation
        if generation <= frontier.generation or lease_seconds < 1:
            raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)
        waiting = source.pin.waiting
        if (waiting is None or source.phase != "resume_hold"
                or (source.turn_id, source.session_id) != (key.turn_id, key.session_id)
                or source.pin.checkpoint_id != frontier.checkpoint_id
                or waiting.original_generation != frontier.generation
                or source.pin.receipt_id != frontier.split_receipt_id):
            raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)
        owner = self._turn_values(key)
        occupied = select(self._sessions.c.id).filter_by(
            **self._session_values(key.session_id), active_turn_id=owner["turn_id"],
        ).exists()

        async def write(session: AsyncSession):
            changed = await session.execute(self._turns.update().filter_by(
                **owner, generation=generation, state="accepted",
                split_source_receipt=source.pin.receipt_id.encode("utf-8"),
            ).where(occupied, self._turns.c.lease_until <= self._now).values(
                generation=generation + 1, lease_until=self._now + lease_seconds,
            ))
            if changed.rowcount != 1:
                return False, None
            return True, await self._capture_instruction_frontier(session, key)

        return await self._context.run_mutation(write, domain="session.recover")

    async def confirmed_split_regrant(
        self, key: SessionTurnKey, *, generation: int, receipt_id: str,
    ) -> tuple[bool, SessionInstructionFrontier | None]:
        from sqlalchemy import select

        cols = self._turns.c
        owner = self._turn_values(key)
        occupied = select(self._sessions.c.id).filter_by(
            **self._session_values(key.session_id), active_turn_id=owner["turn_id"],
        ).exists()
        statement = select(cols.request_json["instruction_frontier"]).where(
            *(cols[name] == value for name, value in owner.items()),
            occupied,
            cols.generation == generation, cols.state == "accepted", cols.lease_until > self._now,
            cols.split_source_receipt == receipt_id.encode("utf-8"),
        )
        async with self._context.engine.connect() as connection:
            row = (await connection.execute(statement)).one_or_none()
        return (False, None) if row is None else (True, SessionInstructionFrontier.from_json(row[0]))

    def _require_checkpoint_store(self, step_store: SqlStepStore) -> None:
        if self._transcript_write_binding is None:
            raise AIError(ErrorCode.STORAGE_DEPENDENCY_NOT_READY)
        _store, _archive, native = self._transcript_write_binding
        if (step_store.storage_context.engine is not self._context.engine
                or step_store.store_key != native.store_key):
            raise AIError(ErrorCode.STORAGE_CAPABILITY_MISSING)

    async def _capture_instruction_frontier(
        self, session: AsyncSession, key: SessionTurnKey,
    ) -> SessionInstructionFrontier | None:
        from sqlalchemy import select

        # The successful regrant UPDATE owns this row until the transaction
        # commits. Capture after it, not from the earlier authorization read.
        columns = self._turns.c
        statement = select(columns.request_json["instruction_frontier"]).where(
            *(columns[name] == value for name, value in self._turn_values(key).items()),
        )
        result = await session.execute(statement)
        return SessionInstructionFrontier.from_json(result.scalar_one())

    async def take_over_with_checkpoint(
        self, key: SessionTurnKey, *, generation: int, lease_seconds: int, step_store: SqlStepStore,
    ) -> tuple[bool, ContinuableSnapshot | None, SessionInstructionFrontier | None]:
        """Regrant the canonical lease and freeze its stable native checkpoint frontier."""
        self._require_checkpoint_store(step_store)

        async def write(session: AsyncSession):
            if not await self._take_over(session, key, generation=generation,
                                         lease_seconds=lease_seconds, cancelling=False):
                return False, None, None
            checkpoint = await step_store.capture_turn_checkpoint(
                session, run_id=key.turn_id, conversation_id=key.session_id,
            )
            instructions = await self._capture_instruction_frontier(session, key)
            return True, checkpoint, instructions

        acquired, checkpoint, instructions = await self._context.run_mutation(write, domain="session.recover")
        return acquired, None if checkpoint is None else await checkpoint(), instructions


    async def take_over_ready_checkpoint(
        self, frontier: CanonicalDeferredFrontier, *, lease_seconds: int, step_store: SqlStepStore,
    ) -> tuple[bool, ContinuableSnapshot | None, SessionInstructionFrontier | None]:
        """Adopt a settled frontier exactly once, without another turn or owner."""
        if lease_seconds < 1:
            raise ValueError("the turn lease must be positive")
        self._require_checkpoint_store(step_store)
        key = frontier.key

        async def write(session: AsyncSession):
            # Only this generation can have installed the immutable frontier.
            # The CAS serializes against ongoing receipt settlement/withdrawal.
            acquired = await session.execute(self._turns.update().filter_by(
                **self._turn_values(key), generation=frontier.generation, state="ready", lease_until=None,
            ).values(
                state="accepted", generation=frontier.generation + 1,
                lease_until=self._now + lease_seconds, result_json=None,
            ))
            if acquired.rowcount != 1:
                return None, None
            checkpoint = await step_store.capture_turn_checkpoint(
                session, run_id=key.turn_id, conversation_id=key.session_id, replay_key=frontier.checkpoint_id,
            )
            if checkpoint is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            return checkpoint, await self._capture_instruction_frontier(session, key)

        checkpoint, instructions = await self._context.run_mutation(write, domain="session.recover")
        return (False, None, None) if checkpoint is None else (True, await checkpoint(), instructions)
