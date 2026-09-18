#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Conversation-domain repository implementations."""

from dataclasses import replace
from datetime import datetime
from linktools.core import environ
from ...core import OperationKind, OperationLedgerInput, Page, ResourceKind, SessionStatus, canonical_sha256, validate_agent_id
from ...errors import AIError, ErrorCode
from ._codec import _decode_enveloped_domain
from ._contracts import ConversationCursor, ConversationHistoryIndexNodeRecord, ConversationHistoryRecord, HistoryQuality, SessionRecord, SessionTurnCommitRef, SessionTurnRef, TranscriptHeadRecord, TranscriptOwnerDomain
from ._history_index import build_fork_index_node_from_roots
from ._plan import RuntimeDomain
from ._store import FactQuery, RecordQuery, StateStore, StateTransaction, StoredFact, StoredRecord, sequence_key, sortable_identity, stream_digest, subject_digest
from ._repository_common import (
    RepositoryBase as _RepositoryBase,
    ResourceRepository as _ResourceRepository,
    append_operation as _append_operation,
    decode_record_cursor as _decode_record_cursor,
    projected_record as _projected_record,
    record_cursor as _record_cursor,
    replace_checked as _replace_checked,
    require_repository_tenant as _require_repository_tenant,
    require_tenant as _require_tenant,
    validate_page_limit as _validate_page_limit,
)

_logger = environ.get_logger("ai.runtime.state.repositories")


class ConversationHistoryRepositoryImpl(_RepositoryBase):
    """Persist immutable branch descriptors and their skew prefix index."""

    def __init__(self, store: StateStore, *, namespace: str, tenant_id: str) -> None:
        super().__init__(
            store,
            namespace=namespace,
            tenant_id=tenant_id,
            domain=RuntimeDomain.CONVERSATION,
        )

    async def create(
        self, record: ConversationHistoryRecord
    ) -> ConversationHistoryRecord:
        _require_tenant(record, self._tenant_id)

        async def mutate(transaction: StateTransaction) -> ConversationHistoryRecord:
            return await self.create_in_transaction(transaction, record)

        return await self._store.mutate(mutate)

    async def create_in_transaction(
        self,
        transaction: StateTransaction,
        record: ConversationHistoryRecord,
    ) -> ConversationHistoryRecord:
        _require_tenant(record, self._tenant_id)
        history_key = self._key("conversation_history", record.history_id)
        head_key = self._key("transcript_head", record.history_id)
        records = await transaction.get_records((history_key, head_key))
        current = records.get(history_key)
        head = records.get(head_key)
        if current is not None:
            existing = await self._decode_history(current)
            if existing != record:
                raise AIError(ErrorCode.IDEMPOTENCY_CONFLICT)
            if head is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            return existing
        if head is not None:
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        await transaction.insert_records(
            (
                self._stored("conversation_history", record.history_id, record),
                self._stored(
                    "transcript_head",
                    record.history_id,
                    TranscriptHeadRecord(
                        TranscriptOwnerDomain.CONVERSATION,
                        record.history_id,
                        0,
                        0,
                        HistoryQuality.COMPLETE,
                    ),
                ),
            )
        )
        _logger.debug(
            "conversation history admitted with head: history=%s",
            record.history_id,
        )
        return record

    async def get(
        self,
        history_id: str,
        *,
        tenant_id: str,
    ) -> ConversationHistoryRecord | None:
        if tenant_id != self._tenant_id:
            return None
        record = await self._record(self._key("conversation_history", history_id))
        return None if record is None else await self._decode_history(record)

    async def get_in_transaction(
        self,
        transaction: StateTransaction,
        history_id: str,
        *,
        tenant_id: str,
    ) -> ConversationHistoryRecord | None:
        _require_repository_tenant(tenant_id, self._tenant_id)
        record = await transaction.get_record(
            self._key("conversation_history", history_id)
        )
        return None if record is None else await self._decode_history(record)

    async def local_head_in_transaction(
        self,
        transaction: StateTransaction,
        history_id: str,
    ) -> int:
        """Read one branch's local canonical message count."""
        record = await transaction.get_record(self._key("transcript_head", history_id))
        if record is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        head = await self._decode(record, TranscriptHeadRecord)
        return head.message_count

    async def get_index_node_in_transaction(
        self,
        transaction: StateTransaction,
        node_id: str,
    ) -> ConversationHistoryIndexNodeRecord:
        """Read exactly one skew index node without traversing its children."""
        record = await transaction.get_record(
            self._key("conversation_index_node", node_id)
        )
        if record is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        node = _decode_enveloped_domain(
            record.data,
            ConversationHistoryIndexNodeRecord,
        )
        if node.node_id != node_id:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return node

    async def get_index_node(
        self,
        node_id: str,
        *,
        tenant_id: str,
    ) -> ConversationHistoryIndexNodeRecord:
        _require_repository_tenant(tenant_id, self._tenant_id)
        return await self._store.read(
            lambda transaction: self.get_index_node_in_transaction(
                transaction,
                node_id,
            )
        )

    async def get_forest_roots_in_transaction(
        self,
        transaction: StateTransaction,
        head_id: str | None,
        *,
        max_roots: int,
    ) -> tuple[ConversationHistoryIndexNodeRecord, ...]:
        if max_roots < 1:
            raise ValueError("max_roots must be positive")
        roots: list[ConversationHistoryIndexNodeRecord] = []
        cursor = head_id
        visited: set[str] = set()
        while cursor is not None and len(roots) < max_roots:
            if cursor in visited:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            visited.add(cursor)
            node = await self.get_index_node_in_transaction(transaction, cursor)
            roots.append(node)
            cursor = node.next_forest_id
        if cursor is not None and max_roots > 2:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return tuple(roots)

    async def get_forest_roots(
        self,
        head_id: str | None,
        *,
        tenant_id: str,
        max_roots: int,
    ) -> tuple[ConversationHistoryIndexNodeRecord, ...]:
        _require_repository_tenant(tenant_id, self._tenant_id)
        return await self._store.read(
            lambda transaction: self.get_forest_roots_in_transaction(
                transaction,
                head_id,
                max_roots=max_roots,
            )
        )

    async def insert_index_node_in_transaction(
        self,
        transaction: StateTransaction,
        node: ConversationHistoryIndexNodeRecord,
    ) -> None:
        key = self._key("conversation_index_node", node.node_id)
        if await transaction.get_record(key) is not None:
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        await transaction.insert_record(
            self._stored("conversation_index_node", node.node_id, node)
        )

    async def fork(
        self,
        source_history_id: str,
        child_history_id: str,
        *,
        session_id: str,
        tenant_id: str,
    ) -> ConversationHistoryRecord:
        _require_repository_tenant(tenant_id, self._tenant_id)

        async def mutate(transaction: StateTransaction) -> ConversationHistoryRecord:
            source = await self.get_in_transaction(
                transaction,
                source_history_id,
                tenant_id=tenant_id,
            )
            if source is None:
                raise AIError(ErrorCode.SESSION_HISTORY_UNAVAILABLE)
            local_messages = await self.local_head_in_transaction(
                transaction,
                source_history_id,
            )
            prefix_head = source.prefix_index_head_id
            if local_messages > 0:
                roots = await self.get_forest_roots_in_transaction(
                    transaction,
                    prefix_head,
                    max_roots=2,
                )
                node = build_fork_index_node_from_roots(
                    roots,
                    source_history_id=source_history_id,
                    source_local_message_count=local_messages,
                )
                if node is None:
                    pass
                elif isinstance(node, str):
                    prefix_head = node
                else:
                    await self.insert_index_node_in_transaction(transaction, node)
                    prefix_head = node.node_id
            inherited_messages = source.inherited_message_count + local_messages
            child = ConversationHistoryRecord(
                history_id=child_history_id,
                session_id=session_id,
                tenant_id=tenant_id,
                parent_history_id=source_history_id,
                prefix_index_head_id=prefix_head,
                inherited_message_count=inherited_messages,
            )
            key = self._key("conversation_history", child_history_id)
            current = await transaction.get_record(key)
            if current is None:
                return await self.create_in_transaction(transaction, child)
            existing = await self._decode_history(current)
            if existing == child:
                return existing
            if (
                existing.session_id == session_id
                and existing.parent_history_id is None
                and existing.inherited_message_count == 0
            ):
                await _replace_checked(
                    transaction,
                    replace(
                        self._stored(
                            "conversation_history",
                            child_history_id,
                            child,
                        ),
                        storage_version=current.storage_version + 1,
                    ),
                    current.storage_version,
                )
                return child
            raise AIError(ErrorCode.IDEMPOTENCY_CONFLICT)

        return await self._store.mutate(mutate)

    async def _decode_history(self, record: StoredRecord) -> ConversationHistoryRecord:
        return _decode_enveloped_domain(record.data, ConversationHistoryRecord)


class SessionRepositoryImpl(_ResourceRepository[SessionRecord]):
    def __init__(self, store: StateStore, *, namespace: str, tenant_id: str) -> None:
        super().__init__(
            store,
            namespace=namespace,
            tenant_id=tenant_id,
            domain=RuntimeDomain.CONVERSATION,
            kind="session",
            resource_kind=ResourceKind.SESSION,
            value_type=SessionRecord,
        )


    def _timeline_sequence_key(self, session_id: str) -> bytes:
        return sequence_key(
            self._namespace,
            self._tenant_id,
            self._domain.value,
            "session_turn",
            [session_id],
        )

    def _timeline_stream(self, session_id: str) -> bytes:
        return stream_digest(
            self._namespace,
            self._tenant_id,
            self._domain.value,
            "session_turn",
            [session_id],
        )

    def _timeline_commit_key(self, session_id: str, sequence: int) -> bytes:
        return self._key("session_turn_commit", [session_id, sequence])

    def _stored_timeline_commit(
        self, value: SessionTurnCommitRef
    ) -> StoredRecord:
        identity = [value.session_id, value.sequence]
        return StoredRecord(
            self._timeline_commit_key(value.session_id, value.sequence),
            self._partition("session_turn_commit"),
            self._scope("session_turn_commit", "session", value.session_id),
            None,
            "session_turn_commit",
            sortable_identity(identity),
            None,
            0,
            None,
            0,
            None,
            {
                "version": 1,
                "session_id": value.session_id,
                "sequence": value.sequence,
                "execution_id": value.execution_id,
                "start_message_index": value.start_message_index,
                "end_message_index": value.end_message_index,
            },
        )

    @staticmethod
    def _timeline_subject(execution_id: str) -> bytes:
        return subject_digest(execution_id)

    def _decode_timeline_turn(
        self, session_id: str, fact: StoredFact
    ) -> SessionTurnRef:
        if (
            fact.kind != "session_turn"
            or fact.owner_key_digest != self._key("session", session_id)
            or set(fact.data) != {"version", "execution_id"}
            or fact.data.get("version") != 1
            or not isinstance(fact.data.get("execution_id"), str)
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        execution_id = str(fact.data["execution_id"])
        if fact.subject_digest != self._timeline_subject(execution_id):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return SessionTurnRef(session_id, fact.sequence, execution_id)

    def _decode_timeline_commit(
        self, session_id: str, sequence: int, record: StoredRecord
    ) -> SessionTurnCommitRef:
        if (
            record.key_digest != self._timeline_commit_key(session_id, sequence)
            or record.partition_digest != self._partition("session_turn_commit")
            or record.scope_digest
            != self._scope("session_turn_commit", "session", session_id)
            or record.parent_digest is not None
            or record.kind != "session_turn_commit"
            or record.sort_key != sortable_identity([session_id, sequence])
            or record.state is not None
            or set(record.data)
            != {
                "version",
                "session_id",
                "sequence",
                "execution_id",
                "start_message_index",
                "end_message_index",
            }
            or record.data.get("version") != 1
            or record.data.get("session_id") != session_id
            or record.data.get("sequence") != sequence
            or not isinstance(record.data.get("execution_id"), str)
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        execution_id = str(record.data["execution_id"])
        start = record.data.get("start_message_index")
        end = record.data.get("end_message_index")
        if (
            isinstance(start, bool)
            or not isinstance(start, int)
            or isinstance(end, bool)
            or not isinstance(end, int)
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        try:
            return SessionTurnCommitRef(
                session_id, sequence, execution_id, start, end
            )
        except ValueError as error:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error

    def _list_generation_key(self, owner_principal_id: str | None = None) -> bytes:
        return sequence_key(
            self._namespace,
            self._tenant_id,
            self._domain.value,
            "session_list_owner"
            if owner_principal_id is not None
            else "session_list_tenant",
            owner_principal_id if owner_principal_id is not None else [],
        )

    async def _bump_list_generation(
        self,
        transaction: StateTransaction,
        owner_principal_id: str,
    ) -> None:
        await transaction.reserve_sequences(
            {
                self._list_generation_key(owner_principal_id): 1,
                self._list_generation_key(): 1,
            }
        )

    async def create(self, value: SessionRecord) -> SessionRecord:
        _require_tenant(value, self._tenant_id)
        _require_explicit_session_agent_id(value)
        value = _ensure_session_history(value)

        async def mutate(transaction: StateTransaction) -> SessionRecord:
            await transaction.insert_records(
                (
                    self._stored(
                        "session",
                        value.session_id,
                        value,
                        scope=self._scope("session", "owner", value.owner_principal_id),
                        state=value.status.value,
                    ),
                    self._stored(
                        "conversation_history",
                        value.history_id,
                        _new_session_history(value),
                    ),
                    self._stored(
                        "transcript_head",
                        value.history_id,
                        _empty_conversation_transcript_head(value.history_id),
                    ),
                )
            )
            await self._bump_list_generation(transaction, value.owner_principal_id)
            _logger.debug(
                "session admitted with history: session=%s history=%s",
                value.session_id,
                value.history_id,
            )
            return value

        return await self._store.mutate(mutate)

    async def create_with_operation(
        self, record: SessionRecord, *, operation: OperationLedgerInput
    ) -> tuple[SessionRecord, bool]:
        _require_tenant(record, self._tenant_id)
        _require_tenant(operation, self._tenant_id)
        _require_explicit_session_agent_id(record)
        record = _ensure_session_history(record)

        async def mutate(transaction: StateTransaction) -> tuple[SessionRecord, bool]:
            _, replayed = await _append_operation(transaction, self, operation)
            if replayed:
                current = await transaction.get_record(
                    self._key("session", record.session_id)
                )
                if current is None:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                return await self._decode(current, SessionRecord), True
            await transaction.insert_records(
                (
                    self._stored(
                        "session",
                        record.session_id,
                        record,
                        scope=self._scope(
                            "session", "owner", record.owner_principal_id
                        ),
                        state=record.status.value,
                    ),
                    self._stored(
                        "conversation_history",
                        record.history_id,
                        _new_session_history(record),
                    ),
                    self._stored(
                        "transcript_head",
                        record.history_id,
                        _empty_conversation_transcript_head(record.history_id),
                    ),
                )
            )
            await self._bump_list_generation(transaction, record.owner_principal_id)
            _logger.debug(
                "session operation admitted with history: session=%s history=%s",
                record.session_id,
                record.history_id,
            )
            return record, False

        return await self._store.mutate(mutate)

    async def create_fork_with_operation(
        self,
        source_session_id: str,
        target: SessionRecord,
        *,
        expected_source_revision: int,
        operation: OperationLedgerInput,
    ) -> tuple[SessionRecord, bool]:
        _require_tenant(target, self._tenant_id)
        _require_tenant(operation, self._tenant_id)
        _require_explicit_session_agent_id(target)
        if (
            operation.resource_kind is not ResourceKind.SESSION
            or operation.resource_id != target.session_id
            or operation.operation_kind is not OperationKind.SESSION_FORK
        ):
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        if expected_source_revision < 0:
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        target = _ensure_session_history(replace(target, history_id=None))

        async def mutate(transaction: StateTransaction) -> tuple[SessionRecord, bool]:
            _, replayed = await _append_operation(
                transaction,
                self,
                operation,
            )
            if replayed:
                return await self._replay_fork_in_transaction(
                    transaction,
                    source_session_id=source_session_id,
                    target=target,
                )
            source_stored = await transaction.get_record(
                self._key("session", source_session_id)
            )
            if source_stored is None:
                raise AIError(ErrorCode.STORAGE_NOT_FOUND)
            source = await self._decode(source_stored, SessionRecord)
            if source.tenant_id != self._tenant_id:
                raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)
            if source.revision != expected_source_revision:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            if source.history_id is None:
                raise AIError(ErrorCode.SESSION_HISTORY_UNAVAILABLE)
            if (
                await transaction.guard_record(
                    self._key("session", source_session_id),
                    expected_storage_version=source_stored.storage_version,
                )
                is None
            ):
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            child_history_id = target.history_id
            if child_history_id is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            source_history_key = self._key("conversation_history", source.history_id)
            target_key = self._key("session", target.session_id)
            child_history_key = self._key("conversation_history", child_history_id)
            source_head_key = self._key("transcript_head", source.history_id)
            related = await transaction.get_records(
                (
                    source_history_key,
                    target_key,
                    child_history_key,
                    source_head_key,
                )
            )
            source_history_stored = related.get(source_history_key)
            if source_history_stored is None:
                raise AIError(ErrorCode.SESSION_HISTORY_UNAVAILABLE)
            source_history = await self._decode_history(source_history_stored)
            if (
                source_history.session_id != source.session_id
                or source_history.tenant_id != self._tenant_id
            ):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            target_stored = related.get(target_key)
            child_stored = related.get(child_history_key)
            source_head_stored = related.get(source_head_key)
            if source_head_stored is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            source_head = _decode_enveloped_domain(
                source_head_stored.data,
                TranscriptHeadRecord,
            )
            local_messages = source_head.message_count
            histories = ConversationHistoryRepositoryImpl(
                self._store,
                namespace=self._namespace,
                tenant_id=self._tenant_id,
            )
            prefix_head = source_history.prefix_index_head_id
            if local_messages > 0:
                roots = await histories.get_forest_roots_in_transaction(
                    transaction,
                    prefix_head,
                    max_roots=2,
                )
                node = build_fork_index_node_from_roots(
                    roots,
                    source_history_id=source.history_id,
                    source_local_message_count=local_messages,
                )
                if isinstance(node, str):
                    prefix_head = node
                else:
                    await histories.insert_index_node_in_transaction(
                        transaction,
                        node,
                    )
                    prefix_head = node.node_id
            inherited = source_history.inherited_message_count + local_messages
            turn_head = await transaction.get_sequence(
                self._timeline_sequence_key(source_session_id)
            )
            if source.active_execution_id is not None:
                if turn_head < 1:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                timeline_cutoff = turn_head - 1
            else:
                timeline_cutoff = turn_head
            if timeline_cutoff > 0:
                timeline_parent = source.session_id
                timeline_parent_cutoff = timeline_cutoff
            else:
                timeline_parent = source.timeline_parent_session_id
                timeline_parent_cutoff = source.timeline_parent_turn_sequence
            child = ConversationHistoryRecord(
                history_id=child_history_id,
                session_id=target.session_id,
                tenant_id=self._tenant_id,
                parent_history_id=source.history_id,
                prefix_index_head_id=prefix_head,
                inherited_message_count=inherited,
            )
            expected_target = replace(
                target,
                history_id=child_history_id,
                history_quality="complete",
                timeline_parent_session_id=timeline_parent,
                timeline_parent_turn_sequence=timeline_parent_cutoff,
                continuation=(
                    None
                    if target.continuation is None
                    else replace(
                        target.continuation,
                        history_id=child_history_id,
                    )
                ),
            )
            if target_stored is not None or child_stored is not None:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            await transaction.insert_records(
                (
                    self._stored(
                        "session",
                        expected_target.session_id,
                        expected_target,
                        scope=self._scope(
                            "session",
                            "owner",
                            expected_target.owner_principal_id,
                        ),
                        state=expected_target.status.value,
                    ),
                    self._stored(
                        "conversation_history",
                        child.history_id,
                        child,
                    ),
                    self._stored(
                        "transcript_head",
                        child.history_id,
                        _empty_conversation_transcript_head(child.history_id),
                    ),
                )
            )
            await self._bump_list_generation(
                transaction,
                expected_target.owner_principal_id,
            )
            _logger.info(
                "session fork committed: source=%s target=%s inherited=%s",
                source_session_id,
                expected_target.session_id,
                inherited,
            )
            return expected_target, False

        return await self._store.mutate(mutate)

    async def _replay_fork_in_transaction(
        self,
        transaction: StateTransaction,
        *,
        source_session_id: str,
        target: SessionRecord,
    ) -> tuple[SessionRecord, bool]:
        target_history_id = target.history_id
        if target_history_id is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        source_key = self._key("session", source_session_id)
        target_key = self._key("session", target.session_id)
        child_key = self._key("conversation_history", target_history_id)
        head_key = self._key("transcript_head", target_history_id)
        related = await transaction.get_records(
            (source_key, target_key, child_key, head_key)
        )
        source_stored = related.get(source_key)
        target_stored = related.get(target_key)
        child_stored = related.get(child_key)
        head_stored = related.get(head_key)
        if (
            source_stored is None
            or target_stored is None
            or child_stored is None
            or head_stored is None
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        source = await self._decode(source_stored, SessionRecord)
        existing_target = await self._decode(target_stored, SessionRecord)
        child = await self._decode_history(child_stored)
        if (
            source.session_id != source_session_id
            or source.tenant_id != self._tenant_id
            or source.history_id is None
            or existing_target.session_id != target.session_id
            or existing_target.history_id != target_history_id
            or existing_target.tenant_id != self._tenant_id
            or existing_target.owner_principal_id != target.owner_principal_id
            or existing_target.agent_id != target.agent_id
            or child.session_id != target.session_id
            or child.tenant_id != self._tenant_id
            or child.parent_history_id != source.history_id
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        head = _decode_enveloped_domain(
            head_stored.data,
            TranscriptHeadRecord,
        )
        if (
            head.owner_domain is not TranscriptOwnerDomain.CONVERSATION
            or head.owner_id != target_history_id
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        _logger.info(
            "session fork replayed: source=%s target=%s",
            source_session_id,
            existing_target.session_id,
        )
        return existing_target, True


    async def _visible_history_count_in_transaction(
        self,
        transaction: StateTransaction,
        record: ConversationHistoryRecord,
    ) -> int:
        histories = ConversationHistoryRepositoryImpl(
            self._store,
            namespace=self._namespace,
            tenant_id=self._tenant_id,
        )
        local_messages = await histories.local_head_in_transaction(
            transaction,
            record.history_id,
        )
        return record.inherited_message_count + local_messages

    async def timeline_head(self, session_id: str, *, tenant_id: str) -> int:
        _require_repository_tenant(tenant_id, self._tenant_id)
        return await self._store.read(
            lambda transaction: transaction.get_sequence(
                self._timeline_sequence_key(session_id)
            )
        )

    async def list_timeline_turns(
        self,
        session_id: str,
        *,
        tenant_id: str,
        start_sequence: int,
        end_sequence: int,
    ) -> tuple[SessionTurnRef, ...]:
        _require_repository_tenant(tenant_id, self._tenant_id)
        if start_sequence < 1 or end_sequence < start_sequence:
            raise ValueError("session timeline range is invalid")
        if start_sequence == end_sequence:
            return ()
        facts = await self._store.read(
            lambda transaction: transaction.list_facts(
                FactQuery(
                    self._timeline_stream(session_id),
                    after_sequence=start_sequence - 1,
                    limit=end_sequence - start_sequence,
                )
            )
        )
        selected = tuple(
            fact for fact in facts if fact.sequence < end_sequence
        )
        if tuple(fact.sequence for fact in selected) != tuple(
            range(start_sequence, end_sequence)
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return tuple(
            self._decode_timeline_turn(session_id, fact) for fact in selected
        )

    async def list_timeline_commits(
        self,
        session_id: str,
        *,
        tenant_id: str,
        start_sequence: int,
        end_sequence: int,
    ) -> tuple[SessionTurnCommitRef, ...]:
        _require_repository_tenant(tenant_id, self._tenant_id)
        if start_sequence < 1 or end_sequence < start_sequence:
            raise ValueError("session timeline range is invalid")
        if start_sequence == end_sequence:
            return ()
        sequences = tuple(range(start_sequence, end_sequence))
        keys = tuple(
            self._timeline_commit_key(session_id, sequence)
            for sequence in sequences
        )
        records = await self._store.read(
            lambda transaction: transaction.get_records(keys)
        )
        return tuple(
            self._decode_timeline_commit(session_id, sequence, records[key])
            for sequence, key in zip(sequences, keys)
            if key in records
        )

    async def commit_timeline_turn_in_transaction(
        self,
        transaction: StateTransaction,
        session_id: str,
        *,
        tenant_id: str,
        execution_id: str,
        start_message_index: int,
        end_message_index: int,
    ) -> SessionTurnCommitRef:
        _require_repository_tenant(tenant_id, self._tenant_id)
        subject = self._timeline_subject(execution_id)
        turns = await transaction.list_facts(
            FactQuery(
                self._timeline_stream(session_id),
                subject_digest=subject,
            )
        )
        if len(turns) != 1:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        turn = self._decode_timeline_turn(session_id, turns[0])
        if turn.execution_id != execution_id:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        commit_key = self._timeline_commit_key(session_id, turn.sequence)
        existing = await transaction.get_record(commit_key)
        if existing is not None:
            committed = self._decode_timeline_commit(
                session_id, turn.sequence, existing
            )
            if committed.execution_id != execution_id:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if (
                committed.end_message_index != end_message_index
                or committed.start_message_index != start_message_index
            ):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            return committed
        if start_message_index < 0 or end_message_index <= start_message_index:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        committed = SessionTurnCommitRef(
            session_id,
            turn.sequence,
            execution_id,
            start_message_index,
            end_message_index,
        )
        await transaction.insert_record(self._stored_timeline_commit(committed))
        return committed

    async def list(
        self, *, tenant_id: str, owner_principal_id: str | None = None
    ) -> tuple[SessionRecord, ...]:
        if tenant_id != self._tenant_id:
            return ()
        scope = (
            None
            if owner_principal_id is None
            else self._scope("session", "owner", owner_principal_id)
        )

        async def read(transaction: StateTransaction) -> tuple[SessionRecord, ...]:
            records = await transaction.list_records(
                RecordQuery(
                    partition_digest=self._partition("session")
                    if scope is None
                    else None,
                    scope_digest=scope,
                    kind="session",
                )
            )
            return tuple(
                [await self._decode(record, SessionRecord) for record in records]
            )

        return await self._store.read(read)

    async def list_page(
        self,
        *,
        tenant_id: str,
        owner_principal_id: str | None,
        cursor: str | None,
        limit: int,
        snapshot: int | None = None,
    ) -> tuple[int, Page[SessionRecord]]:
        if tenant_id != self._tenant_id:
            return 0, Page(())
        _validate_page_limit(limit)
        scope = (
            None
            if owner_principal_id is None
            else self._scope(
                "session",
                "owner",
                owner_principal_id,
            )
        )

        async def read(
            transaction: StateTransaction,
        ) -> tuple[int, Page[SessionRecord]]:
            generation = await transaction.get_sequence(
                self._list_generation_key(owner_principal_id)
            )
            if snapshot is not None and snapshot != generation:
                raise AIError(ErrorCode.CURSOR_INVALID)
            after_sort_key, after_key_digest = _decode_record_cursor(cursor)
            records = await transaction.list_records(
                RecordQuery(
                    partition_digest=self._partition("session")
                    if scope is None
                    else None,
                    scope_digest=scope,
                    kind="session",
                    after_sort_key=after_sort_key,
                    after_key_digest=after_key_digest,
                    limit=min(limit + 1, 1000),
                )
            )
            if limit == 1000 and len(records) == 1000:
                last = records[-1]
                probe = await transaction.list_records(
                    RecordQuery(
                        partition_digest=(
                            self._partition("session")
                            if scope is None
                            else None
                        ),
                        scope_digest=scope,
                        kind="session",
                        after_sort_key=last.sort_key,
                        after_key_digest=last.key_digest,
                        limit=1,
                    )
                )
                if probe:
                    records = (*records, probe[0])
            values = tuple(
                [
                    await self._decode(record, SessionRecord)
                    for record in records[:limit]
                ]
            )
            next_cursor = (
                _record_cursor(records[limit - 1]) if len(records) > limit else None
            )
            return generation, Page(values, next_cursor)

        return await self._store.read(read)

    async def compare_and_swap_with_operation(
        self,
        session_id: str,
        *,
        tenant_id: str,
        expected_revision: int,
        next_record: SessionRecord,
        operation: OperationLedgerInput,
    ) -> tuple[SessionRecord, bool]:
        if tenant_id != self._tenant_id:
            raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)
        _require_tenant(next_record, self._tenant_id)
        _require_tenant(operation, self._tenant_id)

        async def mutate(transaction: StateTransaction) -> tuple[SessionRecord, bool]:
            current = await transaction.get_record(self._key("session", session_id))
            if current is None:
                raise AIError(ErrorCode.STORAGE_NOT_FOUND)
            value = await self._decode(current, SessionRecord)
            if value.revision != expected_revision:
                raise AIError(ErrorCode.SESSION_REVISION_CONFLICT)
            proposed = next_record
            if (
                value.active_execution_id is not None
                and proposed.active_execution_id != value.active_execution_id
            ):
                proposed = replace(
                    proposed, active_execution_id=value.active_execution_id
                )
            _, replayed = await _append_operation(transaction, self, operation)
            if replayed:
                return value, True
            candidate = _projected_record(self, current, proposed)
            await _replace_checked(transaction, candidate, current.storage_version)
            await self._bump_list_generation(transaction, value.owner_principal_id)
            return proposed, False

        return await self._store.mutate(mutate)

    async def compare_and_swap(
        self,
        session_id: str,
        *,
        tenant_id: str,
        expected_revision: int,
        next_record: SessionRecord,
    ) -> SessionRecord:
        _require_repository_tenant(tenant_id, self._tenant_id)
        _require_tenant(next_record, self._tenant_id)

        async def mutate(transaction: StateTransaction) -> SessionRecord:
            record = await transaction.get_record(self._key("session", session_id))
            if record is None:
                raise AIError(ErrorCode.STORAGE_NOT_FOUND)
            current = await self._decode(record, SessionRecord)
            if current.revision != expected_revision:
                raise AIError(ErrorCode.SESSION_REVISION_CONFLICT)
            proposed = next_record
            if (
                current.active_execution_id is not None
                and proposed.active_execution_id != current.active_execution_id
            ):
                proposed = replace(
                    proposed, active_execution_id=current.active_execution_id
                )
            await _replace_checked(
                transaction,
                _projected_record(self, record, proposed),
                record.storage_version,
            )
            await self._bump_list_generation(transaction, current.owner_principal_id)
            return proposed

        return await self._store.mutate(mutate)

    async def admit_execution(
        self,
        session_id: str,
        *,
        tenant_id: str,
        execution_id: str,
        expected: ConversationCursor | None,
    ) -> SessionRecord:
        return await self._admission(
            session_id,
            tenant_id=tenant_id,
            execution_id=execution_id,
            expected=expected,
            release=False,
        )

    async def admit_execution_in_transaction(
        self,
        transaction: StateTransaction,
        session_id: str,
        *,
        tenant_id: str,
        execution_id: str,
        expected: ConversationCursor | None,
    ) -> SessionRecord:
        return await self._admission(
            session_id,
            tenant_id=tenant_id,
            execution_id=execution_id,
            expected=expected,
            release=False,
            transaction=transaction,
        )

    async def release_execution(
        self, session_id: str, *, tenant_id: str, execution_id: str
    ) -> SessionRecord:
        return await self._admission(
            session_id,
            tenant_id=tenant_id,
            execution_id=execution_id,
            expected=None,
            release=True,
        )

    async def release_execution_in_transaction(
        self,
        transaction: StateTransaction,
        session_id: str,
        *,
        tenant_id: str,
        execution_id: str,
    ) -> SessionRecord:
        return await self._admission(
            session_id,
            tenant_id=tenant_id,
            execution_id=execution_id,
            expected=None,
            release=True,
            transaction=transaction,
        )

    async def _admission(
        self,
        session_id: str,
        *,
        tenant_id: str,
        execution_id: str,
        expected: ConversationCursor | None,
        release: bool,
        transaction: StateTransaction | None = None,
    ) -> SessionRecord:
        _require_repository_tenant(tenant_id, self._tenant_id)

        async def mutate(transaction: StateTransaction) -> SessionRecord:
            current = await transaction.get_record(self._key("session", session_id))
            if current is None:
                raise AIError(ErrorCode.STORAGE_NOT_FOUND)
            value = await self._decode(current, SessionRecord)
            if release:
                if value.active_execution_id is None:
                    return value
                if value.active_execution_id != execution_id:
                    return value
                next_value = replace(value, active_execution_id=None)
            else:
                if (
                    value.active_execution_id == execution_id
                    and value.continuation == expected
                ):
                    return value
                if value.status is not SessionStatus.OPEN:
                    raise AIError(ErrorCode.SESSION_CONFLICT)
                if (
                    value.active_execution_id is not None
                    or value.continuation != expected
                ):
                    raise AIError(ErrorCode.SESSION_BUSY)
                next_value = replace(value, active_execution_id=execution_id)
            candidate = _projected_record(self, current, next_value)
            await _replace_checked(transaction, candidate, current.storage_version)
            if not release:
                sequence = await transaction.next_sequence(
                    self._timeline_sequence_key(session_id)
                )
                await transaction.insert_fact(
                    StoredFact(
                        self._timeline_stream(session_id),
                        sequence,
                        self._key("session", session_id),
                        "session_turn",
                        self._timeline_subject(execution_id),
                        None,
                        {"version": 1, "execution_id": execution_id},
                    )
                )
            return next_value

        if transaction is not None:
            return await mutate(transaction)
        try:
            return await self._store.mutate(mutate)
        except AIError as error:
            if error.code is ErrorCode.STORAGE_CONFLICT and not release:
                latest = await self.get(session_id, tenant_id=tenant_id)
                if latest is not None and latest.active_execution_id not in {
                    None,
                    execution_id,
                }:
                    raise AIError(ErrorCode.SESSION_BUSY) from error
            raise

    async def transition_status(
        self,
        session_id: str,
        *,
        tenant_id: str,
        expected: frozenset[SessionStatus],
        next_status: SessionStatus,
        closed_at: datetime | None = None,
        require_no_active: bool = False,
    ) -> SessionRecord:
        if tenant_id != self._tenant_id:
            raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)

        async def mutate(transaction: StateTransaction) -> SessionRecord:
            record = await transaction.get_record(self._key("session", session_id))
            if record is None:
                raise AIError(ErrorCode.STORAGE_NOT_FOUND)
            current = await self._decode(record, SessionRecord)
            if current.status not in expected:
                raise AIError(ErrorCode.SESSION_CONFLICT)
            if require_no_active and current.active_execution_id is not None:
                raise AIError(ErrorCode.SESSION_ACTIVE_EXECUTIONS)
            now = await transaction.now()
            next_value = replace(
                current,
                status=next_status,
                closed_at=closed_at,
                revision=current.revision + 1,
                updated_at=now,
            )
            await _replace_checked(
                transaction,
                _projected_record(self, record, next_value),
                record.storage_version,
            )
            await self._bump_list_generation(transaction, current.owner_principal_id)
            return next_value

        return await self._store.mutate(mutate)

    async def advance_continuation(
        self,
        session_id: str,
        *,
        tenant_id: str,
        execution_id: str,
        expected: ConversationCursor | None,
        next_cursor: ConversationCursor,
        history_quality: str | None = None,
    ) -> SessionRecord:
        if tenant_id != self._tenant_id:
            raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)

        return await self._store.mutate(
            lambda transaction: self.advance_continuation_in_transaction(
                transaction,
                session_id,
                tenant_id=tenant_id,
                execution_id=execution_id,
                expected=expected,
                next_cursor=next_cursor,
                history_quality=history_quality,
            )
        )

    async def get_in_transaction(
        self,
        transaction: StateTransaction,
        session_id: str,
        *,
        tenant_id: str,
    ) -> SessionRecord:
        if tenant_id != self._tenant_id:
            raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)
        record = await transaction.get_record(self._key("session", session_id))
        if record is None:
            raise AIError(ErrorCode.STORAGE_NOT_FOUND)
        return await self._decode(record, SessionRecord)

    async def _decode_history(self, record: StoredRecord) -> ConversationHistoryRecord:
        return _decode_enveloped_domain(record.data, ConversationHistoryRecord)

    async def advance_continuation_in_transaction(
        self,
        transaction: StateTransaction,
        session_id: str,
        *,
        tenant_id: str,
        execution_id: str,
        expected: ConversationCursor | None,
        next_cursor: ConversationCursor,
        release_execution: bool = False,
        history_quality: str | None = None,
    ) -> SessionRecord:
        if tenant_id != self._tenant_id:
            raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)

        record = await transaction.get_record(self._key("session", session_id))
        if record is None:
            raise AIError(ErrorCode.STORAGE_NOT_FOUND)
        current = await self._decode(record, SessionRecord)
        if current.continuation == next_cursor:
            return current
        if (
            current.active_execution_id != execution_id
            or current.continuation != expected
            or current.status
            not in {
                SessionStatus.OPEN,
                SessionStatus.CLOSING,
                SessionStatus.CLEANUP_REQUIRED,
            }
        ):
            raise AIError(ErrorCode.SESSION_BUSY)
        now = await transaction.now()
        next_value = replace(
            current,
            continuation=next_cursor,
            history_quality=(
                current.history_quality if history_quality is None else history_quality
            ),
            active_execution_id=None
            if release_execution
            else current.active_execution_id,
            revision=current.revision + 1,
            updated_at=now,
        )
        await _replace_checked(
            transaction,
            _projected_record(self, record, next_value),
            record.storage_version,
        )
        await self._bump_list_generation(transaction, current.owner_principal_id)
        return next_value

    async def complete_execution_in_transaction(
        self,
        transaction: StateTransaction,
        session_id: str,
        *,
        tenant_id: str,
        execution_id: str,
        expected: ConversationCursor | None,
        next_cursor: ConversationCursor,
        history_quality: str | None = None,
    ) -> SessionRecord:
        return await self.advance_continuation_in_transaction(
            transaction,
            session_id,
            tenant_id=tenant_id,
            execution_id=execution_id,
            expected=expected,
            next_cursor=next_cursor,
            release_execution=True,
            history_quality=history_quality,
        )


def _session_history_id(session_id: str, tenant_id: str) -> str:
    return canonical_sha256(
        {
            "kind": "conversation_history",
            "session_id": session_id,
            "tenant_id": tenant_id,
        }
    )


def _ensure_session_history(value: SessionRecord) -> SessionRecord:
    if value.history_id is not None:
        return value
    return replace(
        value,
        history_id=_session_history_id(value.session_id, value.tenant_id),
    )


def _new_session_history(value: SessionRecord) -> ConversationHistoryRecord:
    if value.history_id is None:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return ConversationHistoryRecord(
        history_id=value.history_id,
        session_id=value.session_id,
        tenant_id=value.tenant_id,
        parent_history_id=None,
        prefix_index_head_id=None,
        inherited_message_count=0,
    )


def _empty_conversation_transcript_head(
    history_id: str,
) -> TranscriptHeadRecord:
    return TranscriptHeadRecord(
        TranscriptOwnerDomain.CONVERSATION,
        history_id,
        0,
        0,
        HistoryQuality.COMPLETE,
    )


def _require_explicit_session_agent_id(value: SessionRecord) -> None:
    if value.agent_id is None:
        raise AIError(ErrorCode.AGENT_ID_INVALID)
    try:
        validate_agent_id(value.agent_id)
    except TypeError as error:
        raise AIError(ErrorCode.AGENT_ID_INVALID) from error


__all__ = [
    "ConversationHistoryRepositoryImpl",
    "SessionRepositoryImpl",
]
