from __future__ import annotations

from pathlib import Path


def load(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def save(path: str, text: str) -> None:
    Path(path).write_text(text, encoding="utf-8")


def replace_once(path: str, old: str, new: str) -> None:
    text = load(path)
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: expected one replacement, found {count}: {old[:80]!r}")
    save(path, text.replace(old, new, 1))


def insert_before_once(path: str, marker: str, content: str) -> None:
    replace_once(path, marker, content + marker)


def insert_after_once(path: str, marker: str, content: str) -> None:
    replace_once(path, marker, marker + content)


# ---------------------------------------------------------------------------
# Session repository: one compact local turn index plus one sparse committed
# conversation-range index. Both are facts owned by the Session record.
# ---------------------------------------------------------------------------
path = "linktools-ai/src/linktools/ai/runtime/state/_repositories.py"
replace_once(
    path,
    '''    SessionRecord,\n    ToolOperationAdmission,\n''',
    '''    SessionRecord,\n    SessionTurnCommitRef,\n    SessionTurnRef,\n    ToolOperationAdmission,\n''',
)
insert_after_once(
    path,
    '''            value_type=SessionRecord,\n        )\n''',
    '''\n\n    def _timeline_sequence_key(self, session_id: str) -> bytes:\n        return sequence_key(\n            self._namespace,\n            self._tenant_id,\n            self._domain.value,\n            "session_turn",\n            [session_id],\n        )\n\n    def _timeline_stream(self, session_id: str, kind: str) -> bytes:\n        return stream_digest(\n            self._namespace,\n            self._tenant_id,\n            self._domain.value,\n            kind,\n            [session_id],\n        )\n\n    @staticmethod\n    def _timeline_subject(execution_id: str) -> bytes:\n        return hashlib.sha256(canonical_json_bytes(execution_id)).digest()\n\n    def _decode_timeline_turn(\n        self, session_id: str, fact: StoredFact\n    ) -> SessionTurnRef:\n        if (\n            fact.kind != "session_turn"\n            or fact.owner_key_digest != self._key("session", session_id)\n            or set(fact.data) != {"version", "execution_id"}\n            or fact.data.get("version") != 1\n            or not isinstance(fact.data.get("execution_id"), str)\n        ):\n            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)\n        execution_id = str(fact.data["execution_id"])\n        if fact.subject_digest != self._timeline_subject(execution_id):\n            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)\n        return SessionTurnRef(session_id, fact.sequence, execution_id)\n\n    def _decode_timeline_commit(\n        self, session_id: str, fact: StoredFact\n    ) -> SessionTurnCommitRef:\n        if (\n            fact.kind != "session_turn_commit"\n            or fact.owner_key_digest != self._key("session", session_id)\n            or set(fact.data)\n            != {\n                "version",\n                "execution_id",\n                "start_message_index",\n                "end_message_index",\n            }\n            or fact.data.get("version") != 1\n            or not isinstance(fact.data.get("execution_id"), str)\n        ):\n            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)\n        execution_id = str(fact.data["execution_id"])\n        start = fact.data.get("start_message_index")\n        end = fact.data.get("end_message_index")\n        if (\n            isinstance(start, bool)\n            or not isinstance(start, int)\n            or isinstance(end, bool)\n            or not isinstance(end, int)\n            or fact.subject_digest != self._timeline_subject(execution_id)\n        ):\n            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)\n        try:\n            return SessionTurnCommitRef(\n                session_id, fact.sequence, execution_id, start, end\n            )\n        except ValueError as error:\n            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error\n''',
)
replace_once(
    path,
    '''            inherited = source_history.inherited_message_count + local_messages\n            child = ConversationHistoryRecord(\n''',
    '''            inherited = source_history.inherited_message_count + local_messages\n            turn_head = await transaction.get_sequence(\n                self._timeline_sequence_key(source_session_id)\n            )\n            if source.active_execution_id is not None:\n                if turn_head < 1:\n                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)\n                timeline_cutoff = turn_head - 1\n            else:\n                timeline_cutoff = turn_head\n            if timeline_cutoff > 0:\n                timeline_parent = source.session_id\n                timeline_parent_cutoff = timeline_cutoff\n            else:\n                timeline_parent = source.timeline_parent_session_id\n                timeline_parent_cutoff = source.timeline_parent_turn_sequence\n            child = ConversationHistoryRecord(\n''',
)
replace_once(
    path,
    '''                history_quality="complete",\n                continuation=(\n''',
    '''                history_quality="complete",\n                timeline_parent_session_id=timeline_parent,\n                timeline_parent_turn_sequence=timeline_parent_cutoff,\n                continuation=(\n''',
)
insert_before_once(
    path,
    '''    async def list(\n        self, *, tenant_id: str, owner_principal_id: str | None = None\n    ) -> tuple[SessionRecord, ...]:\n''',
    '''    async def timeline_head(self, session_id: str, *, tenant_id: str) -> int:\n        _require_repository_tenant(tenant_id, self._tenant_id)\n        return await self._store.read(\n            lambda transaction: transaction.get_sequence(\n                self._timeline_sequence_key(session_id)\n            )\n        )\n\n    async def list_timeline_turns(\n        self,\n        session_id: str,\n        *,\n        tenant_id: str,\n        start_sequence: int,\n        end_sequence: int,\n    ) -> tuple[SessionTurnRef, ...]:\n        _require_repository_tenant(tenant_id, self._tenant_id)\n        if start_sequence < 1 or end_sequence < start_sequence:\n            raise ValueError("session timeline range is invalid")\n        if start_sequence == end_sequence:\n            return ()\n        facts = await self._store.read(\n            lambda transaction: transaction.list_facts(\n                FactQuery(\n                    self._timeline_stream(session_id, "session_turn"),\n                    after_sequence=start_sequence - 1,\n                    limit=end_sequence - start_sequence,\n                )\n            )\n        )\n        selected = tuple(\n            fact for fact in facts if fact.sequence < end_sequence\n        )\n        if tuple(fact.sequence for fact in selected) != tuple(\n            range(start_sequence, end_sequence)\n        ):\n            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)\n        return tuple(\n            self._decode_timeline_turn(session_id, fact) for fact in selected\n        )\n\n    async def list_timeline_commits(\n        self,\n        session_id: str,\n        *,\n        tenant_id: str,\n        start_sequence: int,\n        end_sequence: int,\n    ) -> tuple[SessionTurnCommitRef, ...]:\n        _require_repository_tenant(tenant_id, self._tenant_id)\n        if start_sequence < 1 or end_sequence < start_sequence:\n            raise ValueError("session timeline range is invalid")\n        if start_sequence == end_sequence:\n            return ()\n        facts = await self._store.read(\n            lambda transaction: transaction.list_facts(\n                FactQuery(\n                    self._timeline_stream(session_id, "session_turn_commit"),\n                    after_sequence=start_sequence - 1,\n                    limit=end_sequence - start_sequence,\n                )\n            )\n        )\n        return tuple(\n            self._decode_timeline_commit(session_id, fact)\n            for fact in facts\n            if fact.sequence < end_sequence\n        )\n\n    async def commit_timeline_turn_in_transaction(\n        self,\n        transaction: StateTransaction,\n        session_id: str,\n        *,\n        tenant_id: str,\n        execution_id: str,\n        start_message_index: int | None,\n        end_message_index: int,\n    ) -> SessionTurnCommitRef:\n        _require_repository_tenant(tenant_id, self._tenant_id)\n        subject = self._timeline_subject(execution_id)\n        turns = await transaction.list_facts(\n            FactQuery(\n                self._timeline_stream(session_id, "session_turn"),\n                subject_digest=subject,\n                latest=True,\n            )\n        )\n        if len(turns) != 1:\n            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)\n        turn = self._decode_timeline_turn(session_id, turns[0])\n        if turn.execution_id != execution_id:\n            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)\n        commit_stream = self._timeline_stream(session_id, "session_turn_commit")\n        existing = await transaction.list_facts(\n            FactQuery(\n                commit_stream,\n                after_sequence=turn.sequence - 1,\n                limit=1,\n            )\n        )\n        if existing and existing[0].sequence == turn.sequence:\n            committed = self._decode_timeline_commit(session_id, existing[0])\n            if committed.execution_id != execution_id:\n                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)\n            if start_message_index is not None and (\n                committed.start_message_index != start_message_index\n                or committed.end_message_index != end_message_index\n            ):\n                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)\n            return committed\n        if (\n            start_message_index is None\n            or start_message_index < 0\n            or end_message_index <= start_message_index\n        ):\n            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)\n        committed = SessionTurnCommitRef(\n            session_id,\n            turn.sequence,\n            execution_id,\n            start_message_index,\n            end_message_index,\n        )\n        await transaction.insert_fact(\n            StoredFact(\n                commit_stream,\n                turn.sequence,\n                self._key("session", session_id),\n                "session_turn_commit",\n                subject,\n                None,\n                {\n                    "version": 1,\n                    "execution_id": execution_id,\n                    "start_message_index": start_message_index,\n                    "end_message_index": end_message_index,\n                },\n            )\n        )\n        return committed\n\n''',
)
replace_once(
    path,
    '''            candidate = _projected_record(self, current, next_value)\n            await _replace_checked(transaction, candidate, current.storage_version)\n            return next_value\n''',
    '''            candidate = _projected_record(self, current, next_value)\n            await _replace_checked(transaction, candidate, current.storage_version)\n            if not release:\n                sequence = await transaction.next_sequence(\n                    self._timeline_sequence_key(session_id)\n                )\n                await transaction.insert_fact(\n                    StoredFact(\n                        self._timeline_stream(session_id, "session_turn"),\n                        sequence,\n                        self._key("session", session_id),\n                        "session_turn",\n                        self._timeline_subject(execution_id),\n                        None,\n                        {"version": 1, "execution_id": execution_id},\n                    )\n                )\n            return next_value\n''',
)

path = "linktools-ai/src/linktools/ai/runtime/state/_commands.py"
replace_once(
    path,
    '''                        await self._promote_history_in_transaction(\n                            conversation_transaction,\n                            session,\n                            prepared_conversation[0],\n                        )\n                        await self._conversation.complete_execution_in_transaction(\n''',
    '''                        history = await self._promote_history_in_transaction(\n                            conversation_transaction,\n                            session,\n                            prepared_conversation[0],\n                        )\n                        await self._conversation.complete_execution_in_transaction(\n''',
)
needle = '''                            history_quality="complete",\n                        )\n'''
addition = '''                            history_quality="complete",\n                        )\n                        local_start = min(\n                            (\n                                chunk.first_message_index\n                                for chunk in prepared_conversation[0].chunks\n                            ),\n                            default=None,\n                        )\n                        await self._conversation.commit_timeline_turn_in_transaction(\n                            conversation_transaction,\n                            session_id,\n                            tenant_id=commit.execution.tenant_id,\n                            execution_id=commit.execution.execution_id,\n                            start_message_index=(\n                                None\n                                if local_start is None\n                                else history.inherited_message_count + local_start\n                            ),\n                            end_message_index=(\n                                history.inherited_message_count\n                                + prepared_conversation.target_transcript_message_count\n                            ),\n                        )\n'''
text = load(path)
if text.count(needle) < 1:
    raise RuntimeError("terminal same-group continuation marker missing")
text = text.replace(needle, addition, 1)
save(path, text)
replace_once(
    path,
    '''                await self._promote_history_in_transaction(\n                    conversation_transaction,\n                    session,\n                    prepared_conversation[0],\n                )\n                await self._conversation.advance_continuation_in_transaction(\n''',
    '''                history = await self._promote_history_in_transaction(\n                    conversation_transaction,\n                    session,\n                    prepared_conversation[0],\n                )\n                await self._conversation.advance_continuation_in_transaction(\n''',
)
needle = '''                    history_quality="complete",\n                )\n\n            if _same_group(conversation_stores):\n'''
replacement = '''                    history_quality="complete",\n                )\n                local_start = min(\n                    (\n                        chunk.first_message_index\n                        for chunk in prepared_conversation[0].chunks\n                    ),\n                    default=None,\n                )\n                await self._conversation.commit_timeline_turn_in_transaction(\n                    conversation_transaction,\n                    session_id,\n                    tenant_id=commit.execution.tenant_id,\n                    execution_id=commit.execution.execution_id,\n                    start_message_index=(\n                        None\n                        if local_start is None\n                        else history.inherited_message_count + local_start\n                    ),\n                    end_message_index=(\n                        history.inherited_message_count\n                        + prepared_conversation.target_transcript_message_count\n                    ),\n                )\n\n            if _same_group(conversation_stores):\n'''
replace_once(path, needle, replacement)

print("session timeline patch part 2 applied")
