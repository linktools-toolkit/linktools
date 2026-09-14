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
        raise RuntimeError(f"{path}: expected one replacement, found {count}: {old[:100]!r}")
    save(path, text.replace(old, new, 1))


path = "linktools-ai/src/linktools/ai/runtime/state/_repositories.py"
replace_once(
    path,
    '''    def _timeline_stream(self, session_id: str, kind: str) -> bytes:\n        return stream_digest(\n            self._namespace,\n            self._tenant_id,\n            self._domain.value,\n            kind,\n            [session_id],\n        )\n''',
    '''    def _timeline_stream(self, session_id: str) -> bytes:\n        return stream_digest(\n            self._namespace,\n            self._tenant_id,\n            self._domain.value,\n            "session_turn",\n            [session_id],\n        )\n\n    def _timeline_commit_key(self, session_id: str, sequence: int) -> bytes:\n        return self._key("session_turn_commit", [session_id, sequence])\n\n    def _stored_timeline_commit(\n        self, value: SessionTurnCommitRef\n    ) -> StoredRecord:\n        identity = [value.session_id, value.sequence]\n        return StoredRecord(\n            self._timeline_commit_key(value.session_id, value.sequence),\n            self._partition("session_turn_commit"),\n            self._scope("session_turn_commit", "session", value.session_id),\n            None,\n            "session_turn_commit",\n            sortable_identity(identity),\n            None,\n            0,\n            None,\n            0,\n            None,\n            {\n                "version": 1,\n                "session_id": value.session_id,\n                "sequence": value.sequence,\n                "execution_id": value.execution_id,\n                "start_message_index": value.start_message_index,\n                "end_message_index": value.end_message_index,\n            },\n        )\n''',
)
replace_once(
    path,
    '''    def _decode_timeline_commit(\n        self, session_id: str, fact: StoredFact\n    ) -> SessionTurnCommitRef:\n        if (\n            fact.kind != "session_turn_commit"\n            or fact.owner_key_digest != self._key("session", session_id)\n            or set(fact.data)\n            != {\n                "version",\n                "execution_id",\n                "start_message_index",\n                "end_message_index",\n            }\n            or fact.data.get("version") != 1\n            or not isinstance(fact.data.get("execution_id"), str)\n        ):\n            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)\n        execution_id = str(fact.data["execution_id"])\n        start = fact.data.get("start_message_index")\n        end = fact.data.get("end_message_index")\n        if (\n            isinstance(start, bool)\n            or not isinstance(start, int)\n            or isinstance(end, bool)\n            or not isinstance(end, int)\n            or fact.subject_digest != self._timeline_subject(execution_id)\n        ):\n            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)\n        try:\n            return SessionTurnCommitRef(\n                session_id, fact.sequence, execution_id, start, end\n            )\n        except ValueError as error:\n            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error\n''',
    '''    def _decode_timeline_commit(\n        self, session_id: str, sequence: int, record: StoredRecord\n    ) -> SessionTurnCommitRef:\n        if (\n            record.key_digest != self._timeline_commit_key(session_id, sequence)\n            or record.partition_digest != self._partition("session_turn_commit")\n            or record.scope_digest\n            != self._scope("session_turn_commit", "session", session_id)\n            or record.parent_digest is not None\n            or record.kind != "session_turn_commit"\n            or record.sort_key != sortable_identity([session_id, sequence])\n            or record.state is not None\n            or set(record.data)\n            != {\n                "version",\n                "session_id",\n                "sequence",\n                "execution_id",\n                "start_message_index",\n                "end_message_index",\n            }\n            or record.data.get("version") != 1\n            or record.data.get("session_id") != session_id\n            or record.data.get("sequence") != sequence\n            or not isinstance(record.data.get("execution_id"), str)\n        ):\n            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)\n        execution_id = str(record.data["execution_id"])\n        start = record.data.get("start_message_index")\n        end = record.data.get("end_message_index")\n        if (\n            isinstance(start, bool)\n            or not isinstance(start, int)\n            or isinstance(end, bool)\n            or not isinstance(end, int)\n        ):\n            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)\n        try:\n            return SessionTurnCommitRef(\n                session_id, sequence, execution_id, start, end\n            )\n        except ValueError as error:\n            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error\n''',
)
text = load(path).replace('self._timeline_stream(session_id, "session_turn")', 'self._timeline_stream(session_id)')
save(path, text)
replace_once(
    path,
    '''        facts = await self._store.read(\n            lambda transaction: transaction.list_facts(\n                FactQuery(\n                    self._timeline_stream(session_id, "session_turn_commit"),\n                    after_sequence=start_sequence - 1,\n                    limit=end_sequence - start_sequence,\n                )\n            )\n        )\n        return tuple(\n            self._decode_timeline_commit(session_id, fact)\n            for fact in facts\n            if fact.sequence < end_sequence\n        )\n''',
    '''        sequences = tuple(range(start_sequence, end_sequence))\n        keys = tuple(\n            self._timeline_commit_key(session_id, sequence)\n            for sequence in sequences\n        )\n        records = await self._store.read(\n            lambda transaction: transaction.get_records(keys)\n        )\n        return tuple(\n            self._decode_timeline_commit(session_id, sequence, records[key])\n            for sequence, key in zip(sequences, keys)\n            if key in records\n        )\n''',
)
replace_once(
    path,
    '''        commit_stream = self._timeline_stream(session_id, "session_turn_commit")\n        existing = await transaction.list_facts(\n            FactQuery(\n                commit_stream,\n                after_sequence=turn.sequence - 1,\n                limit=1,\n            )\n        )\n        if existing and existing[0].sequence == turn.sequence:\n            committed = self._decode_timeline_commit(session_id, existing[0])\n            if committed.execution_id != execution_id:\n                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)\n            if start_message_index is not None and (\n                committed.start_message_index != start_message_index\n                or committed.end_message_index != end_message_index\n            ):\n                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)\n            return committed\n''',
    '''        commit_key = self._timeline_commit_key(session_id, turn.sequence)\n        existing = await transaction.get_record(commit_key)\n        if existing is not None:\n            committed = self._decode_timeline_commit(\n                session_id, turn.sequence, existing\n            )\n            if committed.execution_id != execution_id:\n                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)\n            if start_message_index is not None and (\n                committed.start_message_index != start_message_index\n                or committed.end_message_index != end_message_index\n            ):\n                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)\n            return committed\n''',
)
replace_once(
    path,
    '''        await transaction.insert_fact(\n            StoredFact(\n                commit_stream,\n                turn.sequence,\n                self._key("session", session_id),\n                "session_turn_commit",\n                subject,\n                None,\n                {\n                    "version": 1,\n                    "execution_id": execution_id,\n                    "start_message_index": start_message_index,\n                    "end_message_index": end_message_index,\n                },\n            )\n        )\n''',
    '''        await transaction.insert_record(self._stored_timeline_commit(committed))\n''',
)

print("session timeline sparse-commit fix applied")
