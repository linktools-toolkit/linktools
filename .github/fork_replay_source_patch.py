#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import re


path = Path("linktools-ai/src/linktools/ai/runtime/state/_repositories.py")
text = path.read_text(encoding="utf-8")
pattern = re.compile(
    r"    async def _replay_fork_in_transaction\(.*?(?=\n\n    async def _visible_history_count_in_transaction\()",
    re.DOTALL,
)
replacement = '''    async def _replay_fork_in_transaction(
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
'''
text, count = pattern.subn(replacement, text, count=1)
if count != 1:
    raise RuntimeError(f"fork replay method match count: {count}")
path.write_text(text, encoding="utf-8")

path = Path("tests/ai/test_runtime_storage_io_session.py")
text = path.read_text(encoding="utf-8")
old = "        assert _parameter_count(batched[0]) == 3\n"
if text.count(old) != 1:
    raise RuntimeError("fork replay IO assertion did not match exactly once")
path.write_text(
    text.replace(old, "        assert _parameter_count(batched[0]) == 4\n", 1),
    encoding="utf-8",
)
