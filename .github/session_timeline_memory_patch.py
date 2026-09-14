from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    file = Path(path)
    text = file.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: expected one replacement, found {count}")
    file.write_text(text.replace(old, new, 1), encoding="utf-8")


steps = "linktools-ai/src/linktools/ai/runtime/state/_steps.py"
replace_once(
    steps,
    '''    async def load_model_context(self, *, run_id: str) -> tuple[object, ...]:\n        snapshot = await self.latest_snapshot(run_id=run_id, include_interrupted=True)\n        return () if snapshot is None else tuple(snapshot.messages)\n\n\nclass StateStepArchive(StepStore):\n''',
    '''    def _session_snapshot(self, history_id: str) -> ContinuableSnapshot | None:\n        if self._runtime_domain is not RuntimeDomain.CONVERSATION:\n            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)\n        candidates: list[ContinuableSnapshot] = []\n        for run in self._runs.values():\n            if run.metadata.get("history_id") != history_id:\n                continue\n            snapshot = self.latest_snapshot_local(run.run_id)\n            if snapshot is not None:\n                candidates.append(snapshot)\n        if not candidates:\n            return None\n        return max(\n            candidates,\n            key=lambda value: (len(value.messages), value.timestamp, value.run_id),\n        )\n\n    async def session_message_count(\n        self,\n        history_id: str,\n        *,\n        tenant_id: str,\n    ) -> int:\n        del tenant_id\n        snapshot = self._session_snapshot(history_id)\n        if snapshot is None:\n            raise AIError(ErrorCode.SESSION_HISTORY_UNAVAILABLE)\n        return len(snapshot.messages)\n\n    async def iter_session_message_range(\n        self,\n        history_id: str,\n        *,\n        tenant_id: str,\n        start: int,\n        end: int,\n    ) -> AsyncIterator[object]:\n        del tenant_id\n        if start < 0 or end < start:\n            raise AIError(ErrorCode.STORAGE_CONFLICT)\n        snapshot = self._session_snapshot(history_id)\n        if snapshot is None:\n            raise AIError(ErrorCode.SESSION_HISTORY_UNAVAILABLE)\n        if end > len(snapshot.messages):\n            raise AIError(ErrorCode.SESSION_HISTORY_UNAVAILABLE)\n        for message in snapshot.messages[start:end]:\n            yield message\n\n    async def load_model_context(self, *, run_id: str) -> tuple[object, ...]:\n        snapshot = await self.latest_snapshot(run_id=run_id, include_interrupted=True)\n        return () if snapshot is None else tuple(snapshot.messages)\n\n\nclass StateStepArchive(StepStore):\n''',
)

local = "linktools-ai/src/linktools/ai/runtime/_local.py"
replace_once(
    local,
    '''        if not isinstance(conversation_archive, StateStepArchive):\n            if session.continuation == intent.next_cursor:\n                return\n            if session.status is SessionStatus.CLOSED:\n                raise AIError(ErrorCode.SESSION_CONFLICT)\n            if session.active_execution_id != checkpoint.execution_id:\n                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)\n            if session.continuation != intent.expected_cursor:\n                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)\n            await self._conversation.sessions.advance_continuation(\n                intent.session_id,\n                tenant_id=checkpoint.tenant_id,\n                execution_id=checkpoint.execution_id,\n                expected=intent.expected_cursor,\n                next_cursor=intent.next_cursor,\n            )\n            return\n        history_id = session.history_id or intent.next_cursor.history_id\n        if history_id is None:\n            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)\n        end_message_index = await conversation_archive.session_message_count(\n            history_id,\n            tenant_id=checkpoint.tenant_id,\n        )\n''',
    '''        session_message_count = getattr(\n            conversation_archive, "session_message_count", None\n        )\n        if session_message_count is None:\n            if session.continuation == intent.next_cursor:\n                return\n            if session.status is SessionStatus.CLOSED:\n                raise AIError(ErrorCode.SESSION_CONFLICT)\n            if session.active_execution_id != checkpoint.execution_id:\n                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)\n            if session.continuation != intent.expected_cursor:\n                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)\n            await self._conversation.sessions.advance_continuation(\n                intent.session_id,\n                tenant_id=checkpoint.tenant_id,\n                execution_id=checkpoint.execution_id,\n                expected=intent.expected_cursor,\n                next_cursor=intent.next_cursor,\n            )\n            return\n        history_id = session.history_id or intent.next_cursor.history_id\n        if history_id is None:\n            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)\n        end_message_index = await session_message_count(\n            history_id,\n            tenant_id=checkpoint.tenant_id,\n        )\n''',
)

timeline_test = "tests/ai/test_session_timeline.py"
replace_once(
    timeline_test,
    '''async def _materialize_conversation(state: RuntimeState) -> tuple[str, int]:\n''',
    '''async def _materialize_conversation(\n    state: RuntimeState, history_id: str\n) -> tuple[str, int]:\n''',
)
replace_once(
    timeline_test,
    '''            metadata={"agent_name": "agent"},\n''',
    '''            metadata={"agent_name": "agent", "history_id": history_id},\n''',
)
replace_once(
    timeline_test,
    '''        run_id, message_count = await _materialize_conversation(state)\n''',
    '''        assert created.history_id is not None\n        run_id, message_count = await _materialize_conversation(\n            state, created.history_id\n        )\n''',
)

recovery_test = "tests/ai/test_session_timeline_recovery.py"
replace_once(
    recovery_test,
    '''from linktools.ai.runtime.state._steps import StateStepArchive\n''',
    '''''',
)
replace_once(
    recovery_test,
    '''        archive = state.steps.read_store(RuntimeDomain.CONVERSATION)\n        assert isinstance(archive, StateStepArchive)\n''',
    '''        archive = state.steps.read_store(RuntimeDomain.CONVERSATION)\n''',
)
replace_once(
    recovery_test,
    '''            metadata={"history_id": session.history_id},\n''',
    '''            metadata={"history_id": session.history_id},\n''',
)

print("session timeline in-memory compatibility patch applied")
