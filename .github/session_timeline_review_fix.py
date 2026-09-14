#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def replace_once(path: str, old: str, new: str) -> None:
    target = ROOT / path
    text = target.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: expected one replacement, found {count}")
    target.write_text(text.replace(old, new), encoding="utf-8")


def append_once(path: str, marker: str, content: str) -> None:
    target = ROOT / path
    text = target.read_text(encoding="utf-8")
    if marker in text:
        raise RuntimeError(f"{path}: marker already exists")
    target.write_text(text.rstrip() + "\n\n" + content.rstrip() + "\n", encoding="utf-8")


replace_once(
    "linktools-ai/src/linktools/ai/runtime/_session.py",
    '''        if isinstance(message, ModelRequest):
            projected = tuple(
                item for item in projected if item.item_kind == "tool_result"
            )
        elif not isinstance(message, ModelResponse):
            continue
''',
    '''        if isinstance(message, ModelRequest):
            projected = tuple(
                item for item in projected if item.item_kind == "tool_result"
            )
        elif isinstance(message, ModelResponse):
            projected = tuple(
                item
                for item in projected
                if item.item_kind in {"assistant", "thinking", "tool_call"}
            )
        else:
            continue
''',
)

replace_once(
    "linktools-ai/src/linktools/ai/runtime/_local.py",
    '''        session_message_count = getattr(
            conversation_archive, "session_message_count", None
        )
        if session_message_count is None:
            if session.continuation == intent.next_cursor:
                return
            if session.status is SessionStatus.CLOSED:
                raise AIError(ErrorCode.SESSION_CONFLICT)
            if session.active_execution_id != checkpoint.execution_id:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if session.continuation != intent.expected_cursor:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            await self._conversation.sessions.advance_continuation(
                intent.session_id,
                tenant_id=checkpoint.tenant_id,
                execution_id=checkpoint.execution_id,
                expected=intent.expected_cursor,
                next_cursor=intent.next_cursor,
            )
            return
        history_id = session.history_id or intent.next_cursor.history_id
        if history_id is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        end_message_index = await session_message_count(
            history_id,
            tenant_id=checkpoint.tenant_id,
        )
''',
    '''        history_id = session.history_id or intent.next_cursor.history_id
        if history_id is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        end_message_index = await self._steps.session_message_count(
            history_id,
            tenant_id=checkpoint.tenant_id,
        )
''',
)
replace_once(
    "linktools-ai/src/linktools/ai/runtime/_local.py",
    '''        start_message_index = (
            0
            if intent.expected_cursor is None
            else intent.expected_cursor.message_count
        )
        if start_message_index is None or end_message_index <= start_message_index:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
''',
    '''        if intent.expected_cursor is None:
            start_message_index = 0
        else:
            start_message_index = intent.expected_cursor.message_count
            if start_message_index is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if end_message_index <= start_message_index:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
''',
)

replace_once(
    "linktools-ai/src/linktools/ai/runtime/state/_steps.py",
    '''    async def session_message_count(
        self,
        history_id: str,
        *,
        tenant_id: str,
    ) -> int:
        archive = self._archives.get(RuntimeDomain.CONVERSATION)
        if not isinstance(archive, StateStepArchive):
            raise AIError(ErrorCode.STORAGE_DEPENDENCY_NOT_READY)
        return await archive.transcript_repository.history_message_count(
            history_id,
            tenant_id=tenant_id,
        )
''',
    '''    async def session_message_count(
        self,
        history_id: str,
        *,
        tenant_id: str,
    ) -> int:
        archive = self._archives.get(RuntimeDomain.CONVERSATION)
        if isinstance(archive, StateStepArchive):
            return await archive.transcript_repository.history_message_count(
                history_id,
                tenant_id=tenant_id,
            )
        if isinstance(archive, InMemoryStepArchive):
            return await archive.session_message_count(
                history_id,
                tenant_id=tenant_id,
            )
        raise AIError(ErrorCode.STORAGE_DEPENDENCY_NOT_READY)
''',
)

replace_once(
    "linktools-ai/src/linktools/ai/runtime/state/_retention.py",
    '''    async def release_session(
        self,
        session_id: str,
        *,
        tenant_id: str,
        continuation: ConversationCursor | None,
    ) -> None:
        if RuntimeDomain.CONVERSATION not in self._transient_domains:
            return
        if continuation is not None:
            await self._steps.release_archive(
                RuntimeDomain.CONVERSATION, continuation.step_run_id
            )
        await self._objects.release_object_scope(
            RuntimeDomain.CONVERSATION, owner_scope=f"session:{session_id}"
        )
''',
    '''    async def release_session(
        self,
        session_id: str,
        *,
        tenant_id: str,
        continuation: ConversationCursor | None,
    ) -> None:
        # Forked sessions may still reference transient conversation state.
        del session_id, tenant_id, continuation
''',
)

replace_once(
    "linktools-ai/src/linktools/ai/runtime/state/_repositories.py",
    '''    sequence_key,
    sortable_identity,
    stream_digest,
''',
    '''    sequence_key,
    sortable_identity,
    stream_digest,
    subject_digest,
''',
)
replace_once(
    "linktools-ai/src/linktools/ai/runtime/state/_repositories.py",
    '''    @staticmethod
    def _timeline_subject(execution_id: str) -> bytes:
        return hashlib.sha256(canonical_json_bytes(execution_id)).digest()
''',
    '''    @staticmethod
    def _timeline_subject(execution_id: str) -> bytes:
        return subject_digest(execution_id)
''',
)
replace_once(
    "linktools-ai/src/linktools/ai/runtime/state/_repositories.py",
    '''                self._timeline_stream(session_id),
                subject_digest=subject,
                latest=True,
''',
    '''                self._timeline_stream(session_id),
                subject_digest=subject,
''',
)
replace_once(
    "linktools-ai/src/linktools/ai/runtime/state/_repositories.py",
    '''        execution_id: str,
        start_message_index: int | None,
        end_message_index: int,
''',
    '''        execution_id: str,
        start_message_index: int,
        end_message_index: int,
''',
)
replace_once(
    "linktools-ai/src/linktools/ai/runtime/state/_repositories.py",
    '''            if committed.end_message_index != end_message_index or (
                start_message_index is not None
                and committed.start_message_index != start_message_index
            ):
''',
    '''            if (
                committed.end_message_index != end_message_index
                or committed.start_message_index != start_message_index
            ):
''',
)
replace_once(
    "linktools-ai/src/linktools/ai/runtime/state/_repositories.py",
    '''        if (
            start_message_index is None
            or start_message_index < 0
            or end_message_index <= start_message_index
        ):
''',
    '''        if start_message_index < 0 or end_message_index <= start_message_index:
''',
)

replace_once(
    "linktools-ai/src/linktools/ai/runtime/state/_contracts.py",
    '''        execution_id: str,
        start_message_index: int | None,
        end_message_index: int,
    ) -> SessionTurnCommitRef: ...
''',
    '''        execution_id: str,
        start_message_index: int,
        end_message_index: int,
    ) -> SessionTurnCommitRef: ...
''',
)

replace_once(
    "linktools-ai/src/linktools/ai/runtime/state/_commands.py",
    '''def _timeline_turn_message_range(
    history: ConversationHistoryRecord,
    prepared: PreparedStepSnapshotBatch,
    expected_cursor: ConversationCursor | None,
) -> tuple[int | None, int]:
    if not prepared.snapshots:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    local_start = min(
        (chunk.first_message_index for chunk in prepared.snapshots[0].chunks),
        default=None,
    )
    end = history.inherited_message_count + prepared.target_transcript_message_count
    if local_start is not None:
        start: int | None = history.inherited_message_count + local_start
    elif expected_cursor is None:
        if history.inherited_message_count != 0:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        start = 0
    else:
        start = expected_cursor.message_count
    if start is not None and end <= start:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return start, end
''',
    '''def _timeline_turn_message_range(
    history: ConversationHistoryRecord,
    prepared: PreparedStepSnapshotBatch,
    expected_cursor: ConversationCursor | None,
) -> tuple[int, int]:
    if not prepared.snapshots:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    local_start = min(
        (chunk.first_message_index for chunk in prepared.snapshots[0].chunks),
        default=None,
    )
    end = history.inherited_message_count + prepared.target_transcript_message_count
    if local_start is not None:
        start = history.inherited_message_count + local_start
    elif expected_cursor is None:
        if history.inherited_message_count != 0:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        start = 0
    elif expected_cursor.message_count is None:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    else:
        start = expected_cursor.message_count
    if end <= start:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return start, end
''',
)
replace_once(
    "linktools-ai/src/linktools/ai/runtime/state/_commands.py",
    '''        timeline_range: tuple[int | None, int] | None = None
''',
    '''        timeline_range: tuple[int, int] | None = None
''',
)

replace_once(
    "tests/ai/test_session_timeline_recovery.py",
    '''        backend._steps = object()
''',
    '''        backend._steps = state.steps
''',
)

replace_once(
    "tests/ai/test_session_timeline.py",
    '''from linktools.ai.runtime._session import DefaultSessionService
from linktools.ai.runtime.service_api import ExecutionView
''',
    '''import linktools.ai.runtime._session as session_module
from linktools.ai.runtime._session import DefaultSessionService
from linktools.ai.runtime.service_api import ExecutionView, SessionHistoryItem
''',
)
append_once(
    "tests/ai/test_session_timeline.py",
    "test_timeline_projection_ignores_unrecognized_response_items",
    '''def test_timeline_projection_ignores_unrecognized_response_items(monkeypatch) -> None:
    def project(_message):
        return (
            SessionHistoryItem(1, "assistant", "visible"),
            SessionHistoryItem(2, "provider_internal", {"secret": "hidden"}),
        )

    monkeypatch.setattr(session_module, "project_session_history_message", project)
    response = ModelResponse(parts=[TextPart(content="visible")])

    items = session_module._timeline_items((response,))

    assert [item.item_kind for item in items] == ["assistant"]
    assert [item.content for item in items] == ["visible"]
''',
)

append_once(
    "tests/ai/test_session_timeline_runtime.py",
    "test_in_memory_fork_survives_parent_close",
    '''@pytest.mark.asyncio
async def test_in_memory_fork_survives_parent_close(tmp_path: Path) -> None:
    workspace = _runtime_usage_workspace(tmp_path / "workspace-fork")

    async with Runtime.open(
        workspace,
        models=_RuntimeUsageModels(),  # type: ignore[arg-type]
        state=RuntimeState.in_memory(),
    ) as runtime:
        parent = await runtime.agent("default").create_session("parent")
        parent_turn = await parent.run("before fork", timeout_seconds=10)
        assert parent_turn.status is ExecutionStatus.SUCCEEDED

        child = await parent.fork("child")
        await parent.close()
        child_turn = await child.run("after fork", timeout_seconds=10)
        assert child_turn.status is ExecutionStatus.SUCCEEDED

        page = await child.timeline()
        assert [turn.execution_id for turn in page.items] == [
            parent_turn.execution_id,
            child_turn.execution_id,
        ]
        assert [turn.user_input["prompt"] for turn in page.items] == [
            {"kind": "text", "text": "before fork"},
            {"kind": "text", "text": "after fork"},
        ]
        assert all(turn.conversation_committed for turn in page.items)
''',
)

replace_once(
    "tests/ai/test_session_timeline_storage.py",
    '''from linktools.ai.core import SessionStatus
''',
    '''from linktools.ai.core import SessionStatus
from linktools.ai.errors import AIError, ErrorCode
''',
)
replace_once(
    "tests/ai/test_session_timeline_storage.py",
    '''from linktools.ai.runtime.state._contracts import (
    ConversationCursor,
    ConversationHistoryRecord,
    SessionRecord,
)
''',
    '''from linktools.ai.runtime.state._contracts import (
    ConversationCursor,
    ConversationHistoryRecord,
    SessionRecord,
)
from linktools.ai.runtime.state._store import StoredFact
''',
)
append_once(
    "tests/ai/test_session_timeline_storage.py",
    "test_timeline_commit_rejects_duplicate_admission_fact",
    '''@pytest.mark.asyncio
async def test_timeline_commit_rejects_duplicate_admission_fact() -> None:
    state = RuntimeState.in_memory()
    await state.initialize(namespace="session-timeline-duplicate", tenant_id="tenant")
    try:
        repository = state.conversation.sessions
        await repository.create(_session())
        await repository.admit_execution(
            "session",
            tenant_id="tenant",
            execution_id="duplicate",
            expected=None,
        )

        async def duplicate(transaction) -> None:
            sequence = await transaction.next_sequence(
                repository._timeline_sequence_key("session")
            )
            await transaction.insert_fact(
                StoredFact(
                    repository._timeline_stream("session"),
                    sequence,
                    repository._key("session", "session"),
                    "session_turn",
                    repository._timeline_subject("duplicate"),
                    None,
                    {"version": 1, "execution_id": "duplicate"},
                )
            )

        await repository.state_store.mutate(duplicate)

        async def commit(transaction) -> None:
            await repository.commit_timeline_turn_in_transaction(
                transaction,
                "session",
                tenant_id="tenant",
                execution_id="duplicate",
                start_message_index=0,
                end_message_index=2,
            )

        with pytest.raises(AIError) as captured:
            await repository.state_store.mutate(commit)
        assert captured.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
    finally:
        await state.close()
''',
)

print("session timeline review fixes applied")
