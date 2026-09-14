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


path = "linktools-ai/src/linktools/ai/runtime/_session.py"
replace_once(path, "from typing import Protocol\n", "from typing import Protocol, cast\n")
replace_once(
    path,
    "from pydantic_ai.messages import ModelRequest, ModelResponse\n",
    "from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse\n",
)
replace_once(
    path,
    '''    SessionRecord,\n    SessionTurnRef,\n)\n''',
    '''    SessionRecord,\n    SessionTurnCommitRef,\n    SessionTurnRef,\n)\n''',
)
replace_once(
    path,
    "def _timeline_items(messages: tuple[object, ...]) -> tuple[SessionTurnItem, ...]:\n",
    "def _timeline_items(messages: tuple[ModelMessage, ...]) -> tuple[SessionTurnItem, ...]:\n",
)
replace_once(
    path,
    '''    async def iter_session_messages(\n        self,\n        history_id: str,\n        *,\n        tenant_id: str,\n    ) -> AsyncIterator[object]: ...\n\n    async def load_session_model_context(\n''',
    '''    async def iter_session_messages(\n        self,\n        history_id: str,\n        *,\n        tenant_id: str,\n    ) -> AsyncIterator[object]: ...\n\n    async def iter_session_message_range(\n        self,\n        history_id: str,\n        *,\n        tenant_id: str,\n        start: int,\n        end: int,\n    ) -> AsyncIterator[ModelMessage]: ...\n\n    async def load_session_model_context(\n''',
)
replace_once(
    path,
    '''    ) -> Page[SessionTurn]:\n        if not 1 <= limit <= 200:\n            raise AIError(ErrorCode.PAGE_LIMIT_INVALID)\n''',
    '''    ) -> Page[SessionTurn]:\n        if (\n            isinstance(limit, bool)\n            or not isinstance(limit, int)\n            or not 1 <= limit <= 200\n        ):\n            raise AIError(ErrorCode.PAGE_LIMIT_INVALID)\n''',
)
replace_once(
    path,
    '''            commits: dict[tuple[str, int], object] = {}\n            messages: dict[str, tuple[object, ...]] = {}\n''',
    '''            commits: dict[tuple[str, int], SessionTurnCommitRef] = {}\n            messages: dict[str, tuple[ModelMessage, ...]] = {}\n''',
)
replace_once(
    path,
    '''                loaded = tuple(\n                    [\n                        item\n                        async for item in self._transcript_store.iter_session_message_range(\n                            record.history_id,\n                            tenant_id=record.tenant_id,\n                            start=range_start,\n                            end=range_end,\n                        )\n                    ]\n                )\n''',
    '''                loaded = cast(\n                    tuple[ModelMessage, ...],\n                    tuple(\n                        [\n                            item\n                            async for item in self._transcript_store.iter_session_message_range(\n                                record.history_id,\n                                tenant_id=record.tenant_id,\n                                start=range_start,\n                                end=range_end,\n                            )\n                        ]\n                    ),\n                )\n''',
)

path = "linktools-ai/src/linktools/ai/runtime/state/_commands.py"
replace_once(
    path,
    '''            else:\n                await self._materialize_prepared_snapshot_with_reconciliation(\n                    self._conversation_steps,\n                    conversation_run,\n                    prepared_conversation,\n                )\n                for attempt in range(2):\n                    try:\n                        await self._conversation.advance_continuation(\n                            session_id,\n                            tenant_id=commit.execution.tenant_id,\n                            execution_id=commit.execution.execution_id,\n                            expected=expected_cursor,\n                            next_cursor=next_cursor,\n                        )\n                        break\n                    except AIError as error:\n                        if error.code is not ErrorCode.STORAGE_COMMIT_UNKNOWN or attempt == 1:\n                            raise\n                        session = await self._conversation.get(\n                            session_id,\n                            tenant_id=commit.execution.tenant_id,\n                        )\n                        if session is not None and session.continuation == next_cursor:\n                            break\n\n        execution_stores = [self._execution.state_store]\n''',
    '''            else:\n                await self._materialize_prepared_snapshot_with_reconciliation(\n                    self._conversation_steps,\n                    conversation_run,\n                    prepared_conversation,\n                )\n\n                async def commit_conversation_state(\n                    transaction: StateTransaction,\n                ) -> None:\n                    session = await self._conversation.get_in_transaction(\n                        transaction,\n                        session_id,\n                        tenant_id=commit.execution.tenant_id,\n                    )\n                    history = await self._promote_history_in_transaction(\n                        transaction,\n                        session,\n                        prepared_conversation[0],\n                    )\n                    await self._conversation.advance_continuation_in_transaction(\n                        transaction,\n                        session_id,\n                        tenant_id=commit.execution.tenant_id,\n                        execution_id=commit.execution.execution_id,\n                        expected=expected_cursor,\n                        next_cursor=next_cursor,\n                        release_execution=False,\n                        history_quality="complete",\n                    )\n                    local_start = min(\n                        (\n                            chunk.first_message_index\n                            for chunk in prepared_conversation[0].chunks\n                        ),\n                        default=None,\n                    )\n                    await self._conversation.commit_timeline_turn_in_transaction(\n                        transaction,\n                        session_id,\n                        tenant_id=commit.execution.tenant_id,\n                        execution_id=commit.execution.execution_id,\n                        start_message_index=(\n                            None\n                            if local_start is None\n                            else history.inherited_message_count + local_start\n                        ),\n                        end_message_index=(\n                            history.inherited_message_count\n                            + prepared_conversation.target_transcript_message_count\n                        ),\n                    )\n\n                for attempt in range(2):\n                    try:\n                        await self._conversation.state_store.mutate(\n                            commit_conversation_state\n                        )\n                        break\n                    except AIError as error:\n                        if error.code is not ErrorCode.STORAGE_COMMIT_UNKNOWN or attempt == 1:\n                            raise\n                        session = await self._conversation.get(\n                            session_id,\n                            tenant_id=commit.execution.tenant_id,\n                        )\n                        if session is not None and session.continuation == next_cursor:\n                            break\n\n        execution_stores = [self._execution.state_store]\n''',
)

print("session timeline patch part 5 applied")
