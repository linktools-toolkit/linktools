from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    file = Path(path)
    text = file.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: expected one replacement, found {count}")
    file.write_text(text.replace(old, new, 1), encoding="utf-8")


local = "linktools-ai/src/linktools/ai/runtime/_local.py"
replace_once(
    local,
    '''            loaded_context = LoadedModelContext(())\n            session_history_source = (\n                current.lineage_kind is ExecutionLineageKind.SESSION_RESUME\n                and history_id is not None\n            )\n            if source_replay_history is not None:\n                history = source_replay_history\n            elif recovery_history_run_id is not None:\n                loaded_context = await self._steps.load_loaded_model_context(\n                    RuntimeDomain.RECOVERY,\n                    recovery_history_run_id,\n                )\n                history = list(loaded_context.model_messages())\n            else:\n                if session_history_source:\n                    loaded_context = await self._steps.load_loaded_model_context(\n                        RuntimeDomain.CONVERSATION,\n                        history_id,\n                    )\n                if session_history_source and loaded_context.messages:\n                    history = list(loaded_context.model_messages())\n                else:\n                    history = cast("list[ModelMessage]", await self._history(current))\n''',
    '''            loaded_context = LoadedModelContext(())\n            session_history_source = (\n                current.lineage_kind is ExecutionLineageKind.SESSION_RESUME\n                and history_id is not None\n            )\n            session_history_owner: str | None = None\n            if session_history_source:\n                if isinstance(\n                    self._step_reads[RuntimeDomain.CONVERSATION],\n                    StateStepArchive,\n                ):\n                    session_history_owner = history_id\n                elif session is not None and session.continuation is not None:\n                    session_history_owner = session.continuation.step_run_id\n            if source_replay_history is not None:\n                history = source_replay_history\n            elif recovery_history_run_id is not None:\n                loaded_context = await self._steps.load_loaded_model_context(\n                    RuntimeDomain.RECOVERY,\n                    recovery_history_run_id,\n                )\n                history = list(loaded_context.model_messages())\n            elif session_history_source:\n                if session_history_owner is not None:\n                    loaded_context = await self._steps.load_loaded_model_context(\n                        RuntimeDomain.CONVERSATION,\n                        session_history_owner,\n                    )\n                history = list(loaded_context.model_messages())\n            else:\n                history = cast("list[ModelMessage]", await self._history(current))\n''',
)

test = "tests/ai/test_session_timeline_runtime.py"
replace_once(
    test,
    '''        result = await session.run("hello", timeout_seconds=10)\n        assert result.status is ExecutionStatus.SUCCEEDED\n\n        page = await session.timeline()\n        assert len(page.items) == 1\n        turn = page.items[0]\n        assert turn.execution_id == result.execution_id\n        assert turn.status is ExecutionStatus.SUCCEEDED\n        assert turn.user_input == {\n            "version": 1,\n            "prompt": {"kind": "text", "text": "hello"},\n            "files": [],\n        }\n        assert turn.conversation_committed is True\n        assert [item.item_kind for item in turn.items] == ["assistant"]\n        assert page.next_cursor is None\n''',
    '''        first = await session.run("hello", timeout_seconds=10)\n        second = await session.run("again", timeout_seconds=10)\n        assert first.status is ExecutionStatus.SUCCEEDED\n        assert second.status is ExecutionStatus.SUCCEEDED\n\n        page = await session.timeline()\n        assert [turn.execution_id for turn in page.items] == [\n            first.execution_id,\n            second.execution_id,\n        ]\n        assert [turn.status for turn in page.items] == [\n            ExecutionStatus.SUCCEEDED,\n            ExecutionStatus.SUCCEEDED,\n        ]\n        assert [turn.user_input["prompt"] for turn in page.items] == [\n            {"kind": "text", "text": "hello"},\n            {"kind": "text", "text": "again"},\n        ]\n        assert all(turn.conversation_committed for turn in page.items)\n        assert [\n            [item.item_kind for item in turn.items] for turn in page.items\n        ] == [["assistant"], ["assistant"]]\n        assert page.next_cursor is None\n''',
)

print("volatile Session context owner fixed")
