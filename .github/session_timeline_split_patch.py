from pathlib import Path

path = Path("linktools-ai/src/linktools/ai/runtime/state/_commands.py")
text = path.read_text(encoding="utf-8")
old = '''                local_start = min(\n                    (\n                        chunk.first_message_index\n                        for chunk in prepared_conversation[0].chunks\n                    ),\n                    default=None,\n                )\n                await self._conversation.commit_timeline_turn_in_transaction(\n                    conversation_transaction,\n                    session_id,\n                    tenant_id=commit.execution.tenant_id,\n                    execution_id=commit.execution.execution_id,\n                    start_message_index=(\n                        None\n                        if local_start is None\n                        else history.inherited_message_count + local_start\n                    ),\n                    end_message_index=(\n                        history.inherited_message_count\n                        + prepared_conversation.target_transcript_message_count\n                    ),\n                )\n'''
new = '''                if timeline_range is None:\n                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)\n                await self._conversation.commit_timeline_turn_in_transaction(\n                    conversation_transaction,\n                    session_id,\n                    tenant_id=commit.execution.tenant_id,\n                    execution_id=commit.execution.execution_id,\n                    start_message_index=timeline_range[0],\n                    end_message_index=timeline_range[1],\n                )\n'''
if text.count(old) != 1:
    raise RuntimeError(f"expected one remaining conversation timeline range block, found {text.count(old)}")
path.write_text(text.replace(old, new, 1), encoding="utf-8")
print("split-storage timeline branch unified")
