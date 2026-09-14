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


path = "tests/ai/test_file_input_regressions.py"
replace_once(
    path,
    '''        assert stored.codec == "text"\n        assert stored.payload == StoredPayload.inline_text("plain text")\n''',
    '''        assert stored.codec == "text"\n        assert stored.payload == StoredPayload.inline_text("plain text")\n        assert stored.view == {\n            "version": 1,\n            "prompt": {"kind": "text", "text": "plain text"},\n            "files": [],\n        }\n''',
)
replace_once(
    path,
    '''        assert prepared.stored_user_input is not None\n        assert isinstance(prepared.request.user_prompt, tuple)\n        assert session.reads == ["evidence.txt"]\n\n        replay = await materializer.restore(prepared.stored_user_input)\n''',
    '''        assert prepared.stored_user_input is not None\n        assert isinstance(prepared.request.user_prompt, tuple)\n        assert session.reads == ["evidence.txt"]\n        assert prepared.stored_user_input.view == {\n            "version": 1,\n            "prompt": {"kind": "text", "text": "inspect"},\n            "files": [\n                {\n                    "path": "evidence.txt",\n                    "media_type": "text/plain",\n                    "size": 8,\n                    "digest": (\n                        "ee8250fb76e094b34b471f13a73dbbe51d1ae142e9df59d7c0d31ec20f0a0a8e"\n                    ),\n                }\n            ],\n        }\n        view_text = str(prepared.stored_user_input.view)\n        assert "Workspace file path" not in view_text\n        assert "ZXZpZGVuY2U=" not in view_text\n\n        replay = await materializer.restore(prepared.stored_user_input)\n''',
)

save(
    "tests/ai/test_session_timeline.py",
    '''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Web Session timeline projection regressions."""

from dataclasses import replace
from datetime import datetime, timezone

import pytest
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    SystemPromptPart,
    TextPart,
    ToolReturnPart,
    UserPromptPart,
)

from linktools.ai.agent import AgentBindingSnapshot, AgentSpec, bind_output
from linktools.ai.core import (
    ExecutionLineageKind,
    ExecutionStatus,
    HmacCursorSigner,
    Principal,
    SessionStatus,
    TenantAuthorizationPolicy,
    UsageMetrics,
    step_conversation_id,
)
from linktools.ai.runtime._session import DefaultSessionService
from linktools.ai.runtime.service_api import ExecutionView
from linktools.ai.runtime.state import RuntimeDomain, RuntimeState
from linktools.ai.runtime.state._contracts import (
    ConversationCursor,
    ExecutionRecord,
    ResultRecord,
    SessionRecord,
    StoredUserInput,
)
from linktools.ai.runtime.state._step_contracts import ContinuableSnapshot, RunRecord
from linktools.ai.storage import StoredPayload


def _binding() -> AgentBindingSnapshot:
    output = bind_output()
    return AgentBindingSnapshot(
        AgentSpec(id="agent"),
        (),
        (),
        (),
        (),
        (),
        (),
        output,
        "0" * 64,
    )


def _session() -> SessionRecord:
    now = datetime.now(timezone.utc)
    return SessionRecord(
        session_id="session",
        tenant_id="tenant",
        owner_principal_id="owner",
        status=SessionStatus.OPEN,
        revision=0,
        cwd=None,
        metadata={},
        created_at=now,
        updated_at=now,
        closed_at=None,
        active_execution_id=None,
        agent_id="agent",
    )


def _execution(
    execution_id: str,
    *,
    status: ExecutionStatus,
    prompt: str,
    error_code: str | None = None,
) -> ExecutionRecord:
    now = datetime.now(timezone.utc)
    result = None
    if status in {ExecutionStatus.SUCCEEDED, ExecutionStatus.FAILED, ExecutionStatus.CANCELLED}:
        result = ResultRecord(
            execution_id,
            "tenant",
            StoredPayload.inline_json({"text": "ok"})
            if status is ExecutionStatus.SUCCEEDED
            else None,
            "end_turn" if status is ExecutionStatus.SUCCEEDED else "error",
            UsageMetrics(),
            now,
        )
    return ExecutionRecord(
        execution_id=execution_id,
        tenant_id="tenant",
        session_id="session",
        parent_execution_id=None,
        root_execution_id=execution_id,
        source_execution_id=None,
        base_execution_id=None,
        lineage_kind=ExecutionLineageKind.SESSION_RESUME,
        status=status,
        revision=1,
        event_sequence=1,
        agent_run_sequence=1 if status is ExecutionStatus.SUCCEEDED else 0,
        error_code=error_code,
        safe_error_details={"visible": "safe"} if error_code else {},
        created_at=now,
        updated_at=now,
        mode="run",
        planning=False,
        thinking=False,
        binding=_binding(),
        principal_id="owner",
        principal_kind="user",
        stored_user_input=StoredUserInput(
            "text",
            StoredPayload.inline_text(prompt),
            {
                "version": 1,
                "prompt": {"kind": "text", "text": prompt},
                "files": [],
            },
        ),
        result=result,
    )


class _Executions:
    def __init__(self, values: dict[str, ExecutionRecord]) -> None:
        self.values = values
        self.get_many_calls = 0

    async def get_many(self, execution_ids, *, tenant_id: str):
        assert tenant_id == "tenant"
        self.get_many_calls += 1
        return {value: self.values[value] for value in execution_ids}

    async def get(self, execution_id: str, *, tenant_id: str):
        assert tenant_id == "tenant"
        return self.values.get(execution_id)


class _ExecutionService:
    async def wait(self, *args, **kwargs):
        raise AssertionError("timeline must not wait for an execution")


async def _materialize_conversation(state: RuntimeState) -> tuple[str, int]:
    run_id = "timeline-conversation-run"
    conversation_id = step_conversation_id(
        namespace="session-timeline",
        tenant_id="tenant",
        execution_id="success",
    )
    now = datetime.now(timezone.utc)
    await state.steps.register_run(
        RunRecord(
            run_id=run_id,
            conversation_id=conversation_id,
            parent_run_id=None,
            agent_name="agent",
            metadata={"agent_name": "agent"},
            started_at=now,
        )
    )
    messages = [
        ModelRequest(
            parts=[
                SystemPromptPart(content="SECRET SYSTEM INSTRUCTION"),
                UserPromptPart(content="original prompt"),
                UserPromptPart(content='Workspace file path: "secret.txt"'),
                RetryPromptPart(content="INTERNAL RETRY PROMPT"),
                ToolReturnPart(
                    tool_name="lookup",
                    tool_call_id="tool-1",
                    content={"ok": True},
                ),
            ],
            conversation_id=conversation_id,
        ),
        ModelResponse(
            parts=[TextPart(content="visible answer")],
            conversation_id=conversation_id,
        ),
    ]
    await state.steps.save_snapshot(
        ContinuableSnapshot(
            run_id=run_id,
            step_index=1,
            messages=messages,
            conversation_id=conversation_id,
            parent_run_id=None,
            agent_name="agent",
            timestamp=now,
            state="complete",
        )
    )
    await state.steps.materialize_conversation(step_run_id=run_id)
    return run_id, len(messages)


@pytest.mark.asyncio
async def test_session_timeline_restores_original_prompt_without_runtime_instructions() -> None:
    state = RuntimeState.in_memory()
    await state.initialize(namespace="session-timeline", tenant_id="tenant")
    try:
        created = await state.conversation.sessions.create(_session())
        success = _execution(
            "success",
            status=ExecutionStatus.SUCCEEDED,
            prompt="original prompt",
        )
        failed = _execution(
            "failed",
            status=ExecutionStatus.FAILED,
            prompt="failed prompt",
            error_code="MODEL_UNAVAILABLE",
        )

        await state.conversation.sessions.admit_execution(
            "session",
            tenant_id="tenant",
            execution_id="success",
            expected=None,
        )
        run_id, message_count = await _materialize_conversation(state)

        async def commit_success(transaction):
            await state.conversation.sessions.commit_timeline_turn_in_transaction(
                transaction,
                "session",
                tenant_id="tenant",
                execution_id="success",
                start_message_index=0,
                end_message_index=message_count,
            )
            await state.conversation.sessions.advance_continuation_in_transaction(
                transaction,
                "session",
                tenant_id="tenant",
                execution_id="success",
                expected=None,
                next_cursor=ConversationCursor(
                    run_id,
                    history_id=created.history_id,
                    message_count=message_count,
                ),
                release_execution=True,
                history_quality="complete",
            )

        await state.conversation.sessions.state_store.mutate(commit_success)

        await state.conversation.sessions.admit_execution(
            "session",
            tenant_id="tenant",
            execution_id="failed",
            expected=ConversationCursor(
                run_id,
                history_id=created.history_id,
                message_count=message_count,
            ),
        )
        await state.conversation.sessions.release_execution(
            "session",
            tenant_id="tenant",
            execution_id="failed",
        )

        executions = _Executions({"success": success, "failed": failed})
        service = DefaultSessionService(
            state.conversation,
            executions,  # type: ignore[arg-type]
            TenantAuthorizationPolicy(),
            _ExecutionService(),  # type: ignore[arg-type]
            HmacCursorSigner("session", b"session-timeline-key"),
            history_reader=object(),  # type: ignore[arg-type]
            transcript_store=state.steps.read_store(RuntimeDomain.CONVERSATION),
        )
        principal = Principal("owner", "tenant")

        latest = await service.timeline(
            "session",
            principal=principal,
            limit=1,
        )
        assert len(latest.items) == 1
        assert latest.items[0].execution_id == "failed"
        assert latest.items[0].user_input == {
            "version": 1,
            "prompt": {"kind": "text", "text": "failed prompt"},
            "files": [],
        }
        assert latest.items[0].conversation_committed is False
        assert latest.items[0].items == ()
        assert latest.items[0].error_code == "MODEL_UNAVAILABLE"
        assert latest.next_cursor is not None

        older = await service.timeline(
            "session",
            principal=principal,
            cursor=latest.next_cursor,
            limit=1,
        )
        assert len(older.items) == 1
        turn = older.items[0]
        assert turn.execution_id == "success"
        assert turn.user_input == {
            "version": 1,
            "prompt": {"kind": "text", "text": "original prompt"},
            "files": [],
        }
        assert turn.conversation_committed is True
        assert [item.item_kind for item in turn.items] == ["tool_result", "assistant"]
        rendered = repr(turn)
        assert "visible answer" in rendered
        assert "SECRET SYSTEM INSTRUCTION" not in rendered
        assert "Workspace file path" not in rendered
        assert "INTERNAL RETRY PROMPT" not in rendered
        assert older.next_cursor is None
        assert executions.get_many_calls == 2
    finally:
        await state.close()
''',
)

print("session timeline patch part 4 applied")
