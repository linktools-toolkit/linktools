#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""History source selection consumes control views, not execution payload leaves."""

from dataclasses import replace

import pytest
from pydantic_ai.messages import ModelRequest, UserPromptPart

from linktools.ai.core import ExecutionLineageKind, ExecutionStatus, Principal, StopReason, TenantAuthorizationPolicy, UsageMetrics
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.history._service import DefaultExecutionHistoryService
from linktools.ai.persistence._execution_content import ExecutionContent
from linktools.ai.persistence.contracts import ResultRecord, StoredUserInput
from linktools.ai.storage import StoredPayload

from .test_history_request_association import _history
from .test_history_projection_conformance import _record


@pytest.mark.asyncio
async def test_history_views_skip_large_execution_leaves_for_root_and_children(monkeypatch: pytest.MonkeyPatch) -> None:
    async with _history() as history:
        recorder, = history.recorders
        recorder.append_transcript_message(ModelRequest(parts=[UserPromptPart("actual history")]))
        await history.archive()
        repository = history.state.execution.executions
        previous = await repository.get("execution", tenant_id="tenant")
        assert previous is not None
        large = "not history " * 10000
        result = ResultRecord(StoredPayload.inline_json(large), StopReason.END_TURN, UsageMetrics(), previous.updated_at)
        root = replace(previous, status=ExecutionStatus.SUCCEEDED,
            stored_user_input=StoredUserInput("text", StoredPayload.inline_text(large)), result=result)
        await repository.state_store.mutate(lambda tx: tx.delete_record(repository.execution_record_key("execution")))
        await repository.create_execution(root)
        child = replace(_record(ExecutionStatus.STARTED, 0), execution_id="child",
            parent_execution_id="execution", parent_invocation_id="invocation", root_execution_id="execution",
            lineage_kind=ExecutionLineageKind.SUBAGENT,
            stored_user_input=StoredUserInput("text", StoredPayload.inline_text(large)))
        await repository.create_execution(child)

        async def forbid(*args: object, **kwargs: object) -> None:
            raise AssertionError("history must not expand execution input/output leaves")

        monkeypatch.setattr(ExecutionContent, "read_payloads", forbid)
        monkeypatch.setattr(repository, "get", forbid)
        monkeypatch.setattr(repository, "list_children", forbid)
        service = DefaultExecutionHistoryService(repository, TenantAuthorizationPolicy(), history.reader)
        visible = await service.history("execution", principal=Principal("principal", "tenant"), include_content=True)
        assert len(visible.items) == 1
        assert visible.items[0].content == "actual history"
        hidden = await service.history("execution", principal=Principal("principal", "tenant"), include_content=False)
        assert len(hidden.items) == 1 and hidden.items[0].content is None
        assert hidden.items[0].item_kind == visible.items[0].item_kind


@pytest.mark.asyncio
@pytest.mark.parametrize("restart", (False, True))
async def test_empty_successful_history_uses_real_restart_policy(restart: bool) -> None:
    async with _history(2) as history:
        for recorder in history.recorders:
            await recorder.record_event("AGENT_RUN_STARTED", 0)
            await recorder.record_event("AGENT_RUN_SUCCEEDED", 1)
        await history.archive()
        repository = history.state.execution.executions
        previous = await repository.get("execution", tenant_id="tenant")
        assert previous is not None
        await repository.state_store.mutate(lambda tx: tx.delete_record(repository.execution_record_key("execution")))
        await repository.create_execution(replace(previous, status=ExecutionStatus.SUCCEEDED, restart_checkpoints=restart))
        viewed = await repository.get_view("execution", tenant_id="tenant")
        assert viewed is not None and viewed[1].restart_checkpoints is restart
        if restart:
            page = await history.reader.history("execution", tenant_id="tenant", cursor=None, limit=100, agent_run_seq=2)
            assert page.items == ()
        else:
            with pytest.raises(AIError) as caught:
                await history.reader.history("execution", tenant_id="tenant", cursor=None, limit=100, agent_run_seq=2)
            assert caught.value.code is ErrorCode.EXECUTION_HISTORY_UNAVAILABLE
