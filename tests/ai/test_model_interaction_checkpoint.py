#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Recovery checkpoints preserve in-flight model request ownership."""

import json
from pathlib import Path

import pytest
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RequestUsage

from linktools.ai.errors import ErrorCode
from linktools.ai.runtime import RuntimeStorage
from linktools.ai.runtime._agent_run_recorder import AgentRunRecorder
from linktools.ai.runtime._journal import ModelRequestJournal
from linktools.ai.runtime.state import RuntimeDomain
from linktools.ai.runtime.state._step_contracts import AgentRunCheckpoint, AgentRunRecord


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "filesystem", "sqlite"))
@pytest.mark.parametrize("status", ("SUCCEEDED", "FAILED", "CANCELLED"))
async def test_checkpoint_preserves_live_request_until_terminal(
    tmp_path: Path,
    backend: str,
    status: str,
) -> None:
    storage = (
        RuntimeStorage.in_memory()
        if backend == "memory"
        else RuntimeStorage.filesystem(tmp_path / "state")
        if backend == "filesystem"
        else RuntimeStorage.sqlite(tmp_path / "state.db")
    )
    await storage.initialize(namespace="live-request", tenant_id="tenant")
    try:
        store = storage.run_store
        capture = AgentRunRecorder(store, execution_id="execution", agent_run_id="run")
        await capture.register_agent_run(AgentRunRecord("run"))
        journal = ModelRequestJournal(
            source_namespace="live-request",
            tenant_id="tenant",
            execution_id="execution",
            agent_run_id="run",
        )
        recovery = store.read_store(RuntimeDomain.RECOVERY)
        for sequence in (1, 2):
            request = ModelRequest(parts=[UserPromptPart(f"request-{sequence}")])
            messages = (*capture.transcript_messages(), request)
            fact = journal.begin(sequence)
            capture.begin_model_interaction(
                fact, TestModel(), messages, None, ModelRequestParameters(), False,
            )
            capture.prepare_model_interaction(
                fact, TestModel(), messages, None, ModelRequestParameters(), False,
            )
            capture.append_transcript_message(request)
            checkpoint = AgentRunCheckpoint(
                agent_run_id="run",
                step_index=sequence,
                messages=list(capture.transcript_messages()),
                state="complete",
                transcript_message_count_before=0,
            )
            await capture.save_checkpoint(checkpoint)
            await capture.save_checkpoint(checkpoint)
            await store.materialize_recovery_checkpoint(
                agent_run_id="run", require_complete=True,
            )
            assert await recovery.model_interaction_count(agent_run_id="run") == sequence - 1
            staged = await store.list_model_interactions(agent_run_id="run")
            assert staged[-1].model_request_seq == sequence
            assert staged[-1].status == "RUNNING"

            response = (
                ModelResponse(parts=[TextPart(f"response-{sequence}")])
                if status == "SUCCEEDED" else None
            )
            error_code = ErrorCode.MODEL_API_ERROR.value if status == "FAILED" else None
            finished = journal.finish(fact.model_request_seq, status=status)
            capture.finish_model_interaction(
                finished,
                model=TestModel(),
                response=response,
                status=status,
                error_code=error_code,
                duration_ns=sequence,
                usage=RequestUsage(input_tokens=sequence * 3, output_tokens=sequence * 2),
            )
            if response is not None:
                capture.append_transcript_message(response)

        await capture.save_checkpoint(
            AgentRunCheckpoint(
                agent_run_id="run",
                step_index=2,
                messages=list(capture.transcript_messages()),
                state="complete" if status == "SUCCEEDED" else "interrupted",
                transcript_message_count_before=0,
            )
        )
        await store.materialize_recovery_checkpoint(
            agent_run_id="run", require_complete=status == "SUCCEEDED",
        )
        archived = await recovery.list_model_interactions(agent_run_id="run")
        assert [value.model_request_seq for value in archived] == [1, 2]
        assert [value.status for value in archived] == [status, status]
        assert [value.error_code for value in archived] == [error_code, error_code]
        assert [value.duration_ns for value in archived] == [1, 2]
        assert [value.usage.input_tokens for value in archived] == [3, 6]
        assert [value.usage.output_tokens for value in archived] == [2, 4]
        resolved = await recovery.resolve_model_interactions(archived)
        for sequence, (request_messages, response_messages, envelope) in enumerate(resolved, 1):
            assert request_messages[-1].parts[0].content == f"request-{sequence}"
            assert json.loads(envelope)["streaming"] is False
            if status == "SUCCEEDED":
                assert response_messages[0].parts[0].content == f"response-{sequence}"
            else:
                assert response_messages is None
    finally:
        await storage.close()
