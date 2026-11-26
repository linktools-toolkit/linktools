#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A native Step Graph publishes one retained conversation turn without Trace."""

from contextlib import AsyncExitStack
from pathlib import Path

import pytest
from pydantic_ai.messages import BinaryContent, ModelResponse, TextPart, UserPromptPart
from pydantic_ai_harness.step_persistence import StepPersistence

from linktools.ai.capability import ExecutionCapability
from linktools.ai.core import CommitPolicy, TaskStatus, TenantAuthorizationPolicy
from linktools.ai.history import History
from linktools.ai.inputs import AgentTaskInput
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.runtime import Runtime, RuntimeFeatures
from linktools.ai.session import CreateSessionRequest
from linktools.ai.task import Task, TaskGraph, TaskNode
from linktools.ai.task.engine import TaskEngine
from linktools.ai.workspace import Workspace

from ._runtime_test_helpers import _Models, _UsageFunctionModel, model_features, workspace_features


@pytest.mark.asyncio
@pytest.mark.parametrize("attachment", ("none", "file", "binary", "mixed"))
async def test_graph_step_reopens_exact_session_result_without_trace(tmp_path: Path, attachment: str) -> None:
    calls = 0
    seen: list[bytes] = []
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    file = workspace / "source.txt"
    file.write_bytes(b"workspace-file")
    second = workspace / "second.txt"
    second.write_bytes(b"second-file")
    files = {"file": ("source.txt",), "mixed": ("source.txt", "second.txt")}.get(attachment, ())
    binary, tail = b"binary-content", b"binary-tail"
    prompt: str | tuple[str | BinaryContent, ...] = "graph question"
    if attachment == "binary":
        prompt = ("graph question", BinaryContent(binary, media_type="application/octet-stream"))
    elif attachment == "mixed":
        prompt = (
            "graph question", BinaryContent(binary, media_type="application/octet-stream"),
            "following text", BinaryContent(tail, media_type="application/octet-stream"),
        )
    expected = {
        "none": [], "file": [b"workspace-file"], "binary": [binary],
        "mixed": [b"workspace-file", b"second-file", binary, tail],
    }[attachment]

    async def model(_messages, _info):
        nonlocal calls
        calls += 1
        seen.extend(
            item.data for message in _messages for part in message.parts
            if isinstance(part, UserPromptPart) and not isinstance(part.content, str)
            for item in part.content if isinstance(item, BinaryContent)
        )
        return ModelResponse(parts=[TextPart("retained graph output")])

    namespace = "graph-step-no-trace"
    storage = PersistenceStorage.from_root(tmp_path, exclusive_writer=True)
    class HostAuthorization(TenantAuthorizationPolicy):
        pass

    authorization = HostAuthorization()
    async with AsyncExitStack() as stack:
        state = await stack.enter_async_context(Persistence.open(namespace, storage=storage))
        native = await state.create_step_store()
        stack.push_async_callback(native.close)
        features = workspace_features(namespace, workspace=Workspace.load(workspace)) if files else RuntimeFeatures(namespace)
        features.capability(ExecutionCapability(retain_history=True, commit=CommitPolicy.sync()), scope="runtime")
        features.capability(StepPersistence(store=native, capture_frontier=True), scope="runtime")
        features.agent("default", model="default", allow_tools=(), allow_skills=(), allow_subagents=())
        model_features(_Models(_UsageFunctionModel(model)), features)
        runtime = await stack.enter_async_context(Runtime.open(
            persistence=state, features=features, authorization=authorization,
        ))
        principal = runtime.default_principal
        created = await runtime.sessions.create(CreateSessionRequest(principal, "graph-session", "create-session"))
        task = Task.from_agent("native.agent", runtime.agents.get())
        engine = await stack.enter_async_context(TaskEngine.open(
            task, persistence=state, context=runtime.context, authorization=authorization,
        ))
        graph = TaskGraph("source-graph", (TaskNode(
            "agent", task=task, input=AgentTaskInput(prompt, files=files, session_id=created.session_id),
        ),))
        run = await engine.start(graph, principal=principal, idempotency_key="start-graph")
        completed = (await run.wait(timeout_seconds=10)).result
        assert completed.status is TaskStatus.SUCCEEDED
        execution_id = (await run.execution("agent")).execution_id
        session = await runtime.sessions.get(created.session_id, principal=principal)
        items = (await session.timeline())["items"]
        assert [item["turn_id"] for item in items] == [execution_id]
        assert items[0]["conversation_committed"] and items[0]["output"] == {"text": "retained graph output"}
        execution = await state.execution.executions.get(execution_id, tenant_id=principal.tenant_id)
        assert execution is not None and execution.requires_task_invocation_capture
        assert execution.restart_checkpoints and not execution.durable_trace and execution.session_id is None
        assert await state.conversation.sessions.get(created.session_id, tenant_id=principal.tenant_id) is None

    file.unlink()
    second.unlink()
    async with (
        Persistence.open(namespace, storage=storage) as reopened,
        History.open(persistence=reopened, authorization=authorization) as history,
    ):
        turn = await history.turn_result(created.session_id, execution_id, principal=principal)
        assert turn["conversation_committed"] and turn["output"] == {"text": "retained graph output"}
        projected = turn["user_input"]["attachments"]
        assert sorted(item["size"] for item in projected) == sorted(map(len, expected))
        assert all(set(item) == {"attachment_id", "source", "media_type", "size", "position"} for item in projected)
    assert calls == 1
    assert seen == expected
