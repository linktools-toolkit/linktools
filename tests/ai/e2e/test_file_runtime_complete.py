#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests/ai/e2e/test_file_runtime_complete.py — File-backed Runtime end-to-end
run-completion contract.

Drives the real path: FilesystemStorage -> build_runtime -> Runtime.run -> agent
returns. Asserts the cross-store commit leaves exactly one of each artifact
(no duplicate session messages, no duplicate checkpoint, no duplicate
RunCompleted event, no RunFailed) -- the invariants that broke before the
AgentEngine delegated completion to the FilesystemRunCommitCoordinator."""

import asyncio

from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from linktools.ai.agent.spec import AgentSpec, PromptSpec
from linktools.ai.model.policy import ModelPolicy
from linktools.ai.run.models import RunStatus
from linktools.ai.session.models import MessageRole
from linktools.ai.runtime import Runtime, build_runtime
from linktools.ai.storage.facade import FilesystemStorage
from linktools.ai.storage.filesystem.commit import FilesystemRunCommitCoordinator


def _model_fn(messages, info: AgentInfo) -> ModelResponse:
    return ModelResponse(parts=[TextPart(content='{"response": {"msg": "ok"}}')])


def _registry():
    from linktools.ai.model.registry import ModelRegistry
    from linktools.ai.model.resolver import ModelResolver

    registry = ModelRegistry()
    registry.register("test-model", model=FunctionModel(_model_fn))
    return ModelResolver(registry=registry)


def test_file_runtime_complete_has_one_of_each_artifact(tmp_path):
    storage = FilesystemStorage(root=tmp_path)
    runtime = build_runtime(
        storage=storage,
        model_resolver=_registry(),
        commit_coordinator=FilesystemRunCommitCoordinator.from_storage(storage),
    )
    spec = AgentSpec(
        id="agent-1",
        name="e2e-agent",
        model=ModelPolicy(primary="test-model"),
        instructions=PromptSpec(instructions="hi"),
    )

    async def _run():
        return await runtime.run(spec, "say hello")

    result = asyncio.run(_run())
    assert "ok" in str(result.output)

    async def _verify():
        # The run_id travels on every session message the run wrote.
        sessions_dir = tmp_path / "sessions"
        run_ids = set()
        for session_path in sessions_dir.iterdir() if sessions_dir.exists() else []:
            for m in await storage.sessions.list_messages(session_path.name):
                run_ids.add(m.run_id)
        assert run_ids, "no run_id found among session messages"
        run_id = run_ids.pop()

        record = await storage.runs.get(run_id)
        assert record is not None
        assert record.status is RunStatus.SUCCEEDED

        messages = await storage.sessions.list_messages(record.session_id)
        user_count = sum(1 for m in messages if m.role is MessageRole.USER)
        assistant_count = sum(1 for m in messages if m.role is MessageRole.ASSISTANT)
        assert user_count == 1, f"expected 1 USER, got {user_count}"
        assert assistant_count == 1, f"expected 1 ASSISTANT, got {assistant_count}"

        checkpoint = await storage.checkpoints.latest(run_id)
        assert checkpoint is not None, "no checkpoint written"
        assert checkpoint.sequence == 1, (
            f"expected checkpoint sequence 1, got {checkpoint.sequence}"
        )

        page = await storage.events.list(run_id, limit=100)
        payload_types = [type(e.payload).__name__ for e in page.items]
        assert payload_types.count("RunCompleted") == 1, payload_types
        assert "RunFailed" not in payload_types, payload_types

    asyncio.run(_verify())
