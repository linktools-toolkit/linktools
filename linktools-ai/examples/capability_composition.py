#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Offline capability composition, current Skill reads and admitted Memory reuse.

All LinkTools imports are public. The model scripts call real capability tools;
they do not connect to a model provider. Temporary state is not a deployment
migration. Reopening and retrying a completed execution is not crash recovery.
"""

import asyncio
import json
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from tempfile import TemporaryDirectory

from pydantic_ai.capabilities import AbstractCapability, CombinedCapability
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, ToolCallPart, ToolReturnPart, UserPromptPart
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, FunctionModel
from pydantic_ai.tools import RunContext
from pydantic_ai_harness.step_persistence import SqliteStepStore, StepPersistence

from linktools.ai.asset import AssetKey, AssetStore, InMemoryAssetBackend
from linktools.ai.capability import (
    AgentContext, AssetCapability, ExecutionCapability,
    FileSystem, MemoryCapability, SessionCapability, ShellCapability, TraceCapability,
)
from linktools.ai.core import ExecutionStatus, JsonValue
from linktools.ai.persistence import PersistenceStorage
from linktools.ai.runtime import Runtime, RuntimeFeatures
from linktools.ai.storage import StorageOverlay
from linktools.ai.workspace import Workspace


class OfflineModel:
    """The application's complete ModelBinding for deterministic local calls."""

    provider = "test"

    def __init__(
        self,
        name: str,
        respond: Callable[[list[ModelMessage], AgentInfo], ModelResponse],
    ) -> None:
        self.model_identity = "test:" + name
        self.contract: dict[str, JsonValue] = {"provider": "test", "model": name}
        self.respond = respond

    def materialize(self) -> FunctionModel:
        async def stream(
            messages: list[ModelMessage], info: AgentInfo,
        ) -> AsyncIterator[str | dict[int, DeltaToolCall]]:
            response = self.respond(messages, info)
            for index, part in enumerate(response.parts):
                if isinstance(part, TextPart):
                    yield part.content
                elif isinstance(part, ToolCallPart):
                    yield {index: DeltaToolCall(part.tool_name, json.dumps(part.args_as_dict()), tool_call_id=part.tool_call_id)}
                else:
                    raise TypeError("The example model only produces text and tool calls")

        return FunctionModel(self.respond, stream_function=stream, model_name=self.model_identity)


class ObserveRuns(AbstractCapability[AgentContext[None]]):
    """An application-owned native hook with an explicitly shared event sink."""

    id = "example.observe-runs"

    def __init__(self, executions: list[str]) -> None:
        self.executions = executions

    async def before_run(self, ctx: RunContext[AgentContext[None]]) -> None:
        self.executions.append(ctx.deps.execution_id)


async def captured_assets_and_workspace(root: Path) -> None:
    """Asset owns versions; Workspace owns files/processes; tools own meaning."""
    root.mkdir()
    (root / "hello.txt").write_text("workspace content", encoding="utf-8")
    backend = InMemoryAssetBackend()
    store = AssetStore(StorageOverlay(backend, writer=backend))
    skill = AssetKey("skill", "guide/SKILL.md")
    resource = AssetKey("skill", "guide/reference.txt")
    returned: dict[str, object] = {}
    runs: list[str] = []

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        request = messages[-1]
        assert isinstance(request, ModelRequest)
        results = [part for part in request.parts if isinstance(part, ToolReturnPart)]
        if results:
            returned.update((part.tool_call_id, part.content) for part in results)
            return ModelResponse(parts=[TextPart("resources read")])
        names = {tool.name for tool in info.function_tools}
        assert {"load_skill", "files_list_files", "run_command"} <= names
        return ModelResponse(parts=[
            ToolCallPart("load_skill", {"skill_id": "guide"}, tool_call_id="skill"),
            ToolCallPart("load_skill", {"skill_id": "guide", "path": "reference.txt"}, tool_call_id="reference"),
            ToolCallPart("files_list_files", {}, tool_call_id="files"),
            ToolCallPart("run_command", {"command": "printf capability-example"}, tool_call_id="command"),
        ])

    try:
        await store.initialize()
        await store.put(skill, b"---\nname: guide\ndescription: Example guide\n---\nCaptured instructions")
        await store.put(resource, b"captured resource")
        features = RuntimeFeatures("capability-assets")
        features.capability(AssetCapability(store))
        features.capability(CombinedCapability([
            FileSystem(Workspace.load(root)).prefix_tools("files"),
            ShellCapability(),
            ObserveRuns(runs),
            TraceCapability(),
        ], id="example.workspace-policy"))
        features.model("offline", OfflineModel("resources", respond))
        features.agent("default", model="offline", allow_tools=("list_files", "run_command"),
                       allow_skills=("guide",), allow_subagents=(), allow_capabilities=())
        resolution = await features.resolve()
        async with Runtime.open("capability-assets", storage=PersistenceStorage.in_memory(), features=features.borrow()) as runtime:
            await store.put(resource, b"new authoring resource")
            assert resolution.asset_reader is not None
            assert await resolution.asset_reader.get(resource) == b"captured resource"
            current = await runtime.features.resolve()
            assert await current.asset_reader.get(resource) == b"new authoring resource"
            execution = await runtime.agents.get().start("inspect current resources")
            outcome = await execution.wait(timeout_seconds=30)
            assert outcome.result.status is ExecutionStatus.SUCCEEDED
            assert "Captured instructions" in str(returned["skill"])
            assert "new authoring resource" in str(returned["reference"])
            assert "hello.txt" in str(returned["files"])
            assert "capability-example" in str(returned["command"])
            assert runs == [execution.execution_id]
            assert (await execution.model_interactions()).items
        assert store.ready
    finally:
        await store.close()
    assert not store.ready
    print("Current Skill resources, wrapped FileSystem, Shell and native hook passed")


async def admitted_memory_after_reopen(root: Path) -> None:
    """A retry uses current tools while retaining its admitted Memory scope."""
    observations: list[tuple[str, set[str]]] = []
    recalled: list[object] = []

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        request = messages[-1]
        assert isinstance(request, ModelRequest)
        returns = [part for part in request.parts if isinstance(part, ToolReturnPart)]
        if returns:
            recalled.extend(part.content for part in returns if part.tool_name == "read_memory")
            return ModelResponse(parts=[TextPart("done")])
        prompt = next(part.content for part in reversed(request.parts) if isinstance(part, UserPromptPart))
        assert isinstance(prompt, str)
        observations.append((prompt, {tool.name for tool in info.function_tools}))
        if prompt == "remember":
            return ModelResponse(parts=[ToolCallPart("write_memory", {"content": "admitted note"}, tool_call_id="write")])
        if prompt == "recall":
            return ModelResponse(parts=[ToolCallPart("read_memory", {"file": "MEMORY.md"}, tool_call_id="read")])
        return ModelResponse(parts=[TextPart("fresh execution")])

    original = RuntimeFeatures("memory-original")
    original.capability(ExecutionCapability(retain_history=True))
    original.capability(MemoryCapability(scope="example-notes"))
    original.model("offline", OfflineModel("memory", respond))
    original.agent("default", model="offline", allow_tools=("write_memory", "read_memory"),
                   allow_skills=(), allow_subagents=())
    async with Runtime.open("capability-memory", storage=PersistenceStorage.from_root(root, exclusive_writer=True),
                            features=original) as runtime:
        first = await runtime.agents.get().run("remember", timeout_seconds=30)
        assert first.result.status is ExecutionStatus.SUCCEEDED
        execution_id = first.result.execution_id

    current = RuntimeFeatures("memory-current")
    current.capability(ExecutionCapability(retain_history=True))
    current.model("offline", OfflineModel("memory", respond))
    current.agent("default", model="offline", allow_tools=(), allow_skills=(), allow_subagents=())
    async with Runtime.open("capability-memory", storage=PersistenceStorage.from_root(root, exclusive_writer=True),
                            features=current) as runtime:
        fresh = await runtime.agents.get().run("fresh", timeout_seconds=30)
        assert fresh.result.status is ExecutionStatus.SUCCEEDED
        current.capability(MemoryCapability(scope="new-host-notes"))
        source = await runtime.executions.get(execution_id)
        retry = await source.retry("recall", idempotency_key="recall")
        assert (await retry.wait(timeout_seconds=30)).result.status is ExecutionStatus.SUCCEEDED
    assert observations == [
        ("remember", {"read_memory", "write_memory"}),
        ("fresh", set()),
        ("recall", {"read_memory", "write_memory"}),
    ]
    assert len(recalled) == 1 and "admitted note" in str(recalled[0])
    print("Fresh host policy and admitted Memory retry after reopen passed")


async def planning_and_sessions(root: Path) -> None:
    """Planning tools are independent from ordinary tool selection and DAGs."""
    returned: list[object] = []
    runs: list[str] = []

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        request = messages[-1]
        assert isinstance(request, ModelRequest)
        results = [part for part in request.parts if isinstance(part, ToolReturnPart)]
        if results:
            returned.extend(part.content for part in results)
            return ModelResponse(parts=[TextPart("planned")])
        assert {tool.name for tool in info.function_tools} == {"write_plan"}
        return ModelResponse(parts=[ToolCallPart("write_plan", {"items": [
            {"id": "inspect", "content": "Inspect the example", "status": "completed"},
        ]}, tool_call_id="plan")])

    # The official database-backed store opens and closes a connection per call.
    # Runtime borrows the store; its lifetime also covers reads after Runtime closes.
    steps = SqliteStepStore(database=root / "steps.db")
    features = RuntimeFeatures("planning-session")
    features.capability(CombinedCapability([
        ExecutionCapability(retain_history=True),
        StepPersistence(store=steps, capture_frontier=True),
        TraceCapability(), SessionCapability(), ObserveRuns(runs),
    ], id="example.session-policy"))
    features.model("offline", OfflineModel("planning", respond))
    features.agent("default", model="offline", allow_tools=(), allow_skills=(),
                   allow_subagents=(), allow_capabilities=())
    async with Runtime.open("capability-planning", storage=PersistenceStorage.from_root(root, exclusive_writer=True),
                            features=features) as runtime:
        session = runtime.agents.get().session("example")
        outcome = await session.run("make a plan", planning=True, timeout_seconds=30)
        assert outcome.result.status is ExecutionStatus.SUCCEEDED
        execution = await runtime.executions.get(outcome.result.execution_id)
        assert (await session.history()).items
        assert (await execution.transcript(include_content=True)).items
        assert (await execution.model_interactions()).items
        assert returned and runs == [execution.execution_id]
    saved_runs = await steps.list_runs()
    assert len(saved_runs) == 1
    assert await steps.latest_snapshot(run_id=saved_runs[0].run_id) is not None
    print("Planning, Session, retained history, native StepPersistence and Trace composition passed")


async def main() -> None:
    with TemporaryDirectory(prefix="linktools-capabilities-") as directory:
        root = Path(directory)
        await captured_assets_and_workspace(root / "workspace")
        await admitted_memory_after_reopen(root / "memory")
        await planning_and_sessions(root / "planning")


if __name__ == "__main__":
    asyncio.run(main())
