#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Native workspace composition keeps tool authority and lifecycle explicit."""

from pathlib import Path

import pytest
from pydantic_ai import Agent
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.workspaces import CommandResult, WorkspaceRef

from linktools.ai.capability import FileSystem, PydanticToolControlCapability, ShellCapability
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime import RuntimeFeatures
from linktools.ai.workspace import ToolPermissionPolicy, ToolPermissionRule, Workspace


class _CommandBackend:
    ref = WorkspaceRef(provider="test", id="one")

    def __init__(self) -> None:
        self.calls = []
        self.closed = False

    async def run_command(self, command, *, timeout_seconds=None):
        self.calls.append((command, timeout_seconds))
        return "native output"

    async def start_command(self, command):
        raise AssertionError("unexpected background command")

    async def check_command(self, command_id):
        raise AssertionError("unexpected background command")

    async def stop_command(self, command_id):
        raise AssertionError("unexpected background command")

    async def close(self) -> None:
        self.closed = True

    async def working_dir(self) -> str:
        return "/workspace"

    async def run(self, command, *, shell=False, env=None, timeout=None):
        self.calls.append((command, shell, env, timeout))
        return CommandResult(exit_code=0, stdout="native output", stderr="")


def _command_model(messages, info):
    if not any(isinstance(part, ToolReturnPart) for message in messages for part in message.parts):
        return ModelResponse(parts=[ToolCallPart("run_command", {"command": "printf example"})])
    return ModelResponse(parts=[TextPart("done")])


def test_filesystem_keeps_file_surface_without_shell(tmp_path: Path) -> None:
    tools = FileSystem(Workspace.load(tmp_path)).tool_definitions()
    assert {tool.name for tool in tools} == {
        "attach_files", "create_directory", "edit_file", "file_info", "find_files",
        "list_directory", "list_files", "read_file", "search_files", "write_file",
    }


def test_shell_metadata_declares_effect_certainty() -> None:
    captured = {tool.name: tool for tool in ShellCapability().tool_definitions()}
    for name, tool in captured.items():
        assert tool.metadata["linktools.ai.tool_class"] == "shell"
        assert tool.metadata["linktools.ai.effect_policy"] == (
            "none" if name == "check_command" else "non_replay_safe"
        )


@pytest.mark.asyncio
async def test_shell_only_features_capture_tools_before_runtime_binding() -> None:
    features = RuntimeFeatures("native-shell")
    features.capability(ShellCapability())
    capture = await features.capture()
    tools = [item for item in capture.contributions if item.kind == "tool"]
    assert {tool.id for tool in tools} == {"run_command", "start_command", "check_command", "stop_command"}
    assert tools[0].contract["metadata"]["linktools.ai.tool_class"] == "shell"
    assert capture.workspace is None


def test_agent_scoped_shell_cannot_bypass_runtime_tool_admission() -> None:
    features = RuntimeFeatures("native-shell")
    with pytest.raises(AIError) as raised:
        features.capability(ShellCapability(), scope="agent")
    assert raised.value.code is ErrorCode.CAPABILITY_RESOLUTION_INVALID


@pytest.mark.asyncio
async def test_native_shell_executes_on_explicit_backend_and_borrows_it() -> None:
    backend = _CommandBackend()
    capability = ShellCapability().bind(("run_command",), backend=backend)
    result = await Agent(FunctionModel(_command_model), capabilities=[PydanticToolControlCapability(), capability]).run(
        "run the command", workspace=backend,
    )
    assert result.output == "done"
    assert len(backend.calls) == 1
    assert backend.calls[0][0].endswith("printf example")


@pytest.mark.asyncio
async def test_native_shell_denial_precedes_backend_side_effect() -> None:
    backend = _CommandBackend()
    capability = ShellCapability().bind(
        ("run_command",), backend=backend,
        policy=ToolPermissionPolicy(rules=(ToolPermissionRule(decision="deny", tool_class="shell"),)),
    )
    result = await Agent(FunctionModel(_command_model), capabilities=[PydanticToolControlCapability(), capability]).run(
        "run the command", workspace=backend,
    )
    assert result.output == "done"
    assert backend.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("with_files", (False, True))
async def test_runtime_native_shell_uses_admitted_tools_and_explicit_environment(tmp_path: Path, with_files: bool) -> None:
    from linktools.ai.persistence import Persistence, PersistenceStorage
    from linktools.ai.runtime import Runtime
    from linktools.ai.workspace import LocalWorkspaceBackend
    from .test_explicit_runtime_features import _FeatureFunctionModel, _FeatureModels

    features = RuntimeFeatures("native-shell-runtime", workspace_factory=LocalWorkspaceBackend)
    if with_files:
        features.capability(FileSystem(Workspace.load(tmp_path)))
    features.capability(ShellCapability())
    features.agent("default", allow_tools=("terminal:*",))
    features.model("default", _FeatureModels(_FeatureFunctionModel(_command_model)))
    async with Persistence.open("native-shell", storage=PersistenceStorage.in_memory()) as persistence:
        async with Runtime.attach(persistence=persistence, features=features) as runtime:
            execution = await runtime.agents.get().run("execute command", timeout_seconds=10)
            assert execution.result.status.value == "SUCCEEDED", execution.result
            assert "done" in str(execution.result.output)


@pytest.mark.asyncio
async def test_managed_boundary_forwards_native_toolset_lifecycle_per_run() -> None:
    from copy import copy
    from pydantic_ai.models.test import TestModel
    from pydantic_ai.tools import RunContext
    from pydantic_ai.toolsets import FunctionToolset
    from pydantic_ai.usage import RunUsage
    from linktools.ai.runtime._tool_boundary import BoundaryToolset

    events = []

    class LifecycleToolset(FunctionToolset):
        async def for_run(self, ctx):
            child = copy(self)
            child.run_id = ctx.run_id
            events.append(("run", child.run_id))
            return child

        async def for_run_step(self, ctx):
            events.append(("step", self.run_id))
            return self

        async def __aenter__(self):
            events.append(("enter", self.run_id))
            return self

        async def __aexit__(self, *args):
            events.append(("exit", self.run_id))

    boundary = BoundaryToolset((LifecycleToolset(),), {}, id="lifecycle")
    for run_id in ("first", "second"):
        ctx = RunContext(deps=None, model=TestModel(), usage=RunUsage(), run_id=run_id)
        bound = await boundary.for_run(ctx)
        async with bound:
            await bound.for_run_step(ctx)
            assert await bound.get_tools(ctx) == {}
    assert events == [
        (phase, run_id) for run_id in ("first", "second") for phase in ("run", "enter", "step", "exit")
    ]


@pytest.mark.asyncio
async def test_command_handles_belong_to_one_environment_even_with_shared_root(tmp_path: Path) -> None:
    import shlex
    import sys
    from linktools.ai.workspace import LocalWorkspaceBackend

    first, second = LocalWorkspaceBackend(tmp_path), LocalWorkspaceBackend(tmp_path)
    command = f"{shlex.quote(sys.executable)} -c {shlex.quote('import time; time.sleep(60)')}"
    try:
        started = await first.start_command(command)
        command_id = started.splitlines()[0].removeprefix("command_id: ")
        assert "running" in await first.check_command(command_id)
        with pytest.raises(AIError) as foreign:
            await second.stop_command(command_id)
        assert foreign.value.code is ErrorCode.REQUEST_FIELD_INVALID
        assert "running" in await first.check_command(command_id)
        await first.stop_command(command_id)
    finally:
        await first.close()
        await second.close()
    replacement = LocalWorkspaceBackend(tmp_path)
    try:
        with pytest.raises(AIError) as stale:
            await replacement.check_command(command_id)
        assert stale.value.code is ErrorCode.REQUEST_FIELD_INVALID
        assert not (tmp_path / ".pydantic-ai-harness").exists()
    finally:
        await replacement.close()


@pytest.mark.asyncio
async def test_shell_only_runtime_does_not_require_resource_projection_extension() -> None:
    from linktools.ai.persistence import Persistence, PersistenceStorage
    from linktools.ai.runtime import Runtime
    from linktools.ai.workspace import SupportsResources
    from .test_explicit_runtime_features import _FeatureFunctionModel, _FeatureModels

    backends = []

    def factory(root, *, resources):
        assert resources == ()
        backend = _CommandBackend()
        assert not isinstance(backend, SupportsResources)
        backends.append(backend)
        return backend

    features = RuntimeFeatures("commands-without-resource-projections", workspace_factory=factory)
    features.capability(ShellCapability())
    features.agent("default", allow_tools=("run_command",))
    features.model("default", _FeatureModels(_FeatureFunctionModel(_command_model)))
    async with Persistence.open("commands-only", storage=PersistenceStorage.in_memory()) as persistence:
        async with Runtime.attach(persistence=persistence, features=features) as runtime:
            execution = await runtime.agents.get().run("execute command", timeout_seconds=10)
            assert execution.result.status.value == "SUCCEEDED", execution.result
    executing = [backend for backend in backends if backend.calls]
    assert len(executing) == 1 and executing[0].closed


@pytest.mark.asyncio
async def test_runtime_rejects_owned_provider_without_close_before_activation() -> None:
    from linktools.ai.persistence import Persistence, PersistenceStorage
    from linktools.ai.runtime import Runtime

    activated = []

    class NoCleanup:
        ref = None

        async def working_dir(self):
            activated.append(True)
            return "/workspace"

    features = RuntimeFeatures("missing-workspace-cleanup", workspace_factory=lambda root, **kwargs: NoCleanup())
    async with Persistence.open("no-cleanup", storage=PersistenceStorage.in_memory()) as persistence:
        with pytest.raises(AIError) as rejected:
            async with Runtime.attach(persistence=persistence, features=features):
                raise AssertionError("provider without cleanup was accepted")
        assert rejected.value.code is ErrorCode.SANDBOX_UNAVAILABLE
        assert rejected.value.safe_details["reason"] == "workspace_cleanup_required"
    assert not activated
