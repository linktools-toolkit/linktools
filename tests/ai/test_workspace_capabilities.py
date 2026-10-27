#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Workspace capability projection and materialization contracts."""
from linktools.ai.model import create_model

from ._runtime_test_helpers import workspace_features

import asyncio
from pathlib import Path

from ._runtime_test_helpers import FileToolsStub

import pytest
from linktools.ai.agent import AgentCompiler
from linktools.ai.asset import AssetStore, DirectoryAssetBackend, PrefixAssetPathAdapter
from linktools.ai.capability import (
    CapabilityContribution,
    ToolCallFailed,
    tool_class_from_metadata,
    filesystem_capabilities,
    filesystem_tool_declarations,
)
from linktools.ai.runtime import RuntimeFeatures
from linktools.ai.errors import AIError, ErrorCode

from linktools.ai.spec import (
    AgentSpec,
    MCPServerSpec,
    mcp_server_selector,
    mcp_tool_selector,
)
from linktools.ai.storage import StorageOverlay
from linktools.ai.runtime._tool_boundary import (
    ManagedToolDescriptor,
    BoundaryToolset,
)
from linktools.ai.workspace import (
    WorkspaceResource,
    SupportsFileTools,
    ToolPermissionRule,
    Workspace,
    ToolPermissionPolicy,
)
from pydantic_ai.capabilities import AbstractCapability
from linktools.ai.capability import ShellCapability
from pydantic_ai.exceptions import ApprovalRequired
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import RunContext
from pydantic_ai.usage import RunUsage


def _workspace_tool_contributions(workspace: Workspace):
    return tuple(
        item for item in workspace_features("workspace", workspace=workspace)._contributions
        if item.kind == "tool"
    )


class _RecordingSession(FileToolsStub):
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[object, ...], dict[str, object]]] = []
        self.closed = 0

    async def canonicalize_path(self, path: str) -> str:
        return path

    async def _record(self, name: str, *args: object, **kwargs: object) -> str:
        self.calls.append((name, args, kwargs))
        return name


    async def inspect_file(self, path: str, *, max_bytes=None):
        return {"path": path, "is_text": True, "mime_type": "text/plain",
                "type_source": "extension", "size": 1, "state": "test-state", "bytes_read": 1}

    async def read_text(self, path: str, **kwargs):
        await self._record("read_text", path, **kwargs)
        return {"path": path, "content": "x", "state": "test-state", "bytes_read": 1}

    async def write_file(
        self,
        path: str,
        content: str,
        *,
        old_text: "str | None" = None,
        expected_hash: "str | None" = None,
    ) -> str:
        return await self._record("write_file", path, content, old_text=old_text, expected_hash=expected_hash)







    async def run_command(
        self,
        command: str,
        *,
        timeout_seconds: "float | None" = None,
    ) -> str:
        return await self._record("run_command", command, timeout_seconds=timeout_seconds)

    async def start_command(self, command: str) -> str:
        return await self._record("start_command", command)

    async def check_command(self, command_id: str) -> str:
        return await self._record("check_command", command_id)

    async def stop_command(self, command_id: str) -> str:
        return await self._record("stop_command", command_id)

    async def close(self) -> None:
        self.closed += 1


class _RecordingSandbox:
    def __init__(self) -> None:
        self.sessions: list[_RecordingSession] = []

    def __call__(
        self,
        root: Path,
        *,
        resources: tuple[WorkspaceResource, ...] = (),
    ) -> SupportsFileTools:
        del root, resources
        session = _RecordingSession()
        self.sessions.append(session)
        return session


class _FailingCloseSession(_RecordingSession):
    async def close(self) -> None:
        self.closed += 1
        raise RuntimeError("close failed")


class _BlockingCloseSession(_RecordingSession):
    def __init__(self) -> None:
        super().__init__()
        self.close_started = asyncio.Event()
        self.close_release = asyncio.Event()

    async def close(self) -> None:
        self.closed += 1
        self.close_started.set()
        await self.close_release.wait()


class _SpoofedSandboxCapability(AbstractCapability[object]):
    id = "linktools.workspace-sandbox"


def _tool_contract(tool: object) -> dict[str, object]:
    definition = tool.tool_def  # type: ignore[attr-defined]
    return {
        "version": 1,
        "revision": next(item.revision for item in filesystem_tool_declarations()
                         if item.name == definition.name),
        "description": definition.description,
        "parameters": definition.parameters_json_schema,
        "return_schema": definition.return_schema,
        "strict": definition.strict,
        "metadata": definition.metadata,
    }


def test_workspace_constructor_normalizes_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    workspace = Workspace(Path("project"), {})
    assert workspace.root == (tmp_path / "project").resolve()


def test_workspace_tool_contributions_are_stable_and_classified(tmp_path: Path) -> None:
    workspace = Workspace.load(tmp_path)
    contributions = _workspace_tool_contributions(workspace)

    assert tuple(sorted(item.id for item in contributions)) == (
        "check_command",
        "list_files",
        "read_file",
        "run_command",
        "search_files",
        "start_command",
        "stop_command",
        "write_file",
    )
    assert all(item.kind == "tool" for item in contributions)
    assert all(item.revision == 1 for item in contributions)
    assert {item.id: tool_class_from_metadata(item.value.tool_def.metadata) for item in contributions} == {
        **{item.name: item.metadata["linktools.ai.tool_class"] for item in filesystem_tool_declarations()},
        **{name: "shell" for name in ("check_command", "run_command", "start_command", "stop_command")},
    }
    assert tuple(item.revision for item in contributions) == tuple(
        item.revision for item in _workspace_tool_contributions(workspace)
    )


@pytest.mark.parametrize(
    ("selectors", "tool_classes"),
    (
        (("file:read",), {"filesystem.read"}),
        (("file:write",), {"filesystem.write"}),
        (("file:*",), {"filesystem.read", "filesystem.write"}),
        (("terminal:*",), {"shell"}),
        (("file:read", "read_file"), {"filesystem.read"}),
    ),
)
@pytest.mark.asyncio
async def test_workspace_selector_expands_registered_tool_declarations(
    tmp_path: Path,
    selectors: tuple[str, ...],
    tool_classes: set[str],
) -> None:
    workspace = Workspace.load(tmp_path)
    snapshot = await workspace_features("workspace", workspace=workspace).capture()
    spec = AgentSpec(
        "agent",
        allow_tools=selectors,
        allow_skills=(),
        allow_subagents=(),
        allow_capabilities=(),
    )
    compiler = AgentCompiler(
        models={"default": create_model(api='openai', model="gpt-test")},
        candidates=snapshot.contributions,
        agents={"agent": spec},
    )

    definition = compiler.compile(spec)
    expected = {
        declaration.name
        for declaration in filesystem_tool_declarations()
        if tool_class_from_metadata(declaration.metadata) in tool_classes
    }
    if "shell" in tool_classes:
        expected.update(tool.name for tool in ShellCapability().tool_definitions())
    assert {item.id for item in definition.selected_tools} == expected


@pytest.mark.asyncio
async def test_workspace_selector_validation_and_candidate_boundaries(
    tmp_path: Path,
) -> None:
    with pytest.raises(AIError) as error:
        AgentSpec("agent", allow_tools=("*", "file:delete"))
    assert error.value.code is ErrorCode.CAPABILITY_RESOLUTION_INVALID

    workspace = Workspace.load(tmp_path)
    snapshot = await workspace_features("workspace", workspace=workspace).capture()
    for selectors in (("new_tool",), ("*", "new_tool"), ("edit_file",), ("*", "edit_file")):
        spec = AgentSpec("agent", allow_tools=selectors)
        compiler = AgentCompiler(
            models={"default": create_model(api='openai', model="gpt-test")},
            candidates=snapshot.contributions,
            agents={"agent": spec},
        )
        with pytest.raises(AIError) as error:
            compiler.compile(spec)
        assert error.value.code is ErrorCode.CAPABILITY_RESOLUTION_INVALID

    for field in ("allow_skills", "allow_subagents", "allow_capabilities"):
        kwargs = {
            "allow_tools": (),
            "allow_skills": (),
            "allow_subagents": (),
            "allow_capabilities": (),
            field: ("*", "missing"),
        }
        spec = AgentSpec("agent", **kwargs)
        compiler = AgentCompiler(
            models={"default": create_model(api='openai', model="gpt-test")},
            candidates=(),
            agents={"agent": spec},
        )
        with pytest.raises(AIError) as error:
            compiler.compile(spec)
        assert error.value.code is ErrorCode.CAPABILITY_RESOLUTION_INVALID

    empty = AgentSpec("agent", allow_tools=())
    empty_compiler = AgentCompiler(
        models={"default": create_model(api='openai', model="gpt-test")},
        candidates=(),
        agents={"agent": empty},
    )
    assert empty_compiler.compile(empty).selected_tools == ()


def test_global_tool_wildcard_preserves_exact_mcp_requirement() -> None:
    server = MCPServerSpec("server", "python")
    exact = mcp_tool_selector(server.id, "required")
    spec = AgentSpec(
        "agent",
        allow_tools=("*", exact),
        allow_skills=(),
        allow_subagents=(),
        allow_capabilities=(),
    )
    compiler = AgentCompiler(
        models={"default": create_model(api='openai', model="gpt-test")},
        candidates=(CapabilityContribution.from_declaration(server),),
        agents={"agent": spec},
    )

    definition = compiler.compile(spec)

    assert mcp_server_selector(server.id) in definition.mcp_policy
    assert exact in definition.mcp_policy


@pytest.mark.asyncio
async def test_workspace_group_preserves_custom_asset_path_discovery(tmp_path: Path) -> None:
    workspace = Workspace.load(tmp_path / "workspace")
    declaration_root = tmp_path / "declarations"
    agent_dir = declaration_root / "custom-agents"
    agent_dir.mkdir(parents=True)
    audit = agent_dir / "audit" / "AGENT.md"
    audit.parent.mkdir(parents=True)
    audit.write_text("---\nmodel: default\n---\n", encoding="utf-8")
    backend = DirectoryAssetBackend(
        str(declaration_root),
        path_adapter=PrefixAssetPathAdapter(
            {
                "agent": "custom-agents",
                "skill": "custom-skills",
                "mcp": "custom-mcp",
            }
        ),
        kinds=("agent", "skill", "mcp"),
    )
    store = AssetStore(StorageOverlay(backend))
    await store.initialize()
    try:
        snapshot = await workspace_features(
            "workspace",
            workspace=workspace,
            assets=store,
        ).capture()
    finally:
        await store.close()

    identities = {
        (item.kind, item.id)
        for item in snapshot.contributions
    }
    assert ("agent", "audit") in identities
    assert ("tool", "read_file") in identities
    assert ("tool", "run_command") in identities


@pytest.mark.asyncio
async def test_filesystem_tool_declarations_do_not_depend_on_sandbox_selection(
    tmp_path: Path,
) -> None:
    sandbox = _RecordingSandbox()
    workspace = Workspace.load(tmp_path)
    groups = (
        workspace_features("workspace", workspace=workspace),
        workspace_features("workspace", workspace=workspace, workspace_factory=sandbox),
        workspace_features("workspace", workspace=workspace, workspace_factory=None),
    )
    snapshots = [await group.capture() for group in groups]
    projected = tuple(
        tuple(
            (item.id, item.revision, item.contract)
            for item in snapshot.contributions
        )
        for snapshot in snapshots
    )
    assert projected[0] == projected[1] == projected[2]
    assert sandbox.sessions == []


def test_filesystem_capabilities_materialize_one_sandbox_group(tmp_path: Path) -> None:
    workspace = Workspace.load(tmp_path)

    with pytest.raises(AIError) as raised:
        filesystem_capabilities(workspace, ("read_file",))
    assert raised.value.code is ErrorCode.SANDBOX_SESSION_CLOSED
    assert filesystem_capabilities(workspace, ()) == ()


def test_filesystem_capabilities_reject_unknown_tool_names(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown workspace tools"):
        filesystem_capabilities(Workspace.load(tmp_path), ("missing_tool",))


def test_workspace_sandbox_capability_id_is_reserved() -> None:
    group = RuntimeFeatures[object]("custom")
    with pytest.raises(AIError) as raised:
        group.capability(_SpoofedSandboxCapability())
    assert raised.value.code is ErrorCode.CAPABILITY_RESOLUTION_INVALID


@pytest.mark.asyncio
async def test_workspace_runtime_tool_contracts_match_durable_contributions(tmp_path: Path) -> None:
    sandbox = _RecordingSandbox()
    workspace = Workspace.load(tmp_path)
    contributions = _workspace_tool_contributions(workspace)
    expected = {item.id: item.contract for item in contributions
                if item.contract["metadata"]["linktools.ai.tool_class"] != "shell"}
    capability = filesystem_capabilities(
        workspace,
        (
                                    "list_files",
            "read_file",
            "search_files",
            "write_file",
        ),
        backend=sandbox(root=workspace.root),
    )[0]
    run_toolset = capability.get_toolset()

    assert {
        name: _tool_contract(tool)
        for name, tool in run_toolset.tools.items()  # type: ignore[attr-defined]
    } == expected
    await run_toolset.__aexit__(None, None, None)


@pytest.mark.asyncio
async def test_workspace_capability_uses_the_caller_owned_session(tmp_path: Path) -> None:
    sandbox = _RecordingSandbox()
    workspace = Workspace.load(tmp_path)
    capability = filesystem_capabilities(
        workspace,
        ("read_file",),
        backend=sandbox(root=workspace.root),
    )[0]
    toolset = capability.get_toolset()

    assert len(sandbox.sessions) == 1
    await toolset.tools["read_file"].function("sample.txt")  # type: ignore[attr-defined]
    assert [name for name, _, _ in sandbox.sessions[0].calls] == ["read_text"]
    await toolset.__aexit__(None, None, None)
    await sandbox.sessions[0].close()
    assert [session.closed for session in sandbox.sessions] == [1]


@pytest.mark.parametrize(
    ("decision", "expected_error"),
    (("deny", ToolCallFailed), ("ask", ApprovalRequired)),
)
@pytest.mark.asyncio
async def test_permission_rejection_has_no_sandbox_operation_side_effect(
    tmp_path: Path,
    decision: str,
    expected_error: type[BaseException],
) -> None:
    sandbox = _RecordingSandbox()
    workspace = Workspace.load(tmp_path)
    session = sandbox(root=workspace.root)
    capability = filesystem_capabilities(
        workspace,
        ("read_file",),
        backend=session,
    )[0]
    run_toolset = capability.get_toolset()
    boundary = BoundaryToolset(
        (run_toolset,),
        {
            "read_file": ManagedToolDescriptor(
                effect_owner="none",
                effect_policy="none",
                tool_class="filesystem.read",
                workspace_path_fields=("path",),
            )
        },
        id="workspace-boundary",
            permission_policy=ToolPermissionPolicy(
                (ToolPermissionRule(decision, tool_name="read_file"),)  # type: ignore[arg-type]
            ),
        workspace_backend=session,
    )
    context = RunContext(
        deps=None,
        model=TestModel(),
        usage=RunUsage(),
        run_id="run",
        tool_call_id="call",
    )
    tools = await boundary.get_tools(context)

    with pytest.raises(expected_error):
        await boundary.call_tool(
            "read_file",
            {"path": "sample.txt"},
            context,
            tools["read_file"],
        )
    assert sandbox.sessions[0].calls == []
    await run_toolset.__aexit__(None, None, None)
    await session.close()


@pytest.mark.asyncio
async def test_custom_sandbox_does_not_fallback_to_host_filesystem(tmp_path: Path) -> None:
    sandbox = _RecordingSandbox()
    workspace = Workspace.load(tmp_path)
    session = sandbox(root=workspace.root)
    capability = filesystem_capabilities(
        workspace,
        ("write_file",),
        backend=session,
    )[0]
    run_toolset = capability.get_toolset()

    await run_toolset.tools["write_file"].function("host.txt", "content")  # type: ignore[attr-defined]
    assert not (tmp_path / "host.txt").exists()
    assert sandbox.sessions[0].calls == [
        ("write_file", ("host.txt", "content"), {"old_text": None, "expected_hash": None})
    ]
    await run_toolset.__aexit__(None, None, None)
    await session.close()


@pytest.mark.asyncio
async def test_workspace_sandbox_close_failure_propagates_without_primary_error(tmp_path: Path) -> None:
    session = _FailingCloseSession()

    with pytest.raises(RuntimeError, match="close failed"):
        await session.close()
    assert session.closed == 1


@pytest.mark.asyncio
async def test_workspace_sandbox_close_is_completed_during_cancellation(tmp_path: Path) -> None:
    session = _BlockingCloseSession()
    closing = asyncio.create_task(session.close())
    await session.close_started.wait()

    closing.cancel()
    await asyncio.sleep(0)
    session.close_release.set()
    with pytest.raises(asyncio.CancelledError):
        await closing
    assert session.closed == 1


@pytest.mark.asyncio
async def test_disabled_sandbox_fails_before_workspace_tool_execution(tmp_path: Path) -> None:
    workspace = Workspace.load(tmp_path)
    group = workspace_features("workspace", workspace=workspace, workspace_factory=None)

    assert group.workspace_factory is None
    with pytest.raises(AIError) as raised:
        filesystem_capabilities(workspace, ("read_file",), backend=None)
    assert raised.value.code is ErrorCode.SANDBOX_SESSION_CLOSED


@pytest.mark.asyncio
async def test_workspace_group_does_not_discover_declarations(
    tmp_path: Path,
) -> None:
    storage = tmp_path / ".linktools"
    agent = storage / "agents" / "broken"
    agent.parent.mkdir(parents=True)
    agent.write_text("not a valid agent declaration", encoding="utf-8")
    sandbox = _RecordingSandbox()
    workspace = Workspace.load(tmp_path)

    group = workspace_features("workspace", workspace=workspace, workspace_factory=sandbox)
    snapshot = await group.capture()

    assert group.workspace is workspace
    assert snapshot.workspace_factory is sandbox
    assert snapshot
    assert all(item.kind == "tool" or (item.kind == "capability" and item.id in {"runtime.filesystem", "runtime.shell"})
               for item in snapshot.contributions)
    assert sandbox.sessions == []
