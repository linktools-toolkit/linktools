#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Resource membership constrains the existing Sandbox read policy."""

import asyncio
import json
import sys
from pathlib import Path

import pytest
from pydantic_ai import Agent
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import FunctionModel

from linktools.ai.capability import AgentContext, PydanticToolControlCapability, filesystem_capabilities
from linktools.ai.core import service_principal
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.workspace import LocalWorkspaceBackend, WorkspaceReadPolicy, WorkspaceResource, Workspace
from linktools.ai.workspace.sandbox_worker import _dispatch

from ._runtime_test_helpers import is_tool_retry

pytestmark = pytest.mark.asyncio


@pytest.fixture
def resource_tree(tmp_path: Path) -> tuple[Path, WorkspaceResource]:
    root, package = tmp_path / "workspace", tmp_path / "package"
    root.mkdir()
    package.mkdir()
    names = ("visible.txt", "unlisted.txt", "nested/visible.txt", "nested/unlisted.txt",
             "blocked/declared.txt", "private/unlisted.txt")
    for name in names:
        target = package / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(f"synthetic {name}\n", encoding="utf-8")
    return root, WorkspaceResource("package", package, files={
        name: package / name for name in ("visible.txt", "nested/visible.txt", "blocked/declared.txt")
    })


def read_policy() -> WorkspaceReadPolicy:
    return WorkspaceReadPolicy((), {"package": ("visible.txt", "nested/**")})


@pytest.mark.parametrize("through_worker", (False, True))
@pytest.mark.parametrize("method", ("read_text", "inspect_file"))
async def test_file_reads_require_membership_and_read_policy(
    resource_tree: tuple[Path, WorkspaceResource], through_worker: bool, method: str,
) -> None:
    root, resource = resource_tree
    session = LocalWorkspaceBackend(root, read_policy=read_policy(), resources=(resource,))
    await session.working_dir()
    try:
        for relative in ("visible.txt", "nested/visible.txt"):
            path = str(resource.source / relative)
            result = (await _dispatch(session, method, {"path": path}) if through_worker
                      else await getattr(session, method)(path))
            assert "visible.txt" in str(result)
        for relative in ("unlisted.txt", "nested/unlisted.txt", "blocked/declared.txt"):
            path = str(resource.source / relative)
            with pytest.raises(AIError) as caught:
                if through_worker:
                    await _dispatch(session, method, {"path": path})
                else:
                    await getattr(session, method)(path)
            assert caught.value.code is ErrorCode.AUTHORIZATION_DENIED
            assert "synthetic" not in str(caught.value.safe_details)
        assert await session.read_bytes(str(resource.source / "visible.txt")) == b"synthetic visible.txt\n"
        with pytest.raises(PermissionError):
            await session.read_bytes(str(resource.source / "unlisted.txt"))

    finally:
        await session.close()


@pytest.mark.parametrize("through_worker", (False, True))
@pytest.mark.parametrize("method", ("list_files", "search_text"))
async def test_directory_results_only_expose_permitted_resource_members(
    resource_tree: tuple[Path, WorkspaceResource], through_worker: bool, method: str,
) -> None:
    root, resource = resource_tree
    session = LocalWorkspaceBackend(root, read_policy=read_policy(), resources=(resource,))
    await session.working_dir()
    try:
        params = {"path": str(resource.source)}
        if method.startswith("search"):
            params["pattern"] = "synthetic"
        elif method == "list_files":
            params["include_glob"] = "**"
        for directory in (resource.source, resource.source / "nested"):
            params["path"] = str(directory)
            result = (await _dispatch(session, method, params) if through_worker
                      else await getattr(session, method)(**params))
            assert "visible.txt" in result
            if directory == resource.source:
                assert "nested" in result
            assert all(name not in result for name in ("unlisted", "blocked", "private"))
        for directory in ("private", "blocked"):
            params["path"] = str(resource.source / directory)
            with pytest.raises(AIError) as caught:
                if through_worker:
                    await _dispatch(session, method, params)
                else:
                    await getattr(session, method)(**params)
            assert caught.value.code is ErrorCode.AUTHORIZATION_DENIED
    finally:
        await session.close()


async def test_resource_root_is_hidden_when_no_declared_member_is_permitted(
    resource_tree: tuple[Path, WorkspaceResource],
) -> None:
    root, resource = resource_tree
    session = LocalWorkspaceBackend(root, read_policy=WorkspaceReadPolicy(
        (), {"package": ("unlisted.txt",)},
    ), resources=(resource,))
    await session.working_dir()
    try:
        assert session.resource_path("package") is None
        with pytest.raises(AIError) as caught:
            await session.list_files(str(resource.source))
        assert caught.value.code is ErrorCode.AUTHORIZATION_DENIED
    finally:
        await session.close()


async def test_source_only_directory_resource_keeps_its_directory_grant(
    resource_tree: tuple[Path, WorkspaceResource],
) -> None:
    root, resource = resource_tree
    directory = WorkspaceResource("package", resource.source)
    session = LocalWorkspaceBackend(root, read_policy=WorkspaceReadPolicy(
        (), {"package": ("**",)},
    ), resources=(directory,))
    await session.working_dir()
    try:
        assert session.resource_path("package") == str(resource.source)
        assert await session.read_bytes(str(resource.source / "unlisted.txt")) == b"synthetic unlisted.txt\n"
    finally:
        await session.close()


async def test_missing_declared_member_does_not_prevent_other_authorized_reads(
    resource_tree: tuple[Path, WorkspaceResource],
) -> None:
    root, resource = resource_tree
    missing = resource.source / "blocked/declared.txt"
    missing.unlink()
    session = LocalWorkspaceBackend(root, read_policy=WorkspaceReadPolicy(
        (), {"package": ("**",)},
    ), resources=(resource,))
    await session.working_dir()
    try:
        assert await session.read_bytes(str(resource.source / "visible.txt")) == b"synthetic visible.txt\n"
        with pytest.raises(FileNotFoundError):
            await session.read_bytes(str(missing))

    finally:
        await session.close()


async def test_workspace_relative_grant_is_independent_of_resource_subset(tmp_path: Path) -> None:
    package = tmp_path / "package"
    package.mkdir()
    for name in ("listed.txt", "unlisted.txt"):
        (package / name).write_text(name, encoding="utf-8")
    session = LocalWorkspaceBackend(tmp_path, read_policy=WorkspaceReadPolicy(
        ("package/**",), {"package": ("**",)},
    ), resources=(WorkspaceResource(
        "package", package, files={"listed.txt": package / "listed.txt"},
    ),))
    await session.working_dir()
    try:
        assert await session.read_bytes("package/unlisted.txt") == b"unlisted.txt"
        with pytest.raises(PermissionError):
            await session.read_bytes(str(package / "unlisted.txt"))

    finally:
        await session.close()


async def test_nested_resource_keeps_the_existing_longest_root_selection(
    resource_tree: tuple[Path, WorkspaceResource],
) -> None:
    root, resource = resource_tree
    nested = resource.source / "nested"
    session = LocalWorkspaceBackend(root, read_policy=WorkspaceReadPolicy(
        (), {"package": ("**",), "child": ("**",)},
    ), resources=(WorkspaceResource("package", resource.source), WorkspaceResource(
        "child", nested, files={"visible.txt": nested / "visible.txt"},
    )))
    await session.working_dir()
    try:
        assert await session.read_bytes(str(nested / "visible.txt")) == b"synthetic nested/visible.txt\n"
        with pytest.raises(PermissionError):
            await session.read_bytes(str(nested / "unlisted.txt"))

    finally:
        await session.close()


async def test_native_workspace_rejects_unlisted_input_and_borrows_session(
    resource_tree: tuple[Path, WorkspaceResource],
) -> None:
    root, resource = resource_tree
    session = LocalWorkspaceBackend(root, read_policy=WorkspaceReadPolicy(
        (), {"package": ("**",)},
    ), resources=(resource,))
    await session.working_dir()
    requests = 0

    def respond(messages, info):
        nonlocal requests
        requests += 1
        if requests == 1:
            return ModelResponse(parts=[ToolCallPart("read_file", {
                "path": str(resource.source / "unlisted.txt"),
            }, tool_call_id="read")])
        assert any(is_tool_retry(part) for message in messages for part in message.parts)
        assert "synthetic unlisted.txt" not in str(messages)
        return ModelResponse(parts=[TextPart("denied")])

    try:
        caps = filesystem_capabilities(Workspace(root, {}), ("read_file",), backend=session)
        agent = Agent(model=FunctionModel(respond), capabilities=[PydanticToolControlCapability(), *caps])
        result = await agent.run("Read a file", deps=AgentContext(
            None, service_principal("tenant", "reader"), "resource-test", None, "one", {},
        ))
        assert result.output == "denied" and requests == 2
        assert await session.read_bytes(str(resource.source / "visible.txt")) == b"synthetic visible.txt\n"
    finally:
        await session.close()


async def test_sandbox_opens_reads_and_closes_without_importing_runtime(tmp_path: Path) -> None:
    (tmp_path / "note.txt").write_text("independent", encoding="utf-8")
    script = '''
import asyncio, json, sys
from pathlib import Path
from linktools.ai.workspace import LocalWorkspaceBackend
async def run():
    session = LocalWorkspaceBackend(Path(sys.argv[1]))
    try:
        assert (await session.read_text("note.txt"))["content"] == "independent"
    finally:
        await session.close()
    assert not any(n == "linktools.ai.runtime" or n.startswith("linktools.ai.runtime.") for n in sys.modules)
    print(json.dumps({"independent": True}))
asyncio.run(run())
'''
    process = await asyncio.create_subprocess_exec(
        sys.executable, "-c", script, str(tmp_path),
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await process.communicate()
    assert process.returncode == 0, stderr.decode()
    assert json.loads(stdout) == {"independent": True}
