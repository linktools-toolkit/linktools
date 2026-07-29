#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""BuiltinToolProvider (contract): resolves builtin:file/terminal into the file/
terminal FunctionToolset, and rejects resolution without a sandbox."""

import pytest

from linktools.ai.agent.tool.providers.builtin import BuiltinToolProvider
from linktools.ai.agent.tool.exposure import ToolExposurePolicy
from linktools.ai.agent.assembly.provider import AgentFeatureContext
from linktools.ai.agent.assembly.models import AgentFeatureRef
from linktools.ai.errors import AgentFeatureNotFoundError, AgentAssemblyError
from linktools.ai.agent.tool.sandbox.local import LocalSandbox


def _ctx(sandbox, agent_id="a1"):
    return AgentFeatureContext(
        agent_id=agent_id,
        execution_id="e1",
        root_execution_id="e1",
        parent_execution_id=None,
        session_id="s1",
        tenant_id="t1",
        user_id="u1",
        workspace=None,
        sandbox=sandbox,
    )


@pytest.mark.asyncio
async def test_builtin_file_exposes_only_file_tools(tmp_path):
    backend = LocalSandbox(runtime_dir=str(tmp_path))
    bundle = await BuiltinToolProvider().resolve(
        AgentFeatureRef("builtin", "file"), _ctx(backend)
    )
    names = tuple(
        md.descriptor.name for md in bundle.tools
    )
    assert set(names) == {
        "list_dir",
        "read_file",
        "write_file",
        "batch_files",
        "apply_patch",
    }


@pytest.mark.asyncio
async def test_builtin_terminal_exposes_only_bash(tmp_path):
    backend = LocalSandbox(runtime_dir=str(tmp_path))
    bundle = await BuiltinToolProvider().resolve(
        AgentFeatureRef("builtin", "terminal"), _ctx(backend)
    )
    assert tuple(
        md.descriptor.name for md in bundle.tools
    ) == ("bash",)


@pytest.mark.asyncio
async def test_builtin_wildcard_exposes_both(tmp_path):
    backend = LocalSandbox(runtime_dir=str(tmp_path))
    bundle = await BuiltinToolProvider().resolve(
        AgentFeatureRef("builtin", "*"), _ctx(backend)
    )
    names = {md.descriptor.name for md in bundle.tools}
    assert "bash" in names and "read_file" in names


@pytest.mark.asyncio
async def test_builtin_without_execution_backend_raises(tmp_path):
    with pytest.raises(AgentAssemblyError, match="requires a sandbox"):
        await BuiltinToolProvider().resolve(AgentFeatureRef("builtin", "file"), _ctx(None))


@pytest.mark.asyncio
async def test_builtin_unknown_name_raises(tmp_path):
    backend = LocalSandbox(runtime_dir=str(tmp_path))
    with pytest.raises(AgentFeatureNotFoundError, match="unknown builtin"):
        await BuiltinToolProvider().resolve(AgentFeatureRef("builtin", "nope"), _ctx(backend))


@pytest.mark.asyncio
async def test_builtin_file_read_exposes_only_read_tools(tmp_path):
    backend = LocalSandbox(runtime_dir=str(tmp_path))
    bundle = await BuiltinToolProvider().resolve(
        AgentFeatureRef("builtin", "file-read"), _ctx(backend)
    )
    names = {md.descriptor.name for md in bundle.tools}
    assert names == {"list_dir", "read_file"}
    # read-only categories on the descriptors
    cats = {md.descriptor.category for md in bundle.tools}
    assert cats == {"file-read"}


@pytest.mark.asyncio
async def test_builtin_file_write_exposes_only_write_tools(tmp_path):
    backend = LocalSandbox(runtime_dir=str(tmp_path))
    bundle = await BuiltinToolProvider().resolve(
        AgentFeatureRef("builtin", "file-write"), _ctx(backend)
    )
    names = {md.descriptor.name for md in bundle.tools}
    assert names == {"write_file", "batch_files", "apply_patch"}
    descs = tuple(md.descriptor for md in bundle.tools)
    assert all(d.mutating for d in descs)
    assert {d.category for d in descs} == {"file-write"}


@pytest.mark.asyncio
async def test_builtin_file_maps_to_read_plus_write(tmp_path):
    """builtin:file is a legitimate ref mapping to read + write tools (subject
    to Exposure Policy)."""
    backend = LocalSandbox(runtime_dir=str(tmp_path))
    bundle = await BuiltinToolProvider().resolve(
        AgentFeatureRef("builtin", "file"), _ctx(backend)
    )
    names = {md.descriptor.name for md in bundle.tools}
    assert {
        "list_dir",
        "read_file",
        "write_file",
        "batch_files",
        "apply_patch",
    } == names
