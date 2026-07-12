#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""BuiltinProvider (contract): resolves builtin:file/terminal into the file/
terminal FunctionToolset, and rejects resolution without an execution backend."""

import pytest

from linktools.ai.capability.builtin import BuiltinProvider
from linktools.ai.capability.exposure import CapabilityToolExposurePolicy
from linktools.ai.capability.provider import CapabilityContext
from linktools.ai.capability.models import CapabilityRef
from linktools.ai.errors import CapabilityNotFoundError, CapabilityResolutionError
from linktools.ai.execution.local import LocalExecutionBackend


def _ctx(execution, agent_id="a1"):
    return CapabilityContext(
        agent_id=agent_id,
        exposure_policy=CapabilityToolExposurePolicy(),
        execution=execution,
    )


@pytest.mark.asyncio
async def test_builtin_file_exposes_only_file_tools(tmp_path):
    backend = LocalExecutionBackend(runtime_dir=str(tmp_path))
    bundle = await BuiltinProvider().resolve(
        CapabilityRef("builtin", "file"), _ctx(backend)
    )
    names = tuple(
        md.descriptor.name for c in bundle.tool_contributions for md in c.tools
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
    backend = LocalExecutionBackend(runtime_dir=str(tmp_path))
    bundle = await BuiltinProvider().resolve(
        CapabilityRef("builtin", "terminal"), _ctx(backend)
    )
    assert tuple(
        md.descriptor.name for c in bundle.tool_contributions for md in c.tools
    ) == ("bash",)


@pytest.mark.asyncio
async def test_builtin_wildcard_exposes_both(tmp_path):
    backend = LocalExecutionBackend(runtime_dir=str(tmp_path))
    bundle = await BuiltinProvider().resolve(
        CapabilityRef("builtin", "*"), _ctx(backend)
    )
    names = {md.descriptor.name for c in bundle.tool_contributions for md in c.tools}
    assert "bash" in names and "read_file" in names


@pytest.mark.asyncio
async def test_builtin_without_execution_backend_raises(tmp_path):
    with pytest.raises(CapabilityResolutionError, match="execution backend"):
        await BuiltinProvider().resolve(CapabilityRef("builtin", "file"), _ctx(None))


@pytest.mark.asyncio
async def test_builtin_unknown_name_raises(tmp_path):
    backend = LocalExecutionBackend(runtime_dir=str(tmp_path))
    with pytest.raises(CapabilityNotFoundError, match="unknown builtin"):
        await BuiltinProvider().resolve(CapabilityRef("builtin", "nope"), _ctx(backend))


@pytest.mark.asyncio
async def test_builtin_file_read_exposes_only_read_tools(tmp_path):
    backend = LocalExecutionBackend(runtime_dir=str(tmp_path))
    bundle = await BuiltinProvider().resolve(
        CapabilityRef("builtin", "file-read"), _ctx(backend)
    )
    names = {md.descriptor.name for c in bundle.tool_contributions for md in c.tools}
    assert names == {"list_dir", "read_file"}
    # read-only categories on the descriptors
    cats = {md.descriptor.category for c in bundle.tool_contributions for md in c.tools}
    assert cats == {"file-read"}


@pytest.mark.asyncio
async def test_builtin_file_write_exposes_only_write_tools(tmp_path):
    backend = LocalExecutionBackend(runtime_dir=str(tmp_path))
    bundle = await BuiltinProvider().resolve(
        CapabilityRef("builtin", "file-write"), _ctx(backend)
    )
    names = {md.descriptor.name for c in bundle.tool_contributions for md in c.tools}
    assert names == {"write_file", "batch_files", "apply_patch"}
    descs = tuple(md.descriptor for c in bundle.tool_contributions for md in c.tools)
    assert all(d.mutating for d in descs)
    assert {d.category for d in descs} == {"file-write"}


@pytest.mark.asyncio
async def test_builtin_file_maps_to_read_plus_write(tmp_path):
    """builtin:file is a legitimate ref mapping to read + write tools (subject
    to Exposure Policy)."""
    backend = LocalExecutionBackend(runtime_dir=str(tmp_path))
    bundle = await BuiltinProvider().resolve(
        CapabilityRef("builtin", "file"), _ctx(backend)
    )
    names = {md.descriptor.name for c in bundle.tool_contributions for md in c.tools}
    assert {
        "list_dir",
        "read_file",
        "write_file",
        "batch_files",
        "apply_patch",
    } == names
