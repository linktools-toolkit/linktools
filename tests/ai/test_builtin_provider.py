#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""BuiltinProvider (spec §12): resolves builtin:file/terminal into the file/
terminal FunctionToolset, and rejects resolution without an execution backend."""

import pytest

from linktools.ai.capability import (
    BuiltinProvider,
    CapabilityContext,
    CapabilityToolExposurePolicy,
)
from linktools.ai.capability.ref import CapabilityRef
from linktools.ai.errors import CapabilityNotFoundError, CapabilityResolutionError
from linktools.ai.execution.local import LocalExecutionBackend


def _ctx(execution, agent_id="a1"):
    return CapabilityContext(
        agent_id=agent_id, exposure_policy=CapabilityToolExposurePolicy(),
        execution=execution,
    )


@pytest.mark.asyncio
async def test_builtin_file_exposes_only_file_tools(tmp_path):
    backend = LocalExecutionBackend(runtime_dir=str(tmp_path))
    bundle = await BuiltinProvider().resolve(CapabilityRef("builtin", "file"), _ctx(backend))
    names = tuple(bundle.toolsets[0].tools.keys())
    assert set(names) == {"list_dir", "read_file", "write_file", "batch_files", "apply_patch"}


@pytest.mark.asyncio
async def test_builtin_terminal_exposes_only_bash(tmp_path):
    backend = LocalExecutionBackend(runtime_dir=str(tmp_path))
    bundle = await BuiltinProvider().resolve(CapabilityRef("builtin", "terminal"), _ctx(backend))
    assert tuple(bundle.toolsets[0].tools.keys()) == ("bash",)


@pytest.mark.asyncio
async def test_builtin_wildcard_exposes_both(tmp_path):
    backend = LocalExecutionBackend(runtime_dir=str(tmp_path))
    bundle = await BuiltinProvider().resolve(CapabilityRef("builtin", "*"), _ctx(backend))
    names = set(bundle.toolsets[0].tools.keys())
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
    bundle = await BuiltinProvider().resolve(CapabilityRef("builtin", "file-read"), _ctx(backend))
    names = set(bundle.toolsets[0].tools.keys())
    assert names == {"list_dir", "read_file"}
    # read-only categories on the descriptors
    cats = {d.category for d in bundle.tool_contributions[0].descriptors}
    assert cats == {"file-read"}


@pytest.mark.asyncio
async def test_builtin_file_write_exposes_only_write_tools(tmp_path):
    backend = LocalExecutionBackend(runtime_dir=str(tmp_path))
    bundle = await BuiltinProvider().resolve(CapabilityRef("builtin", "file-write"), _ctx(backend))
    names = set(bundle.toolsets[0].tools.keys())
    assert names == {"write_file", "batch_files", "apply_patch"}
    descs = bundle.tool_contributions[0].descriptors
    assert all(d.mutating for d in descs)
    assert {d.category for d in descs} == {"file-write"}


@pytest.mark.asyncio
async def test_builtin_file_deprecated_maps_to_read_plus_write(tmp_path):
    """builtin:file is deprecated but still functional -- maps to read + write,
    still subject to Exposure Policy, and emits a DeprecationWarning."""
    backend = LocalExecutionBackend(runtime_dir=str(tmp_path))
    with pytest.warns(DeprecationWarning, match="deprecated"):
        bundle = await BuiltinProvider().resolve(CapabilityRef("builtin", "file"), _ctx(backend))
    names = set(bundle.toolsets[0].tools.keys())
    assert {"list_dir", "read_file", "write_file", "batch_files", "apply_patch"} == names
