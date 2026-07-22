#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for run.preparation: RunPreparationCoordinator builds + persists a real
ExecutionManifest into RunDefinitionSnapshot, and records a
resumability verdict. The persisted manifest must deserialize to a
valid ExecutionManifest carrying the spec's runnable id / model / tool
descriptor fingerprints -- not the old placeholder shape."""

import asyncio

from linktools.ai.agent.spec import AgentSpec, ModelPolicy, PromptSpec, ToolRef
from linktools.ai.run.context import RunContext
from linktools.ai.run.manifest import (
    Resumability,
    manifest_from_dict,
)
from linktools.ai.run.models import RunnableType
from linktools.ai.run.preparation import RunPreparationCoordinator
from linktools.ai.storage.filesystem.definition import FilesystemRunDefinitionStore


def _spec() -> AgentSpec:
    return AgentSpec(
        id="agent-prep",
        name="prep-agent",
        model=ModelPolicy(primary="test-model"),
        instructions=PromptSpec(instructions="hi"),
        tools=(ToolRef(kind="builtin", name="t1"), ToolRef(kind="builtin", name="t2")),
    )


def _context() -> RunContext:
    return RunContext(
        run_id="run-prep-1",
        root_run_id="run-prep-1",
        parent_run_id=None,
        session_id="session-prep",
        runnable_id="agent-prep",
        runnable_type=RunnableType.AGENT,
        user_id=None,
        tenant_id=None,
        workspace=None,
    )


def test_prepare_persists_real_execution_manifest(tmp_path):
    store = FilesystemRunDefinitionStore(root=tmp_path)
    coordinator = RunPreparationCoordinator(store)

    async def _run():
        return await coordinator.prepare_agent_run(spec=_spec(), context=_context())

    snapshot = asyncio.run(_run())

    # The persisted manifest deserializes to a real ExecutionManifest carrying
    # spec-level fields (runnable id/type, model name, tool descriptor
    # fingerprints) -- not the old {runnable_id, tools:[{kind,name}]} placeholder.
    manifest = manifest_from_dict(dict(snapshot.manifest))
    assert manifest.runnable_id == "agent-prep"
    assert manifest.runnable_type == "agent"
    assert manifest.model_name == "test-model"
    assert {t.name for t in manifest.tool_descriptors} == {"t1", "t2"}
    for tool in manifest.tool_descriptors:
        assert tool.descriptor_fingerprint and len(tool.descriptor_fingerprint) == 64

    # The snapshot records a resumability verdict (RESUMABLE in this phase --
    # versionability is determined from the compiled run in a follow-up).
    assert snapshot.resumability == Resumability.RESUMABLE.value

    # Round-trips through the store: the manifest + resumability survive get().
    fetched = asyncio.run(store.get("run-prep-1"))
    assert fetched is not None
    assert fetched.resumability == Resumability.RESUMABLE.value
    re_manifest = manifest_from_dict(dict(fetched.manifest))
    assert re_manifest.runnable_id == "agent-prep"
    assert {t.name for t in re_manifest.tool_descriptors} == {"t1", "t2"}


def test_prepare_manifest_descriptor_fingerprints_are_stable(tmp_path):
    # Re-preparing the same spec yields identical descriptor fingerprints.
    store = FilesystemRunDefinitionStore(root=tmp_path)
    coordinator = RunPreparationCoordinator(store)

    async def _prepare(rid):
        ctx = RunContext(
            run_id=rid,
            root_run_id=rid,
            parent_run_id=None,
            session_id="s",
            runnable_id="agent-prep",
            runnable_type=RunnableType.AGENT,
            user_id=None,
            tenant_id=None,
            workspace=None,
        )
        return await coordinator.prepare_agent_run(spec=_spec(), context=ctx)

    a = asyncio.run(_prepare("run-a"))
    b = asyncio.run(_prepare("run-b"))
    fps_a = {t.name: t.descriptor_fingerprint for t in manifest_from_dict(dict(a.manifest)).tool_descriptors}
    fps_b = {t.name: t.descriptor_fingerprint for t in manifest_from_dict(dict(b.manifest)).tool_descriptors}
    assert fps_a == fps_b
