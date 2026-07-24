#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""2 (v4 guide ): RunDefinitionStore is a required capability, not
optional. These tests pin the two guarantees the spec states in :

1. build_runtime fails fast when Storage has no RunDefinitionStore --
   the error surfaces at build time, not when a subagent/worker tool first
   pauses on approval and Runtime.resume(child_run_id) cannot find a snapshot.
2. A subagent run always persists a RunDefinitionSnapshot for its child run
 -- the unconditional prepare_agent_run in the subagent executor.
"""

import asyncio
from dataclasses import dataclass

import pytest

from linktools.ai.subagent.executor import SubagentExecutor
from linktools.ai.agent.spec import AgentSpec, PromptSpec
from linktools.ai.errors import RuntimeInitializationError
from linktools.ai.model.policy import ModelPolicy
from linktools.ai.run.models import RunInput
from linktools.ai.runtime import Runtime, build_runtime
from linktools.ai.storage.facade import FilesystemStorage
from linktools.ai.storage.filesystem.commit import FilesystemRunCommitCoordinator


def test_runtime_build_rejects_storage_without_run_definitions(tmp_path):
    """a Storage whose run_definitions is None must be rejected at build
    time with RuntimeInitializationError, before any run/resume is attempted."""
    storage = FilesystemStorage(root=tmp_path)
    # run_definitions is a required field now, but a caller can still pass None
    # explicitly -- that must fail fast rather than silently disabling resume.
    object.__setattr__(storage, "run_definitions", None)

    with pytest.raises(RuntimeInitializationError):
        build_runtime(
            storage=storage,
            commit_coordinator=FilesystemRunCommitCoordinator.from_storage(storage),
        )


@dataclass
class _FakeResult:
    output: object


class _FakeCompiler:
    async def compile(self, spec):
        return ("compiled", spec)


class _FakeRunner:
    async def dispatch(self, request):
        assert isinstance(request.input, RunInput)
        return _FakeResult(output="child-output")


def test_subagent_run_persists_child_run_definition_snapshot(tmp_path):
    """a subagent (child) run persists a RunDefinitionSnapshot so a later
    Runtime.resume(child_run_id) can restore its spec + identity after an
    approval pause. The subagent executor must call prepare_agent_run for every
    child run -- this test fails if that call is re-gated behind an Optional."""
    storage = FilesystemStorage(root=tmp_path)
    executor = SubagentExecutor(
        storage=storage,
        compiler=_FakeCompiler(),
        dispatcher=_FakeRunner(),
    )
    spec = AgentSpec(
        id="child-agent",
        name="child",
        model=ModelPolicy(primary="any"),
        instructions=PromptSpec(instructions="do the work"),
    )

    async def _run():
        # parent=None -> the child run is its own root. The executor mints the
        # child run id and returns it on SubagentResult.run_id.
        result = await executor.execute(
            agent_spec=spec,
            task="hi",
            context=None,
            parent=None,
            scope=None,
            timeout_seconds=None,
        )
        assert result.status == "succeeded"
        snapshot = await storage.run_definitions.get(result.run_id)
        assert snapshot is not None, "subagent run persisted no definition snapshot"
        assert snapshot.runnable_id == "child-agent"
        # The fingerprint is recomputed from the spec on resume and must match,
        # else resume refuses (tamper/drift detection).
        from linktools.ai.run.definition import spec_fingerprint

        assert snapshot.spec_fingerprint == spec_fingerprint(spec)
        return result

    asyncio.run(_run())
