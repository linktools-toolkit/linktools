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
from linktools.ai.runtime.persistence.facade import FilesystemStorage
from linktools.ai.run.persistence.commit import FilesystemRunCommitCoordinator


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
    """Stands in for CoordinatorRunDispatcher: implements the two-step
    open_child + dispatch contract, creating the resumable snapshot in
    dispatch (mirroring RunCoordinator.dispatch_child) so this test exercises
    the real RunPreparationCoordinator primitive the production dispatcher uses."""

    def __init__(self, storage):
        from linktools.ai.run.preparation import RunPreparationCoordinator

        self._preparation = RunPreparationCoordinator(storage.run_definitions)

    async def open_child(self, parent_context, session_policy, metadata):
        import uuid

        from linktools.ai.run.dispatch import ChildRunHandle

        child = str(uuid.uuid4())
        return ChildRunHandle(
            run_id=child,
            session_id=str(uuid.uuid4()),
            root_run_id=(
                parent_context.root_run_id or parent_context.run_id or child
            ),
            parent_run_id=parent_context.run_id,
            parent_session_id=session_policy.parent_session_id,
            user_id=parent_context.user_id,
            tenant_id=parent_context.tenant_id,
            session_needs_create=True,
        )

    async def dispatch(self, request):
        from linktools.ai.run.context import RunContext
        from linktools.ai.run.models import RunnableType

        handle = request.handle
        compiled = request.compiled_agent
        spec = compiled[1] if isinstance(compiled, tuple) else compiled.spec
        context = RunContext(
            run_id=handle.run_id,
            root_run_id=handle.root_run_id,
            parent_run_id=handle.parent_run_id,
            session_id=handle.session_id,
            runnable_id=request.metadata.get("runnable_id") or spec.id,
            runnable_type=RunnableType.AGENT,
            user_id=handle.user_id,
            tenant_id=handle.tenant_id,
            workspace=None,
        )
        await self._preparation.prepare_agent_run(spec=spec, context=context)
        assert isinstance(request.input, RunInput)
        return _FakeResult(output="child-output")


def test_subagent_run_persists_child_run_definition_snapshot(tmp_path):
    """a subagent (child) run persists a RunDefinitionSnapshot so a later
    Runtime.resume(child_run_id) can restore its spec + identity after an
    approval pause. The snapshot is created by the dispatcher's dispatch step
    (RunCoordinator.dispatch_child -> prepare_agent_run); the executor only
    allocates the child id via open_child and hands the handle to dispatch --
    this test fails if that snapshot-creating step is dropped."""
    from linktools.ai.run.context import RunContext
    from linktools.ai.run.models import RunnableType

    storage = FilesystemStorage(root=tmp_path)
    executor = SubagentExecutor(
        storage=storage,
        compiler=_FakeCompiler(),
        dispatcher=_FakeRunner(storage),
    )
    spec = AgentSpec(
        id="child-agent",
        name="child",
        model=ModelPolicy(primary="any"),
        instructions=PromptSpec(instructions="do the work"),
    )
    parent = RunContext(
        run_id="parent-run",
        root_run_id="parent-run",
        parent_run_id=None,
        session_id="parent-session",
        runnable_id="parent-agent",
        runnable_type=RunnableType.AGENT,
        user_id=None,
        tenant_id=None,
        workspace=None,
    )

    async def _run():
        result = await executor.execute(
            agent_spec=spec,
            task="hi",
            context=None,
            parent=parent,
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
