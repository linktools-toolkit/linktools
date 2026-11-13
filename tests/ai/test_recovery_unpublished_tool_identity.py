#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Split tool admission retains identity before a native event is published."""
from dataclasses import replace
from pathlib import Path

import pytest
from pydantic_ai_harness.step_persistence import RunRecord, StepEvent

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.execution import TOOL_OPERATION_ID_METADATA_KEY
from linktools.ai.persistence import Persistence, PersistenceStorage, PersistenceDomain
from linktools.ai.runtime._local import LocalExecutionBackend

from .test_tool_execution_boundary import _admit, _request


@pytest.mark.asyncio
@pytest.mark.parametrize("conflict", (False, True))
async def test_event_reference_lookup_does_not_scan_effect_ledger(tmp_path: Path, conflict: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    config = PersistenceStorage.sqlite(tmp_path / "recovery.db")
    async with Persistence.open("tool-boundary", tenant_id="tenant", storage=config) as state:
        receipt = await _admit(state)
        native = await state.create_step_store()
        await native.register_run(RunRecord(run_id="run"))
        assert await native.list_events(run_id="run") == []
        # An independent execution using the same provider call/run is not ours.
        await state.checkpoint.tools.admit(replace(_request(), execution_id="other", tool_operation_id="other"))
        await native.close()
    async with Persistence.open("tool-boundary", tenant_id="tenant", storage=config) as state:
        native = await state.create_step_store()
        state.run_store.bind_step_store(native)
        if conflict:
            await native.append_event(StepEvent(run_id="run", kind="tool_call_started", step_index=1,
                tool_name="tool", tool_call_id="call", idempotency_key="started",
                metadata={TOOL_OPERATION_ID_METADATA_KEY: "conflicting"}))
        backend = object.__new__(LocalExecutionBackend)
        backend._tenant_id = "tenant"
        backend._tool_operations = state.checkpoint.tools
        backend._run_stores = {domain: state.run_store.read_store(domain) for domain in
            (PersistenceDomain.CHECKPOINT, PersistenceDomain.TRACE, PersistenceDomain.CONVERSATION)}
        execution = replace(await state.execution.executions.get_fresh("execution", tenant_id="tenant"),
                            restart_checkpoints=True)
        async def forbidden(*args, **kwargs):
            raise AssertionError("ordinary event identity lookup must not scan the effect ledger")
        monkeypatch.setattr(state.checkpoint.tools, "list_by_agent_run", forbidden)
        assert await backend._tool_operation_references(execution, "run") == (
            {"call": "conflicting"} if conflict else {})
        current = await state.checkpoint.tools.get_operation("operation", tenant_id="tenant")
        assert current == receipt.operation.value
        assert current.fence == receipt.operation.value.fence
        await native.close()


def _crash_before_tool_event(database: str, effects: str, native_database: str) -> None:
    import os
    from linktools.ai.runtime._tool import RuntimeToolOperationBridge
    from ._session_tool_test_helpers import _crash_session_process

    os.environ["LINKTOOLS_TEST_STEP_DATABASE"] = native_database
    publish = RuntimeToolOperationBridge._publish_native_event

    async def crash(self, event, **kwargs):
        if event.kind == "tool_call_started":
            os._exit(91)
        return await publish(self, event, **kwargs)

    RuntimeToolOperationBridge._publish_native_event = crash
    _crash_session_process(database, effects, "sqlite", "before_tool_event")


@pytest.mark.asyncio
async def test_public_split_recovery_preserves_unknown_admission_after_process_exit(tmp_path: Path) -> None:
    import asyncio
    import multiprocessing
    from linktools.ai.capability import TraceCapability
    from linktools.ai.core import ToolOperationStatus
    from linktools.ai.runtime import Runtime
    from ._runtime_test_helpers import native_checkpoint, model_features
    from ._session_tool_test_helpers import _application, _ToolModels

    database, effects, native_database = (tmp_path / name for name in ("runtime.db", "effects", "native.db"))
    process = multiprocessing.get_context("spawn").Process(target=_crash_before_tool_event,
        args=(str(database), str(effects), str(native_database)))
    process.start()
    try:
        await asyncio.to_thread(process.join, 30)
        assert process.exitcode == 91
    finally:
        if process.is_alive():
            process.kill()
            await asyncio.to_thread(process.join)
    assert not effects.exists()
    calls: list[str] = []
    app = _application(calls, effect_policy="non_replay_safe", effect_log=effects)
    app.capability(native_checkpoint(database=native_database), scope="runtime")
    app.capability(TraceCapability(), scope="runtime")
    async with Persistence.open("session-tool-crash", storage=PersistenceStorage.sqlite(database)) as state:
        async with Runtime.attach(persistence=state, features=model_features(_ToolModels(), app), auto_recover=False) as runtime:
            execution = await runtime.agents.get("default").session("session").start("inspect", idempotency_key="turn-1")
            tenant = runtime.default_principal.tenant_id
            before = await state.checkpoint.tools.list_by_execution(execution.execution_id, tenant_id=tenant)
            assert len(before) == 1 and before[0].status is ToolOperationStatus.CLAIMED
            await execution.recover()
            with pytest.raises(AIError) as error:
                await execution.wait(timeout_seconds=15)
            assert error.value.code is ErrorCode.TOOL_EFFECT_UNKNOWN
            after = await state.checkpoint.tools.list_by_execution(execution.execution_id, tenant_id=tenant)
            assert len(after) == 1
            assert after[0].tool_operation_id == before[0].tool_operation_id
            assert after[0].fence == before[0].fence
            assert after[0].status is ToolOperationStatus.EFFECT_UNKNOWN
            assert calls == [] and not effects.exists()
            with pytest.raises(AIError) as repeated:
                await execution.recover()
                await execution.wait(timeout_seconds=15)
            assert repeated.value.code is ErrorCode.TOOL_EFFECT_UNKNOWN
            final = await state.checkpoint.tools.list_by_execution(execution.execution_id, tenant_id=tenant)
            assert final == after
            assert calls == [] and not effects.exists()
