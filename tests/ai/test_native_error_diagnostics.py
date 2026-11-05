#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Execution diagnostics and native events retain original exception text."""

from pathlib import Path

import pytest
from pydantic_ai_harness.step_persistence import InMemoryStepStore, StepPersistence
from sqlalchemy.ext.asyncio import create_async_engine

from linktools.ai.migrate import provision_database
from linktools.ai.persistence import Persistence, PersistenceStorage, SqlStepStore
from linktools.ai.runtime import Runtime, RuntimeFeatures

from ._runtime_test_helpers import _Models, _UsageFunctionModel, model_features


class _CustomStepPersistence(StepPersistence):
    pass


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "sql"))
@pytest.mark.parametrize("capability_type", (StepPersistence, _CustomStepPersistence))
async def test_native_and_execution_errors_keep_original_diagnostics_without_trace(
    tmp_path: Path, backend: str, capability_type: type[StepPersistence],
) -> None:
    message = "request failed: Authorization: Bearer fixture-token status=401"

    def model(_messages, _info):
        raise RuntimeError(message)

    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'state.db'}")
    await provision_database(engine)
    native = (SqlStepStore(engine, namespace="native-errors", tenant_id="default")
              if backend == "sql" else InMemoryStepStore())
    features = RuntimeFeatures("native-errors")
    features.capability(capability_type(store=native, capture_frontier=True))
    features.agent("default", model="default", allow_tools=())
    try:
        async with Persistence.open("native-errors", storage=PersistenceStorage.sql(engine)) as state:
            async with Runtime.attach(
                persistence=state, features=model_features(_Models(_UsageFunctionModel(model)), features),
            ) as runtime:
                session = await runtime.agents.get().create_session("session")
                execution = await session.start("question", idempotency_key="input")
                result = (await execution.wait(timeout_seconds=5)).result
                assert result is not None and result.error_diagnostics is not None
                assert result.error_diagnostics.exception_message == message
                assert message in result.error_diagnostics.traceback
        async with Persistence.open("native-errors", storage=PersistenceStorage.sql(engine)) as state:
            saved = await state.execution.executions.get(execution.execution_id, tenant_id="default")
            assert saved is not None and saved.error_diagnostics == result.error_diagnostics
        runs = await native.list_runs()
        assert len(runs) == 1
        events = await native.list_events(run_id=runs[0].run_id)
        errors = {event.kind: event.error for event in events if event.error is not None}
        assert errors == {
            "model_request_failed": repr(RuntimeError(message)),
            "run_failed": repr(RuntimeError(message)),
        }
        snapshot = await native.latest_snapshot(run_id=runs[0].run_id, include_interrupted=True)
        assert snapshot is not None and snapshot.messages
    finally:
        if isinstance(native, SqlStepStore):
            await native.close()
        await engine.dispose()
