from types import SimpleNamespace
from typing import Any, cast

import pytest
from pydantic_ai.tools import RunContext

from linktools.ai.runtime._capabilities import (
    _MissingToolOperationBridge,
    _RuntimeStepPersistence,
)


@pytest.mark.asyncio
async def test_runtime_step_persistence_for_run_preserves_config_and_resets_state() -> None:
    bridge = _MissingToolOperationBridge()
    background_tasks: set[Any] = set()
    pauses: list[int] = []
    pause_sink = pauses.append
    persistence = _RuntimeStepPersistence(
        tool_operations=bridge,
        run_id="run",
        plan_mode=True,
        background_tasks=background_tasks,
        deferred_pause_sink=pause_sink,
    )
    calls = persistence._calls
    persistence._last_observed_step_index = 7

    ctx = cast(RunContext[None], SimpleNamespace(run_id="ignored"))
    materialized = await persistence.for_run(ctx)

    assert isinstance(materialized, _RuntimeStepPersistence)
    assert materialized is not persistence
    assert materialized.tool_operations is bridge
    assert materialized.plan_mode is True
    assert materialized.background_tasks is background_tasks
    assert materialized.deferred_pause_sink is pause_sink
    assert materialized._calls == {}
    assert materialized._calls is not calls
    assert materialized._last_observed_step_index is None
