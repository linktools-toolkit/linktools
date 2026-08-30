#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regression coverage for cancellation close failure propagation."""

from types import SimpleNamespace

import pytest

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime._local import LocalExecutionBackend, _WorkerFailure


@pytest.mark.asyncio
async def test_local_execution_close_surfaces_recorded_worker_failure() -> None:
    backend = object.__new__(LocalExecutionBackend)
    backend._accepting = True
    backend._tasks = {}
    backend._executor = SimpleNamespace(pending_background_tasks=())
    backend._subagent_dispatcher = None
    backend._checkpoint_tasks = set()
    backend._execution_durable_tasks = {}
    backend._worker_failures = {
        "execution": _WorkerFailure(
            ErrorCode.STORAGE_INTEGRITY_ERROR,
            {"phase": "local_execution_worker", "execution_id": "execution"},
        )
    }

    with pytest.raises(AIError) as error:
        await backend.close()

    assert error.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
    assert error.value.safe_details == {
        "phase": "local_execution_worker",
        "execution_id": "execution",
    }
    assert backend._worker_failures
