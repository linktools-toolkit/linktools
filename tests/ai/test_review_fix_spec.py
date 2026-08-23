#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Focused RuntimeState and transaction invariant evidence."""


import pytest
from linktools.ai.runtime.state import RuntimeState


@pytest.mark.asyncio
async def test_configured_runtime_mutation_requires_active_transaction() -> None:
    state = RuntimeState.in_memory()
    await state.initialize(namespace="mutation-guard", tenant_id="tenant")
    try:
        with pytest.raises(RuntimeError, match="storage mutation outside transaction"):
            state.execution.executions._mark_changed()
    finally:
        await state.close()


@pytest.mark.asyncio
async def test_runtime_state_close_is_idempotent_and_single_use(tmp_path) -> None:
    state = RuntimeState.filesystem(tmp_path / "runtime")
    await state.initialize(namespace="review", tenant_id="tenant")
    await state.close()
    await state.close()
    assert state.ready is False
    with pytest.raises(Exception):
        await state.initialize(namespace="review", tenant_id="tenant")
