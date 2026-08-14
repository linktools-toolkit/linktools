#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Focused RuntimeState and transaction invariant evidence."""

import asyncio

import pytest

from linktools.ai.runtime.state import RuntimeState
from linktools.ai.runtime.state._memory import build_in_memory_runtime


@pytest.mark.asyncio
async def test_configured_runtime_mutation_requires_active_transaction() -> None:
    runtime = build_in_memory_runtime(namespace="mutation-guard")
    await runtime.initialize()
    try:
        with pytest.raises(RuntimeError, match="storage mutation outside transaction"):
            runtime.components[0]._mark_changed()
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_runtime_state_close_is_idempotent_and_single_use(tmp_path) -> None:
    state = RuntimeState.filesystem(tmp_path / "runtime")
    await state.initialize(namespace="review", tenant_id="tenant")
    await state.close()
    await state.close()
    assert state.lifecycle == "closed"
    with pytest.raises(Exception):
        await state.initialize(namespace="review", tenant_id="tenant")
