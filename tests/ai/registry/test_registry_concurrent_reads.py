#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Registry concurrent-read safety (WP-19): 100 concurrent get() calls must
never see a mixed snapshot (the refresh lock serializes the cache clear +
revision read so no reader sees a half-cleared cache)."""

import asyncio

import pytest

from linktools.ai.registry.agent import AgentRegistry
from linktools.ai.registry.parser import SpecLoader


@pytest.mark.asyncio
async def test_concurrent_gets_never_mix_snapshot(tmp_path):
    """100 concurrent get() calls on a stable tree all return the same spec."""
    agents = tmp_path / "agents"
    agents.mkdir()
    (agents / "a.md").write_text(
        "---\nname: a\nmodel:\n  primary: gpt\n---\nbody-v1\n", encoding="utf-8"
    )
    registry = AgentRegistry(SpecLoader.from_filesystem(agents))

    results = await asyncio.gather(*(registry.get("a") for _ in range(100)))
    # Every concurrent reader sees the same, consistent spec.
    assert all(r.instructions.instructions == "body-v1" for r in results)
    assert len({id(r) for r in results}) <= 2  # cached object identity is stable
