#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Batch authoring inherits defaults once from the semantic owner."""

import pytest

from linktools.ai.capability import TraceCapability
from linktools.ai.runtime import RuntimeFeatures
from linktools.ai.core import CommitPolicy
from linktools.ai.storage import InMemoryCacheBackend


@pytest.mark.parametrize("options,trace_expected", [
    ({}, (128, 3.0)),
    ({"batch_size": 7}, (7, 3.0)),
    ({"flush_interval": 0.25}, (128, 0.25)),
    ({"batch_size": 7, "flush_interval": 0.25}, (7, 0.25)),
])
@pytest.mark.asyncio
async def test_trace_authoring_policy_resolves_owner_defaults(options, trace_expected):
    buffer = InMemoryCacheBackend(max_bytes=65536)
    policy = CommitPolicy.batch(buffer=buffer, **options)
    trace = TraceCapability(commit=policy)
    assert (trace.commit.batch_size, trace.commit.flush_interval) == trace_expected
    assert trace.commit.buffer is buffer
    assert policy.batch_size == options.get("batch_size")
    assert policy.flush_interval == options.get("flush_interval")
    features = RuntimeFeatures("trace-defaults")
    features.capability(trace)
    capture = await features.capture()
    declaration = next(item.value for item in capture.contributions if item.id == trace.id)
    assert declaration.commit == trace.commit
    assert declaration.commit.buffer is buffer


@pytest.mark.parametrize("options", [
    {"batch_size": 0}, {"batch_size": True}, {"batch_size": 1.5},
    {"flush_interval": 0}, {"flush_interval": True},
    {"flush_interval": float("inf")}, {"flush_interval": float("nan")},
])
def test_partial_batch_authoring_rejects_invalid_explicit_values(options):
    with pytest.raises(ValueError):
        CommitPolicy.batch(**options)


def test_custom_owner_resolves_only_missing_timing_and_preserves_other_modes():
    policy = CommitPolicy.batch(batch_size=5)
    assert policy.resolve_batch(batch_size=17, flush_interval=2) == CommitPolicy.batch(
        batch_size=5, flush_interval=2,
    )
    for policy in (CommitPolicy.sync(), CommitPolicy.terminal()):
        assert policy.resolve_batch(batch_size=17, flush_interval=2) is policy
    with pytest.raises(ValueError):
        CommitPolicy.batch().resolve_batch(batch_size=0, flush_interval=2)
    with pytest.raises(ValueError):
        CommitPolicy.batch().resolve_batch(batch_size=17, flush_interval=None)


@pytest.mark.asyncio
async def test_custom_capability_must_resolve_batch_policy_before_runtime_open(tmp_path, monkeypatch):
    from .test_capability_atomic_writes import _opened

    with pytest.raises(ValueError, match="resolve batch timing"):
        async with _opened(
            tmp_path, monkeypatch, backend="memory",
            policies=(CommitPolicy.batch(), CommitPolicy.sync()),
        ):
            pytest.fail("unresolved custom policy reached Runtime")


@pytest.mark.asyncio
async def test_custom_owner_defaults_preserve_runtime_batch_publication(tmp_path, monkeypatch):
    from .test_capability_atomic_writes import _Buffer, _opened

    buffer = _Buffer()
    policy = CommitPolicy.batch(buffer=buffer).resolve_batch(batch_size=9, flush_interval=12)
    async with _opened(tmp_path, monkeypatch, backend="memory", policies=(policy, policy)) as opened:
        audit, index = await opened.start()
        assert all(template.commit == policy for template in opened.templates)
        await audit.submit("audit")
        await index.submit("index")
        assert await buffer.pending() == 2
        assert opened.probe.writes == []
        await audit.flush()
        assert await buffer.pending() == 0
        assert len(opened.probe.writes) == 2
        assert opened.probe.writes[0].frame is opened.probe.writes[1].frame
        await opened.finish()
    assert not buffer.closed
