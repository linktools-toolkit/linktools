#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests/ai/events/test_envelope.py"""
from datetime import datetime, timezone

from linktools.ai.events.envelope import EventEnvelope
from linktools.ai.events.payloads import RunStarted


def test_event_envelope_construction():
    envelope = EventEnvelope(
        event_id="evt-1", sequence=1, occurred_at=datetime.now(timezone.utc),
        run_id="run-1", root_run_id="run-1", parent_run_id=None, session_id="session-1",
        runnable_id="agent-1", payload=RunStarted(run_id="run-1", runnable_id="agent-1"),
    )
    assert envelope.event_id == "evt-1"
    assert isinstance(envelope.payload, RunStarted)
    assert envelope.payload.run_id == "run-1"


def test_event_envelope_is_frozen():
    import pytest
    envelope = EventEnvelope(
        event_id="evt-1", sequence=1, occurred_at=datetime.now(timezone.utc),
        run_id="run-1", root_run_id="run-1", parent_run_id=None, session_id="session-1",
        runnable_id="agent-1", payload=RunStarted(run_id="run-1", runnable_id="agent-1"),
    )
    with pytest.raises(Exception):
        envelope.sequence = 2
