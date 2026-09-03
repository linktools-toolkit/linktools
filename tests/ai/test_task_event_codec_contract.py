#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Persisted Task event codec compatibility contracts."""

import pytest

from linktools.ai.core import JsonValue
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime.state._store import StoredFact
from linktools.ai.runtime.state._task_repository import _decode_task_event


def _fact(kind: str, data: dict[str, JsonValue]) -> StoredFact:
    return StoredFact(
        b"s" * 32,
        1,
        b"o" * 32,
        kind,
        None,
        None,
        data,
    )


def _node_event(**overrides: JsonValue) -> dict[str, JsonValue]:
    data: dict[str, JsonValue] = {
        "version": 1,
        "occurred_at": "2026-09-02T00:00:00+00:00",
        "status": "RUNNING",
        "previous_status": "READY",
        "node_id": "node",
        "owner": "worker",
        "fence": 1,
        "execution_id": None,
        "result_digest": None,
        "error_code": None,
        "error_digest": None,
    }
    data.update(overrides)
    return data


def test_task_event_future_version_precedes_unknown_kind_classification() -> None:
    fact = _fact("FUTURE_EVENT", {"version": 2})

    with pytest.raises(AIError) as raised:
        _decode_task_event("graph", fact)

    assert raised.value.code is ErrorCode.STORAGE_VERSION_UNSUPPORTED


def test_task_event_corrupt_v1_payload_fails_closed() -> None:
    fact = _fact(
        "GRAPH_ADMITTED",
        {
            "version": 1,
            "occurred_at": "not-a-datetime",
            "status": "PENDING",
        },
    )

    with pytest.raises(AIError) as raised:
        _decode_task_event("graph", fact)

    assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


def test_task_event_graph_changed_rejects_node_only_ready_status() -> None:
    fact = _fact(
        "GRAPH_CHANGED",
        {
            "version": 1,
            "occurred_at": "2026-09-02T00:00:00+00:00",
            "status": "READY",
            "previous_status": "PENDING",
        },
    )

    with pytest.raises(AIError) as raised:
        _decode_task_event("graph", fact)

    assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


@pytest.mark.parametrize(
    "payload",
    (
        _node_event(
            status="SUCCEEDED",
            previous_status="RUNNING",
            owner=None,
        ),
        _node_event(
            status="SUCCEEDED",
            previous_status="RUNNING",
            result_digest="a" * 64,
        ),
        _node_event(
            status="FAILED",
            previous_status="RUNNING",
            owner=None,
        ),
        _node_event(owner=None),
        _node_event(result_digest="a" * 64),
        _node_event(status="READY", previous_status="PENDING"),
        _node_event(status="PENDING", result_digest="a" * 64),
        _node_event(
            status="CANCELLED",
            previous_status="RUNNING",
            owner=None,
            result_digest="a" * 64,
        ),
    ),
)
def test_task_node_event_semantically_invalid_v1_state_fails_closed(
    payload: dict[str, JsonValue],
) -> None:
    with pytest.raises(AIError) as raised:
        _decode_task_event("graph", _fact("NODE_CHANGED", payload))

    assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


def test_task_node_event_allows_running_reclaim_with_existing_execution() -> None:
    event = _decode_task_event(
        "graph",
        _fact(
            "NODE_CHANGED",
            _node_event(
                previous_status="RUNNING",
                owner="replacement-worker",
                fence=2,
                execution_id="execution",
            ),
        ),
    )

    assert event.status.value == "RUNNING"
    assert event.previous_status is event.status
    assert event.owner == "replacement-worker"
    assert event.fence == 2
    assert event.execution_id == "execution"
