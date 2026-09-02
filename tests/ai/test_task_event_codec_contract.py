#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Persisted Task event codec compatibility contracts."""

import pytest

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime.state._store import StoredFact
from linktools.ai.runtime.state._task_repository import _decode_task_event


def _fact(kind: str, data: dict[str, object]) -> StoredFact:
    return StoredFact(
        b"s" * 32,
        1,
        b"o" * 32,
        kind,
        None,
        None,
        data,
    )


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
