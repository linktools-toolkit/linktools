#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime storage maintenance persistence contracts."""

import hashlib

import pytest

from linktools.ai.core import ExecutionEventType
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime import RuntimeDomain
from linktools.ai.runtime.state._maintenance import RuntimeStorageInspection
from linktools.ai.runtime.state._store import StoredFact, StoredRecord
from linktools.ai.task import TaskEventType


class _NoObjects:
    def object_store(self, domain: RuntimeDomain) -> object:
        raise AssertionError(f"unexpected object reference in {domain.value}")


def _inspection() -> RuntimeStorageInspection:
    return RuntimeStorageInspection(
        {},
        _NoObjects(),
        durable_domains=frozenset(),
    )


def _record(kind: str, data: dict[str, object]) -> StoredRecord:
    return StoredRecord(
        hashlib.sha256(f"key:{kind}".encode()).digest(),
        hashlib.sha256(f"partition:{kind}".encode()).digest(),
        None,
        None,
        kind,
        kind,
        None,
        0,
        None,
        0,
        None,
        data,
    )


def _fact(kind: str, data: dict[str, object]) -> StoredFact:
    return StoredFact(
        hashlib.sha256(f"stream:{kind}".encode()).digest(),
        1,
        hashlib.sha256(f"owner:{kind}".encode()).digest(),
        kind,
        None,
        None,
        data,
    )


@pytest.mark.parametrize(
    ("kind", "data"),
    (
        ("agent_plan", {"version": 1, "items": []}),
        (
            "session_turn_commit",
            {
                "version": 1,
                "session_id": "session",
                "sequence": 1,
                "execution_id": "execution",
                "start_message_index": 0,
                "end_message_index": 2,
            },
        ),
    ),
)
def test_maintenance_accepts_reference_free_raw_records(
    kind: str,
    data: dict[str, object],
) -> None:
    references: dict[int, set[str]] = {}

    _inspection()._collect_references(
        RuntimeDomain.CONVERSATION,
        (_record(kind, data),),
        (),
        (),
        references,
    )

    assert references == {}


@pytest.mark.parametrize(
    ("kind", "data"),
    (
        ("session_turn", {"version": 1, "execution_id": "execution"}),
        (TaskEventType.GRAPH_ADMITTED.value, {"version": 1}),
        (ExecutionEventType.EXECUTION_CREATED.value, {"kind": "business"}),
    ),
)
def test_maintenance_accepts_reference_free_raw_facts(
    kind: str,
    data: dict[str, object],
) -> None:
    references: dict[int, set[str]] = {}

    _inspection()._collect_references(
        RuntimeDomain.EXECUTION,
        (),
        (_fact(kind, data),),
        (),
        references,
    )

    assert references == {}


@pytest.mark.parametrize(
    ("kind", "record"),
    (
        ("agent_plan", True),
        ("session_turn", False),
        (TaskEventType.GRAPH_ADMITTED.value, False),
    ),
)
def test_maintenance_rejects_future_reference_free_raw_versions(
    kind: str,
    record: bool,
) -> None:
    inspection = _inspection()

    with pytest.raises(AIError) as raised:
        inspection._collect_references(
            RuntimeDomain.EXECUTION,
            (_record(kind, {"version": 2}),) if record else (),
            () if record else (_fact(kind, {"version": 2}),),
            (),
            {},
        )

    assert raised.value.code is ErrorCode.STORAGE_VERSION_UNSUPPORTED


def test_maintenance_rejects_unknown_raw_persistence_kind() -> None:
    with pytest.raises(AIError) as raised:
        _inspection()._collect_references(
            RuntimeDomain.EXECUTION,
            (),
            (_fact("unknown_raw_fact", {"version": 1}),),
            (),
            {},
        )

    assert raised.value.code is ErrorCode.STORAGE_VERSION_UNSUPPORTED
