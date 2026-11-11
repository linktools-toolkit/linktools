#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Malformed stored record owners report storage errors, not request errors."""

from datetime import datetime, timezone

import pytest

from linktools.ai.core import BudgetUsage, RunBudget
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.evaluation import CaseRef, DatasetRef
from linktools.ai.model import StoredUserInput
from linktools.ai.persistence._budget_records import BudgetModelReservation, BudgetToolReservation
from linktools.ai.persistence._evaluation_records import EvaluationContentTombstone
from linktools.ai.persistence.codec import _CURRENT_CODEC, _decode_domain, _encode_persisted_domain
from linktools.ai.persistence.contracts import ModelInteractionRecord, TaskPreparedInputRecord
from linktools.ai.storage import StoredPayload
from linktools.ai.task import TaskRef

_NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)
_PREPARED = TaskPreparedInputRecord(
    "graph", "node", "tenant", "a" * 64, TaskRef("task", 1), "b" * 64, (),
    StoredUserInput("text", StoredPayload.inline_text("input")), "c" * 64, "d" * 64, 1,
)


@pytest.mark.parametrize("value,field,invalid", (
    (DatasetRef("dataset", 1), "id", "x" * 129),
    (CaseRef("dataset", "case", 1), "dataset_id", "x" * 129),
    (CaseRef("dataset", "case", 1), "case_id", "x" * 129),
    (ModelInteractionRecord("run", 0, 1, "agent", None, {}, None, None, None,
                            "RUNNING", None, None, None, _NOW, None), "agent_run_id", "x" * 129),
    (_PREPARED, "graph_id", "x" * 129),
    (_PREPARED, "node_id", "x" * 129),
    (BudgetUsage("execution:run", RunBudget()), "scope_id", "execution:" + "x" * 129),
    (BudgetModelReservation("scope", "request", "owner", "admission"), "request_id", "x" * 129),
    (BudgetModelReservation("graph:graph", "request", "owner", "admission"), "scope_id", "graph:" + "x" * 129),
    (BudgetToolReservation("graph:graph", "call", "admission"), "scope_id", "graph:" + "x" * 129),
    (EvaluationContentTombstone("evaluation_report", "report", "owner", "a" * 64, _NOW), "identity", "x" * 129),
    (EvaluationContentTombstone("evaluation_dataset", ["dataset", 1], "owner", "a" * 64, _NOW), "identity", ["x" * 129, 1]),
    (EvaluationContentTombstone("evaluation_case", ["dataset", "case", 1], "owner", "a" * 64, _NOW), "identity", ["dataset", "x" * 129, 1]),
))
def test_persisted_owner_validation_maps_to_storage_integrity(value: object, field: str, invalid: object) -> None:
    payload = _encode_persisted_domain(value)
    assert _decode_domain(payload, type(value), _CURRENT_CODEC, persisted=True) == value
    payload["fields"][field] = invalid
    with pytest.raises(AIError) as raised:
        _decode_domain(payload, type(value), _CURRENT_CODEC, persisted=True)
    assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
