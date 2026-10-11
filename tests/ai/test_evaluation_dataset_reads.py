#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Dataset retention reads are bounded without dropping case validation."""

from dataclasses import replace
from datetime import timedelta
from pathlib import Path

import pytest
from sqlalchemy import event

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.evaluation import CaseContract, CaseRef, DatasetContract, DatasetRef, TaskCaseInput
from linktools.ai.runtime.state._evaluation_records import EvaluationCaseRecord, EvaluationDatasetRecord
from linktools.ai.runtime.state._sql import SqlStateStore

from .test_evaluation_persistence import NOW, storage_for


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "filesystem", "sqlite"))
@pytest.mark.parametrize("fault", (None, "missing", "wrong_type"))
async def test_dataset_expiry_preserves_all_case_deadlines_and_integrity(
    tmp_path: Path, backend: str, fault: str | None,
) -> None:
    storage = storage_for(backend, tmp_path)
    await storage.initialize(namespace="dataset-reads", tenant_id="tenant")
    try:
        repository = storage.evaluation.records
        cases = tuple(CaseContract(CaseRef("dataset", str(index), 1), TaskCaseInput(input={}))
                      for index in range(65 if fault is None else 3))
        dataset = DatasetContract(DatasetRef("dataset", 1), tuple(case.ref for case in cases), "task_input")
        earliest = NOW + timedelta(hours=1)
        dataset_record = EvaluationDatasetRecord(dataset, "owner", NOW, NOW + timedelta(hours=3))
        records = [repository._stored("evaluation_dataset", [dataset.ref.id, dataset.ref.revision], dataset_record)]
        for index, case in enumerate(cases):
            if index == len(cases) - 1 and fault == "missing":
                continue
            value = (dataset_record if index == len(cases) - 1 and fault == "wrong_type" else
                     EvaluationCaseRecord(case, "owner", NOW,
                        earliest if index == len(cases) - 1 else None))
            records.append(repository._stored("evaluation_case", [case.ref.dataset_id, case.ref.case_id, case.ref.revision], value))
        await repository.state_store.mutate(lambda tx: tx.insert_records(tuple(records)))
        statements = []
        def capture(_connection, _cursor, sql, _parameters, _context, _many):
            if sql.startswith("SELECT"):
                statements.append(sql)
        sql_store = repository.state_store if isinstance(repository.state_store, SqlStateStore) else None
        if sql_store is not None:
            event.listen(sql_store.context.engine.sync_engine, "before_cursor_execute", capture)
        try:
            if fault is None:
                assert await repository.dataset_expiry(dataset.ref) == earliest
                if sql_store is not None:
                    assert len(statements) <= 3, "expiry reads must batch case identities"
            else:
                with pytest.raises(AIError) as raised:
                    await repository.dataset_expiry(dataset.ref)
                assert raised.value.code is (ErrorCode.EVALUATION_EVIDENCE_UNAVAILABLE
                                             if fault == "missing" else ErrorCode.STORAGE_INTEGRITY_ERROR)
        finally:
            if sql_store is not None:
                event.remove(sql_store.context.engine.sync_engine, "before_cursor_execute", capture)
        if fault is None:
            # A later operation reads a changed retained deadline rather than reusing a cache.
            row = records[-1]
            case = EvaluationCaseRecord(cases[-1], "owner", NOW, None)
            changed = replace(repository._stored("evaluation_case", [cases[-1].ref.dataset_id,
                              cases[-1].ref.case_id, cases[-1].ref.revision], case), storage_version=1)
            assert await repository.state_store.mutate(lambda tx: tx.replace_record(
                changed, expected_storage_version=row.storage_version))
            assert await repository.dataset_expiry(dataset.ref) == dataset_record.content_expires_at
    finally:
        await storage.close()
