#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Retention discovery batches case reads without changing expiry decisions."""

import asyncio
import json
from collections import Counter
from contextlib import asynccontextmanager
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

import pytest
from sqlalchemy import event

from linktools.ai.core import canonical_json_bytes
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.evaluation import CaseContract, CaseRef, DatasetContract, DatasetRef, TaskCaseInput, TrialPlan
from linktools.ai.persistence import Persistence
from linktools.ai.persistence import _sql
from linktools.ai.persistence._evaluation_records import (
    EvaluationCaseRecord, EvaluationContentTombstone, EvaluationDatasetRecord,
)

from .test_evaluation_persistence import NOW, experiment, storage_for


@asynccontextmanager
async def _retention_repository(tmp_path, backend, states, *, dataset_state="live"):
    async with Persistence.open("retention-reads", storage=storage_for(backend, tmp_path), tenant_id="tenant") as persistence:
        repository = persistence.evaluation.records
        cases = tuple(CaseContract(CaseRef("dataset", str(index), 1), TaskCaseInput(input={"index": index}))
                      for index in range(len(states)))
        dataset = DatasetContract(DatasetRef("dataset", 1), tuple(case.ref for case in cases), "task_input")
        record = experiment()
        manifest = replace(record.manifest, trials=tuple(
            TrialPlan(str(index), case.ref, "target", 1) for index, case in enumerate(cases)
        ))
        record = replace(record, manifest=manifest, manifest_digest=manifest.digest)
        await repository.reserve_experiment(record)
        dataset_record = EvaluationDatasetRecord(dataset, "owner", NOW,
            NOW if dataset_state == "expired" else None)
        rows = []
        if dataset_state not in {"missing", "tombstone"}:
            rows.append(repository._stored("evaluation_dataset", [dataset.ref.id, dataset.ref.revision], dataset_record))
        if dataset_state == "tombstone":
            identity = [dataset.ref.id, dataset.ref.revision]
            rows.append(repository._stored("evaluation_content_tombstone", ["evaluation_dataset", identity],
                EvaluationContentTombstone("evaluation_dataset", identity, "owner", dataset.digest, NOW)))
        for case, state in zip(cases, states):
            identity = [case.ref.dataset_id, case.ref.case_id, case.ref.revision]
            if state not in {"missing", "tombstone"}:
                value = (dataset_record if state == "malformed" else
                         EvaluationCaseRecord(case, "owner", NOW, NOW if state == "expired" else None))
                rows.append(repository._stored("evaluation_case", identity, value))
            if state in {"tombstone", "live_with_tombstone"}:
                rows.append(repository._stored("evaluation_content_tombstone", ["evaluation_case", identity],
                    EvaluationContentTombstone("evaluation_case", identity, "owner", case.digest, NOW)))
        await repository.state_store.mutate(lambda tx: tx.insert_records(tuple(rows)))
        yield repository, record, cases


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "local_sqlite", "sqlite"))
@pytest.mark.parametrize("states,expired", (
    (("live", "live"), False),
    (("missing", "live"), False),
    (("missing", "expired"), True),
    (("tombstone", "malformed"), True),
    (("expired", "malformed"), True),
    (("live_with_tombstone", "live"), False),
    (("malformed", "expired"), None),
    (("missing", "malformed"), None),
))
async def test_retention_case_batches_preserve_order_and_missing_semantics(
    tmp_path: Path, backend: str, states: tuple[str, ...], expired: bool | None,
) -> None:
    async with _retention_repository(tmp_path, backend, states) as (repository, record, _cases):
        if expired is None:
            with pytest.raises(AIError) as raised:
                await repository.list_expired(now=NOW, limit=10, owner_principal_id="owner")
            assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
        else:
            assert await repository.list_expired(now=NOW, limit=10, owner_principal_id="owner") == (
                (record,) if expired else ()
            )
        # An unrelated owner never inspects this dataset, including malformed content.
        assert await repository.list_expired(now=NOW, limit=10, owner_principal_id="other") == ()


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "sqlite"))
@pytest.mark.parametrize("dataset_state,expired", (("missing", False), ("tombstone", True), ("expired", True)))
async def test_retention_dataset_decision_precedes_case_decoding(
    tmp_path: Path, backend: str, dataset_state: str, expired: bool,
) -> None:
    async with _retention_repository(tmp_path, backend, ("malformed",), dataset_state=dataset_state) as (repository, record, _cases):
        assert await repository.list_expired(now=NOW, limit=10) == ((record,) if expired else ())


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "sqlite"))
async def test_retention_reads_fresh_case_expiry_on_each_discovery(tmp_path: Path, backend: str) -> None:
    async with _retention_repository(tmp_path, backend, ("live",) * 65) as (repository, record, cases):
        assert await repository.list_expired(now=NOW, limit=10) == ()
        case = cases[-1]
        identity = [case.ref.dataset_id, case.ref.case_id, case.ref.revision]
        row = repository._stored("evaluation_case", identity, EvaluationCaseRecord(case, "owner", NOW, NOW))
        assert await repository.state_store.mutate(lambda tx: tx.replace_record(
            replace(row, storage_revision=1), expected_storage_revision=0,
        ))
        assert await repository.list_expired(now=NOW, limit=10) == (record,)
        assert await repository.list_expired(now=NOW - timedelta(seconds=1), limit=10) == ()


@pytest.mark.asyncio
@pytest.mark.parametrize("case_count", (1, 64, 65))
async def test_retention_case_selects_are_bounded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case_count: int) -> None:
    async with _retention_repository(tmp_path, "sqlite", ("live",) * case_count) as (repository, _record, _cases):
        counts = Counter()
        engine = repository.state_store.context.engine.sync_engine
        original = _sql._record_from_row

        def materialize(row):
            result = original(row)
            counts["records"] += 1
            counts["logical_json_bytes"] += len(canonical_json_bytes(result.data))
            return result

        def statement(_connection, _cursor, sql, _parameters, _context, _many):
            counts[sql.lstrip().split()[0].upper()] += 1

        def begin(_connection):
            counts["transactions"] += 1

        def commit(_connection):
            counts["commits"] += 1

        monkeypatch.setattr(_sql, "_record_from_row", materialize)
        event.listen(engine, "before_cursor_execute", statement)
        event.listen(engine, "begin", begin)
        event.listen(engine, "commit", commit)
        try:
            assert await repository.list_expired(now=NOW, limit=10, owner_principal_id="owner") == ()
        finally:
            event.remove(engine, "before_cursor_execute", statement)
            event.remove(engine, "begin", begin)
            event.remove(engine, "commit", commit)
        print(json.dumps({"case_count": case_count, **counts}, sort_keys=True))
        assert counts["SELECT"] <= 2 + (case_count + 63) // 64
        assert counts["INSERT"] == counts["UPDATE"] == counts["DELETE"] == 0
        assert counts["transactions"] == counts["commits"] == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ("cancel", "read_error"))
async def test_retention_case_batch_failure_propagates_without_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str,
) -> None:
    async with _retention_repository(tmp_path, "sqlite", ("live",) * 65) as (repository, _record, cases):
        original = _sql._record_from_row
        keys = {repository._key("evaluation_case", [case.ref.dataset_id, case.ref.case_id, case.ref.revision]) for case in cases}
        calls = 0

        def read(row):
            nonlocal calls
            value = original(row)
            if value.key_digest in keys:
                calls += 1
                if failure == "cancel":
                    raise asyncio.CancelledError
                raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)
            return value

        with monkeypatch.context() as patch:
            patch.setattr(_sql, "_record_from_row", read)
            with pytest.raises(asyncio.CancelledError if failure == "cancel" else AIError) as raised:
                await repository.list_expired(now=NOW, limit=10)
            if failure == "read_error":
                assert raised.value.code is ErrorCode.STORAGE_RECOVERY_REQUIRED
        assert calls == 1
        assert await repository.list_expired(now=NOW, limit=10) == ()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ("evaluation_case", "evaluation_content_tombstone"))
async def test_retention_fails_closed_on_corrupt_row_in_selected_case_batch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str,
) -> None:
    async with _retention_repository(tmp_path, "sqlite", ("expired", "live_with_tombstone")) as (repository, record, cases):
        case = cases[-1]
        identity = [case.ref.dataset_id, case.ref.case_id, case.ref.revision]
        key = repository._key(kind, identity if kind == "evaluation_case" else ["evaluation_case", identity])
        original = _sql._record_from_row

        def read(row):
            value = original(row)
            if value.key_digest == key:
                return original(dict(row, payload_json=[]))
            return value

        with monkeypatch.context() as patch:
            patch.setattr(_sql, "_record_from_row", read)
            with pytest.raises(AIError) as raised:
                await repository.list_expired(now=NOW, limit=10)
            assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
        assert await repository.list_expired(now=NOW, limit=10) == (record,)
