#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Dataset capture addressing preserves structured case identities."""

from linktools.ai.persistence import Persistence
from linktools.ai.core import ExecutionContext

from pathlib import Path

import pytest

from linktools.ai.core import service_principal
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.evaluation import CaseRef, CaseSpec, DatasetRef, DatasetSpec

from linktools.ai.persistence import PersistenceStorage
from linktools.ai.task.engine import TaskEngine
from linktools.ai.evaluation.engine import Evaluations


@pytest.mark.asyncio
async def test_dataset_capture_keys_preserve_structured_case_identity(tmp_path: Path) -> None:
    principal = service_principal("dataset-identity", "owner")
    specs = tuple(
        DatasetSpec(DatasetRef(dataset, 1), (
            CaseSpec.agent(CaseRef(dataset, case, 1), prompt=prompt),
        ))
        for dataset, case, prompt in (
            ("group:one", "case", "first prompt"),
            ("group", "one:case", "second prompt"),
        )
    )
    async with (
        Persistence.open('dataset-identity', storage=PersistenceStorage.filesystem(tmp_path), tenant_id=principal.tenant_id) as persistence_30,
        TaskEngine.open(persistence=persistence_30, context=ExecutionContext(None)) as task_engine,
        Evaluations.open(task_engine) as evaluations,
    ):
        captures = []
        for index, spec in enumerate(specs):
            ref = await evaluations.publish_dataset(
                spec, principal=principal, idempotency_key=f"dataset-{index}",
            )
            assert ref == spec.ref
            assert await evaluations.publish_dataset(
                spec, principal=principal, idempotency_key=f"dataset-{index}",
            ) == ref
            cases = await evaluations.list_cases(ref, principal=principal)
            assert tuple(case.ref for case in cases.items) == (spec.cases[0].ref,)
            captures.append(cases.items[0].input)
        assert captures[0] != captures[1]
        changed = DatasetSpec(specs[0].ref, (
            CaseSpec.agent(specs[0].cases[0].ref, prompt="changed prompt"),
        ))
        with pytest.raises(AIError) as caught:
            await evaluations.publish_dataset(
                changed, principal=principal, idempotency_key="changed-dataset",
            )
        assert caught.value.code is ErrorCode.IDEMPOTENCY_CONFLICT
