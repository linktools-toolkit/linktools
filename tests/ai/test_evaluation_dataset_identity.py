#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Dataset capture addressing preserves structured case identities."""

from pathlib import Path

import pytest

from linktools.ai.core import service_principal
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.evaluation import CaseRef, CaseSpec, DatasetRef, DatasetSpec
from linktools.ai.model import ModelRegistry
from linktools.ai.runtime import Runtime, RuntimeContext, RuntimeStorage


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
    async with Runtime.open(
        "dataset-identity", models=ModelRegistry(), storage=RuntimeStorage.filesystem(tmp_path),
        context=RuntimeContext(None, tenant_id=principal.tenant_id),
    ) as runtime:
        captures = []
        for index, spec in enumerate(specs):
            ref = await runtime.evaluations.publish_dataset(
                spec, principal=principal, idempotency_key=f"dataset-{index}",
            )
            assert ref == spec.ref
            assert await runtime.evaluations.publish_dataset(
                spec, principal=principal, idempotency_key=f"dataset-{index}",
            ) == ref
            cases = await runtime.evaluations.list_cases(ref, principal=principal)
            assert tuple(case.ref for case in cases.items) == (spec.cases[0].ref,)
            captures.append(cases.items[0].input)
        assert captures[0] != captures[1]
        changed = DatasetSpec(specs[0].ref, (
            CaseSpec.agent(specs[0].cases[0].ref, prompt="changed prompt"),
        ))
        with pytest.raises(AIError) as caught:
            await runtime.evaluations.publish_dataset(
                changed, principal=principal, idempotency_key="changed-dataset",
            )
        assert caught.value.code is ErrorCode.IDEMPOTENCY_CONFLICT
