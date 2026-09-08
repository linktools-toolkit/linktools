#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import asyncio

import pytest

from linktools.ai.core import canonical_sha256
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime import RuntimeState
from linktools.ai.runtime.state import (
    AttachmentEntry,
    AttachmentPresentation,
    ContentRef,
    Locator,
    ModelExposureEntry,
    PathOrigin,
    managed_attachment_path,
)
from linktools.ai.runtime.state._exposure_repository import ModelExposureRepository
from linktools.ai.storage import ObjectRef


def _entry(owner: str, *, digest: str = "a" * 64) -> AttachmentEntry:
    return AttachmentEntry(
        managed_attachment_path("p", owner, 0),
        "evidence.txt",
        "text/plain",
        AttachmentPresentation(None, None),
        ContentRef(
            "execution",
            f"input-prepare:{owner}",
            ObjectRef("memory", "content", digest, 8),
        ),
    )


def _exposure_entry(owner: str, *, digest: str = "a" * 64) -> ModelExposureEntry:
    return ModelExposureEntry(
        canonical_sha256(["activation", owner, digest]),
        Locator("state:execution", "records", "b" * 64),
        0,
        _entry(owner, digest=digest),
    )


@pytest.mark.asyncio
async def test_model_exposure_fact_is_semantically_idempotent() -> None:
    state = RuntimeState.in_memory()
    await state.initialize(namespace="exposure-test", tenant_id="default")
    try:
        repository = ModelExposureRepository(
            state.recovery.checkpoints.state_store,
            namespace="exposure-test",
            tenant_id="default",
        )
        origin = PathOrigin(1, "exposure-test", "posix", "/workspace")
        owner = "c" * 64
        entry = _exposure_entry(owner)

        first, replay = await asyncio.gather(
            repository.put(
                execution_id="execution",
                step_run_id="run",
                run_step=0,
                path_origin=origin,
                entries=(entry,),
            ),
            repository.put(
                execution_id="execution",
                step_run_id="run",
                run_step=0,
                path_origin=origin,
                entries=(entry,),
            ),
        )

        assert replay == first
        assert first.exposure_id == canonical_sha256(
            ["exposure-test", "default", "execution", "run", 0]
        )
        assert await repository.get(
            execution_id="execution",
            step_run_id="run",
            run_step=0,
        ) == first
    finally:
        await state.close()


@pytest.mark.asyncio
async def test_model_exposure_same_subject_rejects_semantic_drift() -> None:
    state = RuntimeState.in_memory()
    await state.initialize(namespace="exposure-test", tenant_id="default")
    try:
        repository = ModelExposureRepository(
            state.recovery.checkpoints.state_store,
            namespace="exposure-test",
            tenant_id="default",
        )
        origin = PathOrigin(1, "exposure-test", "posix", "/workspace")
        owner = "c" * 64
        await repository.put(
            execution_id="execution",
            step_run_id="run",
            run_step=1,
            path_origin=origin,
            entries=(_exposure_entry(owner),),
        )

        with pytest.raises(AIError) as raised:
            await repository.put(
                execution_id="execution",
                step_run_id="run",
                run_step=1,
                path_origin=origin,
                entries=(_exposure_entry(owner, digest="d" * 64),),
            )
        assert raised.value.code is ErrorCode.STORAGE_CONFLICT
    finally:
        await state.close()
