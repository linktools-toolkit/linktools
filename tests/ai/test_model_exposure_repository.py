#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import asyncio
from datetime import datetime, timezone

import pytest

from linktools.ai.agent import AgentBindingSnapshot
from linktools.ai.agent._output import bind_output
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
from linktools.ai.runtime.state._contracts import (
    RecoveryCheckpoint,
    RecoveryCheckpointState,
    RecoveryExecutionInput,
    RecoveryHandoffPhase,
    RecoveryIdempotencyInput,
)
from linktools.ai.runtime.state._exposure_repository import ModelExposureRepository
from linktools.ai.spec import AgentSpec
from linktools.ai.storage import ObjectRef


def _binding() -> AgentBindingSnapshot:
    output = bind_output()
    return AgentBindingSnapshot(
        version=1,
        agent_spec=AgentSpec("default"),
        model={"route_id": "default", "model_identity": "test:model"},
        selected=(),
        subagents=(),
        output_mode=output.mode,
        output_schema=output.schema_definition,
        binding_digest="a" * 64,
    )


def _checkpoint(execution_id: str) -> RecoveryCheckpoint:
    now = datetime.now(timezone.utc)
    return RecoveryCheckpoint(
        execution_id,
        "default",
        RecoveryExecutionInput(
            user_prompt="prompt",
            user_prompt_codec="text",
            principal_id="owner",
            principal_kind="user",
            session_id=None,
            memory_scope=None,
            binding_digest="a" * 64,
            lineage_kind="run",
            parent_execution_id=None,
            root_execution_id=execution_id,
            source_execution_id=None,
            base_execution_id=None,
            conversation_step_run_id=None,
            idempotency=RecoveryIdempotencyInput("scope", "key", "digest"),
            mode="run",
            planning=False,
            thinking=False,
            binding=_binding(),
        ),
        None,
        0,
        RecoveryCheckpointState.ADMITTED,
        RecoveryHandoffPhase.NONE,
        None,
        None,
        None,
        0,
        now,
        now,
    )


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
        await state.recovery.checkpoints.create(_checkpoint("execution"))
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
        await state.recovery.checkpoints.create(_checkpoint("execution"))
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
