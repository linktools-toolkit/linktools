#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Closing storage persists queued projections that reference external transcripts."""

from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pytest
from pydantic_ai.messages import ModelRequest, UserPromptPart
from pydantic_ai.models import ModelRequestParameters

from linktools.ai.runtime import RuntimeDomain, RuntimeStorage
from linktools.ai.runtime._model_interaction import StagedContextProjection, StagedModelInteraction, request_envelope
from linktools.ai.runtime._transcript_staging import StagedTranscript
from linktools.ai.runtime.state._contracts import TranscriptSpanRef
from linktools.ai.runtime.state._step_contracts import AgentRunRecord
from linktools.ai.storage import InMemoryCacheBackend, ObjectStoreInspection

from .test_step_projection import _execution


@pytest.mark.asyncio
async def test_close_persists_queued_projection_referencing_external_transcript(
    tmp_path: Path,
) -> None:
    class Cache(InMemoryCacheBackend):
        reads = 0

        async def get(self, key: str) -> bytes | None:
            self.reads += 1
            return await super().get(key)

    cache = Cache(max_bytes=1024 * 1024)
    root = tmp_path / "state"
    storage = RuntimeStorage.filesystem(root, cache=cache)
    await storage.initialize(
        namespace="close-projection", tenant_id="tenant", module_caches={RuntimeDomain.TRACE: cache},
    )
    message = ModelRequest(parts=[UserPromptPart("external transcript " * 4096)])
    try:
        execution = replace(_execution(), revision=1)
        repository = storage.execution.executions
        await repository.create_with_history_head(execution)
        await repository.state_store.mutate(
            lambda transaction: repository.admit_history_producer_in_transaction(transaction, execution)
        )
        store = storage.run_store
        store.trace_persistence.configure("run", cache=cache, flush_interval=60.0)
        store.bind_execution_producer("run", execution_id="execution", producer_generation=1)
        await store.register_agent_run(AgentRunRecord("run"), execution_id="execution")
        store.stage_transcript("run", StagedTranscript((message,)))
        await store.flush_execution_projection("run", execution_id="execution", producer_generation=1)
        archive = store.read_store(RuntimeDomain.TRACE)
        objects = storage.object_store(RuntimeDomain.TRACE)
        assert isinstance(objects, ObjectStoreInspection)
        assert [item async for item in objects.list_objects()]

        _envelope, raw = request_envelope(
            model_settings=None, parameters=ModelRequestParameters(), streaming=False,
        )
        envelope_digest, _size = store.intern_payload("run", raw)
        now = datetime.now(timezone.utc)
        store.stage_model_interaction(StagedModelInteraction(
            agent_run_id="run", step_index=1, model_request_seq=1,
            purpose="agent", output_retry_index=None, model={},
            request_context=StagedContextProjection((
                TranscriptSpanRef(RuntimeDomain.TRACE, "run", 0, 1),
            )),
            request_envelope_digest=envelope_digest, response_context=None,
            status="CANCELLED", error_code=None, duration_ns=0, usage=None,
            started_at=now, finished_at=now,
        ))
        await store.flush_execution_projection(
            "run", execution_id="execution", producer_generation=1, deferred=True,
        )
        assert await archive.list_model_interactions(agent_run_id="run") == []
        reads_before_close = cache.reads
        await storage.close()
        assert cache.reads > reads_before_close
    finally:
        await storage.close()

    reopened = RuntimeStorage.filesystem(root, cache=cache)
    await reopened.initialize(
        namespace="close-projection", tenant_id="tenant", read_only=True,
        module_caches={RuntimeDomain.TRACE: cache},
    )
    try:
        archive = reopened.run_store.read_store(RuntimeDomain.TRACE)
        interactions = await archive.list_model_interactions(agent_run_id="run")
        assert len(interactions) == 1
        assert interactions[0].request_context is not None
        context = await archive.transcript_repository.load_projected_context(
            "run", interactions[0].request_context,
        )
        assert context.model_messages() == (message,)
    finally:
        await reopened.close()
