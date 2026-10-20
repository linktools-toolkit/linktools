#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared response content preserves independent evidence and ordered occurrences."""

import asyncio
import hashlib
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import pytest
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, UserPromptPart
from pydantic_ai.models.function import AgentInfo
from sqlalchemy.ext.asyncio import create_async_engine

from linktools.ai.capability import RuntimeFeatures, TraceCapability
from linktools.ai.core import agent_run_id, canonical_json_bytes
from linktools.ai.migrate import provision_runtime_database
from linktools.ai.runtime import Runtime, RuntimeStorage
from linktools.ai.runtime.state import RuntimeDomain
from linktools.ai.runtime.state._archive import ContentArchive, ContentRecord
from linktools.ai.runtime.state._contracts import ArchivedContextBlock, ContextProjection, ModelInteractionRecord
from linktools.ai.runtime.state._history import encode_model_messages
from linktools.ai.runtime.state._step_archive import StateStepArchive
from linktools.ai.runtime.state._store import FactQuery, RecordQuery, StateTransaction
from linktools.ai.storage import ObjectStat, SqlObjectStore

from .test_explicit_runtime_features import _FeatureFunctionModel, _FeatureModels


@asynccontextmanager
async def _storage(backend: str, root: Path) -> AsyncIterator[RuntimeStorage]:
    engine = None
    if backend == "memory":
        storage = RuntimeStorage.in_memory()
    elif backend == "filesystem":
        storage = RuntimeStorage.filesystem(root / "state")
    elif backend == "sqlite":
        storage = RuntimeStorage.sqlite(root / "state.db")
    else:
        engine = create_async_engine(f"sqlite+aiosqlite:///{root / 'state.db'}")
        await provision_runtime_database(engine)
        storage = RuntimeStorage.sql(engine)
    try:
        yield storage
    finally:
        if engine is not None:
            await engine.dispose()


def _features() -> RuntimeFeatures:
    features = RuntimeFeatures("response-dedup")
    features.agent("default", model="default", allow_tools=())
    features.capability(TraceCapability(), scope="runtime")
    return features


def _archived_block(projection: ContextProjection | None) -> ArchivedContextBlock:
    assert projection is not None and len(projection.items) == 1
    block, = projection.items
    assert isinstance(block, ArchivedContextBlock)
    return block


async def _transcript_content_ids(archive: StateStepArchive, run_id: str) -> tuple[str, ...]:
    transcript = archive.transcript_repository
    facts = await archive.state_store.read(
        lambda transaction: transaction.list_facts(FactQuery(transcript.transcript_stream(run_id))),
    )
    chunks = tuple(transcript.decode_chunk(fact) for fact in facts)
    content = await transcript.content_archive()
    groups = await content.content_ids_many(tuple(chunk.content for chunk in chunks))
    ids = tuple(identity for group in groups for identity in group)
    assert len(ids) == await transcript.transcript_message_count(run_id)
    return ids


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "filesystem", "sqlite", "sql"))
@pytest.mark.parametrize("size", (32, 65536), ids=("short", "long"))
async def test_trace_response_payload_storage_and_metadata_reads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend: str, size: int,
) -> None:
    content = ("response evidence 中文 " * (size // 21 + 1))[:size]

    async def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart(content)], timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc))

    async with _storage(backend, tmp_path) as storage:
        async with Runtime.open(
            "response-dedup", storage=storage, features=_features(),
            models=_FeatureModels(_FeatureFunctionModel(model)),
        ) as runtime:
            objects = runtime._state.archive_object_store(RuntimeDomain.TRACE)
            domain_objects = runtime._state.object_store(RuntimeDomain.TRACE)
            sources = {"archive": objects}
            if domain_objects is not objects:
                sources["domain"] = domain_objects
            # SQL routes can wrap one physical object table in distinct Python instances.
            physical_sources = {}
            for label, source in sources.items():
                identity = ((id(source._context.engine), source.store_id) if isinstance(source, SqlObjectStore)
                            else (id(source),))
                physical_sources.setdefault(identity, (label, source))
            puts: list[tuple[str, str, int]] = []

            for label, source in sources.items():
                original_put = source.put

                async def put(key: str, chunks: AsyncIterator[bytes], *, expected_size: int, expected_digest: str,
                              label: str = label, original_put=original_put) -> ObjectStat:
                    result = await original_put(key, chunks, expected_size=expected_size, expected_digest=expected_digest)
                    puts.append((label, key, expected_size))
                    return result

                monkeypatch.setattr(source, "put", put)
            execution = await runtime.agents.get().start("question")
            assert (await execution.wait(timeout_seconds=10)).result.status == "SUCCEEDED"
            run_id = agent_run_id(
                namespace=runtime.namespace, tenant_id=runtime.tenant_id,
                execution_id=execution.execution_id, agent_run_seq=1,
            )
            archive = runtime._state.run_store.read_store(RuntimeDomain.TRACE)
            records = await archive.list_model_interactions(agent_run_id=run_id)
            assert len(records) == 1 and isinstance(records[0], ModelInteractionRecord)
            block = _archived_block(records[0].response_context)
            assert block.root.count == 1
            content_archive = await archive.transcript_repository.content_archive()
            response_ids = await content_archive.content_ids(block.root)
            transcript_ids = await _transcript_content_ids(archive, run_id)
            assert transcript_ids[1:2] == response_ids
            response_content, = await content_archive.content_records(response_ids)
            response_records = await content_archive.store.read(lambda transaction: transaction.list_records(
                RecordQuery(kind="archive_content", sort_key_prefix=response_content.identity),
            ))
            assert len(response_records) == 1
            assert ContentRecord.from_json(response_records[0].data) == response_content
            persisted = await archive.state_store.read(
                lambda transaction: transaction.list_records(RecordQuery(kind="model_interaction")),
            )
            archive_records = await content_archive.store.read(lambda transaction: transaction.scan_records())
            stats = [(label, item) for label, source in physical_sources.values()
                     async for item in source.list_objects()]
            actual_bytes = 0
            for label, item in stats:
                raw = b"".join([chunk async for chunk in sources[label].open(item.key)])
                assert len(raw) == item.size and hashlib.sha256(raw).hexdigest() == item.digest
                actual_bytes += len(raw)
            assert actual_bytes == sum(item.size for _, item in stats)
            if size == 32:
                assert response_content.payload.ref is None
            else:
                response_ref = response_content.payload.ref
                assert response_ref is not None
                assert [label for label, item in stats if item.key == response_ref.key] == ["archive"]
                assert [(label, count) for label, key, count in puts if key == response_ref.key] == [
                    ("archive", response_ref.size),
                ]
            report = {
                "backend": backend, "response_chars": size,
                "interaction_metadata_bytes": sum(len(canonical_json_bytes(item.data)) for item in persisted),
                "object_count": len(stats), "object_bytes": actual_bytes,
                "object_puts": len(puts), "object_put_bytes": sum(size for _, _, size in puts),
                "archive_object_count": sum(item.key.startswith("v1/archive/") for _, item in stats),
                "domain_object_count": sum(not item.key.startswith("v1/archive/") for _, item in stats),
                "physical_store_count": len(physical_sources),
                "archive_metadata_bytes": sum(len(canonical_json_bytes(item.data)) for item in archive_records),
                "response_content_records": len(response_records),
                "response_payload_bytes": response_content.payload.size,
            }

            async def reject_content(key: str) -> AsyncIterator[bytes]:
                raise AssertionError("metadata read loaded object content")
                yield b""

            with monkeypatch.context() as patch:
                for source in sources.values():
                    patch.setattr(source, "open", reject_content)

                async def reject_body(*args: object, **kwargs: object) -> bytes:
                    raise AssertionError("metadata read decoded archived message content")

                patch.setattr(ContentArchive, "_read_content", reject_body)

                async def reject_projection(*args: object, **kwargs: object) -> None:
                    raise AssertionError("metadata read resolved full model context")

                patch.setattr(archive.transcript_repository, "load_projected_contexts", reject_projection)
                patch.setattr(archive.transcript_repository, "load_projected_context", reject_projection)
                page = await execution.model_interactions()
                assert len(page.items) == 1
                assert not page.items[0].content_included
                assert page.items[0].request == {} and page.items[0].response is None
                assert await archive.resolve_model_interactions(records, include_content=False) == [(None, None, None, None)]
            opens: list[tuple[str, str]] = []
            with monkeypatch.context() as patch:
                for label, source in sources.items():
                    original_open = source.open

                    async def count_open(key: str, *, label: str = label, original_open=original_open) -> AsyncIterator[bytes]:
                        opens.append((label, key))
                        async for chunk in original_open(key):
                            yield chunk

                    patch.setattr(source, "open", count_open)
                public = await execution.model_interactions(include_content=True)
                assert public.items[0].response is not None
            report["full_content_object_opens"] = len(opens)
            report["full_content_archive_opens"] = sum(label == "archive" for label, _ in opens)
            report["metadata_content_loads"] = 0
            print("TRACE_RESPONSE_STORAGE " + json.dumps(report, sort_keys=True))
            resolved = await archive.resolve_model_interactions(records)
            assert resolved[0][1][0].parts[0].content == content


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "filesystem", "sqlite", "sql"))
@pytest.mark.parametrize("outcome", ("FAILED", "CANCELLED"))
async def test_unsuccessful_requests_without_response_preserve_no_response(
    tmp_path: Path, backend: str, outcome: str,
) -> None:
    entered = asyncio.Event()

    async def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        entered.set()
        if outcome == "FAILED":
            raise RuntimeError("controlled model failure")
        await asyncio.Event().wait()
        raise AssertionError("cancelled request resumed")

    async with _storage(backend, tmp_path) as storage:
        async with Runtime.open(
            "response-dedup", storage=storage, features=_features(),
            models=_FeatureModels(_FeatureFunctionModel(model)),
        ) as runtime:
            execution = await runtime.agents.get().start("question")
            if outcome == "CANCELLED":
                await asyncio.wait_for(entered.wait(), 5)
                await execution.cancel()
            assert (await execution.wait(timeout_seconds=10)).result.status == outcome
            page = await execution.model_interactions(include_content=True)
            assert len(page.items) == 1
            assert page.items[0].status == outcome
            assert page.items[0].response is None
            assert page.items[0].request["messages"]


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "filesystem", "sqlite", "sql"))
async def test_compaction_response_retains_independent_evidence(
    tmp_path: Path, backend: str,
) -> None:
    from linktools.ai.runtime import ExecutionInputContext

    responses: list[ModelResponse] = []

    async def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        response = ModelResponse(parts=[TextPart("retained summary")], timestamp=datetime(
            2026, 1, 1, second=len(responses), tzinfo=timezone.utc,
        ))
        responses.append(response)
        return response

    history: list[ModelMessage] = []
    for index in range(20):
        history.extend((
            ModelRequest(parts=[UserPromptPart(f"user-{index} " + "x" * 240)]),
            ModelResponse(parts=[TextPart(f"answer-{index} " + "y" * 240)]),
        ))
    async with _storage(backend, tmp_path) as storage:
        async with Runtime.open(
            "response-dedup", storage=storage, features=_features(),
            models=_FeatureModels(_FeatureFunctionModel(model, profile={"context_window": 1000})),
        ) as runtime:
            execution = await runtime.agents.get().start(
                "continue", input_context=ExecutionInputContext.from_messages(history),
            )
            assert (await execution.wait(timeout_seconds=10)).result.status == "SUCCEEDED"
            run_id = agent_run_id(
                namespace=runtime.namespace, tenant_id=runtime.tenant_id,
                execution_id=execution.execution_id, agent_run_seq=1,
            )
            archive = runtime._state.run_store.read_store(RuntimeDomain.TRACE)
            records = await archive.list_model_interactions(agent_run_id=run_id)
            assert [record.purpose for record in records] == ["compaction", "agent"]
            blocks = tuple(_archived_block(record.response_context) for record in records)
            assert all(block.root.count == 1 for block in blocks)
            content_archive = await archive.transcript_repository.content_archive()
            summary_ids, answer_ids = await content_archive.content_ids_many(tuple(block.root for block in blocks))
            transcript_ids = await _transcript_content_ids(archive, run_id)
            assert summary_ids[0] not in transcript_ids
            assert answer_ids[0] in transcript_ids
            assert await content_archive.read_contents(summary_ids, codec="model-message")
            resolved = await archive.resolve_model_interactions(records)
            assert [value[1][0].parts[0].content for value in resolved] == ["retained summary"] * 2
            public = await execution.model_interactions(include_content=True)
            assert all(item.response is not None for item in public.items)


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "filesystem", "sqlite", "sql"))
async def test_identical_responses_keep_occurrences_through_checkpoint_reopen_and_cleanup(
    tmp_path: Path, backend: str,
) -> None:
    from pydantic_ai.models import ModelRequestParameters
    from pydantic_ai.models.test import TestModel
    from linktools.ai.runtime._agent_run_recorder import AgentRunRecorder
    from linktools.ai.runtime._journal import ModelRequestJournal
    from linktools.ai.runtime.state import RuntimeState
    from linktools.ai.runtime.state._step_contracts import AgentRunCheckpoint, AgentRunRecord
    from linktools.ai.storage import ObjectRef, ObjectStoreMaintenance
    from .test_evaluation_retention import Offline

    response = ModelResponse(
        parts=[TextPart("identical response " * 4000)],
        timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    async with _storage(backend, tmp_path) as storage:
        state = RuntimeState(storage)
        await state.initialize(namespace="response-checkpoint", tenant_id="tenant")
        try:
            capture = AgentRunRecorder(state.run_store, execution_id=None, agent_run_id="run")
            await capture.register_agent_run(AgentRunRecord("run"))
            journal = ModelRequestJournal(
                source_namespace="response-checkpoint", tenant_id="tenant",
                execution_id="execution", agent_run_id="run",
            )
            for sequence in (1, 2):
                capture.append_transcript_message(ModelRequest(parts=[UserPromptPart(f"question-{sequence}")]))
                messages = capture.transcript_messages()
                fact = journal.begin(sequence)
                capture.begin_model_interaction(fact, TestModel(), messages, None, ModelRequestParameters(), False)
                capture.prepare_model_interaction(fact, TestModel(), messages, None, ModelRequestParameters(), False)
                finished = journal.finish(fact.model_request_seq, status="SUCCEEDED")
                capture.finish_model_interaction(
                    finished, model=TestModel(), response=response, status="SUCCEEDED",
                    error_code=None, duration_ns=1, usage=None,
                )
                await capture.record_model_event(
                    finished, phase="completed", response=response, include_observation=True,
                )
                await capture.save_checkpoint(AgentRunCheckpoint(
                    agent_run_id="run", step_index=sequence, messages=list(capture.transcript_messages()),
                    state="complete", transcript_message_count_before=0,
                ))
                await state.run_store.checkpoint_persistence.materialize_recovery_checkpoint(
                    agent_run_id="run", require_complete=True,
                )
            if backend != "memory":
                await state.close()
                state = RuntimeState(storage)
                await state.initialize(namespace="response-checkpoint", tenant_id="tenant")
            archive = state.run_store.read_store(RuntimeDomain.CHECKPOINT)
            records = await archive.list_model_interactions(agent_run_id="run")
            assert [record.model_request_seq for record in records] == [1, 2]
            blocks = tuple(_archived_block(record.response_context) for record in records)
            assert all(block.root.count == 1 for block in blocks)
            content_archive = await archive.transcript_repository.content_archive()
            first_ids, second_ids = await content_archive.content_ids_many(tuple(block.root for block in blocks))
            assert len(first_ids) == 1 and first_ids == second_ids
            transcript_ids = await _transcript_content_ids(archive, "run")
            assert len(transcript_ids) == 4
            assert transcript_ids[0] != transcript_ids[2]
            assert transcript_ids[1:2] == transcript_ids[3:4] == first_ids
            response_content, = await content_archive.content_records(first_ids)
            raw = encode_model_messages((response,))
            assert response_content.raw_digest == hashlib.sha256(raw).hexdigest()
            assert response_content.raw_size == len(raw)
            response_records = await content_archive.store.read(lambda transaction: transaction.list_records(
                RecordQuery(kind="archive_content", sort_key_prefix=response_content.identity),
            ))
            assert len(response_records) == 1
            checkpoint = await archive.latest_checkpoint(agent_run_id="run")
            assert len(checkpoint.messages) == 4
            assert checkpoint.messages[1] == checkpoint.messages[3] == response
            resolved = await archive.resolve_model_interactions(records)
            assert [value[1] for value in resolved] == [(response,), (response,)]
            objects = state.archive_object_store(RuntimeDomain.CHECKPOINT)
            candidates = tuple([(RuntimeDomain.CHECKPOINT, ObjectRef(
                objects.store_id, item.key, item.digest, item.size,
            )) async for item in objects.list_objects()])
            assert candidates
            protected = await state.purge_unreferenced_objects(candidates, exclusive=Offline())
            assert set(protected.retained) == set(candidates)
            assert await archive.resolve_model_interactions(records) == resolved
            await archive.release_agent_run("run")
            assert await archive.list_model_interactions(agent_run_id="run") == []
            assert await archive.latest_checkpoint(agent_run_id="run") is None
            retained = await state.purge_unreferenced_objects(candidates, exclusive=Offline())
            assert set(retained.retained) == set(candidates)
            assert await archive.transcript_repository.load_messages("run") == tuple(checkpoint.messages)

            # Releasing the run leaves transcript roots; only this fixture's owners may be removed.
            transcript = archive.transcript_repository
            await transcript.create_head("unrelated")
            unrelated_head = await transcript.get_head("unrelated")

            async def remove_retained_history(transaction: StateTransaction) -> None:
                head_key = transcript.head_key("run")
                children = await transaction.list_records(RecordQuery(parent_digest=head_key))
                await transaction.delete_records((
                    *(item.key_digest for item in children),
                    transcript.projection_key("run"), head_key,
                ))

            await archive.state_store.mutate(remove_retained_history)
            assert await transcript.get_head("run") is None
            assert await transcript.load_projection("run") is None
            assert await transcript.get_head("unrelated") == unrelated_head
            assert await archive.state_store.read(lambda transaction: transaction.list_facts(
                FactQuery(transcript.transcript_stream("run")),
            )) == ()
            deleted = await state.purge_unreferenced_objects(candidates, exclusive=Offline())
            assert await content_archive.store.read(lambda transaction: transaction.list_records(
                RecordQuery(kind="archive_content", sort_key_prefix=response_content.identity),
            )) == ()
            assert await transcript.get_head("unrelated") == unrelated_head
            if isinstance(objects, ObjectStoreMaintenance):
                assert set(deleted.deleted) == set(candidates)
            else:
                assert set(deleted.blocked) == set(candidates)
                assert all([await objects.stat(reference.key) is not None for _, reference in candidates])
        finally:
            await state.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "sqlite"))
@pytest.mark.parametrize("outcome", ("FAILED", "CANCELLED"))
async def test_interrupted_partial_response_shares_retained_transcript(
    tmp_path: Path, backend: str, outcome: str,
) -> None:
    from pydantic_ai.models.function import FunctionModel

    entered = asyncio.Event()
    content = "partial response " * 2048

    async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        yield content
        entered.set()
        if outcome == "FAILED":
            raise RuntimeError("controlled partial stream failure")
        await asyncio.Event().wait()

    async with _storage(backend, tmp_path) as storage:
        async with Runtime.open(
            "response-dedup", storage=storage, features=_features(),
            models=_FeatureModels(FunctionModel(stream_function=stream)),
        ) as runtime:
            execution = await runtime.agents.get().start("question")
            if outcome == "CANCELLED":
                await asyncio.wait_for(entered.wait(), 5)
                await execution.cancel()
            assert (await execution.wait(timeout_seconds=10)).result.status == outcome
            run_id = agent_run_id(
                namespace=runtime.namespace, tenant_id=runtime.tenant_id,
                execution_id=execution.execution_id, agent_run_seq=1,
            )
            archive = runtime._state.run_store.read_store(RuntimeDomain.TRACE)
            record, = await archive.list_model_interactions(agent_run_id=run_id)
            assert record.status == outcome
            block = _archived_block(record.response_context)
            assert block.root.count == 1
            content_archive = await archive.transcript_repository.content_archive()
            response_ids = await content_archive.content_ids(block.root)
            transcript_ids = await _transcript_content_ids(archive, run_id)
            assert transcript_ids[1:2] == response_ids
            request, response, envelope, requests = await archive.resolve_model_interaction(record)
            assert response[0].parts[0].content == content
            assert response[0].state == "interrupted"
            transcript = await archive.transcript_repository.load_messages(run_id)
            assert transcript == (*request, *response)
