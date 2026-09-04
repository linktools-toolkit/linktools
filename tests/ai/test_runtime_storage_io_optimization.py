#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Focused I/O invariants for Runtime storage optimization."""

import hashlib
from collections.abc import Awaitable, Callable
from dataclasses import replace
from pathlib import Path
from typing import TypeVar

import pytest
from linktools.ai.migrate import provision_database
from linktools.ai.runtime import RuntimeState
from linktools.ai.runtime.state import FactQuery, RuntimeDomain
from linktools.ai.runtime.state._contracts import (
    ContextProjection,
    LoadedModelContext,
)
from linktools.ai.runtime.state._history import TranscriptRepository
from linktools.ai.runtime.state._readmodel import (
    ExecutionReadModelBuild,
    ExecutionReadModelRepository,
    ExecutionReadModelStatus,
)
from linktools.ai.runtime.state._store import StateStore, StateTransaction
from linktools.ai.storage import FilesystemObjectStore, SqlObjectStore
from linktools.ai.storage import _object as object_module
from sqlalchemy.ext.asyncio import create_async_engine

pytestmark = pytest.mark.asyncio

ResultT = TypeVar("ResultT")


async def _chunks(value: bytes):
    yield value


class _CountingStateStore:
    def __init__(self, delegate: StateStore) -> None:
        self._delegate = delegate
        self.mutation_count = 0

    async def read(
        self,
        operation: Callable[[StateTransaction], Awaitable[ResultT]],
    ) -> ResultT:
        return await self._delegate.read(operation)

    async def mutate(
        self,
        operation: Callable[[StateTransaction], Awaitable[ResultT]],
    ) -> ResultT:
        self.mutation_count += 1
        return await self._delegate.mutate(operation)


async def test_filesystem_object_store_syncs_payload_before_publish(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    original_sync = object_module._sync_file
    original_publish = object_module._publish_filesystem_object

    def sync(path: Path) -> None:
        events.append("sync")
        original_sync(path)

    def publish(*args, **kwargs):
        events.append("publish")
        return original_publish(*args, **kwargs)

    monkeypatch.setattr(object_module, "_sync_file", sync)
    monkeypatch.setattr(object_module, "_publish_filesystem_object", publish)
    store = FilesystemObjectStore(tmp_path / "objects")
    payload = b"filesystem-payload"
    digest = hashlib.sha256(payload).hexdigest()

    stat = await store.put(
        "payload",
        _chunks(payload),
        expected_size=len(payload),
        expected_digest=digest,
    )

    assert stat.digest == digest
    assert events[:2] == ["sync", "publish"]


async def test_sql_object_store_does_not_sync_staging_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "objects.db"
    engine = create_async_engine(f"sqlite+aiosqlite:///{path}")
    await provision_database(engine)
    store = SqlObjectStore(engine)
    payload = b"sql-payload"
    digest = hashlib.sha256(payload).hexdigest()

    def fail_sync(_path: Path) -> None:
        raise AssertionError("SQL staging payload must not be fsynced")

    monkeypatch.setattr(object_module, "_sync_file", fail_sync)
    try:
        stat = await store.put(
            "payload",
            _chunks(payload),
            expected_size=len(payload),
            expected_digest=digest,
        )
        assert stat.digest == digest
        assert await store.stat("payload") == stat
    finally:
        await engine.dispose()


async def test_session_model_context_reuses_observed_projection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = TranscriptRepository(
        object(),  # type: ignore[arg-type]
        object_store=None,
        namespace="io-history",
        tenant_id="tenant",
        runtime_domain=RuntimeDomain.CONVERSATION,
    )
    projection = ContextProjection((), "projection")
    expected = LoadedModelContext(())
    projection_reads = 0

    async def load_projection(_owner_id: str) -> ContextProjection:
        nonlocal projection_reads
        projection_reads += 1
        return projection

    async def load_projected(
        owner_id: str,
        observed: ContextProjection,
    ) -> LoadedModelContext:
        assert owner_id == "history"
        assert observed is projection
        return expected

    monkeypatch.setattr(repository, "load_projection", load_projection)
    monkeypatch.setattr(
        repository,
        "_load_model_context_from_projection",
        load_projected,
    )

    result = await repository.load_session_model_context(
        "history",
        tenant_id="tenant",
    )

    assert result is expected
    assert projection_reads == 1


async def test_message_spans_pass_observed_head_to_seek(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = TranscriptRepository(
        object(),  # type: ignore[arg-type]
        object_store=None,
        namespace="io-history",
        tenant_id="tenant",
        runtime_domain=RuntimeDomain.EXECUTION,
    )
    head = replace(repository.empty_head("run"), message_count=1)
    head_reads = 0

    async def get_head(owner_id: str):
        nonlocal head_reads
        assert owner_id == "run"
        head_reads += 1
        return head

    async def stop_at_seek(
        owner_id: str,
        view_index: int,
        **kwargs,
    ) -> int | None:
        assert owner_id == "run"
        assert view_index == 0
        assert kwargs["observed_head"] is head
        raise RuntimeError("seek-observed")

    monkeypatch.setattr(repository, "get_head", get_head)
    monkeypatch.setattr(repository, "_seek_fact_sequence", stop_at_seek)

    with pytest.raises(RuntimeError, match="seek-observed"):
        await repository.load_message_spans("run", ((0, 1),))

    assert head_reads == 1


async def test_execution_read_model_batches_streams_by_ordinal() -> None:
    state = RuntimeState.in_memory()
    await state.initialize(namespace="io-readmodel", tenant_id="tenant")
    try:
        delegate = state.execution.executions.state_store
        store = _CountingStateStore(delegate)
        repository = ExecutionReadModelRepository(
            store,  # type: ignore[arg-type]
            namespace="io-readmodel",
            tenant_id="tenant",
        )
        owner, fence, claimed = await repository._claim("execution")
        assert claimed is None
        build = ExecutionReadModelBuild(
            "execution",
            "tenant",
            "source",
            tuple({"index": index} for index in range(129)),
            tuple({"index": index} for index in range(257)),
            ({"index": 0},),
        )

        await repository._write_build(build, owner, fence)

        assert store.mutation_count == 4
        stored = await delegate.read(
            lambda transaction: transaction.get_record(
                repository._record_key("execution")
            )
        )
        assert stored is not None
        value = repository._decode_record(stored)
        assert value.status is ExecutionReadModelStatus.COMPLETE
        assert (value.trace_count, value.history_count, value.transcript_count) == (
            129,
            257,
            1,
        )
        history_facts = await delegate.read(
            lambda transaction: transaction.list_facts(
                FactQuery(repository._stream("execution", "history"))
            )
        )
        assert tuple(len(fact.data["items"]) for fact in history_facts) == (
            128,
            128,
            1,
        )
    finally:
        await state.close()


async def test_empty_execution_read_model_uses_one_publish_mutation() -> None:
    state = RuntimeState.in_memory()
    await state.initialize(namespace="io-readmodel-empty", tenant_id="tenant")
    try:
        delegate = state.execution.executions.state_store
        store = _CountingStateStore(delegate)
        repository = ExecutionReadModelRepository(
            store,  # type: ignore[arg-type]
            namespace="io-readmodel-empty",
            tenant_id="tenant",
        )
        owner, fence, claimed = await repository._claim("execution")
        assert claimed is None
        await repository._write_build(
            ExecutionReadModelBuild(
                "execution",
                "tenant",
                "source",
                (),
                (),
                (),
            ),
            owner,
            fence,
        )

        assert store.mutation_count == 2
    finally:
        await state.close()
