#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from linktools.ai.persistence import Persistence
"""Task artifact publication uses shared persistence and replays consistently."""

import hashlib
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

import pytest

from linktools.ai.core import TaskStatus
from linktools.ai.errors import AIError, ErrorCode

from linktools.ai.persistence import PersistenceStorage
from linktools.ai.task.engine import TaskEngine
from linktools.ai.persistence import PersistenceDomain
from linktools.ai.task import Task, TaskGraph, TaskNode, TaskNodeContext

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.asyncio
async def test_task_artifact_publish_readback_and_idempotency() -> None:
    body = b"published task artifact"
    digest = hashlib.sha256(body).hexdigest()
    storage = PersistenceStorage.in_memory()

    async def chunks() -> AsyncIterator[bytes]:
        yield body

    async def publish(context: TaskNodeContext[None]) -> str:
        assert context.artifacts is not None
        record = await context.artifacts.publish(
            "report.txt", chunks(), mime_type="text/plain",
            expected_size=len(body), expected_digest=digest,
        )
        assert record.mime_type == "text/plain"
        assert record.execution_id == context.execution_id
        assert record.producer == f"task:{context.graph_id}:{context.node_id}"
        assert record.object_ref.digest == digest
        assert record.size == len(body)
        stored = await persistence_67.artifact.records.get_metadata(
            record.artifact_id, tenant_id=context.principal.tenant_id,
        )
        assert stored == record
        objects = persistence_67.object_store(PersistenceDomain.ARTIFACT)
        assert b"".join([chunk async for chunk in objects.open(record.object_ref.key)]) == body
        replay = await context.artifacts.publish(
            "report.txt", chunks(), mime_type="text/plain",
            expected_size=len(body), expected_digest=digest,
        )
        assert replay == record
        for mime_type, size, expected_digest in (
            ("application/json", len(body), digest),
            ("text/plain", len(body) + 1, digest),
            ("text/plain", len(body), "b" * 64),
        ):
            with pytest.raises(AIError) as conflict:
                await context.artifacts.publish(
                    "report.txt", chunks(), mime_type=mime_type,
                    expected_size=size, expected_digest=expected_digest,
                )
            assert conflict.value.code is ErrorCode.IDEMPOTENCY_CONFLICT
        assert await persistence_67.artifact.records.get_metadata(
            record.artifact_id, tenant_id=context.principal.tenant_id,
        ) == record
        return record.artifact_id

    task = Task("artifact.publish", publish, effect_policy="none")
    graph = TaskGraph("artifact-publish", (TaskNode("publish", task=task),))
    async with Persistence.open('artifact-publish', storage=storage) as persistence_67, TaskEngine.open(task, persistence=persistence_67) as engine:
        run = await engine.start(graph, idempotency_key="artifact-publish")
        assert (await run.wait(timeout_seconds=10)).result.status is TaskStatus.SUCCEEDED
        assert isinstance(await run.result("publish"), str)


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "sqlite"))
@pytest.mark.parametrize("difference", ("same", "mime", "content"))
async def test_task_artifact_concurrent_publication_preserves_semantic_identity(tmp_path: "Path", monkeypatch: pytest.MonkeyPatch, backend: str, difference: str) -> None:
    import asyncio
    storage = PersistenceStorage.in_memory() if backend == "memory" else PersistenceStorage.sqlite(tmp_path / "state.sqlite")
    async with Persistence.open("artifact-race", storage=storage) as state:
        put = state.artifact.records.put_metadata
        arrived = 0
        ready = asyncio.Event()
        async def synchronized(record):
            nonlocal arrived
            arrived += 1
            if arrived == 2:
                ready.set()
            await asyncio.wait_for(ready.wait(), timeout=3)
            return await put(record)
        monkeypatch.setattr(state.artifact.records, "put_metadata", synchronized)
        async def body(ctx):
            async def publish(data, mime):
                async def chunks():
                    yield data
                return await ctx.artifacts.publish("report", chunks(), mime_type=mime, expected_size=len(data), expected_digest=hashlib.sha256(data).hexdigest())
            results = await asyncio.gather(
                publish(b"first", "text/plain"),
                publish(b"other" if difference == "content" else b"first", "application/json" if difference == "mime" else "text/plain"),
                return_exceptions=True,
            )
            if difference == "same":
                assert not any(isinstance(r, BaseException) for r in results)
                assert results[0] == results[1]
            else:
                successes = [r for r in results if not isinstance(r, BaseException)]
                errors = [r for r in results if isinstance(r, AIError)]
                assert len(successes) == len(errors) == 1
                assert errors[0].code is ErrorCode.IDEMPOTENCY_CONFLICT
            return "done"
        task = Task("publish", body, effect_policy="none")
        async with TaskEngine.open(task, persistence=state) as engine:
            run = await engine.start(TaskGraph("graph", (TaskNode("node", task=task),)), idempotency_key="start")
            assert (await run.wait(timeout_seconds=5)).result.status is TaskStatus.SUCCEEDED


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", ("before", "after", "cancel", "object", "readback"))
async def test_task_artifact_publication_resolves_only_metadata_commit_uncertainty(monkeypatch: pytest.MonkeyPatch, fault: str) -> None:
    import asyncio
    async with Persistence.open("artifact-unknown", storage=PersistenceStorage.in_memory()) as state:
        put = state.artifact.records.put_metadata
        error = asyncio.CancelledError() if fault == "cancel" else AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN)
        async def uncertain(record):
            if fault in {"after", "readback"}:
                await put(record)
            if fault == "readback":
                monkeypatch.setattr(state.artifact.records, "get_metadata", unavailable_metadata)
            raise error
        async def unavailable_object(*args, **kwargs):
            raise error
        async def unavailable_metadata(*args, **kwargs):
            raise AIError(ErrorCode.STORAGE_UNAVAILABLE)
        if fault == "object":
            monkeypatch.setattr(state.object_store(PersistenceDomain.ARTIFACT), "put", unavailable_object)
        else:
            monkeypatch.setattr(state.artifact.records, "put_metadata", uncertain)
        async def body(ctx):
            data = b"content"
            async def chunks():
                yield data
            async def publish():
                return await ctx.artifacts.publish("report", chunks(), mime_type="text/plain", expected_size=len(data), expected_digest=hashlib.sha256(data).hexdigest())
            if fault == "readback":
                with pytest.raises(AIError) as raised:
                    await publish()
                assert raised.value.code is ErrorCode.STORAGE_COMMIT_UNKNOWN
            elif fault == "after":
                record = await publish()
                assert record == await state.artifact.records.get_metadata(record.artifact_id, tenant_id=state.tenant_id)
                assert await publish() == record
            else:
                with pytest.raises(type(error)) as raised:
                    await publish()
                assert raised.value is error
            return "done"
        task = Task("publish", body, effect_policy="none")
        async with TaskEngine.open(task, persistence=state) as engine:
            run = await engine.start(TaskGraph("graph", (TaskNode("node", task=task),)), idempotency_key="start")
            assert (await run.wait(timeout_seconds=5)).result.status is TaskStatus.SUCCEEDED
