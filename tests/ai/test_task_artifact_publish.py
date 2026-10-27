#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from linktools.ai.persistence import Persistence
"""Task artifact publication uses shared persistence and replays consistently."""

import hashlib
from collections.abc import AsyncIterator

import pytest

from linktools.ai.core import TaskStatus
from linktools.ai.errors import AIError, ErrorCode

from linktools.ai.persistence import PersistenceStorage
from linktools.ai.task.engine import TaskEngine
from linktools.ai.persistence import PersistenceDomain
from linktools.ai.task import Task, TaskGraph, TaskNode, TaskNodeContext


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
