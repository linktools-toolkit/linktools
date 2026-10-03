#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Derived input capture plans retain source authority without preflight writes."""

from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pytest

from linktools.ai.core import JsonValue, TaskStatus
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime import CaptureInputRequest, Runtime, RuntimeStorage
from linktools.ai.runtime.state import RuntimeDomain
from linktools.ai.task import Task, TaskGraph, TaskInvocationInputRef, TaskNode, TaskNodeContext, TaskRef
from .test_evaluation_consumers import CONTEXT, PRINCIPAL, FixtureModels


async def _source_capture(runtime: Runtime) -> TaskInvocationInputRef:
    async def produce(context: TaskNodeContext[None]) -> JsonValue:
        return "accepted dependency"

    async def consume(context: TaskNodeContext[None]) -> JsonValue:
        return await context.read_dependency("value")

    producer = Task("capture.produce", produce, effect_policy="none")
    consumer = Task("capture.consume", consume, effect_policy="none")
    run = await runtime.tasks.bind(producer, consumer).start(TaskGraph("source", (
        TaskNode("value", task=producer),
        TaskNode("read", ("value",), task=consumer, input={"question": "accepted input"}),
    )), principal=PRINCIPAL, idempotency_key="source")
    assert (await run.wait(timeout_seconds=15)).status is TaskStatus.SUCCEEDED
    execution = await run.execution("read")
    capture = await runtime.executions.capture_input(execution.execution_id,
        CaptureInputRequest(PRINCIPAL, "capture", context_policy="clean"))
    assert isinstance(capture, TaskInvocationInputRef)
    return capture


@pytest.mark.asyncio
async def test_derived_input_description_matches_publication_without_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage = RuntimeStorage.filesystem(tmp_path)
    async with Runtime.open("input-planning", models=FixtureModels(), context=CONTEXT,
                            storage=storage) as runtime:
        source = await _source_capture(runtime)
        captures = runtime._input_captures
        objects = storage.object_store(RuntimeDomain.TASK)

        async def reject_write(*args: object, **kwargs: object) -> object:
            raise AssertionError("input planning must not publish payloads")

        with monkeypatch.context() as patched:
            patched.setattr(objects, "put", reject_write)
            contract = await captures.resolve_task_input(source, principal=PRINCIPAL)
            planned = await captures.describe_task_input(contract, principal=PRINCIPAL,
                source_capture=source, idempotency_key="evaluation:first")
        with pytest.raises(AIError) as missing:
            await captures.read_task(planned, principal=PRINCIPAL)
        assert missing.value.code is ErrorCode.INPUT_CAPTURE_UNAVAILABLE
        published = await captures.create_task_input(contract, principal=PRINCIPAL,
            source_capture=source, idempotency_key="evaluation:first")
        assert published == planned
        assert await captures.create_task_input(contract, principal=PRINCIPAL,
            source_capture=source, idempotency_key="evaluation:first") == planned
        other = await captures.create_task_input(contract, principal=PRINCIPAL,
            source_capture=source, idempotency_key="evaluation:second")
        assert other.capture_id != planned.capture_id
        assert await captures.read_dependency(planned, "value", principal=PRINCIPAL) == "accepted dependency"


@pytest.mark.asyncio
@pytest.mark.parametrize("changed", ["execution", "task", "binding", "dependency"])
async def test_derived_input_publication_cannot_replace_its_authorized_source(
    tmp_path: Path, changed: str,
) -> None:
    async with Runtime.open("input-authority", models=FixtureModels(), context=CONTEXT,
                            storage=RuntimeStorage.filesystem(tmp_path)) as runtime:
        source = await _source_capture(runtime)
        captures = runtime._input_captures
        contract = await captures.resolve_task_input(source, principal=PRINCIPAL)
        if changed == "execution":
            contract = replace(contract, source_execution_id="different-source")
        elif changed == "task":
            contract = replace(contract, task_ref=TaskRef("different.task", 1))
        elif changed == "binding":
            contract = replace(contract, binding={**contract.binding, "revision": 2})
        else:
            dependency = contract.dependencies[0]
            contract = replace(contract, dependencies=(replace(dependency,
                source_ref=replace(dependency.source_ref, graph_id="private-source")),))
        for operation in (captures.describe_task_input, captures.create_task_input):
            with pytest.raises(AIError) as rejected:
                await operation(contract, principal=PRINCIPAL, source_capture=source, idempotency_key="invalid")
            assert rejected.value.code is ErrorCode.INPUT_CAPTURE_UNAVAILABLE


@pytest.mark.asyncio
@pytest.mark.parametrize("materialized", [False, True])
async def test_owned_input_expiry_fences_only_its_reserved_derivation(
    tmp_path: Path, materialized: bool,
) -> None:
    storage = RuntimeStorage.filesystem(tmp_path)
    async with Runtime.open("input-fence", models=FixtureModels(), context=CONTEXT,
                            storage=storage) as runtime:
        source = await _source_capture(runtime)
        captures = runtime._input_captures
        contract = await captures.resolve_task_input(source, principal=PRINCIPAL)
        planned = await captures.describe_task_input(contract, principal=PRINCIPAL,
            source_capture=source, idempotency_key="evaluation:expired")
        if materialized:
            assert await captures.create_task_input(contract, principal=PRINCIPAL,
                source_capture=source, idempotency_key="evaluation:expired") == planned
        candidates = await captures.task_input_objects((planned,), principal=PRINCIPAL)
        assert len(candidates) == 1
        if materialized:
            with pytest.raises(AIError) as retained:
                await captures.expire_task_input_objects(candidates, principal=PRINCIPAL, now=datetime.now(timezone.utc))
            assert retained.value.code is ErrorCode.STORAGE_CONFLICT
            await storage.object_store(RuntimeDomain.TASK).delete_object(candidates[0].key, expected_digest=candidates[0].digest)
        receipt_objects = tuple(replace(reference, store_id="runtime") for reference in candidates)
        await captures.expire_task_input_objects(receipt_objects, principal=PRINCIPAL, now=datetime.now(timezone.utc))
        with pytest.raises(AIError) as rejected:
            await captures.create_task_input(contract, principal=PRINCIPAL,
                source_capture=source, idempotency_key="evaluation:expired")
        assert rejected.value.code is ErrorCode.INPUT_CAPTURE_UNAVAILABLE
        with pytest.raises(AIError) as unreadable:
            await captures.read_task(planned, principal=PRINCIPAL)
        assert unreadable.value.code is ErrorCode.INPUT_CAPTURE_UNAVAILABLE
        other = await captures.create_task_input(contract, principal=PRINCIPAL,
            source_capture=source, idempotency_key="evaluation:active")
        assert await captures.read_dependency(other, "value", principal=PRINCIPAL) == "accepted dependency"
        assert await captures.read_dependency(source, "value", principal=PRINCIPAL) == "accepted dependency"


@pytest.mark.asyncio
async def test_retained_input_capture_cannot_be_expired_before_object_cleanup(tmp_path: Path) -> None:
    async with Runtime.open("input-transfer", models=FixtureModels(), context=CONTEXT,
                            storage=RuntimeStorage.filesystem(tmp_path)) as runtime:
        source = await _source_capture(runtime)
        captures = runtime._input_captures
        derived = await captures.task_input(source, principal=PRINCIPAL, idempotency_key="evaluation:submitted")
        candidates = await captures.task_input_objects((derived,), principal=PRINCIPAL)
        assert len(candidates) == 1
        with pytest.raises(AIError) as retained:
            await captures.expire_task_input_objects(candidates, principal=PRINCIPAL, now=datetime.now(timezone.utc))
        assert retained.value.code is ErrorCode.STORAGE_CONFLICT
        assert (await captures.read_task(derived, principal=PRINCIPAL)).source_execution_id == source.source_execution_id
        assert await captures.read_dependency(derived, "value", principal=PRINCIPAL) == "accepted dependency"
