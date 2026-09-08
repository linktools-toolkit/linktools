#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import asyncio
from collections.abc import Mapping
from pathlib import Path

import pytest
from pydantic_ai.models.test import TestModel

from linktools.ai.core import ExecutionStatus, JsonValue
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime import Runtime, RuntimeState
from linktools.ai.runtime.state._attachment_repository import AttachmentRepository
from linktools.ai.workspace import Workspace


class _TextModelBinding:
    route_id = "default"
    provider = "test"
    model_identity = "test:test"
    fingerprint = "d" * 64
    semantic_payload: dict[str, JsonValue] = {
        "provider": "test",
        "model": "test",
    }

    def materialize(self) -> TestModel:
        return TestModel(custom_output_text="ok")


class _TextModels:
    def snapshot(self) -> "_TextModels":
        return self

    def resolve(self, route_id: str) -> _TextModelBinding:
        if route_id != "default":
            raise AssertionError(route_id)
        return _TextModelBinding()

    def restore(
        self,
        payload: Mapping[str, JsonValue],
        *,
        route_id: str | None = None,
    ) -> _TextModelBinding:
        if (
            route_id not in {None, "default"}
            or dict(payload) != _TextModelBinding.semantic_payload
        ):
            raise AIError(ErrorCode.MODEL_CONNECTION_NOT_FOUND)
        return _TextModelBinding()


def _await_chain(value: object) -> str:
    parts: list[str] = []
    current = value
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        frame = getattr(current, "cr_frame", None) or getattr(current, "ag_frame", None)
        code = None if frame is None else frame.f_code
        if code is not None:
            parts.append(f"{Path(code.co_filename).name}:{frame.f_lineno}:{code.co_name}")
        next_value = getattr(current, "cr_await", None)
        if next_value is None:
            next_value = getattr(current, "ag_await", None)
        current = next_value
    return " -> ".join(parts)


@pytest.mark.asyncio
async def test_managed_path_admission_replays_without_source_read(tmp_path: Path) -> None:
    source = tmp_path / "evidence.txt"
    source.write_bytes(b"immutable evidence")
    workspace = Workspace.load(tmp_path, workspace_id="portable-runtime")
    state = RuntimeState.in_memory()
    key = "managed-runtime-replay-key"

    async with Runtime.open(
        workspace,
        models=_TextModels(),  # type: ignore[arg-type]
        state=state,
    ) as runtime:
        pending = await runtime.agent("default").start(
            "inspect the attachment",
            attachments=("evidence.txt",),
            idempotency_key=key,
        )
        await asyncio.sleep(1)
        backend = runtime.execution._backend  # type: ignore[attr-defined]
        task = None if backend is None else backend._tasks.get(pending.execution_id)
        record = await state.execution.executions.get(
            pending.execution_id,
            tenant_id="default",
        )
        if task is not None and not task.done():
            pytest.fail(
                "managed worker stalled before terminal: "
                f"status={None if record is None else record.status.value} "
                f"chain={_await_chain(task.get_coro())}"
            )
        first = await pending.wait(timeout_seconds=10)
        assert first.status is ExecutionStatus.SUCCEEDED

        execution = await state.execution.executions.get(
            first.execution_id,
            tenant_id="default",
        )
        assert execution is not None
        assert len(execution.attachment_manifest) == 1
        assert execution.input_digest is not None
        assert execution.path_origin is not None

        repository = AttachmentRepository(
            state.execution.executions.state_store,
            namespace=workspace.workspace_id,
            tenant_id="default",
        )
        owner = repository.prepare_key("execution.run", key)
        preparation = await repository.get_prepare(owner, tenant_id="default")
        assert preparation is not None
        assert preparation.status == "ADOPTED"
        assert preparation.input is None
        assert preparation.target is not None
        assert preparation.target.at.resource == "state:execution"
        assert preparation.target.at.key == repository._key(
            "execution", first.execution_id
        ).hex()

        checkpoint = await state.recovery.checkpoints.get(
            first.execution_id,
            tenant_id="default",
        )
        assert checkpoint is not None
        assert checkpoint.input.attachment_manifest == execution.attachment_manifest
        assert checkpoint.input.input_digest == execution.input_digest
        assert checkpoint.input.path_origin == execution.path_origin

        source.unlink()
        replayed = await runtime.agent("default").run(
            "inspect the attachment",
            attachments=("evidence.txt",),
            idempotency_key=key,
            timeout_seconds=10,
        )
        assert replayed.execution_id == first.execution_id
        assert replayed.status is ExecutionStatus.SUCCEEDED
