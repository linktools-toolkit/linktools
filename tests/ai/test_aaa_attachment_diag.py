#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import asyncio
from collections.abc import Mapping
from pathlib import Path

import pytest
from pydantic_ai.models.test import TestModel

from linktools.ai.core import JsonValue
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime import Runtime, RuntimeState
from linktools.ai.workspace import Workspace


class _Binding:
    route_id = "default"
    provider = "test"
    model_identity = "test:test"
    fingerprint = "d" * 64
    semantic_payload: dict[str, JsonValue] = {"provider": "test", "model": "test"}

    def materialize(self) -> TestModel:
        return TestModel(custom_output_text="ok")


class _Models:
    def snapshot(self) -> "_Models":
        return self

    def resolve(self, route_id: str) -> _Binding:
        if route_id != "default":
            raise AssertionError(route_id)
        return _Binding()

    def restore(
        self,
        payload: Mapping[str, JsonValue],
        *,
        route_id: str | None = None,
    ) -> _Binding:
        if route_id not in {None, "default"} or dict(payload) != _Binding.semantic_payload:
            raise AIError(ErrorCode.MODEL_CONNECTION_NOT_FOUND)
        return _Binding()


def _await_chain(value: object) -> str:
    result: list[str] = []
    current = value
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        frame = getattr(current, "cr_frame", None) or getattr(current, "ag_frame", None)
        if frame is not None:
            code = frame.f_code
            result.append(f"{Path(code.co_filename).name}:{frame.f_lineno}:{code.co_name}")
        current = getattr(current, "cr_await", None) or getattr(current, "ag_await", None)
    return " -> ".join(result)


@pytest.mark.asyncio
async def test_managed_worker_task_snapshot(tmp_path: Path) -> None:
    (tmp_path / "evidence.txt").write_bytes(b"immutable evidence")
    state = RuntimeState.in_memory()
    async with Runtime.open(
        Workspace.load(tmp_path, workspace_id="portable-runtime"),
        models=_Models(),  # type: ignore[arg-type]
        state=state,
    ) as runtime:
        execution = await runtime.agent("default").start(
            "inspect the attachment",
            attachments=("evidence.txt",),
            idempotency_key="diag-managed-runtime",
        )
        await asyncio.sleep(1)
        current = asyncio.current_task()
        tasks = sorted(
            f"{task.get_name()}={_await_chain(task.get_coro())}"
            for task in asyncio.all_tasks()
            if task is not current and not task.done()
        )
        record = await state.execution.executions.get(
            execution.execution_id,
            tenant_id="default",
        )
        pytest.fail(
            f"status={None if record is None else record.status.value}; "
            + " || ".join(tasks)
        )
