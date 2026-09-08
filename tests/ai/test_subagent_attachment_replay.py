#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Subagent attachment replay must resolve the adopted child before source access."""

from types import SimpleNamespace

import pytest

from linktools.ai.core import (
    ExecutionStatus,
    Principal,
    UsageMetrics,
    canonical_sha256,
)
from linktools.ai.runtime._subagent import SubagentAttachmentRuntime, SubagentDispatcher
from linktools.ai.runtime._subagent_attachment import SubagentAttachmentPreparer
from linktools.ai.runtime.service_api import ExecutionHandle, ExecutionResult
from linktools.ai.runtime.state import (
    AttachmentEntry,
    AttachmentPresentation,
    ContentRef,
    InputAttachmentPart,
    InputTextPart,
    InputV2,
    PathOrigin,
    PreparedInput,
    RuntimeDomain,
    input_v2_digest,
    managed_attachment_path,
)
from linktools.ai.spec import SubagentRef
from linktools.ai.storage import ObjectRef


def _prepared() -> PreparedInput:
    entry = AttachmentEntry(
        managed_attachment_path("p", "a" * 64, 0),
        "evidence.txt",
        "text/plain",
        AttachmentPresentation(None, None),
        ContentRef(
            RuntimeDomain.EXECUTION.value,
            None,
            ObjectRef("memory", "body", "2d711642b726b04401627ca9fbac32f5c8530fb1903cc4db02258717921a4881", 1),
        ),
    )
    prompt = InputV2(
        2,
        (
            InputTextPart("text", "inspect"),
            InputAttachmentPart("attachment", 0),
        ),
        (),
        (),
    )
    manifest = (entry,)
    return PreparedInput(
        1,
        "linktools-input-v2",
        prompt,
        manifest,
        "b" * 64,
        input_v2_digest(prompt, manifest),
        PathOrigin(1, "workspace", "posix", "/workspace"),
    )


class _ReplayPreparer(SubagentAttachmentPreparer):
    def __init__(self, prepared: PreparedInput) -> None:
        self.prepared = prepared
        self.adopted_calls = 0
        self.replay_calls = 0

    async def adopted(
        self,
        task: str,
        attachments: tuple[str, ...],
        *,
        idempotency_key: str,
    ) -> bool:
        del task, attachments, idempotency_key
        self.adopted_calls += 1
        return True

    def replay(
        self,
        task: str,
        attachments: tuple[str, ...],
        execution: object,
    ) -> PreparedInput:
        del task, attachments, execution
        self.replay_calls += 1
        return self.prepared


class _ExecutionService:
    def __init__(self) -> None:
        self.starts = 0

    async def start_subagent(
        self,
        binding_digest: str,
        request: object,
        *,
        parent_execution_id: str,
        root_execution_id: str,
    ) -> ExecutionHandle:
        del binding_digest, request, parent_execution_id, root_execution_id
        self.starts += 1
        return ExecutionHandle("child")

    async def wait(
        self,
        execution_id: str,
        *,
        principal: Principal,
    ) -> ExecutionResult:
        del principal
        output = {"ok": True}
        return ExecutionResult(
            execution_id,
            ExecutionStatus.SUCCEEDED,
            output,
            canonical_sha256(output),
            UsageMetrics(),
            None,
            {},
        )


@pytest.mark.asyncio
async def test_subagent_attachment_replay_hits_child_before_parent_source_grant() -> None:
    prepared = _prepared()
    preparer = _ReplayPreparer(prepared)
    execution = _ExecutionService()
    dispatcher = SubagentDispatcher(
        None,  # type: ignore[arg-type]
        None,  # type: ignore[arg-type]
        execution,  # type: ignore[arg-type]
    )
    existing = SimpleNamespace(
        execution_id="child",
        parent_execution_id="parent",
        root_execution_id="root",
        binding_digest="binding",
        planning=False,
        thinking=False,
    )
    find_calls = 0
    grant_calls = 0

    async def find_child(idempotency_key: str):  # type: ignore[no-untyped-def]
        nonlocal find_calls
        assert idempotency_key.startswith("subagent:")
        find_calls += 1
        return existing

    async def grant(path: str):  # type: ignore[no-untyped-def]
        nonlocal grant_calls
        grant_calls += 1
        raise AssertionError(f"replay must not read parent source: {path}")

    dispatcher.bind_attachment_runtime(
        "parent",
        SubagentAttachmentRuntime(preparer, grant, find_child),  # type: ignore[arg-type]
    )

    result = await dispatcher.dispatch(
        parent_execution_id="parent",
        root_execution_id="root",
        memory_scope=None,
        principal=Principal("principal", "tenant", "service"),
        ref=SubagentRef("agent", "child"),
        mode="run",
        user_prompt="inspect",
        attachments=("gone.txt",),
        invocation_id="same-call",
    )

    assert result["execution_id"] == "child"
    assert find_calls == 1
    assert grant_calls == 0
    assert preparer.adopted_calls == 1
    assert preparer.replay_calls == 1
    assert execution.starts == 1
