#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from collections.abc import Sequence

import pytest
from pydantic_ai.messages import BinaryContent, UploadedFile

from linktools.ai.capability import WorkspaceAccess
from linktools.ai.core import Principal
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime import ExecutionRequest
from linktools.ai.runtime._input import ExecutionInputMaterializer
from linktools.ai.workspace import SandboxSession, Workspace


class _CountingSession:
    def __init__(self, values: dict[str, bytes]) -> None:
        self.values = values
        self.reads: list[tuple[str, int | None]] = []

    async def canonicalize_path(self, path: str) -> str:
        return path

    async def read_bytes(self, path: str, *, max_bytes: int | None = None) -> bytes:
        self.reads.append((path, max_bytes))
        value = self.values[path]
        if max_bytes is not None and len(value) > max_bytes:
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        return value

    async def close(self) -> None:
        return None


class _CountingSandbox:
    def __init__(self, session: _CountingSession) -> None:
        self.session = session

    async def open(self) -> SandboxSession:
        return self.session  # type: ignore[return-value]


def _materializer(values: dict[str, bytes]) -> tuple[ExecutionInputMaterializer, _CountingSession]:
    session = _CountingSession(values)
    access = WorkspaceAccess(_CountingSandbox(session))  # type: ignore[arg-type]
    workspace = Workspace.load(".")
    return ExecutionInputMaterializer(access, workspace.policy), session


def test_native_user_content_is_canonical_and_durable() -> None:
    materializer, _session = _materializer({})
    value = ("Inspect historical input", "Return a concise result")

    async def check() -> None:
        stored = await materializer.store(value, tenant_id="tenant")
        restored = await materializer.restore(stored)
        assert restored == value
        await materializer.close()

    import asyncio

    asyncio.run(check())


@pytest.mark.asyncio
async def test_binary_content_is_stored_with_fixed_wire_timestamp() -> None:
    materializer, session = _materializer({"evidence.txt": b"error"})
    try:
        canonical_files = await materializer.canonicalize_files(
            ("evidence.txt", "evidence.txt")
        )
        canonical = await materializer.materialize(
            ("Inspect this file",),
            canonical_files,
        )
        stored = await materializer.store(canonical, tenant_id="tenant")
        assert len(session.reads) == 1
        assert canonical[0] == "Inspect this file"
        assert isinstance(canonical[1], BinaryContent)
        assert canonical[1].identifier == "evidence.txt"
        assert await materializer.restore(stored) == canonical
    finally:
        await materializer.close()


@pytest.mark.asyncio
async def test_unknown_file_media_type_fails_before_read() -> None:
    materializer, session = _materializer({"evidence.unknown": b"error"})
    try:
        files = await materializer.canonicalize_files(("evidence.unknown",))
        with pytest.raises(AIError) as raised:
            await materializer.materialize("Inspect this file", files)
        assert raised.value.code is ErrorCode.REQUEST_FIELD_INVALID
        assert raised.value.safe_details == {
            "field": "files",
            "reason": "media_type_unknown",
        }
        assert session.reads == []
    finally:
        await materializer.close()


def test_uploaded_file_is_rejected_at_request_boundary() -> None:
    uploaded = UploadedFile("file-123", "openai", media_type="text/plain")

    with pytest.raises(AIError) as raised:
        ExecutionRequest(
            user_prompt=("Inspect this file", uploaded),
            principal=Principal("user", "tenant", "local_trusted"),
            idempotency_key="user-content-request",
            memory_scope=None,
            mode="run",
            planning=False,
            thinking=False,
        )

    assert raised.value.code is ErrorCode.REQUEST_FIELD_INVALID
    assert raised.value.safe_details == {
        "field": "user_prompt",
        "reason": "uploaded_file_not_durable",
    }


def test_execution_request_keeps_canonical_user_content() -> None:
    prompt: Sequence[object] = ("Inspect historical input", "Return a result")
    request = ExecutionRequest(
        user_prompt=prompt,  # type: ignore[arg-type]
        principal=Principal("user", "tenant", "local_trusted"),
        idempotency_key="user-content-request",
        memory_scope=None,
        mode="run",
        planning=False,
        thinking=False,
    )

    assert request.user_prompt == tuple(prompt)
