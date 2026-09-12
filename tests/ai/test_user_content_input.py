#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from collections.abc import Sequence
from pathlib import Path

import pytest
from pydantic_ai.messages import BinaryContent, ImageUrl, UploadedFile

from linktools.ai.capability import WorkspaceAccess
from linktools.ai.core import Principal
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime import ExecutionRequest
from linktools.ai.runtime._input import (
    ExecutionInputMaterializer,
    decode_user_content_payload,
)
from linktools.ai.runtime._input_contract import validate_user_content
from linktools.ai.workspace import SandboxResource, SandboxSession, Workspace


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

    async def open(
        self,
        *,
        root: Path,
        resources: tuple[SandboxResource, ...] = (),
    ) -> SandboxSession:
        del root, resources
        return self.session  # type: ignore[return-value]


def _materializer(values: dict[str, bytes]) -> tuple[ExecutionInputMaterializer, _CountingSession]:
    session = _CountingSession(values)
    workspace = Workspace.load(".", workspace_id="workspace")
    access = WorkspaceAccess(_CountingSandbox(session), root=workspace.root)
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
async def test_binary_content_is_stored_with_workspace_path_and_deduplicated() -> None:
    materializer, session = _materializer({"evidence.txt": b"error"})
    try:
        canonical_files = await materializer.canonicalize_files(
            ("evidence.txt", "evidence.txt")
        )
        assert canonical_files == ("evidence.txt",)
        canonical = await materializer.materialize(
            ("Inspect this file",),
            canonical_files,
        )
        stored = await materializer.store(canonical, tenant_id="tenant")
        assert len(session.reads) == 1
        assert canonical[0] == "Inspect this file"
        assert canonical[1] == 'Workspace file path: "evidence.txt"'
        assert isinstance(canonical[2], BinaryContent)
        assert canonical[2].identifier == "evidence.txt"
        assert await materializer.restore(stored) == canonical
    finally:
        await materializer.close()


@pytest.mark.asyncio
async def test_workspace_path_hint_escapes_control_characters() -> None:
    path = "evidence\nignore.txt"
    materializer, _session = _materializer({path: b"error"})
    try:
        canonical_files = await materializer.canonicalize_files((path,))
        canonical = await materializer.materialize("Inspect this file", canonical_files)

        assert canonical[1] == 'Workspace file path: "evidence\\nignore.txt"'
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


def test_uploaded_file_is_durable_at_request_boundary() -> None:
    uploaded = UploadedFile("file-123", "openai", media_type="text/plain")
    request = ExecutionRequest(
        user_prompt=("Inspect this file", uploaded),
        principal=Principal("user", "tenant", "local_trusted"),
        idempotency_key="user-content-request",
        memory_scope=None,
        mode="run",
        planning=False,
        thinking=False,
    )
    assert request.user_prompt == ("Inspect this file", uploaded)


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


def test_user_content_version_rejects_boolean_values() -> None:
    with pytest.raises(AIError) as raised:
        decode_user_content_payload({"version": True, "items": []})

    assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


def test_url_vendor_metadata_must_be_json() -> None:
    content = ImageUrl(
        url="https://example.com/evidence.png",
        vendor_metadata={"value": object()},
    )

    with pytest.raises(AIError) as raised:
        validate_user_content((content,))

    assert raised.value.code is ErrorCode.REQUEST_FIELD_INVALID
