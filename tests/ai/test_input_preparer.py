#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

import pytest
from linktools.ai.core import Principal
from linktools.ai.runtime._attachment import (
    DefaultAttachmentService,
    InputPreparer,
)
from linktools.ai.runtime._object import RuntimeObjectKeyFactory, read_runtime_object
from linktools.ai.runtime.state import (
    InputAttachmentPart,
    InputTextPart,
    RuntimeDomain,
    RuntimeState,
    managed_attachment_locator,
)
from linktools.ai.runtime.state._attachment_repository import AttachmentRepository
from linktools.ai.workspace import DisabledSandbox, SandboxSession, Workspace
from pydantic_ai.messages import BinaryContent


class _ReadSession:
    def __init__(self, values: dict[str, bytes]) -> None:
        self.values = values
        self.reads: list[str] = []
        self.closed = 0

    async def read_bytes(self, path: str) -> bytes:
        self.reads.append(path)
        return self.values[path]

    async def close(self) -> None:
        self.closed += 1


class _ReadSandbox:
    def __init__(self, session: _ReadSession) -> None:
        self.session = session
        self.opens = 0

    async def open(self) -> SandboxSession:
        self.opens += 1
        return self.session  # type: ignore[return-value]


async def _preparer(
    workspace: Workspace,
) -> tuple[
    RuntimeState,
    AttachmentRepository,
    DefaultAttachmentService,
    InputPreparer,
]:
    state = RuntimeState.in_memory()
    await state.initialize(namespace="workspace", tenant_id="tenant")
    repository = AttachmentRepository(
        state.execution.executions.state_store,
        namespace="workspace",
        tenant_id="tenant",
    )
    keys = RuntimeObjectKeyFactory("workspace")
    return (
        state,
        repository,
        DefaultAttachmentService(repository, state, keys),
        InputPreparer(repository, state, keys, workspace),
    )


@pytest.mark.asyncio
async def test_input_preparer_externalizes_binary_and_path_then_replays_ready(
    tmp_path: Path,
) -> None:
    path = tmp_path / "evidence.png"
    path.write_bytes(b"path-image")
    workspace = Workspace.load(tmp_path, workspace_id="workspace")
    state, repository, _uploads, preparer = await _preparer(workspace)
    principal = Principal("user", "tenant", "local_trusted")
    prompt = (
        "inspect",
        BinaryContent(data=b"direct-image", media_type="image/png"),
    )
    try:
        prepared = await preparer.prepare(
            prompt,
            ("evidence.png",),
            principal=principal,
            scope="execution.run",
            idempotency_key="prepare-key-" + "a" * 23,
        )
        assert prepared.user_prompt_codec == "linktools-input-v2"
        assert len(prepared.attachment_manifest) == 2
        assert isinstance(prepared.user_prompt.parts[0], InputTextPart)
        assert prepared.user_prompt.parts[0].text == "inspect"
        assert prepared.user_prompt.parts[1] == InputAttachmentPart("attachment", 0)
        assert prepared.user_prompt.available == (1,)
        assert tuple((item.relative, item.index) for item in prepared.user_prompt.sources) == (
            ("evidence.png", 1),
        )
        assert await read_runtime_object(
            state.object_store(RuntimeDomain.EXECUTION),
            prepared.attachment_manifest[0].content.object,
        ) == b"direct-image"
        assert await read_runtime_object(
            state.object_store(RuntimeDomain.EXECUTION),
            prepared.attachment_manifest[1].content.object,
        ) == b"path-image"

        owner_key = repository.prepare_key(
            "execution.run",
            "prepare-key-" + "a" * 23,
        )
        current = await repository.get_prepare(owner_key, tenant_id="tenant")
        assert current is not None and current.status == "READY"
        assert current.slots == () and current.input == prepared

        path.unlink()
        replayed = await preparer.prepare(
            prompt,
            ("evidence.png",),
            principal=principal,
            scope="execution.run",
            idempotency_key="prepare-key-" + "a" * 23,
        )
        assert replayed == prepared
    finally:
        await state.close()


@pytest.mark.asyncio
async def test_input_preparer_reads_duplicate_path_once(tmp_path: Path) -> None:
    session = _ReadSession({"evidence.png": b"image"})
    sandbox = _ReadSandbox(session)
    workspace = Workspace.load(
        tmp_path,
        workspace_id="workspace",
        sandbox=sandbox,  # type: ignore[arg-type]
    )
    state, _repository, _uploads, preparer = await _preparer(workspace)
    principal = Principal("user", "tenant", "local_trusted")
    try:
        prepared = await preparer.prepare(
            "inspect",
            ("evidence.png", "evidence.png"),
            principal=principal,
            scope="execution.run",
            idempotency_key="prepare-key-" + "b" * 23,
        )
        assert len(prepared.attachment_manifest) == 1
        assert prepared.user_prompt.available == (0,)
        assert session.reads == ["evidence.png"]
        assert sandbox.opens == 1
        assert session.closed == 1
    finally:
        await state.close()


@pytest.mark.asyncio
async def test_virtual_upload_prepare_does_not_open_disabled_sandbox(tmp_path: Path) -> None:
    workspace = Workspace.load(
        tmp_path,
        workspace_id="workspace",
        sandbox=DisabledSandbox(),
    )
    state, repository, uploads, preparer = await _preparer(workspace)
    principal = Principal("user", "tenant", "local_trusted")
    try:
        uploaded = await uploads.upload(
            b"image",
            media_type="image/png",
            name="upload.png",
            principal=principal,
            idempotency_key="upload-key-" + "c" * 24,
        )
        prepared = await preparer.prepare(
            "inspect",
            (uploaded.path,),
            principal=principal,
            scope="execution.run",
            idempotency_key="prepare-key-" + "d" * 23,
        )
        assert len(prepared.attachment_manifest) == 1
        assert prepared.user_prompt.available == (0,)
        kind, owner, slot = managed_attachment_locator(uploaded.path)
        assert (kind, slot) == ("u", 0)
        upload = await repository.get_upload(owner, tenant_id="tenant")
        assert upload is not None and upload.status == "HELD"
    finally:
        await state.close()
