#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from linktools.ai.persistence import Persistence
"""External input mismatches fail the execution without declaring Runtime storage corrupt."""

import asyncio
import base64
import hashlib
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models.function import FunctionModel

from linktools.ai.asset import AssetKey, AssetVersionRef
from linktools.ai.capability import AgentContext, ExecutionCapability, ReferencedInputsCapability, TraceCapability
from linktools.ai.core import ExecutionStatus, service_principal
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.model import ReferencedContent
from linktools.ai.runtime import Runtime, RuntimeFeatures
from linktools.ai.persistence import PersistenceStorage
from linktools.ai.spec import CapabilityRef
from linktools.ai.storage import StorageEntryRevision


_BODY = b"synthetic-admitted-file-content-never-retained"
_REF = AssetVersionRef(
    AssetKey("document", "immutable-input"), "configured-files", StorageEntryRevision(1),
    hashlib.sha256(_BODY).hexdigest(), len(_BODY),
)
_SOURCE = CapabilityRef("capability", "input-files", 1)


@pytest.mark.asyncio
@pytest.mark.parametrize("returned", (b"different-length-input", b"x" * len(_BODY)), ids=("size", "digest"))
async def test_wrong_input_bytes_commit_failed_result_and_reopen_without_loading(
    tmp_path: Path, returned: bytes,
) -> None:
    loads: list[ReferencedContent] = []
    dispatches: list[bool] = []

    async def load(ctx: AgentContext[None], content: ReferencedContent) -> bytes:
        assert ctx.principal == principal
        assert content.ref == _REF
        loads.append(content)
        return returned

    async def respond(messages: list[ModelMessage], info: object) -> AsyncIterator[str]:
        dispatches.append(True)
        yield "unexpected provider invocation"

    class Binding:
        provider = "fixture"
        model_identity = "fixture:reference-failure"
        contract = {"provider": "fixture", "input_mime_types": ["application/pdf"]}

        def materialize(self) -> FunctionModel:
            return FunctionModel(stream_function=respond)

    features = RuntimeFeatures("reference-failure")
    features.model("default", Binding())
    features.agent("default", allow_tools=(), allow_skills=(), allow_subagents=())
    features.capability(ExecutionCapability(retain_history=True), scope="runtime")
    features.capability(TraceCapability(record_http_bodies=True), scope="runtime")
    features.capability(ReferencedInputsCapability(load), id=_SOURCE.id,
                        revision=_SOURCE.revision, scope="agent")
    content = ReferencedContent(_REF, media_type="application/pdf", source=_SOURCE)
    async with Persistence.open('reference-failure', storage=PersistenceStorage.filesystem(tmp_path)) as persistence_67, Runtime.attach(persistence=persistence_67, features=features) as runtime:
        principal = runtime.default_principal
        execution = await runtime.agents.get().start(["inspect", content], principal=principal)
        result = (await execution.wait(timeout_seconds=10)).result
        assert result.status is ExecutionStatus.FAILED
        assert result.error_code == ErrorCode.ASSET_CONTENT_MISMATCH.value
        assert result.error_diagnostics is not None
        assert result.error_diagnostics.content_omission_reason == "referenced_content_lifecycle"
        authoritative = await runtime.executions.result(execution.execution_id, principal=principal)
        assert authoritative.status is ExecutionStatus.FAILED
        assert authoritative.error_code == result.error_code
        interactions = (await execution.model_interactions(include_content=True)).items
        assert len(interactions) == 1
        assert interactions[0].status == "FAILED"
        assert interactions[0].error_code == result.error_code
        execution_id = execution.execution_id
    assert not dispatches and len(loads) == 1
    async with Persistence.open('reference-failure', storage=PersistenceStorage.filesystem(tmp_path)) as persistence_85, Runtime.attach(persistence=persistence_85, features=features) as reopened:
        restored = await reopened.executions.result(execution_id, principal=principal)
        assert restored.status is ExecutionStatus.FAILED
        assert restored.error_code == ErrorCode.ASSET_CONTENT_MISMATCH.value
        assert restored.error_diagnostics.to_payload() == result.error_diagnostics.to_payload()
    assert not dispatches and len(loads) == 1
    stored = b"\n".join(path.read_bytes() for path in tmp_path.rglob("*") if path.is_file())
    for value in (_BODY, returned):
        assert value not in stored
        assert base64.b64encode(value) not in stored


@pytest.mark.asyncio
@pytest.mark.parametrize("error", (AIError(ErrorCode.STORAGE_INTEGRITY_ERROR), asyncio.CancelledError()))
async def test_loader_infrastructure_failure_and_cancellation_are_not_reclassified(
    error: BaseException,
) -> None:
    async def load(ctx: AgentContext[None], content: ReferencedContent) -> bytes:
        raise error

    context = AgentContext(None, service_principal("tenant", "admitted"),
                           "fixture", None, "execution", {})
    with pytest.raises(type(error)) as caught:
        await ReferencedInputsCapability(load).load(
            context, ReferencedContent(_REF, media_type="application/pdf", source=_SOURCE),
        )
    assert caught.value is error


def test_asset_version_integrity_contract_remains_a_storage_error() -> None:
    with pytest.raises(AIError) as caught:
        _REF.verify_content(b"changed")
    assert caught.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
