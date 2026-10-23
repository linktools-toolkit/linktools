#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Business-tool attachment references survive durable replay and SDK consumption."""

import asyncio
import base64
import json
import multiprocessing
from collections.abc import Awaitable, Callable
from multiprocessing.connection import Connection
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic_ai.messages import (
    BinaryContent,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextContent,
    TextPart,
    ToolCallPart,
    ToolReturn,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo

from linktools.ai.capability import (
    AgentContext, CheckpointCapability, ExecutionCapability,
    ReferencedInputsCapability, TraceCapability,
)
from linktools.ai.core import (
    AuthorizationAction, ExecutionStatus, Principal, ResourceRef,
    TenantAuthorizationPolicy, service_principal,
)
from linktools.ai.model import ReferencedContent
from linktools.ai.runtime import Runtime, RuntimeFeatures, RuntimeStorage
from linktools.ai.runtime._tool import ToolOperationDecision

from ._runtime_test_helpers import _Models, _UsageFunctionModel, model_features
from .test_referenced_content import _BODY, _SOURCE, _content
from .test_tool_return_persistence import _tool_bridge


@pytest.mark.asyncio
@pytest.mark.parametrize("replay", (False, True))
async def test_business_tool_reference_is_consumed_once_after_live_or_durable_return(
    replay: bool,
) -> None:
    original = ToolReturn(
        return_value={"name": "synthetic-document"},
        content=[_content()],
        metadata={"fixture": "referenced-tool"},
    )
    bridge = _tool_bridge()
    payload = await bridge._result_payload(
        ToolOperationDecision("operation", "worker", 1, True), original,
    )
    stored = await bridge._payload_bytes(payload)
    assert _BODY not in stored and base64.b64encode(_BODY) not in stored
    recovered = await bridge._decode_result(SimpleNamespace(
        result_payload=payload, tool_operation_id="operation",
    ))
    assert isinstance(recovered, ToolReturn)
    assert recovered.return_value == original.return_value
    assert recovered.metadata == original.metadata
    assert recovered.tools is None
    assert recovered.content == original.content
    assert isinstance(recovered.content[0], ReferencedContent)

    calls = {"requests": 0, "tool": 0, "load": 0}
    principals = []

    async def get_file(context: AgentContext[None]) -> ToolReturn:
        calls["tool"] += 1
        principals.append(context.principal)
        return recovered if replay else original

    async def finish(context: AgentContext[None]) -> str:
        return "done"

    async def authorized_load(context: AgentContext[None], content: ReferencedContent) -> bytes:
        calls["load"] += 1
        principals.append(context.principal)
        assert content == _content()
        return _BODY

    async def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        calls["requests"] += 1
        content = [
            item for message in messages if isinstance(message, ModelRequest)
            for part in message.parts if isinstance(part, UserPromptPart)
            and not isinstance(part.content, str) for item in part.content
        ]
        binaries = [item for item in content if isinstance(item, BinaryContent)]
        if calls["requests"] == 1:
            return ModelResponse(parts=[ToolCallPart("get_file", {}, tool_call_id="load")])
        if calls["requests"] == 2:
            assert len(binaries) == 1 and binaries[0].data == _BODY
            assert not any(isinstance(item, ReferencedContent) for item in content)
            return ModelResponse(parts=[ToolCallPart("finish", {}, tool_call_id="finish")])
        assert calls["requests"] == 3 and not binaries
        summaries = [
            item for item in content if isinstance(item, TextContent)
            and (item.metadata or {}).get("referenced_content") is not None
        ]
        assert len(summaries) == 1
        assert json.loads(summaries[0].content)["size"] == len(_BODY)
        descriptor = summaries[0].metadata["referenced_content"]
        assert descriptor["ref"] == _content().ref.to_payload()
        assert descriptor["source"] == {
            "kind": _SOURCE.kind, "id": _SOURCE.id, "revision": _SOURCE.revision,
        }
        return ModelResponse(parts=[TextPart("confirmed")])

    group = RuntimeFeatures("referenced-tool-replay")
    group.tool(get_file, effect_policy="replay_safe")
    group.tool(finish, effect_policy="replay_safe")
    group.capability(ReferencedInputsCapability(authorized_load),
                     id=_SOURCE.id, revision=_SOURCE.revision, scope="agent")
    group.capability(TraceCapability(record_http_bodies=True), scope="runtime")
    group.agent("default", allow_tools=("get_file", "finish"),
                allow_skills=(), allow_subagents=())
    async with Runtime.open(
        "referenced-tool-replay", storage=RuntimeStorage.in_memory(),
        features=model_features(_Models(_UsageFunctionModel(model)), group),
    ) as runtime:
        execution = await runtime.agents.get().start("Inspect the referenced document.")
        result = (await execution.wait(timeout_seconds=10)).result
        assert result.status is ExecutionStatus.SUCCEEDED, result.error_diagnostics
        assert result.output == {"text": "confirmed"}
        assert principals == [runtime.default_principal, runtime.default_principal]
    assert calls == {"requests": 3, "tool": 1, "load": 1}


_ADMITTED = Principal("admitted-user", "default")
_RECOVERY_ACTOR = service_principal("default", "recovery-operator")
_RECOVERY_NAMESPACE = "referenced-pending-recovery"


class _RecoveryAuthorization(TenantAuthorizationPolicy):
    async def authorize(
        self, principal: Principal, action: AuthorizationAction, resource: ResourceRef,
    ) -> None:
        if (principal == _RECOVERY_ACTOR and action is AuthorizationAction.EXECUTION_RECOVER
            and resource.tenant_id == _ADMITTED.tenant_id):
            return
        await super().authorize(principal, action, resource)


def _recovery_features(
    loader: Callable[[AgentContext[None], ReferencedContent], Awaitable[bytes]],
) -> RuntimeFeatures:
    group = RuntimeFeatures(_RECOVERY_NAMESPACE, authorization=_RecoveryAuthorization())
    group.capability(ReferencedInputsCapability(loader), id=_SOURCE.id,
                     revision=_SOURCE.revision, scope="agent")
    group.capability(CheckpointCapability(), scope="runtime")
    group.capability(ExecutionCapability(retain_history=True), scope="runtime")
    group.capability(TraceCapability(record_http_bodies=True), scope="runtime")
    group.agent("default", allow_tools=(), allow_skills=(), allow_subagents=())
    return model_features(_Models(_UsageFunctionModel(_recovery_model)), group)


async def _recovery_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    binaries = [
        item for message in messages if isinstance(message, ModelRequest)
        for part in message.parts if isinstance(part, UserPromptPart)
        and not isinstance(part.content, str) for item in part.content
        if isinstance(item, BinaryContent)
    ]
    assert len(binaries) == 1 and binaries[0].data == _BODY
    return ModelResponse(parts=[TextPart("recovered reference")])


def _pending_reference_process(connection: Connection, path: str) -> None:
    async def run() -> None:
        async def pending_load(context: AgentContext[None], content: ReferencedContent) -> bytes:
            assert context.principal == _ADMITTED and content == _content()
            connection.send(context.execution_id)
            await asyncio.Event().wait()
            raise AssertionError("the interrupted loader must not return bytes")

        async with Runtime.open(
            _RECOVERY_NAMESPACE, storage=RuntimeStorage.filesystem(Path(path)),
            features=_recovery_features(pending_load),
        ) as runtime:
            await runtime.agents.get().start(["Inspect", _content()], principal=_ADMITTED)
            await asyncio.Event().wait()

    asyncio.run(run())


@pytest.mark.asyncio
async def test_pending_reference_reopens_from_checkpoint_with_original_principal(
    tmp_path: Path,
) -> None:
    path = tmp_path / "state"
    context = multiprocessing.get_context("spawn")
    reader, writer = context.Pipe(duplex=False)
    owner = context.Process(target=_pending_reference_process, args=(writer, str(path)))
    owner.start()
    writer.close()
    try:
        assert await asyncio.to_thread(reader.poll, 15), "pending loader was not entered"
        execution_id = reader.recv()
        owner.terminate()
        await asyncio.to_thread(owner.join, 10)
        assert not owner.is_alive()
    finally:
        if owner.is_alive():
            owner.kill()
            await asyncio.to_thread(owner.join, 10)
        reader.close()
        owner.close()

    loaded = []

    async def authorized_load(agent_context: AgentContext[None], content: ReferencedContent) -> bytes:
        assert agent_context.principal == _ADMITTED
        assert content == _content() and content.source == _SOURCE
        loaded.append((agent_context.principal, content))
        return _BODY

    async with Runtime.open(
        _RECOVERY_NAMESPACE, storage=RuntimeStorage.filesystem(path),
        features=_recovery_features(authorized_load), auto_recover=False,
    ) as runtime:
        checkpoint = await runtime._state.recovery.checkpoints.get(
            execution_id, tenant_id=_ADMITTED.tenant_id,
        )
        assert checkpoint is not None
        assert loaded == []
        await runtime.executions.recover(
            execution_id, principal=_RECOVERY_ACTOR, idempotency_key="recover-reference",
        )
        result = (await runtime.executions.wait(
            execution_id, principal=_ADMITTED, timeout_seconds=10,
        )).result
        assert result.status is ExecutionStatus.SUCCEEDED, result.error_diagnostics
        assert result.output == {"text": "recovered reference"}
    async with Runtime.open(
        _RECOVERY_NAMESPACE, storage=RuntimeStorage.filesystem(path),
        features=_recovery_features(authorized_load), auto_recover=False,
    ) as reopened:
        history = await reopened.executions.history(
            execution_id, principal=_ADMITTED, include_content=True,
        )
        history_bytes = json.dumps([item.content for item in history.items]).encode()
        assert _content().ref.digest.encode() in history_bytes
        assert _BODY not in history_bytes and base64.b64encode(_BODY) not in history_bytes
    assert loaded == [(_ADMITTED, _content())]
    stored = b"\n".join(file.read_bytes() for file in path.rglob("*") if file.is_file())
    assert _BODY not in stored and base64.b64encode(_BODY) not in stored
