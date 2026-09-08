#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Portable attachment delegation stays explicit and child-owned."""

import json
from collections.abc import AsyncIterator, Mapping
from pathlib import Path

import pytest
from pydantic_ai.messages import (
    BinaryContent,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, FunctionModel

from linktools.ai.capability import CapabilityGroup
from linktools.ai.core import ExecutionStatus, JsonValue
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime import Runtime, RuntimeState
from linktools.ai.runtime.state._attachment_repository import AttachmentRepository
from linktools.ai.workspace import Workspace


def _contains_binary(messages: list[ModelMessage], body: bytes) -> bool:
    for message in messages:
        if not isinstance(message, ModelRequest):
            continue
        for part in message.parts:
            if not isinstance(part, UserPromptPart) or isinstance(part.content, str):
                continue
            if any(
                isinstance(item, BinaryContent) and item.data == body
                for item in part.content
            ):
                return True
    return False


def _has_delegate_result(messages: list[ModelMessage]) -> bool:
    return any(
        isinstance(message, ModelRequest)
        and any(
            isinstance(part, ToolReturnPart) and part.tool_name == "delegate_task"
            for part in message.parts
        )
        for message in messages
    )


class _RouteBinding:
    provider = "test"

    def __init__(self, models: "_SubagentModels", route_id: str) -> None:
        self._models = models
        self.route_id = route_id
        self.model_identity = f"test:{route_id}"
        self.fingerprint = ("a" if route_id == "parent" else "b") * 64
        self.semantic_payload: dict[str, JsonValue] = {
            "provider": "test",
            "model": route_id,
        }

    def materialize(self) -> FunctionModel:
        models = self._models
        route_id = self.route_id

        async def request(
            messages: list[ModelMessage],
            info: AgentInfo,
        ) -> ModelResponse:
            del info
            if route_id == "parent":
                models.parent_seen.append(list(messages))
            else:
                models.child_seen.append(list(messages))
            return ModelResponse(parts=[TextPart("ok")])

        async def stream(
            messages: list[ModelMessage],
            info: AgentInfo,
        ) -> AsyncIterator[str | dict[int, DeltaToolCall]]:
            if route_id == "child":
                models.child_seen.append(list(messages))
                if not _contains_binary(messages, models.body):
                    raise AssertionError("child model did not receive delegated attachment body")
                models.child_binary_seen = True
                yield "child done"
                return

            models.parent_seen.append(list(messages))
            if _contains_binary(messages, models.body):
                models.parent_binary_seen = True
            if _has_delegate_result(messages):
                yield "parent done"
                return
            if "delegate_task" not in {tool.name for tool in info.function_tools}:
                raise AssertionError("delegate_task is not available")
            yield {
                0: DeltaToolCall(
                    name="delegate_task",
                    json_args=json.dumps(
                        {
                            "subagent_id": "child",
                            "task": "inspect the evidence",
                            "attachments": ["evidence.txt"],
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    tool_call_id="delegate-with-attachment-1",
                )
            }

        return FunctionModel(function=request, stream_function=stream)


class _SubagentModels:
    def __init__(self, body: bytes) -> None:
        self.body = body
        self.parent_seen: list[list[ModelMessage]] = []
        self.child_seen: list[list[ModelMessage]] = []
        self.parent_binary_seen = False
        self.child_binary_seen = False

    def snapshot(self) -> "_SubagentModels":
        return self

    def resolve(self, route_id: str) -> _RouteBinding:
        if route_id not in {"parent", "child"}:
            raise AIError(ErrorCode.MODEL_CONNECTION_NOT_FOUND)
        return _RouteBinding(self, route_id)

    def restore(
        self,
        payload: Mapping[str, JsonValue],
        *,
        route_id: str | None = None,
    ) -> _RouteBinding:
        resolved = route_id
        if resolved is None:
            candidate = payload.get("model")
            resolved = candidate if isinstance(candidate, str) else None
        if resolved not in {"parent", "child"}:
            raise AIError(ErrorCode.MODEL_CONNECTION_NOT_FOUND)
        expected = {"provider": "test", "model": resolved}
        if dict(payload) != expected:
            raise AIError(ErrorCode.MODEL_CONNECTION_NOT_FOUND)
        return _RouteBinding(self, resolved)


def _group() -> CapabilityGroup[object]:
    group: CapabilityGroup[object] = CapabilityGroup("subagent-attachment-test")
    group.agent(
        "parent",
        model="parent",
        allow_tools=(),
        allow_skills=(),
        allow_subagents=("child",),
    )
    group.agent(
        "child",
        model="child",
        allow_tools=(),
        allow_skills=(),
        allow_subagents=(),
    )
    return group


@pytest.mark.asyncio
async def test_delegate_attachment_freezes_parent_source_and_activates_only_child(
    tmp_path: Path,
) -> None:
    body = b"delegated evidence body"
    (tmp_path / "evidence.txt").write_bytes(body)
    workspace = Workspace.load(tmp_path, workspace_id="portable-subagent")
    state = RuntimeState.in_memory()
    models = _SubagentModels(body)

    async with Runtime.open(
        workspace,
        models=models,  # type: ignore[arg-type]
        state=state,
        capabilities=(_group(),),
    ) as runtime:
        result = await runtime.agent("parent").run(
            "delegate the evidence review",
            idempotency_key="parent-subagent-attachment-key",
            timeout_seconds=10,
        )

        assert result.status is ExecutionStatus.SUCCEEDED
        assert models.child_binary_seen
        assert not models.parent_binary_seen

        parent = await state.execution.executions.get(
            result.execution_id,
            tenant_id="default",
        )
        assert parent is not None
        assert parent.attachment_manifest == ()
        assert parent.input_digest is None

        children = await state.execution.executions.list_children(
            result.execution_id,
            tenant_id="default",
        )
        assert len(children) == 1
        child = children[0]
        assert child.status is ExecutionStatus.SUCCEEDED
        assert child.parent_execution_id == result.execution_id
        assert child.root_execution_id == result.execution_id
        assert len(child.attachment_manifest) == 1
        assert child.input_digest is not None
        assert child.path_origin is not None

        source = await AttachmentRepository(
            state.execution.executions.state_store,
            namespace=workspace.workspace_id,
            tenant_id="default",
        ).get_source(
            result.execution_id,
            "evidence.txt",
            tenant_id="default",
        )
        assert source is not None
        assert source.entry.path.startswith("virtual:attachments/e.")
        assert child.attachment_manifest[0].path.startswith("virtual:attachments/p.")
        assert child.attachment_manifest[0].content == source.entry.content
