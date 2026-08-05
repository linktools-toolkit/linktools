#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Pure conversion between ACP schemas and Linktools domain values."""

import hashlib
import json
from typing import Any, Mapping

from ..execution.domain import RunStatus
from ..execution.live_events import (
    AssistantTextDelta,
    AssistantThoughtDelta,
    ExecutionEvent,
    PlanUpdated,
    ToolCallCompleted,
    ToolCallFailed,
    ToolCallProgress,
    ToolCallStarted,
    UsageUpdated,
)
from ..prompt import (
    AudioPromptPart,
    EmbeddedResourcePromptPart,
    ImagePromptPart,
    PromptValidationError,
    ResourceLinkPromptPart,
    TextPromptPart,
    UserPrompt,
    decode_base64,
)
from ..json import canonical_json_bytes
from .protocol import request_error


class AcpCodec:
    def __init__(
        self, *, image: bool = False, audio: bool = False, embedded: bool = False
    ) -> None:
        self.image = image
        self.audio = audio
        self.embedded = embedded

    def decode_prompt(self, blocks: "list[Any]") -> UserPrompt:
        if not blocks:
            raise request_error("empty_prompt")
        parts: "list[Any]" = []
        for block in blocks:
            kind = getattr(block, "type", None)
            if kind == "text":
                parts.append(TextPromptPart(block.text))
            elif kind == "image":
                if not self.image:
                    raise request_error("unsupported_content_type")
                parts.append(ImagePromptPart(self._decode(block.data), block.mime_type))
            elif kind == "audio":
                if not self.audio:
                    raise request_error("unsupported_content_type")
                parts.append(AudioPromptPart(self._decode(block.data), block.mime_type))
            elif kind == "resource_link":
                parts.append(ResourceLinkPromptPart(block.uri, block.name, block.mime_type))
            elif kind == "resource":
                if not self.embedded:
                    raise request_error("unsupported_content_type")
                resource = block.resource
                if getattr(resource, "text", None) is not None:
                    parts.append(EmbeddedResourcePromptPart(resource.mime_type, text=resource.text))
                else:
                    parts.append(
                        EmbeddedResourcePromptPart(
                            resource.mime_type, data=self._decode(resource.blob)
                        )
                    )
            else:
                raise request_error("unsupported_content_type")
        try:
            return UserPrompt(tuple(parts))
        except ValueError as exc:
            raise request_error("invalid_prompt") from exc

    def decode_mcp_server(self, descriptor: Any) -> Any:
        from ..agent.mcp.spec import MCPServerSpec

        kind = self._mcp_transport(descriptor)
        name = getattr(descriptor, "name", "")
        if not name:
            raise request_error("invalid_mcp_descriptor")
        if kind == "stdio":
            env = {item.name: item.value for item in getattr(descriptor, "env", ())}
            return MCPServerSpec(
                id=name,
                name=name,
                transport="stdio",
                command=(descriptor.command, *tuple(getattr(descriptor, "args", ()))),
                env=env,
            )
        if kind in {"http", "sse"}:
            headers = {item.name: item.value for item in getattr(descriptor, "headers", ())}
            return MCPServerSpec(
                id=name,
                name=name,
                transport=kind,
                url=descriptor.url,
                headers=headers,
            )
        raise request_error("unsupported_mcp_transport", details={"transport": kind})

    def decode_mcp_servers(self, descriptors: "list[Any] | None") -> "tuple[Any, ...]":
        return tuple(self.decode_mcp_server(item) for item in descriptors or ())

    @staticmethod
    def mcp_server_fingerprint(server: Any) -> str:
        payload = {
            "id": server.id,
            "transport": server.transport,
            "command": list(server.command) if server.command is not None else None,
            "url": server.url,
            "cwd": server.cwd,
            "timeout_seconds": server.timeout_seconds,
            "tool_prefix": server.tool_prefix,
            "enabled_tools": list(server.enabled_tools) if server.enabled_tools is not None else None,
            "disabled_tools": list(server.disabled_tools),
        }
        return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()

    def encode_event(self, event: ExecutionEvent) -> "Any | None":
        import acp.schema as schema

        if isinstance(event, AssistantTextDelta):
            return schema.AgentMessageChunk(
                content=schema.TextContentBlock(type="text", text=event.text),
                messageId=event.execution_id,
                sessionUpdate="agent_message_chunk",
            )
        if isinstance(event, AssistantThoughtDelta):
            return schema.AgentThoughtChunk(
                content=schema.TextContentBlock(type="text", text=event.text),
                messageId=event.execution_id,
                sessionUpdate="agent_thought_chunk",
            )
        if isinstance(event, ToolCallStarted):
            return schema.ToolCallStart(
                toolCallId=event.tool_call_id,
                title=event.tool_name,
                status="pending",
                rawInput=event.arguments,
                sessionUpdate="tool_call",
            )
        if isinstance(event, (ToolCallProgress, ToolCallCompleted, ToolCallFailed)):
            return schema.ToolCallProgress(
                toolCallId=event.tool_call_id,
                title=event.tool_name,
                status=event.status,
                rawInput=event.arguments,
                rawOutput=(
                    event.error
                    if isinstance(event, ToolCallFailed)
                    else getattr(event, "result", None)
                ),
                sessionUpdate="tool_call_update",
            )
        if isinstance(event, PlanUpdated):
            return schema.AgentPlanUpdate(entries=list(event.entries), sessionUpdate="plan")
        if isinstance(event, UsageUpdated):
            used = getattr(event, "context_tokens_used", None)
            size = getattr(event, "context_window_size", None)
            if not isinstance(used, int) or isinstance(used, bool) or not isinstance(size, int) or isinstance(size, bool) or not 0 <= used <= size:
                return None
            return schema.UsageUpdate(used=used, size=size, sessionUpdate="usage_update")
        return None

    def encode_history(self, session_id: str, views: Any) -> "tuple[Any, ...]":
        updates: "list[Any]" = []
        for turn_index, view in enumerate(views):
            status = getattr(getattr(view, "status", None), "value", getattr(view, "status", None))
            capture = getattr(getattr(view, "capture_state", None), "value", getattr(view, "capture_state", None))
            if status != "completed" or capture != "complete":
                raise request_error("incomplete_history", session_id=session_id, details={"turnIndex": turn_index, "captureState": capture})
            updates.extend(self._encode_turn(session_id, turn_index, view))
        return tuple(updates)

    def encode_session(self, record: Any) -> Any:
        import acp.schema as schema

        return schema.SessionInfo(
            sessionId=record.id,
            cwd=record.workspace.cwd,
            additionalDirectories=list(record.workspace.additional_directories),
            title=record.title,
            updatedAt=record.updated_at.isoformat(),
        )

    @staticmethod
    def encode_stop_reason(status: RunStatus) -> str:
        if status is RunStatus.COMPLETED:
            return "end_turn"
        if status is RunStatus.CANCELLED:
            return "cancelled"
        raise RuntimeError(f"execution is not a successful terminal state: {status.value}")

    @staticmethod
    def mcp_descriptor_fingerprint(descriptor: object) -> str:
        if hasattr(descriptor, "model_dump"):
            raw = descriptor.model_dump(mode="json", by_alias=True, exclude_none=True)
        elif isinstance(descriptor, Mapping):
            raw = dict(descriptor)
        else:
            raw = {"value": str(descriptor)}
        encoded = json.dumps(
            AcpCodec._without_secrets(raw), sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode()
        return hashlib.sha256(encoded).hexdigest()

    @staticmethod
    def _decode(value: str) -> bytes:
        try:
            return decode_base64(value)
        except PromptValidationError as exc:
            raise request_error("invalid_prompt") from exc

    def _encode_turn(self, session_id: str, turn_index: int, view: Any) -> "list[Any]":
        import acp.schema as schema

        updates: "list[Any]" = []
        tool_ids: "set[str]" = set()
        messages = getattr(view, "messages", None)
        if messages is None:
            messages = getattr(view, "delta_messages", ())
        for message_index, message in enumerate(messages):
            if not isinstance(message, Mapping):
                self._unsupported(session_id, turn_index, message_index, type(message).__name__)
            kind = message.get("kind")
            parts = message.get("parts")
            if kind not in {"request", "response"} or not isinstance(parts, (list, tuple)):
                self._unsupported(session_id, turn_index, message_index, str(kind))
            message_id = f"{view.run_id}:{message_index}"
            for part_index, part in enumerate(parts):
                if not isinstance(part, Mapping):
                    self._unsupported(session_id, turn_index, part_index, type(part).__name__)
                part_type = part.get("type")
                if kind == "request" and part_type == "user_prompt":
                    for content in self._user_content(part, session_id, turn_index, part_index):
                        updates.append(schema.UserMessageChunk(content=content, messageId=message_id, sessionUpdate="user_message_chunk"))
                elif kind == "response" and part_type == "text":
                    updates.append(schema.AgentMessageChunk(content=schema.TextContentBlock(type="text", text=self._text(part, session_id, turn_index, part_index)), messageId=message_id, sessionUpdate="agent_message_chunk"))
                elif kind == "response" and part_type in {"thinking", "reasoning"}:
                    updates.append(schema.AgentThoughtChunk(content=schema.TextContentBlock(type="text", text=self._text(part, session_id, turn_index, part_index)), messageId=message_id, sessionUpdate="agent_thought_chunk"))
                elif kind == "response" and part_type == "tool_call":
                    call_id = self._call_id(part, session_id, turn_index, part_index)
                    tool_ids.add(call_id)
                    updates.append(schema.ToolCallStart(toolCallId=call_id, title=str(part.get("tool_name", "tool")), status="pending", rawInput=part.get("arguments"), sessionUpdate="tool_call"))
                elif part_type == "tool_result":
                    call_id = self._call_id(part, session_id, turn_index, part_index)
                    if call_id not in tool_ids:
                        self._unsupported(session_id, turn_index, part_index, "tool_result_without_call")
                    status = "completed" if str(part.get("status", "success")) in {"success", "completed", "ok"} else "failed"
                    updates.append(schema.ToolCallProgress(toolCallId=call_id, title=str(part.get("tool_name", "tool")), status=status, rawOutput=part.get("result") if status == "completed" else {"error": part.get("result")}, sessionUpdate="tool_call_update"))
                elif part_type not in {"system_prompt", "metadata"}:
                    self._unsupported(session_id, turn_index, part_index, str(part_type))
        return updates

    @staticmethod
    def _text(part: Mapping[str, Any], session_id: str, turn_index: int, part_index: int) -> str:
        value = part.get("content")
        if not isinstance(value, str):
            AcpCodec._unsupported(session_id, turn_index, part_index, str(part.get("type")))
        return value

    @staticmethod
    def _call_id(part: Mapping[str, Any], session_id: str, turn_index: int, part_index: int) -> str:
        value = part.get("call_id")
        if not isinstance(value, str) or not value:
            AcpCodec._unsupported(session_id, turn_index, part_index, str(part.get("type")))
        return value

    @staticmethod
    def _user_content(part: Mapping[str, Any], session_id: str, turn_index: int, part_index: int) -> "tuple[Any, ...]":
        import acp.schema as schema

        content = part.get("content")
        if isinstance(content, str):
            return (schema.TextContentBlock(type="text", text=content),)
        if not isinstance(content, list):
            AcpCodec._unsupported(session_id, turn_index, part_index, "user_prompt")
        mapped = []
        for item in content:
            kind = item.get("kind") or item.get("part_kind") if isinstance(item, Mapping) else None
            if kind == "text" and isinstance(item.get("content"), str):
                mapped.append(schema.TextContentBlock(type="text", text=item["content"]))
            elif kind in {"image-url", "audio-url", "document-url"} and isinstance(item.get("url"), str):
                mapped.append(schema.ResourceContentBlock(type="resource_link", name=str(kind), uri=item["url"]))
            else:
                AcpCodec._unsupported(session_id, turn_index, part_index, str(kind))
        return tuple(mapped)

    @staticmethod
    def _unsupported(session_id: str, turn_index: int, part_index: int, part_type: str) -> None:
        raise request_error("unsupported_history_part", session_id=session_id, details={"turnIndex": turn_index, "partIndex": part_index, "partType": part_type})

    @staticmethod
    def _mcp_transport(descriptor: Any) -> str:
        kind = getattr(descriptor, "type", None)
        if kind is not None:
            return str(kind)
        return {"McpServerStdio": "stdio", "McpServerHttp": "http", "McpServerSse": "sse", "McpServerAcp": "acp"}.get(type(descriptor).__name__, "unknown")

    @staticmethod
    def _without_secrets(value: Any) -> Any:
        names = {"authorization", "api_key", "apikey", "env", "environment", "headers", "password", "secret", "token"}
        if isinstance(value, Mapping):
            return {key: AcpCodec._without_secrets(item) for key, item in value.items() if str(key).lower().replace("-", "_") not in names}
        if isinstance(value, (list, tuple)):
            return [AcpCodec._without_secrets(item) for item in value]
        return value


__all__ = ["AcpCodec"]
