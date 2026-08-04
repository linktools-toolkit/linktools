#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Preflight and map persisted Runtime turns to ACP history updates."""

from typing import Any, Mapping

from .errors import request_error


class AcpHistoryMapper:
    """Own the only persisted-history-to-ACP mapping implementation."""

    def preflight(self, session_id: str, views: Any) -> "tuple[Any, ...]":
        updates: "list[Any]" = []
        for turn_index, view in enumerate(views):
            capture_state = getattr(getattr(view, "capture_state", None), "value", getattr(view, "capture_state", None))
            status = getattr(view, "status", None)
            if getattr(status, "value", status) != "completed" or capture_state != "complete":
                raise request_error(
                    "incomplete_history",
                    session_id=session_id,
                    details={
                        "turnIndex": turn_index,
                        "captureState": capture_state,
                    },
                )
            updates.extend(self._map_turn(session_id, turn_index, view))
        return tuple(updates)

    def _map_turn(self, session_id: str, turn_index: int, view: Any) -> "list[Any]":
        import acp.schema as schema

        updates: "list[Any]" = []
        tool_ids: "set[str]" = set()
        messages = getattr(view, "messages", ())
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
                    for content in self._user_content(
                        part,
                        session_id,
                        turn_index,
                        part_index,
                    ):
                        updates.append(
                            schema.UserMessageChunk(
                                content=content,
                                messageId=message_id,
                                sessionUpdate="user_message_chunk",
                            )
                        )
                elif kind == "response" and part_type == "text":
                    updates.append(
                        schema.AgentMessageChunk(
                            content=schema.TextContentBlock(type="text", text=self._text_content(part, session_id, turn_index, part_index)),
                            messageId=message_id,
                            sessionUpdate="agent_message_chunk",
                        )
                    )
                elif kind == "response" and part_type in {"thinking", "reasoning"}:
                    updates.append(
                        schema.AgentThoughtChunk(
                            content=schema.TextContentBlock(type="text", text=self._text_content(part, session_id, turn_index, part_index)),
                            messageId=message_id,
                            sessionUpdate="agent_thought_chunk",
                        )
                    )
                elif kind == "response" and part_type == "tool_call":
                    call_id = self._call_id(part, session_id, turn_index, part_index)
                    tool_ids.add(call_id)
                    updates.append(
                        schema.ToolCallStart(
                            toolCallId=call_id,
                            title=str(part.get("tool_name", "tool")),
                            status="pending",
                            rawInput=part.get("arguments"),
                            sessionUpdate="tool_call",
                        )
                    )
                elif part_type == "tool_result":
                    call_id = self._call_id(part, session_id, turn_index, part_index)
                    if call_id not in tool_ids:
                        self._unsupported(session_id, turn_index, part_index, "tool_result_without_call")
                    outcome = str(part.get("status", "success"))
                    status = "completed" if outcome in {"success", "completed", "ok"} else "failed"
                    updates.append(
                        schema.ToolCallProgress(
                            toolCallId=call_id,
                            title=str(part.get("tool_name", "tool")),
                            status=status,
                            rawOutput=part.get("result") if status == "completed" else {"error": part.get("result")},
                            sessionUpdate="tool_call_update",
                        )
                    )
                elif part_type in {"system_prompt", "metadata"}:
                    continue
                else:
                    self._unsupported(session_id, turn_index, part_index, str(part_type))
            usage = message.get("usage")
            if kind == "response" and isinstance(usage, Mapping):
                used = usage.get("context_tokens_used")
                size = usage.get("context_window_size")
                if (
                    isinstance(used, int)
                    and not isinstance(used, bool)
                    and isinstance(size, int)
                    and not isinstance(size, bool)
                    and 0 <= used <= size
                ):
                    updates.append(
                        schema.UsageUpdate(
                            used=used,
                            size=size,
                            sessionUpdate="usage_update",
                        )
                    )
        return updates

    @staticmethod
    def _text_content(part: Mapping[str, Any], session_id: str, turn_index: int, part_index: int) -> str:
        content = part.get("content")
        if not isinstance(content, str):
            AcpHistoryMapper._unsupported(session_id, turn_index, part_index, str(part.get("type")))
        return content

    @staticmethod
    def _user_content(
        part: Mapping[str, Any],
        session_id: str,
        turn_index: int,
        part_index: int,
    ) -> "tuple[Any, ...]":
        import acp.schema as schema

        content = part.get("content")
        if isinstance(content, str):
            return (schema.TextContentBlock(type="text", text=content),)
        if not isinstance(content, list):
            AcpHistoryMapper._unsupported(
                session_id,
                turn_index,
                part_index,
                str(part.get("type")),
            )
        mapped = []
        for item in content:
            if not isinstance(item, Mapping):
                AcpHistoryMapper._unsupported(
                    session_id,
                    turn_index,
                    part_index,
                    type(item).__name__,
                )
            kind = item.get("kind") or item.get("part_kind")
            if kind == "text":
                text = item.get("content")
                if not isinstance(text, str):
                    AcpHistoryMapper._unsupported(session_id, turn_index, part_index, "text")
                mapped.append(schema.TextContentBlock(type="text", text=text))
            elif kind in {"image-url", "audio-url", "document-url"}:
                uri = item.get("url")
                if not isinstance(uri, str) or not uri:
                    AcpHistoryMapper._unsupported(session_id, turn_index, part_index, str(kind))
                mapped.append(
                    schema.ResourceContentBlock(
                        type="resource_link",
                        name=str(kind),
                        uri=uri,
                    )
                )
            else:
                AcpHistoryMapper._unsupported(session_id, turn_index, part_index, str(kind))
        return tuple(mapped)

    @staticmethod
    def _call_id(part: Mapping[str, Any], session_id: str, turn_index: int, part_index: int) -> str:
        call_id = part.get("call_id")
        if not isinstance(call_id, str) or not call_id:
            AcpHistoryMapper._unsupported(session_id, turn_index, part_index, str(part.get("type")))
        return call_id

    @staticmethod
    def _unsupported(session_id: str, turn_index: int, part_index: int, part_type: str) -> None:
        raise request_error(
            "unsupported_history_part",
            session_id=session_id,
            details={
                "turnIndex": turn_index,
                "partIndex": part_index,
                "partType": part_type,
            },
        )


__all__ = ["AcpHistoryMapper"]
