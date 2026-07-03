#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Session history snapshots: context.json read/write and per-call prompt sidecars."""

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic_ai.messages import (
    ModelMessage,
    ModelMessagesTypeAdapter,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    SystemPromptPart,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)

from ..core.model_runtime import RuntimeModelConfig
from ..support.utils import safe_filename

logger = logging.getLogger("linktools.ai.session.history")

# Mirror the legacy context.json trimming budget.
_CONTEXT_MAX_MESSAGES = 80
_CONTEXT_MAX_CHARS = 160000


@dataclass(frozen=True, slots=True)
class SessionContextSnapshot:
    session_id: str
    messages: "list[ModelMessage]"
    model: RuntimeModelConfig
    token_usage: "dict[str, Any]"
    llm_call: "dict[str, Any]"


@dataclass(frozen=True, slots=True)
class CallPromptSnapshot:
    call_id: str
    messages: "list[dict[str, Any]]"


def load_message_history(session_dir: Path) -> "list[ModelMessage]":
    """Restore pydantic-ai message history from `context.json`, or [] if absent."""
    path = session_dir / "context.json"
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        raw = payload.get("messages")
        if not raw:
            return []
        return list(ModelMessagesTypeAdapter.validate_python(raw))
    except Exception:
        logger.warning("failed to load message history from %s", path, exc_info=True)
        return []


def _trim_messages(messages: "list[ModelMessage]") -> "list[ModelMessage]":
    """Keep the history within the legacy 80-message / 160K-char budget.

    pydantic-ai groups one request and its response into separate ModelMessage
    objects, so trimming whole messages from the front preserves request/response
    pairing without orphaning tool returns.
    """
    trimmed = messages[-_CONTEXT_MAX_MESSAGES:] if len(messages) > _CONTEXT_MAX_MESSAGES else list(messages)
    while len(trimmed) > 1:
        dumped = ModelMessagesTypeAdapter.dump_python(trimmed, mode="json")
        if len(json.dumps(dumped, ensure_ascii=False, default=str)) <= _CONTEXT_MAX_CHARS:
            break
        trimmed = trimmed[1:]
    return trimmed


def write_session_context(session_dir: Path, snapshot: SessionContextSnapshot) -> None:
    """Persist message history + per-call metadata to `context.json`.

    Preserves the legacy schema (trace_id/session_id/llm_calls/messages/...) so
    downstream readers and `_next_llm_seq` keep working.
    """
    path = session_dir / "context.json"
    existing: "dict[str, Any]" = {}
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            existing = {}

    trimmed = _trim_messages(snapshot.messages)
    stored = ModelMessagesTypeAdapter.dump_python(trimmed, mode="json")
    llm_calls = list(existing.get("llm_calls") or [])
    llm_calls.append(snapshot.llm_call)

    payload = {
        "session_id": snapshot.session_id,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "current_model": {"model_type": snapshot.model.model_type, "model": snapshot.model.model},
        "llm_calls": llm_calls,
        "token_usage": snapshot.token_usage,
        "message_count": len(stored),
        "messages": stored,
    }
    session_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


# ---------------------------------------------------------------------------
# Per-call prompt sidecar (session_dir/calls/<call_id>.json), read by the
# console UI's "prompt" inspector via /v1/traces/{trace_id}/files/{path}.
# ---------------------------------------------------------------------------

def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(_content_to_text(item) for item in content)
    return str(content)


def _model_message_to_dicts(message: ModelMessage) -> "list[dict[str, Any]]":
    """Convert a pydantic-ai `ModelMessage` into legacy OpenAI-style chat dicts."""
    out: "list[dict[str, Any]]" = []
    if isinstance(message, ModelRequest):
        for part in message.parts:
            if isinstance(part, SystemPromptPart):
                out.append({"role": "system", "content": _content_to_text(part.content)})
            elif isinstance(part, UserPromptPart):
                out.append({"role": "user", "content": _content_to_text(part.content)})
            elif isinstance(part, ToolReturnPart):
                out.append({
                    "role": "tool",
                    "tool_call_id": part.tool_call_id,
                    "content": part.model_response_str(),
                })
            elif isinstance(part, RetryPromptPart):
                content = part.model_response()
                if part.tool_call_id:
                    out.append({"role": "tool", "tool_call_id": part.tool_call_id, "content": content})
                else:
                    out.append({"role": "user", "content": content})
    elif isinstance(message, ModelResponse):
        text_parts: "list[str]" = []
        reasoning_parts: "list[str]" = []
        tool_calls: "list[dict[str, Any]]" = []
        for part in message.parts:
            if isinstance(part, TextPart):
                text_parts.append(part.content)
            elif isinstance(part, ThinkingPart):
                reasoning_parts.append(part.content)
            elif isinstance(part, ToolCallPart):
                args = part.args
                if not isinstance(args, str):
                    args = json.dumps(args, ensure_ascii=False, default=str)
                tool_calls.append({
                    "id": part.tool_call_id,
                    "type": "function",
                    "function": {"name": part.tool_name, "arguments": args},
                })
        entry: "dict[str, Any]" = {"role": "assistant", "content": "\n".join(text_parts)}
        if reasoning_parts:
            entry["reasoning_content"] = "\n".join(reasoning_parts)
        if tool_calls:
            entry["tool_calls"] = tool_calls
        out.append(entry)
    return out


def new_call_messages(
    history: "list[ModelMessage]", all_messages: "list[ModelMessage]", *, system_prompt: str = "",
) -> "list[dict[str, Any]]":
    """Render the messages added by this call (plus a leading system message) for the prompt sidecar.

    `instructions=`-based capabilities (vs the legacy `system_prompt=` kwarg) never produce a
    `SystemPromptPart` in message history, so the message-scan fallback below can no longer find
    one — `system_prompt` (the raw string from `LlmCallContext.system_prompt`) is the real source now.
    """
    new_messages = all_messages[len(history):]
    out: "list[dict[str, Any]]" = []
    for msg in new_messages:
        out.extend(_model_message_to_dicts(msg))
    if not any(d.get("role") == "system" for d in out):
        for msg in all_messages:
            system = next((d for d in _model_message_to_dicts(msg) if d.get("role") == "system"), None)
            if system:
                out.insert(0, system)
                break
        else:
            if system_prompt:
                out.insert(0, {"role": "system", "content": system_prompt})
    return out


def write_call_prompt(session_dir: Path, snapshot: CallPromptSnapshot) -> None:
    """Persist the per-call prompt sidecar consumed by the console's prompt inspector."""
    if not snapshot.call_id or not snapshot.messages:
        return
    path = session_dir / "calls" / f"{safe_filename(snapshot.call_id, 'call')}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"call_id": snapshot.call_id, "messages": snapshot.messages}, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
