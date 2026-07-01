#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Model configuration, pydantic-ai runtime factory, and session persistence.

The hand-rolled OpenAI-compatible ReAct loop that previously lived here has been
replaced by pydantic-ai (see `base.py`). This module now owns:

- the shared error types raised/caught across the pipeline
  (`ModelClientUnavailable`, `ModelOutputError`, `ModelTurnLimitExceeded`);
- the `RuntimeModelConfig` dataclass and `load_runtime_model_config`, which resolve
  a named model from `config.{env}.yaml` `models.*` (with `env:` interpolation);
- `build_model`/`ModelBundle`, the pydantic-ai model factory built on top of
  `RuntimeModelConfig`;
- `build_mcp_toolset`, mapping `MCPServerSpec` onto pydantic-ai `MCPToolset`s;
- session history persistence to `session_dir/context.json` and the per-call
  prompt sidecar under `session_dir/calls/<call_id>.json`.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic_ai.mcp import MCPToolset
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
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.profiles.openai import OpenAIModelProfile
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.settings import ModelSettings
from pydantic_ai.usage import UsageLimits

import logging

from .environment import AgentEnvironment
from ..support.config import load_yaml_file
from ..support.utils import resolve_ref as _resolve_ref, safe_filename
from .mcp_client import _resolve_arg
from .registry import MCPServerSpec

logger = logging.getLogger("linktools.ai.core.model_runtime")

# Mirror the legacy context.json trimming budget.
_CONTEXT_MAX_MESSAGES = 80
_CONTEXT_MAX_CHARS = 160000


class ModelClientUnavailable(RuntimeError):
    def __init__(self, message: str, diagnostics: dict[str, Any] | None = None):
        super().__init__(message)
        self.diagnostics = diagnostics or {}


class ModelOutputError(RuntimeError):
    """Model response could not be parsed/validated as the expected structured output."""

    def __init__(self, message: str, diagnostics: dict[str, Any] | None = None):
        super().__init__(message)
        self.diagnostics = diagnostics or {}


class ModelTurnLimitExceeded(RuntimeError):
    """The agent exhausted its per-call turn/request budget (UsageLimits.request_limit)
    without producing a result, typically from looping on tool calls."""

    def __init__(self, message: str, diagnostics: dict[str, Any] | None = None):
        super().__init__(message)
        self.diagnostics = diagnostics or {}


@dataclass(slots=True)
class RuntimeModelConfig:
    model_type: str
    protocol: str
    model: str | None
    base_url: str | None
    api_key: str | None
    auth_token: str | None
    timeout_seconds: int
    raw: dict[str, Any]

    @property
    def token(self) -> str | None:
        return self.auth_token or self.api_key


def load_runtime_model_config(env: AgentEnvironment, model_type: str) -> RuntimeModelConfig:
    """Load and resolve a named model config from config.{env}.yaml `models.*`."""
    path = env.config_root / f"config.{env.env}.yaml"
    if not path.exists():
        raise ModelClientUnavailable(f"model config not found: {path}")
    config = load_yaml_file(path, resolve_env=True)
    raw = (config.get("models") or {}).get(model_type)
    if not isinstance(raw, dict):
        raise ModelClientUnavailable(f"model config missing: {model_type}")

    def _required(key: str) -> str:
        value = raw.get(key)
        if not value:
            raise ModelClientUnavailable(f"model config '{model_type}' missing required field: {key}")
        return str(value)

    def _optional(key: str) -> str | None:
        value = raw.get(key)
        return str(value) if value else None

    return RuntimeModelConfig(
        model_type=model_type,
        protocol=_required("protocol"),
        model=_required("model"),
        base_url=_required("base_url"),
        api_key=_optional("api_key"),
        auth_token=_optional("auth_token"),
        timeout_seconds=int(raw.get("timeout_seconds", 300)),
        raw=raw,
    )


# ---------------------------------------------------------------------------
# pydantic-ai model factory
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class ModelBundle:
    """A configured model plus the per-call execution limits derived from config."""

    config: RuntimeModelConfig
    model: OpenAIChatModel
    settings: ModelSettings
    usage_limits: UsageLimits


@dataclass(frozen=True, slots=True)
class SessionContextSnapshot:
    trace_id: str
    session_id: str
    messages: list[ModelMessage]
    model: RuntimeModelConfig
    token_usage: dict[str, Any]
    llm_call: dict[str, Any]


@dataclass(frozen=True, slots=True)
class CallPromptSnapshot:
    call_id: str
    messages: list[dict[str, Any]]


def _normalize_base_url(config: RuntimeModelConfig) -> str:
    base = (config.base_url or "").rstrip("/").removesuffix("/v1")
    if not base:
        raise ModelClientUnavailable(
            f"{config.model_type}: openai protocol requires base_url"
        )
    return f"{base}/v1"


def build_model(env: AgentEnvironment, model_type: str) -> ModelBundle:
    """Build an `OpenAIChatModel` (+ settings/limits) for the named model type.

    Reuses `load_runtime_model_config` so the OpenAI-compatible base_url/token are
    resolved identically to the legacy client.
    """
    config = load_runtime_model_config(env, model_type)
    if config.protocol != "openai":
        raise ModelClientUnavailable(
            f"{config.model_type}: unsupported protocol '{config.protocol}' (use 'openai')"
        )
    provider = OpenAIProvider(base_url=_normalize_base_url(config), api_key=config.token)
    # The gateway routes to various OpenAI-compatible models, including reasoning/
    # "thinking mode" models (e.g. deepseek-v4-flash) that reject `tool_choice:
    # "required"` with HTTP 400. Disabling this lets pydantic-ai fall back to
    # `tool_choice: "auto"` for structured output, which all backends accept.
    profile = OpenAIModelProfile(openai_supports_tool_choice_required=False)
    model = OpenAIChatModel(config.model or "", provider=provider, profile=profile)

    raw = config.raw
    settings = ModelSettings(
        max_tokens=int(raw.get("max_output_tokens", 4096)),
        timeout=float(config.timeout_seconds),
        parallel_tool_calls=True,
    )
    # max_turns historically bounded the number of model requests per call; map it
    # onto pydantic-ai's request limit (one request per turn).
    max_turns = int(raw.get("max_turns", 10))
    usage_limits = UsageLimits(request_limit=max(1, max_turns))
    return ModelBundle(config=config, model=model, settings=settings, usage_limits=usage_limits)


# ---------------------------------------------------------------------------
# MCP toolsets (native, via fastmcp transports)
# ---------------------------------------------------------------------------

def _mcp_transport(spec: MCPServerSpec):
    """Map an MCPServerSpec onto a fastmcp transport (env-resolved like mcp.py)."""
    from fastmcp.client.transports import (
        SSETransport,
        StdioTransport,
        StreamableHttpTransport,
    )

    if spec.mcp_type == "stdio":
        command = str(_resolve_ref(spec.command) or sys.executable)
        base_dir = spec.base_dir or Path()
        args = [_resolve_arg(base_dir, item) for item in spec.args]
        if not args:
            default_script = base_dir / "script.py"
            if default_script.exists():
                args = [str(default_script.resolve())]
        if not args:
            raise ValueError(f"Server '{spec.name}': stdio requires args or script.py")
        env = os.environ.copy()
        env.update({str(k): str(_resolve_ref(v) or "") for k, v in spec.env.items()})
        return StdioTransport(command=command, args=args, env=env)

    if spec.mcp_type == "sse":
        endpoint = str(_resolve_ref(spec.url) or "")
        if not endpoint:
            raise ValueError(f"SSE MCP server '{spec.name}' requires url")
        headers = {str(k): str(_resolve_ref(v) or "") for k, v in spec.headers.items()}
        return SSETransport(endpoint, headers=headers)

    if spec.mcp_type == "http":
        endpoint = str(_resolve_ref(spec.url) or "")
        if not endpoint:
            raise ValueError(f"HTTP MCP server '{spec.name}' requires url")
        headers = {str(k): str(_resolve_ref(v) or "") for k, v in spec.headers.items()}
        return StreamableHttpTransport(endpoint, headers=headers)

    raise ValueError(f"Unsupported MCP type: {spec.mcp_type}")


def build_mcp_toolset(spec: MCPServerSpec):
    """Build a prefixed `MCPToolset` for a server spec.

    The `.prefixed(prefix)` wrapper renders tool names as `{prefix}_{name}`, so a
    prefix of `mcp__<server>_` reproduces the legacy `mcp__<server>__<tool>`
    naming. Result-envelope normalization and hook firing are handled by
    `HookedMCPCapability.wrap_tool_execute` (capabilities/mcp.py), which wraps the
    `pydantic_ai.capabilities.MCP` capability built from this toolset.
    """
    server = spec.server_name or spec.name
    toolset = MCPToolset(_mcp_transport(spec), id=spec.name)
    return toolset.prefixed(f"mcp__{server}_")


# ---------------------------------------------------------------------------
# Session message-history persistence (context.json)
# ---------------------------------------------------------------------------

def load_message_history(session_dir: Path) -> list[ModelMessage]:
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


def _trim_messages(messages: list[ModelMessage]) -> list[ModelMessage]:
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
    existing: dict[str, Any] = {}
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
        "trace_id": snapshot.trace_id,
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


def _model_message_to_dicts(message: ModelMessage) -> list[dict[str, Any]]:
    """Convert a pydantic-ai `ModelMessage` into legacy OpenAI-style chat dicts."""
    out: list[dict[str, Any]] = []
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
        text_parts: list[str] = []
        reasoning_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []
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
        entry: dict[str, Any] = {"role": "assistant", "content": "\n".join(text_parts)}
        if reasoning_parts:
            entry["reasoning_content"] = "\n".join(reasoning_parts)
        if tool_calls:
            entry["tool_calls"] = tool_calls
        out.append(entry)
    return out


def new_call_messages(
    history: list[ModelMessage], all_messages: list[ModelMessage], *, system_prompt: str = "",
) -> list[dict[str, Any]]:
    """Render the messages added by this call (plus a leading system message) for the prompt sidecar.

    `instructions=`-based capabilities (vs the legacy `system_prompt=` kwarg) never produce a
    `SystemPromptPart` in message history, so the message-scan fallback below can no longer find
    one — `system_prompt` (the raw string from `LlmCallContext.system_prompt`) is the real source now.
    """
    new_messages = all_messages[len(history):]
    out: list[dict[str, Any]] = []
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
