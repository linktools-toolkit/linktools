#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Local Agent runtime used by the CLI and ACP composition roots."""

import asyncio
import inspect
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from linktools.core import environ

from pydantic_ai import Agent, AgentRunResultEvent
from pydantic_ai.messages import (
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    ModelMessage,
    ModelMessagesTypeAdapter,
    PartDeltaEvent,
    PartEndEvent,
    PartStartEvent,
    TextPart,
    TextPartDelta,
    ThinkingPart,
    ThinkingPartDelta,
)
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from .project import LocalProject
from .tools import build_local_tools

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Mapping

logger = environ.get_logger("ai.local.runtime")


@dataclass(frozen=True, slots=True)
class LocalRunResult:
    session_id: str
    run_id: str
    output: str


@dataclass(frozen=True, slots=True)
class LocalSession:
    session_id: str
    cwd: Path


class LocalAgentRuntime:
    """Run one local Agent while keeping session history under the project root."""

    def __init__(
        self,
        project: LocalProject,
        *,
        model: "str | object | None" = None,
        base_url: "str | None" = None,
        api_key: "str | None" = None,
        agent_factory: "Callable[..., Agent[Any]] | None" = None,
        tools: "tuple[Callable[..., Any], ...] | None" = None,
    ) -> None:
        self.project = project
        self._model = model
        self._base_url = base_url or os.getenv("OPENAI_BASE_URL")
        self._api_key = api_key or os.getenv("OPENAI_API_KEY")
        self._agent_factory = agent_factory or Agent
        self._tools = tools if tools is not None else build_local_tools(project.root)
        self._agents: "dict[str | None, Agent[Any]]" = {}
        self._sessions: "dict[str, LocalSession]" = {}
        self._tasks: "dict[str, asyncio.Task[LocalRunResult]]" = {}
        self._lock = asyncio.Lock()

    @property
    def sessions_root(self) -> Path:
        return self.project.root / ".linktools" / "sessions"

    async def run(
        self,
        session_id: str,
        prompt: str,
        *,
        cwd: "str | Path | None" = None,
        agent_id: "str | None" = None,
        on_text: "Callable[[str], Awaitable[None]] | None" = None,
        on_event: "Callable[[Mapping[str, object]], Awaitable[None]] | None" = None,
    ) -> LocalRunResult:
        _validate_session_id(session_id)
        if not prompt.strip():
            raise ValueError("prompt must not be empty")
        session = await self.open_session(session_id, cwd=cwd)
        async with self._lock:
            task = asyncio.create_task(self._run(session, prompt, agent_id, on_text, on_event))
            self._tasks[session_id] = task
        try:
            return await task
        finally:
            async with self._lock:
                if self._tasks.get(session_id) is task:
                    self._tasks.pop(session_id, None)

    async def cancel(self, session_id: str) -> bool:
        task = self._tasks.get(session_id)
        if task is None:
            return False
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        return True

    async def open_session(self, session_id: str, *, cwd: "str | Path | None" = None) -> LocalSession:
        _validate_session_id(session_id)
        existing = self._sessions.get(session_id)
        if existing is not None:
            return existing
        session = LocalSession(session_id, Path(cwd or self.project.root).expanduser().resolve())
        session_file = self._session_file(session_id)
        if session_file.exists():
            ModelMessagesTypeAdapter.validate_json(session_file.read_bytes())
        self.sessions_root.mkdir(parents=True, exist_ok=True)
        self._sessions[session_id] = session
        logger.info("local session opened: session=%s cwd=%s", session_id, session.cwd)
        return session

    async def list_sessions(self, *, cwd: "str | None" = None) -> "tuple[LocalSession, ...]":
        self.sessions_root.mkdir(parents=True, exist_ok=True)
        values = []
        for path in sorted(self.sessions_root.glob("*.json")):
            session_id = path.stem
            try:
                await self.open_session(session_id)
            except ValueError:
                continue
            session = self._sessions[session_id]
            if cwd is None or session.cwd.as_posix() == Path(cwd).expanduser().resolve().as_posix():
                values.append(session)
        return tuple(values)

    async def fork_session(self, source_id: str, *, cwd: "str | None" = None) -> LocalSession:
        source = await self.open_session(source_id)
        target = await self.open_session(uuid4().hex, cwd=cwd or source.cwd)
        source_messages = self._load_history(source.session_id)
        self._save_history(target.session_id, source_messages)
        return target

    async def close_session(self, session_id: str) -> None:
        await self.cancel(session_id)
        self._sessions.pop(session_id, None)

    async def _run(
        self,
        session: LocalSession,
        prompt: str,
        agent_id: "str | None",
        on_text: "Callable[[str], Awaitable[None]] | None",
        on_event: "Callable[[Mapping[str, object]], Awaitable[None]] | None",
    ) -> LocalRunResult:
        history = self._load_history(session.session_id)
        agent = self._get_agent(agent_id)
        logger.info("local Agent run started: session=%s history=%s", session.session_id, len(history))
        final_result = None
        async with agent.run_stream_events(
            prompt,
            message_history=history or None,
            conversation_id=session.session_id,
        ) as events:
            output_parts: list[str] = []
            async for event in events:
                if isinstance(event, AgentRunResultEvent):
                    final_result = event.result
                    continue
                mapped = _map_event(event)
                if mapped is None:
                    continue
                if mapped["type"] == "text":
                    text = str(mapped["text"])
                    output_parts.append(text)
                    if on_text is not None:
                        await _notify(on_text, text)
                if on_event is not None:
                    await _notify(on_event, mapped)
        if final_result is None:
            raise RuntimeError("Agent ended without a result")
        messages = final_result.all_messages()
        run_id = final_result.run_id
        self._save_history(session.session_id, messages)
        logger.info("local Agent run completed: session=%s run=%s", session.session_id, run_id)
        return LocalRunResult(session.session_id, run_id, "".join(output_parts) or str(final_result.output))

    def _get_agent(self, agent_id: "str | None") -> "Agent[Any]":
        if agent_id in self._agents:
            return self._agents[agent_id]
        model = self._model or self.project.config.get("model") or os.getenv("OPENAI_MODEL") or "gpt-4o-mini"
        if isinstance(model, str) and model != "test":
            model = OpenAIChatModel(
                model.removeprefix("openai:"),
                provider=OpenAIProvider(
                    base_url=self._base_url or self.project.config.get("base_url"),
                    api_key=self._api_key,
                ),
            )
        instructions = self._instructions(agent_id)
        tools = () if model == "test" else self._tools
        agent = self._agent_factory(model, name=agent_id or "linktools-local", instructions=instructions, tools=tools)
        self._agents[agent_id] = agent
        return agent

    def _instructions(self, agent_id: "str | None") -> str:
        configured = self.project.config.get("instructions")
        if isinstance(configured, str) and configured.strip():
            return configured
        selected = agent_id or str(self.project.config.get("default_agent", "default"))
        agent_file = self.project.root / ".linktools" / "agents" / f"{selected}.md"
        if agent_file.is_file():
            return agent_file.read_text(encoding="utf-8")
        return "You are the Linktools local coding assistant. Be concise and practical."

    def _session_file(self, session_id: str) -> Path:
        return self.sessions_root / f"{session_id}.json"

    def _load_history(self, session_id: str) -> "list[ModelMessage]":
        path = self._session_file(session_id)
        if not path.exists():
            return []
        return list(ModelMessagesTypeAdapter.validate_json(path.read_bytes()))

    def _save_history(self, session_id: str, messages: "list[ModelMessage]") -> None:
        self.sessions_root.mkdir(parents=True, exist_ok=True)
        target = self._session_file(session_id)
        temporary = target.with_suffix(".tmp")
        temporary.write_bytes(ModelMessagesTypeAdapter.dump_json(messages))
        temporary.replace(target)


def _validate_session_id(session_id: str) -> None:
    if not session_id or session_id in {".", ".."} or "/" in session_id or "\\" in session_id:
        raise ValueError("session id must not contain path separators")


def _map_event(event: object) -> "dict[str, object] | None":
    if isinstance(event, PartStartEvent):
        if isinstance(event.part, TextPart) and event.part.content:
            return {"type": "text", "text": event.part.content}
        if isinstance(event.part, ThinkingPart) and event.part.content:
            return {"type": "thinking", "text": event.part.content}
    elif isinstance(event, PartDeltaEvent):
        if isinstance(event.delta, TextPartDelta) and event.delta.content_delta:
            return {"type": "text", "text": event.delta.content_delta}
        if isinstance(event.delta, ThinkingPartDelta) and event.delta.content_delta:
            return {"type": "thinking", "text": event.delta.content_delta}
    elif isinstance(event, PartEndEvent) and isinstance(event.part, TextPart) and event.part.content:
        return {"type": "text_end"}
    elif isinstance(event, FunctionToolCallEvent):
        part = event.part
        return {
            "type": "tool",
            "id": part.tool_call_id,
            "name": part.tool_name,
            "phase": "start",
            "arguments": part.args_as_dict(),
        }
    elif isinstance(event, FunctionToolResultEvent):
        part = event.part
        return {
            "type": "tool",
            "id": part.tool_call_id,
            "name": part.tool_name,
            "phase": "end",
            "ok": part.outcome == "success",
            "detail": str(part.content),
        }
    return None


async def _notify(callback: "Callable[..., object]", value: object) -> None:
    result = callback(value)
    if inspect.isawaitable(result):
        await result


__all__ = ["LocalAgentRuntime", "LocalRunResult", "LocalSession"]
