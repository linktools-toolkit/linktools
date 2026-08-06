#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Local Agent runtime used by the CLI and ACP composition roots."""

import asyncio
from collections.abc import Awaitable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol
from uuid import uuid4

from linktools.core import environ

from pydantic_ai.messages import ModelMessage, ModelMessagesTypeAdapter

from ..agent.runner import LocalAgentRunner
from ..core.errors import ErrorCode, LinktoolsAIError
from ..core.json import JsonValue
from ..storage.files import write_bytes_atomic
from .project import LocalProject
from .record import LocalExecutionRecord, LocalRecordStore

logger = environ.get_logger("ai.local.runtime")


class TextHandler(Protocol):
    async def __call__(self, text: str) -> None: ...


class EventHandler(Protocol):
    def __call__(self, event: 'dict[str, JsonValue]') -> 'Awaitable[None] | None': ...


@dataclass(frozen=True, slots=True)
class LocalRunResult:
    execution_id: str
    session_id: str
    run_id: str
    output: str


@dataclass(frozen=True, slots=True)
class LocalSession:
    session_id: str
    cwd: Path
    session_revision: int = 0


class LocalAgentRuntime:
    """Run one local Agent while keeping session history under the project root."""

    def __init__(
        self,
        project: LocalProject,
        *,
        runner: LocalAgentRunner,
    ) -> None:
        self.project = project
        self._runner = runner
        self._sessions: "dict[str, LocalSession]" = {}
        self._tasks: "dict[str, asyncio.Task[LocalRunResult]]" = {}
        self._session_runs: "dict[str, set[str]]" = {}
        self._idempotency_tasks: "dict[tuple[str, str], asyncio.Task[LocalRunResult]]" = {}
        self._idempotency_results: "dict[tuple[str, str], LocalRunResult]" = {}
        self._idempotency_prompts: "dict[tuple[str, str], str]" = {}
        self._idempotency_agents: "dict[tuple[str, str], str | None]" = {}
        self._shutdown_ids: set[str] = set()
        self._records = LocalRecordStore(project.storage_root, project.project_id, work_root=project.root)
        self._history_locks: "dict[str, asyncio.Lock]" = {}
        self._lock = asyncio.Lock()

    @property
    def sessions_root(self) -> Path:
        return self.project.storage_root / ".linktools" / "sessions"

    async def run(
        self,
        session_id: str,
        prompt: str,
        *,
        cwd: "str | Path | None" = None,
        agent_id: "str | None" = None,
        idempotency_key: "str | None" = None,
        on_text: "TextHandler | None" = None,
        on_event: "EventHandler | None" = None,
    ) -> LocalRunResult:
        _validate_session_id(session_id)
        if not prompt.strip():
            raise ValueError("prompt must not be empty")
        session = await self.open_session(session_id, cwd=cwd)
        await self._records.initialize()
        execution_id = uuid4().hex
        created_at = datetime.now(timezone.utc)
        idempotency_identity = None if idempotency_key is None else (session_id, idempotency_key)
        existing_result: LocalRunResult | None = None
        existing_task: asyncio.Task[LocalRunResult] | None = None
        async with self._lock:
            if idempotency_identity is not None:
                previous_prompt = self._idempotency_prompts.get(idempotency_identity)
                if previous_prompt is not None and previous_prompt != prompt:
                    raise LinktoolsAIError(ErrorCode.IDEMPOTENCY_CONFLICT)
                previous_agent = self._idempotency_agents.get(idempotency_identity)
                if idempotency_identity in self._idempotency_agents and previous_agent != agent_id:
                    raise LinktoolsAIError(ErrorCode.IDEMPOTENCY_CONFLICT)
                existing_result = self._idempotency_results.get(idempotency_identity)
                existing_task = self._idempotency_tasks.get(idempotency_identity)
                if existing_result is None and existing_task is None:
                    self._idempotency_prompts[idempotency_identity] = prompt
                    self._idempotency_agents[idempotency_identity] = agent_id
            if existing_result is not None or existing_task is not None:
                task = None
            else:
                task = asyncio.create_task(self._run(session, prompt, agent_id, on_text, on_event, execution_id))
                self._tasks[execution_id] = task
                self._session_runs.setdefault(session_id, set()).add(execution_id)
                if idempotency_identity is not None:
                    self._idempotency_tasks[idempotency_identity] = task
                await self._records.save(
                    LocalExecutionRecord(
                        self.project.project_id,
                        session_id,
                        session.session_revision,
                        session.cwd.as_posix(),
                        execution_id,
                        "PENDING_START",
                        created_at,
                        created_at,
                        None,
                    )
                )
                await self._records.save(
                    LocalExecutionRecord(
                        self.project.project_id,
                        session_id,
                        session.session_revision,
                        session.cwd.as_posix(),
                        execution_id,
                        "STARTED",
                        created_at,
                        datetime.now(timezone.utc),
                        None,
                    )
                )
        if existing_result is not None:
            return existing_result
        if existing_task is not None:
            return await asyncio.shield(existing_task)
        if task is None:
            raise RuntimeError("local execution task was not created")
        try:
            result = await task
            completed = LocalRunResult(execution_id, result.session_id, result.run_id, result.output)
            await self._save_record(execution_id, session, created_at, "SUCCEEDED", None)
            if idempotency_identity is not None:
                async with self._lock:
                    self._idempotency_results[idempotency_identity] = completed
            return completed
        except asyncio.CancelledError:
            status = "PROCESS_SHUTDOWN" if execution_id in self._shutdown_ids else "CANCELLED"
            reason = "PROCESS_SHUTDOWN" if status == "PROCESS_SHUTDOWN" else "cancelled"
            await self._save_record(execution_id, session, created_at, status, reason)
            raise
        except Exception as error:
            await self._save_record(execution_id, session, created_at, "FAILED", str(error))
            raise
        finally:
            async with self._lock:
                self._tasks.pop(execution_id, None)
                session_runs = self._session_runs.get(session_id)
                if session_runs is not None:
                    session_runs.discard(execution_id)
                    if not session_runs:
                        self._session_runs.pop(session_id, None)
                self._shutdown_ids.discard(execution_id)
                if idempotency_identity is not None:
                    self._idempotency_tasks.pop(idempotency_identity, None)

    async def cancel(self, session_id: str) -> bool:
        async with self._lock:
            tasks = tuple(
                self._tasks[execution_id]
                for execution_id in self._session_runs.get(session_id, ())
                if execution_id in self._tasks
            )
        if not tasks:
            return False
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        return True

    async def shutdown(self) -> None:
        async with self._lock:
            execution_ids = tuple(self._tasks)
            tasks = tuple(self._tasks.values())
            self._shutdown_ids.update(execution_ids)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        logger.info("local runtime shutdown: executions=%s", len(execution_ids))

    async def open_session(self, session_id: str, *, cwd: "str | Path | None" = None) -> LocalSession:
        _validate_session_id(session_id)
        existing = self._sessions.get(session_id)
        if existing is not None:
            return existing
        session_cwd = Path(cwd or self.project.root).expanduser().resolve()
        try:
            session_cwd.relative_to(self.project.root)
        except ValueError as error:
            raise ValueError("session cwd must be inside the project root") from error
        session = LocalSession(session_id, session_cwd)
        session_file = self._session_file(session_id)
        if session_file.exists():
            ModelMessagesTypeAdapter.validate_json(session_file.read_bytes())
        self.sessions_root.mkdir(parents=True, exist_ok=True)
        self._sessions[session_id] = session
        logger.info(
            "local session opened: session=%s cwd=%s storage=%s",
            session_id,
            session.cwd,
            self.project.storage_root,
        )
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
        await self._save_history(target.session_id, source_messages, [])
        return target

    async def close_session(self, session_id: str, *, force: bool = False) -> None:
        async with self._lock:
            active = bool(self._session_runs.get(session_id))
        if active and not force:
            raise LinktoolsAIError(ErrorCode.SESSION_ACTIVE_EXECUTIONS)
        if active:
            await self.cancel(session_id)
        self._sessions.pop(session_id, None)

    async def _run(
        self,
        session: LocalSession,
        prompt: str,
        agent_id: "str | None",
        on_text: "TextHandler | None",
        on_event: "EventHandler | None",
        execution_id: str,
    ) -> LocalRunResult:
        history = self._load_history(session.session_id)
        logger.info("local Agent run started: session=%s history=%s", session.session_id, len(history))
        async def on_agent_event(event: 'dict[str, JsonValue]') -> None:
            if event.get("type") == "text" and on_text is not None:
                await _notify_text(on_text, str(event.get("text", "")))
            if on_event is not None:
                await _notify_event(on_event, event)

        result = await self._runner.run(
            agent_id,
            prompt,
            history,
            session.session_id,
            on_event=on_agent_event,
        )
        messages = result.messages
        run_id = result.run_id
        await self._save_history(session.session_id, messages, history)
        logger.info("local Agent run completed: session=%s run=%s", session.session_id, run_id)
        return LocalRunResult(execution_id, session.session_id, run_id, result.output)

    def _session_file(self, session_id: str) -> Path:
        return self.sessions_root / f"{session_id}.json"

    def _load_history(self, session_id: str) -> "list[ModelMessage]":
        path = self._session_file(session_id)
        if not path.exists():
            return []
        return list(ModelMessagesTypeAdapter.validate_json(path.read_bytes()))

    async def _save_history(
        self,
        session_id: str,
        messages: "list[ModelMessage]",
        base_history: "list[ModelMessage]",
    ) -> None:
        self.sessions_root.mkdir(parents=True, exist_ok=True)
        target = self._session_file(session_id)
        lock = self._history_locks.setdefault(session_id, asyncio.Lock())
        delta = messages[len(base_history):] if messages[:len(base_history)] == base_history else messages
        async with lock:
            current = self._load_history(session_id)
            merged = current + delta
            write_bytes_atomic(target, ModelMessagesTypeAdapter.dump_json(merged), fsync=True)

    async def _save_record(
        self,
        execution_id: str,
        session: LocalSession,
        created_at: datetime,
        status: str,
        reason: 'str | None',
    ) -> None:
        await self._records.save(
            LocalExecutionRecord(
                self.project.project_id,
                session.session_id,
                session.session_revision,
                session.cwd.as_posix(),
                execution_id,
                status,
                created_at,
                datetime.now(timezone.utc),
                reason,
            )
        )


def _validate_session_id(session_id: str) -> None:
    if not session_id or session_id in {".", ".."} or "/" in session_id or "\\" in session_id:
        raise ValueError("session id must not contain path separators")


async def _notify_text(handler: TextHandler, text: str) -> None:
    try:
        await handler(text)
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.warning("local text observer failed")


async def _notify_event(handler: EventHandler, event: 'dict[str, JsonValue]') -> None:
    try:
        pending = handler(event)
        if pending is not None:
            await pending
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.warning("local event observer failed")


__all__ = ["EventHandler", "LocalAgentRuntime", "LocalRunResult", "LocalSession", "TextHandler"]
