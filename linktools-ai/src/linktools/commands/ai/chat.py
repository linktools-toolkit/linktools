#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""`lt ai`: single-entry agent CLI.

The whole CLI lives in this one file as a `BaseCommandGroup` with flat
``@subcommand`` methods (chat / run / sessions / session / approvals / approve /
reject), mirroring the cntr `__main__.py` pattern. Shared model/runtime/spec
wiring lives in `.support`.
"""

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING

from linktools.cli import BaseCommandGroup, CommandError, subcommand, subcommand_argument
from linktools.core import environ
from linktools.ai.agent.approval import ApprovalStatus
from linktools.ai.model.registry import (
    ModelClientUnavailable,
    ModelOutputError,
    ModelTurnLimitExceeded,
)
from linktools.system import get_user

from .support import (
    build_agent_spec,
    build_runtime,
    build_storage,
    ensure_session,
    validate_session_id,
)

if TYPE_CHECKING:
    from argparse import Namespace

_EXIT_WORDS = {"exit", "quit"}


class Command(BaseCommandGroup):
    """AI agent tools: chat / run / sessions / approvals."""

    @property
    def name(self) -> str:
        return "ai"

    @property
    def order(self) -> str:
        # Preserve the historical sort key so `ai` keeps its position in the
        # top-level `lt` tree (the package used to declare `__order__`).
        return "\x1f200-ai"

    @property
    def known_errors(self) -> "list[type[BaseException]]":
        return super().known_errors + [
            ModelClientUnavailable, ModelOutputError, ModelTurnLimitExceeded,
        ]

    # ------------------------------------------------------------------ chat

    @subcommand("chat", help="interactive agent chat session")
    @subcommand_argument("--model", default=None, help="model name (default: $OPENAI_MODEL)")
    @subcommand_argument("--base-url", default=None, help="OpenAI-compatible base url (default: $OPENAI_BASE_URL)")
    @subcommand_argument("--api-key", default=None, help="api key (default: $OPENAI_API_KEY)")
    @subcommand_argument("--session", default="main", help="session id (default: main)")
    @subcommand_argument("--workdir", default=None, help="agent working directory (default: current directory)")
    def on_chat(self, model=None, base_url=None, api_key=None, session="main", workdir=None):
        args = _namespace(model=model, base_url=base_url, api_key=api_key,
                          session=session, workdir=workdir)
        return asyncio.run(self._chat_async(args))

    async def _chat_async(self, args: "Namespace") -> "int | None":
        runtime = build_runtime(args)
        spec = build_agent_spec(args)
        session_id = validate_session_id(args.session)
        storage = build_storage()
        await ensure_session(storage, session_id)

        workdir = Path(args.workdir) if args.workdir else Path.cwd()
        self.logger.info(f"session: {session_id} (workdir: {workdir})")
        while True:
            try:
                line = await asyncio.to_thread(input, "> ")
            except EOFError:
                break
            line = line.strip()
            if not line:
                continue
            if line in _EXIT_WORDS:
                break
            try:
                await self._run_turn(runtime, spec, session_id, line)
            except asyncio.CancelledError:
                self.logger.warning("turn cancelled")
            except KeyboardInterrupt:
                self.logger.warning("turn cancelled")
        return 0

    async def _run_turn(self, runtime, spec, session_id: str, line: str) -> None:
        async for event in runtime.run_stream(spec, line, session_id=session_id):
            if event["type"] == "text":
                print(event["text"], end="", flush=True)
            elif event["type"] == "tool":
                print(f"\n[tool: {event['name']} {event['phase']}"
                      f"{' ok' if event.get('ok') else ''}]")
        print()

    # ------------------------------------------------------------------- run

    @subcommand("run", help="run agent with a single prompt")
    @subcommand_argument("prompt", help="the prompt")
    @subcommand_argument("--model", default=None, help="model name (default: $OPENAI_MODEL)")
    @subcommand_argument("--base-url", default=None, help="OpenAI-compatible base url (default: $OPENAI_BASE_URL)")
    @subcommand_argument("--api-key", default=None, help="api key (default: $OPENAI_API_KEY)")
    @subcommand_argument("--session", default="main", help="session id (default: main)")
    @subcommand_argument("--workdir", default=None, help="agent working directory (default: current directory)")
    def on_run(self, prompt: str, model=None, base_url=None, api_key=None,
               session="main", workdir=None):
        args = _namespace(model=model, base_url=base_url, api_key=api_key,
                          session=session, workdir=workdir)
        return asyncio.run(self._run_async(args, prompt))

    async def _run_async(self, args: "Namespace", prompt: str) -> "int | None":
        runtime = build_runtime(args)
        spec = build_agent_spec(args)
        session_id = validate_session_id(args.session)
        storage = build_storage()
        await ensure_session(storage, session_id)
        result = await runtime.run(spec, prompt, session_id=session_id)
        print(result.output)
        return 0

    # -------------------------------------------------------- sessions / session

    @subcommand("sessions", help="list all sessions")
    def on_sessions(self):
        storage = build_storage()
        return asyncio.run(self._sessions_async(storage))

    async def _sessions_async(self, storage) -> "int | None":
        records = await _list_sessions(storage)
        if not records:
            self.logger.info("no sessions")
            return 0
        for rec in records:
            self.logger.info(
                f"{rec.id}\t{rec.status.value}\t{rec.updated_at.isoformat()}"
            )
        return 0

    @subcommand("session", help="show a session's messages")
    @subcommand_argument("session_id", help="session id")
    def on_session(self, session_id: str):
        storage = build_storage()
        return asyncio.run(self._session_async(storage, session_id))

    async def _session_async(self, storage, session_id: str) -> "int | None":
        session_id = validate_session_id(session_id)
        record = await storage.sessions.get(session_id)
        if record is None:
            raise CommandError(f'session "{session_id}" not found')
        messages = await storage.sessions.list_messages(session_id)
        self.logger.info(f"session: {record.id} ({record.status.value}, "
                         f"{len(messages)} messages)")
        for msg in messages:
            content = msg.content if isinstance(msg.content, str) else repr(msg.content)
            self.logger.info(f"[{msg.role.value}] {content}")
        return 0

    # ----------------------------------------------- approvals / approve / reject

    @subcommand("approvals", help="list pending approval requests")
    def on_approvals(self):
        storage = build_storage()
        return asyncio.run(self._approvals_async(storage))

    async def _approvals_async(self, storage) -> "int | None":
        requests = await _list_pending_approvals(storage)
        if not requests:
            self.logger.info("no pending approvals")
            return 0
        for req in requests:
            self.logger.info(
                f"{req.id}\trun={req.run_id}\ttool={req.tool_name}"
            )
        return 0

    @subcommand("approve", help="approve a pending request")
    @subcommand_argument("approval_id", help="approval request id")
    def on_approve(self, approval_id: str):
        storage = build_storage()
        return asyncio.run(
            self._resolve_async(storage, approval_id, approved=True, reason=None)
        )

    @subcommand("reject", help="reject a pending request")
    @subcommand_argument("approval_id", help="approval request id")
    @subcommand_argument("--reason", default=None, help="rejection reason")
    def on_reject(self, approval_id: str, reason=None):
        storage = build_storage()
        return asyncio.run(
            self._resolve_async(storage, approval_id, approved=False, reason=reason)
        )

    async def _resolve_async(self, storage, approval_id: str, *,
                             approved: bool, reason: "str | None") -> "int | None":
        request = await storage.approvals.get(approval_id)
        if request is None:
            raise CommandError(f'approval "{approval_id}" not found')
        if request.status != ApprovalStatus.PENDING:
            raise CommandError(
                f'approval "{approval_id}" is already {request.status.value}'
            )
        resolved_by = get_user() or "cli"
        if approved:
            await storage.approvals.approve(
                approval_id, expected_version=request.version,
                resolved_by=resolved_by,
            )
            self.logger.info(f"approved {approval_id}")
        else:
            await storage.approvals.reject(
                approval_id, expected_version=request.version,
                resolved_by=resolved_by, reason=reason,
            )
            self.logger.info(f"rejected {approval_id}")
        return 0


# ---- module helpers --------------------------------------------------------

def _namespace(**kwargs) -> SimpleNamespace:
    """Build an args-like object from subcommand kwargs (cntr pattern)."""
    return SimpleNamespace(**kwargs)


async def _list_sessions(storage) -> list:
    """Enumerate every session record by scanning the sessions directory.

    `SessionStore` exposes no `list()` method, so we glob the on-disk layout
    (`<root>/sessions/{id}/record.json`) and rehydrate each record through the
    store's own `get()` to keep deserialization encapsulated.
    """
    root = Path(environ.get_data_path("ai")) / "sessions"
    records = []
    for record_path in sorted(root.glob("*/record.json")):
        session_id = record_path.parent.name
        record = await storage.sessions.get(session_id)
        if record is not None:
            records.append(record)
    return records


async def _list_pending_approvals(storage) -> list:
    """Enumerate pending approval requests across all runs.

    `ApprovalStore.list_pending(run_id)` is scoped to a single run; there is no
    global listing API. We glob `<root>/approvals/requests/*.json`, rehydrate
    each request via the store's `get()`, and keep only pending ones.
    """
    root = Path(environ.get_data_path("ai")) / "approvals" / "requests"
    requests = []
    for request_path in sorted(root.glob("*.json")):
        approval_id = request_path.stem
        request = await storage.approvals.get(approval_id)
        if request is not None and request.status == ApprovalStatus.PENDING:
            requests.append(request)
    return requests


# NOTE: this module deliberately does NOT expose a module-level ``command``.
# The package ``linktools.commands.ai`` exposes ``command = Command()`` so the
# package is discovered as the single top-level ``lt ai`` command (see the
# package-walk handling in `linktools.cli.command.iter_module_commands`).
# Run standalone via: python -m linktools.commands.ai.chat
if __name__ == "__main__":
    Command().main()
