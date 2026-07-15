#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""`lt ai chat`: interactive agent chat session.

Each turn mints its own run id so Ctrl+C can cancel the in-flight Run through
the runtime (not just the REPL), and a paused turn offers an
interactive Approve / Reject / Later choice that drives ``Runtime.resume``
without re-passing an AgentSpec."""

import asyncio
from typing import TYPE_CHECKING

from linktools.ai.model.registry import (
    ModelClientUnavailable,
    ModelOutputError,
    ModelTurnLimitExceeded,
)
from linktools.cli import BaseCommand

from .assembly import build_project_bundle, load_agent_spec
from .support import (
    announce_paused,
    ensure_session,
    new_run_id,
    resolve_approval,
    validate_session_id,
)

if TYPE_CHECKING:
    from argparse import Namespace

    from linktools.cli import CommandParser

_EXIT_WORDS = {"exit", "quit"}
# Interactive approval choices offered when a turn pauses.
_APPROVE = "approve"
_REJECT = "reject"
_LATER = "later"


class Command(BaseCommand):
    """interactive agent chat session"""

    @property
    def known_errors(self) -> "list[type[BaseException]]":
        return super().known_errors + [
            ModelClientUnavailable,
            ModelOutputError,
            ModelTurnLimitExceeded,
        ]

    def init_arguments(self, parser: "CommandParser") -> None:
        parser.add_argument(
            "--model", default=None, help="model name (default: $OPENAI_MODEL)"
        )
        parser.add_argument(
            "--base-url",
            default=None,
            help="OpenAI-compatible base url (default: $OPENAI_BASE_URL)",
        )
        parser.add_argument(
            "--api-key", default=None, help="api key (default: $OPENAI_API_KEY)"
        )
        parser.add_argument(
            "--session", default="main", help="session id (default main)"
        )
        parser.add_argument(
            "--workdir",
            default=None,
            help="agent working directory (default: current directory)",
        )

    def run(self, args: "Namespace") -> "int | None":
        return asyncio.run(self._chat_async(args))

    async def _chat_async(self, args: "Namespace") -> "int | None":
        bundle = build_project_bundle(args)
        spec = await load_agent_spec(bundle, None)
        session_id = validate_session_id(args.session)
        await ensure_session(bundle.storage, session_id)

        self.logger.info(f"session: {session_id} (project: {bundle.project.root})")
        while True:
            try:
                line = await asyncio.to_thread(input, "> ")
            except EOFError:
                break
            line = line.strip()
            if not line:
                continue
            if line.startswith("/"):
                # Wrap slash-command dispatch so a failing command (e.g.
                # /resume <stale_id> -> RunNotFoundError) logs and returns to
                # the prompt instead of killing the whole REPL.
                try:
                    spec, session_id, stop = await self._handle_command(
                        bundle, spec, session_id, line
                    )
                except asyncio.CancelledError:
                    self.logger.warning("command cancelled")
                    stop = False
                except Exception as exc:
                    self.logger.error(f"{type(exc).__name__}: {exc}")
                    stop = False
                if stop:
                    break
                continue
            if line in _EXIT_WORDS:
                break
            try:
                await self._run_turn(
                    bundle.runtime, spec, bundle.storage, session_id, line
                )
            except asyncio.CancelledError:
                self.logger.warning("turn cancelled")
            except KeyboardInterrupt:
                self.logger.warning("turn cancelled")
        return 0

    async def _handle_command(self, bundle, spec, session_id, line):
        """Slash-command dispatch. Returns (spec, session_id, stop)."""
        parts = line.split()
        cmd = parts[0]
        if cmd in ("/exit", "/quit"):
            return spec, session_id, True
        if cmd == "/help":
            self.logger.info(
                "commands: /help /agent <name> /new <session> /session "
                "/inspect /approvals /resume <run_id> /exit"
            )
        elif cmd == "/agent":
            name = parts[1] if len(parts) > 1 else bundle.project.default_agent
            spec = await load_agent_spec(bundle, name)
            self.logger.info(f"agent: {getattr(spec, 'name', name)}")
        elif cmd == "/inspect":
            await self._show_inspect(bundle, spec)
        elif cmd == "/new":
            new_session = parts[1] if len(parts) > 1 else "main"
            session_id = validate_session_id(new_session)
            await ensure_session(bundle.storage, session_id)
            self.logger.info(f"session: {session_id}")
        elif cmd == "/session":
            self.logger.info(f"session: {session_id}")
        elif cmd == "/approvals":
            await self._show_approvals(bundle.storage)
        elif cmd == "/resume":
            run_id = parts[1] if len(parts) > 1 else None
            if run_id:
                await self._resume_turn(bundle.runtime, bundle.storage, run_id)
            else:
                self.logger.info("usage: /resume <run_id>")
        else:
            self.logger.info(f"unknown command: {cmd} (try /help)")
        return spec, session_id, False

    async def _show_approvals(self, storage) -> None:
        from .approvals import _list_pending_approvals

        requests = await _list_pending_approvals(storage)
        if not requests:
            self.logger.info("no pending approvals")
            return
        for req in requests:
            self.logger.info(f"{req.id}\trun={req.run_id}\ttool={req.tool_name}")

    async def _show_inspect(self, bundle, spec) -> None:
        # Delegate capability resolution to the runtime's single inspect path.
        inspection = await bundle.runtime.inspect(spec)
        descriptors = getattr(inspection, "tool_descriptors", ()) or ()
        self.logger.info(
            f"agent: {getattr(spec, 'id', '?')} ({len(descriptors)} tools)"
        )
        for descriptor in descriptors:
            self.logger.info(f"  - {getattr(descriptor, 'name', descriptor)}")
        for warning in getattr(inspection, "warnings", ()) or ():
            self.logger.warning(f"  ! {warning}")

    async def _run_turn(
        self, runtime, spec, storage, session_id: str, line: str
    ) -> None:
        # CLI-owned run id so an interrupt cancels this exact turn's Run.
        run_id = new_run_id()
        try:
            async for event in runtime.run_stream(
                spec, line, session_id=session_id, run_id=run_id
            ):
                kind = event["type"]
                if kind == "text":
                    print(event["text"], end="", flush=True)
                elif kind == "tool":
                    print(
                        f"\n[tool: {event['name']} {event['phase']}"
                        f"{' ok' if event.get('ok') else ''}]"
                    )
                elif kind == "paused":
                    # Hand the paused turn to the interactive flow; once it
                    # returns the turn is resolved (or deferred) -- stop the
                    # stream regardless.
                    await self._handle_paused(runtime, storage, event)
                    print()
                    return
            print()
        except asyncio.CancelledError:
            # Ctrl+C mid-run: cancel the Run through the runtime so it stops
            # executing, then keep the REPL alive.
            await runtime.cancel(run_id)
            self.logger.warning("turn cancelled")

    async def _handle_paused(self, runtime, storage, event) -> None:
        await announce_paused(storage, event, self.logger)
        approval_id = event.get("approval_id")
        run_id = event.get("run_id")
        choice = await asyncio.to_thread(self._prompt_approval)
        if choice == _APPROVE:
            await resolve_approval(storage, approval_id, approved=True, reason=None)
            await self._resume_turn(runtime, storage, run_id)
        elif choice == _REJECT:
            await resolve_approval(storage, approval_id, approved=False, reason=None)
            await runtime.cancel(run_id)
            self.logger.info("rejected and cancelled")
        else:
            self.logger.info(
                f"resume later: lt ai approve {approval_id} && lt ai resume {run_id}"
            )

    async def _resume_turn(self, runtime, storage, run_id: str) -> None:
        # Resume re-drives the ORIGINAL spec from the snapshot; the caller
        # passes only the run id.
        try:
            async for event in runtime.resume(run_id):
                kind = event["type"]
                if kind == "text":
                    print(event["text"], end="", flush=True)
                elif kind == "tool":
                    print(
                        f"\n[tool: {event['name']} {event['phase']}"
                        f"{' ok' if event.get('ok') else ''}]"
                    )
                elif kind == "paused":
                    # Paused again mid-resume: surface and stop (do not recurse
                    # into another interactive prompt within the same turn).
                    await announce_paused(storage, event, self.logger)
                    print()
                    return
        except asyncio.CancelledError:
            await runtime.cancel(run_id)
            self.logger.warning("turn cancelled")

    @staticmethod
    def _prompt_approval() -> str:
        try:
            raw = input("approve/reject/later [later]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return _LATER
        if raw.startswith("a"):
            return _APPROVE
        if raw.startswith("r"):
            return _REJECT
        return _LATER


command = Command()
